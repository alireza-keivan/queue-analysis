import logging
import os
import tempfile
import time

import boto3
from botocore.config import Config as BotoConfig
import requests
import runpod

from app.queue_management import (
    load_config,
    cap_check,
    release_cap,
    video_writer,
    model_creator,
    video_processor,
)
from app.track_diagnostics import collect_track_history

BUCKET_NAME = "Alireza-keivan"
BUCKET_REGION = "us-east-005"

# Config is static, safe to load once at cold start.
config = load_config()


def upload_annotated_video(file_path, job_id):
    # rp_upload's own upload helper can't determine Backblaze's region from
    # this endpoint format and silently signs requests with the wrong region,
    # causing SignatureDoesNotMatch. Building the client directly with the
    # correct region avoids that. request/response checksum calculation is
    # also restricted to "when_required" since Backblaze's signature
    # verification doesn't handle botocore's newer default checksum headers.
    client = boto3.client(
        "s3",
        endpoint_url=os.environ["BUCKET_ENDPOINT_URL"],
        aws_access_key_id=os.environ["BUCKET_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["BUCKET_SECRET_ACCESS_KEY"],
        region_name=BUCKET_REGION,
        config=BotoConfig(
            signature_version="s3v4",
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )
    # Keyed per job. A single fixed key meant every job silently overwrote the
    # previous one's video, so an older job's presigned URL quietly started
    # serving the newest job's output instead of its own.
    key = f"queue-analysis-outputs/{job_id}.mp4"
    client.upload_file(file_path, BUCKET_NAME, key)
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET_NAME, "Key": key},
        ExpiresIn=7 * 24 * 3600,
    )

def handler(event):
    job_id = event["id"]
    job_input = event["input"]
    video_url = job_input.get("video_url")
    if not video_url:
        return {"error": "No 'video_url' provided."}

    # Per-request overrides; fall back to queue.yaml. Turning annotate off
    # skips rendering, encoding and uploading the output video entirely.
    target_fps = job_input.get("target_fps", config.get("TARGET_FPS", 10))
    # Defaults to False: measured at +9.5s on an 11.2s job (+85%) for output
    # nothing downstream reads. Callers that want the video ask for it.
    annotate = job_input.get("annotate", False)
    diagnostic = job_input.get("diagnostic", False)
    # [[x, y], ...] in the source video's native pixel coordinates, picked in
    # the dashboard's browser-side ROI editor. Falls back to queue.yaml's
    # QUEUE_REGION when not supplied - see app/queue_management.py:model_creator.
    region = job_input.get("region")
    # Detection confidence / NMS IOU thresholds - same override-or-fall-back
    # pattern as region, exposed in the dashboard's advanced settings panel.
    conf = job_input.get("conf")
    iou = job_input.get("iou")

    input_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    # .mp4, not .avi: the writer now prefers H.264, and .avi is not playable
    # in a browser regardless of codec - this file is handed to a client.
    output_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    # Only the path is ever used (by VideoWriter/boto3), so close the handle
    # now rather than leaking a file descriptor per job on a long-lived worker.
    output_tmp.close()
    cap = None
    writer = None
    t_job_start = time.perf_counter()
    try:
        a = time.perf_counter()
        with requests.get(video_url, stream=True) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=1 << 20):
                input_tmp.write(chunk)
        input_tmp.close()
        t_download = time.perf_counter() - a
        input_mb = os.path.getsize(input_tmp.name) / 1e6

        cap = cap_check(input_tmp.name)
        if cap is None:
            return {"error": "Could not open downloaded video."}

        # Fresh QueueManager per job: reused tracker state (track_history,
        # ID counter, persist=True tracking) leaked across unrelated videos
        # when this was created once at cold start.
        a = time.perf_counter()
        queue_manager = model_creator(config, region=region, conf=conf, iou=iou)
        t_model_load = time.perf_counter() - a

        if diagnostic:
            # Standalone analysis pass - full-frame, not ROI-filtered, and
            # never touches upload/annotation. Only the GPU-bound collection
            # runs here; the churn scoring is pure CPU math with no model or
            # cv2 involved, so it deliberately does NOT run on this paid GPU
            # worker - run score_diagnostic.py against this output instead.
            # See app/track_diagnostics.py.
            a = time.perf_counter()
            history, stride, src_fps = collect_track_history(cap, queue_manager, target_fps=target_fps)
            t_diag = time.perf_counter() - a
            release_cap(cap)
            cap = None
            logging.info(f"Diagnostic collection took {t_diag:.1f}s over {len(history)} frames")
            return {
                "job_id": job_id,
                "model_load_s": round(t_model_load, 1),
                "collection_s": round(t_diag, 1),
                "stride": stride,
                "src_fps": src_fps,
                "history": [
                    {str(tid): list(box) for tid, box in frame.items()}
                    for frame in history
                ],
            }

        # Only built when actually annotating - constructing it regardless
        # allocated an encoder and created an output file for every job,
        # including the ones that never write a single frame to it.
        writer = video_writer(cap, output_tmp.name, target_fps) if annotate else None
        a = time.perf_counter()
        snapshots, tracks, profile = video_processor(
            cap, queue_manager, writer, job_id,
            target_fps=target_fps, annotate=annotate,
        )
        t_process = time.perf_counter() - a

        # Release before upload: VideoWriter buffers frames until closed,
        # so the file on disk isn't complete until this happens.
        release_cap(cap)
        cap = None
        if writer is not None:
            writer.release()
            writer = None
        output_mb = os.path.getsize(output_tmp.name) / 1e6

        a = time.perf_counter()
        annotated_video_url = upload_annotated_video(output_tmp.name, job_id) if annotate else None
        t_upload = time.perf_counter() - a

        t_total = time.perf_counter() - t_job_start
        # One logging.info() per line - a single large multi-line entry was
        # observed to get silently truncated by RunPod's log pipeline.
        for line in [
            "=" * 62,
            f"JOB COST BREAKDOWN  job={job_id}",
            f"  download    {t_download:7.1f}s  ({input_mb:.0f} MB in)",
            f"  model load  {t_model_load:7.1f}s  <- paid on EVERY job",
            f"  processing  {t_process:7.1f}s",
            f"  upload      {t_upload:7.1f}s  ({output_mb:.0f} MB out)",
            f"  TOTAL       {t_total:7.1f}s   "
            f"(non-processing overhead: "
            f"{(t_total - t_process) / t_total * 100:.0f}%)",
            "=" * 62,
        ]:
            logging.info(line)

        return {
            "job_id": job_id,
            "snapshots": snapshots,
            "tracks": tracks,
            "annotated_video_url": annotated_video_url,
            # Returned, not just logged: these numbers are otherwise visible
            # only in RunPod's log console, so no API caller (or dashboard)
            # can see where a job's time actually went.
            "cost": {
                "download_s": round(t_download, 2),
                "model_load_s": round(t_model_load, 2),
                "processing_s": round(t_process, 2),
                "upload_s": round(t_upload, 2),
                "total_s": round(t_total, 2),
                "input_mb": round(input_mb, 1),
                "output_mb": round(output_mb, 1),
                "annotated": annotate,
                "overhead_pct": round((t_total - t_process) / t_total * 100, 1) if t_total else 0.0,
            },
            "profile": profile,
        }
    finally:
        if cap is not None:
            release_cap(cap)
        if writer is not None:
            writer.release()
        os.unlink(input_tmp.name)
        os.unlink(output_tmp.name)


runpod.serverless.start({"handler": handler})
