# handler.py
import os
import tempfile

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

BUCKET_NAME = "Alireza-keivan"
BUCKET_REGION = "us-east-005"

# Config is static, safe to load once at cold start.
config = load_config()


def upload_annotated_video(file_path):
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
    key = "queue-analysis-outputs/annotated.avi"
    client.upload_file(file_path, BUCKET_NAME, key)
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET_NAME, "Key": key},
        ExpiresIn=7 * 24 * 3600,
    )


def handler(event):
    job_input = event["input"]
    video_url = job_input.get("video_url")
    if not video_url:
        return {"error": "No 'video_url' provided."}

    input_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    output_tmp = tempfile.NamedTemporaryFile(suffix=".avi", delete=False)
    cap = None
    writer = None
    try:
        with requests.get(video_url, stream=True) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=1 << 20):
                input_tmp.write(chunk)
        input_tmp.close()

        cap = cap_check(input_tmp.name)
        if cap is None:
            return {"error": "Could not open downloaded video."}

        # Fresh QueueManager per job: reused tracker state (track_history,
        # ID counter, persist=True tracking) leaked across unrelated videos
        # when this was created once at cold start.
        queue_manager = model_creator(config)

        writer = video_writer(cap, output_tmp.name)
        results = video_processor(cap, queue_manager, writer)

        # Release before upload: VideoWriter buffers frames until closed,
        # so the file on disk isn't complete until this happens.
        release_cap(cap)
        cap = None
        writer.release()
        writer = None

        annotated_video_url = upload_annotated_video(output_tmp.name)

        return {
            "queue_count": results.queue_count,
            "total_tracks": results.total_tracks,
            "annotated_video_url": annotated_video_url,
        }
    finally:
        if cap is not None:
            release_cap(cap)
        if writer is not None:
            writer.release()
        os.unlink(input_tmp.name)
        os.unlink(output_tmp.name)


runpod.serverless.start({"handler": handler})
