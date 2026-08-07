import cv2
from ultralytics import solutions
import yaml
import logging
import numpy as np
import uuid
import time
import statistics

def load_config(file_path="app/queue.yaml"):
    # Read video file
    with open(file_path, "r") as file:
        try: 
            config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            logging.error(f"Error reading YAML file: {e}")
            return None
    return config

def cap_check(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        logging.error("Could not open webcam.")
        return None
    return cap

def release_cap(check):
    if check:
        check.release()
        #cv2.destroyAllWindows()
        return "cap released"
    else:
        logging.error("CAP IS NOT OPENED")
        return None

def frame_stride(capture, target_fps):
    """How many source frames to advance per processed frame.

    Returns (stride, effective_fps). Single source of truth so the writer's
    playback rate always matches the rate frames are actually processed at.
    """
    src_fps = capture.get(cv2.CAP_PROP_FPS) or float(target_fps)
    stride = max(1, round(src_fps / target_fps))
    return stride, src_fps / stride

def video_writer(capture, path, target_fps):
    # Create a VideoWriter object
    w, h = (int(capture.get(x))
                    for x in (
                    cv2.CAP_PROP_FRAME_WIDTH,
                    cv2.CAP_PROP_FRAME_HEIGHT))

    # Must be the processed rate, not the source rate, or playback runs fast.
    _, effective_fps = frame_stride(capture, target_fps)

    writer = cv2.VideoWriter(path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                effective_fps, (w, h))
    return writer

def model_creator(config):
    # Initialize queue manager object
    queuemanager = solutions.QueueManager(
        show=False,  # display the output
        model=config["MODEL"],  # path to the YOLO26 model file
        region=config["QUEUE_REGION"],
        tracker=config["TRACKER"],
        conf=config["CONF"],
        iou=config["IOU"],
        classes=[0]
    )
    return queuemanager

def video_processor(cap, queue_manager, writer, job_id, target_fps=10, annotate=True):
    # Process video
    region = np.array(queue_manager.region, dtype=np.int32)  # built once, reused every frame
    src_fps = cap.get(cv2.CAP_PROP_FPS) or float(target_fps)
    stride, effective_fps = frame_stride(cap, target_fps)
    logging.info(
        f"Source {src_fps:.2f}fps -> processing every {stride} frame(s) "
        f"= {effective_fps:.2f}fps"
    )

    src_idx = 0     # position in the source video (drives real video time)
    frame_idx = 0   # frames actually run through the model

    open_tracks = {}       # track_id -> entry_time (video-relative seconds)
    completed_tracks = []  # [{job_id, track_id, dwell_seconds}, ...]
    snapshots = []         # [{job_id, timestamp, queue_count, inside_ids, outside_ids}, ...]

    # --- profiling accumulators (cheap; one summary logged at the end) ---
    t_decode = t_track = t_roi = t_write = t_log = 0.0
    ms_pre = ms_inf = ms_post = 0.0
    unique_ids = set()
    peak_concurrent = 0
    t_loop_start = time.perf_counter()

    while True:
        # grab() advances the decoder without converting the frame to BGR;
        # retrieve() does that conversion. Skipped frames never pay for it.
        a = time.perf_counter()
        if not cap.grab():
            t_decode += time.perf_counter() - a
            logging.info("Video frame is empty or processing is complete.")
            break

        frame_pos = src_idx
        src_idx += 1
        if frame_pos % stride != 0:
            t_decode += time.perf_counter() - a
            continue

        success, im0 = cap.retrieve()
        t_decode += time.perf_counter() - a
        if not success:
            logging.info("Video frame is empty or processing is complete.")
            break

        a = time.perf_counter()
        results = queue_manager(im0)
        t_track += time.perf_counter() - a
        # Ultralytics' own per-stage numbers for the frame it just ran.
        speed = getattr(queue_manager.tracks, "speed", None)
        if speed:
            ms_pre += speed.get("preprocess", 0.0)
            ms_inf += speed.get("inference", 0.0)
            ms_post += speed.get("postprocess", 0.0)

        # Real position in the video, so timings stay correct despite skipping.
        current_time = frame_pos / src_fps

        a = time.perf_counter()
        inside_ids = []
        for track_id, box in zip(queue_manager.track_ids, queue_manager.boxes):
            point = (float((box[0] + box[2]) / 2), float(box[3]))  # bottom-center of box
            if cv2.pointPolygonTest(region, point, False) >= 0:
                inside_ids.append(track_id)

        # Entries: newly inside the ROI, not already being tracked as "in".
        for track_id in inside_ids:
            if track_id not in open_tracks:
                open_tracks[track_id] = current_time

        # Exits: was inside, no longer is.
        for track_id in list(open_tracks.keys()):
            if track_id not in inside_ids:
                entry_time = open_tracks.pop(track_id)
                completed_tracks.append({
                    "job_id": job_id,
                    "track_id": track_id,
                    "dwell_seconds": current_time - entry_time,
                })

        # One snapshot per processed frame (every 0.1s at target_fps=10) -
        # fine enough for the dashboard's hover inspector to be meaningful.
        outside_ids = [tid for tid in queue_manager.track_ids if tid not in inside_ids]
        snapshots.append({
            "job_id": job_id,
            "timestamp": current_time,
            "queue_count": len(inside_ids),
            "inside_ids": inside_ids,
            "outside_ids": outside_ids,
        })
        t_roi += time.perf_counter() - a

        unique_ids.update(inside_ids)
        peak_concurrent = max(peak_concurrent, len(inside_ids))

        # Per-frame logging is itself a measurable cost at 60fps, so sample it.
        a = time.perf_counter()
        if frame_idx % 100 == 0:
            logging.info(f"Frame {frame_idx}: inside ROI = {inside_ids}")
        t_log += time.perf_counter() - a

        a = time.perf_counter()
        if annotate:
            writer.write(results.plot_im)  # write the processed frame.
        t_write += time.perf_counter() - a
        frame_idx += 1

    t_loop = time.perf_counter() - t_loop_start

    # Video ended while these were still inside the ROI - close them out
    # using the last processed frame as their exit, so they aren't silently
    # dropped from the data.
    final_time = src_idx / src_fps
    for track_id, entry_time in open_tracks.items():
        completed_tracks.append({
            "job_id": job_id,
            "track_id": track_id,
            "dwell_seconds": final_time - entry_time,
        })

    _log_profile(job_id, frame_idx, src_idx, src_fps, effective_fps, t_loop,
                 t_decode, t_track, t_roi, t_write, t_log,
                 ms_pre, ms_inf, ms_post,
                 unique_ids, peak_concurrent, completed_tracks)

    return snapshots, completed_tracks


def _log_profile(job_id, frames, src_frames, src_fps, effective_fps, t_loop,
                 t_decode, t_track, t_roi, t_write, t_log,
                 ms_pre, ms_inf, ms_post, unique_ids, peak_concurrent,
                 completed_tracks):
    """One screenshot-friendly block of cost + tracking-quality numbers."""
    if frames == 0:
        logging.info("PROFILE: no frames processed")
        return

    def per_frame(total_seconds):
        return total_seconds / frames * 1000.0

    video_seconds = src_frames / src_fps if src_fps else 0.0
    dwells = [t["dwell_seconds"] for t in completed_tracks]
    short = [d for d in dwells if d < 1.0]

    lines = [
        "",
        "=" * 62,
        f"PROFILE  job={job_id}",
        f"  video={video_seconds:.1f}s  src_frames={src_frames} @ {src_fps:.2f}fps",
        f"  processed={frames} frames @ {effective_fps:.2f}fps  "
        f"(skipped {src_frames - frames} = {(1 - frames / src_frames) * 100 if src_frames else 0:.0f}%)",
        f"  loop wall={t_loop:.1f}s   speed={video_seconds / t_loop if t_loop else 0:.2f}x realtime"
        f"   ({per_frame(t_loop):.1f} ms/processed frame)",
        "-" * 62,
        "  STAGE                ms/frame     % of loop",
        f"    decode          {per_frame(t_decode):9.2f}   {t_decode / t_loop * 100:7.1f}%",
        f"    track+solution  {per_frame(t_track):9.2f}   {t_track / t_loop * 100:7.1f}%",
        f"        preprocess  {ms_pre / frames:9.2f}",
        f"        inference   {ms_inf / frames:9.2f}",
        f"        postprocess {ms_post / frames:9.2f}",
        f"        tracker/etc {per_frame(t_track) - (ms_pre + ms_inf + ms_post) / frames:9.2f}",
        f"    roi_check       {per_frame(t_roi):9.2f}   {t_roi / t_loop * 100:7.1f}%",
        f"    annotate+write  {per_frame(t_write):9.2f}   {t_write / t_loop * 100:7.1f}%",
        f"    logging         {per_frame(t_log):9.2f}   {t_log / t_loop * 100:7.1f}%",
        "-" * 62,
        "  TRACKING QUALITY",
        f"    unique track ids seen in ROI : {len(unique_ids)}",
        f"    peak concurrent in ROI       : {peak_concurrent}",
        f"    completed dwell records      : {len(dwells)}",
    ]
    if dwells:
        pct = len(short) / len(dwells) * 100
        lines += [
            f"    dwell < 1.0s (likely churn)  : {len(short)}  ({pct:.0f}%)",
            f"    median dwell                 : {statistics.median(dwells):.2f}s",
            f"    max dwell                    : {max(dwells):.2f}s",
        ]
    lines += ["=" * 62, ""]
    # One logging.info() call per line, not one call for the whole block -
    # RunPod's log pipeline appears to silently truncate a single very long
    # multi-line log entry (this block went missing past the header on a
    # real run). Many small entries survive where one large one didn't.
    for line in lines:
        logging.info(line)

def queue_management():
    config = load_config()
    target_fps = config.get("TARGET_FPS", 10)
    cap = cap_check(config["CAP"])
    writer = video_writer(cap, config["VIDEO_WRITER"], target_fps)
    queue_manager = model_creator(config)
    job_id = str(uuid.uuid4())
    snapshots, completed_tracks = video_processor(
        cap, queue_manager, writer, job_id, target_fps=target_fps
    )
    logging.info("Queue management processing complete.")

    release_cap(cap)
    writer.release()
    return snapshots, completed_tracks

if __name__ == "__main__":
    queue_management()