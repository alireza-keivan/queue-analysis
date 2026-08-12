"""RunPod Serverless handler for the CPU-only endpoint.

Deploy this on a CPU worker type (no GPU attached, much cheaper per second
than the GPU endpoint in handler.py). It only ever does pure-Python/CPU work
- currently the track-diagnostic churn scoring - so it must never import cv2,
torch, or ultralytics; see app/track_diagnostics.py's lazy-import comment for
why that import boundary matters.

Job input:
    {
        "task": "score_diagnostic",
        "history": [...],   # from the GPU handler's diagnostic-mode output
        "stride": int,
        "src_fps": float,
    }
"""
import logging

import runpod

from app.track_diagnostics import score_track_churn


def handler(event):
    job_input = event["input"]
    task = job_input.get("task")

    if task == "score_diagnostic":
        history = [
            {int(tid): tuple(box) for tid, box in frame.items()}
            for frame in job_input["history"]
        ]
        # frame_size is optional: older collection payloads predate it, and
        # score_track_churn degrades honestly without it (reports the split as
        # unavailable rather than guessing frame bounds from box coordinates).
        frame_size = job_input.get("frame_size")
        if frame_size:
            frame_size = tuple(frame_size)
        report = score_track_churn(
            history, job_input["stride"], job_input["src_fps"],
            frame_size=frame_size,
        )
        logging.info(f"Scored diagnostic: {report}")
        return {"diagnostic": report}

    return {"error": f"Unknown task '{task}'. Supported: 'score_diagnostic'."}


runpod.serverless.start({"handler": handler})