# handler.py
import os
import tempfile

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

# Loaded once at cold start, reused across every request on this worker.
config = load_config()
queue_manager = model_creator(config)


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

        writer = video_writer(cap, output_tmp.name)
        results = video_processor(cap, queue_manager, writer)

        return {
            "queue_count": results.queue_count,
            "total_tracks": results.total_tracks,
        }
    finally:
        if cap is not None:
            release_cap(cap)
        if writer is not None:
            writer.release()
        os.unlink(input_tmp.name)
        os.unlink(output_tmp.name)


runpod.serverless.start({"handler": handler})
