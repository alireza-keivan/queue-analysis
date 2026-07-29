# handler.py
import runpod, requests, tempfile, os
from ultralytics import YOLO   # or however your pipeline loads

# Load the model ONCE at container start, not per request
model = YOLO("weights.pt")

def handler(event):
    job_input = event["input"]
    video_url  = job_input.get("video_url")
    sample_fps = job_input.get("sample_fps", 2)   # sample frames, don't process all 108k
    if not video_url:
        return {"error": "No 'video_url' provided."}

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    try:
        with requests.get(video_url, stream=True) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=1 << 20):
                tmp.write(chunk)
        tmp.close()

        # --- your existing queue-analysis logic here ---
        # detection + tracking + per-ROI counting -> queue length / wait times
        results = analyze_queue(model, tmp.name, sample_fps)

        return {"results": results}
    finally:
        os.unlink(tmp.name)

runpod.serverless.start({"handler": handler})