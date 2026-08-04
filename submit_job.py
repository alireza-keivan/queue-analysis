"""Submit a video to the RunPod endpoint, then send its results into storage-api.

Usage:
    RUNPOD_API_KEY=...        \
    RUNPOD_ENDPOINT_ID=...    \
    STORAGE_API_URL=http://localhost:8000  \
        python submit_job.py <video_url>

STORAGE_API_URL defaults to http://localhost:8000 if not set.
"""
import os
import sys

import requests

RUNPOD_API_KEY = os.environ["RUNPOD_API_KEY"]
RUNPOD_ENDPOINT_ID = os.environ["RUNPOD_ENDPOINT_ID"]
STORAGE_API_URL = os.environ.get("STORAGE_API_URL", "http://localhost:8000")


def run_job(video_url):
    response = requests.post(
        f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/runsync",
        headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
        json={"input": {"video_url": video_url}},
        timeout=600,
    )
    response.raise_for_status()
    body = response.json()

    if body.get("status") != "COMPLETED":
        raise RuntimeError(f"RunPod job did not complete cleanly: {body}")
    if "output" not in body:
        raise RuntimeError(f"Unexpected response shape, no 'output' key: {body}")

    return body["output"]


def relay_to_storage(result):
    job_id = result["job_id"]
    sent_snapshots = 0
    sent_tracks = 0

    for snapshot in result["snapshots"]:
        r = requests.post(f"{STORAGE_API_URL}/snapshots", json=snapshot)
        r.raise_for_status()
        sent_snapshots += 1

    for track in result["tracks"]:
        r = requests.post(f"{STORAGE_API_URL}/tracks", json=track)
        r.raise_for_status()
        sent_tracks += 1

    print(f"job {job_id}: sent {sent_snapshots} snapshots, {sent_tracks} tracks "
          f"to {STORAGE_API_URL}")
    if result.get("annotated_video_url"):
        print(f"annotated video: {result['annotated_video_url']}")


def main():
    if len(sys.argv) != 2:
        print("usage: python submit_job.py <video_url>")
        sys.exit(1)
    video_url = sys.argv[1]

    print("submitting job to RunPod...")
    result = run_job(video_url)
    print(f"job finished: {len(result['snapshots'])} snapshots, "
          f"{len(result['tracks'])} tracks")

    relay_to_storage(result)


if __name__ == "__main__":
    main()