"""Dashboard service.

Owns the UI and job orchestration. Never touches the database directly - all
data goes through storage-api, which remains the sole owner of the SQLite file.

Browser -> dashboard -> storage-api -> SQLite
                     -> RunPod (GPU processing)
"""
import asyncio
import logging
import os
import time

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(message)s")

STORAGE_API_URL = os.environ.get("STORAGE_API_URL", "http://storage-api:8000")
STORAGE_API_SECRET = os.environ.get("STORAGE_API_SECRET", "")
STORAGE_HEADERS = {"X-API-Key": STORAGE_API_SECRET}
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "")
# Optional: n8n's webhook URL for the alert workflow. If unset, jobs simply
# aren't announced to n8n - nothing else changes.
N8N_WEBHOOK_URL = os.environ.get("N8N_WEBHOOK_URL", "")

# A cold worker plus a full video can take minutes; the browser waits on this.
RUNPOD_TIMEOUT = 900.0

app = FastAPI(title="queue-analysis dashboard")


class SubmitRequest(BaseModel):
    video_url: str
    target_fps: int = 10
    annotate: bool = True
    # [[x, y], ...] in source-video pixel coords, picked in the browser ROI
    # editor. None means "use queue.yaml's default region" - never written to
    # any file, just passed through on this one job's request.
    region: list[list[float]] | None = None


@app.get("/api/health")
async def health():
    """Reports whether the pieces the UI depends on are actually reachable."""
    storage_ok = False
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{STORAGE_API_URL}/jobs", headers=STORAGE_HEADERS)
            storage_ok = r.status_code == 200
    except httpx.HTTPError:
        pass
    return {
        "storage_api": storage_ok,
        "runpod_configured": bool(RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID),
    }


@app.get("/api/jobs")
async def list_jobs():
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f"{STORAGE_API_URL}/jobs", headers=STORAGE_HEADERS)
        r.raise_for_status()
        return r.json()


@app.get("/api/jobs/{job_id}")
async def job_detail(job_id: str):
    async with httpx.AsyncClient(timeout=30.0) as client:
        snapshots = await client.get(
            f"{STORAGE_API_URL}/snapshots", params={"job_id": job_id}, headers=STORAGE_HEADERS
        )
        tracks = await client.get(
            f"{STORAGE_API_URL}/tracks", params={"job_id": job_id}, headers=STORAGE_HEADERS
        )
        snapshots.raise_for_status()
        tracks.raise_for_status()
    return {
        "job_id": job_id,
        "snapshots": snapshots.json(),
        "tracks": tracks.json(),
    }


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.delete(f"{STORAGE_API_URL}/jobs/{job_id}", headers=STORAGE_HEADERS)
        r.raise_for_status()
        return r.json()


@app.post("/api/jobs")
async def submit_job(req: SubmitRequest):
    """Run a video through RunPod, then persist the results via storage-api.

    This is submit_job.py's logic, moved server-side so the UI can drive it.
    """
    if not RUNPOD_API_KEY or not RUNPOD_ENDPOINT_ID:
        raise HTTPException(
            status_code=503,
            detail="RUNPOD_API_KEY / RUNPOD_ENDPOINT_ID are not set on the dashboard service.",
        )

    auth = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    deadline = time.monotonic() + RUNPOD_TIMEOUT

    async with httpx.AsyncClient(timeout=RUNPOD_TIMEOUT) as client:
        try:
            response = await client.post(
                f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/runsync",
                headers=auth,
                json={
                    "input": {
                        "video_url": req.video_url,
                        "target_fps": req.target_fps,
                        "annotate": req.annotate,
                        **({"region": req.region} if req.region else {}),
                    }
                },
            )
        except httpx.HTTPError as exc:
            logging.error(f"RunPod unreachable: {exc}")
            raise HTTPException(status_code=502, detail=f"RunPod unreachable: {exc}")

        if response.status_code != 200:
            logging.error(f"RunPod returned {response.status_code}: {response.text[:800]}")
            raise HTTPException(
                status_code=502,
                detail=f"RunPod returned {response.status_code}: {response.text[:400]}",
            )

        body = response.json()

        # /runsync only blocks up to RunPod's own internal wait window. A
        # cold worker plus a slow job can outlast that window, in which case
        # it hands back whatever status it has (IN_QUEUE/IN_PROGRESS) instead
        # of the final result - that's not a failure, just not finished yet.
        # Poll /status until it actually reaches a terminal state.
        job_id_for_poll = body.get("id")
        while body.get("status") in ("IN_QUEUE", "IN_PROGRESS") and job_id_for_poll:
            if time.monotonic() > deadline:
                logging.error(f"Timed out waiting for RunPod job: {str(body)[:800]}")
                raise HTTPException(
                    status_code=504,
                    detail=f"Timed out waiting for RunPod job to finish: {str(body)[:400]}",
                )
            await asyncio.sleep(3.0)
            poll = await client.get(
                f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/status/{job_id_for_poll}",
                headers=auth,
            )
            poll.raise_for_status()
            body = poll.json()

    if body.get("status") != "COMPLETED" or "output" not in body:
        logging.error(f"Job did not complete: {str(body)[:800]}")
        raise HTTPException(
            status_code=502, detail=f"Job did not complete: {str(body)[:400]}"
        )

    result = body["output"]
    if "error" in result:
        logging.error(f"Handler error: {result['error']}")
        raise HTTPException(status_code=400, detail=f"Handler error: {result['error']}")

    # Relay into storage-api. Done after processing succeeded, so a storage
    # failure here doesn't hide a successful (already paid for) GPU run.
    # Bulk endpoints: one round trip and one commit per table instead of one
    # of each per row (a few hundred snapshots per job otherwise means a few
    # hundred HTTP calls and SQLite commits).
    snapshots = result.get("snapshots", [])
    tracks = result.get("tracks", [])
    async with httpx.AsyncClient(timeout=60.0) as client:
        if snapshots:
            r = await client.post(f"{STORAGE_API_URL}/snapshots/bulk", json=snapshots, headers=STORAGE_HEADERS)
            r.raise_for_status()
        if tracks:
            r = await client.post(f"{STORAGE_API_URL}/tracks/bulk", json=tracks, headers=STORAGE_HEADERS)
            r.raise_for_status()

    # Ping n8n so its alert workflow can react. Best-effort: n8n fetches its
    # own data from storage-api once triggered, so this carries no payload
    # beyond an identifier - and a notification failure here shouldn't hide
    # an otherwise-successful, already-paid-for GPU job.
    if N8N_WEBHOOK_URL:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(N8N_WEBHOOK_URL, json={"job_id": result.get("job_id")})
        except httpx.HTTPError:
            pass

    return {
        "job_id": result.get("job_id"),
        "snapshot_count": len(result.get("snapshots", [])),
        "track_count": len(result.get("tracks", [])),
        "annotated_video_url": result.get("annotated_video_url"),
    }


@app.get("/")
async def index():
    return FileResponse("static/index.html")


app.mount("/", StaticFiles(directory="static"), name="static")