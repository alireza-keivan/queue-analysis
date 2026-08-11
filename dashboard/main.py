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
from fastapi import FastAPI, HTTPException, Query
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
# Optional: a Google Sheet published-to-web embed URL (File > Share > Publish
# to web, embed tab) - shown as-is in the dashboard if set. This is the same
# sheet n8n's workflow already logs job results into.
GOOGLE_SHEET_EMBED_URL = os.environ.get("GOOGLE_SHEET_EMBED_URL", "")

# A cold worker plus a full video can take minutes; the browser waits on this.
RUNPOD_TIMEOUT = 900.0

app = FastAPI(title="queue-analysis dashboard")


class SubmitRequest(BaseModel):
    video_url: str
    target_fps: int = 10
    # False by default - rendering/encoding/uploading the annotated video was
    # measured at +85% job time for output the metrics never use. The UI
    # checkbox is unchecked to match; tick it when you want the video.
    annotate: bool = False
    # [[x, y], ...] in source-video pixel coords, picked in the browser ROI
    # editor. None means "use queue.yaml's default region" - never written to
    # any file, just passed through on this one job's request.
    region: list[list[float]] | None = None
    # Detection confidence / NMS IOU thresholds. Same "None = queue.yaml
    # default, otherwise a per-job override" pattern as region.
    conf: float | None = None
    iou: float | None = None


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


async def _storage(client: httpx.AsyncClient, method: str, path: str, **kwargs) -> httpx.Response:
    """Call storage-api and turn either failure mode into a legible 502.

    Both modes previously reached the browser as a bare "500 Internal Server
    Error" with no detail: an unreachable storage-api raised httpx.ConnectError,
    and an error *response* raised via raise_for_status - in both cases the
    real cause (`no such column: inside_ids`, or "the service isn't running")
    was visible only in a server log the user never sees.
    """
    try:
        response = await client.request(method, f"{STORAGE_API_URL}{path}",
                                        headers=STORAGE_HEADERS, **kwargs)
    except httpx.HTTPError as exc:
        logging.error(f"storage-api unreachable for {method} {path}: {exc}")
        raise HTTPException(
            status_code=502,
            detail=f"storage-api is unreachable at {STORAGE_API_URL} - is the service running?",
        )

    if not response.is_success:
        body = response.text[:300]
        logging.error(f"storage-api {method} {path} failed: {response.status_code} {body}")
        raise HTTPException(
            status_code=502,
            detail=f"storage-api {method} {path} failed ({response.status_code}): {body}",
        )
    return response


@app.get("/api/config")
async def config():
    """Static UI config the frontend can't get from anywhere else - not
    secrets (those stay server-side), just "should this panel render"."""
    return {"google_sheet_embed_url": GOOGLE_SHEET_EMBED_URL or None}


@app.get("/api/jobs")
async def list_jobs(
    since: str | None = Query(default=None),
    until: str | None = Query(default=None),
):
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await _storage(client, "GET", "/jobs", params={
            k: v for k, v in {"since": since, "until": until}.items() if v is not None
        })
        return r.json()


@app.get("/api/jobs/{job_id}")
async def job_detail(job_id: str):
    async with httpx.AsyncClient(timeout=30.0) as client:
        snapshots = await _storage(client, "GET", "/snapshots", params={"job_id": job_id})
        tracks = await _storage(client, "GET", "/tracks", params={"job_id": job_id})
    return {
        "job_id": job_id,
        "snapshots": snapshots.json(),
        "tracks": tracks.json(),
    }


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await _storage(client, "DELETE", f"/jobs/{job_id}")
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
                        **({"conf": req.conf} if req.conf is not None else {}),
                        **({"iou": req.iou} if req.iou is not None else {}),
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

    # Relay into storage-api. Bulk endpoints: one round trip and one commit
    # per table instead of one of each per row (a few hundred snapshots per
    # job otherwise means a few hundred HTTP calls and SQLite commits).
    #
    # The GPU run is already finished and already paid for by this point, so
    # a storage failure must say so explicitly rather than surfacing as a
    # bare 500 that looks like the whole job failed - the expensive half
    # succeeded and only persistence was lost.
    snapshots = result.get("snapshots", [])
    tracks = result.get("tracks", [])
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            if snapshots:
                await _storage(client, "POST", "/snapshots/bulk", json=snapshots)
            if tracks:
                await _storage(client, "POST", "/tracks/bulk", json=tracks)
    except HTTPException as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                f"GPU job {result.get('job_id')} COMPLETED "
                f"({len(snapshots)} snapshots, {len(tracks)} tracks) but the results "
                f"could not be saved: {exc.detail}"
            ),
        )

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
        # Per-stage timings straight from the GPU worker. Passed through so
        # the cost of a job is inspectable from the browser/API instead of
        # only from RunPod's log console.
        "cost": result.get("cost"),
        "profile": result.get("profile"),
    }


@app.get("/")
async def index():
    return FileResponse("static/index.html")


app.mount("/", StaticFiles(directory="static"), name="static")