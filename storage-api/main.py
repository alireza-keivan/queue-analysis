import json
import os
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request

from db import connect
from models import JobSummary, SnapshotIn, SnapshotOut, TrackIn, TrackOut

# No default: an unset secret must fail closed, not silently allow every
# request through. See verify_api_key below.
API_SECRET = os.environ.get("STORAGE_API_SECRET", "")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One connection, opened once, reused for every request. SQLite only
    # allows one writer at a time regardless, so pooling multiple
    # connections wouldn't buy real write concurrency here.
    app.state.db = await connect()
    yield
    await app.state.db.close()


async def verify_api_key(x_api_key: str = Header(default="")):
    # compare_digest avoids leaking the secret's length/prefix through
    # response-timing differences.
    if not API_SECRET or not secrets.compare_digest(x_api_key, API_SECRET):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key")


app = FastAPI(
    title="queue-analysis storage-api",
    lifespan=lifespan,
    dependencies=[Depends(verify_api_key)],  # applies to every route below
)


def get_db(request: Request):
    return request.app.state.db


async def _fetch_job_summaries(
    db, job_id: str | None = None,
    since: str | None = None, until: str | None = None,
) -> list[JobSummary]:
    # job_id is an exact match; since/until filter on created_at (inclusive)
    # and only ever come from list_jobs, never combined with job_id in
    # practice, but there's no reason to forbid it.
    clauses, params = [], []
    if job_id is not None:
        clauses.append("job_id = ?")
        params.append(job_id)
    if since is not None:
        clauses.append("created_at >= ?")
        params.append(since)
    if until is not None:
        clauses.append("created_at <= ?")
        params.append(until)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params = tuple(params)

    cursor = await db.execute(
        f"""
        SELECT job_id, COUNT(*), MAX(queue_count), AVG(queue_count),
               MAX(timestamp), MIN(id), MIN(created_at)
        FROM snapshots
        {where}
        GROUP BY job_id
        ORDER BY MIN(id) DESC
        """,
        params,
    )
    snapshot_rows = await cursor.fetchall()

    # Tracks carry no created_at of their own - they belong to the same job_id
    # as the snapshots already matched above, so re-filtering by date here
    # would just be filtering on a value that isn't in this table.
    track_where = "WHERE job_id = ?" if job_id is not None else ""
    track_params = (job_id,) if job_id is not None else ()
    cursor = await db.execute(
        f"""
        SELECT job_id, COUNT(*), AVG(dwell_seconds), MAX(dwell_seconds)
        FROM tracks
        {track_where}
        GROUP BY job_id
        """,
        track_params,
    )
    track_stats = {r[0]: (r[1], r[2], r[3]) for r in await cursor.fetchall()}

    jobs = []
    for jid, snap_count, peak, avg_q, duration, _, created_at in snapshot_rows:
        count, avg_dwell, max_dwell = track_stats.get(jid, (0, 0.0, 0.0))
        jobs.append(
            JobSummary(
                job_id=jid,
                snapshot_count=snap_count,
                track_count=count,
                peak_queue=peak or 0,
                avg_queue=avg_q or 0.0,
                duration_seconds=duration or 0.0,
                avg_dwell=avg_dwell or 0.0,
                max_dwell=max_dwell or 0.0,
                created_at=created_at,
            )
        )
    return jobs


@app.get("/jobs", response_model=list[JobSummary])
async def list_jobs(
    since: str | None = Query(default=None, description="ISO date/datetime, inclusive lower bound on created_at"),
    until: str | None = Query(default=None, description="ISO date/datetime, inclusive upper bound on created_at"),
    db=Depends(get_db),
):
    """One row per processing run, newest first. since/until filter by the
    job's created_at (string comparison - both are 'YYYY-MM-DD[ HH:MM:SS]',
    which sorts correctly as text)."""
    return await _fetch_job_summaries(db, since=since, until=until)


@app.get("/jobs/{job_id}", response_model=JobSummary)
async def get_job(job_id: str, db=Depends(get_db)):
    """Aggregate for exactly one job. This is what n8n's alert workflow
    should call - /jobs returns every job ever recorded, which fans out
    into one execution per historical job instead of just the new one."""
    jobs = await _fetch_job_summaries(db, job_id=job_id)
    if not jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[0]


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str, db=Depends(get_db)):
    """Remove every row belonging to one run."""
    cur_s = await db.execute("DELETE FROM snapshots WHERE job_id = ?", (job_id,))
    cur_t = await db.execute("DELETE FROM tracks WHERE job_id = ?", (job_id,))
    await db.commit()
    return {"job_id": job_id, "snapshots_deleted": cur_s.rowcount,
            "tracks_deleted": cur_t.rowcount}


@app.post("/snapshots", response_model=SnapshotOut)
async def create_snapshot(snapshot: SnapshotIn, db=Depends(get_db)):
    cursor = await db.execute(
        "INSERT INTO snapshots (job_id, timestamp, queue_count, inside_ids, outside_ids) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            snapshot.job_id, snapshot.timestamp, snapshot.queue_count,
            json.dumps(snapshot.inside_ids), json.dumps(snapshot.outside_ids),
        ),
    )
    await db.commit()
    return SnapshotOut(id=cursor.lastrowid, **snapshot.model_dump())


@app.post("/snapshots/bulk")
async def create_snapshots_bulk(snapshots: list[SnapshotIn], db=Depends(get_db)):
    """Same insert as POST /snapshots, but one round trip and one commit for
    the whole batch instead of one of each per row - a full job's worth of
    snapshots (hundreds) was measured at ~7ms/row through the single-row
    endpoint, almost all of it per-request overhead, not the insert itself."""
    if not snapshots:
        return {"inserted": 0}
    await db.executemany(
        "INSERT INTO snapshots (job_id, timestamp, queue_count, inside_ids, outside_ids) "
        "VALUES (?, ?, ?, ?, ?)",
        [
            (s.job_id, s.timestamp, s.queue_count,
             json.dumps(s.inside_ids), json.dumps(s.outside_ids))
            for s in snapshots
        ],
    )
    await db.commit()
    return {"inserted": len(snapshots)}


@app.get("/snapshots", response_model=list[SnapshotOut])
async def list_snapshots(job_id: str | None = Query(default=None), db=Depends(get_db)):
    cols = "id, job_id, timestamp, queue_count, inside_ids, outside_ids"
    if job_id is not None:
        cursor = await db.execute(
            f"SELECT {cols} FROM snapshots WHERE job_id = ?", (job_id,),
        )
    else:
        cursor = await db.execute(f"SELECT {cols} FROM snapshots")
    rows = await cursor.fetchall()
    return [
        SnapshotOut(
            id=r[0], job_id=r[1], timestamp=r[2], queue_count=r[3],
            inside_ids=json.loads(r[4]), outside_ids=json.loads(r[5]),
        )
        for r in rows
    ]


@app.post("/tracks", response_model=TrackOut)
async def create_track(track: TrackIn, db=Depends(get_db)):
    cursor = await db.execute(
        "INSERT INTO tracks (job_id, track_id, dwell_seconds) VALUES (?, ?, ?)",
        (track.job_id, track.track_id, track.dwell_seconds),
    )
    await db.commit()
    return TrackOut(id=cursor.lastrowid, **track.model_dump())


@app.post("/tracks/bulk")
async def create_tracks_bulk(tracks: list[TrackIn], db=Depends(get_db)):
    """See create_snapshots_bulk - same reasoning, applied to tracks."""
    if not tracks:
        return {"inserted": 0}
    await db.executemany(
        "INSERT INTO tracks (job_id, track_id, dwell_seconds) VALUES (?, ?, ?)",
        [(t.job_id, t.track_id, t.dwell_seconds) for t in tracks],
    )
    await db.commit()
    return {"inserted": len(tracks)}


@app.get("/tracks", response_model=list[TrackOut])
async def list_tracks(job_id: str | None = Query(default=None), db=Depends(get_db)):
    if job_id is not None:
        cursor = await db.execute(
            "SELECT id, job_id, track_id, dwell_seconds FROM tracks WHERE job_id = ?",
            (job_id,),
        )
    else:
        cursor = await db.execute("SELECT id, job_id, track_id, dwell_seconds FROM tracks")
    rows = await cursor.fetchall()
    return [
        TrackOut(id=r[0], job_id=r[1], track_id=r[2], dwell_seconds=r[3])
        for r in rows
    ]