import json
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Query, Request

from db import connect
from models import JobSummary, SnapshotIn, SnapshotOut, TrackIn, TrackOut


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One connection, opened once, reused for every request. SQLite only
    # allows one writer at a time regardless, so pooling multiple
    # connections wouldn't buy real write concurrency here.
    app.state.db = await connect()
    yield
    await app.state.db.close()


app = FastAPI(title="queue-analysis storage-api", lifespan=lifespan)


def get_db(request: Request):
    return request.app.state.db


@app.get("/jobs", response_model=list[JobSummary])
async def list_jobs(db=Depends(get_db)):
    """One row per processing run, newest first."""
    cursor = await db.execute(
        """
        SELECT job_id, COUNT(*), MAX(queue_count), AVG(queue_count),
               MAX(timestamp), MIN(id)
        FROM snapshots
        GROUP BY job_id
        ORDER BY MIN(id) DESC
        """
    )
    snapshot_rows = await cursor.fetchall()

    cursor = await db.execute(
        """
        SELECT job_id, COUNT(*), AVG(dwell_seconds), MAX(dwell_seconds)
        FROM tracks
        GROUP BY job_id
        """
    )
    track_stats = {r[0]: (r[1], r[2], r[3]) for r in await cursor.fetchall()}

    jobs = []
    for job_id, snap_count, peak, avg_q, duration, _ in snapshot_rows:
        count, avg_dwell, max_dwell = track_stats.get(job_id, (0, 0.0, 0.0))
        jobs.append(
            JobSummary(
                job_id=job_id,
                snapshot_count=snap_count,
                track_count=count,
                peak_queue=peak or 0,
                avg_queue=avg_q or 0.0,
                duration_seconds=duration or 0.0,
                avg_dwell=avg_dwell or 0.0,
                max_dwell=max_dwell or 0.0,
            )
        )
    return jobs


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