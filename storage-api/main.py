from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Query, Request

from db import connect
from models import SnapshotIn, SnapshotOut, TrackIn, TrackOut


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


@app.post("/snapshots", response_model=SnapshotOut)
async def create_snapshot(snapshot: SnapshotIn, db=Depends(get_db)):
    cursor = await db.execute(
        "INSERT INTO snapshots (job_id, timestamp, queue_count) VALUES (?, ?, ?)",
        (snapshot.job_id, snapshot.timestamp.isoformat(), snapshot.queue_count),
    )
    await db.commit()
    return SnapshotOut(id=cursor.lastrowid, **snapshot.model_dump())


@app.get("/snapshots", response_model=list[SnapshotOut])
async def list_snapshots(job_id: str | None = Query(default=None), db=Depends(get_db)):
    if job_id is not None:
        cursor = await db.execute(
            "SELECT id, job_id, timestamp, queue_count FROM snapshots WHERE job_id = ?",
            (job_id,),
        )
    else:
        cursor = await db.execute("SELECT id, job_id, timestamp, queue_count FROM snapshots")
    rows = await cursor.fetchall()
    return [SnapshotOut(id=r[0], job_id=r[1], timestamp=r[2], queue_count=r[3]) for r in rows]


@app.post("/tracks", response_model=TrackOut)
async def create_track(track: TrackIn, db=Depends(get_db)):
    cursor = await db.execute(
        "INSERT INTO tracks (job_id, track_id, entry_time, exit_time) VALUES (?, ?, ?, ?)",
        (track.job_id, track.track_id, track.entry_time.isoformat(), track.exit_time.isoformat()),
    )
    await db.commit()
    return TrackOut(id=cursor.lastrowid, **track.model_dump())


@app.get("/tracks", response_model=list[TrackOut])
async def list_tracks(job_id: str | None = Query(default=None), db=Depends(get_db)):
    if job_id is not None:
        cursor = await db.execute(
            "SELECT id, job_id, track_id, entry_time, exit_time FROM tracks WHERE job_id = ?",
            (job_id,),
        )
    else:
        cursor = await db.execute("SELECT id, job_id, track_id, entry_time, exit_time FROM tracks")
    rows = await cursor.fetchall()
    return [
        TrackOut(id=r[0], job_id=r[1], track_id=r[2], entry_time=r[3], exit_time=r[4])
        for r in rows
    ]