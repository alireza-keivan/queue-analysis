from datetime import datetime

from pydantic import BaseModel


class SnapshotIn(BaseModel):
    job_id: str
    timestamp: datetime
    queue_count: int


class SnapshotOut(SnapshotIn):
    id: int


class TrackIn(BaseModel):
    job_id: str
    track_id: int
    entry_time: datetime
    exit_time: datetime


class TrackOut(TrackIn):
    id: int