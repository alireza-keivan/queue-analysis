from pydantic import BaseModel


class SnapshotIn(BaseModel):
    job_id: str
    timestamp: float  # video-relative seconds, not wall-clock
    queue_count: int


class SnapshotOut(SnapshotIn):
    id: int


class TrackIn(BaseModel):
    job_id: str
    track_id: int
    dwell_seconds: float  # how long this track was inside the ROI


class TrackOut(TrackIn):
    id: int