from pydantic import BaseModel


class SnapshotIn(BaseModel):
    job_id: str
    timestamp: float  # video-relative seconds, not wall-clock
    queue_count: int
    inside_ids: list[int] = []
    outside_ids: list[int] = []


class SnapshotOut(SnapshotIn):
    id: int


class TrackIn(BaseModel):
    job_id: str
    track_id: int
    dwell_seconds: float  # how long this track was inside the ROI


class TrackOut(TrackIn):
    id: int


class JobSummary(BaseModel):
    """Aggregates for one processing run, so the dashboard's job list does not
    have to pull every row just to show headline numbers."""
    job_id: str
    # Plain 1, 2, 3... in the order jobs were first seen - what clients
    # actually see; job_id (a UUID) stays around for API calls and debugging.
    seq: int
    snapshot_count: int
    track_count: int
    peak_queue: int
    avg_queue: float
    duration_seconds: float
    avg_dwell: float
    max_dwell: float
    # Wall-clock insert time (UTC, "YYYY-MM-DD HH:MM:SS"), not video-relative.
    created_at: str