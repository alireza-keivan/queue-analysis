import aiosqlite

DB_PATH = "data/storage.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    queue_count INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_snapshots_job_id ON snapshots (job_id);

CREATE TABLE IF NOT EXISTS tracks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    track_id INTEGER NOT NULL,
    dwell_seconds REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tracks_job_id ON tracks (job_id);
"""


async def connect() -> aiosqlite.Connection:
    db = await aiosqlite.connect(DB_PATH)
    # One writer at a time either way (SQLite), but WAL lets reads happen
    # without waiting on an in-progress write, and busy_timeout makes a
    # second writer wait instead of raising "database is locked".
    await db.execute("PRAGMA journal_mode=WAL;")
    await db.execute("PRAGMA busy_timeout=5000;")
    await db.executescript(SCHEMA)
    await db.commit()
    return db