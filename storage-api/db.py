import os

import aiosqlite

DB_PATH = "data/storage.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    queue_count INTEGER NOT NULL,
    inside_ids TEXT NOT NULL DEFAULT '[]',   -- JSON array of track_ids; SQLite has no array type
    outside_ids TEXT NOT NULL DEFAULT '[]'
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


# Columns added to a table after it first shipped. CREATE TABLE IF NOT
# EXISTS does nothing at all when the table is already there - it does not
# reconcile columns - so a DB file created before one of these existed keeps
# the old shape forever, and every query naming the new column dies with
# "no such column". Each entry is applied only if the column is missing.
MIGRATIONS = (
    ("snapshots", "inside_ids", "TEXT NOT NULL DEFAULT '[]'"),
    ("snapshots", "outside_ids", "TEXT NOT NULL DEFAULT '[]'"),
)


async def _apply_migrations(db: aiosqlite.Connection) -> None:
    for table, column, decl in MIGRATIONS:
        cursor = await db.execute(f"PRAGMA table_info({table})")
        existing = {row[1] for row in await cursor.fetchall()}
        if column not in existing:
            # A NOT NULL column can only be added when it carries a default,
            # which is what backfills the pre-existing rows.
            await db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


async def connect() -> aiosqlite.Connection:
    # sqlite3 (and aiosqlite, which wraps it) will not create a missing
    # parent directory - it only creates the .db file itself.
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    db = await aiosqlite.connect(DB_PATH)
    # One writer at a time either way (SQLite), but WAL lets reads happen
    # without waiting on an in-progress write, and busy_timeout makes a
    # second writer wait instead of raising "database is locked".
    await db.execute("PRAGMA journal_mode=WAL;")
    await db.execute("PRAGMA busy_timeout=5000;")
    await db.executescript(SCHEMA)
    # Runs after the schema script, so the table is guaranteed to exist -
    # this only ever adds columns to an already-created older table.
    await _apply_migrations(db)
    await db.commit()
    return db