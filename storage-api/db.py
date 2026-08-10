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
    outside_ids TEXT NOT NULL DEFAULT '[]',
    -- Wall-clock insert time (UTC), NOT video-relative like `timestamp` above -
    -- this is "when was this job's data saved", used as the job's displayed date.
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
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
#
# `backfill`, when set, is a second statement run once right after the ALTER.
# This is required, not optional, for any default that isn't a plain
# constant (e.g. CURRENT_TIMESTAMP): SQLite's ALTER TABLE ADD COLUMN accepts
# a non-constant default only on a table with zero rows (nothing to
# backfill) - on any table that already has data, e.g. this project's real
# storage.db, it raises "Cannot add a column with non-constant default".
# A plain constant default sidesteps that restriction; the follow-up UPDATE
# (an ordinary statement, not a column default, so the restriction doesn't
# apply) does the actual backfill.
MIGRATIONS = (
    ("snapshots", "inside_ids", "TEXT NOT NULL DEFAULT '[]'", None),
    ("snapshots", "outside_ids", "TEXT NOT NULL DEFAULT '[]'", None),
    # Pre-existing rows get the migration's own run time, not their real
    # insert time (SQLite has no way to recover that) - an honest
    # approximation, not a fabricated backfill.
    (
        "snapshots", "created_at", "TEXT NOT NULL DEFAULT ''",
        "UPDATE snapshots SET created_at = CURRENT_TIMESTAMP WHERE created_at = ''",
    ),
)


async def _apply_migrations(db: aiosqlite.Connection) -> None:
    for table, column, decl, backfill in MIGRATIONS:
        cursor = await db.execute(f"PRAGMA table_info({table})")
        existing = {row[1] for row in await cursor.fetchall()}
        if column not in existing:
            await db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
            if backfill:
                await db.execute(backfill)


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