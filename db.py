"""
db.py — SQLite Database Layer for Object Observations

Schema (Task 5 — extended):
  id              INTEGER PRIMARY KEY
  timestamp       TEXT
  frame_number    INTEGER
  object_class    TEXT
  tracking_id     INTEGER       ← raw ByteTrack ID
  confidence      REAL
  bounding_box    TEXT           ← JSON {"x1","y1","x2","y2"}
  zone            TEXT
  global_track_id INTEGER       ← stable ReID-stitched ID  (NEW)
  crop_path       TEXT           ← path to cropped object image  (NEW)
  frame_path      TEXT           ← path to saved keyframe  (NEW)
"""

import datetime
import sqlite3
import os

DB_PATH = "observations.db"


def _get_conn():
    return sqlite3.connect(DB_PATH)


# ── Schema ───────────────────────────────────────────────────────────────────
def init_db():
    """Create the observations table if it doesn't exist (idempotent)."""
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            frame_number INTEGER,
            object_class TEXT,
            tracking_id INTEGER,
            confidence REAL,
            bounding_box TEXT,
            zone TEXT,
            global_track_id INTEGER,
            crop_path TEXT,
            frame_path TEXT
        )
    """)
    conn.commit()
    conn.close()


def window_cutoff(seconds: float) -> str:
    """Timestamp string matching how observations are stored."""
    cutoff = datetime.datetime.now() - datetime.timedelta(seconds=seconds)
    return str(cutoff)


def prune_older_than(seconds: float) -> int:
    """Delete sightings older than ``seconds``. Returns how many rows were removed."""
    init_db()
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM observations WHERE timestamp < ?",
        (window_cutoff(seconds),),
    )
    removed = cursor.rowcount
    conn.commit()
    conn.close()
    return removed


def max_frame_number() -> int:
    """Highest frame number already stored, or 0 when the log is empty."""
    init_db()
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT COALESCE(MAX(frame_number), 0) FROM observations")
    value = cursor.fetchone()[0]
    conn.close()
    return int(value or 0)


def reset_db():
    """Drop and recreate the observations table (used before a new file run)."""
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS observations")
    conn.commit()
    conn.close()
    init_db()


# ── Insert ───────────────────────────────────────────────────────────────────
def insert_observation(timestamp, frame_number, object_class, tracking_id,
                       confidence, bounding_box, zone,
                       global_track_id=None, crop_path=None, frame_path=None):
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO observations
            (timestamp, frame_number, object_class, tracking_id, confidence,
             bounding_box, zone, global_track_id, crop_path, frame_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (timestamp, frame_number, object_class, tracking_id, confidence,
          bounding_box, zone, global_track_id, crop_path, frame_path))
    conn.commit()
    conn.close()


# ── Basic Queries ────────────────────────────────────────────────────────────
_COLUMNS = [
    "id", "timestamp", "frame_number", "object_class", "tracking_id",
    "confidence", "bounding_box", "zone", "global_track_id", "crop_path",
    "frame_path",
]


def _rows_to_dicts(rows):
    return [dict(zip(_COLUMNS, row)) for row in rows]


def get_all_observations():
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM observations ORDER BY frame_number ASC")
    rows = cursor.fetchall()
    conn.close()
    return _rows_to_dicts(rows)


def get_observations_for_frame(frame_number):
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM observations WHERE frame_number = ?",
                   (frame_number,))
    rows = cursor.fetchall()
    conn.close()
    return _rows_to_dicts(rows)


# ── Task 5 — Advanced Queries ───────────────────────────────────────────────
def get_latest_observation(object_class: str):
    """Return the most recent observation matching *object_class* (case-insensitive)."""
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM observations
        WHERE LOWER(object_class) = LOWER(?)
        ORDER BY id DESC
        LIMIT 1
    """, (object_class,))
    rows = cursor.fetchall()
    conn.close()
    return _rows_to_dicts(rows)[0] if rows else None


def get_object_history(object_class: str = None, global_track_id: int = None):
    """
    Return chronological zone-transition history.

    Supply *object_class* for a class-level query or *global_track_id* for a
    specific tracked instance.  Returns a list of dicts with zone-change events.
    """
    conn = _get_conn()
    cursor = conn.cursor()
    if global_track_id is not None:
        cursor.execute("""
            SELECT * FROM observations
            WHERE global_track_id = ?
            ORDER BY id ASC
        """, (global_track_id,))
    elif object_class is not None:
        cursor.execute("""
            SELECT * FROM observations
            WHERE LOWER(object_class) = LOWER(?)
            ORDER BY id ASC
        """, (object_class,))
    else:
        conn.close()
        return []
    rows = cursor.fetchall()
    conn.close()

    all_obs = _rows_to_dicts(rows)
    if not all_obs:
        return []

    # Collapse consecutive same-zone entries into transitions
    transitions = []
    prev_zone = None
    for obs in all_obs:
        if obs["zone"] != prev_zone:
            transitions.append(obs)
            prev_zone = obs["zone"]
    return transitions


def get_distinct_tracked_classes():
    """Return sorted list of distinct object class names in the database."""
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT object_class FROM observations ORDER BY object_class")
    rows = cursor.fetchall()
    conn.close()
    return [r[0] for r in rows]


def get_observation_summary():
    """Return per-class summary: class, count, last zone, last frame."""
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT object_class,
               COUNT(*) as cnt,
               MAX(frame_number) as last_frame
        FROM observations
        GROUP BY object_class
        ORDER BY cnt DESC
    """)
    rows = cursor.fetchall()
    conn.close()

    summaries = []
    for cls, cnt, last_frame in rows:
        latest = get_latest_observation(cls)
        summaries.append({
            "object_class": cls,
            "total_detections": cnt,
            "last_seen_frame": latest["frame_number"] if latest else last_frame,
            "last_zone": latest["zone"] if latest else "Unknown",
            "last_confidence": latest["confidence"] if latest else 0,
        })
    return summaries


# ── Entrypoint ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    print("Database initialized with extended schema.")
