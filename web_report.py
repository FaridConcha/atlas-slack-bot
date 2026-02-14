"""
ATLAS Web Report — Storage layer.

Stores full analysis payloads in SQLite and returns report URLs.
No LLM calls. Deterministic storage only.

Usage (from bot.py):
    from web_report import generate_and_store_report, get_report
"""

import json
import os
import re
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Force flush on every print so Render logs appear in real time
import functools
print = functools.partial(print, flush=True)  # noqa: A001

ATLAS_WEB_BASE_URL = os.environ.get("ATLAS_WEB_BASE_URL", "http://localhost:8000")

# Use /opt/render/project/.data/ on Render (persists across sleep/wake, lost on deploy)
# Fallback to local dir for development
_RENDER_DATA_DIR = Path("/opt/render/project/.data")
if _RENDER_DATA_DIR.exists():
    _RENDER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DB_PATH = str(_RENDER_DATA_DIR / "reports.db")
else:
    DB_PATH = str(Path(__file__).parent / "reports.db")

print(f"[WEB_REPORT] Database path: {DB_PATH}")


def _get_db() -> sqlite3.Connection:
    """Open (and auto-create) the reports database."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS reports (
            report_id   TEXT PRIMARY KEY,
            created_at  TEXT NOT NULL,
            symbol      TEXT NOT NULL,
            payload_json TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def _sanitize_id(raw: str) -> str:
    """Replace non-alphanumeric chars (dots, etc.) with dashes for URL safety."""
    return re.sub(r'[^a-zA-Z0-9_-]', '-', raw)


def _make_serializable(obj: Any) -> Any:
    """Recursively convert numpy/non-JSON types so json.dumps succeeds."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    # numpy scalars / anything with .item()
    if hasattr(obj, 'item'):
        return obj.item()
    # Fallback: force to string
    return str(obj)


_REPORT_ID_RE = re.compile(r'^[A-Za-z0-9_-]{1,120}$')


def _cleanup_old_reports():
    """Delete reports older than 30 days."""
    conn = _get_db()
    try:
        conn.execute(
            "DELETE FROM reports WHERE created_at < datetime('now', '-30 days')"
        )
        conn.commit()
    except Exception:
        pass
    finally:
        conn.close()


def generate_and_store_report(
    symbol: str,
    company_name: str | None,
    thread_ts: str,
    summary: dict,
    v8_extended: dict,
    provenance: dict,
) -> dict:
    """
    Persist a full analysis payload and return its URL.

    Returns:
        {"report_id": str, "report_url": str}
    """
    short_uuid = uuid.uuid4().hex[:8]
    safe_ts = _sanitize_id(thread_ts)
    report_id = f"{symbol}-{safe_ts}-{short_uuid}"
    created_at = datetime.now(timezone.utc).isoformat()

    payload = {
        "report_id": report_id,
        "created_at": created_at,
        "engine_version": "ATLAS V10",
        "symbol": symbol,
        "company_name": company_name,
        "thread_ts": thread_ts,
        "summary": _make_serializable(summary),
        "v8_extended": _make_serializable(v8_extended),
        "provenance": _make_serializable(provenance),
    }

    payload_json = json.dumps(payload, default=str)

    conn = _get_db()
    try:
        conn.execute(
            "INSERT INTO reports (report_id, created_at, symbol, payload_json) VALUES (?, ?, ?, ?)",
            (report_id, created_at, symbol, payload_json),
        )
        conn.commit()
    finally:
        conn.close()

    report_url = f"{ATLAS_WEB_BASE_URL}/r/{report_id}"
    print(f"[WEB_REPORT] Stored {report_id} ({len(payload_json)} bytes) → {report_url}")

    _cleanup_old_reports()

    return {"report_id": report_id, "report_url": report_url}


def get_report(report_id: str) -> dict | None:
    """Fetch a stored report by ID. Returns parsed payload dict or None."""
    if not _REPORT_ID_RE.match(report_id):
        return None
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT payload_json FROM reports WHERE report_id = ?", (report_id,)
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None
    return json.loads(row[0])
