"""Dev-only structured log of file-processing failures.

Writes one JSON line per failure to ~/Library/Logs/cosma/failed-items.jsonl
when COSMA_DEV=1 is set. Intended for the "fix → retry-all → observe"
dev loop: after each code change, reset the log, retry all FAILED files
via POST /api/queue/reindex, and tail the file to see what still breaks.

No-op in production (env var not set) so shipped builds never touch disk.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

_DEV_MODE = os.environ.get("COSMA_DEV") == "1"
_LOG_PATH = Path.home() / "Library/Logs/cosma/failed-items.jsonl"
_WRITE_LOCK = Lock()


def is_enabled() -> bool:
    return _DEV_MODE


def log_failed_file(
    *,
    file_path: str,
    extension: str | None,
    error: str,
    phase: str,
) -> None:
    """Append a single failure record. Silently no-ops outside dev mode."""
    if not _DEV_MODE:
        return
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "file_path": file_path,
        "extension": extension,
        "phase": phase,
        "error": error,
    }
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with _WRITE_LOCK:
            with _LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(line)
    except OSError:
        pass  # Never let dev logging break the pipeline.


def reset_log() -> None:
    """Truncate the log. Call before kicking off a retry-all iteration."""
    if not _DEV_MODE:
        return
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _LOG_PATH.write_text("", encoding="utf-8")
    except OSError:
        pass
