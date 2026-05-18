"""Database access layer for the applications table.

Kept separate from `Database` so the apps source has its own
boundary — the rest of the codebase touches Application rows
through this thin shim, not through hand-written SQL. The FTS5
sync is handled by triggers in schema.sql; this layer just
issues INSERT/UPDATE/DELETE against the base table and lets the
triggers do their job.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from cosma_backend.logging import get_logger
from cosma_backend.models import Application

if TYPE_CHECKING:
    from cosma_backend.db.database import Database

logger = get_logger(__name__)


_UPSERT_SQL = """
INSERT INTO applications (
    app_path, bundle_id, display_name, short_version,
    category, description, use_cases, icon_path,
    indexed_at, updated_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(app_path) DO UPDATE SET
    bundle_id     = excluded.bundle_id,
    display_name  = excluded.display_name,
    short_version = excluded.short_version,
    category      = excluded.category,
    description   = excluded.description,
    -- use_cases is filled by the (slow, LLM-driven) enrichment pass
    -- in a future revision. Preserve any existing value across
    -- routine re-scans so we don't have to re-pay the LLM tax just
    -- because Info.plist's mtime advanced.
    use_cases     = COALESCE(applications.use_cases, excluded.use_cases),
    icon_path     = excluded.icon_path,
    updated_at    = excluded.updated_at
"""


class ApplicationsRepository:
    def __init__(self, db: "Database") -> None:
        self._db = db

    async def upsert_many(self, apps: list[Application]) -> int:
        """Insert or update many apps in a single transaction.

        Returns the count handed in (a row that fell through to UPDATE
        still counts — for "scan summary" purposes the user cares
        about how many bundles we processed, not how many were new).
        """
        if not apps:
            return 0
        now = int(time.time())
        rows = [
            (
                a.app_path, a.bundle_id, a.display_name, a.short_version,
                a.category, a.description, a.use_cases, a.icon_path,
                now, now,
            )
            for a in apps
        ]
        async with self._db.acquire() as conn:
            await conn.executemany(_UPSERT_SQL, rows)
            await conn.commit()
        return len(rows)

    async def delete_missing(self, present_app_paths: set[str]) -> int:
        """Remove rows whose bundles are no longer on disk.

        Run after a full scan: pass the set of app_paths the
        discoverer just saw; rows not in that set get deleted. The
        FTS sync trigger cleans up `applications_fts` for us.
        """
        if not present_app_paths:
            # Empty disk would imply the user uninstalled everything;
            # that's almost certainly a discoverer error (path bug,
            # permissions). Don't nuke the table.
            logger.warning("Apps scan returned zero results — "
                           "refusing to delete the entire applications "
                           "table as a safety guard.")
            return 0
        async with self._db.acquire() as conn:
            placeholders = ",".join("?" * len(present_app_paths))
            sql = f"DELETE FROM applications WHERE app_path NOT IN ({placeholders})"
            await conn.execute(sql, tuple(present_app_paths))
            # asqlite's Cursor doesn't expose rowcount; SQLite's
            # changes() function is the portable way to read the
            # delta of the most recent INSERT/UPDATE/DELETE on this
            # connection. We read it inside the same connection
            # acquire so no other write can interleave.
            cur = await conn.execute("SELECT changes()")
            row = (await cur.fetchall())[0]
            count = int(row[0])
            await conn.commit()
            if count:
                logger.info("Pruned uninstalled apps", count=count)
            return count

    async def list_all(self, *, limit: int = 1000, offset: int = 0) -> list[Application]:
        async with self._db.acquire() as conn:
            async with conn.execute(
                "SELECT * FROM applications ORDER BY display_name "
                "COLLATE NOCASE LIMIT ? OFFSET ?",
                (limit, offset),
            ) as cur:
                rows = await cur.fetchall()
        return [Application.from_row(r) for r in rows]

    async def count(self) -> int:
        async with self._db.acquire() as conn:
            async with conn.execute("SELECT count(*) FROM applications") as cur:
                row = (await cur.fetchall())[0]
        return int(row[0])
