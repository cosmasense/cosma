"""Tests for search fusion across files + applications.

Step 6 wiring lands before Step 4 (the apps discoverer), so the
critical contract here is twofold:

  * When the apps table is empty, the apps query returns [] cleanly
    and search() / search_with_apps() still work — no exceptions, no
    slowdown beyond a sub-ms FTS round-trip.
  * When apps ARE present (we insert directly via SQL), they show
    up in `search_with_apps().apps` separately from file results
    and never get mixed into the files list.
"""

from pathlib import Path
from unittest.mock import AsyncMock

import asqlite
import pytest
import sqlite_vec

from cosma_backend.searcher.searcher import HybridSearcher, SearchBundle


SCHEMA_PATH = Path(__file__).resolve().parents[2] / "src" / "cosma_backend" / "schema.sql"


def _init_conn(conn):
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


async def _make_db():
    """Wrap an in-memory schema-initialized DB in a thin shim that
    speaks the Database API the searcher consumes. We only need a
    handful of methods so we duck-type instead of pulling in the
    real Database class (which has heavy init)."""
    pool = await asqlite.create_pool(":memory:", init=_init_conn)
    conn = await pool.acquire()
    schema = SCHEMA_PATH.read_text()
    await conn.executescript(schema)

    class _DB:
        def __init__(self):
            self._conn = conn

        async def search_similar_files(self, *a, **kw):
            return []

        async def keyword_search(self, *a, **kw):
            return []

        async def applications_keyword_search(self, query, limit=20, allow_operators=False):
            # Reuse the real implementation by binding it here — we
            # want this exercised against the real FTS table.
            from cosma_backend.db.database import Database
            # Call the unbound method with a fake `self` that exposes
            # the acquire() context the real method uses.
            real = Database.applications_keyword_search
            return await real(self, query, limit=limit, allow_operators=allow_operators)

        def acquire(self):
            # Re-use the single fixture connection as a no-op context.
            db_self = self

            class _Ctx:
                async def __aenter__(self_inner):
                    return db_self._conn

                async def __aexit__(self_inner, *args):
                    return None

            return _Ctx()

        async def close(self):
            await pool.release(conn)
            await pool.close()

    return _DB(), pool, conn


@pytest.mark.unit
class TestEmptyAppsTable:
    """Step 6 has to be safe to ship before Step 4 puts any rows into
    the apps table. Pin that the apps half is a no-op zero-result
    branch when the table is empty."""

    @pytest.mark.asyncio
    async def test_search_with_apps_returns_empty_apps_list_on_empty_table(self):
        db, pool, conn = await _make_db()
        try:
            searcher = HybridSearcher(db, embedder=AsyncMock())
            # Embedder is mocked; semantic returns []; keyword returns
            # []; apps returns []. The whole call must succeed and
            # return an empty bundle.
            bundle = await searcher.search_with_apps("safari")
            assert isinstance(bundle, SearchBundle)
            assert bundle.files == []
            assert bundle.apps == []
        finally:
            await db.close()


@pytest.mark.unit
class TestAppsAppearInBundle:
    @pytest.mark.asyncio
    async def test_app_match_surfaces_in_apps_list_not_files(self):
        db, pool, conn = await _make_db()
        try:
            await conn.execute(
                """INSERT INTO applications (app_path, bundle_id, display_name,
                                              category, description)
                   VALUES ('/Applications/Logic Pro.app', 'com.apple.logic10',
                           'Logic Pro', 'public.app-category.music',
                           'Professional music production')"""
            )

            searcher = HybridSearcher(db, embedder=AsyncMock())
            bundle = await searcher.search_with_apps("music production")

            assert bundle.files == []  # No files in this DB
            assert len(bundle.apps) == 1
            hit = bundle.apps[0]
            assert hit.application.display_name == "Logic Pro"
            assert hit.application.category == "public.app-category.music"
            assert hit.relevance_score > 0
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_apps_search_via_bundle_id(self):
        # Apps without a description should still be findable by
        # their bundle id — useful for "what's that helper from
        # com.docker.docker doing on my system?"
        db, pool, conn = await _make_db()
        try:
            await conn.execute(
                """INSERT INTO applications (app_path, bundle_id, display_name)
                   VALUES ('/Applications/Docker.app', 'com.docker.docker', 'Docker')"""
            )

            searcher = HybridSearcher(db, embedder=AsyncMock())
            bundle = await searcher.search_with_apps("docker")

            assert len(bundle.apps) == 1
            assert bundle.apps[0].application.display_name == "Docker"
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_apps_limit_independent_from_files_limit(self):
        db, pool, conn = await _make_db()
        try:
            # 20 apps whose display_name shares a token the query
            # actually matches. FTS5 unicode61 doesn't stem
            # "utilities" → "utility0", so we put the search token
            # directly in the display_name.
            for i in range(20):
                await conn.execute(
                    "INSERT INTO applications (app_path, display_name, category) "
                    "VALUES (?, ?, 'public.app-category.utilities')",
                    (f"/Applications/Util{i}.app", f"utility tool {i}"),
                )

            searcher = HybridSearcher(db, embedder=AsyncMock())
            bundle = await searcher.search_with_apps(
                "utility", limit=5, apps_limit=3,
            )

            assert len(bundle.apps) == 3
            # files limit unaffected.
            assert len(bundle.files) == 0
        finally:
            await db.close()


@pytest.mark.unit
class TestSearchBundleStruct:
    """Backward-compat: ``search()`` still returns ``list[SearchResult]``
    so any caller that hasn't switched to the bundle API doesn't
    break. ``search_with_apps()`` is the new entrypoint."""

    @pytest.mark.asyncio
    async def test_legacy_search_returns_plain_list(self):
        db, pool, conn = await _make_db()
        try:
            searcher = HybridSearcher(db, embedder=AsyncMock())
            results = await searcher.search("anything")
            assert isinstance(results, list)
            # Doesn't accidentally start carrying the apps.
            assert all(not hasattr(r, "application") for r in results)
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_bundle_total_ms_populated(self):
        db, pool, conn = await _make_db()
        try:
            searcher = HybridSearcher(db, embedder=AsyncMock())
            bundle = await searcher.search_with_apps("x")
            assert bundle.total_ms >= 0.0  # not strict-positive (in-memory is fast)
        finally:
            await db.close()
