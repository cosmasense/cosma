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

        async def applications_keyword_search(self, query, limit=20, allow_operators=False, path_pattern=None):
            # Reuse the real implementation by binding it here — we
            # want this exercised against the real FTS table.
            from cosma_backend.db.database import Database
            # Call the unbound method with a fake `self` that exposes
            # the acquire() context the real method uses.
            real = Database.applications_keyword_search
            return await real(self, query, limit=limit, allow_operators=allow_operators, path_pattern=path_pattern)

        async def search_similar_applications(self, query_embedding, limit=10, threshold=None, path_pattern=None):
            # Same trick — bind the real method against this shim's
            # acquire() context. Lets the unified-RRF code path that
            # fires an apps semantic query in `search_with_apps` find
            # a callable (it skips when self.embedder is None, but
            # tests still pass a mock embedder).
            from cosma_backend.db.database import Database
            real = Database.search_similar_applications
            return await real(self, query_embedding, limit=limit, threshold=threshold, path_pattern=path_pattern)

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
    async def test_apps_compete_for_limit_slots(self):
        # Apps now share the same hard cap as files instead of having
        # an independent ``apps_limit``. With 20 matching apps and
        # ``limit=5``, exactly 5 should come back — the rest fall off
        # the top-N. This is the regression test for "apps always show
        # ~8 even when files dominate the ranking."
        db, pool, conn = await _make_db()
        try:
            for i in range(20):
                await conn.execute(
                    "INSERT INTO applications (app_path, display_name, category) "
                    "VALUES (?, ?, 'public.app-category.utilities')",
                    (f"/Applications/Util{i}.app", f"utility tool {i}"),
                )

            searcher = HybridSearcher(db, embedder=AsyncMock())
            bundle = await searcher.search_with_apps("utility", limit=5)

            assert len(bundle.apps) == 5
            assert len(bundle.files) == 0
        finally:
            await db.close()


@pytest.mark.unit
class TestSearchScope:
    """``scope`` lets the frontend run only one half of the unified
    search — used by the ``@Applications`` token to suppress files and
    by a future ``@Files`` token to suppress apps."""

    @pytest.mark.asyncio
    async def test_scope_applications_skips_file_branches(self):
        db, pool, conn = await _make_db()
        try:
            await conn.execute(
                "INSERT INTO applications (app_path, display_name) "
                "VALUES ('/Applications/Safari.app', 'Safari')"
            )
            searcher = HybridSearcher(db, embedder=AsyncMock())
            bundle = await searcher.search_with_apps("safari", scope="applications")
            assert len(bundle.apps) == 1
            assert bundle.files == []
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_scope_files_returns_no_apps(self):
        db, pool, conn = await _make_db()
        try:
            await conn.execute(
                "INSERT INTO applications (app_path, display_name) "
                "VALUES ('/Applications/Safari.app', 'Safari')"
            )
            searcher = HybridSearcher(db, embedder=AsyncMock())
            bundle = await searcher.search_with_apps("safari", scope="files")
            # No files in the DB; apps suppressed by scope.
            assert bundle.apps == []
            assert bundle.files == []
        finally:
            await db.close()


@pytest.mark.unit
class TestPathPatternGlob:
    """``path_pattern`` is the backend half of the frontend's
    ``@*.pdf``-style filter. It's pushed into the SQL LIKE clause so
    only matching paths feed the RRF."""

    @pytest.mark.asyncio
    async def test_app_glob_matches_display_name(self):
        # Both apps share a description token so they both pass FTS;
        # the glob is what discriminates. Photoshop survives the
        # ``*Photo*`` filter, Safari does not.
        db, pool, conn = await _make_db()
        try:
            await conn.execute(
                "INSERT INTO applications (app_path, display_name, description) "
                "VALUES ('/Applications/Photoshop.app', 'Photoshop', 'editor app')"
            )
            await conn.execute(
                "INSERT INTO applications (app_path, display_name, description) "
                "VALUES ('/Applications/Safari.app', 'Safari', 'browser app')"
            )
            searcher = HybridSearcher(db, embedder=AsyncMock())
            bundle = await searcher.search_with_apps(
                "app", scope="applications", path_pattern="*Photo*",
            )
            names = [a.application.display_name for a in bundle.apps]
            assert "Photoshop" in names
            assert "Safari" not in names
        finally:
            await db.close()


@pytest.mark.unit
class TestRelativeThreshold:
    """``alpha_floor`` drops the weak tail. A single dominant match
    must not pull in 7 noise matches just to pad the result list."""

    @pytest.mark.asyncio
    async def test_alpha_floor_drops_weak_tail(self):
        # Two apps, only one matches the query meaningfully (because
        # FTS only ranks rows that match MATCH ?). The non-matching
        # app has no FTS hit so it's not in the ranked list at all —
        # confirms the post-RRF threshold doesn't accidentally pull
        # in unranked rows.
        db, pool, conn = await _make_db()
        try:
            await conn.execute(
                "INSERT INTO applications (app_path, display_name) "
                "VALUES ('/Applications/Logic Pro.app', 'Logic Pro')"
            )
            await conn.execute(
                "INSERT INTO applications (app_path, display_name) "
                "VALUES ('/Applications/Safari.app', 'Safari')"
            )
            searcher = HybridSearcher(db, embedder=AsyncMock())
            bundle = await searcher.search_with_apps("logic")
            assert len(bundle.apps) == 1
            assert bundle.apps[0].application.display_name == "Logic Pro"
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
