"""Tests for app semantic embedding + fused search.

Three contracts pinned here:

  * The indexer embeds every newly-discovered app and stores the
    text hash so a re-scan with no Info.plist changes triggers
    zero embedder calls.
  * Search-side fusion runs keyword + semantic in parallel and
    RRF-merges the results, so an "intent" query (no literal
    name match) can surface an app whose description happens to
    contain a semantically close phrase.
  * The keyword-only fallback still works when the embedder is
    absent (tests + minimal-deploy scenarios).
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import asqlite
import numpy as np
import pytest
import sqlite_vec

from cosma_backend.applications import (
    ApplicationsIndexer,
    ApplicationsRepository,
)
from cosma_backend.applications.applications_indexer import (
    _embedding_text,
    _hash_text,
)
from cosma_backend.models import Application
from cosma_backend.searcher.searcher import HybridSearcher


SCHEMA_PATH = Path(__file__).resolve().parents[2] / "src" / "cosma_backend" / "schema.sql"


def _init_conn(conn):
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


async def _make_db():
    pool = await asqlite.create_pool(":memory:", init=_init_conn)
    conn = await pool.acquire()
    await conn.executescript(SCHEMA_PATH.read_text())

    class _DB:
        # Borrow the real Database's serialize/normalize helpers so
        # the repository's vec0 INSERT goes through the same code
        # path production does.
        def __init__(self):
            from cosma_backend.db.database import Database
            self._normalize_embedding_dimensions = (
                Database._normalize_embedding_dimensions.__get__(self)
            )
            self._serialize_vector = Database._serialize_vector.__get__(self)

        def acquire(_self):
            class _Ctx:
                async def __aenter__(self):
                    return conn

                async def __aexit__(self, *a):
                    return None

            return _Ctx()

        async def applications_keyword_search(_self, query, limit=20, allow_operators=False):
            from cosma_backend.db.database import Database
            real = Database.applications_keyword_search
            return await real(_self, query, limit=limit, allow_operators=allow_operators)

        async def search_similar_applications(_self, query_embedding, limit=10, threshold=None):
            from cosma_backend.db.database import Database
            real = Database.search_similar_applications
            return await real(_self, query_embedding, limit=limit, threshold=threshold)

        async def search_similar_files(self, *a, **kw):
            return []

        async def keyword_search(self, *a, **kw):
            return []

    return _DB(), pool, conn


class _FakeEmbedder:
    """Deterministic embedder: hashes text → 1536-d unit vector.

    Two strings with the same hash get identical vectors; different
    strings get different ones. Good enough to exercise the indexer
    + semantic-search wiring without pulling SentenceTransformer
    into the unit test path.
    """

    model_name = "fake-deterministic"

    def __init__(self):
        self.calls = 0
        self._fixed_overrides: dict[str, np.ndarray] = {}

    def set_fixed(self, text: str, vec: np.ndarray) -> None:
        self._fixed_overrides[text] = vec.astype(np.float32)

    async def embed_text_async(self, text, priority=False):
        self.calls += 1
        if text in self._fixed_overrides:
            return self._fixed_overrides[text]
        # Deterministic vector from text hash → 1536-d. We seed RNG
        # off the hash so identical text always produces the same
        # vector — the contract we care about.
        seed = int.from_bytes(_hash_text(text).encode()[:8], "big")
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(1536).astype(np.float32)
        v /= max(np.linalg.norm(v), 1e-8)
        return v


@pytest.fixture
def fake_discoverer():
    """Discoverer stub that yields a fixed list of apps. Beats
    standing up a synthetic /Applications tree for every test."""
    class _Disc:
        def __init__(self, apps: list[Application]):
            self._apps = apps

        def discover(self) -> list[Application]:
            return list(self._apps)

    return _Disc


@pytest.mark.unit
class TestIndexerEmbedsApps:
    @pytest.mark.asyncio
    async def test_first_scan_embeds_every_app(self, fake_discoverer):
        db, pool, conn = await _make_db()
        try:
            apps = [
                Application(app_path="/Applications/A.app", display_name="A"),
                Application(app_path="/Applications/B.app", display_name="B"),
            ]
            embedder = _FakeEmbedder()
            indexer = ApplicationsIndexer(
                ApplicationsRepository(db), fake_discoverer(apps), embedder=embedder,
            )
            result = await indexer.index_now()
            assert result.discovered == 2
            assert result.embedded == 2
            assert embedder.calls == 2

            # Embeddings landed in the vec0 table.
            cur = await conn.execute(
                "SELECT count(*) FROM application_embeddings"
            )
            assert (await cur.fetchall())[0][0] == 2

            # Text-hash columns populated so the next scan can skip.
            cur = await conn.execute(
                "SELECT embedding_text_hash FROM applications ORDER BY display_name"
            )
            hashes = [r[0] for r in await cur.fetchall()]
            assert all(h is not None for h in hashes)
        finally:
            await pool.release(conn); await pool.close()

    @pytest.mark.asyncio
    async def test_rescan_skips_unchanged_apps(self, fake_discoverer):
        db, pool, conn = await _make_db()
        try:
            apps = [Application(app_path="/Applications/A.app", display_name="A")]
            embedder = _FakeEmbedder()
            indexer = ApplicationsIndexer(
                ApplicationsRepository(db), fake_discoverer(apps), embedder=embedder,
            )
            await indexer.index_now()
            assert embedder.calls == 1
            result2 = await indexer.index_now()  # nothing changed
            assert result2.embedded == 0
            assert embedder.calls == 1  # still 1, the second scan was a no-op
        finally:
            await pool.release(conn); await pool.close()

    @pytest.mark.asyncio
    async def test_changed_description_triggers_reembed(self, fake_discoverer):
        # Edit the app's description between scans → hash changes →
        # we should embed again.
        db, pool, conn = await _make_db()
        try:
            apps_v1 = [Application(
                app_path="/Applications/A.app", display_name="A",
                description="original blurb",
            )]
            apps_v2 = [Application(
                app_path="/Applications/A.app", display_name="A",
                description="totally different copy",
            )]
            embedder = _FakeEmbedder()
            repo = ApplicationsRepository(db)
            disc1 = fake_discoverer(apps_v1)
            indexer = ApplicationsIndexer(repo, disc1, embedder=embedder)
            await indexer.index_now()
            # Swap in the v2 discoverer for the next scan.
            indexer._discoverer = fake_discoverer(apps_v2)
            result = await indexer.index_now()
            assert result.embedded == 1
            assert embedder.calls == 2
        finally:
            await pool.release(conn); await pool.close()

    @pytest.mark.asyncio
    async def test_no_embedder_keeps_keyword_only_behavior(self, fake_discoverer):
        # Backward-compat: ApplicationsIndexer constructed without an
        # embedder still upserts apps cleanly; just no vectors get
        # written and `embedded` stays 0.
        db, pool, conn = await _make_db()
        try:
            apps = [Application(app_path="/Applications/A.app", display_name="A")]
            indexer = ApplicationsIndexer(
                ApplicationsRepository(db), fake_discoverer(apps), embedder=None,
            )
            result = await indexer.index_now()
            assert result.upserted == 1
            assert result.embedded == 0
            cur = await conn.execute("SELECT count(*) FROM application_embeddings")
            assert (await cur.fetchall())[0][0] == 0
        finally:
            await pool.release(conn); await pool.close()

    @pytest.mark.asyncio
    async def test_embedder_failure_isolated_to_one_app(self, fake_discoverer):
        # If the embedder raises on one app's text, the scan should
        # still embed the rest.
        db, pool, conn = await _make_db()
        try:
            apps = [
                Application(app_path="/Applications/Good.app", display_name="Good"),
                Application(app_path="/Applications/Bad.app", display_name="Bad"),
            ]

            class _FlakyEmbedder(_FakeEmbedder):
                async def embed_text_async(_self, text, priority=False):
                    if "Bad" in text:
                        raise RuntimeError("simulated embed failure")
                    return await _FakeEmbedder.embed_text_async(_self, text, priority=priority)

            embedder = _FlakyEmbedder()
            indexer = ApplicationsIndexer(
                ApplicationsRepository(db), fake_discoverer(apps), embedder=embedder,
            )
            result = await indexer.index_now()
            assert result.embedded == 1  # only Good
            cur = await conn.execute(
                "SELECT count(*) FROM application_embeddings"
            )
            assert (await cur.fetchall())[0][0] == 1
        finally:
            await pool.release(conn); await pool.close()


@pytest.mark.unit
class TestEmbeddingTextBody:
    """The text body fed to the embedder determines what the
    semantic-search ranking knows about. Pin its shape so a future
    edit can't accidentally drop the description (which is the
    field carrying the "what is this app for" signal)."""

    def test_includes_all_signal_fields(self):
        app = Application(
            app_path="/Applications/X.app",
            display_name="Pixelmator",
            description="Powerful image editor",
            use_cases="Edit and retouch photos",
            category="public.app-category.graphics-design",
            bundle_id="com.pixelmatorteam.pixelmator",
        )
        text = _embedding_text(app)
        assert "Pixelmator" in text
        assert "image editor" in text
        assert "Edit and retouch photos" in text
        # Category trailing component, with dashes humanized.
        assert "graphics design" in text
        assert "com.pixelmatorteam.pixelmator" in text

    def test_missing_fields_are_skipped(self):
        text = _embedding_text(Application(display_name="OnlyName"))
        assert text == "OnlyName"


@pytest.mark.unit
class TestFusedSearch:
    @pytest.mark.asyncio
    async def test_semantic_intent_query_finds_app_without_literal_match(self, fake_discoverer):
        """The point of the whole feature: "music production" should
        find Logic Pro even if the literal phrase doesn't appear in
        Logic Pro's display_name. We rig the embedder so the query
        vector lands closest to Logic Pro's stored embedding."""
        db, pool, conn = await _make_db()
        try:
            logic = Application(
                app_path="/Applications/Logic Pro.app",
                display_name="Logic Pro",
                description="Professional audio production",
                category="public.app-category.music",
            )
            misc = Application(
                app_path="/Applications/Calculator.app",
                display_name="Calculator",
            )

            embedder = _FakeEmbedder()
            # Pin specific vectors so the query is unambiguously
            # closer to Logic Pro than to Calculator.
            logic_vec = np.zeros(1536, dtype=np.float32); logic_vec[0] = 1
            calc_vec  = np.zeros(1536, dtype=np.float32); calc_vec[1] = 1
            query_vec = np.zeros(1536, dtype=np.float32); query_vec[0] = 0.99; query_vec[1] = 0.01
            embedder.set_fixed(_embedding_text(logic), logic_vec)
            embedder.set_fixed(_embedding_text(misc), calc_vec)
            embedder.set_fixed("music production", query_vec)

            indexer = ApplicationsIndexer(
                ApplicationsRepository(db),
                fake_discoverer([logic, misc]),
                embedder=embedder,
            )
            await indexer.index_now()

            searcher = HybridSearcher(db, embedder=embedder)
            bundle = await searcher.search_with_apps("music production")
            assert len(bundle.apps) >= 1
            # Logic Pro must rank first.
            assert bundle.apps[0].application.display_name == "Logic Pro"
        finally:
            await pool.release(conn); await pool.close()

    @pytest.mark.asyncio
    async def test_keyword_match_still_works(self, fake_discoverer):
        # The literal-name path stays intact: searching "Docker"
        # finds Docker via FTS even if the semantic side returns
        # nothing useful.
        db, pool, conn = await _make_db()
        try:
            docker = Application(
                app_path="/Applications/Docker.app",
                display_name="Docker",
                bundle_id="com.docker.docker",
            )
            embedder = _FakeEmbedder()
            indexer = ApplicationsIndexer(
                ApplicationsRepository(db),
                fake_discoverer([docker]),
                embedder=embedder,
            )
            await indexer.index_now()

            searcher = HybridSearcher(db, embedder=embedder)
            bundle = await searcher.search_with_apps("docker")
            assert any(a.application.display_name == "Docker" for a in bundle.apps)
        finally:
            await pool.release(conn); await pool.close()
