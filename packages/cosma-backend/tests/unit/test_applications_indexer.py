"""Tests for the Applications source: discoverer, repository, indexer.

We build fake .app bundles in tmp_path so the tests don't depend on
what happens to be installed on the dev machine. The bundles are
real plist-shaped trees so the same plistlib code that runs against
/Applications runs against these — the only thing mocked is the
search root.
"""

from pathlib import Path
import plistlib
from unittest.mock import AsyncMock

import asqlite
import pytest
import sqlite_vec

from cosma_backend.applications import (
    ApplicationsDiscoverer,
    ApplicationsIndexer,
    ApplicationsRepository,
)


SCHEMA_PATH = Path(__file__).resolve().parents[2] / "src" / "cosma_backend" / "schema.sql"


def _init_conn(conn):
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


def _make_app(root: Path, name: str, **plist) -> Path:
    """Construct a minimal .app bundle on disk.

    Layout is `<root>/<name>.app/Contents/Info.plist` with the
    caller-supplied plist dict. Returns the bundle path so tests can
    assert against it.
    """
    bundle = root / f"{name}.app"
    contents = bundle / "Contents"
    resources = contents / "Resources"
    resources.mkdir(parents=True)
    with (contents / "Info.plist").open("wb") as f:
        plistlib.dump(plist, f)
    return bundle


@pytest.mark.unit
class TestApplicationsDiscoverer:
    def test_minimal_bundle_yields_application(self, tmp_path: Path):
        _make_app(
            tmp_path, "Safari",
            CFBundleDisplayName="Safari",
            CFBundleIdentifier="com.apple.Safari",
            CFBundleShortVersionString="17.4",
            LSApplicationCategoryType="public.app-category.productivity",
        )

        apps = ApplicationsDiscoverer(search_roots=(str(tmp_path),)).discover()
        assert len(apps) == 1
        a = apps[0]
        assert a.display_name == "Safari"
        assert a.bundle_id == "com.apple.Safari"
        assert a.short_version == "17.4"
        assert a.category == "public.app-category.productivity"
        assert a.app_path.endswith("/Safari.app")

    def test_fallback_to_bundle_name_then_directory(self, tmp_path: Path):
        # CFBundleDisplayName missing → falls back to CFBundleName.
        _make_app(tmp_path, "FromBundleName", CFBundleName="Friendly Name")
        # Both missing → falls back to directory stem ("FromDir").
        _make_app(tmp_path, "FromDir")

        apps = sorted(
            ApplicationsDiscoverer(search_roots=(str(tmp_path),)).discover(),
            key=lambda a: a.app_path,
        )
        names = [a.display_name for a in apps]
        assert "Friendly Name" in names
        assert "FromDir" in names

    def test_skips_bundles_without_info_plist(self, tmp_path: Path):
        # An incomplete bundle (no Info.plist) is silently skipped —
        # broken installers in /Applications shouldn't crash the scan.
        broken = tmp_path / "Broken.app"
        (broken / "Contents").mkdir(parents=True)
        # Plus a good one alongside it.
        _make_app(tmp_path, "Good")

        apps = ApplicationsDiscoverer(search_roots=(str(tmp_path),)).discover()
        assert [a.display_name for a in apps] == ["Good"]

    def test_ignores_non_app_entries(self, tmp_path: Path):
        # Random files/folders at the search root are ignored — only
        # .app bundles count as apps.
        (tmp_path / "not-an-app.txt").write_text("ignore me\n")
        (tmp_path / "random-folder").mkdir()
        _make_app(tmp_path, "Real")

        apps = ApplicationsDiscoverer(search_roots=(str(tmp_path),)).discover()
        assert [a.display_name for a in apps] == ["Real"]

    def test_copyright_description_is_dropped(self, tmp_path: Path):
        # Apps that put "Copyright 2024 Apple Inc." in CFBundleGetInfoString
        # shouldn't poison the FTS description column.
        _make_app(
            tmp_path, "CopyOnly",
            CFBundleGetInfoString="Copyright (c) 2024 Apple Inc. All rights reserved.",
        )
        apps = ApplicationsDiscoverer(search_roots=(str(tmp_path),)).discover()
        assert apps[0].description is None

    def test_real_description_is_kept(self, tmp_path: Path):
        _make_app(
            tmp_path, "RealDesc",
            CFBundleGetInfoString="Professional audio workstation",
        )
        apps = ApplicationsDiscoverer(search_roots=(str(tmp_path),)).discover()
        assert apps[0].description == "Professional audio workstation"

    def test_icon_path_resolves_when_present(self, tmp_path: Path):
        bundle = _make_app(tmp_path, "WithIcon", CFBundleIconFile="AppIcon")
        # CFBundleIconFile is "AppIcon"; the .icns is implicit. We
        # write the file so the resolver finds it.
        (bundle / "Contents" / "Resources" / "AppIcon.icns").write_bytes(b"icns")
        apps = ApplicationsDiscoverer(search_roots=(str(tmp_path),)).discover()
        assert apps[0].icon_path is not None
        assert apps[0].icon_path.endswith("/AppIcon.icns")

    def test_search_root_deduplication(self, tmp_path: Path):
        # Same .app symlinked from two roots only surfaces once.
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        root_a.mkdir()
        root_b.mkdir()
        _make_app(root_a, "Shared")
        (root_b / "Shared.app").symlink_to(root_a / "Shared.app")
        apps = ApplicationsDiscoverer(
            search_roots=(str(root_a), str(root_b)),
        ).discover()
        assert len(apps) == 1


async def _make_repo() -> tuple[ApplicationsRepository, asqlite.Pool, asqlite.Connection]:
    pool = await asqlite.create_pool(":memory:", init=_init_conn)
    conn = await pool.acquire()
    await conn.executescript(SCHEMA_PATH.read_text())

    class _DB:
        def acquire(_self):
            db_holder = _self

            class _Ctx:
                async def __aenter__(self2):
                    return conn

                async def __aexit__(self2, *a):
                    return None

            return _Ctx()

    repo = ApplicationsRepository(_DB())
    return repo, pool, conn


@pytest.mark.unit
class TestApplicationsRepository:
    @pytest.mark.asyncio
    async def test_upsert_inserts_new_rows(self):
        repo, pool, conn = await _make_repo()
        try:
            from cosma_backend.models import Application
            count = await repo.upsert_many([
                Application(app_path="/Applications/A.app", display_name="A"),
                Application(app_path="/Applications/B.app", display_name="B"),
            ])
            assert count == 2
            assert await repo.count() == 2
        finally:
            await pool.release(conn); await pool.close()

    @pytest.mark.asyncio
    async def test_upsert_preserves_use_cases_on_rescan(self):
        # The repository must not clobber an LLM-enriched use_cases
        # value when a routine re-scan with no use_cases comes in.
        # Otherwise every 6 h tick would erase the enrichment data.
        repo, pool, conn = await _make_repo()
        try:
            from cosma_backend.models import Application
            await repo.upsert_many([
                Application(
                    app_path="/Applications/Note.app",
                    display_name="Note",
                    use_cases="A simple notes app",
                ),
            ])
            await repo.upsert_many([
                # No use_cases in the rescan payload (discoverer doesn't
                # produce it).
                Application(
                    app_path="/Applications/Note.app",
                    display_name="Note",
                ),
            ])
            apps = await repo.list_all()
            assert apps[0].use_cases == "A simple notes app"
        finally:
            await pool.release(conn); await pool.close()

    @pytest.mark.asyncio
    async def test_delete_missing_removes_uninstalled(self):
        repo, pool, conn = await _make_repo()
        try:
            from cosma_backend.models import Application
            await repo.upsert_many([
                Application(app_path="/Applications/Keep.app", display_name="Keep"),
                Application(app_path="/Applications/Gone.app", display_name="Gone"),
            ])
            await repo.delete_missing({"/Applications/Keep.app"})
            apps = await repo.list_all()
            names = [a.display_name for a in apps]
            assert names == ["Keep"]
        finally:
            await pool.release(conn); await pool.close()

    @pytest.mark.asyncio
    async def test_delete_missing_refuses_to_wipe_on_empty_set(self):
        # If the discoverer reported zero apps, that's almost
        # certainly a bug, not "the user uninstalled everything."
        # Guard rail: don't nuke the table.
        repo, pool, conn = await _make_repo()
        try:
            from cosma_backend.models import Application
            await repo.upsert_many([
                Application(app_path="/Applications/Stay.app", display_name="Stay"),
            ])
            removed = await repo.delete_missing(set())
            assert removed == 0
            assert await repo.count() == 1
        finally:
            await pool.release(conn); await pool.close()


@pytest.mark.unit
class TestApplicationsIndexerEndToEnd:
    """Scan a synthetic /Applications root via the real indexer."""

    @pytest.mark.asyncio
    async def test_scan_populates_table_and_fts(self, tmp_path: Path):
        _make_app(
            tmp_path, "Pixelmator",
            CFBundleDisplayName="Pixelmator",
            CFBundleIdentifier="com.pixelmatorteam.pixelmator",
            CFBundleGetInfoString="Powerful image editor for the Mac",
            LSApplicationCategoryType="public.app-category.graphics-design",
        )

        repo, pool, conn = await _make_repo()
        try:
            discoverer = ApplicationsDiscoverer(search_roots=(str(tmp_path),))
            indexer = ApplicationsIndexer(repo, discoverer=discoverer)
            result = await indexer.index_now()
            assert result.discovered == 1
            assert result.upserted == 1
            assert result.pruned == 0

            # Through FTS, "image editor" must find Pixelmator (description
            # tokenizes into the FTS index).
            cur = await conn.execute(
                "SELECT count(*) FROM applications_fts "
                "WHERE applications_fts MATCH 'image editor'"
            )
            assert (await cur.fetchall())[0][0] == 1
        finally:
            await pool.release(conn); await pool.close()

    @pytest.mark.asyncio
    async def test_uninstall_prunes(self, tmp_path: Path):
        bundle = _make_app(tmp_path, "Tmp", CFBundleDisplayName="Tmp")
        repo, pool, conn = await _make_repo()
        try:
            indexer = ApplicationsIndexer(
                repo, ApplicationsDiscoverer(search_roots=(str(tmp_path),)),
            )
            await indexer.index_now()
            assert await repo.count() == 1
            # Simulate uninstall — but leave one decoy so the empty-set
            # safety guard doesn't fire.
            import shutil
            shutil.rmtree(bundle)
            _make_app(tmp_path, "Keeper", CFBundleDisplayName="Keeper")
            result = await indexer.index_now()
            assert result.pruned == 1
            apps = await repo.list_all()
            assert [a.display_name for a in apps] == ["Keeper"]
        finally:
            await pool.release(conn); await pool.close()
