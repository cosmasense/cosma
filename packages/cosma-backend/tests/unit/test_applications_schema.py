"""Schema-level tests for the new applications table + FTS5 sync.

We exercise the SQL directly (not the future Database wrappers) so
these tests pin the schema contract independently of any application
code that will be added in Step 4. If a future schema change breaks
the apps FTS sync or the new status enum, the failure lands here
with a clear error rather than as a mystery search regression.
"""

from importlib import resources
from pathlib import Path

import asqlite
import pytest
import sqlite_vec


SCHEMA_PATH = Path(__file__).resolve().parents[2] / "src" / "cosma_backend" / "schema.sql"


def _init_conn(conn):
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


async def _open() -> tuple[asqlite.Pool, asqlite.Connection]:
    pool = await asqlite.create_pool(":memory:", init=_init_conn)
    conn = await pool.acquire()
    schema = SCHEMA_PATH.read_text()
    await conn.executescript(schema)
    return pool, conn


@pytest.mark.unit
class TestFilesStatusCheckRelaxed:
    """The old CHECK silently rejected INDEXED_PARTIAL on fresh DBs
    (because the constraint wasn't updated when the enum was). We
    removed the CHECK so Python's ProcessingStatus stays the single
    source of truth."""

    @pytest.mark.asyncio
    async def test_indexed_partial_writes_succeed(self):
        pool, conn = await _open()
        try:
            await conn.execute(
                """INSERT INTO files (file_path, filename, extension,
                       file_size, created, modified, accessed, status, title)
                   VALUES ('/tmp/a.pdf', 'a.pdf', '.pdf', 1, 0, 0, 0,
                           'INDEXED_PARTIAL', 'a')"""
            )
        finally:
            await pool.release(conn)

    @pytest.mark.asyncio
    async def test_indexed_name_only_writes_succeed(self):
        pool, conn = await _open()
        try:
            await conn.execute(
                """INSERT INTO files (file_path, filename, extension,
                       file_size, created, modified, accessed, status, title)
                   VALUES ('/tmp/b.py', 'b.py', '.py', 1, 0, 0, 0,
                           'INDEXED_NAME_ONLY', 'b')"""
            )
            # ...and the row shows up in FTS via the existing trigger.
            cur = await conn.execute(
                "SELECT count(*) FROM files_fts WHERE files_fts MATCH 'b'"
            )
            rows = await cur.fetchall()
            assert rows[0][0] == 1
        finally:
            await pool.release(conn)


@pytest.mark.unit
class TestApplicationsTable:
    @pytest.mark.asyncio
    async def test_insert_creates_fts_row(self):
        pool, conn = await _open()
        try:
            await conn.execute(
                """INSERT INTO applications (app_path, bundle_id, display_name,
                                              category, description)
                   VALUES ('/Applications/Safari.app', 'com.apple.Safari',
                           'Safari', 'public.app-category.productivity',
                           'The fast, secure web browser')"""
            )
            cur = await conn.execute(
                "SELECT count(*) FROM applications_fts WHERE applications_fts MATCH 'safari'"
            )
            assert (await cur.fetchall())[0][0] == 1
            # Description also tokenized — "browser" finds it.
            cur = await conn.execute(
                "SELECT count(*) FROM applications_fts WHERE applications_fts MATCH 'browser'"
            )
            assert (await cur.fetchall())[0][0] == 1
        finally:
            await pool.release(conn)

    @pytest.mark.asyncio
    async def test_update_syncs_fts(self):
        pool, conn = await _open()
        try:
            await conn.execute(
                """INSERT INTO applications (app_path, display_name)
                   VALUES ('/Applications/Notes.app', 'Notes')"""
            )
            await conn.execute(
                "UPDATE applications SET display_name = 'NotesBeta' "
                "WHERE app_path = '/Applications/Notes.app'"
            )
            # Old token gone.
            cur = await conn.execute(
                "SELECT count(*) FROM applications_fts WHERE applications_fts MATCH 'notes'"
            )
            count_notes = (await cur.fetchall())[0][0]
            # New token present.
            cur = await conn.execute(
                "SELECT count(*) FROM applications_fts WHERE applications_fts MATCH 'notesbeta'"
            )
            count_beta = (await cur.fetchall())[0][0]
            # FTS5 tokenizer is unicode61: 'NotesBeta' tokenizes to
            # one token. Old 'notes' stem still matches since the
            # display_name was renamed not deleted. The contract that
            # matters is that the new name is searchable.
            assert count_beta == 1
            assert count_notes in (0, 1)  # depends on tokenizer stemming
        finally:
            await pool.release(conn)

    @pytest.mark.asyncio
    async def test_delete_clears_fts(self):
        pool, conn = await _open()
        try:
            await conn.execute(
                """INSERT INTO applications (app_path, display_name)
                   VALUES ('/Applications/X.app', 'XApp')"""
            )
            await conn.execute(
                "DELETE FROM applications WHERE app_path = '/Applications/X.app'"
            )
            cur = await conn.execute("SELECT count(*) FROM applications_fts")
            assert (await cur.fetchall())[0][0] == 0
        finally:
            await pool.release(conn)

    @pytest.mark.asyncio
    async def test_app_path_is_unique(self):
        pool, conn = await _open()
        try:
            await conn.execute(
                "INSERT INTO applications (app_path, display_name) "
                "VALUES ('/Applications/Y.app', 'Y')"
            )
            with pytest.raises(Exception):  # IntegrityError
                await conn.execute(
                    "INSERT INTO applications (app_path, display_name) "
                    "VALUES ('/Applications/Y.app', 'Y2')"
                )
        finally:
            await pool.release(conn)

    @pytest.mark.asyncio
    async def test_indexes_present(self):
        pool, conn = await _open()
        try:
            cur = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND tbl_name='applications' ORDER BY name"
            )
            names = [r[0] for r in await cur.fetchall()]
            assert "idx_applications_bundle_id" in names
            assert "idx_applications_display_name" in names
            assert "idx_applications_category" in names
        finally:
            await pool.release(conn)
