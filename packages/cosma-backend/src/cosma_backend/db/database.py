"""
Database Implementation

Async SQLite database with:
- Connection pooling via asqlite
- sqlite-vec extension for vector similarity search
- FTS5 for full-text search with BM25 ranking
- Automatic schema migrations

Usage:
    db = await connect("/path/to/app.db")
    async with db.acquire() as conn:
        await conn.execute(...)
    await db.close()
"""

from __future__ import annotations

import asyncio
import datetime
from sqlite3 import Row
import struct
from types import TracebackType
from typing import TYPE_CHECKING, Optional, Self, Type

import asqlite
import numpy as np
import sqlite_vec

from cosma_backend.db.errors import DatabaseClosingError
from cosma_backend.logging import get_logger
from cosma_backend.models import File
from cosma_backend.models.watch import WatchedDirectory
from cosma_backend.utils.bundled import get_bundled_file_text

if TYPE_CHECKING:
    from sqlite3 import Connection as Sqlite3Connection

# Schema file bundled with the package
SCHEMA_FILE = "./schema.sql"

# All embeddings are normalized to this dimension for consistent storage
# This allows mixing different embedding models (e.g., 768-dim local, 512-dim OpenAI)
EMBEDDING_STORAGE_DIMENSIONS = 1536

logger = get_logger(__name__)


def to_timestamp(dt: datetime.datetime | None):
    return int(dt.timestamp()) if dt else None


class Database:
    pool: asqlite.Pool
    _closed: bool

    # ====== Python Magic Functions ======
    # __aenter__ and __aexit__ are for async context managers

    def __init__(self, pool: asqlite.Pool):
        self.pool = pool
        self._closed = False

    @classmethod
    async def from_path(cls, path: str) -> Self:
        def init_conn(conn: Sqlite3Connection):
            # WAL mode: crash-safe journaling that survives sudden termination
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            # synchronous=NORMAL is the SQLite-recommended setting for WAL.
            # It still survives application crashes and OS crashes — the
            # only loss window is a power failure during the last in-flight
            # transaction. Cuts the per-commit fsync count in half. See
            # https://sqlite.org/wal.html#performance_considerations.
            conn.execute("PRAGMA synchronous=NORMAL")
            # Memory-map up to 256 MiB of the DB file for reads. SQLite
            # serves hot pages directly from the kernel page cache without
            # going through pread(). Speeds up bulk skip-check + FTS
            # MATCH against a warm DB. No memory cost when the DB is
            # smaller than the cap.
            conn.execute("PRAGMA mmap_size=268435456")
            # 20 MiB page cache (10× the SQLite default). Keeps recently-
            # touched files, FTS tokens, and queue items in memory across
            # the request churn instead of paging through pread().
            conn.execute("PRAGMA cache_size=-20000")
            # Sort/aggregate scratch space lives in RAM, not in a temp
            # file. Tiny win on `get_files_under_directory_summary` and
            # any GROUP_CONCAT path through the FTS triggers.
            conn.execute("PRAGMA temp_store=MEMORY")
            # Cap the WAL at ~32 MB and trigger an automatic checkpoint
            # every 1000 dirty pages. SQLite's default auto-checkpoint is
            # 1000 pages too, but it can be inhibited by long-lived
            # readers — and during heavy enqueue + search overlap we
            # observed a single user accumulate an 852 MB WAL because no
            # writer ever got to run a checkpoint cleanly. Once the WAL
            # is that size, every read merges 800+ MB of pending frames
            # before serving, which froze the UI for tens of seconds.
            # `journal_size_limit` tells SQLite to truncate the WAL
            # back down on the next checkpoint instead of letting it
            # grow without bound; the explicit autocheckpoint is a
            # belt-and-suspenders default.
            conn.execute("PRAGMA wal_autocheckpoint=1000")
            conn.execute("PRAGMA journal_size_limit=33554432")  # 32 MiB
            # initialize sqlite_vec in each connection
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)

        pool = await asqlite.create_pool(path, init=init_conn)

        # perform migrations (or create tables)
        schema = get_bundled_file_text(SCHEMA_FILE)

        async with pool.acquire() as conn:
            await conn.executescript(schema)

        instance = cls(pool)
        # Background checkpoint task: PASSIVE every 60s as a safety net
        # for the autocheckpoint above. PASSIVE is non-blocking — it
        # only checkpoints frames not currently held by readers, so it
        # can't stall the request path. See `_checkpoint_loop`.
        instance._start_checkpoint_task()
        return instance

    def _start_checkpoint_task(self) -> None:
        loop = asyncio.get_event_loop()
        self._checkpoint_task = loop.create_task(self._checkpoint_loop())

    async def _checkpoint_loop(self) -> None:
        """Run PRAGMA wal_checkpoint(PASSIVE) every 60s.

        PASSIVE is the safe variant: it never blocks readers and never
        forces a writer to wait. If readers are holding old frames it
        simply skips them and tries again next tick. That's exactly the
        right policy for a UI-driven workload — we'd rather have a
        slowly-shrinking WAL than a brief stall every minute.
        """
        while not self._closed:
            try:
                await asyncio.sleep(60)
                if self._closed:
                    return
                async with self.pool.acquire() as conn:
                    await conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            except asyncio.CancelledError:
                return
            except Exception as e:
                # Don't take down the loop for a transient error.
                # Log via aiosqlite — a checkpoint failure is operational
                # info, not a fatal condition.
                from cosma_backend.logging import get_logger
                get_logger(__name__).warning(
                    "WAL checkpoint failed; will retry next tick",
                    error=str(e),
                )
        
    async def close(self) -> None:
        """Close the connection pool."""
        self._closed = True
        # Stop the background checkpoint task before closing the pool —
        # otherwise it may try to acquire from a half-closed pool and
        # log a spurious warning during shutdown.
        task = getattr(self, "_checkpoint_task", None)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        # Final checkpoint before close so the next launch starts with
        # a clean WAL — readers cleaned up by the cancel above mean
        # TRUNCATE can fully reclaim the file.
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception:
            pass
        await self.pool.close()

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """Async context manager exit - close pool if not already closed."""
        if not self._closed:
            self._closed = True
            await self.pool.close()

    # ====== Helper Functions ======

    def acquire(self) -> asqlite._AcquireProxyContextManager:
        """Acquire a connection from the pool."""
        if self._closed:
            raise DatabaseClosingError()
        return self.pool.acquire()

    # ====== File Operations ======

    async def get_file_by_path(self, file_path: str) -> Optional[File]:
        """Get file by its path."""
        SQL = "SELECT * FROM files WHERE file_path = ?"
        async with self.acquire() as conn:
            async with conn.execute(SQL, (file_path,)) as cursor:
                row = await cursor.fetchone()
                if row:
                    return File.from_row(row)
        return None
    
    async def get_files_under_directory_summary(self, directory_path: str) -> dict[str, tuple[str, datetime.datetime | None]]:
        """Bulk-load (status, modified) for every file under ``directory_path``.

        Used by `pipeline.enqueue_directory` to avoid issuing one
        ``SELECT * FROM files WHERE file_path = ?`` per file during
        directory enumeration. For a 10k-file watched folder, the
        per-file approach was 10k DB roundtrips per startup discovery
        sweep — visibly responsible for the post-launch CPU spike. One
        prefix-LIKE query is dramatically cheaper because SQLite can
        use the file_path index for the range scan.

        Returns a dict keyed by file_path → (status_str, modified_dt).
        Empty dict (not None) when nothing matches, so callers can
        unconditionally `.get()` without a None check.
        """
        # Trailing slash on the prefix so we don't match siblings that
        # share a path prefix with the directory name.
        prefix = directory_path.rstrip("/") + "/"
        SQL = "SELECT file_path, status, modified FROM files WHERE file_path LIKE ?"
        out: dict[str, tuple[str, datetime.datetime | None]] = {}

        def _parse_mtime(value) -> Optional[datetime.datetime]:
            # Mirror models.file.from_row's parse_timestamp logic so the
            # caller can compare against File.modified directly. Stored
            # as a unix timestamp; tolerate already-datetime values for
            # a future schema change.
            if not value:
                return None
            if isinstance(value, datetime.datetime):
                return value
            try:
                return datetime.datetime.fromtimestamp(value)
            except (ValueError, TypeError, OSError):
                return None

        # asqlite's Cursor supports `await cursor.fetchall()` but not
        # `async for cursor` (no __aiter__). The previous code crashed
        # with TypeError the first time it was actually exercised —
        # discovered when test_db_bench hit this path for the first
        # time. fetchall() loads the whole rowset into memory; that's
        # fine for this method's use case (one bulk skip-check per
        # directory enqueue, capped by the watched-folder size).
        async with self.acquire() as conn:
            cursor = await conn.execute(SQL, (prefix + "%",))
            rows = await cursor.fetchall()
            for row in rows:
                out[row["file_path"]] = (
                    row["status"], _parse_mtime(row["modified"]),
                )
        return out

    async def touch_files_timestamps(self, file_paths: list[str]) -> int:
        """Bulk-update updated_at on a list of file_paths in one statement.

        Replaces N individual `update_file_timestamp` calls with a single
        UPDATE ... WHERE file_path IN (...). Used by enqueue_directory
        as the "mark this file as still on disk" half of the
        mark-and-sweep stale-file cleanup.

        Chunked at 500 paths per statement to stay safely under SQLite's
        SQLITE_LIMIT_VARIABLE_NUMBER (default 32766 in modern SQLite, but
        older builds capped at 999). Returns total rows affected.
        """
        if not file_paths:
            return 0
        total = 0
        CHUNK = 500
        async with self.acquire() as conn:
            for i in range(0, len(file_paths), CHUNK):
                chunk = file_paths[i:i + CHUNK]
                placeholders = ",".join("?" for _ in chunk)
                SQL = f"UPDATE files SET updated_at = (strftime('%s', 'now')) WHERE file_path IN ({placeholders})"
                cursor = await conn.execute(SQL, tuple(chunk))
                total += cursor.get_cursor().rowcount
        return total

    async def get_file_by_hash(self, content_hash: str) -> Optional[File]:
        """Get file by content hash."""
        SQL = "SELECT * FROM files WHERE content_hash = ?"
        async with self.acquire() as conn:
            async with conn.execute(SQL, (content_hash,)) as cursor:
                row = await cursor.fetchone()
                if row:
                    return File.from_row(row)
        return None
    
    async def upsert_file(self, file_data: File) -> int:
        """
        Insert or update a file record.
        Returns the file ID.
        """
        
        # Check if exists
        async with self.acquire() as conn:
            existing = await conn.fetchone(
                "SELECT 1 FROM files WHERE file_path = ?",
                (file_data.file_path,)
            )
                
        
        if existing:
            # Update
            SQL = """
                UPDATE files 
                SET filename=?, extension=?, file_size=?, created=?, modified=?, accessed=?,
                    content_type=?, content_hash=?,
                    summary=?, title=?, status=?, processing_error=?,
                    parsed_at=?, summarized_at=?, embedded_at=?, updated_at=(strftime('%s', 'now'))
                WHERE file_path=?
                RETURNING id
            """
            params = (
                file_data.filename, file_data.extension, file_data.file_size,
                to_timestamp(file_data.created), to_timestamp(file_data.modified), to_timestamp(file_data.accessed),
                file_data.content_type, file_data.content_hash,
                file_data.summary, file_data.title, file_data.status.name, file_data.processing_error,
                to_timestamp(file_data.parsed_at), to_timestamp(file_data.summarized_at), to_timestamp(file_data.embedded_at),
                file_data.file_path
            )
        else:
            # Insert
            SQL = """
                INSERT INTO files (
                    file_path, filename, extension, file_size, created, modified, accessed,
                    content_type, content_hash, summary, title,
                    status, processing_error, parsed_at, summarized_at, embedded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id
            """
            params = (
                file_data.file_path, file_data.filename, file_data.extension,
                file_data.file_size, to_timestamp(file_data.created), to_timestamp(file_data.modified), to_timestamp(file_data.accessed),
                file_data.content_type, file_data.content_hash,
                file_data.summary, file_data.title, file_data.status.name, file_data.processing_error,
                to_timestamp(file_data.parsed_at), to_timestamp(file_data.summarized_at), to_timestamp(file_data.embedded_at)
            )
        
        async with self.acquire() as conn:
            async with conn.execute(SQL, params) as cursor:
                row = await cursor.fetchone()
                file_data.id = row[0]
                
            # Update keywords in file_keywords table
            if file_data.keywords:
                # Delete existing keywords
                await conn.execute(
                    "DELETE FROM file_keywords WHERE file_id = ?",
                    (file_data.id,)
                )
                
                # Insert new keywords
                for keyword in file_data.keywords:
                    await conn.execute(
                        "INSERT OR IGNORE INTO file_keywords (file_id, keyword) VALUES (?, ?)",
                        (file_data.id, keyword)
                    )
            
            return file_data.id
                
    # ====== Vector Embedding Operations ======

    def _serialize_vector(self, vector: np.ndarray) -> bytes:
        """Serialize numpy array to bytes for sqlite-vec storage."""
        # Ensure vector is float32
        vector = vector.astype(np.float32)
        # Pack as bytes
        return struct.pack(f"{len(vector)}f", *vector)

    def _deserialize_vector(self, blob: bytes, dimensions: int) -> np.ndarray:
        """Deserialize bytes from sqlite-vec to numpy array."""
        # Unpack bytes to float array
        values = struct.unpack(f"{dimensions}f", blob)
        return np.array(values, dtype=np.float32)

    def _normalize_embedding_dimensions(self, embedding: np.ndarray, target_dimensions: int = EMBEDDING_STORAGE_DIMENSIONS) -> np.ndarray:
        """
        Normalize embedding to target dimensions by padding with zeros or truncating.

        Args:
            embedding: Input embedding vector
            target_dimensions: Target dimension count (default 1536)

        Returns:
            Normalized embedding vector
        """
        current_dims = embedding.shape[0]

        if current_dims == target_dimensions:
            return embedding
        if current_dims < target_dimensions:
            # Pad with zeros
            padded = np.zeros(target_dimensions, dtype=embedding.dtype)
            padded[:current_dims] = embedding
            return padded
        # Truncate to target dimensions
        return embedding[:target_dimensions]
                
    async def upsert_file_embeddings(
        self, file: File, *, first_embed: bool = False,
    ) -> None:
        """
        Insert or update embedding for a file.

        Args:
            file: file with .id, .embedding, .embedding_model, .embedding_dimensions set.
            first_embed: True iff the caller knows this file has never had an
                embedding row before (e.g., resuming from PARSED/SUMMARIZED, or
                a brand-new file). Lets us skip the leading DELETE — vec0
                doesn't support INSERT OR REPLACE so we usually have to delete
                first, but on a first-time insert there's nothing to delete and
                the round-trip is wasted. Saves ~0.5 ms per file on a hot DB,
                more on cold.
        """
        logger.debug("Inserting embedding", file_id=file.id, model=file.embedding_model, dimensions=file.embedding_dimensions)

        # Normalize embedding to standard storage dimensions for consistent vector search
        normalized_embedding = self._normalize_embedding_dimensions(file.embedding)
        embedding_blob = self._serialize_vector(normalized_embedding)

        async with self.acquire() as conn:
            if not first_embed:
                # vec0 virtual tables don't support INSERT OR REPLACE.
                # Re-embed paths must clear the prior row; first-time
                # embed paths skip this round-trip entirely.
                await conn.execute(
                    "DELETE FROM file_embeddings WHERE file_id = ?",
                    (file.id,)
                )

            # Insert into vec0 table with normalized dimensions
            await conn.execute(
                """
                INSERT INTO file_embeddings(file_id, embedding_model, embedding_dimensions, embedding)
                VALUES (?, ?, ?, ?)
                """,
                (file.id, file.embedding_model, EMBEDDING_STORAGE_DIMENSIONS, embedding_blob)
            )

            logger.info("Embedding inserted successfully", file_id=file.id)
            
    async def search_similar_files(self, query_embedding: np.ndarray, limit: int = 10, threshold: float | None = None, directory: str | None = None) -> list[tuple[File, float]]:
        """
        Search for files similar to the query embedding.

        Args:
            query_embedding: Query vector as numpy array
            limit: Maximum number of results
            threshold: Optional similarity threshold (lower is more similar)
            directory: Optional directory path to limit search scope

        Returns:
            List of tuples (FileMetadata, distance)
        """
        logger.debug("Searching similar files", limit=limit, threshold=threshold, directory=directory)

        # Normalize and serialize query embedding
        normalized_embedding = self._normalize_embedding_dimensions(query_embedding)
        query_blob = self._serialize_vector(normalized_embedding)

        # Build SQL query with k parameter for knn search
        SQL = """
        SELECT
            f.*,
            GROUP_CONCAT(fk.keyword, '||') as keywords_str,
            distance
        FROM file_embeddings
        INNER JOIN files f ON file_embeddings.file_id = f.id
        LEFT JOIN file_keywords fk ON f.id = fk.file_id
        WHERE embedding MATCH ? AND k = ?
        """

        if threshold is not None:
            SQL += f" AND distance <= {threshold}"

        if directory is not None:
            SQL += " AND (f.file_path LIKE ? || '/%' OR f.file_path = ?)"

        SQL += """
        GROUP BY f.id
        ORDER BY distance
        LIMIT ?
        """

        params = [query_blob, limit]
        if directory is not None:
            params.extend([directory, directory])
        params.append(limit)

        async with self.acquire() as conn:
            rows = await conn.fetchall(SQL, tuple(params))

            results = []
            for row in rows:
                file = File.from_row(row)
                distance = row["distance"]
                results.append((file, distance))

            logger.info("Found similar files", count=len(results))
            return results

    async def get_file_embedding(self, file_id: int) -> tuple[np.ndarray, str, int] | None:
        """
        Get embedding vector for a file.

        Args:
            file_id: ID of the file (integer)

        Returns:
            Tuple of (embedding, model_name, dimensions) or None if not found
        """
        SQL = """
        SELECT embedding, embedding_model, embedding_dimensions
        FROM file_embeddings
        WHERE file_id = ?
        """

        async with self.acquire() as conn:
            row = await conn.fetchone(SQL, (file_id,))

            if not row:
                return None

            # Deserialize embedding using storage dimensions
            embedding = self._deserialize_vector(row["embedding"], EMBEDDING_STORAGE_DIMENSIONS)

            # Truncate back to original model dimensions if needed
            original_dimensions = row["embedding_dimensions"]
            if original_dimensions < EMBEDDING_STORAGE_DIMENSIONS:
                embedding = embedding[:original_dimensions]

            return (embedding, row["embedding_model"], row["embedding_dimensions"])

    async def delete_embedding(self, file_id: int) -> bool:
        """
        Delete embedding for a file.

        Args:
            file_id: ID of the file (integer)

        Returns:
            True if deleted, False if not found
        """
        SQL = "DELETE FROM file_embeddings WHERE file_id = ?"

        async with self.acquire() as conn:
            cursor = await conn.execute(SQL, (file_id,))
            rows_affected = cursor.get_cursor().rowcount

            if rows_affected > 0:
                logger.info("Embedding deleted", file_id=file_id)
                return True

            return False

    async def delete_file(self, file_path: str) -> File | None:
        """
        Delete a file.

        Args:
            file_path: Path of the file to delete

        Returns:
            True if deleted, False if not found
        """
        async with self.acquire() as conn:
            # Delete file record
            row = await conn.fetchone("DELETE FROM files WHERE file_path = ? RETURNING *", (file_path,))

        if not row:
            return None

        return File.from_row(row)

    async def delete_files_in_directory(self, directory_path: str) -> list[File]:
        """
        Delete all files within a directory (including subdirectories).

        Args:
            directory_path: Path of the directory

        Returns:
            List of deleted File objects
        """
        async with self.acquire() as conn:
            # Delete all files in this directory
            SQL = """
                DELETE FROM files
                WHERE file_path LIKE ? || '/%' OR file_path = ?
                RETURNING *
            """
            rows = await conn.fetchall(SQL, (directory_path, directory_path))

            deleted_files = [File.from_row(row) for row in rows]
            logger.info("Deleted files in directory", directory=directory_path, count=len(deleted_files))
            return deleted_files

    async def add_watched_directory(self, watched_dir: WatchedDirectory) -> int:
        """
        Add a directory to the watched_directories table.

        Args:
            watched_dir: WatchedDirectory instance to add

        Returns:
            The ID of the watched directory record
        """
        SQL = """
        INSERT INTO watched_directories (path, recursive, file_pattern, last_scan)
        VALUES (?, ?, ?, (strftime('%s', 'now')))
        ON CONFLICT(path) DO UPDATE SET
            is_active = 1,
            recursive = excluded.recursive,
            file_pattern = excluded.file_pattern,
            last_scan = (strftime('%s', 'now'))
        RETURNING id
        """
        
        async with self.acquire() as conn:
            async with conn.execute(SQL, (
                watched_dir.path_str,
                1 if watched_dir.recursive else 0,
                watched_dir.file_pattern
            )) as cursor:
                row = await cursor.fetchone()
                watched_dir.id = row[0]
                logger.info("Watched directory added", path=watched_dir.path_str, id=watched_dir.id, recursive=watched_dir.recursive)
                return watched_dir.id

    async def get_watched_directories(self, active_only: bool = True) -> list[WatchedDirectory]:
        """
        Get all watched directories from the database.

        Args:
            active_only: If True, only return active directories (default: True)

        Returns:
            List of WatchedDirectory instances
        """
        SQL = """
        SELECT id, path, is_active, recursive, file_pattern, last_scan, created_at, updated_at
        FROM watched_directories
        """
        
        if active_only:
            SQL += " WHERE is_active = 1"
        
        SQL += " ORDER BY created_at"
        
        async with self.acquire() as conn:
            rows = await conn.fetchall(SQL)
            
            directories = []
            for row in rows:
                watched_dir = WatchedDirectory.from_row(row)
                directories.append(watched_dir)
            
            logger.info("Retrieved watched directories", count=len(directories), active_only=active_only)
            return directories

    async def count_files_in_directory(self, directory_path: str) -> int:
        """Count indexed files under a watched directory path."""
        async with self.acquire() as conn:
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM files WHERE file_path LIKE ? || '/' || '%' OR file_path = ?",
                (directory_path, directory_path),
            )
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def delete_watched_directory(self, job_id: int) -> WatchedDirectory | None:
        """
        Delete a watched directory by ID and clean up all associated files.

        Args:
            job_id: ID of the watched directory to delete

        Returns:
            WatchedDirectory instance if deleted, None if not found
        """
        async with self.acquire() as conn:
            # First, get the watched directory to retrieve its path
            get_dir_sql = "SELECT * FROM watched_directories WHERE id = ?"
            row = await conn.fetchone(get_dir_sql, (job_id,))
            
            if not row:
                logger.warning("Watched directory not found for deletion", job_id=job_id)
                return None
            
            watched_dir = WatchedDirectory.from_row(row)
            directory_path = watched_dir.path_str
            
            # Delete all files in this directory (using LIKE for path matching)
            # This handles both direct files and files in subdirectories
            delete_files_sql = """
                DELETE FROM files 
                WHERE file_path LIKE ? || '/%' OR file_path = ?
                RETURNING id, file_path
            """
            deleted_files = await conn.fetchall(delete_files_sql, (directory_path, directory_path))
            
            # Delete the watched directory
            delete_dir_sql = "DELETE FROM watched_directories WHERE id = ?"
            await conn.execute(delete_dir_sql, (job_id,))
            
            logger.info(
                "Watched directory and associated files deleted", 
                job_id=job_id, 
                path=directory_path,
                files_deleted=len(deleted_files)
            )
            
            return watched_dir

    async def keyword_search(self, query: str, limit: int = 20, directory: str | None = None, allow_operators: bool = False) -> list[tuple[File, float]]:
        """
        Perform keyword search using FTS5 with BM25 ranking.

        Args:
            query: Search query
            limit: Maximum number of results
            directory: Optional directory path to limit search scope
            allow_operators: If True, preserve AND/OR/NOT operators

        Returns:
            List of tuples (File, relevance_score)
        """
        # Import here to avoid circular import
        from cosma_backend.searcher.fts5_query import parse_fts5_query

        sanitized_query = parse_fts5_query(query, allow_operators=allow_operators)
        if not sanitized_query:
            return []

        logger.debug("Performing keyword search", query=sanitized_query, limit=limit, directory=directory)

        # FTS5 query with BM25 ranking
        # You can use advanced syntax like: "housing AND (apartment OR lease)"
        # BM25 parameters: k1=1.2 (term frequency saturation), b=0.75 (length normalization)
        SQL = """
        SELECT 
            f.*,
            bm25(files_fts) AS relevance_score
        FROM files_fts fts
        JOIN files f ON f.id = fts.rowid
        WHERE files_fts MATCH ?
        """

        params = [sanitized_query]
        
        if directory is not None:
            SQL += " AND (f.file_path LIKE ? || '/%' OR f.file_path = ?)"
            params.extend([directory, directory])

        SQL += """
        ORDER BY rank
        LIMIT ?;
        """
        
        params.append(limit)

        async with self.acquire() as conn:
            try:
                async with conn.execute(SQL, tuple(params)) as cursor:
                    rows = await cursor.fetchall()
            except Exception as e:
                logger.error("SQL query failed", error=str(e), query=sanitized_query)
                raise

            results = []
            for row in rows:
                file = File.from_row(row)
                # BM25 scores are negative (less negative = better match)
                # Convert to positive score (0-1 range approximately)
                relevance_score = abs(row["relevance_score"])
                results.append((file, relevance_score))

            logger.info("Keyword search completed", count=len(results))
            return results

    async def get_fts5_suggestions(self, prefix: str, limit: int = 10) -> list[str]:
        """
        Get autocomplete suggestions using FTS5 prefix matching.

        Args:
            prefix: Search prefix
            limit: Maximum number of suggestions

        Returns:
            List of suggested terms (filenames and titles matching the prefix)
        """
        # Import here to avoid circular import
        from cosma_backend.searcher.fts5_query import build_prefix_query

        fts5_query = build_prefix_query(prefix)
        if not fts5_query:
            return []

        logger.debug("Getting FTS5 suggestions", prefix=prefix, fts5_query=fts5_query)

        SQL = """
        SELECT DISTINCT f.filename, f.title
        FROM files_fts fts
        JOIN files f ON f.id = fts.rowid
        WHERE files_fts MATCH ?
        ORDER BY rank
        LIMIT ?
        """

        async with self.acquire() as conn:
            try:
                rows = await conn.fetchall(SQL, (fts5_query, limit * 2))
            except Exception as e:
                logger.warning("FTS5 suggestions query failed", error=str(e), prefix=prefix)
                return []

            suggestions = []
            prefix_lower = prefix.lower()

            for row in rows:
                # Add filename if it matches prefix
                filename = row["filename"]
                if filename and filename.lower().startswith(prefix_lower):
                    suggestions.append(filename)

                # Add title words that match prefix
                title = row["title"]
                if title:
                    for word in title.split():
                        if word.lower().startswith(prefix_lower) and len(word) > 2:
                            suggestions.append(word)

                if len(suggestions) >= limit:
                    break

            # Deduplicate and limit
            unique_suggestions = list(dict.fromkeys(suggestions))[:limit]

            logger.debug("Generated FTS5 suggestions", count=len(unique_suggestions))
            return unique_suggestions

    async def update_file_timestamp(self, file_path: str) -> bool:
        """
        Update the updated_at timestamp for a file to the current datetime.
    This is used to track which files are still present in the filesystem.

        Args:
            file_path: Path of the file to update

        Returns:
            True if updated, False if file not found
        """
        SQL = "UPDATE files SET updated_at = (strftime('%s', 'now')) WHERE file_path = ?"
        
        async with self.acquire() as conn:
            cursor = await conn.execute(SQL, (file_path,))
            rows_affected = cursor.get_cursor().rowcount
            
            if rows_affected > 0:
                logger.debug("File timestamp updated", file_path=file_path)
                return True
            
            return False

    async def delete_files_not_updated_since(self, timestamp: datetime.datetime, directory_path: str) -> list[Row]:
        """
        Delete files that have not been updated since the given timestamp within a specific directory.
        This is used to remove files that are no longer present in the filesystem.

        Args:
            timestamp: datetime
            directory_path: Directory path to limit deletion scope (only files under this path will be deleted)

        Returns:
            List of file paths that were deleted
        """
        directory_pattern = f"{directory_path}/%"
        
        async with self.acquire() as conn:
            # Delete the files (cascading deletes will handle embeddings and keywords)
            SQL_DELETE = "DELETE FROM files WHERE updated_at < ? AND (file_path LIKE ? OR file_path = ?) RETURNING *"
            rows = await conn.fetchall(SQL_DELETE, (to_timestamp(timestamp), directory_pattern, directory_path))
            
            logger.info("Deleted stale files", count=len(rows), timestamp=timestamp, directory=directory_path)
            return rows


    async def get_files_by_status(self, status: str, limit: int = 50, offset: int = 0) -> tuple[list[File], int]:
        """
        Get files filtered by processing status, ordered by updated_at DESC.

        Args:
            status: Processing status string (e.g. 'FAILED', 'COMPLETE')
            limit: Maximum number of files to return
            offset: Number of files to skip

        Returns:
            Tuple of (list of File objects, total count matching the status)
        """
        COUNT_SQL = "SELECT COUNT(*) as cnt FROM files WHERE status = ?"
        SELECT_SQL = """
            SELECT * FROM files
            WHERE status = ?
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
        """

        async with self.acquire() as conn:
            count_row = await conn.fetchone(COUNT_SQL, (status,))
            total_count = count_row["cnt"] if count_row else 0

            rows = await conn.fetchall(SELECT_SQL, (status, limit, offset))
            files = [File.from_row(row) for row in rows]

        return files, total_count

    # ====== Queue Item Persistence ======

    async def upsert_queue_item(self, item: dict) -> None:
        """Insert or replace a queue item."""
        SQL = """
            INSERT OR REPLACE INTO queue_items
                (id, file_path, action, status, enqueued_at, cooldown_expires_at, dest_path, retry_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        async with self.acquire() as conn:
            await conn.execute(SQL, (
                item["id"],
                item["file_path"],
                item["action"],
                item["status"],
                item["enqueued_at"],
                item["cooldown_expires_at"],
                item.get("dest_path"),
                item.get("retry_count", 0),
            ))

    async def get_queue_items(self) -> list[dict]:
        """Get all persisted queue items."""
        SQL = "SELECT * FROM queue_items"
        async with self.acquire() as conn:
            rows = await conn.fetchall(SQL)
            return [dict(row) for row in rows]

    async def delete_queue_item(self, item_id: str) -> None:
        """Delete a queue item by id."""
        SQL = "DELETE FROM queue_items WHERE id = ?"
        async with self.acquire() as conn:
            await conn.execute(SQL, (item_id,))

    async def delete_queue_items_under(self, directory: str) -> int:
        """Delete queue items whose file_path is under the given directory."""
        SQL = "DELETE FROM queue_items WHERE file_path LIKE ? || '/%' OR file_path = ?"
        async with self.acquire() as conn:
            cursor = await conn.execute(SQL, (directory, directory))
            return cursor.get_cursor().rowcount


async def connect(path: str) -> Database:
    return await Database.from_path(path)
