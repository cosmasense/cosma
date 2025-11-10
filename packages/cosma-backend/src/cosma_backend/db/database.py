from __future__ import annotations

import datetime
import logging
from sqlite3 import Row
import struct
from types import TracebackType
from typing import TYPE_CHECKING, Optional, Self, Type
import asqlite
import sqlite_vec
import numpy as np

from cosma_backend.logging import sm
from cosma_backend.models import File
from cosma_backend.models.watch import WatchedDirectory
from cosma_backend.utils.bundled import get_bundled_file_text

if TYPE_CHECKING:
    from sqlite3 import Connection as Sqlite3Connection

# The schema file is bundled with the distribution
SCHEMA_FILE = "./schema.sql"

logger = logging.getLogger(__name__)


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
            # initialize sqlite_vec in each connection
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)

        pool = await asqlite.create_pool(path, init=init_conn)

        # perform migrations (or create tables)
        schema = get_bundled_file_text(SCHEMA_FILE)

        async with pool.acquire() as conn:
            await conn.executescript(schema)

        return cls(pool)
        
    async def close(self):
        await self.pool.close()

    async def __aenter__(self, ):
        pass

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        if self._closed is False:
            await self.pool.close()
            self._closed = True

    # ====== Helper Functions ======
    def acquire(self) -> asqlite._AcquireProxyContextManager:
        return self.pool.acquire()

    # ====== Database Logic ======
    async def insert_file(self, file: File):
        SQL = """INSERT INTO files (filename, extension, created, modified, summary)
                 VALUES (?, ?, ?, ?, ?)
                 RETURNING id;
              """

        async with self.acquire() as conn:
            await conn.execute(SQL, (
                file.filename,
                file.extension,
                file.created,
                file.modified,
                file.summary,
            ))
            
    # Add these methods to your Database class:
    
    async def get_file_by_path(self, file_path: str) -> Optional[File]:
        """Get file by its path."""
        SQL = "SELECT * FROM files WHERE file_path = ?"
        async with self.acquire() as conn:
            async with conn.execute(SQL, (file_path,)) as cursor:
                row = await cursor.fetchone()
                if row:
                    return File.from_row(row)
        return None
    
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
                        "INSERT INTO file_keywords (file_id, keyword) VALUES (?, ?)",
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

    def _normalize_embedding_dimensions(self, embedding: np.ndarray, target_dimensions: int = 1536) -> np.ndarray:
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
                
    async def upsert_file_embeddings(self, file: File) -> None:
        """
        Insert or update embedding for a file.

        Args:
            file_id: ID of the file
            embedding: Embedding vector as numpy array
            model_name: Name of the model used
            dimensions: Actual dimensions of the embedding
        """
        logger.debug(sm("Inserting embedding", file_id=file.id, model=file.embedding_model, dimensions=file.embedding_dimensions))

        # Normalize embedding to 1536 dimensions for consistent storage
        normalized_embedding = self._normalize_embedding_dimensions(file.embedding)

        # Serialize embedding
        embedding_blob = self._serialize_vector(normalized_embedding)

        async with self.acquire() as conn:
            # Check if embedding already exists
            existing = await conn.fetchone(
                "SELECT 1 FROM file_embeddings WHERE file_id = ?",
                (file.id,)
            )

            # Always delete existing embeddings first (vec0 virtual tables don't support INSERT OR REPLACE)
            await conn.execute(
                "DELETE FROM file_embeddings WHERE file_id = ?",
                (file.id,)
            )
            # await conn.execute(
            #     "DELETE FROM embedding_metadata WHERE file_id = ?",
            #     (file.id,)
            # )

            # Insert into vec0 table (using normalized dimensions)
            await conn.execute(
                """
                INSERT INTO file_embeddings(file_id, embedding_model, embedding_dimensions, embedding)
                VALUES (?, ?, ?, ?)
                """,
                (file.id, file.embedding_model, 1536, embedding_blob)
            )

            # Insert metadata
            # await conn.execute(
            #     """
            #     INSERT INTO embedding_metadata(file_id, model_name, model_dimensions)
            #     VALUES (?, ?, ?)
            #     """,
            #     (file.id, file.embedding_model, file.embedding_dimensions)
            # )

            logger.info(sm("Embedding inserted successfully", file_id=file.id))
            
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
        logger.debug(sm("Searching similar files", limit=limit, threshold=threshold, directory=directory))

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

            logger.info(sm("Found similar files", count=len(results)))
            return results

    async def get_file_embedding(self, file_id: str) -> tuple[np.ndarray, str, int] | None:
        """
        Get embedding vector for a file.

        Args:
            file_id: ID of the file

        Returns:
            Tuple of (embedding, model_name, dimensions) or None
        """
        SQL = """
        SELECT
            fe.embedding,
            em.model_name,
            em.model_dimensions
        FROM file_embeddings fe
        INNER JOIN embedding_metadata em ON fe.file_id = em.file_id
        WHERE fe.file_id = ?
        """

        async with self.acquire() as conn:
            row = await conn.fetchone(SQL, (file_id,))

            if not row:
                return None

            # Deserialize embedding using normalized dimensions (1536) since that's how it's stored
            # The vector was normalized to 1536 dimensions before storage in line 514
            embedding = self._deserialize_vector(row["embedding"], 1536)
            
            # Truncate back to original model dimensions if needed
            original_dimensions = row["model_dimensions"]
            if original_dimensions < 1536:
                embedding = embedding[:original_dimensions]

            return (embedding, row["model_name"], row["model_dimensions"])

    async def delete_embedding(self, file_id: str) -> bool:
        """
        Delete embedding for a file.

        Args:
            file_id: ID of the file

        Returns:
            True if deleted, False if not found
        """
        SQL = "DELETE FROM file_embeddings WHERE file_id = ?"

        async with self.acquire() as conn:
            cursor = await conn.execute(SQL, (file_id,))
            rows_affected = cursor.get_cursor().rowcount

            if rows_affected > 0:
                # Also delete metadata
                # await conn.execute("DELETE FROM embedding_metadata WHERE file_id = ?", (file_id,))
                logger.info(sm("Embedding deleted", file_id=file_id))
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
                logger.info(sm("Watched directory added", path=watched_dir.path_str, id=watched_dir.id, recursive=watched_dir.recursive))
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
            
            logger.info(sm("Retrieved watched directories", count=len(directories), active_only=active_only))
            return directories

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
                logger.warning(sm("Watched directory not found for deletion", job_id=job_id))
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
            
            logger.info(sm(
                "Watched directory and associated files deleted", 
                job_id=job_id, 
                path=directory_path,
                files_deleted=len(deleted_files)
            ))
            
            return watched_dir

    async def keyword_search(self, query: str, limit: int = 20, directory: str | None = None) -> list[tuple[File, float]]:
        """
        Perform keyword search using FTS5 with BM25 ranking.

        Args:
            query: Search query
            limit: Maximum number of results
            directory: Optional directory path to limit search scope

        Returns:
            List of tuples (File, relevance_score)
        """
        
        sanitized_query = f'"{query.replace('"', '""')}"'
        logger.debug(sm("Performing keyword search", query=sanitized_query, limit=limit, directory=directory))

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
                logger.error(sm("SQL query failed", error=str(e), query=sanitized_query))
                raise

            results = []
            for row in rows:
                file = File.from_row(row)
                # BM25 scores are negative (less negative = better match)
                # Convert to positive score (0-1 range approximately)
                relevance_score = abs(row["relevance_score"])
                results.append((file, relevance_score))

            logger.info(sm("Keyword search completed", count=len(results)))
            return results

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
                logger.debug(sm("File timestamp updated", file_path=file_path))
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
            
            logger.info(sm("Deleted stale files", count=len(rows), timestamp=timestamp, directory=directory_path))
            return rows


async def connect(path: str) -> Database:
    return await Database.from_path(path)
