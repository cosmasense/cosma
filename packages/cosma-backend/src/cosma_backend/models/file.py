"""
File Model

Central domain model representing a file being indexed. The File object
accumulates data as it progresses through the pipeline stages:

1. DISCOVERED: Basic metadata (path, size, timestamps)
2. PARSED: Extracted content (text/transcript)
3. SUMMARIZED: AI-generated title, summary, keywords
4. COMPLETE: Vector embedding for semantic search

The same File instance flows through the entire pipeline, with each
stage populating additional fields.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sqlite3
from typing import Any, Optional, List, Self, TYPE_CHECKING

import numpy as np

from cosma_backend.logging import get_logger
from cosma_backend.models.status import ProcessingStatus

if TYPE_CHECKING:
    from cosma_backend.api.models import FileResponse

logger = get_logger(__name__)


@dataclass
class File:
    """
    A unified file model that progresses through the pipeline stages.
    Each stage adds more data to the model.
    """
    # Stage 0: Discovery (file system metadata)
    path: Path
    file_path: str
    filename: str
    extension: str
    file_size: int
    created: datetime
    modified: datetime
    accessed: datetime
    
    # Stage 1: Parsing (content extraction)
    id: Optional[int] = None
    content_type: Optional[str] = None
    content: Optional[str] = None
    content_hash: Optional[str] = None
    parsed_at: Optional[datetime] = None
    
    # Stage 2: Summarization (AI processing)
    summary: Optional[str] = None
    title: Optional[str] = None
    keywords: Optional[List[str]] = None
    summarized_at: Optional[datetime] = None
    
    # Stage 3: Embedding (vector representation)
    embedding: Optional[np.ndarray] = None
    embedding_model: Optional[str] = None
    embedding_dimensions: Optional[int] = None
    embedded_at: Optional[datetime] = None
    
    # Meta
    status: ProcessingStatus = ProcessingStatus.DISCOVERED
    processing_error: Optional[str] = None
    # DB bookkeeping: time the row was last written (set automatically by
    # `upsert_file` via strftime('%s','now')). This is the correct value to
    # show as "processed at" in the Recent/Failed lists — `modified` is the
    # filesystem mtime and can be years old.
    updated_at: Optional[datetime] = None

    # Transient data (not persisted to DB)
    # Used to pass video frames to the summarizer for vision analysis
    extra_images: Optional[List[bytes]] = None  # JPEG-encoded frames
    # True when the file is taking the embed-only path (no LLM
    # summary). Four distinct reasons trigger this — distinguished by
    # `partial_kind` below:
    #   * "parser_failed": the parser actually failed (codec
    #     unsupported even after H.264 transcode, no transcript and
    #     no frames, ffmpeg missing). Final status = FAILED,
    #     surfaces in Failed tab so the user can debug.
    #   * "user_elected": the user-configured filter classified this
    #     file as metadata-only. Final status = INDEXED_PARTIAL —
    #     this is intentional, not a failure.
    #   * "oversize": file exceeds parser.max_file_size_mb. We
    #     deliberately don't transcode/OCR a 60 GB Blu-ray remux;
    #     name + metadata is all the user can ever search by anyway.
    #     Final status = INDEXED_PARTIAL — by-design partial, not a
    #     failure.
    #   * "no_content": every extractor returned empty content (truly
    #     blank PDF, password-only DOCX with no body text, etc.).
    #     The parser ran cleanly — there just wasn't anything to
    #     extract. Final status = INDEXED_PARTIAL — by-design partial,
    #     not a failure.
    # In all four cases the file gets an embedding (filename +
    # available metadata) so semantic search can still surface it.
    metadata_only: bool = False
    # Human-readable explanation. Surfaced as processing_error on the
    # row so the user can see *why* a file is partial.
    metadata_only_reason: Optional[str] = None
    # Distinguishes which final status the pipeline should write. See
    # the docstring on `metadata_only` for what each value means.
    # Values: "parser_failed" | "user_elected" | "oversize" | "no_content"
    #       | "filename_only" | None.
    # "filename_only" is Tier C — the file received only an FTS entry on
    # its filename; no parse, no summarize, no semantic embed. Final
    # status = INDEXED_NAME_ONLY.
    partial_kind: Optional[str] = None
    
    @classmethod
    def from_path(cls, path: Path) -> Self:
        path = path.resolve()
        file_stats = path.stat()
        
        modified_at = datetime.fromtimestamp(file_stats.st_mtime)
        created_at = datetime.fromtimestamp(file_stats.st_ctime)
        accessed_at = datetime.fromtimestamp(file_stats.st_atime)
        
        return cls(
            path=path,
            file_path=str(path),
            filename=path.name,
            extension=path.suffix,
            file_size=file_stats.st_size,
            created=created_at,
            modified=modified_at,
            accessed=accessed_at,
        )
    
    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Self:
        """
        Create a File instance from a database row.
        
        Args:
            row: A database row (dict-like object with column names as keys)
        
        Returns:
            A File instance populated with data from the row
        """
        # Helper function to safely get a value from a Row object
        def get_value(key: str) -> Optional[Any]:
            try:
                return row[key]
            except (KeyError, IndexError):
                return None
        
        # Helper function to parse unix timestamps from database
        def parse_timestamp(value) -> Optional[datetime]:
            if not value:
                return None
            
            # If already a datetime object, return it
            if isinstance(value, datetime):
                return value
            
            # Parse unix timestamp
            try:
                return datetime.fromtimestamp(value)
            except (ValueError, AttributeError):
                logger.warning(f"Failed to parse timestamp: {value}")
                return None
        
        # Parse status from string to enum (with fallback for corrupted data)
        try:
            status = ProcessingStatus[row["status"]] if row["status"] else ProcessingStatus.DISCOVERED
        except KeyError:
            logger.warning("Invalid processing status in database, defaulting to DISCOVERED",
                          status=row["status"], file_path=row.get("file_path"))
            status = ProcessingStatus.DISCOVERED
        
        # Parse timestamps (they're stored as UNIX timestamps in the database)
        created = parse_timestamp(row["created"])
        modified = parse_timestamp(row["modified"])
        accessed = parse_timestamp(row["accessed"])
        parsed_at = parse_timestamp(get_value("parsed_at"))
        summarized_at = parse_timestamp(get_value("summarized_at"))
        embedded_at = parse_timestamp(get_value("embedded_at"))
        updated_at = parse_timestamp(get_value("updated_at"))
        
        # Parse keywords if present (stored as comma or || separated string)
        keywords = None
        keywords_value = get_value("keywords") or get_value("keywords_str")
        if keywords_value:
            # Handle both comma and || separators
            keywords = [k.strip() for k in keywords_value.replace("||", ",").split(",") if k.strip()]
        
        return cls(
            id=get_value("id"),
            path=Path(row["file_path"]),
            file_path=row["file_path"],
            filename=row["filename"],
            extension=row["extension"],
            file_size=row["file_size"],
            created=created,
            modified=modified,
            accessed=accessed,
            content_type=get_value("content_type"),
            content_hash=get_value("content_hash"),
            parsed_at=parsed_at,
            summary=get_value("summary"),
            title=get_value("title"),
            keywords=keywords,
            summarized_at=summarized_at,
            embedded_at=embedded_at,
            status=status,
            processing_error=get_value("processing_error"),
            updated_at=updated_at,
        )
    
    def to_response(self) -> "FileResponse":
        """
        Convert this File instance to a FileResponse for API serialization.
        
        Returns:
            A FileResponse instance with the relevant fields from this File
        """
        from cosma_backend.api.models import FileResponse
        
        return FileResponse(
            file_path=self.file_path,
            filename=self.filename,
            extension=self.extension,
            created=self.created,
            modified=self.modified,
            accessed=self.accessed,
            title=self.title,
            summary=self.summary,
            keywords=self.keywords,
        )
