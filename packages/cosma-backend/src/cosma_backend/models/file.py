from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sqlite3
from typing import Any, Optional, List, Self, TYPE_CHECKING
import logging

import numpy as np

from cosma_backend.models.status import ProcessingStatus

if TYPE_CHECKING:
    from cosma_backend.api.models import FileResponse

logger = logging.getLogger(__name__)


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
        
        # Parse status from string to enum
        status = ProcessingStatus[row["status"]] if row["status"] else ProcessingStatus.DISCOVERED
        
        # Parse timestamps (they're stored as UNIX timestamps in the database)
        created = parse_timestamp(row["created"])
        modified = parse_timestamp(row["modified"])
        accessed = parse_timestamp(row["accessed"])
        parsed_at = parse_timestamp(get_value("parsed_at"))
        summarized_at = parse_timestamp(get_value("summarized_at"))
        embedded_at = parse_timestamp(get_value("embedded_at"))
        
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
        )
