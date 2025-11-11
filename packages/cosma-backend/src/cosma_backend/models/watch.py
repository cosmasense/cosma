from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sqlite3
from typing import Any, Optional, Self
import logging

logger = logging.getLogger(__name__)


@dataclass
class WatchedDirectory:
    """
    A model representing a directory being watched for file changes.
    """
    # Core fields
    path: Path
    is_active: bool = True
    recursive: bool = True
    file_pattern: Optional[str] = None
    
    # Database fields
    id: Optional[int] = None
    last_scan: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    @classmethod
    def from_path(cls, path: Path, recursive: bool = True, file_pattern: Optional[str] = None) -> Self:
        """
        Create a WatchedDirectory instance from a path.
        
        Args:
            path: Path to the directory to watch
            recursive: Whether to watch subdirectories recursively
            file_pattern: Optional glob pattern for filtering files (e.g., "*.pdf")
        
        Returns:
            A WatchedDirectory instance
        """
        path = path.resolve()
        
        if not path.exists():
            raise ValueError(f"Path does not exist: {path}")
        
        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")
        
        return cls(
            path=path,
            recursive=recursive,
            file_pattern=file_pattern,
        )
    
    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Self:
        """
        Create a WatchedDirectory instance from a database row.
        
        Args:
            row: A database row (dict-like object with column names as keys)
        
        Returns:
            A WatchedDirectory instance populated with data from the row
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
        
        # Parse timestamps
        last_scan = parse_timestamp(get_value("last_scan"))
        created_at = parse_timestamp(get_value("created_at"))
        updated_at = parse_timestamp(get_value("updated_at"))
        
        return cls(
            id=get_value("id"),
            path=Path(row["path"]),
            is_active=bool(row["is_active"]),
            recursive=bool(row["recursive"]),
            file_pattern=get_value("file_pattern"),
            last_scan=last_scan,
            created_at=created_at,
            updated_at=updated_at,
        )
    
    @property
    def path_str(self) -> str:
        """Return the path as a string."""
        return str(self.path)
    
    def to_response(self) -> "JobResponse":
        """
        Convert this WatchedDirectory instance to a JobResponse for API serialization.
        
        Returns:
            A JobResponse instance with the relevant fields from this WatchedDirectory
        """
        from cosma_backend.api.models import JobResponse
        
        return JobResponse(
            id=self.id,
            path=self.path_str,
            is_active=self.is_active,
            recursive=self.recursive,
            file_pattern=self.file_pattern,
            last_scan=self.last_scan,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )
    
    def __str__(self) -> str:
        """Return a string representation of the watched directory."""
        return f"WatchedDirectory(id={self.id}, path={self.path}, active={self.is_active})"
