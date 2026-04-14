"""
API Response Models

Dataclass definitions for API responses. These are used by quart-schema
for automatic serialization and OpenAPI documentation generation.

Domain models (File, WatchedDirectory) have .to_response() methods
that convert them to these API models.
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class FileResponse:
    """API response model for file metadata"""
    file_path: str
    filename: str
    extension: str
    created: datetime
    modified: datetime
    accessed: datetime
    title: str | None
    summary: str | None
    keywords: list[str] | None = None


@dataclass
class JobResponse:
    """API response model for watched directory jobs"""
    id: int
    path: str
    is_active: bool
    recursive: bool
    file_pattern: str | None
    last_scan: datetime | None
    created_at: datetime | None
    updated_at: datetime | None
    file_count: int = 0        # total indexed files in DB for this directory
    total_files: int = 0       # total files on disk (from last discovery)
