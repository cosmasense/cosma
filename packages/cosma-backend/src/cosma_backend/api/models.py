from dataclasses import dataclass
from datetime import datetime


@dataclass
class FileResponse:
    """Shared API response model for file metadata across endpoints"""
    file_path: str
    filename: str
    extension: str
    created: datetime
    modified: datetime
    accessed: datetime
    title: str | None
    summary: str | None


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
