"""
Index API Blueprint

Handles endpoints related to indexing directories and files.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from quart import Blueprint, current_app
from quart_schema import validate_request, validate_response

from backend.models import File
from backend.pipeline import Pipeline

if TYPE_CHECKING:
    from backend.app import app as current_app

index_bp = Blueprint('index', __name__)


@dataclass
class IndexDirectoryRequest:
    """Request body for indexing a directory"""
    directory_path: str


@dataclass
class IndexDirectoryResponse:
    """Response for directory indexing"""
    success: bool
    message: str
    files_indexed: int


@index_bp.post("/directory")  # type: ignore[return-value]
@validate_request(IndexDirectoryRequest)
@validate_response(IndexDirectoryResponse, 201)
async def index_directory(data: IndexDirectoryRequest) -> tuple[IndexDirectoryResponse, int]:
    """Index all files in a directory"""
    current_app.submit_job(current_app.pipeline.process_directory(data.directory_path))
    
    return IndexDirectoryResponse(
        success=True,
        message=f"Started indexing directory: {data.directory_path}",
        files_indexed=0
    ), 201


@dataclass
class IndexFileRequest:
    """Request body for indexing a single file"""
    file_path: str


@dataclass
class IndexFileResponse:
    """Response for file indexing"""
    success: bool
    message: str
    file_id: int | None


@index_bp.post("/file")  # type: ignore[return-value]
@validate_request(IndexFileRequest)
@validate_response(IndexFileResponse, 201)
async def index_file(data: IndexFileRequest) -> tuple[IndexFileResponse, int]:
    """Index a single file"""
    # TODO: Implement single file indexing
    # 1. Validate file exists
    # 2. Parse file using backend.parser
    # 3. Summarize file using backend.summarizer
    # 4. Insert into database using current_app.db
    await current_app.pipeline.process_file(File.from_path(Path(data.file_path)))
    
    return IndexFileResponse(
        success=True,
        message=f"Successfully indexed file: {data.file_path}",
        file_id=None
    ), 201


@dataclass
class IndexStatusResponse:
    """Response for indexing status"""
    is_indexing: bool
    current_file: str | None
    files_processed: int
    total_files: int


# @index_bp.get("/status")  # type: ignore[return-value]
# @validate_response(IndexStatusResponse, 200)
# async def index_status() -> tuple[IndexStatusResponse, int]:
#     """
#     Get the current status of any ongoing indexing operations.
#     
#     GET /api/index/status
#     
#     Returns:
#         200: Current indexing status
#     """
#     # TODO: Implement status tracking
#     # This could use a global state manager or database table
#     # to track ongoing indexing operations
#     
#     return IndexStatusResponse(
#         is_indexing=False,
#         current_file=None,
#         files_processed=0,
#         total_files=0
#     ), 200
