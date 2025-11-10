"""
Watch API Blueprint

Handles endpoints related to watching directories.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from quart import Blueprint, current_app
from quart_schema import validate_request, validate_response

from cosma_backend.api.models import JobResponse

if TYPE_CHECKING:
    from cosma_backend.app import app as current_app

watch_bp = Blueprint('watch', __name__)


@dataclass
class WatchRequest:
    """Request body for watching a directory"""
    directory_path: str


@dataclass
class WatchResponse:
    """Response for directory watching"""
    success: bool
    message: str
    files_indexed: int


@watch_bp.post("/")  # type: ignore[return-value]
@validate_request(WatchRequest)
@validate_response(WatchResponse, 201)
async def watch_directory(data: WatchRequest) -> tuple[WatchResponse, int]:
    """Watch all files in a directory for changes"""
    # TODO: Implement indexing logic
    # 1. Validate directory exists
    # 2. Extract files using cosma_backend.extractor
    # 3. Parse each file using cosma_backend.parser
    # 4. Summarize each file using cosma_backend.summarizer
    # 5. Insert into database using current_app.db
    
    try:
        await current_app.watcher.start_watching(data.directory_path)
    except ValueError as e:
        # Directory is already being watched or parent directory is watching it
        return WatchResponse(
            success=False,
            message=str(e),
            files_indexed=0
        ), 400
    
    return WatchResponse(
        success=True,
        message=f"Started watching directory: {data.directory_path}",
        files_indexed=0
    ), 201


@dataclass
class WatchStatusResponse:
    """Response for watching status"""
    is_indexing: bool
    current_file: str | None
    files_processed: int
    total_files: int


@dataclass
class JobsListResponse:
    """Response for jobs list"""
    jobs: list[JobResponse]


@watch_bp.get("/jobs")  # type: ignore[return-value]
@validate_response(JobsListResponse, 200)
async def get_jobs() -> tuple[JobsListResponse, int]:
    """
    Get all watched directory jobs.
    
    GET /api/watch/jobs
    
    Returns:
        200: List of all watched directories
    """    
    # Get all watched directories from database
    watched_dirs = await current_app.db.get_watched_directories(active_only=False)
    
    # Convert to API response models
    jobs = [watched_dir.to_response() for watched_dir in watched_dirs]
    
    return JobsListResponse(jobs=jobs), 200


@dataclass
class DeleteJobResponse:
    """Response for deleting a watched directory job"""
    success: bool
    message: str
    job_id: int


@watch_bp.delete("/jobs/<int:job_id>")  # type: ignore[return-value]
@validate_response(DeleteJobResponse, 200)
async def delete_job(job_id: int) -> tuple[DeleteJobResponse, int]:
    """
    Delete a watched directory job by ID.
    
    DELETE /api/watch/jobs/{job_id}
    
    Returns:
        200: Job deleted successfully
        404: Job not found
    """
    # Delete the watched directory from database
    deleted_dir = await current_app.db.delete_watched_directory(job_id)
    
    if deleted_dir:
        return DeleteJobResponse(
            success=True,
            message=f"Successfully deleted watched directory: {deleted_dir.path_str}",
            job_id=job_id
        ), 200
    else:
        return DeleteJobResponse(
            success=False,
            message=f"Watched directory with ID {job_id} not found",
            job_id=job_id
        ), 404


# @watch_bp.get("/status")  # type: ignore[return-value]
# @validate_response(WatchStatusResponse, 200)
# async def watch_status() -> tuple[WatchStatusResponse, int]:
#     """
#     Get the current status of any ongoing watch operations.
#     
#     GET /api/index/status
#     
#     Returns:
#         200: Current watch status
#     """
#     # TODO: Implement status tracking
#     # This could use a global state manager or database table
#     # to track ongoing indexing operations
#     
#     return WatchStatusResponse(
#         is_indexing=False,
#         current_file=None,
#         files_processed=0,
#         total_files=0
#     ), 200
