"""
Files API Blueprint

Handles endpoints related to file operations and retrieval.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from quart import Blueprint, current_app
from quart_schema import validate_request, validate_response

if TYPE_CHECKING:
    from cosma_backend.app import App
    current_app: App

files_bp = Blueprint('files', __name__)


@dataclass
class FileResponse:
    """Response model for a single file"""
    id: int
    filename: str
    extension: str
    created: str
    modified: str
    summary: str
    keywords: list[str] | None


@dataclass
class FilesListResponse:
    """Response model for list of files"""
    files: list[FileResponse]
    total: int
    page: int
    per_page: int


# @files_bp.get("/")  # type: ignore[return-value]
# @validate_response(FilesListResponse, 200)
# async def list_files() -> tuple[FilesListResponse, int]:
#     """
#     Get a list of all indexed files.
#     
#     GET /api/files/
#     
#     Query parameters:
#         page: Page number (default: 1)
#         per_page: Items per page (default: 50)
#         extension: Filter by file extension
#     
#     Returns:
#         200: List of files
#     """
#     # TODO: Implement file listing with pagination
#     # 1. Get query parameters
#     # 2. Query database with filters and pagination
#     # 3. Return formatted response
#     
#     return FilesListResponse(
#         files=[],
#         total=0,
#         page=1,
#         per_page=50
#     ), 200


@files_bp.get("/<int:file_id>")  # type: ignore[return-value]
@validate_response(FileResponse, 200)
async def get_file(file_id: int) -> tuple[FileResponse, int]:
    """Get details of a specific file by ID"""
    # TODO: Implement single file retrieval
    # 1. Query database for file by ID
    # 2. Return file details or 404
    
    async with current_app.db.acquire() as conn:
        file = await conn.fetchone(
            "SELECT * FROM files WHERE id = ?;",
            (file_id,)
        )
    
    if not file:
        return {"error": "File not found"}, 404  # type: ignore
    
    # TODO: Parse the file data properly
    return FileResponse(
        id=file['id'],
        filename=file['filename'],
        extension=file['extension'],
        created=str(file['created']),
        modified=str(file['modified']),
        summary=file['summary'],
        keywords=None  # TODO: Parse keywords from database
    ), 200


@dataclass
class DeleteFileResponse:
    """Response for file deletion"""
    success: bool
    message: str


# @files_bp.delete("/<int:file_id>")  # type: ignore[return-value]
# @validate_response(DeleteFileResponse, 200)
# async def delete_file(file_id: int) -> tuple[DeleteFileResponse, int]:
#     """
#     Delete a file from the index.
#     
#     DELETE /api/files/{file_id}
#     
#     Returns:
#         200: File deleted successfully
#         404: File not found
#     """
#     # TODO: Implement file deletion
#     # 1. Check if file exists
#     # 2. Delete from database
#     # 3. Return success/failure
#     
#     return DeleteFileResponse(
#         success=True,
#         message=f"File {file_id} deleted successfully"
#     ), 200


@dataclass
class FileDetailsResponse:
    """Everything we know about a single indexed file. Powers the
    frontend's "Stats for Nerds" debug panel — surfaces what the LLM
    actually wrote, whether the embedding even exists, and any failure
    reason — so users can diagnose bad search hits (especially images,
    where the vision summary is the only signal the embedder sees)
    without needing to reach for sqlite."""
    found: bool
    file_path: str
    file_id: int | None
    filename: str | None
    extension: str | None
    file_size: int | None
    created: int | None
    modified: int | None
    accessed: int | None
    content_type: str | None
    content_hash: str | None
    parsed_at: int | None
    title: str | None
    summary: str | None
    summarized_at: int | None
    embedded_at: int | None
    status: str | None
    processing_error: str | None
    owner: str | None
    permissions: str | None
    created_at: int | None
    updated_at: int | None
    keywords: list[str]
    has_embedding: bool
    embedding_model: str | None
    embedding_dimensions: int | None


@files_bp.get("/details")  # type: ignore[return-value]
async def get_file_details() -> tuple[dict, int]:  # type: ignore[type-arg]
    """Look up a single file by absolute path and return everything we
    know about it: every column in `files`, its keywords, and whether
    its embedding row exists in `file_embeddings` (with the model name
    + dimension count). Designed for the Cosma Sense "Stats for Nerds"
    debug panel — quickest way to see why a search result looks wrong
    without dropping into sqlite.

    GET /api/files/details/?path=<absolute_path>

    Response is always 200 with `found=false` if the path isn't in the
    index, so the frontend can render a uniform "not indexed" state
    without parsing HTTP statuses.
    """
    from quart import request

    raw_path = request.args.get("path", "").strip()
    if not raw_path:
        return {"error": "path query parameter required"}, 400  # type: ignore

    async with current_app.db.acquire() as conn:
        file_row = await conn.fetchone(
            "SELECT * FROM files WHERE file_path = ?;",
            (raw_path,),
        )

        if not file_row:
            empty = FileDetailsResponse(
                found=False,
                file_path=raw_path,
                file_id=None,
                filename=None,
                extension=None,
                file_size=None,
                created=None,
                modified=None,
                accessed=None,
                content_type=None,
                content_hash=None,
                parsed_at=None,
                title=None,
                summary=None,
                summarized_at=None,
                embedded_at=None,
                status=None,
                processing_error=None,
                owner=None,
                permissions=None,
                created_at=None,
                updated_at=None,
                keywords=[],
                has_embedding=False,
                embedding_model=None,
                embedding_dimensions=None,
            )
            return empty.__dict__, 200

        file_id = file_row["id"]

        keyword_rows = await conn.fetchall(
            "SELECT keyword FROM file_keywords WHERE file_id = ? ORDER BY keyword;",
            (file_id,),
        )
        keywords = [r["keyword"] for r in keyword_rows]

        # file_embeddings is a vec0 virtual table — we just need to
        # know whether the row exists and which model wrote it. The
        # actual embedding blob is huge and useless to the UI.
        embedding_row = await conn.fetchone(
            "SELECT embedding_model, embedding_dimensions "
            "FROM file_embeddings WHERE file_id = ?;",
            (file_id,),
        )

    details = FileDetailsResponse(
        found=True,
        file_path=file_row["file_path"],
        file_id=file_id,
        filename=file_row["filename"],
        extension=file_row["extension"],
        file_size=file_row["file_size"],
        created=file_row["created"],
        modified=file_row["modified"],
        accessed=file_row["accessed"],
        content_type=file_row["content_type"],
        content_hash=file_row["content_hash"],
        parsed_at=file_row["parsed_at"],
        title=file_row["title"],
        summary=file_row["summary"],
        summarized_at=file_row["summarized_at"],
        embedded_at=file_row["embedded_at"],
        status=file_row["status"],
        processing_error=file_row["processing_error"],
        owner=file_row["owner"],
        permissions=file_row["permissions"],
        created_at=file_row["created_at"],
        updated_at=file_row["updated_at"],
        keywords=keywords,
        has_embedding=embedding_row is not None,
        embedding_model=embedding_row["embedding_model"] if embedding_row else None,
        embedding_dimensions=embedding_row["embedding_dimensions"] if embedding_row else None,
    )
    return details.__dict__, 200


@dataclass
class FileStatsResponse:
    """Response for file statistics"""
    total_files: int
    total_size: int
    file_types: dict[str, int]
    last_indexed: str | None


@files_bp.get("/stats")  # type: ignore[return-value]
@validate_response(FileStatsResponse, 200)
async def get_stats() -> tuple[FileStatsResponse, int]:
    """Get statistics about indexed files"""
    # TODO: Implement statistics gathering
    # 1. Count total files
    # 2. Group by extension
    # 3. Get most recent index timestamp
    
    async with current_app.db.acquire() as conn:
        total = await conn.fetchone("SELECT COUNT(*) as count FROM files;")
    
    return FileStatsResponse(
        total_files=total['count'] if total else 0,
        total_size=0,  # TODO: Add size tracking
        file_types={},
        last_indexed=None
    ), 200
