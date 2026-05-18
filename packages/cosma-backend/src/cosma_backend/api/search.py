"""
Search API Blueprint

Handles endpoints related to searching indexed files.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from quart import Blueprint, current_app, request
from quart_schema import validate_request, validate_response

from cosma_backend.api.models import FileResponse

if TYPE_CHECKING:
    from cosma_backend.app import App
    current_app: App

search_bp = Blueprint('search', __name__)


@dataclass
class TypingNudgeResponse:
    paused_until: float
    cancelled_in_flight: int


@search_bp.post("/typing")  # type: ignore[return-value]
@validate_response(TypingNudgeResponse, 200)
async def typing_nudge() -> tuple[TypingNudgeResponse, int]:
    """Frontend signals "user is interacting with the search bar".

    Debounce on the frontend (~200 ms) and POST here on every change.
    The backend (a) extends the search-preempt window so dispatch
    pauses, and (b) cancels in-flight indexing tasks so the embedder
    + LLM get the hardware *now* instead of waiting for the current
    summarize call to finish.

    Cancelled work isn't lost — the persisted-progress refactor
    saves PARSED/SUMMARIZED durably between stages, so the file
    resumes at its last completed stage on the next dispatch pass.

    The endpoint is idempotent and cheap. Spam it as often as the
    user types.
    """
    iq = getattr(current_app, "indexing_queue", None)
    cancelled = 0
    if iq is not None:
        if hasattr(iq, "search_preempt"):
            iq.search_preempt()
        if hasattr(iq, "cancel_in_flight"):
            cancelled = iq.cancel_in_flight()
    paused_until = float(getattr(iq, "_search_preempt_until", 0.0))
    return TypingNudgeResponse(
        paused_until=paused_until,
        cancelled_in_flight=cancelled,
    ), 200


@dataclass
class SearchRequest:
    """Request body for searching files"""
    query: str
    filters: dict[str, str] | None = None
    limit: int = 50
    directory: str | None = None
    offset: int = 0


@dataclass
class SearchResultItem:
    """A single search result"""
    file: FileResponse
    relevance_score: float


@dataclass
class ApplicationResultItem:
    """A single application search hit, returned alongside file
    results so the frontend can render apps as a distinct section
    (the divider between the two lists that lets the user see
    'here's the app, here are the docs' at a glance)."""
    id: int
    app_path: str
    display_name: str
    bundle_id: str | None = None
    short_version: str | None = None
    category: str | None = None
    description: str | None = None
    use_cases: str | None = None
    icon_path: str | None = None
    relevance_score: float = 0.0


@dataclass
class SearchResponse:
    """Response for search queries.

    `apps` is a parallel list, not interleaved into `results`. It
    stays empty until the apps table has rows (Step 4 populates it).
    Clients that don't know about apps continue to read `results`
    and `total_count` exactly as before.
    """
    results: list[SearchResultItem]
    total_count: int = 0
    apps: list[ApplicationResultItem] | None = None


@search_bp.post("/")  # type: ignore[return-value]
@validate_request(SearchRequest)
@validate_response(SearchResponse, 200)
async def search(data: SearchRequest) -> tuple[SearchResponse, int]:
    """Search for files based on query"""
    # Tell the indexing queue a user search is happening so it pauses
    # dispatch and lets the embedder / LLM serve the query immediately.
    # See QueueConfig.search_preempt_seconds.
    iq = getattr(current_app, "indexing_queue", None)
    if iq is not None and hasattr(iq, "search_preempt"):
        iq.search_preempt()
        # Submitting a query is a hard signal — also cancel anything
        # currently consuming hardware so the embedder gets the GPU/
        # CPU immediately. The cancelled tasks resume from their last
        # persisted stage on the next dispatch pass.
        if hasattr(iq, "cancel_in_flight"):
            iq.cancel_in_flight()
    bundle = await current_app.searcher.search_with_apps(
        data.query, directory=data.directory,
    )
    results = bundle.files
    total_count = len(results)
    paginated = results[data.offset:data.offset + data.limit]

    apps_items = [
        ApplicationResultItem(
            id=a.application.id or 0,
            app_path=a.application.app_path,
            display_name=a.application.display_name,
            bundle_id=a.application.bundle_id,
            short_version=a.application.short_version,
            category=a.application.category,
            description=a.application.description,
            use_cases=a.application.use_cases,
            icon_path=a.application.icon_path,
            relevance_score=a.relevance_score,
        )
        for a in bundle.apps
    ]

    return SearchResponse(
        results=[
            SearchResultItem(
                r.file_metadata.to_response(), r.combined_score
            )
            for r in paginated],
        total_count=total_count,
        apps=apps_items or None,
    ), 200


@dataclass
class KeywordSearchRequest:
    """Request body for keyword-based search"""
    keywords: list[str]
    match_all: bool = False


# @search_bp.post("/keywords")  # type: ignore[return-value]
# @validate_request(KeywordSearchRequest)
# @validate_response(SearchResponse, 200)
# async def search_by_keywords(data: KeywordSearchRequest) -> tuple[SearchResponse, int]:
#     """
#     Search for files by keywords.
#     
#     POST /api/search/keywords
#     
#     Request body:
#         {
#             "keywords": ["python", "api", "database"],
#             "match_all": false
#         }
#     
#     Returns:
#         200: Search results matching keywords
#     """
#     # TODO: Implement keyword search
#     # 1. Query files with matching keywords
#     # 2. If match_all=true, require all keywords
#     # 3. If match_all=false, match any keyword
#     # 4. Rank by number of matching keywords
#     
#     return SearchResponse(
#         results=[],
#         total=0,
#         query=f"Keywords: {', '.join(data.keywords)}"
#     ), 200


@dataclass
class SimilarFilesResponse:
    """Response for similar files query"""
    files: list[SearchResultItem]
    total: int


# @search_bp.get("/<int:file_id>/similar")  # type: ignore[return-value]
# @validate_response(SimilarFilesResponse, 200)
# async def find_similar_files(file_id: int) -> tuple[SimilarFilesResponse, int]:
#     """
#     Find files similar to a given file.
#     
#     GET /api/search/{file_id}/similar
#     
#     Query parameters:
#         limit: Maximum number of results (default: 10)
#     
#     Returns:
#         200: Similar files
#         404: Source file not found
#     """
#     # TODO: Implement similarity search
#     # 1. Get the source file
#     # 2. Compare keywords/summaries with other files
#     # 3. Rank by similarity
#     # 4. Return top N results
#     
#     return SimilarFilesResponse(
#         files=[],
#         total=0
#     ), 200


@dataclass
class AutocompleteResponse:
    """Response for autocomplete suggestions"""
    suggestions: list[str]


# @search_bp.get("/autocomplete")  # type: ignore[return-value]
# @validate_response(AutocompleteResponse, 200)
# async def autocomplete() -> tuple[AutocompleteResponse, int]:
#     """
#     Get autocomplete suggestions for search queries.
#     
#     GET /api/search/autocomplete?q=py
#     
#     Query parameters:
#         q: Partial query string
#         limit: Maximum suggestions (default: 10)
#     
#     Returns:
#         200: List of suggestions
#     """
#     # TODO: Implement autocomplete
#     # 1. Get partial query from request args
#     # 2. Search for matching filenames, keywords, or common terms
#     # 3. Return suggestions
#     
#     query = request.args.get('q', '')
#     
#     return AutocompleteResponse(
#         suggestions=[]
#     ), 200