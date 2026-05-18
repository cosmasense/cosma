"""Applications API Blueprint.

Endpoints:
- GET  /api/applications/         — list known apps (paginated)
- GET  /api/applications/count    — total count
- POST /api/applications/rescan/  — force a fresh scan of /Applications

Search-side fusion lives in /api/search; this blueprint is the
admin/listing surface so the future Settings UI can show "we
know about N applications, last scanned X ago" and offer a
manual rescan.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from quart import Blueprint, current_app, request
from quart_schema import validate_response

if TYPE_CHECKING:
    from cosma_backend.app import App
    current_app: App

applications_bp = Blueprint("applications", __name__)


@dataclass
class ApplicationItem:
    id: int
    app_path: str
    display_name: str
    bundle_id: str | None = None
    short_version: str | None = None
    category: str | None = None
    description: str | None = None
    use_cases: str | None = None
    icon_path: str | None = None


@dataclass
class ApplicationsListResponse:
    applications: list[ApplicationItem]
    total: int


@dataclass
class ScanResponse:
    discovered: int
    upserted: int
    pruned: int
    elapsed_seconds: float


def _to_item(a) -> ApplicationItem:
    return ApplicationItem(
        id=a.id or 0,
        app_path=a.app_path,
        display_name=a.display_name,
        bundle_id=a.bundle_id,
        short_version=a.short_version,
        category=a.category,
        description=a.description,
        use_cases=a.use_cases,
        icon_path=a.icon_path,
    )


@applications_bp.get("/")  # type: ignore[return-value]
@validate_response(ApplicationsListResponse, 200)
async def list_applications() -> tuple[ApplicationsListResponse, int]:
    repo = getattr(current_app, "applications_repository", None)
    if repo is None:
        return ApplicationsListResponse(applications=[], total=0), 200

    try:
        limit = int(request.args.get("limit", "500"))
    except (TypeError, ValueError):
        limit = 500
    try:
        offset = int(request.args.get("offset", "0"))
    except (TypeError, ValueError):
        offset = 0
    # Cap limits so a misconfigured client can't drag a thousand-row
    # ICNS-path payload across the wire repeatedly.
    limit = max(1, min(limit, 2000))
    offset = max(0, offset)

    apps = await repo.list_all(limit=limit, offset=offset)
    total = await repo.count()
    return ApplicationsListResponse(
        applications=[_to_item(a) for a in apps],
        total=total,
    ), 200


@applications_bp.get("/count")  # type: ignore[return-value]
async def count_applications() -> tuple[dict, int]:
    repo = getattr(current_app, "applications_repository", None)
    if repo is None:
        return {"total": 0}, 200
    return {"total": await repo.count()}, 200


@applications_bp.post("/rescan/")  # type: ignore[return-value]
@validate_response(ScanResponse, 200)
async def rescan_applications() -> tuple[ScanResponse, int]:
    """Manually trigger an apps rescan.

    The background loop already runs every 6 h; this endpoint is
    for the user-driven "I just installed something and want it
    indexed now" case (and for the test harness).
    """
    indexer = getattr(current_app, "applications_indexer", None)
    if indexer is None:
        return ScanResponse(0, 0, 0, 0.0), 200
    result = await indexer.index_now()
    return ScanResponse(
        discovered=result.discovered,
        upserted=result.upserted,
        pruned=result.pruned,
        elapsed_seconds=result.elapsed_seconds,
    ), 200
