"""
Queue API Blueprint

Endpoints for queue status, pause/resume, item management, and scheduler config.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from quart import Blueprint, current_app, request
from quart_schema import validate_response

from cosma_backend.logging import get_logger

if TYPE_CHECKING:
    from cosma_backend.app import App
    current_app: App

logger = get_logger(__name__)

queue_bp = Blueprint("queue", __name__)


# ------------------------------------------------------------------
# Response models
# ------------------------------------------------------------------

@dataclass
class QueueStatusResponse:
    paused: bool
    manually_paused: bool
    scheduler_paused: bool
    bootstrap_paused: bool
    total_items: int
    cooling_down: int
    waiting: int
    processing: int
    failing_rules: list[str]
    # One-shot user override on top of the scheduler decision.
    #   None  = no override, scheduler is in control
    #   True  = user forced "run now" (overrides a scheduler pause)
    #   False = user forced "pause now" (overrides a scheduler run)
    # Auto-cleared on next scheduler transition.
    user_override: Optional[bool] = None
    # True for ~10s after every /api/search/ hit, so the queue UI can
    # render a "paused for search" badge instead of looking stuck.
    # See QueueConfig.search_preempt_seconds.
    search_preempted: bool = False


@dataclass
class QueueActionResponse:
    success: bool
    message: str


@dataclass
class QueueItemsResponse:
    items: list[dict[str, Any]]
    total_count: int
    offset: int
    limit: int


@dataclass
class SchedulerResponse:
    enabled: bool
    combine_mode: str
    check_interval_seconds: int
    rules: list[dict[str, Any]]
    conditions_met: bool
    warnings: list[str]
    rule_results: list[dict[str, Any]]


@dataclass
class FileListItem:
    file_path: str
    filename: str
    extension: str
    processing_error: str | None
    status: str
    updated_at: int | None


@dataclass
class FileListResponse:
    files: list[dict[str, Any]]
    total_count: int
    offset: int
    limit: int


@dataclass
class ReindexResponse:
    success: bool
    message: str


@dataclass
class MetricsResponse:
    metrics: dict[str, Any]
    models: list[dict[str, Any]]


# ------------------------------------------------------------------
# Queue status / control
# ------------------------------------------------------------------

@queue_bp.get("/status")
@validate_response(QueueStatusResponse, 200)
async def queue_status() -> tuple[QueueStatusResponse, int]:
    status = await current_app.indexing_queue.get_status()

    failing_rules: list[str] = []
    if status.get("scheduler_paused") and hasattr(current_app, "scheduler"):
        for r in current_app.scheduler.last_rule_results:
            if not r.get("passed", True):
                failing_rules.append(r.get("rule", "unknown"))

    return QueueStatusResponse(**status, failing_rules=failing_rules), 200


@queue_bp.post("/pause")
@validate_response(QueueActionResponse, 200)
async def queue_pause() -> tuple[QueueActionResponse, int]:
    current_app.indexing_queue.manual_pause()
    return QueueActionResponse(success=True, message="Queue paused"), 200


@queue_bp.post("/resume")
@validate_response(QueueActionResponse, 200)
async def queue_resume() -> tuple[QueueActionResponse, int]:
    current_app.indexing_queue.manual_resume()
    return QueueActionResponse(success=True, message="Queue resumed"), 200


# ------------------------------------------------------------------
# Queue items
# ------------------------------------------------------------------

@queue_bp.get("/items")
@validate_response(QueueItemsResponse, 200)
async def queue_items() -> tuple[QueueItemsResponse, int]:
    offset = request.args.get("offset", 0, type=int)
    # 10000 = effectively uncapped. Listing failed/recent files is cheap
    # (indexed lookup on a single column) and the UI's "Retry All" button
    # needs to see every failure, not just the first page.
    limit = request.args.get("limit", 10000, type=int)

    all_items = await current_app.indexing_queue.get_items()
    total_count = len(all_items)
    paginated = all_items[offset:offset + limit]

    return QueueItemsResponse(
        items=[i.to_dict() for i in paginated],
        total_count=total_count,
        offset=offset,
        limit=limit,
    ), 200


@queue_bp.delete("/items/<item_id>")
@validate_response(QueueActionResponse, 404)
@validate_response(QueueActionResponse, 200)
async def queue_remove_item(item_id: str) -> tuple[QueueActionResponse, int]:
    removed = await current_app.indexing_queue.remove_item(item_id)
    if removed:
        return QueueActionResponse(success=True, message="Item removed"), 200
    return QueueActionResponse(success=False, message="Item not found"), 404


# ------------------------------------------------------------------
# Scheduler
# ------------------------------------------------------------------

@queue_bp.get("/scheduler")
@validate_response(SchedulerResponse, 200)
async def scheduler_status() -> tuple[SchedulerResponse, int]:
    scheduler = current_app.scheduler
    cfg = scheduler.config
    from dataclasses import asdict
    rules = [asdict(r) for r in cfg.rules]
    return SchedulerResponse(
        enabled=cfg.enabled,
        combine_mode=cfg.combine_mode,
        check_interval_seconds=cfg.check_interval_seconds,
        rules=rules,
        conditions_met=scheduler.conditions_met,
        warnings=scheduler.warnings,
        rule_results=scheduler.last_rule_results,
    ), 200


@queue_bp.put("/scheduler")
@validate_response(SchedulerResponse, 200)
async def scheduler_update() -> tuple[SchedulerResponse, int]:
    data = await request.get_json()
    scheduler = current_app.scheduler
    await scheduler.update_config(data)
    # Persist to settings (including rules)
    from dataclasses import asdict
    cfg = scheduler.config
    rules = [asdict(r) for r in cfg.rules]
    current_app.settings_manager.update({
        "scheduler.enabled": cfg.enabled,
        "scheduler.combine_mode": cfg.combine_mode,
        "scheduler.check_interval_seconds": cfg.check_interval_seconds,
        "scheduler.rules": rules,
    })
    return SchedulerResponse(
        enabled=cfg.enabled,
        combine_mode=cfg.combine_mode,
        check_interval_seconds=cfg.check_interval_seconds,
        rules=rules,
        conditions_met=scheduler.conditions_met,
        warnings=scheduler.warnings,
        rule_results=scheduler.last_rule_results,
    ), 200


@queue_bp.post("/scheduler/test")
async def scheduler_test() -> dict:
    """Evaluate current scheduler rules against live metrics (dry-run)."""
    return await current_app.scheduler.test_rules()


@queue_bp.get("/scheduler/rule-types")
async def scheduler_rule_types() -> dict:
    """Return metadata for each scheduler rule type so clients can build type-specific UIs."""
    from cosma_backend.settings import SCHEDULER_RULE_TYPES
    return {"rule_types": SCHEDULER_RULE_TYPES}


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

@queue_bp.get("/metrics")
@validate_response(MetricsResponse, 200)
async def system_metrics() -> tuple[MetricsResponse, int]:
    from cosma_backend.queue.metrics import SystemMetricsCollector
    collector = SystemMetricsCollector()
    metrics = await collector.collect()

    models: list[dict[str, Any]] = []
    if hasattr(current_app, "model_lifecycle"):
        models = current_app.model_lifecycle.get_model_status()

    return MetricsResponse(metrics=metrics, models=models), 200


# ------------------------------------------------------------------
# Failed / Recent / Reindex
# ------------------------------------------------------------------

def _file_to_list_item(f) -> dict[str, Any]:
    """Convert a File model to a dict suitable for FileListResponse.

    The exposed `updated_at` is the time we actually finished processing
    the file — NOT the DB column of the same name. The DB's `updated_at`
    column gets bumped on every disk-scan sweep (mark-and-sweep stale
    detection in pipeline.enqueue_directory), which means it resets to
    "now" every time the backend launches. Surfacing that to the
    frontend made the Recent tab read "Xs ago" for every file right
    after each launch.

    Pick the most specific completion timestamp available:
      embedded_at > summarized_at > parsed_at > modified
    For COMPLETE rows all three pipeline stamps are set, so we get the
    real "completed at" time. For FAILED rows we fall back to whichever
    stage actually ran. Filesystem `modified` is the last fallback for
    pre-pipeline metadata.
    """
    ts = None
    for attr in ("embedded_at", "summarized_at", "parsed_at"):
        value = getattr(f, attr, None)
        if value is not None:
            ts = int(value.timestamp())
            break
    if ts is None and getattr(f, "modified", None) is not None:
        ts = int(f.modified.timestamp())
    return {
        "file_path": f.file_path,
        "filename": f.filename,
        "extension": f.extension,
        "processing_error": f.processing_error,
        "status": f.status.name if hasattr(f.status, "name") else str(f.status),
        "updated_at": ts,
    }


@queue_bp.get("/failed")
@validate_response(FileListResponse, 200)
async def queue_failed_files() -> tuple[FileListResponse, int]:
    offset = request.args.get("offset", 0, type=int)
    # 10000 = effectively uncapped. Listing failed/recent files is cheap
    # (indexed lookup on a single column) and the UI's "Retry All" button
    # needs to see every failure, not just the first page.
    limit = request.args.get("limit", 10000, type=int)

    files, total_count = await current_app.db.get_files_by_status("FAILED", limit=limit, offset=offset)

    return FileListResponse(
        files=[_file_to_list_item(f) for f in files],
        total_count=total_count,
        offset=offset,
        limit=limit,
    ), 200


@queue_bp.get("/recent")
@validate_response(FileListResponse, 200)
async def queue_recent_files() -> tuple[FileListResponse, int]:
    offset = request.args.get("offset", 0, type=int)
    # 10000 = effectively uncapped. Listing failed/recent files is cheap
    # (indexed lookup on a single column) and the UI's "Retry All" button
    # needs to see every failure, not just the first page.
    limit = request.args.get("limit", 10000, type=int)

    files, total_count = await current_app.db.get_files_by_status("COMPLETE", limit=limit, offset=offset)

    return FileListResponse(
        files=[_file_to_list_item(f) for f in files],
        total_count=total_count,
        offset=offset,
        limit=limit,
    ), 200


@queue_bp.get("/partial")
@validate_response(FileListResponse, 200)
async def queue_partial_files() -> tuple[FileListResponse, int]:
    """List files with status=INDEXED_PARTIAL.

    These were classified as metadata-only by the user's filter config
    and indexed by filename + metadata only (no LLM summary). The
    frontend exposes them as their own tab so the user can review what
    was opted out of full indexing.
    """
    offset = request.args.get("offset", 0, type=int)
    limit = request.args.get("limit", 10000, type=int)

    files, total_count = await current_app.db.get_files_by_status(
        "INDEXED_PARTIAL", limit=limit, offset=offset
    )

    return FileListResponse(
        files=[_file_to_list_item(f) for f in files],
        total_count=total_count,
        offset=offset,
        limit=limit,
    ), 200


@queue_bp.post("/reindex")
@validate_response(ReindexResponse, 400)
@validate_response(ReindexResponse, 200)
async def queue_reindex_file() -> tuple[ReindexResponse, int]:
    from cosma_backend.queue import QueueAction

    data = await request.get_json()
    file_path = data.get("file_path") if data else None

    if not file_path:
        return ReindexResponse(success=False, message="file_path is required"), 400

    # Never let a hiccup here become a 500 — the frontend's per-file
    # retry buttons (and the Retry-All fan-out) surface a 500 as a scary
    # "internal server error" toast. Return success=False instead; the UI
    # already renders that as a retryable "Reindex failed: …".
    try:
        # Delete the existing file record so it gets fully reprocessed.
        # delete_file is best-effort and swallows transient SQLite errors.
        await current_app.db.delete_file(file_path)
        # Enqueue for re-indexing
        await current_app.indexing_queue.enqueue(file_path, QueueAction.INDEX)
    except Exception as e:
        logger.warning("reindex enqueue failed", file_path=file_path, error=str(e))
        return ReindexResponse(success=False, message=f"Could not re-queue {file_path}: {e}"), 200

    return ReindexResponse(success=True, message=f"File enqueued for reindexing: {file_path}"), 200


# ------------------------------------------------------------------
# Retry all FAILED files in one call.
#
# The frontend's "Retry All Failed" button used to fan out one
# POST /api/queue/reindex per failed file — dozens of concurrent
# delete+enqueue round-trips, where a single transient DB error in any
# one of them bubbled up as an Internal Server Error and left the rest
# unretried. This single bulk call does the loop server-side, skipping
# (and logging) any individual file that errors, and returns the count
# that actually got re-queued. Also doubles as the COSMA_DEV
# fix→retry→observe loop endpoint: when COSMA_DEV=1 it additionally
# truncates the failed-items log (no-op otherwise).
# ------------------------------------------------------------------

@queue_bp.post("/retry_all_failed")
async def queue_retry_all_failed() -> tuple[dict, int]:
    from cosma_backend.dev_logging import reset_log  # no-op unless COSMA_DEV=1
    from cosma_backend.queue import QueueAction

    try:
        files, total = await current_app.db.get_files_by_status("FAILED", limit=100000, offset=0)
    except Exception as e:
        logger.warning("retry_all_failed: could not list FAILED files", error=str(e))
        return {"success": False, "message": f"Could not list failed files: {e}", "count": 0}, 200

    reset_log()
    requeued = 0
    skipped = 0
    for f in files:
        try:
            await current_app.db.delete_file(f.file_path)
            await current_app.indexing_queue.enqueue(f.file_path, QueueAction.INDEX)
            requeued += 1
        except Exception as e:
            skipped += 1
            logger.warning("retry_all_failed: skipping file", file_path=f.file_path, error=str(e))

    msg = f"Re-queued {requeued} of {total} failed files"
    if skipped:
        msg += f" ({skipped} skipped — see logs)"
    return {"success": True, "message": msg, "count": requeued, "skipped": skipped}, 200
