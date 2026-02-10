"""
Queue API Blueprint

Endpoints for queue status, pause/resume, item management, and scheduler config.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from quart import Blueprint, current_app, request
from quart_schema import validate_response

if TYPE_CHECKING:
    from cosma_backend.app import app as current_app

queue_bp = Blueprint("queue", __name__)


# ------------------------------------------------------------------
# Response models
# ------------------------------------------------------------------

@dataclass
class QueueStatusResponse:
    paused: bool
    manually_paused: bool
    scheduler_paused: bool
    total_items: int
    cooling_down: int
    waiting: int
    processing: int
    failing_rules: list[str]


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
        for r in getattr(current_app.scheduler, "_last_rule_results", []):
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
    limit = request.args.get("limit", 50, type=int)

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
    scheduler = current_app.scheduler
    from cosma_backend.queue.metrics import SystemMetricsCollector
    collector = SystemMetricsCollector()
    metrics = await collector.collect()

    # Evaluate rules against fresh metrics
    async with scheduler._config_lock:
        met = scheduler._evaluate_rules(metrics)
        rule_results = scheduler._last_rule_results.copy()

    return {
        "conditions_met": met,
        "rule_results": rule_results,
        "metrics": {k: v for k, v in metrics.items() if k != "collected_at"},
    }


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
    """Convert a File model to a dict suitable for FileListResponse."""
    return {
        "file_path": f.file_path,
        "filename": f.filename,
        "extension": f.extension,
        "processing_error": f.processing_error,
        "status": f.status.name if hasattr(f.status, "name") else str(f.status),
        "updated_at": int(f.modified.timestamp()) if f.modified else None,
    }


@queue_bp.get("/failed")
@validate_response(FileListResponse, 200)
async def queue_failed_files() -> tuple[FileListResponse, int]:
    offset = request.args.get("offset", 0, type=int)
    limit = request.args.get("limit", 50, type=int)

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
    limit = request.args.get("limit", 50, type=int)

    files, total_count = await current_app.db.get_files_by_status("COMPLETE", limit=limit, offset=offset)

    return FileListResponse(
        files=[_file_to_list_item(f) for f in files],
        total_count=total_count,
        offset=offset,
        limit=limit,
    ), 200


@queue_bp.post("/reindex")
@validate_response(ReindexResponse, 200)
async def queue_reindex_file() -> tuple[ReindexResponse, int]:
    from cosma_backend.queue import QueueAction

    data = await request.get_json()
    file_path = data.get("file_path") if data else None

    if not file_path:
        return ReindexResponse(success=False, message="file_path is required"), 400

    # Delete the existing file record so it gets fully reprocessed
    await current_app.db.delete_file(file_path)

    # Enqueue for re-indexing
    await current_app.indexing_queue.enqueue(file_path, QueueAction.INDEX)

    return ReindexResponse(success=True, message=f"File enqueued for reindexing: {file_path}"), 200
