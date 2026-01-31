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
    ready: int
    processing: int


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


@dataclass
class MetricsResponse:
    metrics: dict[str, Any]


# ------------------------------------------------------------------
# Queue status / control
# ------------------------------------------------------------------

@queue_bp.get("/status")
@validate_response(QueueStatusResponse, 200)
async def queue_status() -> tuple[QueueStatusResponse, int]:
    status = await current_app.indexing_queue.get_status()
    return QueueStatusResponse(**status), 200


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
    ), 200


@queue_bp.put("/scheduler")
@validate_response(SchedulerResponse, 200)
async def scheduler_update() -> tuple[SchedulerResponse, int]:
    data = await request.get_json()
    scheduler = current_app.scheduler
    scheduler.update_config(data)
    # Persist to settings
    from dataclasses import asdict
    cfg = scheduler.config
    current_app.settings_manager.set_by_path("scheduler.enabled", cfg.enabled)
    current_app.settings_manager.set_by_path("scheduler.combine_mode", cfg.combine_mode)
    current_app.settings_manager.set_by_path("scheduler.check_interval_seconds", cfg.check_interval_seconds)
    rules = [asdict(r) for r in cfg.rules]
    return SchedulerResponse(
        enabled=cfg.enabled,
        combine_mode=cfg.combine_mode,
        check_interval_seconds=cfg.check_interval_seconds,
        rules=rules,
        conditions_met=scheduler.conditions_met,
    ), 200


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

@queue_bp.get("/metrics")
@validate_response(MetricsResponse, 200)
async def system_metrics() -> tuple[MetricsResponse, int]:
    from cosma_backend.queue.metrics import SystemMetricsCollector
    collector = SystemMetricsCollector()
    metrics = await collector.collect()
    return MetricsResponse(metrics=metrics), 200
