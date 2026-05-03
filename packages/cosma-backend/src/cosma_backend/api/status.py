"""
Status API Blueprint

Handles endpoints related to app status.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from quart import Blueprint, abort, current_app, make_response, request, stream_with_context
from quart_schema import validate_request, validate_response

from cosma_backend.utils.pubsub import subscribe

if TYPE_CHECKING:
    from cosma_backend.app import App
    current_app: App

status_bp = Blueprint('status', __name__)


@status_bp.get("/")  # type: ignore[return-value]
async def status():
    """Get current application status and active jobs count"""
    embedder_ready = False
    searcher = getattr(current_app, "searcher", None)
    if searcher is not None:
        embedder = getattr(searcher, "embedder", None)
        if embedder is not None:
            embedder_ready = embedder.is_model_loaded()
    init_progress = getattr(current_app, "_deferred_init_progress", 0.0)
    return {
        "jobs": len(current_app.jobs),
        "embedder_ready": embedder_ready,
        "init_progress": round(init_progress, 2),
    }


@status_bp.get("/version")  # type: ignore[return-value]
async def version():
    """Backend version handshake — the frontend pins against api_version
    on launch and refuses to operate if the running backend exposes a
    higher number than the bundled Swift constant. This is what stops a
    stale frontend from talking to a backend that an auto-upgrade pulled
    forward past it. See cosma_backend/__init__.py for bump procedure.
    """
    from cosma_backend import (
        __version__,
        __api_version__,
        __min_frontend_api_version__,
    )
    return {
        "backend_version": __version__,
        "api_version": __api_version__,
        "min_frontend_api_version": __min_frontend_api_version__,
    }
