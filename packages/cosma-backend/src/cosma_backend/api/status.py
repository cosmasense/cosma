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


@status_bp.get("/vision")  # type: ignore[return-value]
async def vision():
    """Vision capability probe.

    Reports whether the LlamaCpp summarizer's chat handler loaded —
    i.e. whether image files can be summarized from pixels rather than
    just from the parser-metadata placeholder. The frontend uses this
    after the first launch with a new release to decide whether to
    self-heal a stock-PyPI install (no `Qwen3VLChatHandler` in the
    venv) by reinstalling cosma with the cosmasense fork wheel via
    `uv tool install --reinstall cosma --with <wheel-url>`.

    Returns:
        - `vision_available`: bool — True iff the chat handler is loaded
        - `required_handler`: str — class name we look for
        - `handler_loaded`: bool — same as `vision_available` (alias for clarity)
        - `provider`: str — the configured summarizer provider
    """
    pipeline = getattr(current_app, "pipeline", None)
    summarizer = getattr(pipeline, "summarizer", None) if pipeline else None
    provider = getattr(summarizer, "preferred_provider", None) or "unknown"

    handler_loaded = False
    if summarizer is not None and hasattr(summarizer, "_get_llamacpp_summarizer"):
        try:
            llamacpp = await summarizer._get_llamacpp_summarizer()
        except Exception:
            llamacpp = None
        if llamacpp is not None:
            # `chat_handler` is set during `_ensure_loaded`. None means
            # the handler class was missing or the constructor raised.
            handler_loaded = getattr(llamacpp, "chat_handler", None) is not None

    from cosma_backend.summarizer.providers import LlamaCppSummarizer
    return {
        "vision_available": handler_loaded,
        "handler_loaded": handler_loaded,
        "required_handler": LlamaCppSummarizer._REQUIRED_HANDLER_CLASS,
        "provider": provider,
    }
