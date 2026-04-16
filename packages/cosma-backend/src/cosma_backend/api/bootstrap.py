"""
Bootstrap HTTP routes — thin wrapper over cosma_backend.bootstrap.

Endpoints:
- GET  /api/bootstrap/status   → JSON component table
- POST /api/bootstrap/install  → starts (or joins) the install run
- GET  /api/bootstrap/events   → SSE stream of progress events

Why these three are separate:
- The status endpoint is cheap and polled by the wizard's "should I even
  show the setup step?" check.
- POST /install triggers the side-effectful download; splitting it from
  SSE means a client can POST once and then have multiple tabs subscribe
  to events without each triggering another run.
- SSE is a long-lived GET so it plays nicely with browsers and our
  existing SSE plumbing (see api/updates.py).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from quart import Blueprint, abort, current_app, jsonify, make_response, request, stream_with_context

from cosma_backend import bootstrap as bootstrap_core
from cosma_backend.logging import get_logger
from cosma_backend.utils.sse import ServerSentEvent, sse_comment

if TYPE_CHECKING:
    from cosma_backend.app import App
    current_app: App

logger = get_logger(__name__)
bootstrap_bp = Blueprint('bootstrap', __name__)


def _configs():
    """Pull current llama/whisper configs + chosen summarizer provider.

    The provider drives what bootstrap treats as "required" (see
    bootstrap.get_status). Pulling it here keeps the route bodies thin.
    """
    settings = current_app.settings_manager.settings
    return (
        settings.summarizer.llamacpp,
        settings.parser.whisper,
        settings.summarizer.provider,
    )


@bootstrap_bp.get("/status", strict_slashes=False)
async def status():
    """Cheap availability probe. No downloads, no model loads."""
    llama_cfg, whisper_cfg, summ_provider = _configs()
    st = await bootstrap_core.get_status(
        llama_cfg, whisper_cfg, summ_provider=summ_provider,
    )
    # Include the active providers so the frontend can render the right
    # components even on a fresh /status request (before it has fetched
    # settings separately).
    return jsonify({
        **st.to_dict(),
        "summarizer_provider": summ_provider,
        "whisper_provider": whisper_cfg.provider,
    })


@bootstrap_bp.post("/install", strict_slashes=False)
async def install():
    """Kick off (or no-op if already running) a bootstrap install.

    Only installs components required by the user's currently-selected
    providers (so picking "ollama" doesn't trigger a 2 GB GGUF download).
    Returns immediately — real progress is on /events.
    """
    if bootstrap_core.runner.is_running:
        return jsonify({"started": False, "running": True})

    llama_cfg, whisper_cfg, summ_provider = _configs()
    task = asyncio.create_task(
        bootstrap_core.runner.run(llama_cfg, whisper_cfg, summ_provider=summ_provider)
    )
    # Strong reference via current_app.jobs so asyncio doesn't GC the task
    # once the HTTP handler returns (silent-vanish bug).
    current_app.jobs.add(task)
    task.add_done_callback(current_app.jobs.discard)
    return jsonify({"started": True, "running": True, "provider": summ_provider})


@bootstrap_bp.get("/events", strict_slashes=False)
async def events():
    """SSE stream of progress events for the current (or most recent) run."""
    if "text/event-stream" not in request.accept_mimetypes:
        abort(400)

    q = bootstrap_core.runner.subscribe()

    @stream_with_context
    async def generator():
        # Initial comment forces Quart to flush headers immediately so the
        # browser sees an open stream even before the first real event.
        yield sse_comment("bootstrap-stream-open").encode()
        try:
            async for evt in bootstrap_core.stream_events(q):
                # ServerSentEvent.encode() returns str; Quart needs bytes
                # for streamed chunked responses, so wrap once.
                text = ServerSentEvent(
                    data=evt.to_dict(),
                    event="progress",
                ).encode()
                yield text.encode("utf-8")
        except asyncio.CancelledError:
            pass
        finally:
            bootstrap_core.runner.unsubscribe(q)

    response = await make_response(
        generator(),
        {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Transfer-Encoding": "chunked",
        },
    )
    response.timeout = None  # type: ignore[attr-defined]
    return response
