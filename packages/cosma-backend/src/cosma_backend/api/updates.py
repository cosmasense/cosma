"""
Updates API Blueprint

Handles endpoints related to streaming updates.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from quart import Blueprint, abort, current_app, make_response, request, stream_with_context
from quart_schema import validate_request, validate_response

from cosma_backend.logging import get_logger
from cosma_backend.models.update import UpdateOpcode
from cosma_backend.utils.pubsub import subscribe
from cosma_backend.utils.sse import ServerSentEvent, sse_comment

if TYPE_CHECKING:
    from cosma_backend.app import App
    current_app: App

updates_bp = Blueprint('updates', __name__)

logger = get_logger(__name__)


@updates_bp.get("/", strict_slashes=False)  # type: ignore[return-value]
async def updates():
    """Stream real-time updates via Server-Sent Events"""
    if "text/event-stream" not in request.accept_mimetypes:
        abort(400)
    
    @stream_with_context
    async def updates_generator():
        # Keep-alive interval: send a comment if no updates for 15 seconds
        # This prevents proxy/browser timeouts and helps detect dead connections
        KEEPALIVE_INTERVAL = 15.0

        shutdown_event = getattr(current_app, "shutdown_event", None)

        try:
            with subscribe(current_app.updates_hub) as queue:
                while True:
                    # Short-circuit the moment shutdown is signalled; the
                    # uvicorn graceful window is only ~10 s and we don't want
                    # this long-lived stream to eat it all waiting for
                    # queue.get(). Event-wait runs concurrently with the
                    # real queue.get() so normal throughput is unaffected.
                    if shutdown_event is not None and shutdown_event.is_set():
                        return

                    try:
                        update = await asyncio.wait_for(queue.get(), timeout=KEEPALIVE_INTERVAL)

                        if update.opcode is UpdateOpcode.SHUTTING_DOWN:
                            yield update.to_sse().encode()
                            return

                        yield update.to_sse().encode()
                    except asyncio.TimeoutError:
                        yield sse_comment("keepalive")
        except asyncio.CancelledError:
            # Expected on shutdown / client disconnect. Swallow so uvicorn
            # doesn't print a traceback for a benign cancel.
            return

    response = await make_response(
        updates_generator(),
        {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Transfer-Encoding': 'chunked',
        },
    )
    response.timeout = None  # type: ignore[assignment]
    return response
