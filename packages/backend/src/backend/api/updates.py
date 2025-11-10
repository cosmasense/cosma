"""
Updates API Blueprint

Handles endpoints related to streaming updates.
"""

import asyncio
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from quart import Blueprint, abort, current_app, make_response, request, stream_with_context
from quart_schema import validate_request, validate_response

from backend.logging import sm
from backend.models.update import UpdateOpcode
from backend.utils.pubsub import subscribe
from backend.utils.sse import ServerSentEvent, sse_comment

if TYPE_CHECKING:
    from backend.app import app as current_app

updates_bp = Blueprint('updates', __name__)

logger = logging.getLogger(__name__)


@updates_bp.get("/")  # type: ignore[return-value]
async def updates():
    """Stream real-time updates via Server-Sent Events"""
    if "text/event-stream" not in request.accept_mimetypes:
        abort(400)
    
    @stream_with_context
    async def updates_generator():
        # Keep-alive interval: send a comment if no updates for 15 seconds
        # This prevents proxy/browser timeouts and helps detect dead connections
        KEEPALIVE_INTERVAL = 15.0
        
        with subscribe(current_app.updates_hub) as queue:
            while True:
                try:
                    # Wait for an update with timeout
                    update = await asyncio.wait_for(queue.get(), timeout=KEEPALIVE_INTERVAL)
                    
                    if update.opcode is UpdateOpcode.SHUTTING_DOWN:
                        print(update.to_sse().encode())
                        yield update.to_sse().encode()
                        return  # close connection
                        
                    yield update.to_sse().encode()
                except asyncio.TimeoutError:
                    # No updates received within the keepalive interval
                    # Send a keep-alive comment (SSE spec: lines starting with : are comments)
                    yield sse_comment("keepalive")

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
