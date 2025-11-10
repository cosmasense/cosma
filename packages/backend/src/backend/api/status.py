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

from backend.utils.pubsub import subscribe

if TYPE_CHECKING:
    from backend.app import app as current_app

status_bp = Blueprint('status', __name__)


@status_bp.get("/")  # type: ignore[return-value]
async def status():
    """Get current application status and active jobs count"""
    return {
        "jobs": len(current_app.jobs),
    }
