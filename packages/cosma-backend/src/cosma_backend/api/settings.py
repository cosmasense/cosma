"""
Settings API Blueprint

Handles endpoints for reading and updating application settings.
"""

from typing import TYPE_CHECKING

from quart import Blueprint, current_app, request

if TYPE_CHECKING:
    from cosma_backend.app import app as current_app

settings_bp = Blueprint('settings', __name__)


@settings_bp.get("/")
async def get_settings():
    """Return all settings grouped by section."""
    return current_app.settings_manager.to_dict()


@settings_bp.put("/")
async def update_settings():
    """Partial update of settings. Accepts dotted TOML paths."""
    data = await request.get_json()
    if not data or not isinstance(data, dict):
        return {"error": "Request body must be a JSON object"}, 400

    try:
        updated = current_app.settings_manager.update(data)
    except (KeyError, ValueError) as e:
        return {"error": str(e)}, 400

    # Propagate live setting changes to running services
    if "summarizer.idle_unload_seconds" in data:
        if hasattr(current_app, "model_lifecycle"):
            current_app.model_lifecycle.idle_unload_seconds = (
                current_app.settings_manager.settings.summarizer.idle_unload_seconds
            )

    return current_app.settings_manager.to_dict()


@settings_bp.get("/defaults")
async def get_defaults():
    """Return default values for all settings."""
    return current_app.settings_manager.defaults()
