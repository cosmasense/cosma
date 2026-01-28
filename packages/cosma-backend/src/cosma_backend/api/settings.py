"""
Settings API Blueprint

Handles endpoints for reading and updating application settings.
"""

from typing import TYPE_CHECKING

from quart import Blueprint, current_app, request

from cosma_backend.settings import resolve_key

if TYPE_CHECKING:
    from cosma_backend.app import app as current_app

settings_bp = Blueprint('settings', __name__)


@settings_bp.get("/")
async def get_settings():
    """Return all settings grouped by section."""
    return current_app.settings_manager.to_dict()


@settings_bp.put("/")
async def update_settings():
    """Partial update of settings. Accepts flat config keys or TOML paths."""
    data = await request.get_json()
    if not data or not isinstance(data, dict):
        return {"error": "Request body must be a JSON object"}, 400

    # Resolve all keys to canonical flat config keys
    resolved: dict = {}
    for key, value in data.items():
        canonical = resolve_key(key)
        if canonical is None:
            return {"error": f"Unknown setting: {key}"}, 400
        resolved[canonical] = value

    try:
        updated = current_app.settings_manager.update(resolved)
    except KeyError as e:
        return {"error": str(e)}, 400

    # Sync updated values into app.config so they take effect immediately
    for key, value in updated.items():
        current_app.config[key] = value

    return current_app.settings_manager.to_dict()


@settings_bp.get("/defaults")
async def get_defaults():
    """Return default values for all settings."""
    return current_app.settings_manager.defaults()
