"""
REST API Module

All API routes are prefixed with /api when registered.

Blueprints:
- /files    - File CRUD, listing, status filtering
- /index    - Trigger directory/file indexing
- /search   - Hybrid semantic + full-text search
- /watch    - Manage watched directories
- /updates  - SSE endpoint for real-time updates
- /status   - Health check, stats, debug info
- /filters  - Include/exclude pattern management
- /settings - Persistent settings CRUD
- /queue    - Indexing queue status and control

Request/response models are defined in models.py using dataclasses.
Validation is handled by quart-schema decorators.
"""

from quart import Blueprint

from .applications import applications_bp
from .files import files_bp
from .index import index_bp
from .search import search_bp
from .watch import watch_bp
from .updates import updates_bp
from .status import status_bp
from .filters import filters_bp
from .settings import settings_bp
from .queue import queue_bp
from .bootstrap import bootstrap_bp

# Create the main API blueprint
api_blueprint = Blueprint('api', __name__)

# Register sub-blueprints
api_blueprint.register_blueprint(files_bp, url_prefix='/files')
api_blueprint.register_blueprint(index_bp, url_prefix='/index')
api_blueprint.register_blueprint(search_bp, url_prefix='/search')
api_blueprint.register_blueprint(watch_bp, url_prefix='/watch')
api_blueprint.register_blueprint(updates_bp, url_prefix='/updates')
api_blueprint.register_blueprint(status_bp, url_prefix='/status')
api_blueprint.register_blueprint(filters_bp, url_prefix='/filters')
api_blueprint.register_blueprint(settings_bp, url_prefix='/settings')
api_blueprint.register_blueprint(queue_bp, url_prefix='/queue')
api_blueprint.register_blueprint(bootstrap_bp, url_prefix='/bootstrap')
api_blueprint.register_blueprint(applications_bp, url_prefix='/applications')


__all__ = ['api_blueprint']
