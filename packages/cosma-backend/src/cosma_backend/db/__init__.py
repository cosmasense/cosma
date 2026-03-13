"""
Database Module

SQLite database with async support via asqlite, FTS5 for full-text search,
and sqlite-vec for vector similarity search.

Key features:
- Connection pooling with sqlite-vec extension auto-loaded
- Schema migrations run automatically on connect
- File CRUD with FTS5 index synchronization
- Vector embedding storage and k-NN search
- Queue item persistence for crash recovery
"""

from .database import Database as Database
from .database import connect as connect
