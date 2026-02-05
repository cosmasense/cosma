"""
File System Watcher Module

Monitors watched directories for file changes and triggers re-indexing.

Features:
- Uses watchdog library for cross-platform file system events
- Handles CREATE, MODIFY, DELETE, MOVE events
- Applies filter patterns to ignore excluded files/directories
- Enqueues changes to IndexingQueue with debouncing
- Maintains observer per watched directory

The watcher integrates with:
- FilterConfigManager: Apply include/exclude patterns
- IndexingQueue: Debounce and process file changes
- Database: Track watched directories
"""

from .watcher import Watcher as Watcher
