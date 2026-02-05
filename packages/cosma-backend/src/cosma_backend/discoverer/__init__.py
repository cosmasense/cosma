"""
File Discovery Module

Recursively walks directories to discover files for indexing.

Features:
- Yields File objects with basic metadata (path, size, timestamps)
- Applies FilterConfig to exclude unwanted files/directories
- Handles symlinks and permission errors gracefully
- Memory-efficient generator-based discovery

Usage:
    discoverer = Discoverer()
    for file in discoverer.files_in("/path/to/dir", filter_config=config):
        # file is a File object with DISCOVERED status
        pass
"""

from .discoverer import Discoverer as Discoverer
