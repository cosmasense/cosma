"""Applications source.

A separate "what apps are installed on this Mac" indexer. Apps live in
their own `applications` table (see schema.sql) so search results can
render an app-divider next to file hits — "I forgot what it's called
but it's the one that does X" is a search pattern, not a file system
question.
"""

from .applications_discoverer import ApplicationsDiscoverer
from .applications_indexer import (
    DEFAULT_RESCAN_INTERVAL_SECONDS,
    ApplicationsIndexer,
    ScanResult,
)
from .applications_repository import ApplicationsRepository

__all__ = [
    "DEFAULT_RESCAN_INTERVAL_SECONDS",
    "ApplicationsDiscoverer",
    "ApplicationsIndexer",
    "ApplicationsRepository",
    "ScanResult",
]
