"""
Synchronous wrapper for the async Cosma client.
"""

import asyncio
import warnings
from typing import Any, Callable, Coroutine, Dict, Iterator, List, Optional

from .client import Client
from .models import Update


class SyncClient:
    """
    Synchronous wrapper around the async Client.

    This class provides a synchronous API for the Cosma client,
    making it easier to use in non-async contexts like CLI commands.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:60534"):
        self._base_url = base_url

    def _run_with_client(
        self,
        async_method: Callable[[Client], Coroutine[Any, Any, Any]],
    ) -> Any:
        """Run an async method with a fresh client and proper cleanup."""
        async def run_with_cleanup():
            client = Client(self._base_url)
            try:
                return await async_method(client)
            finally:
                await client.close()

        # Suppress the "Task was destroyed" warning for urllib3's idle connection watcher
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Task was destroyed.*")
            return asyncio.run(run_with_cleanup())

    @property
    def base_url(self) -> str:
        return self._base_url

    # =========================================================================
    # Index API
    # =========================================================================

    def index_directory(self, directory_path: str) -> Dict[str, Any]:
        """Index all files in a directory for searching"""
        return self._run_with_client(
            lambda c: c.index_directory(directory_path)
        )

    def index_file(self, file_path: str) -> Dict[str, Any]:
        """Index a single file"""
        return self._run_with_client(lambda c: c.index_file(file_path))

    def index_status(self) -> Dict[str, Any]:
        """Get the current status of indexing operations"""
        return self._run_with_client(lambda c: c.index_status())

    # =========================================================================
    # Status API
    # =========================================================================

    def status(self) -> Dict[str, Any]:
        """Get the current status of the backend"""
        return self._run_with_client(lambda c: c.status())

    # =========================================================================
    # Search API
    # =========================================================================

    def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """Search for files based on a query string"""
        return self._run_with_client(lambda c: c.search(query, filters, limit))

    def search_by_keywords(
        self,
        keywords: List[str],
        match_all: bool = False
    ) -> Dict[str, Any]:
        """Search for files by specific keywords"""
        return self._run_with_client(
            lambda c: c.search_by_keywords(keywords, match_all)
        )

    def find_similar_files(self, file_id: int, limit: int = 10) -> Dict[str, Any]:
        """Find files similar to a given file"""
        return self._run_with_client(
            lambda c: c.find_similar_files(file_id, limit)
        )

    def autocomplete(self, q: str, limit: int = 10) -> Dict[str, Any]:
        """Get autocomplete suggestions for search queries"""
        return self._run_with_client(lambda c: c.autocomplete(q, limit))

    # =========================================================================
    # Watch API
    # =========================================================================

    def watch_directory(self, directory_path: str) -> Dict[str, Any]:
        """Start watching a directory for changes"""
        return self._run_with_client(
            lambda c: c.watch_directory(directory_path)
        )

    def get_watch_jobs(self) -> Dict[str, Any]:
        """Get all watched directory jobs"""
        return self._run_with_client(lambda c: c.get_watch_jobs())

    def delete_watch_job(self, job_id: int) -> Dict[str, Any]:
        """Delete a watched directory job by ID"""
        return self._run_with_client(lambda c: c.delete_watch_job(job_id))

    # =========================================================================
    # Files API
    # =========================================================================

    def get_file(self, file_id: int) -> Dict[str, Any]:
        """Get details of a specific file by ID"""
        return self._run_with_client(lambda c: c.get_file(file_id))

    def get_file_stats(self) -> Dict[str, Any]:
        """Get statistics about indexed files"""
        return self._run_with_client(lambda c: c.get_file_stats())

    # =========================================================================
    # Filters API
    # =========================================================================

    def get_filter_config(self) -> Dict[str, Any]:
        """Get the current filter configuration"""
        return self._run_with_client(lambda c: c.get_filter_config())

    def update_filter_config(
        self,
        mode: Optional[str] = None,
        exclude: Optional[List[str]] = None,
        include: Optional[List[str]] = None,
        apply_immediately: bool = True,
    ) -> Dict[str, Any]:
        """Update the filter configuration"""
        return self._run_with_client(
            lambda c: c.update_filter_config(
                mode, exclude, include, apply_immediately
            )
        )

    def add_filter_pattern(
        self,
        pattern: str,
        pattern_type: str = "exclude"
    ) -> Dict[str, Any]:
        """Add a pattern to the filter configuration"""
        return self._run_with_client(
            lambda c: c.add_filter_pattern(pattern, pattern_type)
        )

    def remove_filter_pattern(
        self,
        pattern: str,
        pattern_type: str = "exclude"
    ) -> Dict[str, Any]:
        """Remove a pattern from the filter configuration"""
        return self._run_with_client(
            lambda c: c.remove_filter_pattern(pattern, pattern_type)
        )

    def test_filter_patterns(
        self,
        patterns: List[str],
        file_paths: List[str],
        mode: str = "blacklist",
    ) -> Dict[str, Any]:
        """Test pattern matching against file paths"""
        return self._run_with_client(
            lambda c: c.test_filter_patterns(patterns, file_paths, mode)
        )

    def get_filter_defaults(self) -> Dict[str, Any]:
        """Get the default filter configuration"""
        return self._run_with_client(lambda c: c.get_filter_defaults())

    def apply_filter_changes(self) -> Dict[str, Any]:
        """Apply current filter configuration to database"""
        return self._run_with_client(lambda c: c.apply_filter_changes())

    def reset_filters(self) -> Dict[str, Any]:
        """Reset filter configuration to defaults"""
        return self._run_with_client(lambda c: c.reset_filters())

    # =========================================================================
    # Updates API
    # =========================================================================

    def stream_updates(self) -> Iterator[Update]:
        """
        Stream updates from the server.

        Note: This is a blocking iterator. For async usage, use the
        async Client directly.
        """
        async def stream_all():
            client = Client(self._base_url)
            try:
                async for update in client.stream_updates():
                    yield update
            finally:
                await client.close()

        # For streaming, we need a different approach - collect all at once
        # or use a queue-based approach
        async def get_updates():
            updates = []
            client = Client(self._base_url)
            try:
                async for update in client.stream_updates():
                    updates.append(update)
            finally:
                await client.close()
            return updates

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Task was destroyed.*")
            updates = asyncio.run(get_updates())

        for update in updates:
            yield update
