import niquests
import json
from typing import Optional, Dict, Any, List, AsyncIterator
from urllib.parse import urljoin

from .models import Update, UpdateOpcode
from .exceptions import ServerNotRunningError, APIError


class Client:
    """Async client for the Cosma API."""

    session: niquests.AsyncSession
    base_url: str

    def __init__(self, base_url: str = "http://127.0.0.1:60534"):
        self.session = niquests.AsyncSession()
        self.base_url = base_url
        self.session.headers.update({"Content-Type": "application/json"})

    def _url(self, endpoint: str) -> str:
        """Helper to build full URL"""
        return urljoin(self.base_url, endpoint)

    def _handle_response(self, response: niquests.Response) -> Dict[str, Any]:
        """Handle response and raise on errors"""
        response.raise_for_status()
        return response.json()

    # =========================================================================
    # Index API
    # =========================================================================

    async def index_directory(self, directory_path: str) -> Dict[str, Any]:
        """Index all files in a directory for searching"""
        data = {"directory_path": directory_path}
        try:
            response = await self.session.post(
                self._url("/api/index/directory"),
                data=json.dumps(data)
            )
            return self._handle_response(response)
        except niquests.exceptions.ConnectionError:
            raise ServerNotRunningError(self.base_url)

    async def index_file(self, file_path: str) -> Dict[str, Any]:
        """Index a single file"""
        data = {"file_path": file_path}
        try:
            response = await self.session.post(
                self._url("/api/index/file"),
                data=json.dumps(data)
            )
            return self._handle_response(response)
        except niquests.exceptions.ConnectionError:
            raise ServerNotRunningError(self.base_url)

    async def index_status(self) -> Dict[str, Any]:
        """Get the current status of indexing operations"""
        try:
            response = await self.session.get(self._url("/api/index/status"))
            return self._handle_response(response)
        except niquests.exceptions.ConnectionError:
            raise ServerNotRunningError(self.base_url)

    # =========================================================================
    # Status API
    # =========================================================================

    async def status(self) -> Dict[str, Any]:
        """Get the current status of the backend"""
        try:
            response = await self.session.get(self._url("/api/status"))
            return self._handle_response(response)
        except niquests.exceptions.ConnectionError:
            raise ServerNotRunningError(self.base_url)

    # =========================================================================
    # Search API
    # =========================================================================

    async def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """Search for files based on a query string"""
        data: Dict[str, Any] = {"query": query, "limit": limit}
        if filters:
            data["filters"] = filters

        try:
            response = await self.session.post(
                self._url("/api/search/"),
                data=json.dumps(data)
            )
            return self._handle_response(response)
        except niquests.exceptions.ConnectionError:
            raise ServerNotRunningError(self.base_url)

    async def search_by_keywords(
        self,
        keywords: List[str],
        match_all: bool = False
    ) -> Dict[str, Any]:
        """Search for files by specific keywords"""
        data = {"keywords": keywords, "match_all": match_all}
        try:
            response = await self.session.post(
                self._url("/api/search/keywords"),
                data=json.dumps(data)
            )
            return self._handle_response(response)
        except niquests.exceptions.ConnectionError:
            raise ServerNotRunningError(self.base_url)

    async def find_similar_files(
        self,
        file_id: int,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Find files similar to a given file"""
        try:
            response = await self.session.get(
                self._url(f"/api/search/{file_id}/similar"),
                params={"limit": limit}
            )
            return self._handle_response(response)
        except niquests.exceptions.ConnectionError:
            raise ServerNotRunningError(self.base_url)

    async def autocomplete(self, q: str, limit: int = 10) -> Dict[str, Any]:
        """Get autocomplete suggestions for search queries"""
        try:
            response = await self.session.get(
                self._url("/api/search/autocomplete"),
                params={"q": q, "limit": limit}
            )
            return self._handle_response(response)
        except niquests.exceptions.ConnectionError:
            raise ServerNotRunningError(self.base_url)

    # =========================================================================
    # Watch API
    # =========================================================================

    async def watch_directory(self, directory_path: str) -> Dict[str, Any]:
        """Start watching a directory for changes"""
        data = {"directory_path": directory_path}
        try:
            response = await self.session.post(
                self._url("/api/watch/"),
                data=json.dumps(data)
            )
            return self._handle_response(response)
        except niquests.exceptions.ConnectionError:
            raise ServerNotRunningError(self.base_url)

    async def get_watch_jobs(self) -> Dict[str, Any]:
        """Get all watched directory jobs"""
        try:
            response = await self.session.get(self._url("/api/watch/jobs"))
            return self._handle_response(response)
        except niquests.exceptions.ConnectionError:
            raise ServerNotRunningError(self.base_url)

    async def delete_watch_job(self, job_id: int) -> Dict[str, Any]:
        """Delete a watched directory job by ID"""
        try:
            response = await self.session.delete(
                self._url(f"/api/watch/jobs/{job_id}")
            )
            return self._handle_response(response)
        except niquests.exceptions.ConnectionError:
            raise ServerNotRunningError(self.base_url)

    # =========================================================================
    # Files API
    # =========================================================================

    async def get_file(self, file_id: int) -> Dict[str, Any]:
        """Get details of a specific file by ID"""
        try:
            response = await self.session.get(
                self._url(f"/api/files/{file_id}")
            )
            return self._handle_response(response)
        except niquests.exceptions.ConnectionError:
            raise ServerNotRunningError(self.base_url)

    async def get_file_stats(self) -> Dict[str, Any]:
        """Get statistics about indexed files"""
        try:
            response = await self.session.get(self._url("/api/files/stats"))
            return self._handle_response(response)
        except niquests.exceptions.ConnectionError:
            raise ServerNotRunningError(self.base_url)

    # =========================================================================
    # Filters API
    # =========================================================================

    async def get_filter_config(self) -> Dict[str, Any]:
        """Get the current filter configuration"""
        try:
            response = await self.session.get(self._url("/api/filters/config"))
            return self._handle_response(response)
        except niquests.exceptions.ConnectionError:
            raise ServerNotRunningError(self.base_url)

    async def update_filter_config(
        self,
        mode: Optional[str] = None,
        exclude: Optional[List[str]] = None,
        include: Optional[List[str]] = None,
        apply_immediately: bool = True,
    ) -> Dict[str, Any]:
        """Update the filter configuration"""
        data: Dict[str, Any] = {"apply_immediately": apply_immediately}
        if mode is not None:
            data["mode"] = mode
        if exclude is not None:
            data["exclude"] = exclude
        if include is not None:
            data["include"] = include

        try:
            response = await self.session.put(
                self._url("/api/filters/config"),
                data=json.dumps(data)
            )
            return self._handle_response(response)
        except niquests.exceptions.ConnectionError:
            raise ServerNotRunningError(self.base_url)

    async def add_filter_pattern(
        self,
        pattern: str,
        pattern_type: str = "exclude"
    ) -> Dict[str, Any]:
        """Add a pattern to the filter configuration"""
        data = {"pattern": pattern, "pattern_type": pattern_type}
        try:
            response = await self.session.post(
                self._url("/api/filters/pattern"),
                data=json.dumps(data)
            )
            return self._handle_response(response)
        except niquests.exceptions.ConnectionError:
            raise ServerNotRunningError(self.base_url)

    async def remove_filter_pattern(
        self,
        pattern: str,
        pattern_type: str = "exclude"
    ) -> Dict[str, Any]:
        """Remove a pattern from the filter configuration"""
        data = {"pattern": pattern, "pattern_type": pattern_type}
        try:
            response = await self.session.delete(
                self._url("/api/filters/pattern"),
                data=json.dumps(data)
            )
            return self._handle_response(response)
        except niquests.exceptions.ConnectionError:
            raise ServerNotRunningError(self.base_url)

    async def test_filter_patterns(
        self,
        patterns: List[str],
        file_paths: List[str],
        mode: str = "blacklist",
    ) -> Dict[str, Any]:
        """Test pattern matching against file paths"""
        data = {"patterns": patterns, "file_paths": file_paths, "mode": mode}
        try:
            response = await self.session.post(
                self._url("/api/filters/test"),
                data=json.dumps(data)
            )
            return self._handle_response(response)
        except niquests.exceptions.ConnectionError:
            raise ServerNotRunningError(self.base_url)

    async def get_filter_defaults(self) -> Dict[str, Any]:
        """Get the default filter configuration"""
        try:
            response = await self.session.get(self._url("/api/filters/defaults"))
            return self._handle_response(response)
        except niquests.exceptions.ConnectionError:
            raise ServerNotRunningError(self.base_url)

    async def apply_filter_changes(self) -> Dict[str, Any]:
        """Apply current filter configuration to database"""
        try:
            response = await self.session.post(self._url("/api/filters/apply"))
            return self._handle_response(response)
        except niquests.exceptions.ConnectionError:
            raise ServerNotRunningError(self.base_url)

    async def reset_filters(self) -> Dict[str, Any]:
        """Reset filter configuration to defaults"""
        try:
            response = await self.session.post(self._url("/api/filters/reset"))
            return self._handle_response(response)
        except niquests.exceptions.ConnectionError:
            raise ServerNotRunningError(self.base_url)

    # =========================================================================
    # Settings API
    # =========================================================================

    async def get_settings(self) -> Dict[str, Any]:
        """Get all application settings grouped by section"""
        try:
            response = await self.session.get(self._url("/api/settings/"))
            return self._handle_response(response)
        except niquests.exceptions.ConnectionError:
            raise ServerNotRunningError(self.base_url)

    async def update_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Update application settings (partial update with flat config keys)"""
        try:
            response = await self.session.put(
                self._url("/api/settings/"),
                data=json.dumps(settings)
            )
            return self._handle_response(response)
        except niquests.exceptions.ConnectionError:
            raise ServerNotRunningError(self.base_url)

    async def get_settings_defaults(self) -> Dict[str, Any]:
        """Get default values for all settings"""
        try:
            response = await self.session.get(self._url("/api/settings/defaults"))
            return self._handle_response(response)
        except niquests.exceptions.ConnectionError:
            raise ServerNotRunningError(self.base_url)

    # =========================================================================
    # Updates API (SSE)
    # =========================================================================

    async def stream_updates(self) -> AsyncIterator[Update]:
        """Stream server-sent events from /api/updates"""
        url = self._url("/api/updates")
        try:
            response = await self.session.get(url, stream=True, timeout=30)
            response.raise_for_status()

            async for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8') if isinstance(line, bytes) else line
                    # SSE format: "data: {...}"
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]  # Remove "data: " prefix
                        try:
                            # Parse and create Update instance
                            update = Update.from_sse_data(data_str)
                            yield update
                        except Exception:
                            # Fallback: create a generic INFO update
                            yield Update.create(UpdateOpcode.INFO, message=data_str)
        except niquests.exceptions.ConnectionError:
            raise ServerNotRunningError(self.base_url)

    # =========================================================================
    # Context Manager
    # =========================================================================

    async def close(self):
        """Close the session"""
        await self.session.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
