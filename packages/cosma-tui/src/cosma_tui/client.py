import niquests
import json
from typing import Optional, Dict, Any, List, AsyncIterator
from urllib.parse import urljoin

from .models import Update, UpdateOpcode

class Client:
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

    # Index API (from APIDoc.md)
    async def index_directory(self, directory_path: str) -> Dict[str, Any]:
        """Index all files in a directory for searching"""
        data = {"directory_path": directory_path}
        response = await self.session.post(self._url("/api/index/directory"),
                                   data=json.dumps(data))
        return self._handle_response(response)

    async def index_file(self, file_path: str) -> Dict[str, Any]:
        """Index a single file"""
        data = {"file_path": file_path}
        response = await self.session.post(self._url("/api/index/file"),
                                   data=json.dumps(data))
        return self._handle_response(response)

    async def index_status(self) -> Dict[str, Any]:
        """Get the current status of indexing operations"""
        response = await self.session.get(self._url("/api/index/status"))
        return self._handle_response(response)

    # Search API (from APIDoc.md)
    async def search(self, query: str, filters: Optional[Dict[str, Any]] = None,
              limit: int = 50) -> Dict[str, Any]:
        """Search for files based on a query string"""
        data = {"query": query, "limit": limit}
        if filters:
            data["filters"] = filters
        
        response = await self.session.post(self._url("/api/search/"),
                                   data=json.dumps(data))
        return self._handle_response(response)

    async def search_by_keywords(self, keywords: List[str], match_all: bool = False) -> Dict[str, Any]:
        """Search for files by specific keywords"""
        data = {"keywords": keywords, "match_all": match_all}
        response = await self.session.post(self._url("/api/search/keywords"),
                                   data=json.dumps(data))
        return self._handle_response(response)

    async def find_similar_files(self, file_id: int, limit: int = 10) -> Dict[str, Any]:
        """Find files similar to a given file"""
        response = await self.session.get(self._url(f"/api/search/{file_id}/similar"),
                                  params={"limit": limit})
        return self._handle_response(response)

    async def autocomplete(self, q: str, limit: int = 10) -> Dict[str, Any]:
        """Get autocomplete suggestions for search queries"""
        response = await self.session.get(self._url("/api/search/autocomplete"),
                                  params={"q": q, "limit": limit})
        return self._handle_response(response)

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
                        except Exception as e:
                            # Fallback: create a generic INFO update
                            yield Update.create(UpdateOpcode.INFO, message=data_str)
        except Exception as e:
            raise

    async def close(self):
        """Close the session"""
        await self.session.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
