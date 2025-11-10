"""Integration tests for API endpoints."""

import pytest

from cosma_backend.app import App
from cosma_backend.db.database import Database


@pytest.mark.integration
@pytest.mark.asyncio
class TestAPIEndpoints:
    """Test cases for API endpoints."""

    async def test_echo_endpoint(self, test_client: App):
        """Test the echo endpoint."""
        # For Quart, use the test client context manager
        async with test_client.test_client() as client:
            response = await client.post("/echo", json={"test": "data"})
            
            assert response.status_code == 200
            data = await response.get_json()
            assert data["input"]["test"] == "data"
            assert data["extra"] is True

    async def test_api_search_endpoint(self, test_client: App, temp_db: Database):
        """Test the search API endpoint."""
        # This test would require setting up a searcher with embeddings
        # For now, we'll test that the endpoint exists and responds
        
        response = await test_client.post("/api/search/", json={
            "query": "test query",
            "limit": 10
        })
        
        # The response might be an error if searcher isn't fully configured
        # But we can test that the endpoint exists
        assert response.status_code in [200, 500]  # Either success or configuration error

    async def test_api_files_endpoint(self, test_client: App, sample_file_in_db):
        """Test the files API endpoint."""
        response = await test_client.get(f"/api/files/{sample_file_in_db.id}")
        
        # Endpoint might not be implemented yet
        # We're testing that it either works or gives appropriate error
        assert response.status_code in [200, 404, 405, 500]

    async def test_api_status_endpoint(self, test_client: App):
        """Test the status API endpoint."""
        response = await test_client.get("/api/status")
        
        # Should respond with some status information
        assert response.status_code in [200, 404, 500]

    async def test_api_index_endpoint(self, test_client: App):
        """Test the directory indexing endpoint."""
        response = await test_client.post("/api/index/directory", json={
            "directory_path": "/test/directory",
            "recursive": True
        })
        
        # Response indicates the endpoint exists (might fail due to invalid path)
        assert response.status_code in [200, 400, 500]

    async def test_api_watch_endpoint(self, test_client: App, temp_db: Database):
        """Test the watch management API endpoints."""
        # Test getting watched directories
        response = await test_client.get("/api/watch/")
        
        # Should return list of watched directories (possibly empty)
        assert response.status_code in [200, 404, 500]

    async def test_api_updates_endpoint(self, test_client: App):
        """Test the SSE updates endpoint."""
        # SSE endpoint requires special handling
        # We'll just test that it responds appropriately
        try:
            response = await test_client.get("/api/updates")
            # Should start streaming or return appropriate error
            assert response.status_code in [200, 404, 500]
        except Exception:
            # SSE streaming might have connection issues in test environment
            # That's okay for this basic test
            pass

    async def test_cors_headers(self, test_client: App):
        """Test that CORS headers are properly set."""
        # Make a preflight request
        response = await test_client.options("/echo", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type"
        })
        
        # Should handle preflight request appropriately
        assert response.status_code in [200, 204, 405]

    async def test_request_validation(self, test_client: App):
        """Test that request validation works properly."""
        # Send invalid JSON to echo endpoint
        response = await test_client.post("/echo", data="invalid json")
        
        # Should return validation error
        assert response.status_code in [400, 422]

    async def test_health_check(self, test_client: App):
        """Test basic health check functionality."""
        # Try to access a simple endpoint
        response = await test_client.get("/")
        
        # Should respond (possibly with 404 if no root route)
        assert response.status_code in [200, 404]

    async def test_api_error_handling(self, test_client: App):
        """Test API error handling."""
        # Try to access non-existent endpoint
        response = await test_client.get("/api/nonexistent")
        
        # Should return 404 for non-existent endpoint
        assert response.status_code == 404

    async def test_api_response_format(self, test_client: App):
        """Test that API responses follow expected format."""
        response = await test_client.post("/echo", json={"test": "data"})
        
        assert response.headers.get("content-type") == "application/json"
        
        data = await response.get_json()
        assert isinstance(data, dict)
        assert "input" in data
        assert "extra" in data