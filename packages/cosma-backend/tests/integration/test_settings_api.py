"""Integration tests for Settings API endpoints."""

import pytest

from cosma_backend.app import App
from cosma_backend.db.database import Database


@pytest.mark.integration
@pytest.mark.asyncio
class TestSettingsAPI:
    """Test cases for Settings API endpoints."""

    async def test_get_settings(self, test_client: App):
        """GET /api/settings should return grouped settings."""
        async with test_client.test_client() as client:
            response = await client.get("/api/settings/")
            assert response.status_code == 200
            data = await response.get_json()
            assert "embedder" in data
            assert "summarizer" in data
            assert "parser" in data
            assert data["embedder"]["provider"] == "local"

    async def test_get_settings_defaults(self, test_client: App):
        """GET /api/settings/defaults should return default values."""
        async with test_client.test_client() as client:
            response = await client.get("/api/settings/defaults")
            assert response.status_code == 200
            data = await response.get_json()
            assert "embedder" in data
            assert data["embedder"]["provider"] == "local"

    async def test_update_settings(self, test_client: App):
        """PUT /api/settings should update settings."""
        async with test_client.test_client() as client:
            response = await client.put("/api/settings/", json={
                "AI_PROVIDER": "ollama",
            })
            assert response.status_code == 200
            data = await response.get_json()
            assert data["summarizer"]["provider"] == "ollama"

            # Verify in-memory config was updated
            assert test_client.config["AI_PROVIDER"] == "ollama"

    async def test_update_settings_invalid_key(self, test_client: App):
        """PUT /api/settings with unknown key should return 400."""
        async with test_client.test_client() as client:
            response = await client.put("/api/settings/", json={
                "NONEXISTENT_KEY": "value",
            })
            assert response.status_code == 400
            data = await response.get_json()
            assert "error" in data

    async def test_update_settings_empty_body(self, test_client: App):
        """PUT /api/settings with empty body should return 400."""
        async with test_client.test_client() as client:
            response = await client.put(
                "/api/settings/",
                data="",
                headers={"Content-Type": "application/json"},
            )
            assert response.status_code == 400

    async def test_update_multiple_settings(self, test_client: App):
        """PUT /api/settings should handle multiple keys."""
        async with test_client.test_client() as client:
            response = await client.put("/api/settings/", json={
                "EMBEDDING_DIMENSIONS": 1024,
                "OLLAMA_MODEL": "llama3",
            })
            assert response.status_code == 200
            data = await response.get_json()
            assert data["embedder"]["dimensions"] == 1024
            assert data["summarizer"]["ollama"]["model"] == "llama3"

    async def test_update_settings_type_coercion(self, test_client: App):
        """PUT /api/settings should coerce string values to correct types."""
        async with test_client.test_client() as client:
            response = await client.put("/api/settings/", json={
                "EMBEDDING_DIMENSIONS": "256",
            })
            assert response.status_code == 200
            data = await response.get_json()
            assert data["embedder"]["dimensions"] == 256

    async def test_settings_persist_after_update(self, test_client: App):
        """Settings should be readable after update."""
        async with test_client.test_client() as client:
            await client.put("/api/settings/", json={
                "WHISPER_PROVIDER": "local",
            })

            response = await client.get("/api/settings/")
            assert response.status_code == 200
            data = await response.get_json()
            assert data["parser"]["whisper"]["provider"] == "local"
