"""
Settings API Blueprint

Handles endpoints for reading and updating application settings.
"""

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from quart import Blueprint, current_app, request

from cosma_backend.logging import get_logger

if TYPE_CHECKING:
    from cosma_backend.app import app as current_app

logger = get_logger(__name__)

settings_bp = Blueprint('settings', __name__)


@settings_bp.get("/")
async def get_settings():
    """Return all settings grouped by section."""
    return current_app.settings_manager.to_dict()


@settings_bp.put("/")
async def update_settings():
    """Partial update of settings. Accepts dotted TOML paths."""
    data = await request.get_json()
    if not data or not isinstance(data, dict):
        return {"error": "Request body must be a JSON object"}, 400

    try:
        updated = current_app.settings_manager.update(data)
    except (KeyError, ValueError) as e:
        return {"error": str(e)}, 400

    # Propagate live setting changes to running services
    if "summarizer.idle_unload_seconds" in data:
        if hasattr(current_app, "model_lifecycle"):
            current_app.model_lifecycle.idle_unload_seconds = (
                current_app.settings_manager.settings.summarizer.idle_unload_seconds
            )

    return current_app.settings_manager.to_dict()


@settings_bp.get("/defaults")
async def get_defaults():
    """Return default values for all settings."""
    return current_app.settings_manager.defaults()


@settings_bp.post("/test_model")
async def test_model():
    """
    Quick availability check for the currently configured summarizer model.

    Does NOT load the model — just verifies the backend is reachable and the
    model is known to it. Designed to run as a non-blocking background task.

    Response shape:
        {"ok": true, "provider": "ollama", "model": "...", "detail": "..."}
        {"ok": false, "provider": "ollama", "model": "...", "detail": "...error..."}
    """
    settings = current_app.settings_manager.settings
    provider = (settings.summarizer.provider or "auto").lower()

    # Resolve "auto" to a concrete provider based on env / fallback
    effective_provider = provider
    if provider == "auto":
        if os.getenv("OPENAI_API_KEY"):
            effective_provider = "online"
        else:
            effective_provider = "ollama"

    try:
        if effective_provider == "ollama":
            return await _test_ollama(settings.summarizer.ollama)
        elif effective_provider == "llamacpp":
            return await _test_llamacpp(settings.summarizer.llamacpp)
        elif effective_provider == "online":
            return await _test_online(settings.summarizer.online)
        else:
            return {
                "ok": False,
                "provider": effective_provider,
                "model": "",
                "detail": f"Unknown summarizer provider: {effective_provider}",
            }
    except Exception as e:
        logger.exception("Model availability test crashed",
                        provider=effective_provider, error=str(e))
        return {
            "ok": False,
            "provider": effective_provider,
            "model": "",
            "detail": f"Internal error during test: {e}",
        }


async def _test_ollama(cfg) -> dict:
    """Verify Ollama host is reachable and the model is pulled."""
    model = (cfg.model or "").strip() or "qwen3-vl:2b-instruct"
    host = (cfg.host or "").strip() or "http://localhost:11434"
    url = f"{host.rstrip('/')}/api/tags"

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(url)
    except httpx.HTTPError as e:
        return {
            "ok": False,
            "provider": "ollama",
            "model": model,
            "detail": f"Ollama not reachable at {host}: {e}",
        }

    if resp.status_code != 200:
        return {
            "ok": False,
            "provider": "ollama",
            "model": model,
            "detail": f"Ollama returned HTTP {resp.status_code}",
        }

    try:
        data = resp.json()
    except Exception as e:
        return {
            "ok": False,
            "provider": "ollama",
            "model": model,
            "detail": f"Invalid Ollama response: {e}",
        }

    models = [m.get("name", "") for m in data.get("models", [])]
    # Match exact or by tag prefix (e.g. "qwen3-vl:2b-instruct" vs "qwen3-vl:2b")
    base_name = model.split(":", 1)[0]
    has_model = any(
        m == model or m.startswith(base_name + ":") or m == base_name
        for m in models
    )

    if not has_model:
        return {
            "ok": False,
            "provider": "ollama",
            "model": model,
            "detail": (
                f"Ollama is running but '{model}' is not pulled. "
                f"Run: ollama pull {model}"
            ),
        }

    return {
        "ok": True,
        "provider": "ollama",
        "model": model,
        "detail": f"Ollama model '{model}' is available",
    }


async def _test_llamacpp(cfg) -> dict:
    """Verify llama.cpp model file is locally present or accessible on HF."""
    # Prefer local path if set
    model_path = (cfg.model_path or "").strip()
    if model_path:
        if Path(model_path).expanduser().exists():
            return {
                "ok": True,
                "provider": "llamacpp",
                "model": model_path,
                "detail": f"Local model file exists at {model_path}",
            }
        return {
            "ok": False,
            "provider": "llamacpp",
            "model": model_path,
            "detail": f"Model path does not exist: {model_path}",
        }

    repo_id = (cfg.repo_id or "").strip() or "unsloth/Qwen3-VL-2B-Instruct-GGUF"
    filename = (cfg.filename or "").strip() or "*Q4_K_M.gguf"

    # Check HuggingFace reachability (HEAD on the repo page is lightweight)
    url = f"https://huggingface.co/api/models/{repo_id}"
    try:
        async with httpx.AsyncClient(timeout=5.0, follow_redirects=True) as client:
            resp = await client.get(url)
    except httpx.HTTPError as e:
        return {
            "ok": False,
            "provider": "llamacpp",
            "model": f"{repo_id}/{filename}",
            "detail": f"Cannot reach HuggingFace for {repo_id}: {e}",
        }

    if resp.status_code == 404:
        return {
            "ok": False,
            "provider": "llamacpp",
            "model": f"{repo_id}/{filename}",
            "detail": f"HuggingFace repo not found: {repo_id}",
        }
    if resp.status_code != 200:
        return {
            "ok": False,
            "provider": "llamacpp",
            "model": f"{repo_id}/{filename}",
            "detail": f"HuggingFace returned HTTP {resp.status_code} for {repo_id}",
        }

    return {
        "ok": True,
        "provider": "llamacpp",
        "model": f"{repo_id}/{filename}",
        "detail": f"HuggingFace repo '{repo_id}' is reachable",
    }


async def _test_online(cfg) -> dict:
    """Verify online provider has a valid API key configured."""
    model = (cfg.model or "").strip() or "openai/gpt-4.1-nano-2025-04-14"

    if not os.getenv("OPENAI_API_KEY"):
        return {
            "ok": False,
            "provider": "online",
            "model": model,
            "detail": "OPENAI_API_KEY environment variable is not set",
        }

    return {
        "ok": True,
        "provider": "online",
        "model": model,
        "detail": f"API key is configured for '{model}'",
    }
