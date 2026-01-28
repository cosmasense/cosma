"""
Settings Manager Module

Manages persistent application settings stored in a TOML file.
Settings are loaded from defaults, then TOML file, then env vars (env wins).
API writes update both the TOML file and in-memory config.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from platformdirs import PlatformDirs

from cosma_backend.logging import get_logger

logger = get_logger(__name__)

# Try to import tomli_w for writing TOML
import tomli_w

SETTINGS_FILE = "settings.toml"


# Schema: maps flat app.config keys to TOML path + default + type
# Format: {CONFIG_KEY: {"path": "section.subsection.key", "default": value, "type": type}}
SETTINGS_SCHEMA: dict[str, dict[str, Any]] = {
    # Embedder
    "EMBEDDING_MODEL": {"path": "embedder.model", "default": "text-embedding-3-small", "type": str},
    "EMBEDDING_DIMENSIONS": {"path": "embedder.dimensions", "default": 512, "type": int},
    "LOCAL_EMBEDDING_MODEL": {"path": "embedder.local_model", "default": "intfloat/e5-base-v2", "type": str},
    "LOCAL_EMBEDDING_DIMENSIONS": {"path": "embedder.local_dimensions", "default": 768, "type": int},
    "EMBEDDING_PROVIDER": {"path": "embedder.provider", "default": "local", "type": str},

    # Summarizer
    "AI_PROVIDER": {"path": "summarizer.provider", "default": "auto", "type": str},
    "MAX_TOKENS_PER_REQUEST": {"path": "summarizer.max_tokens_per_request", "default": 100000, "type": int},
    "CHUNK_OVERLAP_TOKENS": {"path": "summarizer.chunk_overlap_tokens", "default": 1000, "type": int},
    "OLLAMA_MODEL": {"path": "summarizer.ollama.model", "default": "qwen3-vl:2b-instruct", "type": str},
    "OLLAMA_HOST": {"path": "summarizer.ollama.host", "default": "http://localhost:11434", "type": str},
    "OLLAMA_MODEL_CONTEXT_LENGTH": {"path": "summarizer.ollama.context_length", "default": 10000, "type": int},
    "ONLINE_MODEL": {"path": "summarizer.online.model", "default": "openai/gpt-4.1-nano-2025-04-14", "type": str},
    "ONLINE_MODEL_CONTEXT_LENGTH": {"path": "summarizer.online.context_length", "default": 128000, "type": int},
    "LLAMACPP_MODEL_CONTEXT_LENGTH": {"path": "summarizer.llamacpp.context_length", "default": 8192, "type": int},
    "LLAMACPP_N_CTX": {"path": "summarizer.llamacpp.n_ctx", "default": 8192, "type": int},
    "LLAMACPP_N_THREADS": {"path": "summarizer.llamacpp.n_threads", "default": 4, "type": int},
    "LLAMACPP_N_GPU_LAYERS": {"path": "summarizer.llamacpp.n_gpu_layers", "default": 0, "type": int},
    "LLAMACPP_VERBOSE": {"path": "summarizer.llamacpp.verbose", "default": False, "type": bool},
    "LLAMACPP_MODEL_PATH": {"path": "summarizer.llamacpp.model_path", "default": "", "type": str},
    "LLAMACPP_REPO_ID": {"path": "summarizer.llamacpp.repo_id", "default": "", "type": str},
    "LLAMACPP_FILENAME": {"path": "summarizer.llamacpp.filename", "default": "", "type": str},

    # Parser
    "EXTRACTION_STRATEGY": {"path": "parser.extraction_strategy", "default": "spotlight_first", "type": str},
    "SPOTLIGHT_ENABLED": {"path": "parser.spotlight_enabled", "default": True, "type": bool},
    "SPOTLIGHT_TIMEOUT_SECONDS": {"path": "parser.spotlight_timeout_seconds", "default": 5, "type": int},
    "WHISPER_PROVIDER": {"path": "parser.whisper.provider", "default": "online", "type": str},
    "ONLINE_WHISPER_MODEL": {"path": "parser.whisper.online_model", "default": "whisper-1", "type": str},
    "LOCAL_WHISPER_MODEL": {"path": "parser.whisper.local_model", "default": "turbo", "type": str},
}

# Reverse mapping: TOML path -> config key
_PATH_TO_KEY = {v["path"]: k for k, v in SETTINGS_SCHEMA.items()}


def resolve_key(key: str) -> str | None:
    """Resolve a key to its flat config key name.

    Accepts flat config keys (e.g. SPOTLIGHT_ENABLED),
    TOML paths (e.g. parser.spotlight_enabled), or
    case-insensitive variants of either.

    Returns the canonical flat config key, or None if not found.
    """
    # Exact flat key match
    upper = key.upper()
    if upper in SETTINGS_SCHEMA:
        return upper
    # TOML path match
    if key in _PATH_TO_KEY:
        return _PATH_TO_KEY[key]
    # Case-insensitive TOML path match
    lower = key.lower()
    for path, flat_key in _PATH_TO_KEY.items():
        if path.lower() == lower:
            return flat_key
    return None


def _coerce(value: Any, target_type: type) -> Any:
    """Coerce a value to the target type."""
    if isinstance(value, target_type):
        return value
    if target_type is bool:
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes")
        return bool(value)
    if target_type is int:
        return int(value)
    if target_type is str:
        return str(value)
    return value


def _get_nested(data: dict, path: str, default: Any = None) -> Any:
    """Get a value from a nested dict using a dotted path."""
    keys = path.split(".")
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _set_nested(data: dict, path: str, value: Any) -> None:
    """Set a value in a nested dict using a dotted path."""
    keys = path.split(".")
    current = data
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


class SettingsManager:
    """Manages persistent settings in a TOML file."""

    def __init__(self, dirs: PlatformDirs):
        self._dirs = dirs
        self._settings: dict[str, Any] = {}
        self._toml_path = Path(dirs.user_config_dir) / SETTINGS_FILE

    @property
    def toml_path(self) -> Path:
        return self._toml_path

    def load(self) -> dict[str, Any]:
        """Load settings: defaults -> TOML -> env vars. Returns flat dict of config keys."""
        # Start with defaults
        for key, schema in SETTINGS_SCHEMA.items():
            self._settings[key] = schema["default"]

        # Overlay TOML file values
        if self._toml_path.exists():
            try:
                with open(self._toml_path, "rb") as f:
                    toml_data = tomllib.load(f)
                for key, schema in SETTINGS_SCHEMA.items():
                    toml_value = _get_nested(toml_data, schema["path"])
                    if toml_value is not None:
                        self._settings[key] = _coerce(toml_value, schema["type"])
                logger.info("Loaded settings from TOML", path=str(self._toml_path))
            except Exception as e:
                logger.warning("Failed to load settings TOML, using defaults", path=str(self._toml_path), error=str(e))
        else:
            logger.info("No settings file found, will create with defaults", path=str(self._toml_path))
            self.save()

        # Overlay env vars (COSMA_ prefix, env wins over TOML)
        for key, schema in SETTINGS_SCHEMA.items():
            env_key = f"COSMA_{key}"
            env_value = os.environ.get(env_key)
            if env_value is not None:
                self._settings[key] = _coerce(env_value, schema["type"])

        return dict(self._settings)

    def save(self) -> None:
        """Write current settings to TOML file."""
        toml_data: dict[str, Any] = {}
        for key, schema in SETTINGS_SCHEMA.items():
            value = self._settings.get(key, schema["default"])
            # Skip empty string defaults to keep TOML clean
            if isinstance(value, str) and value == "" and schema["default"] == "":
                continue
            _set_nested(toml_data, schema["path"], value)

        self._toml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._toml_path, "wb") as f:
            tomli_w.dump(toml_data, f)
        logger.info("Saved settings to TOML", path=str(self._toml_path))

    def get(self, key: str) -> Any:
        """Get a setting value by flat config key."""
        if key in self._settings:
            return self._settings[key]
        if key in SETTINGS_SCHEMA:
            return SETTINGS_SCHEMA[key]["default"]
        raise KeyError(f"Unknown setting: {key}")

    def set(self, key: str, value: Any) -> None:
        """Set a setting value, coerce type, and save."""
        if key not in SETTINGS_SCHEMA:
            raise KeyError(f"Unknown setting: {key}")
        self._settings[key] = _coerce(value, SETTINGS_SCHEMA[key]["type"])
        self.save()

    def update(self, updates: dict[str, Any]) -> dict[str, Any]:
        """Bulk update settings. Accepts flat config keys. Returns updated settings."""
        for key, value in updates.items():
            if key not in SETTINGS_SCHEMA:
                raise KeyError(f"Unknown setting: {key}")
            self._settings[key] = _coerce(value, SETTINGS_SCHEMA[key]["type"])
        self.save()
        return dict(self._settings)

    def to_dict(self) -> dict[str, Any]:
        """Return all settings grouped by top-level TOML section."""
        grouped: dict[str, Any] = {}
        for key, schema in SETTINGS_SCHEMA.items():
            path = schema["path"]
            value = self._settings.get(key, schema["default"])
            _set_nested(grouped, path, value)
        return grouped

    def to_flat_dict(self) -> dict[str, Any]:
        """Return all settings as a flat dict of config keys."""
        result = {}
        for key in SETTINGS_SCHEMA:
            result[key] = self._settings.get(key, SETTINGS_SCHEMA[key]["default"])
        return result

    @staticmethod
    def defaults() -> dict[str, Any]:
        """Return default values grouped by section."""
        grouped: dict[str, Any] = {}
        for key, schema in SETTINGS_SCHEMA.items():
            _set_nested(grouped, schema["path"], schema["default"])
        return grouped

    @staticmethod
    def flat_defaults() -> dict[str, Any]:
        """Return default values as flat dict."""
        return {key: schema["default"] for key, schema in SETTINGS_SCHEMA.items()}
