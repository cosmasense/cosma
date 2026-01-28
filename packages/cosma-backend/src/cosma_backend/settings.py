"""
Settings Manager Module

Manages persistent application settings stored in a TOML file.
Settings are represented as typed dataclasses with attribute access.
"""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, get_type_hints

from platformdirs import PlatformDirs

from cosma_backend.logging import get_logger

logger = get_logger(__name__)

import tomli_w

SETTINGS_FILE = "settings.toml"


# ---------------------------------------------------------------------------
# Dataclass hierarchy
# ---------------------------------------------------------------------------

@dataclass
class OllamaConfig:
    model: str = "qwen3-vl:2b-instruct"
    host: str = "http://localhost:11434"
    context_length: int = 10000


@dataclass
class OnlineConfig:
    model: str = "openai/gpt-4.1-nano-2025-04-14"
    context_length: int = 128000


@dataclass
class LlamaCppConfig:
    context_length: int = 8192
    n_ctx: int = 8192
    n_threads: int = 4
    n_gpu_layers: int = 0
    verbose: bool = False
    model_path: str = ""
    repo_id: str = ""
    filename: str = ""


@dataclass
class SummarizerConfig:
    provider: str = "auto"
    max_tokens_per_request: int = 100000
    chunk_overlap_tokens: int = 1000
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    online: OnlineConfig = field(default_factory=OnlineConfig)
    llamacpp: LlamaCppConfig = field(default_factory=LlamaCppConfig)


@dataclass
class EmbedderConfig:
    provider: str = "local"
    model: str = "text-embedding-3-small"
    dimensions: int = 512
    local_model: str = "intfloat/e5-base-v2"
    local_dimensions: int = 768


@dataclass
class WhisperConfig:
    provider: str = "online"
    online_model: str = "whisper-1"
    local_model: str = "turbo"


@dataclass
class ParserConfig:
    extraction_strategy: str = "spotlight_first"
    spotlight_enabled: bool = True
    spotlight_timeout_seconds: int = 5
    whisper: WhisperConfig = field(default_factory=WhisperConfig)


@dataclass
class Settings:
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    summarizer: SummarizerConfig = field(default_factory=SummarizerConfig)
    parser: ParserConfig = field(default_factory=ParserConfig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Recursively build a dataclass instance from a (possibly partial) dict."""
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        raw = data[f.name]
        field_type = hints[f.name]
        if dataclasses.is_dataclass(field_type) and isinstance(raw, dict):
            kwargs[f.name] = _from_dict(field_type, raw)
        else:
            kwargs[f.name] = _coerce(raw, field_type)
    return cls(**kwargs)


def _get_by_path(obj: Any, path: str) -> Any:
    """Walk a dotted path on a dataclass tree, returning the value."""
    parts = path.split(".")
    current = obj
    for part in parts:
        if not dataclasses.is_dataclass(current):
            raise KeyError(f"Cannot traverse into non-dataclass at '{part}'")
        if not hasattr(current, part):
            raise KeyError(f"Unknown field: '{part}'")
        current = getattr(current, part)
    return current


def _set_by_path(obj: Any, path: str, value: Any) -> None:
    """Walk a dotted path on a dataclass tree and set the leaf value with type coercion."""
    parts = path.split(".")
    current = obj
    for part in parts[:-1]:
        if not dataclasses.is_dataclass(current):
            raise KeyError(f"Cannot traverse into non-dataclass at '{part}'")
        if not hasattr(current, part):
            raise KeyError(f"Unknown field: '{part}'")
        current = getattr(current, part)

    leaf = parts[-1]
    if not hasattr(current, leaf):
        raise KeyError(f"Unknown setting: '{path}'")

    hints = get_type_hints(type(current))
    target_type = hints[leaf]
    setattr(current, leaf, _coerce(value, target_type))


# ---------------------------------------------------------------------------
# SettingsManager
# ---------------------------------------------------------------------------

class SettingsManager:
    """Manages persistent settings backed by a TOML file."""

    def __init__(self, dirs: PlatformDirs):
        self._dirs = dirs
        self._settings = Settings()
        self._toml_path = Path(dirs.user_config_dir) / SETTINGS_FILE

    @property
    def toml_path(self) -> Path:
        return self._toml_path

    @property
    def settings(self) -> Settings:
        return self._settings

    def load(self) -> Settings:
        """Load settings from TOML file, falling back to defaults."""
        self._settings = Settings()  # start from defaults

        if self._toml_path.exists():
            try:
                with open(self._toml_path, "rb") as f:
                    toml_data = tomllib.load(f)
                self._settings = _from_dict(Settings, toml_data)
                logger.info("Loaded settings from TOML", path=str(self._toml_path))
            except Exception as e:
                logger.warning("Failed to load settings TOML, using defaults",
                               path=str(self._toml_path), error=str(e))
        else:
            logger.info("No settings file found, will create with defaults",
                        path=str(self._toml_path))
            self.save()

        return self._settings

    def save(self) -> None:
        """Write current settings to TOML file."""
        data = dataclasses.asdict(self._settings)
        # Remove empty-string fields to keep TOML clean
        _strip_empty_strings(data)
        self._toml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._toml_path, "wb") as f:
            tomli_w.dump(data, f)
        logger.info("Saved settings to TOML", path=str(self._toml_path))

    def to_dict(self) -> dict[str, Any]:
        """Return settings as a nested dict (same structure as TOML)."""
        return dataclasses.asdict(self._settings)

    def get_by_path(self, path: str) -> Any:
        """Get a setting value by dotted path (e.g. 'summarizer.provider')."""
        return _get_by_path(self._settings, path)

    def set_by_path(self, path: str, value: Any) -> None:
        """Set a setting value by dotted path with type coercion, then save."""
        _set_by_path(self._settings, path, value)
        self.save()

    def update(self, updates: dict[str, Any]) -> dict[str, Any]:
        """Bulk update settings. Accepts dotted paths. Returns updated dict."""
        for path, value in updates.items():
            _set_by_path(self._settings, path, value)
        self.save()
        return self.to_dict()

    @staticmethod
    def defaults() -> dict[str, Any]:
        """Return default values as a nested dict."""
        return dataclasses.asdict(Settings())


def _strip_empty_strings(d: dict) -> None:
    """Recursively remove keys whose value is an empty string."""
    keys_to_remove = []
    for k, v in d.items():
        if isinstance(v, dict):
            _strip_empty_strings(v)
        elif isinstance(v, str) and v == "":
            keys_to_remove.append(k)
    for k in keys_to_remove:
        del d[k]
