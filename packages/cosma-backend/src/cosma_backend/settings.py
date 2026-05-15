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
from typing import Any, Callable, get_type_hints

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
    # OpenAI-compatible API endpoint. Default points at OpenAI's hosted
    # API; override to https://api.together.ai/v1, https://api.groq.com/openai/v1,
    # http://localhost:1234/v1 (LM Studio), etc. for compatible providers.
    # Empty string means "use litellm's default for the model's prefix"
    # (e.g. "openai/..." → api.openai.com).
    base_url: str = ""
    # API key for the configured base_url. When set, we install it into
    # OPENAI_API_KEY at summarizer init so litellm picks it up.
    # Empty string means "fall back to whatever's in the environment"
    # (the historical behavior — OPENAI_API_KEY exported before launch).
    api_key: str = ""


@dataclass
class LlamaCppConfig:
    context_length: int = 10000
    # 0 means "let the app pick at startup based on detected RAM" — see
    # cosma_backend.utils.hardware.choose_n_ctx_for_ram. Explicit values
    # in TOML still win. Previous hardcoded 16384 worked on a 16 GB Mac
    # but underused 32+ GB systems and risked OOM on smaller ones.
    n_ctx: int = 0
    n_threads: int = 4
    # n_gpu_layers = -1 offloads all transformer layers to Metal on Apple
    # Silicon. Override to 0 for CPU-only machines.
    n_gpu_layers: int = -1
    verbose: bool = False
    model_path: str = ""
    # Standardized on Qwen3-VL-2B-Instruct: visual-input multimodal, small,
    # and fast enough for interactive file summarization on M-series Macs.
    # Defaults here drive the bootstrap downloader.
    repo_id: str = "unsloth/Qwen3-VL-2B-Instruct-GGUF"
    # Q4_0 (vs Q4_K_M previously): on Apple Silicon, Q4_0's bandwidth
    # exactly matches the M-series memory subsystem — the recommended
    # quant for token generation on Metal. ~10–15% faster decode than
    # Q4_K_M with negligible quality loss for summarization. See
    # cosma/docs/STAGE_PIPELINING_DESIGN.md notes + the May-2026
    # llama.cpp guidance threads.
    filename: str = "*Q4_0.gguf"
    clip_model_path: str = ""
    clip_repo_id: str = "unsloth/Qwen3-VL-2B-Instruct-GGUF"
    clip_filename: str = "mmproj-F16.gguf"
    chat_handler: str = "qwen3-vl"
    enable_thinking: bool = False
    image_min_tokens: int = 1024
    # HF repo for the tokenizer used when counting/chunking tokens. The GGUF
    # repo itself usually has no `config.json` / `tokenizer.json` layout that
    # `AutoTokenizer` accepts, so we point at the upstream un-quantized repo.
    # Setting this to the model's real tokenizer prevents the tiktoken
    # cl100k_base fallback, which miscounts Qwen tokens and causes
    # "token out of bounds" failures at llama.cpp prefill when a chunk
    # actually exceeds n_ctx.
    tokenizer_repo: str = "Qwen/Qwen3-VL-2B-Instruct"
    # Apple-Silicon LLM perf flags. Defaults reflect what we've
    # validated against the cosmasense llama-cpp-python build. Users
    # on a different build can flip these off via settings.toml if
    # something incompatible surfaces.
    flash_attn: bool = True
    kv_cache_quant: bool = True
    prompt_cache_enabled: bool = True
    prompt_cache_capacity_mib: int = 200


@dataclass
class SummarizerConfig:
    # User-selected backend: "llamacpp" (default — fully local, self-contained),
    # "ollama" (local, external daemon), or "online" (OpenAI-style API).
    # "auto" is retained as an explicit opt-in for users who want legacy
    # fallback behavior, but is no longer the default because silent provider
    # switching made failures hard to diagnose (see the original
    # "All AI summarizers failed or are unavailable" report).
    provider: str = "llamacpp"
    max_tokens_per_request: int = 100000
    chunk_overlap_tokens: int = 1000
    max_chunks: int = 10
    idle_unload_seconds: int = 60
    # "Fast mode": cap every file at exactly one chunk's worth of
    # summarize work. Trades coverage for throughput — useful for
    # demos, large-corpus first passes, or low-end Macs where the
    # full multi-chunk path is too slow. The first chunk usually
    # carries the document's first ~6k tokens (with the configured
    # n_ctx) which is enough to produce a reasonable title +
    # summary + keywords for most files. Long docs end up with a
    # head-only summary; the partial-coverage note from the budget
    # logic still applies.
    fast_mode: bool = False
    # Per-file wall-clock budget for the summarize stage. Once we
    # exceed this, the summarizer stops dispatching new chunks and
    # finalizes with whatever chunk summaries it already has — the
    # file ends as COMPLETE with a partial-coverage flag instead of
    # FAILED on a long PDF that never finishes. 0 = no budget (legacy).
    # Default 60 s comes from the user-experience math: a single 25 s
    # mean / 7 s p99 chunk usually fits 1-2 chunks; 60 s comfortably
    # covers the typical 1–3 chunk file but caps the long-tail.
    summarize_budget_seconds: float = 60.0
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    online: OnlineConfig = field(default_factory=OnlineConfig)
    llamacpp: LlamaCppConfig = field(default_factory=LlamaCppConfig)


@dataclass
class EmbedderConfig:
    # "local" eagerly loads + warms the SentenceTransformer at backend
    # startup so the user's first search returns in normal latency
    # instead of paying a 5-15 s cold encode. "lazy_local" defers the
    # load to first use (lower startup CPU but slower first query).
    # "online" needs an API key; falls back to local on failure.
    provider: str = "local"
    model: str = "text-embedding-3-small"
    dimensions: int = 512
    local_model: str = "intfloat/e5-base-v2"
    local_dimensions: int = 768


@dataclass
class WhisperConfig:
    # "local" (default — pywhispercpp bundled, no keys needed) or "online"
    # (OpenAI whisper-1, requires OPENAI_API_KEY). Previous default of
    # "online" meant .mov files silently failed for every user without a
    # key — hence the new local-first default.
    provider: str = "local"
    online_model: str = "whisper-1"
    # "base" = multilingual whisper, ~140 MB first-run download. Multilingual
    # by default because a real Downloads folder has plenty of non-English
    # audio; "base.en" would transcribe that to garbage. Switch to "base.en"
    # for English-only precision, or "small"/"large-v3-turbo" for higher
    # accuracy (turbo is a ~1.5 GB download).
    local_model: str = "base"


@dataclass
class ParserConfig:
    extraction_strategy: str = "spotlight_first"
    spotlight_enabled: bool = True
    spotlight_timeout_seconds: int = 5
    # Hard cap on file size before extraction even starts. Set generously
    # (20 GB) because the per-extractor pipelines stream rather than load
    # whole files: ffmpeg streams audio/video, MarkItDown / PDF readers
    # operate page-by-page, and Whisper consumes audio in chunks. The cap
    # exists only to avoid pathological inputs (corrupted multi-TB files,
    # disk images mistakenly inside a watched folder).
    max_file_size_mb: int = 20480
    whisper: WhisperConfig = field(default_factory=WhisperConfig)


@dataclass
class SchedulerRuleConfig:
    rule: str = ""
    operator: str = ""
    value: Any = None
    enabled: bool = True


@dataclass
class SchedulerConfig:
    enabled: bool = False
    combine_mode: str = "ALL"
    # The scheduler primarily evaluates rules BETWEEN tasks (see
    # IndexingQueue._pre_task_hook). This interval is only used as a backstop
    # for long-term conditions (e.g. time_window) that can change while the
    # queue is idle. Short intervals cause unnecessary start/pause thrashing
    # and mid-task metric polling, so keep it coarse.
    check_interval_seconds: int = 60
    rules: list[SchedulerRuleConfig] = field(default_factory=list)


# Registry describing each scheduler rule type's expected inputs.
# The frontend uses this to render type-specific controls (toggles, sliders, time pickers).
SCHEDULER_RULE_TYPES: dict[str, dict[str, Any]] = {
    "power_source": {
        "label": "Power Source",
        "description": "Require AC power (plugged in)",
        "value_type": "boolean",
        "default_operator": "eq",
        "boolean_labels": {"true": "Plugged in", "false": "On battery"},
    },
    "battery_level": {
        "label": "Battery Level",
        "description": "Minimum battery percentage required",
        "value_type": "percentage",
        "unit": "%",
        "default_operator": "gte",
        "min": 0,
        "max": 100,
    },
    "gpu_usage": {
        "label": "GPU Usage",
        "description": "Maximum GPU utilization allowed",
        "value_type": "percentage",
        "unit": "%",
        "default_operator": "lte",
        "min": 0,
        "max": 100,
    },
    "memory_usage": {
        "label": "Memory Usage",
        "description": "Maximum memory utilization allowed",
        "value_type": "percentage",
        "unit": "%",
        "default_operator": "lte",
        "min": 0,
        "max": 100,
    },
    "memory_pressure": {
        "label": "Memory Pressure",
        "description": "Maximum memory pressure allowed (active+wired %)",
        "value_type": "percentage",
        "unit": "%",
        "default_operator": "lte",
        "min": 0,
        "max": 100,
    },
    "cpu_temperature": {
        "label": "CPU Temperature",
        "description": "Maximum CPU temperature allowed",
        "value_type": "number",
        "unit": "\u00b0C",
        "default_operator": "lte",
        "min": 0,
        "max": 120,
    },
    "fan_speed": {
        "label": "Fan Speed",
        "description": "Maximum fan speed allowed",
        "value_type": "number",
        "unit": "RPM",
        "default_operator": "lte",
        "min": 0,
        "max": 10000,
    },
    "cpu_idle": {
        "label": "CPU Idle",
        "description": "Require CPU to be idle (low usage)",
        "value_type": "boolean",
        "default_operator": "eq",
        "boolean_labels": {"true": "Idle", "false": "Busy"},
    },
    "low_power_mode": {
        "label": "Low Power Mode",
        "description": "Pause when macOS Low Power Mode is active",
        "value_type": "boolean",
        "default_operator": "eq",
        "boolean_labels": {"true": "Active", "false": "Inactive"},
    },
    "time_window": {
        "label": "Time Window",
        "description": "Only process during this time range",
        "value_type": "time_range",
    },
    "queue_size": {
        "label": "Queue Size",
        "description": "Minimum items in queue before processing starts",
        "value_type": "number",
        "default_operator": "gte",
        "min": 0,
    },
}


@dataclass
class QueueConfig:
    cooldown_seconds: int = 60
    initial_cooldown_seconds: int = 5
    # Legacy global cap on total in-flight files. Kept for backward
    # compatibility with existing settings.toml files but NO LONGER
    # CONSULTED at runtime — the effective ceiling is derived from
    # the per-stage caps via `effective_max_concurrency` below. The
    # old behavior was a footgun: shipping `max_concurrency = 2` (an
    # early conservative default) silently throttled parse even
    # though the per-stage `parse_concurrency = 4` looked like it
    # should overlap. New code always uses
    # `parse + summarize + embed` so users only have ONE knob to
    # think about (parse_concurrency).
    max_concurrency: int = 6
    max_retries: int = 3
    file_processing_timeout: int = 300  # seconds per file (5 min default)
    gpu_memory_cap: float = 0.75  # fraction of GPU memory to use (0.0-1.0, default 75%)
    # Per-stage concurrency caps. Each Pipeline stage holds its own
    # asyncio.Semaphore at this size, so files in different stages run
    # in parallel (parse on CPU, summarize on Metal, embed on MPS)
    # without fighting for the same hardware. Tune based on your box:
    #   - parse: I/O + CPU bound; safe to overlap heavily.
    #   - summarize: single Metal LLM context; keep at 1.
    #   - embed: single MPS device; keep at 1 unless you have multiple GPUs.
    # See cosma/docs/STAGE_PIPELINING_DESIGN.md.
    parse_concurrency: int = 4
    summarize_concurrency: int = 1
    embed_concurrency: int = 1
    # Hold-off after model load before indexing starts, so a user search
    # at app launch has the GPU/CPU to itself. See app.py Phase 2.
    indexing_start_grace_seconds: float = 5.0
    # When a search arrives, pause indexing dispatch for this many seconds
    # so the embedder/LLM can serve the query immediately. New searches
    # within the window extend the pause. See queue/indexing_queue.py.
    search_preempt_seconds: float = 10.0

    @property
    def effective_max_concurrency(self) -> int:
        """The actual ceiling on total files in-flight across the
        pipeline. Derived from the per-stage caps so users don't need
        to keep two settings consistent — the single knob that
        actually matters for throughput is `parse_concurrency`
        (summarize and embed are hardware-locked at 1 by Metal /
        single-MPS-device constraints).
        """
        return self.parse_concurrency + self.summarize_concurrency + self.embed_concurrency


@dataclass
class Settings:
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    summarizer: SummarizerConfig = field(default_factory=SummarizerConfig)
    parser: ParserConfig = field(default_factory=ParserConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce(value: Any, target_type: type) -> Any:
    """Coerce a value to the target type."""
    # Handle Any type - return value as-is
    if target_type is Any:
        return value
    # Try isinstance check, but handle types that don't support it
    try:
        if isinstance(value, target_type):
            return value
    except TypeError:
        # Some types (like Any) cannot be used with isinstance()
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


def _is_dataclass_type(t: Any) -> bool:
    """Check if a type is a dataclass, safely handling Any and other special types."""
    try:
        return dataclasses.is_dataclass(t) and isinstance(t, type)
    except TypeError:
        # Some types like Any cannot be used with isinstance()
        return False


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Recursively build a dataclass instance from a (possibly partial) dict."""
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        raw = data[f.name]
        field_type = hints[f.name]
        # Handle list[SomeDataclass] types
        origin = getattr(field_type, "__origin__", None)
        if origin is list and isinstance(raw, list):
            args = getattr(field_type, "__args__", ())
            if args and _is_dataclass_type(args[0]):
                kwargs[f.name] = [_from_dict(args[0], item) if isinstance(item, dict) else item for item in raw]
            else:
                kwargs[f.name] = raw
        elif _is_dataclass_type(field_type) and isinstance(raw, dict):
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


_VALIDATIONS: dict[tuple[str, str], tuple[Callable[[Any], bool], str]] = {
    ("queue", "cooldown_seconds"): (lambda v: v >= 1, "cooldown_seconds must be >= 1"),
    ("queue", "initial_cooldown_seconds"): (lambda v: v >= 0, "initial_cooldown_seconds must be >= 0"),
    ("queue", "max_concurrency"): (lambda v: v >= 1, "max_concurrency must be >= 1"),
    ("queue", "max_retries"): (lambda v: v >= 0, "max_retries must be >= 0"),
    ("scheduler", "check_interval_seconds"): (lambda v: v >= 5, "check_interval_seconds must be >= 5"),
}


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

    # Handle list[SomeDataclass] types - convert dicts to dataclass instances
    origin = getattr(target_type, "__origin__", None)
    if origin is list and isinstance(value, list):
        args = getattr(target_type, "__args__", ())
        if args and _is_dataclass_type(args[0]):
            coerced = [_from_dict(args[0], item) if isinstance(item, dict) else item for item in value]
        else:
            coerced = value
    else:
        coerced = _coerce(value, target_type)

    # Validate if a rule exists for this (parent_name, leaf) pair
    parent_name = type(current).__name__.replace("Config", "").lower()
    validation_key = (parent_name, leaf)
    if validation_key in _VALIDATIONS:
        check_fn, msg = _VALIDATIONS[validation_key]
        if not check_fn(coerced):
            raise ValueError(msg)

    setattr(current, leaf, coerced)


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
                # Apply forward migrations for fields whose defaults moved.
                # Persists immediately so the next launch reads the new value.
                if self._migrate_legacy_defaults():
                    self.save()
            except Exception as e:
                logger.warning("Failed to load settings TOML, using defaults",
                               path=str(self._toml_path), error=str(e))
        else:
            logger.info("No settings file found, will create with defaults",
                        path=str(self._toml_path))
            self.save()

        return self._settings

    def _migrate_legacy_defaults(self) -> bool:
        """Bump fields stuck on a previous release's default value.

        Returns True if anything was migrated (caller should re-save).

        Why this exists: Python dataclass defaults only apply when the
        field is *missing* from the TOML. Once a value has been written
        — including writes that just round-tripped the previous default
        — bumping the dataclass default in code has no effect on
        existing installs. So users who had `max_file_size_mb = 200`
        from the v0.8.x default kept failing 200 MB+ files even after
        the project default became 20 GB. Each entry here is a one-time
        fix-up that only fires when the user's saved value matches the
        previous release's default exactly; explicit user choices
        (anything other than the previous default) are preserved.
        """
        migrated = False
        # parser.max_file_size_mb: 200 → 20480 (v0.8.4)
        if self._settings.parser.max_file_size_mb == 200:
            logger.info("Migrating parser.max_file_size_mb from legacy default 200 to 20480 MB")
            self._settings.parser.max_file_size_mb = 20480
            migrated = True
        return migrated

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
