"""
Local whisper.cpp transcription via the `pywhispercpp` Python binding.

Why pywhispercpp instead of shelling out to a `whisper` / `whisper-cli` binary:
- The previous shell-based code (see git history of media.py) broke because
  the expected binary name varies across installs (Homebrew: `whisper-cpp`,
  upstream: `whisper-cli` since 2024, older: `main`, and `openai-whisper`'s
  pip package also installs a `whisper` binary which is NOT whisper.cpp).
  We were invoking bare "whisper" and picking up whichever one happened to be
  first on $PATH — often the wrong one, often none at all.
- pywhispercpp bundles a prebuilt shared library, so no user has to install
  a binary. It also exposes a clean Python API, letting us keep transcription
  async-friendly by running it in a thread.
- Model files are downloaded once into ~/Library/Application Support/cosma/
  models/whisper/ via our shared paths helper, so they survive reinstalls
  and match where the bootstrap API reports them.

This module is kept under 200 lines intentionally: it has one job
(transcribe an audio file) and hides all pywhispercpp detail from media.py.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
from pathlib import Path
from typing import Optional

from cosma_backend.logging import get_logger
from cosma_backend.paths import ensure_models_dir

logger = get_logger(__name__)

# Default model — small, English-only, ~140 MB. Users can override via
# LOCAL_WHISPER_MODEL env var or the parser.whisper.local_model setting.
# We intentionally avoid "turbo" as the default because pywhispercpp's
# model registry name is "large-v3-turbo" (not "turbo"), and downloading
# ~1.5 GB on first use blocks setup for many minutes.
DEFAULT_MODEL = "base.en"

# Map names that were historically used in settings.toml to names
# pywhispercpp actually understands. Keeps users' existing config working.
_MODEL_ALIASES = {
    "turbo": "large-v3-turbo",
    "large": "large-v3",
}


def _resolve_model_name(requested: Optional[str]) -> str:
    name = (requested or os.getenv("LOCAL_WHISPER_MODEL") or DEFAULT_MODEL).strip()
    return _MODEL_ALIASES.get(name, name)


def is_available() -> bool:
    """True iff pywhispercpp importable. No subprocess, no model load."""
    return importlib.util.find_spec("pywhispercpp") is not None


def expected_model_file(model_name: str) -> Path:
    """Absolute path where pywhispercpp will (or did) store this model."""
    # pywhispercpp names files `ggml-<model>.bin` — match that so the
    # bootstrap "file exists?" check works without having to import the
    # library (cheap startup status).
    return ensure_models_dir("whisper") / f"ggml-{model_name}.bin"


_model_cache: dict[str, object] = {}


def _load_model_sync(model_name: str):
    """Load (and lazily download) a whisper.cpp model via pywhispercpp.

    Called from a thread — pywhispercpp's constructor blocks on download
    and on model load. We cache by name so repeated transcriptions reuse
    the same in-memory model.
    """
    if model_name in _model_cache:
        return _model_cache[model_name]

    from pywhispercpp.model import Model  # type: ignore[import-not-found]

    models_dir = ensure_models_dir("whisper")
    # pywhispercpp's Model accepts a model name (auto-download) or a path.
    # Passing `models_dir` overrides its default cache so everything lands
    # under our managed dir.
    model = Model(model_name, models_dir=str(models_dir), print_progress=False)
    _model_cache[model_name] = model
    return model


async def transcribe(audio_path: Path, model_name: Optional[str] = None) -> Optional[str]:
    """Transcribe an audio file. Returns text or None on failure.

    Runs the blocking whisper.cpp call in a thread so the event loop
    keeps serving HTTP requests during transcription.
    """
    if not is_available():
        logger.warning("pywhispercpp not installed; local whisper unavailable")
        return None

    resolved = _resolve_model_name(model_name)
    try:
        from cosma_backend.pipeline_executor import run_in_pipeline
        model = await run_in_pipeline(_load_model_sync, resolved)
        # `transcribe` returns a list of Segment objects with .text
        segments = await run_in_pipeline(model.transcribe, str(audio_path))
        text = " ".join(seg.text for seg in segments).strip()
        if not text:
            return None
        logger.info("pywhispercpp transcription completed",
                    model=resolved, length=len(text), path=str(audio_path))
        return text
    except Exception as e:
        logger.warning("pywhispercpp transcription failed",
                       model=resolved, error=str(e))
        return None


async def ensure_model_downloaded(
    model_name: Optional[str] = None,
    progress=None,
    stage_label: str = "whisper_model",
) -> tuple[Path, bool]:
    """Pre-download a whisper model with streamed progress.

    We bypass pywhispercpp's internal downloader for two reasons:
      1. It offers no byte-level progress callback — the setup wizard would
         show "Checking whisper model…" forever with no visible movement.
      2. Its download path uses a stdout-bound helper that breaks under
         Quart/hypercorn when the parent process's stdout is pipe-redirected
         (observed: "[Errno 32] Broken pipe" on first-run bootstrap).
    Instead we pull ggml-<name>.bin directly from ggerganov/whisper.cpp on
    HuggingFace using our shared download_hf_file helper, which means the
    same progress-polling path used for llama models drives the whisper row.
    """
    from cosma_backend.downloads import download_hf_file

    resolved = _resolve_model_name(model_name)
    dest = expected_model_file(resolved)
    if dest.exists() and dest.stat().st_size > 0:
        if progress:
            size = dest.stat().st_size
            progress(size, size, stage_label)
        return dest, True

    filename = f"ggml-{resolved}.bin"
    result = await asyncio.to_thread(
        download_hf_file,
        repo_id="ggerganov/whisper.cpp",
        filename=filename,
        kind="whisper",
        progress=progress,
        stage_label=stage_label,
    )
    return result.path, result.skipped
