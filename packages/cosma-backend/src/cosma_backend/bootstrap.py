"""
First-run bootstrap: probe local-AI availability, download missing models.

Why this lives next to app.py (not under api/):
- The logic is used both by the HTTP endpoints (for the setup wizard) AND
  potentially at backend startup (to pre-warm models without forcing the
  user through a wizard). Splitting "what's the state?" and "what to
  download?" from "how do we expose it over HTTP?" keeps each layer small.

Idempotency is the load-bearing property of this module. Every probe/install
step checks first and short-circuits on hit. That's how the setup wizard
can be re-entered after a crash, or skipped entirely on a warm reinstall,
without re-downloading multi-GB model files.

Progress reporting uses a simple shared queue model rather than a pub/sub
hub — there's at most one active install at a time (enforced by a lock),
so a single asyncio.Queue per client suffices.
"""

from __future__ import annotations

import asyncio
import importlib.util
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from cosma_backend.downloads import download_hf_file
from cosma_backend.logging import get_logger
from cosma_backend.paths import (
    ensure_models_dir,
    llama_models_dir,
    whisper_models_dir,
)
from cosma_backend.parser import whisper_local
from cosma_backend.settings import LlamaCppConfig, WhisperConfig

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Status model
# ---------------------------------------------------------------------------

@dataclass
class ComponentStatus:
    """One row of the bootstrap status table."""

    name: str
    present: bool
    path: Optional[str] = None
    detail: Optional[str] = None  # human-readable extra context

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BootstrapStatus:
    ready: bool
    components: list[ComponentStatus] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "components": [c.to_dict() for c in self.components],
        }


def _module_present(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _first_matching_file(directory: Path, pattern: str) -> Optional[Path]:
    """Glob match in a local dir — used to detect already-downloaded models
    without hitting the network. HF stores files under the concrete resolved
    name, which we may not know at probe time, so we glob."""
    if not directory.exists():
        return None
    matches = list(directory.glob(pattern))
    return matches[0] if matches else None


def _required_for(summ_provider: str, whisper_provider: str) -> set[str]:
    """Which component names are *required* for the chosen providers.

    The full component table is always reported (so the UI can show every
    box), but only the required ones count toward the `ready` flag. That
    means a user who picked "ollama" doesn't see the app get stuck waiting
    on llama.cpp downloads they never asked for.
    """
    required: set[str] = set()
    if summ_provider == "llamacpp":
        required.update({"llama_cpp_python", "llama_model", "llama_mmproj"})
    elif summ_provider == "ollama":
        required.update({"ollama_running", "ollama_model"})
    elif summ_provider == "online":
        required.update({"openai_api_key"})
    # "auto" (legacy) falls back, so we don't gate on any single component.

    if whisper_provider == "local":
        required.update({"whisper_cpp", "whisper_model"})
    elif whisper_provider == "online":
        required.update({"openai_api_key"})
    return required


async def get_status(
    llama_cfg: Optional[LlamaCppConfig] = None,
    whisper_cfg: Optional[WhisperConfig] = None,
    summ_provider: str = "llamacpp",
) -> BootstrapStatus:
    """Quick probe — no network, no model loading.

    `ready` is computed only over components required for the chosen
    providers (see _required_for). Components outside that set are still
    reported so Settings can show "Ollama is also installed, by the way",
    but their absence doesn't block the app.
    """
    llama_cfg = llama_cfg or LlamaCppConfig()
    whisper_cfg = whisper_cfg or WhisperConfig()

    comps: list[ComponentStatus] = []

    comps.append(ComponentStatus(
        name="llama_cpp_python",
        present=_module_present("llama_cpp"),
        detail="Python binding for llama.cpp (core summarizer)",
    ))

    # Look in the gguf/ subdir first (new layout); fall back to root of
    # llama/ for any user whose old bootstrap left files there.
    gguf = (_first_matching_file(llama_models_dir() / "gguf", llama_cfg.filename or "*.gguf")
            or _first_matching_file(llama_models_dir(), llama_cfg.filename or "*.gguf"))
    comps.append(ComponentStatus(
        name="llama_model",
        present=gguf is not None,
        path=str(gguf) if gguf else None,
        detail=f"{llama_cfg.repo_id} / {llama_cfg.filename}",
    ))

    mmproj = (_first_matching_file(llama_models_dir() / "mmproj", llama_cfg.clip_filename or "mmproj*.gguf")
              or _first_matching_file(llama_models_dir(), llama_cfg.clip_filename or "mmproj*.gguf"))
    comps.append(ComponentStatus(
        name="llama_mmproj",
        present=mmproj is not None,
        path=str(mmproj) if mmproj else None,
        detail="Vision projector; without it, PNG summarization falls back to text-only",
    ))

    comps.append(ComponentStatus(
        name="whisper_cpp",
        present=whisper_local.is_available(),
        detail="pywhispercpp bundled binding",
    ))

    ggml = whisper_local.expected_model_file(
        whisper_local._resolve_model_name(whisper_cfg.local_model)
    )
    comps.append(ComponentStatus(
        name="whisper_model",
        present=ggml.exists() and ggml.stat().st_size > 0 if ggml.exists() else False,
        path=str(ggml) if ggml.exists() else None,
        detail=f"model={whisper_local._resolve_model_name(whisper_cfg.local_model)}",
    ))

    required = _required_for(summ_provider, whisper_cfg.provider)
    # ready = every *required* component present. If the user picked a
    # provider we don't have a probe for (e.g. "ollama_running"), it's
    # treated as missing — safer to prompt than to silently proceed.
    if required:
        by_name = {c.name: c for c in comps}
        ready = all(by_name.get(r, ComponentStatus(name=r, present=False)).present
                    for r in required)
    else:
        # "auto" / unknown provider — fall back to legacy "all present" check.
        ready = all(c.present for c in comps)
    return BootstrapStatus(ready=ready, components=comps)


# ---------------------------------------------------------------------------
# Install orchestration with progress events
# ---------------------------------------------------------------------------

@dataclass
class ProgressEvent:
    stage: str            # e.g. "llama_model", "whisper_model"
    message: str          # human-readable line
    bytes_done: int = 0
    bytes_total: int = 0
    done: bool = False    # True once this stage is complete
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BootstrapRunner:
    """Single-flight install coordinator.

    Only one install runs at a time; concurrent /install callers attach to
    the same progress stream. Keeping this stateful (vs. spawning a new
    download per request) means users opening the wizard on two windows
    see consistent progress and we don't accidentally download twice.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._subscribers: list[asyncio.Queue[ProgressEvent]] = []
        self._running = False
        self._last_events: list[ProgressEvent] = []  # replay for late subscribers

    def subscribe(self) -> asyncio.Queue[ProgressEvent]:
        q: asyncio.Queue[ProgressEvent] = asyncio.Queue()
        # Replay prior events so a client that connects mid-run sees context.
        for evt in self._last_events:
            q.put_nowait(evt)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[ProgressEvent]) -> None:
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass

    def _emit(self, evt: ProgressEvent) -> None:
        self._last_events.append(evt)
        for q in list(self._subscribers):
            try:
                q.put_nowait(evt)
            except Exception:
                pass

    @property
    def is_running(self) -> bool:
        return self._running

    def broadcast_shutdown(self) -> None:
        """Wake every subscriber with a terminal "complete" event so the SSE
        generators return cleanly instead of blocking on queue.get() past
        the uvicorn graceful-shutdown window."""
        evt = ProgressEvent(stage="complete", message="server shutting down", done=True)
        self._emit(evt)

    async def run(
        self,
        llama_cfg: Optional[LlamaCppConfig] = None,
        whisper_cfg: Optional[WhisperConfig] = None,
        summ_provider: str = "llamacpp",
    ) -> None:
        """Install components required for the chosen provider. Idempotent.

        Only downloads what the user's provider choice actually needs:
          - "llamacpp" → Qwen3-VL GGUF + mmproj (in parallel)
          - "ollama"   → no-op on our side (CosmaManager handles Ollama
                         install + `ollama pull` on the frontend)
          - "online"   → no-op (just needs an API key)
        Whisper is downloaded only when the whisper provider is "local".

        Single-flight: a second concurrent run is dropped — callers should
        subscribe to /events instead of re-POSTing /install.
        """
        if self._lock.locked():
            return
        async with self._lock:
            self._running = True
            self._last_events.clear()
            try:
                tasks: list = []

                if summ_provider == "llamacpp":
                    tasks.append(self._install_llama_gguf(llama_cfg or LlamaCppConfig()))
                    if (llama_cfg or LlamaCppConfig()).clip_repo_id:
                        tasks.append(self._install_llama_mmproj(llama_cfg or LlamaCppConfig()))
                elif summ_provider == "ollama":
                    self._emit(ProgressEvent(
                        stage="ollama_model",
                        message="Ollama models are managed by the app (ollama pull)",
                        done=True,
                    ))
                elif summ_provider == "online":
                    self._emit(ProgressEvent(
                        stage="openai_api_key",
                        message="Online provider uses an API key; nothing to download",
                        done=True,
                    ))

                wcfg = whisper_cfg or WhisperConfig()
                if wcfg.provider == "local":
                    tasks.append(self._install_whisper(wcfg))

                # Parallel downloads. gather with return_exceptions so one
                # failure doesn't cancel the others — we want partial
                # success to be visible in per-stage events rather than
                # wiped by a sibling's exception.
                results = await asyncio.gather(*tasks, return_exceptions=True)
                errors = [r for r in results if isinstance(r, Exception)]

                if errors:
                    # Fail-soft: report the first error as the run-level
                    # error, but emit complete so the UI still unsticks
                    # from "running". The status endpoint's `ready` flag
                    # will reflect which required components are still
                    # missing, and the frontend decides whether to let
                    # the user proceed.
                    msg = "; ".join(str(e) for e in errors[:3])
                    self._emit(ProgressEvent(
                        stage="error", message=msg, error=msg, done=True,
                    ))
                else:
                    self._emit(ProgressEvent(stage="complete", message="ready", done=True))
            except Exception as e:
                logger.exception("bootstrap install failed")
                self._emit(ProgressEvent(
                    stage="error", message=str(e), error=str(e), done=True,
                ))
            finally:
                self._running = False

    # -- Stages --------------------------------------------------------------
    # Each stage is its own coroutine so run() can gather them in parallel.

    async def _install_llama_gguf(self, cfg: LlamaCppConfig) -> None:
        # subdir="gguf" so the parallel mmproj download doesn't share a
        # cache dir with us (see downloads.py for why that matters).
        ensure_models_dir("llama")
        self._emit(ProgressEvent(stage="llama_model", message="Checking Qwen3-VL GGUF…"))
        result = await asyncio.to_thread(
            download_hf_file,
            repo_id=cfg.repo_id,
            filename=cfg.filename,
            kind="llama",
            subdir="gguf",
            progress=self._progress_cb("llama_model"),
            stage_label="llama_model",
        )
        self._emit(ProgressEvent(
            stage="llama_model",
            message="skipped (already present)" if result.skipped else "downloaded",
            done=True,
        ))

    async def _install_llama_mmproj(self, cfg: LlamaCppConfig) -> None:
        ensure_models_dir("llama")
        self._emit(ProgressEvent(stage="llama_mmproj", message="Checking mmproj projector…"))
        result = await asyncio.to_thread(
            download_hf_file,
            repo_id=cfg.clip_repo_id,
            filename=cfg.clip_filename,
            kind="llama",
            subdir="mmproj",
            progress=self._progress_cb("llama_mmproj"),
            stage_label="llama_mmproj",
        )
        self._emit(ProgressEvent(
            stage="llama_mmproj",
            message="skipped (already present)" if result.skipped else "downloaded",
            done=True,
        ))

    async def _install_whisper(self, cfg: WhisperConfig) -> None:
        if not whisper_local.is_available():
            self._emit(ProgressEvent(
                stage="whisper_cpp",
                message="pywhispercpp not installed; skipping whisper model download",
                done=True,
                error="pywhispercpp missing",
            ))
            return

        ensure_models_dir("whisper")
        self._emit(ProgressEvent(stage="whisper_model", message="Checking whisper model…"))
        path, skipped = await whisper_local.ensure_model_downloaded(
            cfg.local_model,
            progress=self._progress_cb("whisper_model"),
            stage_label="whisper_model",
        )
        self._emit(ProgressEvent(
            stage="whisper_model",
            message=f"{'skipped (already present)' if skipped else 'downloaded'}: {path.name}",
            done=True,
        ))

    def _progress_cb(self, stage: str):
        # Throttle: emit at most ~10 events/sec per stage. HF's tqdm can fire
        # hundreds of updates/sec for large files, which would swamp SSE
        # subscribers and make the progress bar jitter.
        last_emit = {"t": 0.0}

        def cb(done: int, total: int, label: str) -> None:
            import time
            now = time.monotonic()
            # Always emit the first and last updates; throttle the middle.
            if done != total and now - last_emit["t"] < 0.1:
                return
            last_emit["t"] = now
            self._emit(ProgressEvent(
                stage=stage,
                message=label,
                bytes_done=done,
                bytes_total=total,
            ))

        return cb


# Module-level singleton — shared by all HTTP handlers so a second request
# attaches to the same run instead of kicking off another.
runner = BootstrapRunner()


async def stream_events(
    q: asyncio.Queue[ProgressEvent],
) -> AsyncIterator[ProgressEvent]:
    """Yield events from a subscriber queue until a terminal event arrives."""
    while True:
        evt = await q.get()
        yield evt
        if evt.done and evt.stage in {"complete", "error"}:
            return
