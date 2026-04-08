"""
Model Lifecycle Manager

Monitors AI model idle time and automatically unloads models
from GPU/memory after a configurable idle period. This helps
reduce memory usage when models aren't actively being used.

Monitored models:
- Summarizer (Ollama, LlamaCpp, or online)
- Embedder (local SentenceTransformer)
- Whisper (audio transcription)

All models are lazily reloaded on next use after being unloaded.

Each model tracks its last_used_at timestamp. When idle time
exceeds the threshold, the model is unloaded to free resources.
Models are lazily reloaded on next use.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Optional

from cosma_backend.logging import get_logger

if TYPE_CHECKING:
    from cosma_backend.summarizer import AutoSummarizer
    from cosma_backend.embedder import AutoEmbedder

logger = get_logger(__name__)

# How often to check model idle times (seconds)
CHECK_INTERVAL_SECONDS = 10


class ModelLifecycleManager:
    """
    Monitors the summarizer and embedder for idle time and unloads
    models when they have been unused for longer than the configured
    threshold, freeing GPU/CPU memory.
    """

    def __init__(
        self,
        summarizer: AutoSummarizer,
        idle_unload_seconds: int = 60,
        embedder: Optional["AutoEmbedder"] = None,
    ):
        self._summarizer = summarizer
        self._embedder = embedder
        self._idle_unload_seconds = idle_unload_seconds
        self._task: Optional[asyncio.Task] = None

    @property
    def idle_unload_seconds(self) -> int:
        return self._idle_unload_seconds

    @idle_unload_seconds.setter
    def idle_unload_seconds(self, value: int) -> None:
        self._idle_unload_seconds = value

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._monitor_loop())
            logger.info(
                "Model lifecycle manager started",
                idle_unload_seconds=self._idle_unload_seconds,
            )

    async def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            logger.info("Model lifecycle manager stopped")

    async def _monitor_loop(self) -> None:
        while True:
            try:
                if self._idle_unload_seconds <= 0:
                    # 0 means never unload
                    await asyncio.sleep(CHECK_INTERVAL_SECONDS)
                    continue

                now = time.time()

                # Check summarizer idle time (only if a model is loaded)
                if self._summarizer.is_any_model_loaded():
                    last_used = self._summarizer.last_used_at
                    if last_used > 0:
                        idle_time = now - last_used
                        if idle_time > self._idle_unload_seconds:
                            logger.info(
                                "Summarizer idle, unloading",
                                idle_seconds=round(idle_time, 1),
                                threshold=self._idle_unload_seconds,
                            )
                            await self._summarizer.unload_models()

                # Check embedder idle time (lazy-reloads on next use)
                if self._embedder is not None and self._embedder.is_model_loaded():
                    last_used = self._embedder.last_used_at
                    if last_used > 0:
                        idle_time = now - last_used
                        if idle_time > self._idle_unload_seconds:
                            logger.info(
                                "Embedder idle, unloading",
                                idle_seconds=round(idle_time, 1),
                                threshold=self._idle_unload_seconds,
                            )
                            await self._embedder.unload_models()

                # Check whisper model idle time
                from cosma_backend.parser.media import (
                    get_whisper_last_used_at,
                    unload_whisper_model,
                )
                whisper_last_used = get_whisper_last_used_at()
                if whisper_last_used is not None:
                    whisper_idle = now - whisper_last_used
                    if whisper_idle > self._idle_unload_seconds:
                        logger.info(
                            "Whisper model idle, unloading",
                            idle_seconds=round(whisper_idle, 1),
                            threshold=self._idle_unload_seconds,
                        )
                        await asyncio.to_thread(unload_whisper_model)

                await asyncio.sleep(CHECK_INTERVAL_SECONDS)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Error in model lifecycle monitor loop")
                await asyncio.sleep(CHECK_INTERVAL_SECONDS)

    def get_model_status(self) -> list[dict]:
        """Return loaded/unloaded status for each managed model."""
        now = time.time()
        models = []

        # Summarizer
        for name, summarizer in self._summarizer.summarizers.items():
            if summarizer is None:
                models.append({
                    "name": f"summarizer ({name})",
                    "loaded": False,
                    "idle_seconds": None,
                })
                continue
            loaded = False
            if hasattr(summarizer, "_model_loaded"):
                loaded = summarizer._model_loaded
            elif hasattr(summarizer, "llm"):
                loaded = summarizer.llm is not None
            elif name == "online":
                loaded = True  # online providers are always "loaded"

            last_used = summarizer.last_used_at if hasattr(summarizer, "last_used_at") else 0.0
            idle = round(now - last_used, 1) if last_used > 0 else None
            models.append({
                "name": f"summarizer ({name})",
                "loaded": loaded,
                "idle_seconds": idle,
            })

        # Embedder
        if self._embedder is not None:
            local = self._embedder.embedders.get("local")
            if local is not None:
                loaded = hasattr(local, "model") and local.model is not None
                last_used = local.last_used_at if hasattr(local, "last_used_at") else 0.0
                idle = round(now - last_used, 1) if last_used > 0 else None
                models.append({
                    "name": "embedder (local)",
                    "loaded": loaded,
                    "idle_seconds": idle,
                })
            online = self._embedder.embedders.get("online")
            if online is not None:
                models.append({
                    "name": "embedder (online)",
                    "loaded": True,
                    "idle_seconds": None,
                })

        # Whisper
        try:
            from cosma_backend.parser.media import get_whisper_last_used_at, _whisper_model
            whisper_loaded = _whisper_model is not None
            whisper_last = get_whisper_last_used_at()
            whisper_idle = round(now - whisper_last, 1) if whisper_last else None
            models.append({
                "name": "whisper",
                "loaded": whisper_loaded,
                "idle_seconds": whisper_idle,
            })
        except Exception:
            pass

        return models
