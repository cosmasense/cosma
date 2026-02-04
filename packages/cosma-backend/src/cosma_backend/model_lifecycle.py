"""
Model Lifecycle Manager

Monitors AI model idle time and automatically unloads models
from GPU/memory after a configurable idle period.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Optional

from cosma_backend.logging import get_logger

if TYPE_CHECKING:
    from cosma_backend.summarizer import AutoSummarizer

logger = get_logger(__name__)

CHECK_INTERVAL_SECONDS = 10


class ModelLifecycleManager:
    """
    Monitors the summarizer for idle time and unloads models when
    they have been unused for longer than the configured threshold.

    The embedding model is intentionally excluded — it stays loaded
    for search responsiveness.
    """

    def __init__(
        self,
        summarizer: AutoSummarizer,
        idle_unload_seconds: int = 60,
    ):
        self._summarizer = summarizer
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

                last_used = self._summarizer.last_used_at
                if last_used > 0:
                    idle_time = time.time() - last_used
                    if idle_time > self._idle_unload_seconds:
                        logger.info(
                            "Models idle, unloading",
                            idle_seconds=round(idle_time, 1),
                            threshold=self._idle_unload_seconds,
                        )
                        await self._summarizer.unload_models()

                await asyncio.sleep(CHECK_INTERVAL_SECONDS)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Error in model lifecycle monitor loop")
                await asyncio.sleep(CHECK_INTERVAL_SECONDS)
