"""High-level orchestrator: discover → upsert → prune.

One-shot ``index_now()`` for explicit scans (startup, API trigger)
and a fire-and-forget background loop for periodic re-scans.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

from cosma_backend.logging import get_logger

from .applications_discoverer import ApplicationsDiscoverer
from .applications_repository import ApplicationsRepository

logger = get_logger(__name__)


# Re-scanning every 6 hours is a generous floor — apps change rarely
# enough that a missed install costs at most one search session of
# "huh, where's that app?" before the next tick. Cheaper than
# subscribing to FSEvents on /Applications, which would also fire
# during every brew update / mas update.
DEFAULT_RESCAN_INTERVAL_SECONDS: int = 6 * 60 * 60


@dataclass
class ScanResult:
    """Lightweight summary returned from ``index_now`` for logs + API."""
    discovered: int
    upserted: int
    pruned: int
    elapsed_seconds: float


class ApplicationsIndexer:
    def __init__(
        self,
        repository: ApplicationsRepository,
        discoverer: Optional[ApplicationsDiscoverer] = None,
        rescan_interval_seconds: int = DEFAULT_RESCAN_INTERVAL_SECONDS,
    ) -> None:
        self._repo = repository
        self._discoverer = discoverer or ApplicationsDiscoverer()
        self._interval = rescan_interval_seconds
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

    async def index_now(self) -> ScanResult:
        """Run a full scan synchronously. Safe to call multiple times.

        Discovery runs in a thread because plistlib is CPU+IO bound
        and we don't want it sharing the event loop with active
        search requests. The DB writes happen back on the event loop
        (the asqlite pool handles thread coordination).
        """
        t_start = time.perf_counter()
        apps = await asyncio.to_thread(self._discoverer.discover)
        upserted = await self._repo.upsert_many(apps)
        present = {a.app_path for a in apps}
        pruned = await self._repo.delete_missing(present)
        elapsed = time.perf_counter() - t_start
        logger.info("Applications scan completed",
                     discovered=len(apps),
                     upserted=upserted,
                     pruned=pruned,
                     elapsed_seconds=round(elapsed, 3))
        return ScanResult(
            discovered=len(apps),
            upserted=upserted,
            pruned=pruned,
            elapsed_seconds=elapsed,
        )

    def start_background_rescan(self) -> None:
        """Fire-and-forget background loop. Idempotent — calling
        twice is a no-op after the first."""
        if self._task is not None and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=5)
            except asyncio.TimeoutError:
                self._task.cancel()

    async def _loop(self) -> None:
        """Initial scan + periodic re-scan.

        We always do one immediate scan so the apps source is usable
        as soon as the backend finishes Phase 1, then sleep the
        interval between subsequent scans. Errors are swallowed so
        the loop survives a transient permission error (e.g. macOS
        Full Disk Access just got revoked) and re-tries next tick.
        """
        try:
            await self.index_now()
        except Exception as e:
            logger.warning("Initial applications scan failed",
                            error=str(e))

        while not self._stop.is_set():
            try:
                # Wait either the full interval or until stop is set.
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self._interval,
                )
            except asyncio.TimeoutError:
                pass
            if self._stop.is_set():
                return
            try:
                await self.index_now()
            except Exception as e:
                logger.warning("Periodic applications scan failed",
                                error=str(e))
