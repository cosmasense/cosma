"""High-level orchestrator: discover → upsert → prune.

One-shot ``index_now()`` for explicit scans (startup, API trigger)
and a fire-and-forget background loop for periodic re-scans.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from cosma_backend.logging import get_logger
from cosma_backend.models import Application

from .applications_discoverer import ApplicationsDiscoverer
from .applications_repository import ApplicationsRepository

if TYPE_CHECKING:
    from cosma_backend.embedder import AutoEmbedder

logger = get_logger(__name__)


def _embedding_text(app: Application) -> str:
    """Build the text body that will be embedded. Order matters
    for the embedder — putting the display name first weights it
    most heavily, and the description / use_cases fill in the
    "what's it for" signal that pure-FTS keyword search misses
    (e.g. 'photo editor' → Pixelmator)."""
    parts: list[str] = [app.display_name]
    if app.description:
        parts.append(app.description)
    if app.use_cases:
        parts.append(app.use_cases)
    if app.category:
        # "public.app-category.photography" → "photography" — the
        # human-readable part is what carries semantic weight.
        last = app.category.split(".")[-1]
        parts.append(last.replace("-", " "))
    if app.bundle_id:
        # Include the bundle id verbatim — semantic value is low but
        # it lets developers searching "com.docker" find Docker.
        parts.append(app.bundle_id)
    return " | ".join(p for p in parts if p)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
    embedded: int
    elapsed_seconds: float


class ApplicationsIndexer:
    def __init__(
        self,
        repository: ApplicationsRepository,
        discoverer: Optional[ApplicationsDiscoverer] = None,
        embedder: Optional["AutoEmbedder"] = None,
        rescan_interval_seconds: int = DEFAULT_RESCAN_INTERVAL_SECONDS,
    ) -> None:
        self._repo = repository
        self._discoverer = discoverer or ApplicationsDiscoverer()
        # Optional so tests can construct without an embedder. Production
        # always passes one — without it, apps stay keyword-only (which
        # is what shipped in 1.2.0 and prompted this follow-up).
        self._embedder = embedder
        self._interval = rescan_interval_seconds
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

    async def index_now(self) -> ScanResult:
        """Run a full scan synchronously. Safe to call multiple times.

        Discovery runs in a thread because plistlib is CPU+IO bound
        and we don't want it sharing the event loop with active
        search requests. The DB writes happen back on the event loop
        (the asqlite pool handles thread coordination). When an
        embedder is wired up, each new-or-changed app gets a
        semantic vector — unchanged apps are skipped via a text-hash
        comparison so periodic re-scans don't burn the embedder
        thread for no reason.
        """
        t_start = time.perf_counter()
        apps = await asyncio.to_thread(self._discoverer.discover)
        upserted = await self._repo.upsert_many(apps)
        present = {a.app_path for a in apps}
        pruned = await self._repo.delete_missing(present)
        embedded = 0
        if self._embedder is not None:
            embedded = await self._embed_changed_apps(apps)
        elapsed = time.perf_counter() - t_start
        logger.info("Applications scan completed",
                     discovered=len(apps),
                     upserted=upserted,
                     pruned=pruned,
                     embedded=embedded,
                     elapsed_seconds=round(elapsed, 3))
        return ScanResult(
            discovered=len(apps),
            upserted=upserted,
            pruned=pruned,
            embedded=embedded,
            elapsed_seconds=elapsed,
        )

    async def _embed_changed_apps(self, apps: list[Application]) -> int:
        """Embed every app whose current text body differs from the
        hash stored on the row. Returns the count actually embedded.

        We deliberately walk apps one at a time — the embedder is
        a single-MPS-device resource, so batching would just serialize
        inside the model anyway. Embedder failures are isolated per
        app: one bad input shouldn't poison the whole scan.
        """
        if not self._embedder:
            return 0
        model_name = getattr(self._embedder, "model_name", "unknown")
        embedded = 0
        for app in apps:
            try:
                # Re-read the persisted row so we have the id + the
                # hash that survived the UPSERT (apps.use_cases is
                # COALESCE'd, so the row may carry an enriched body
                # the in-memory discoverer copy doesn't know about).
                persisted = await self._repo.get_by_app_path(app.app_path)
                if persisted is None or persisted.id is None:
                    continue
                text = _embedding_text(persisted)
                text_hash = _hash_text(text)
                if persisted.embedding_text_hash == text_hash:
                    continue  # unchanged — skip the embedder call
                embedding = await self._embedder.embed_text_async(
                    text, priority=False
                )
                await self._repo.set_embedding(
                    persisted.id,
                    embedding,
                    embedding_model=model_name,
                    text_hash=text_hash,
                )
                embedded += 1
            except Exception as e:
                logger.warning("App embed failed; leaving keyword-only",
                                app=app.app_path, error=str(e))
        return embedded

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
