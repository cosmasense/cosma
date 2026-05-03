"""
Indexing Queue

Core queue with debounce, pause/resume, and a processing loop that feeds
items to the pipeline after their cooldown expires.
"""

from __future__ import annotations

import asyncio
import enum
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, TYPE_CHECKING, Optional

from cosma_backend.db.errors import DatabaseClosingError
from cosma_backend.logging import get_logger
from cosma_backend.models import File
from cosma_backend.models.update import Update, UpdateOpcode

if TYPE_CHECKING:
    from cosma_backend.db.database import Database
    from cosma_backend.pipeline import Pipeline
    from cosma_backend.settings import QueueConfig
    from cosma_backend.utils.pubsub import Hub

logger = get_logger(__name__)


class QueueAction(enum.Enum):
    INDEX = "index"
    DELETE = "delete"
    MOVE = "move"
    EMBED_FALLBACK = "embed_fallback"


class QueueItemStatus(enum.Enum):
    COOLING_DOWN = "cooling_down"
    WAITING = "waiting"
    PROCESSING = "processing"


@dataclass
class QueueItem:
    id: str
    file_path: str
    action: QueueAction
    status: QueueItemStatus
    enqueued_at: float
    cooldown_expires_at: float
    dest_path: Optional[str] = None
    retry_count: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "file_path": self.file_path,
            "action": self.action.value,
            "status": self.status.value,
            "enqueued_at": self.enqueued_at,
            "cooldown_expires_at": self.cooldown_expires_at,
            "dest_path": self.dest_path,
            "retry_count": self.retry_count,
        }


class IndexingQueue:
    """
    Queue that sits between the file watcher and the processing pipeline.

    Files are enqueued with a cooldown (debounce) period. Rapid re-enqueue
    of the same path resets the timer. Once the cooldown expires the item
    becomes READY and is dispatched to ``Pipeline.process_file()``.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        updates_hub: Optional[Hub] = None,
        config: Optional[QueueConfig] = None,
        db: Optional["Database"] = None,
    ):
        from cosma_backend.settings import QueueConfig as _QC

        self._pipeline = pipeline
        self._updates_hub = updates_hub
        self._config = config or _QC()
        self._db = db

        # keyed by file_path (src_path for MOVEs) for fast dedup/lookup
        self._items: dict[str, QueueItem] = {}
        # Reverse lookup: dest_path → src_path for MOVE items, so an INDEX
        # for dest_path can find the existing MOVE and skip redundant work.
        self._dest_to_src: dict[str, str] = {}
        # Pending re-enqueue requests for items currently being processed.
        # Keyed by file_path → (action, dest_path).
        self._pending_reenqueue: dict[str, tuple[QueueAction, Optional[str]]] = {}
        self._lock = asyncio.Lock()

        self._manually_paused = False
        self._scheduler_paused = False
        # User pause/resume now acts as a *one-shot override* on top of
        # the scheduler decision, instead of a permanent flag:
        #
        #   None   → defer to scheduler (rules + bootstrap decide)
        #   True   → user clicked Resume — force-run even if scheduler
        #            says "should pause" right now
        #   False  → user clicked Pause — force-pause even if scheduler
        #            says "should run" right now
        #
        # The override is cleared automatically the next time the
        # scheduler's evaluated outcome *transitions* (e.g. CPU drops
        # back below threshold, time window opens). At that point the
        # user's nudge has served its purpose and the scheduler takes
        # back over. This keeps `_manually_paused` (which is now a
        # synonym for the False state of the override) from getting
        # stuck against an active scheduler rule.
        self._user_override: Optional[bool] = None
        # Set at startup if bootstrap (llama/whisper model downloads) hasn't
        # finished yet. Cleared once the bootstrap runner emits a "complete"
        # event. Without this gate, any file discovered by the watcher during
        # the initial download would hit AutoSummarizer → load_model() →
        # a lazy HF download racing the bootstrap on the same GGUF.
        self._bootstrap_paused = False

        # Search-preempts-indexing: every /api/search/ hit calls
        # search_preempt(), which sets `_search_preempt_until` to
        # `now + search_preempt_seconds`. While we're inside that window,
        # `is_paused` reports True so no new items are dispatched. In-flight
        # items finish normally — we only block the dispatch decision.
        self._search_preempt_until: float = 0.0
        self._search_preempt_seconds: float = float(getattr(
            self._config, "search_preempt_seconds", 10.0,
        ))

        # In-flight task tracking for `cancel_in_flight()`. Keyed by
        # file_path so a search-typing nudge can cancel exactly the
        # tasks currently consuming GPU/CPU. Crash-safe progress
        # (status PARSED/SUMMARIZED durably persisted between stages)
        # means a cancelled task's work isn't lost — its DB row stays
        # at the last completed stage and the pipeline resumes from
        # there on the next dispatch.
        self._in_flight_tasks: dict[str, asyncio.Task] = {}

        self._processing_task: Optional[asyncio.Task] = None
        # Sized off `effective_max_concurrency` (parse + summarize + embed)
        # so the global semaphore is never the binding constraint — the
        # per-stage caps already serialize what needs serializing. The
        # legacy `max_concurrency` field is kept on QueueConfig for TOML
        # compatibility but no longer consulted at runtime; see
        # settings.QueueConfig.effective_max_concurrency.
        self._semaphore = asyncio.Semaphore(self._config.effective_max_concurrency)

        # Optional hook invoked right before each batch of ready items is
        # dispatched. The scheduler installs this so rule evaluation happens
        # at task boundaries (not mid-task). The hook may set
        # ``_scheduler_paused`` via queue.scheduler_pause() — the loop will
        # then wait instead of picking up new work.
        self._pre_task_hook: Optional[Callable[[], Awaitable[Any]]] = None

        # Coalescer for QUEUE_ITEM_ADDED bursts during bulk discovery. New
        # paths land in `_added_buffer`; a single asyncio task drains and
        # publishes them as one QUEUE_BATCH_ADDED event every
        # ``_added_flush_interval`` seconds. Without this, a 10k-file
        # discovery sweep produced 10k SSE events, each triggering frontend
        # state mutations and re-renders that visibly froze the UI even
        # after we moved SSE parsing off the main actor.
        self._added_buffer: list[str] = []
        self._added_buffer_lock = asyncio.Lock()
        self._added_flush_task: Optional[asyncio.Task] = None
        self._added_flush_interval: float = 0.2  # 200 ms
        # Hard upper bound on the buffer — if discovery is slamming us
        # faster than 200 ms, force a flush at this many items rather
        # than letting a single batch grow unbounded.
        self._added_buffer_max: int = 500

    def set_pre_task_hook(self, hook: Optional[Callable[[], Awaitable[Any]]]) -> None:
        """Install a callback invoked once per processing cycle, right
        before ready items are picked. Used by the scheduler to evaluate
        rules between tasks without a fixed polling interval."""
        self._pre_task_hook = hook

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def is_paused(self) -> bool:
        # Bootstrap is non-overridable — if model files are still being
        # downloaded, we MUST stay paused or the pipeline will race the
        # downloader on the same GGUF.
        if self._bootstrap_paused:
            return True
        # Search preemption is also non-overridable: while a user is
        # actively searching, indexing must yield the GPU to the embedder
        # / LLM so query latency stays low. The window is short
        # (default 10 s) so this doesn't meaningfully slow indexing.
        if self.is_search_preempted:
            return True
        # User override beats scheduler in either direction. None means
        # defer to scheduler.
        if self._user_override is False:
            return True
        if self._user_override is True:
            return False
        # No override — keep the existing union of manual + scheduler so
        # legacy `manual_pause()` callers still pause correctly. (After
        # the API switch below, manual_pause() routes through
        # user_override_pause(), so this `_manually_paused` branch is
        # mostly defensive — it covers any direct test or external
        # caller that still sets the old flag.)
        return self._manually_paused or self._scheduler_paused

    @property
    def is_search_preempted(self) -> bool:
        return time.time() < self._search_preempt_until

    def search_preempt(self) -> None:
        """Note that a user-driven search is happening right now. Pauses
        new dispatch for `search_preempt_seconds`. Repeated calls extend
        the pause from "now"."""
        self._search_preempt_until = time.time() + self._search_preempt_seconds

    def cancel_in_flight(self) -> int:
        """Cancel every currently-processing item so the GPU/CPU is
        immediately freed for a user search. Cancelled items' partial
        progress is durable in the DB (status=PARSED or =SUMMARIZED
        depending on which stage they were in) thanks to the persisted-
        progress refactor — they'll be picked up automatically on the
        next dispatch.

        Returns the number of tasks cancelled.

        Note: Python can't safely interrupt OS threads, so an in-flight
        blocking call inside `parser.parse_file` (e.g. a Spotlight
        subprocess wait) finishes its current sub-call. What we cancel
        is the *asyncio orchestrator* that would have run summarize +
        embed for that file; the file ends up at its last persisted
        stage instead of progressing.
        """
        n = 0
        for path, task in list(self._in_flight_tasks.items()):
            if not task.done():
                task.cancel()
                n += 1
        return n

    @property
    def is_manually_paused(self) -> bool:
        # "Manually paused" from the UI's point of view = user explicitly
        # set the override to False. The legacy `_manually_paused` flag
        # is kept for backward compat with code that still reads it.
        return self._manually_paused or self._user_override is False

    @property
    def is_scheduler_paused(self) -> bool:
        return self._scheduler_paused

    @property
    def user_override(self) -> Optional[bool]:
        """Current one-shot override state. None = defer to scheduler."""
        return self._user_override

    @property
    def is_bootstrap_paused(self) -> bool:
        return self._bootstrap_paused

    @property
    def item_count(self) -> int:
        return len(self._items)

    @property
    def queue_size(self) -> int:
        """Return the count of active (non-completed) items in the queue."""
        return len(self._items)

    @property
    def initial_cooldown_seconds(self) -> float:
        return self._config.initial_cooldown_seconds

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Load persisted items and start the background processing loop."""
        await self._load_from_db()
        # Crash recovery: any file row in a non-terminal status from a
        # prior run (DISCOVERED / PARSED / SUMMARIZED) means the previous
        # process exited mid-pipeline. Re-enqueue them — the pipeline's
        # resume-aware process_file will pick up where each one left off.
        # Must run before the processing loop starts (or the watcher
        # delivers any new events) so we don't double-process.
        await self._recover_in_flight_files()
        if self._processing_task is None or self._processing_task.done():
            self._processing_task = asyncio.create_task(self._processing_loop())
            logger.info("Indexing queue processing loop started")
        if self._added_flush_task is None or self._added_flush_task.done():
            self._added_flush_task = asyncio.create_task(self._added_flush_loop())

    async def _recover_in_flight_files(self) -> None:
        """Re-enqueue files left in non-terminal states by a prior crash.

        Looks for status IN (DISCOVERED, PARSED, SUMMARIZED) and pushes
        each onto the queue as a normal INDEX action. The pipeline's
        process_file resumes at the right stage based on the saved row
        (see Pipeline.process_file in pipeline/pipeline.py).

        This is best-effort — a DB error here must not block the queue
        from starting, so we log and continue.
        """
        if self._db is None:
            return
        recovered = 0
        try:
            for status_name in ("DISCOVERED", "PARSED", "SUMMARIZED"):
                files, total = await self._db.get_files_by_status(
                    status_name, limit=100000, offset=0,
                )
                for f in files:
                    # Skip if a queue item already exists (loaded by
                    # _load_from_db) — that path is already scheduled.
                    if f.file_path in self._items:
                        continue
                    # Re-enqueue as INDEX. Use a 0-cooldown override so
                    # recovery is prompt; the normal debounce only matters
                    # for live filesystem events, not crashed-mid-flight
                    # files we already know about.
                    await self.enqueue(
                        Path(f.file_path), QueueAction.INDEX,
                        cooldown_seconds=0,
                    )
                    recovered += 1
            if recovered:
                logger.info("Crash recovery: re-enqueued non-terminal files",
                            count=recovered)
        except Exception:
            logger.exception("Crash recovery scan failed (non-fatal)")

    async def stop(self) -> None:
        """Stop the background processing loop."""
        if self._processing_task is not None and not self._processing_task.done():
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
            logger.info("Indexing queue processing loop stopped")
        if self._added_flush_task is not None and not self._added_flush_task.done():
            self._added_flush_task.cancel()
            try:
                await self._added_flush_task
            except asyncio.CancelledError:
                pass
        # Final drain so anything still in the buffer reaches subscribers
        # before shutdown. Safe to call from any state.
        self._flush_added_buffer()

    async def _added_flush_loop(self) -> None:
        """Periodically drain the QUEUE_ITEM_ADDED coalescing buffer.

        Runs as a long-lived background task. Each tick checks the buffer
        and, if non-empty, emits a single QUEUE_BATCH_ADDED SSE event with
        the accumulated paths. The flush is also triggered synchronously
        from enqueue() when the buffer hits ``_added_buffer_max`` so
        sustained high-rate enqueueing doesn't let the buffer grow
        unbounded between ticks.
        """
        while True:
            try:
                await asyncio.sleep(self._added_flush_interval)
                self._flush_added_buffer()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Queue added-buffer flush failed; continuing")

    def _flush_added_buffer(self) -> None:
        """Emit one QUEUE_BATCH_ADDED for everything in the buffer.

        Cheap and synchronous (no I/O, no DB) so it's safe to call from
        either the periodic flusher or directly from enqueue() under the
        lock when the buffer hits its size cap.
        """
        if not self._added_buffer:
            return
        paths = self._added_buffer
        self._added_buffer = []
        # Single info-level summary per flush so an operator can still
        # see ingestion progress in logs without the per-file noise.
        logger.info("Queue enqueue batch flushed", count=len(paths))
        try:
            self._updates_hub.publish(Update.create(
                UpdateOpcode.QUEUE_BATCH_ADDED,
                count=len(paths),
                paths=paths,
            ))
        except Exception:
            logger.exception("Failed to publish QUEUE_BATCH_ADDED")

    # ------------------------------------------------------------------
    # Pause / Resume
    # ------------------------------------------------------------------

    def manual_pause(self) -> None:
        """User clicked Pause. Acts as a one-shot override that takes
        effect immediately and is cleared next time the scheduler's
        evaluated outcome transitions (so the user's nudge doesn't get
        stuck against rules that were never going to disagree).
        """
        # Keep the legacy flag in sync for any external code still
        # reading `is_manually_paused` directly.
        self._manually_paused = True
        if self._user_override is not False:
            self._user_override = False
            self._publish(Update.create(UpdateOpcode.QUEUE_PAUSED, source="manual"))
            logger.info("Queue paused by user override (one-shot)")

    def manual_resume(self) -> None:
        """User clicked Resume. One-shot override that forces the queue
        to run *now* even if the scheduler's last decision was pause.
        Cleared on the next scheduler transition.
        """
        self._manually_paused = False
        if self._user_override is not True:
            self._user_override = True
            self._publish(Update.create(UpdateOpcode.QUEUE_RESUMED, source="manual"))
            logger.info("Queue resumed by user override (one-shot)")

    def clear_user_override(self) -> None:
        """Drop the one-shot override and let the scheduler decide.

        Called by the scheduler when its evaluated outcome transitions
        (rules went from passing→failing or failing→passing). Also
        callable from any caller that wants to explicitly hand control
        back to the rules — e.g. an admin endpoint.
        """
        if self._user_override is not None:
            prior = self._user_override
            self._user_override = None
            # Mirror to the legacy flag so observers stay consistent.
            self._manually_paused = False
            logger.info("User override cleared (scheduler resumes control)",
                        prior=prior)

    def scheduler_pause(self) -> None:
        if not self._scheduler_paused:
            self._scheduler_paused = True
            self._publish(Update.create(UpdateOpcode.SCHEDULER_PAUSED))
            logger.info("Queue paused by scheduler")

    def scheduler_resume(self) -> None:
        if self._scheduler_paused:
            self._scheduler_paused = False
            self._publish(Update.create(UpdateOpcode.SCHEDULER_RESUMED))
            logger.info("Queue resumed by scheduler")

    def bootstrap_pause(self) -> None:
        if not self._bootstrap_paused:
            self._bootstrap_paused = True
            logger.info("Queue paused: bootstrap in progress")

    def bootstrap_resume(self) -> None:
        if self._bootstrap_paused:
            self._bootstrap_paused = False
            logger.info("Queue resumed: bootstrap complete")

    # ------------------------------------------------------------------
    # Enqueue / Remove
    # ------------------------------------------------------------------

    async def enqueue(
        self,
        file_path: str | Path,
        action: QueueAction,
        dest_path: Optional[str | Path] = None,
        cooldown_seconds: Optional[float] = None,
    ) -> QueueItem:
        """Add or update an item in the queue with debounce.

        ``cooldown_seconds`` overrides the default cooldown for this enqueue
        call (e.g. shorter cooldown for initial directory scans).
        """
        cooldown = cooldown_seconds if cooldown_seconds is not None else self._config.cooldown_seconds
        key = str(Path(file_path).resolve())
        dest_key = str(Path(dest_path).resolve()) if dest_path else None
        now = time.time()

        # Base the debounce window on the file's last-modified time, not
        # on when this event reached us. Watchers can batch / lag by a few
        # seconds; without this a file that was actually saved 10s ago
        # still sat in "cooling down" for another full cooldown window,
        # making indexing feel sluggish. mtime + cooldown is the correct
        # "wait N seconds after the user last touched the file" semantic.
        # Falls back to `now` if the file is gone (deletes) or stat fails.
        try:
            mtime = Path(key).stat().st_mtime
            cooldown_base = max(mtime, now - cooldown)  # clamp to recent past
        except OSError:
            cooldown_base = now
        expires_at = cooldown_base + cooldown

        item_to_persist: Optional[QueueItem] = None

        async with self._lock:
            existing = self._items.get(key)

            # If an INDEX arrives for a path that is the dest of a pending MOVE,
            # the MOVE already covers it — just reset the MOVE's cooldown.
            if existing is None and action == QueueAction.INDEX:
                src_key = self._dest_to_src.get(key)
                if src_key is not None:
                    move_item = self._items.get(src_key)
                    if move_item is not None and move_item.status == QueueItemStatus.COOLING_DOWN:
                        move_item.cooldown_expires_at = expires_at
                        self._publish(Update.create(
                            UpdateOpcode.QUEUE_ITEM_UPDATED, **move_item.to_dict()
                        ))
                        logger.info("INDEX for MOVE dest_path, reset MOVE cooldown",
                                    dest_path=key, src_path=src_key)
                        return move_item
                    # If MOVE is PROCESSING or READY, let the INDEX create a new
                    # item keyed by dest_path — it will re-index after the MOVE.

            if existing is not None:
                if existing.status == QueueItemStatus.PROCESSING:
                    # Item is currently being processed. Stash the new request
                    # so _process_item can re-enqueue it after completion.
                    dp = str(dest_path) if dest_path else None
                    self._pending_reenqueue[key] = (action, dp)
                    logger.info("Item currently processing, pending re-enqueue",
                                file_path=key, action=action.value)
                    return existing

                if existing.status == QueueItemStatus.COOLING_DOWN:
                    # Clean up old dest mapping if action is changing
                    if existing.action == QueueAction.MOVE and existing.dest_path:
                        self._dest_to_src.pop(existing.dest_path, None)

                    # DELETE on a cooling-down item → make it ready immediately
                    if action == QueueAction.DELETE:
                        existing.action = QueueAction.DELETE
                        existing.dest_path = None
                        existing.cooldown_expires_at = now
                        existing.status = QueueItemStatus.WAITING
                        self._publish(Update.create(
                            UpdateOpcode.QUEUE_ITEM_UPDATED, **existing.to_dict()
                        ))
                        item_to_persist = existing
                        logger.debug("Queue item fast-tracked to DELETE", file_path=key)

                    else:
                        # Otherwise reset the cooldown timer
                        existing.action = action
                        existing.cooldown_expires_at = expires_at
                        existing.dest_path = dest_key if dest_key else existing.dest_path
                        # Update dest mapping
                        if action == QueueAction.MOVE and existing.dest_path:
                            self._dest_to_src[existing.dest_path] = key
                        self._publish(Update.create(
                            UpdateOpcode.QUEUE_ITEM_UPDATED, **existing.to_dict()
                        ))
                        item_to_persist = existing
                        logger.debug("Queue item cooldown reset", file_path=key)

                elif existing.status == QueueItemStatus.WAITING:
                    # Item is ready but hasn't been dispatched yet; reset it
                    # back to COOLING_DOWN so the new event is debounced.
                    if existing.action == QueueAction.MOVE and existing.dest_path:
                        self._dest_to_src.pop(existing.dest_path, None)
                    existing.action = action
                    existing.status = QueueItemStatus.COOLING_DOWN
                    existing.cooldown_expires_at = expires_at
                    existing.dest_path = dest_key if dest_key else existing.dest_path
                    if action == QueueAction.MOVE and existing.dest_path:
                        self._dest_to_src[existing.dest_path] = key
                    self._publish(Update.create(
                        UpdateOpcode.QUEUE_ITEM_UPDATED, **existing.to_dict()
                    ))
                    item_to_persist = existing
                    logger.debug("Queue item reset from WAITING to cooldown",
                                 file_path=key, action=action.value)

            if item_to_persist is None and existing is not None:
                # Handled above (PROCESSING stash) — already returned
                pass
            elif item_to_persist is not None:
                # Existing item was updated; will persist below
                pass
            else:
                # New item
                item = QueueItem(
                    id=str(uuid.uuid4()),
                    file_path=key,
                    action=action,
                    status=QueueItemStatus.COOLING_DOWN,
                    enqueued_at=now,
                    cooldown_expires_at=now + cooldown,
                    dest_path=dest_key,
                )
                self._items[key] = item
                if action == QueueAction.MOVE and dest_key:
                    self._dest_to_src[dest_key] = key
                # Buffer the ADDED notification. The flusher publishes a
                # single QUEUE_BATCH_ADDED every ~200 ms (or when the
                # buffer hits _added_buffer_max). This collapses the
                # 10k-event burst from a fresh-folder scan into ~50
                # batched events with negligible UI cost. See _flush_added_buffer.
                self._added_buffer.append(key)
                if len(self._added_buffer) >= self._added_buffer_max:
                    self._flush_added_buffer()
                item_to_persist = item
                # Per-item log demoted to debug — bulk discovery used to
                # emit thousands of "Item enqueued" lines per second,
                # crowding out anything actually useful in the log file.
                # The periodic flusher emits a single info summary instead.
                logger.debug("Item enqueued", file_path=key, action=action.value)

        # Persist outside the lock to avoid blocking concurrent operations.
        if item_to_persist is not None:
            await self._save_item(item_to_persist)

        return item_to_persist if item_to_persist is not None else existing

    async def remove_item(self, item_id: str) -> bool:
        """Remove an item by its ID. Returns True if found and removed."""
        removed_key: Optional[str] = None
        async with self._lock:
            for key, item in self._items.items():
                if item.id == item_id:
                    del self._items[key]
                    self._cleanup_dest_map(item)
                    self._publish(Update.create(
                        UpdateOpcode.QUEUE_ITEM_REMOVED, **item.to_dict()
                    ))
                    removed_key = key
                    break
        if removed_key is not None:
            await self._delete_item_from_db(item_id)
            logger.info("Queue item removed", item_id=item_id, file_path=removed_key)
            return True
        return False

    async def remove_items_under(self, directory: str | Path) -> int:
        """Remove all queued items whose path is under the given directory."""
        dir_path = Path(directory).resolve()
        removed = 0
        async with self._lock:
            keys_to_remove = []
            for k in self._items:
                item_path = Path(k)
                # Check if item_path is the directory itself or is inside it
                try:
                    item_path.relative_to(dir_path)
                    keys_to_remove.append(k)
                except ValueError:
                    # Not under dir_path
                    pass
            for key in keys_to_remove:
                item = self._items.pop(key)
                self._cleanup_dest_map(item)
                self._publish(Update.create(
                    UpdateOpcode.QUEUE_ITEM_REMOVED, **item.to_dict()
                ))
                removed += 1
        if removed and self._db is not None:
            try:
                await self._db.delete_queue_items_under(str(dir_path))
            except Exception:
                logger.exception("Failed to delete queue items under directory from DB")
            logger.info("Removed items under directory", directory=str(dir_path), count=removed)
        return removed

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    async def get_items(self) -> list[QueueItem]:
        async with self._lock:
            return list(self._items.values())

    async def get_item(self, item_id: str) -> Optional[QueueItem]:
        async with self._lock:
            for item in self._items.values():
                if item.id == item_id:
                    return item
            return None

    async def get_status(self) -> dict:
        # Snapshot without taking the lock. During a large initial scan the
        # lock is held hundreds of times per second by enqueue(), and a
        # status call that queued behind those acquires could miss the
        # frontend's 60 s URLSession timeout. A plain list(self._items.values())
        # is safe here: Python dict views over in-memory state are cheap and
        # a racy read that shows a count ±1 is fine for a UI badge.
        items = list(self._items.values())
        return {
            "paused": self.is_paused,
            "manually_paused": self.is_manually_paused,
            "scheduler_paused": self._scheduler_paused,
            "bootstrap_paused": self._bootstrap_paused,
            # New: while True, the queue is paused because a user search
            # just happened. Auto-clears after search_preempt_seconds.
            # Frontend uses this to render a "paused for search" reason
            # badge so users don't think indexing is broken.
            "search_preempted": self.is_search_preempted,
            # One-shot user override surfaced for the UI:
            #   None  → button reflects scheduler decision
            #   True  → user has forced run; button shows "Pause"
            #   False → user has forced pause; button shows "Resume"
            # The override clears automatically on the next scheduler
            # transition; see Scheduler.evaluate_and_apply.
            "user_override": self._user_override,
            "total_items": len(items),
            "cooling_down": sum(1 for i in items if i.status == QueueItemStatus.COOLING_DOWN),
            "waiting": sum(1 for i in items if i.status == QueueItemStatus.WAITING),
            "processing": sum(1 for i in items if i.status == QueueItemStatus.PROCESSING),
        }

    # ------------------------------------------------------------------
    # Processing loop
    # ------------------------------------------------------------------

    async def _processing_loop(self) -> None:
        """Main loop: transition items through cooldown → ready → processing."""
        while True:
            try:
                if self.is_paused:
                    await asyncio.sleep(1)
                    continue

                now = time.time()
                ready_items: list[QueueItem] = []
                items_to_persist: list[QueueItem] = []

                async with self._lock:
                    for item in self._items.values():
                        if (
                            item.status == QueueItemStatus.COOLING_DOWN
                            and now >= item.cooldown_expires_at
                        ):
                            item.status = QueueItemStatus.WAITING
                            items_to_persist.append(item)

                    has_waiting = any(
                        i.status == QueueItemStatus.WAITING
                        for i in self._items.values()
                    )

                # Task-boundary scheduler check: evaluate rules right before
                # we promote WAITING items to PROCESSING. If the hook flips
                # is_paused, we loop back and wait without starting new work.
                # This is the sole "should we run now?" check — there is no
                # fixed polling interval for GPU/CPU/memory rules, so a
                # single file's inference cannot get interrupted mid-way.
                if has_waiting and self._pre_task_hook is not None:
                    try:
                        await self._pre_task_hook()
                    except Exception:
                        logger.exception("pre_task_hook raised — proceeding anyway")
                    if self.is_paused:
                        await asyncio.sleep(1)
                        continue

                async with self._lock:
                    # Only claim as many items as we can actually run at once.
                    # The old behavior marked every WAITING item as PROCESSING,
                    # spawning hundreds of asyncio tasks that all parked on
                    # the semaphore — starving the event loop and making
                    # status=PROCESSING lie about how many were truly running.
                    ready_items = []
                    cap = self._config.effective_max_concurrency
                    for i in self._items.values():
                        if i.status == QueueItemStatus.WAITING:
                            ready_items.append(i)
                            if len(ready_items) >= cap:
                                break
                    # Mark PROCESSING under the lock so concurrent enqueue()
                    # calls see the correct status while items wait for the
                    # semaphore.
                    for item in ready_items:
                        item.status = QueueItemStatus.PROCESSING
                        items_to_persist.append(item)

                # Persist status changes outside the lock to avoid blocking
                # enqueue/remove operations during DB writes.
                for item in items_to_persist:
                    await self._save_item(item)

                if ready_items:
                    # Wrap each item in an asyncio.Task and register it
                    # under its file_path so cancel_in_flight() can find
                    # and cancel exactly the tasks consuming hardware.
                    # Task callbacks deregister on completion.
                    tasks: list[asyncio.Task] = []
                    for item in ready_items:
                        t = asyncio.create_task(self._process_item(item))
                        self._in_flight_tasks[item.file_path] = t
                        # bind file_path to the closure to deregister
                        # the right entry; default-arg trick avoids
                        # late-binding bugs.
                        def _done(fut, fp=item.file_path):
                            self._in_flight_tasks.pop(fp, None)
                        t.add_done_callback(_done)
                        tasks.append(t)
                    await asyncio.gather(*tasks, return_exceptions=True)

                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                raise
            except DatabaseClosingError:
                logger.debug("Queue loop stopping (DB closing during shutdown)")
                return
            except Exception:
                logger.exception("Error in queue processing loop")
                await asyncio.sleep(1)

    async def _process_item(self, item: QueueItem) -> None:
        """Process a single queue item through the pipeline."""
        async with self._semaphore:
            async with self._lock:
                # Item may have been removed or re-enqueued while waiting
                # for the semaphore.  The loop already set status to
                # PROCESSING under the lock, but if the item was removed
                # (or replaced by a new enqueue) we should bail out.
                current = self._items.get(item.file_path)
                if current is not item:
                    return

            self._publish(Update.create(
                UpdateOpcode.QUEUE_ITEM_PROCESSING, **item.to_dict()
            ))

            try:
                timeout = self._config.file_processing_timeout

                if item.action == QueueAction.DELETE:
                    await asyncio.wait_for(
                        self._pipeline.db.delete_file(item.file_path),
                        timeout=timeout,
                    )
                elif item.action == QueueAction.MOVE:
                    async def _do_move() -> None:
                        await self._pipeline.db.delete_file(item.file_path)
                        if item.dest_path:
                            dest_file = File.from_path(Path(item.dest_path))
                            if await self._pipeline.is_supported(dest_file):
                                await self._pipeline.process_file(dest_file)
                    await asyncio.wait_for(_do_move(), timeout=timeout)
                elif item.action == QueueAction.INDEX:
                    file = File.from_path(Path(item.file_path))
                    if await self._pipeline.is_supported(file):
                        await asyncio.wait_for(
                            self._pipeline.process_file(file),
                            timeout=timeout,
                        )
                elif item.action == QueueAction.EMBED_FALLBACK:
                    await asyncio.wait_for(
                        self._pipeline.embed_fallback(item.file_path),
                        timeout=timeout,
                    )

                # Success — remove from queue
                async with self._lock:
                    self._items.pop(item.file_path, None)
                    self._cleanup_dest_map(item)
                self._publish(Update.create(
                    UpdateOpcode.QUEUE_ITEM_COMPLETED, **item.to_dict()
                ))
                await self._delete_item_from_db(item.id)
                logger.debug("Queue item completed", file_path=item.file_path, action=item.action.value)

            except DatabaseClosingError:
                logger.debug("Queue worker stopping (DB closing during shutdown)")
                return
            except asyncio.CancelledError:
                # cancel_in_flight() reached us. Reset the queue item
                # to WAITING with a fresh cooldown so the next dispatch
                # pass picks it up. The file's row in `files` keeps its
                # last persisted stage (PARSED or SUMMARIZED) so the
                # pipeline resumes from the right point — no work lost.
                # Crucially: re-raise CancelledError so asyncio's task
                # bookkeeping is correct. Anything we set on `item`
                # below is fine to do before re-raising.
                logger.info("Queue item cancelled mid-flight",
                            file_path=item.file_path, action=item.action.value)
                async with self._lock:
                    item.status = QueueItemStatus.WAITING
                    item.cooldown_expires_at = time.time() + self._config.initial_cooldown_seconds
                await self._save_item(item)
                raise
            except Exception as e:
                item.retry_count += 1
                # Deterministic failures should not be retried
                non_retryable = (
                    "File size too large" in str(e)
                    or "Unsupported file format" in str(e)
                    or "Too many chunks to summarize" in str(e)
                    or "File too large to summarize" in str(e)
                )
                # EMBED_FALLBACK actions get a single attempt only
                is_fallback = item.action == QueueAction.EMBED_FALLBACK
                if is_fallback or non_retryable or item.retry_count >= self._config.max_retries:
                    async with self._lock:
                        self._items.pop(item.file_path, None)
                        self._cleanup_dest_map(item)
                    item_data = item.to_dict()
                    item_data["error"] = str(e)
                    self._publish(Update.create(
                        UpdateOpcode.QUEUE_ITEM_FAILED, **item_data
                    ))
                    await self._delete_item_from_db(item.id)

                    # For INDEX failures, schedule a fallback embedding pass so
                    # the file is still discoverable via semantic search.
                    if item.action == QueueAction.INDEX:
                        try:
                            await self.enqueue(item.file_path, QueueAction.EMBED_FALLBACK)
                            logger.info("Enqueued fallback embedding after permanent failure",
                                        file_path=item.file_path)
                        except Exception:
                            logger.exception("Failed to enqueue fallback embedding",
                                             file_path=item.file_path)

                    logger.error("Queue item failed permanently",
                                 file_path=item.file_path, error=str(e),
                                 retry_count=item.retry_count)
                else:
                    # Exponential backoff: base * 2^(retry-1), capped at 8x base
                    backoff = self._config.cooldown_seconds * (2 ** (item.retry_count - 1))
                    max_backoff = self._config.cooldown_seconds * 8
                    item.status = QueueItemStatus.COOLING_DOWN
                    item.cooldown_expires_at = time.time() + min(backoff, max_backoff)
                    self._publish(Update.create(
                        UpdateOpcode.QUEUE_ITEM_UPDATED, **item.to_dict()
                    ))
                    await self._save_item(item)
                    logger.warning("Queue item failed, will retry",
                                   file_path=item.file_path, error=str(e),
                                   retry_count=item.retry_count,
                                   backoff_seconds=min(backoff, max_backoff))
            finally:
                # Check if a new event arrived while we were processing
                await self._handle_pending_reenqueue(item.file_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    async def _load_from_db(self) -> None:
        """Load persisted queue items from the database on startup."""
        if self._db is None:
            return
        try:
            rows = await self._db.get_queue_items()
            for row in rows:
                try:
                    action = QueueAction(row["action"])
                except ValueError:
                    logger.error("Invalid queue action type, skipping item",
                                action=row["action"], item_id=row["id"])
                    continue
                # Restore PROCESSING items as WAITING so they get re-processed.
                # Also map legacy "ready" values to WAITING.
                raw_status = row["status"]
                if raw_status in ("processing", "ready"):
                    status = QueueItemStatus.WAITING
                else:
                    status = QueueItemStatus(raw_status)

                item = QueueItem(
                    id=row["id"],
                    file_path=row["file_path"],
                    action=action,
                    status=status,
                    enqueued_at=row["enqueued_at"],
                    cooldown_expires_at=row["cooldown_expires_at"],
                    dest_path=row.get("dest_path"),
                    retry_count=row.get("retry_count", 0),
                )
                self._items[item.file_path] = item
                if action == QueueAction.MOVE and item.dest_path:
                    self._dest_to_src[item.dest_path] = item.file_path

            if rows:
                logger.info("Loaded queue items from database", count=len(rows))
        except Exception:
            logger.exception("Failed to load queue items from database")

    async def _save_item(self, item: QueueItem) -> None:
        """Persist a queue item to the database."""
        if self._db is None:
            return
        try:
            await self._db.upsert_queue_item(item.to_dict())
        except Exception:
            logger.exception("Failed to persist queue item", file_path=item.file_path)

    async def _delete_item_from_db(self, item_id: str) -> None:
        """Remove a queue item from the database."""
        if self._db is None:
            return
        try:
            await self._db.delete_queue_item(item_id)
        except Exception:
            logger.exception("Failed to delete queue item from DB", item_id=item_id)

    async def _handle_pending_reenqueue(self, file_path: str) -> None:
        """Re-enqueue a file if a new event arrived while it was being processed."""
        async with self._lock:
            pending = self._pending_reenqueue.pop(file_path, None)
        if pending is not None:
            action, dest = pending
            logger.info("Re-enqueueing after processing completed",
                        file_path=file_path, action=action.value)
            await self.enqueue(file_path, action, dest_path=dest)

    def _cleanup_dest_map(self, item: QueueItem) -> None:
        """Remove the dest_path → src_path reverse mapping for a MOVE item."""
        if item.action == QueueAction.MOVE and item.dest_path:
            self._dest_to_src.pop(item.dest_path, None)

    def _publish(self, update: Update) -> None:
        if self._updates_hub:
            self._updates_hub.publish(update)
