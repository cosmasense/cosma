"""
File Processing Pipeline

Coordinates the four-stage indexing process:
1. Discovery - Walk filesystem, apply include/exclude filters
2. Parsing - Extract text via Spotlight, MarkItDown, or Whisper
3. Summarization - AI-generated title, summary, keywords
4. Embedding - Vector representation for semantic search

Key behaviors:
- Skips unchanged files (based on content hash)
- Applies fallback indexing for failed files (filename → metadata)
- Emits SSE updates at each stage for UI progress
- Supports both directory-batch and single-file processing
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Optional

from cosma_backend.db import Database
from cosma_backend.db.errors import DatabaseClosingError
from cosma_backend.dev_logging import log_failed_file
from cosma_backend.discoverer import Discoverer
from cosma_backend.embedder import AutoEmbedder
from cosma_backend.logging import get_logger
from cosma_backend.models import File
from cosma_backend.models.status import ProcessingStatus
from cosma_backend.models.update import Update
from cosma_backend.parser import FileParser
from cosma_backend.summarizer import AutoSummarizer
from cosma_backend.utils.pubsub import Hub

if TYPE_CHECKING:
    from cosma_backend.filter import FilterConfig
    from cosma_backend.queue import IndexingQueue

logger = get_logger(__name__)


class PipelineResult:
    """Results from processing a batch of files."""
    def __init__(self):
        self.discovered = 0
        self.skipped = 0
        self.parsed = 0
        self.summarized = 0
        self.embedded = 0
        self.failed = 0
        self.errors: list[tuple[str, str]] = []  # (file_path, error)


class Pipeline:
    """
    Main pipeline orchestrator. Processes files through:
    Discovery → Parsing → Summarization → Embedding
    """
    
    def __init__(
        self,
        db: Database,
        updates_hub: Optional[Hub] = None,
        discoverer: Optional[Discoverer] = None,
        parser: Optional[FileParser] = None,
        summarizer: Optional[AutoSummarizer] = None,
        embedder: Optional[AutoEmbedder] = None,
        parse_concurrency: int = 4,
        summarize_concurrency: int = 1,
        embed_concurrency: int = 1,
    ):
        self.db = db
        self.updates_hub = updates_hub
        self.discoverer = discoverer or Discoverer()
        self.parser = parser or FileParser()
        self.summarizer = summarizer or AutoSummarizer()
        self.embedder = embedder or AutoEmbedder()

        # Per-stage semaphores so files moving through the pipeline don't
        # all collide on the same resource. While file A holds the
        # summarize semaphore (single Metal LLM context), file B can
        # already be embedding on MPS and files C/D can be parsing on
        # CPU. asyncio.Semaphore is the implicit backpressure: a
        # finished-parse waits at `await summarize_sem.acquire()` for
        # zero CPU until the GPU frees up. See
        # cosma/docs/STAGE_PIPELINING_DESIGN.md.
        self._parse_sem = asyncio.Semaphore(parse_concurrency)
        self._summarize_sem = asyncio.Semaphore(summarize_concurrency)
        self._embed_sem = asyncio.Semaphore(embed_concurrency)
    
    async def process_directory(
        self,
        path: str | Path,
        filter_config: Optional["FilterConfig"] = None
    ):
        """
        Process all files in a directory through the full pipeline.
        After processing, deletes any files from the database that weren't seen
        (i.e., files that no longer exist in the filesystem).

        Args:
            path: Root directory to process
            filter_config: Optional filter config to exclude files

        Returns:
            PipelineResult with statistics
        """
        # result = PipelineResult()

        # Publish directory processing started
        self._publish_update(Update.directory_processing_started(str(path)))

        started_processing = datetime.now(timezone.utc)

        # Stage 1: Discovery (with filtering)
        logger.info("Discovering files", path=str(path),
                      has_filter=filter_config is not None)
        for file in self.discoverer.files_in(path, filter_config=filter_config):
            # result.discovered += 1
            
            try:
                # Update the timestamp to mark this file as still present in the filesystem
                await self.db.update_file_timestamp(file.file_path)

                # Check if file needs processing
                should_skip, saved_file = await self._should_skip_file(file)
                if should_skip:
                    logger.info("Skipping processing file", file=file)
                    self._publish_update(Update.file_skipped(
                        file.file_path,
                        file.filename,
                        reason="already processed"
                    ))
                    # result.skipped += 1
                    continue

                # Process the file through the pipeline
                await self.process_file(file)

            except DatabaseClosingError:
                logger.debug("Skipping file processing (DB closing during shutdown)")
                return
            except Exception:
                continue

        try:
            logger.info("Deleting files no longer present in filesystem", started_processing=started_processing, path=str(path))
            rows = await self.db.delete_files_not_updated_since(started_processing, str(path))
            logger.info("Deleted unused files", count=len(rows))
        except DatabaseClosingError:
            logger.debug("Skipped stale-file cleanup (DB closing during shutdown)")
        except Exception as e:
            logger.error("Error while deleting unused files", error=str(e))

        logger.info("Completed processing directory", directory=str(path))
        self._publish_update(Update.directory_processing_completed(str(path)))

    async def enqueue_directory(
        self,
        path: str | Path,
        indexing_queue: "IndexingQueue",
        filter_config: Optional["FilterConfig"] = None,
    ) -> int:
        """
        Discover files in a directory and enqueue them for processing via the
        IndexingQueue instead of processing inline.  This makes items visible
        in the queue view and benefits from cooldown/debounce.

        After enqueueing, schedules a deferred cleanup task to remove stale
        files (files that no longer exist in the filesystem).

        Returns the number of files enqueued.

        Performance note (v0.8.7+): the per-file path used to issue 3
        separate DB roundtrips (`update_file_timestamp` + `get_file_by_path`
        + `enqueue` writes), which scaled badly to 10k-file folders. We
        now bulk-load the directory's saved (status, modified) state into
        a dict once, do skip checks in memory, and batch the
        "mark file as still on disk" timestamp updates into chunked
        UPDATE ... WHERE file_path IN (...) statements at the end.
        """
        from cosma_backend.queue import QueueAction

        self._publish_update(Update.directory_processing_started(str(path)))

        started_processing = datetime.now(timezone.utc)
        enqueued = 0

        # One DB query for the whole directory instead of one per file.
        # Empty dict on error so we degrade to "treat everything as new"
        # rather than crash the discovery sweep.
        try:
            saved_summary = await self.db.get_files_under_directory_summary(str(path))
        except DatabaseClosingError:
            logger.debug("Skipping bulk skip-check (DB closing during shutdown)")
            return enqueued
        except Exception:
            logger.exception("Bulk skip-check failed; degrading to per-file path")
            saved_summary = {}

        # Buffer for batched timestamp touches. Capped so a watched folder
        # with millions of files doesn't blow up resident memory.
        TOUCH_BATCH = 1000
        to_touch: list[str] = []

        async def _flush_touches() -> None:
            nonlocal to_touch
            if not to_touch:
                return
            try:
                await self.db.touch_files_timestamps(to_touch)
            except DatabaseClosingError:
                logger.debug("Skipping timestamp touch (DB closing during shutdown)")
            except Exception:
                logger.exception("Batched timestamp touch failed")
            to_touch = []

        # File walk + filter + skip decision run in a thread, in chunks.
        #
        # Why: ``discoverer.files_in()`` is a synchronous generator
        # that walks the filesystem (os.scandir, stat, filter regex)
        # for every file. Iterating it directly on the event loop
        # blocks every other coroutine for the duration — and in the
        # already-indexed path there are no real awaits to break it
        # up (is_supported is sync-wrapped, the touch is batched
        # every 1000 files). With multiple watchers, ``asyncio.sleep(0)``
        # yields aren't enough either: 3 concurrent enqueue_directory
        # tasks split each yield round between themselves and starve
        # other tasks. Result: a 5-second ``asyncio.sleep`` in the
        # Phase 2 grace took 42 seconds to fire, search requests sat
        # for 30+ s and timed out (see temp/log.txt 20:01:15 → 20:01:57).
        #
        # Fix: pull files in chunks of ``CHUNK_SIZE`` from the sync
        # generator on a background thread, doing the filter +
        # already-indexed check there. The async side wakes once
        # per chunk, calls ``indexing_queue.enqueue`` (brief, yields)
        # for items that need work, and goes back for the next chunk.
        # Loop is never blocked synchronously regardless of folder
        # size or number of watchers.
        is_supported_fn = self.parser.is_supported
        generator = self.discoverer.files_in(path, filter_config=filter_config)
        CHUNK_SIZE = 200

        def _discover_chunk() -> tuple[list[tuple[str, "File"]], bool]:
            """Pull up to CHUNK_SIZE files off the sync generator and
            tag each with its disposition. Runs in a worker thread."""
            out: list[tuple[str, "File"]] = []
            done_flag = False
            for _ in range(CHUNK_SIZE):
                try:
                    f = next(generator)
                except StopIteration:
                    done_flag = True
                    break
                if not is_supported_fn(f):
                    out.append(("unsupported", f))
                    continue
                summary = saved_summary.get(f.file_path)
                if summary is not None:
                    saved_status, saved_mtime = summary
                    if (
                        saved_status in ("COMPLETE", "FAILED")
                        and saved_mtime is not None
                        and saved_mtime.replace(microsecond=0)
                        == f.modified.replace(microsecond=0)
                    ):
                        out.append(("skip", f))
                        continue
                out.append(("enqueue", f))
            return out, done_flag

        done = False
        while not done:
            chunk, done = await asyncio.to_thread(_discover_chunk)
            for action, file in chunk:
                if action == "unsupported":
                    continue
                try:
                    to_touch.append(file.file_path)
                    if len(to_touch) >= TOUCH_BATCH:
                        await _flush_touches()
                    if action == "skip":
                        self._publish_update(Update.file_skipped(
                            file.file_path, file.filename,
                            reason="already processed",
                        ))
                        continue
                    # action == "enqueue"
                    await indexing_queue.enqueue(
                        file.file_path, QueueAction.INDEX,
                        cooldown_seconds=indexing_queue.initial_cooldown_seconds,
                    )
                    enqueued += 1
                except DatabaseClosingError:
                    logger.debug("Skipping enqueue (DB closing during shutdown)")
                    await _flush_touches()
                    return enqueued
                except Exception:
                    logger.exception("Error enqueueing file",
                                     file_path=file.file_path)
                    continue

        # Final flush of any timestamps still buffered.
        await _flush_touches()

        # Stale-file cleanup runs immediately after discovery (all existing
        # files had their timestamps touched above).  Files that were NOT
        # touched are no longer on disk and can be pruned from the DB now.
        try:
            rows = await self.db.delete_files_not_updated_since(started_processing, str(path))
            if rows:
                logger.info("Deleted stale files after enqueue", count=len(rows), directory=str(path))
        except DatabaseClosingError:
            logger.debug("Skipped stale-file cleanup (DB closing during shutdown)")
        except Exception:
            logger.exception("Error during stale-file cleanup")

        # NOTE: We intentionally do NOT publish directory_processing_completed
        # here.  The queue items are still cooling down / processing.  The
        # frontend uses queue-based progress tracking (queue_item_added /
        # queue_item_completed SSE events) to compute the real progress and
        # will mark the folder as complete when all queue items finish.

        logger.info("Enqueued directory for processing", directory=str(path), enqueued=enqueued)
        return enqueued

    async def process_file(self, file: File):
        """
        Process a single file through the pipeline.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Processed File or None if failed
        """
        # if result is None:
        #     result = PipelineResult()

        try:
            # Pre-parse skip check: if the file is already COMPLETE/FAILED in
            # the DB and its mtime hasn't changed, don't re-parse. The watcher
            # enqueues every FileModifiedEvent without consulting the DB, so
            # without this guard noisy filesystem events (or repeated initial
            # discoveries) cause the same file to be parsed/summarized/embedded
            # over and over. _should_skip_file also rejects unsupported types.
            should_skip, saved_file = await self._should_skip_file(file)
            if should_skip:
                self._publish_update(Update.file_skipped(
                    file.file_path,
                    file.filename,
                    reason="already indexed (mtime unchanged)"
                ))
                return

            # Resume-aware orchestration. Each stage method takes its own
            # semaphore so files in different stages run in parallel
            # (parse on CPU, summarize on GPU, embed on MPS) without
            # fighting for the same resource. Persisting after every
            # stage means a crash leaves the row at the last completed
            # state — next launch resumes from there instead of redoing
            # parse + summarize from scratch. See
            # cosma/docs/STAGE_PIPELINING_DESIGN.md.

            # Stage 1: Parse (always runs unless we're resuming from
            # PARSED+ where we already have the content). For FAILED rows
            # we re-run from scratch — the previous failure may have left
            # partial fields the user wants overwritten.
            resume_status = saved_file.status if saved_file else None
            need_parse = resume_status not in (
                ProcessingStatus.PARSED,
                ProcessingStatus.SUMMARIZED,
            )
            if need_parse:
                await self._run_parse_stage(file)

                # Hash-level check (catches mtime-changed-but-content-same).
                # Only meaningful when we just re-parsed; if we resumed
                # from PARSED+, content_hash is already settled.
                if not await self._has_file_changed(file, saved_file=saved_file):
                    logger.info("Skipping processing file, hash not changed",
                                file=file)
                    self._publish_update(Update.file_skipped(
                        file.file_path,
                        file.filename,
                        reason="content not changed"
                    ))
                    # Touch the row so stale-file sweep leaves it alone.
                    if saved_file is not None:
                        try:
                            await self.db.update_file_timestamp(file.file_path)
                        except Exception:
                            logger.debug("update_file_timestamp failed",
                                         file=file.file_path)
                    return
            else:
                # Hydrate the in-memory File with what's already in the DB
                # so summarize/embed don't have to re-parse.
                file.content = saved_file.content
                file.content_hash = saved_file.content_hash
                file.content_type = saved_file.content_type
                file.parsed_at = saved_file.parsed_at
                logger.info("Resuming from PARSED state",
                            file=file.file_path,
                            saved_status=resume_status.name)

            # Stage 2: Summarize (skipped if resuming from SUMMARIZED+).
            need_summarize = resume_status != ProcessingStatus.SUMMARIZED
            if need_summarize:
                await self._run_summarize_stage(file)
            else:
                # Hydrate summary/title/keywords from DB.
                file.title = saved_file.title
                file.summary = saved_file.summary
                file.keywords = saved_file.keywords
                file.summarized_at = saved_file.summarized_at
                logger.info("Resuming from SUMMARIZED state",
                            file=file.file_path)

            # Stage 3: Embed — non-fatal; file is still FTS-searchable
            # without embeddings.
            #
            # `first_embed` lets the DB layer skip the leading
            # `DELETE FROM file_embeddings` when there is nothing to
            # delete. True iff this file has not previously reached the
            # COMPLETE status, which is also the only case where a row
            # could already exist in the vec0 embeddings table.
            first_embed = (
                saved_file is None
                or saved_file.status != ProcessingStatus.COMPLETE
            )
            await self._run_embed_stage(file, first_embed=first_embed)

            # Mark as complete (UI event; row was already persisted by
            # the embed stage with status=COMPLETE).
            self._publish_update(Update.file_complete(file.file_path, file.filename))
            
        except DatabaseClosingError:
            logger.debug("Skipping file processing (DB closing during shutdown)")
            raise
        except Exception as e:
            # result.failed += 1
            # result.errors.append((str(file_path), str(e)))
            logger.error("Pipeline failed for file", file=file, error=e)

            # Publish failure update
            self._publish_update(Update.file_failed(
                file.file_path,
                file.filename,
                error=str(e)
            ))

            # Save failed state to DB if we have file_data
            file.status = ProcessingStatus.FAILED
            file.processing_error = str(e)
            log_failed_file(
                file_path=file.file_path,
                extension=file.extension,
                error=str(e),
                phase="pipeline",
            )
            self._apply_fallback_indexing(file)
            await self._save_to_db(file)

            raise
            
    # ------------------------------------------------------------------
    # Per-stage methods (used by process_file). Each holds its own
    # semaphore so different files in different stages don't fight for
    # the same resource. Each persists progress on success so a crash
    # mid-pipeline leaves a resumable DB state — the next run picks up
    # at the last completed stage instead of redoing parse + summarize
    # from scratch. See cosma/docs/STAGE_PIPELINING_DESIGN.md.
    # ------------------------------------------------------------------

    async def _run_parse_stage(self, file: File) -> None:
        """Parse the file, set status=PARSED, persist."""
        async with self._parse_sem:
            self._publish_update(Update.file_parsing(file.file_path, file.filename))
            await self.parser.parse_file(file)
            self._publish_update(Update.file_parsed(file.file_path, file.filename))

            # Persist parsed content + content_hash so a crash here is
            # recoverable. The status flip to PARSED is the durable
            # checkpoint: on next launch, crash recovery will route this
            # file to the summarize stage instead of re-parsing.
            file.status = ProcessingStatus.PARSED
            await self._save_to_db(file)

    async def _run_summarize_stage(self, file: File) -> None:
        """Summarize, set status=SUMMARIZED, persist."""
        async with self._summarize_sem:
            self._publish_update(Update.file_summarizing(file.file_path, file.filename))
            await self.summarizer.summarize(file)
            file.status = ProcessingStatus.SUMMARIZED
            await self._save_to_db(file)
            self._publish_update(Update.file_summarized(file.file_path, file.filename))

    async def _run_embed_stage(
        self, file: File, *, first_embed: bool = True,
    ) -> None:
        """Embed, set status=COMPLETE, persist embeddings.

        ``first_embed=True`` (the default) tells the DB layer this file
        has no pre-existing row in the vec0 embeddings table, so it
        can skip the leading DELETE. Pass False on a re-embed (e.g.
        forced reindex of a previously-COMPLETE file) so stale vectors
        are cleared.

        Embedding failure is non-fatal — the file is still searchable
        via FTS5 even without semantic vectors, so we mark COMPLETE
        and log a warning instead of failing the whole pipeline.
        """
        async with self._embed_sem:
            try:
                self._publish_update(Update.file_embedding(file.file_path, file.filename))
                await self.embedder.embed(file)
                file.status = ProcessingStatus.COMPLETE
                await self._save_embeddings(file, first_embed=first_embed)
                self._publish_update(Update.file_embedded(file.file_path, file.filename))
            except Exception as embed_err:
                logger.warning(
                    "Embedding failed, file saved without embeddings",
                    file=file.file_path, error=str(embed_err),
                )
                file.status = ProcessingStatus.COMPLETE
                await self._save_to_db(file)

    async def is_supported(self, file: File) -> bool:
        """Check if a file is supported for processing"""
        return self.parser.is_supported(file)
    
    async def _should_skip_file(self, file: File) -> tuple[bool, File | None]:
        """Check if file should be skipped based on DB state.

        Returns (should_skip, saved_file) so callers can reuse the DB record.
        """
        if not await self.is_supported(file):
            logger.debug("Skipping unsupported file", file=file.file_path)
            return True, None

        saved_file = await self.db.get_file_by_path(file.file_path)

        # File not in DB or not yet fully processed - don't skip
        if not saved_file or saved_file.status not in (ProcessingStatus.COMPLETE, ProcessingStatus.FAILED):
            logger.debug("File needs processing", file=file.file_path, status=saved_file.status if saved_file else "not in DB")
            return False, saved_file

        saved_modified = saved_file.modified.replace(microsecond=0)
        current_modified = file.modified.replace(microsecond=0)

        logger.info("Should skip", file=file, saved_modified=saved_modified, current_modified=current_modified)

        return saved_modified == current_modified, saved_file

    async def _has_file_changed(self, file: File, saved_file: File | None = None) -> bool:
        """Check if file content has changed based on hash.

        Accepts an optional pre-fetched ``saved_file`` to avoid a redundant
        DB query when the caller already has it.
        """
        if saved_file is None:
            saved_file = await self.db.get_file_by_path(file.file_path)

        logger.info("Saved file", saved_file=saved_file, status=saved_file.status if saved_file else "N/A")

        if not saved_file or saved_file.status is not ProcessingStatus.COMPLETE:
            return True

        return saved_file.content_hash != file.content_hash
    
    def _apply_fallback_indexing(self, file: File) -> None:
        """Generate searchable title/summary/keywords from the filename so that
        failed files are still discoverable via FTS5."""
        stem = Path(file.filename).stem

        # Title: replace separators with spaces, title-case
        title = re.sub(r"[_\-.]", " ", stem)
        # Split camelCase: insert space before uppercase letters preceded by lowercase
        title = re.sub(r"([a-z])([A-Z])", r"\1 \2", title)
        file.title = title.title()

        file.summary = f"File: {file.filename} (processing failed, indexed by filename)"

        # Keywords: split on separators and camelCase boundaries, deduplicate
        tokens = re.split(r"[_\-.]", stem)
        expanded: list[str] = []
        for token in tokens:
            # Split camelCase within each token
            parts = re.sub(r"([a-z])([A-Z])", r"\1 \2", token).split()
            expanded.extend(parts)
        seen: set[str] = set()
        keywords: list[str] = []
        for t in expanded:
            low = t.lower()
            if low and low not in seen:
                seen.add(low)
                keywords.append(low)
        file.keywords = keywords

    async def embed_fallback(self, file_path: str) -> None:
        """Generate embeddings for a failed file using its fallback title/summary/keywords.

        This makes failed files discoverable via semantic search even though
        full parsing/summarization failed.  The file's status stays FAILED.
        """
        file = await self.db.get_file_by_path(file_path)
        if file is None:
            logger.warning("embed_fallback: file not found in DB", file_path=file_path)
            return
        if not file.title and not file.summary:
            logger.info("embed_fallback: no fallback content, skipping", file_path=file_path)
            return

        logger.info("Generating fallback embedding", file_path=file_path)
        self._publish_update(Update.file_embedding(file.file_path, file.filename))
        await self.embedder.embed(file)
        # Keep the original FAILED status — this file is still incomplete
        file.status = ProcessingStatus.FAILED
        await self._save_embeddings(file)
        self._publish_update(Update.file_embedded(file.file_path, file.filename))
        logger.info("Fallback embedding complete", file_path=file_path)

    async def _save_to_db(self, file: File) -> None:
        """Save file data to database."""
        await self.db.upsert_file(file)
        
    async def _save_embeddings(
        self, file: File, *, first_embed: bool = False,
    ) -> None:
        """Save file embeddings to database. See db.upsert_file_embeddings
        for the meaning of `first_embed`."""
        await self._save_to_db(file)
        await self.db.upsert_file_embeddings(file, first_embed=first_embed)
            
    def _publish_update(self, update: Any):
        if self.updates_hub:
            self.updates_hub.publish(update)
