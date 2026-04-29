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
    ):
        self.db = db
        self.updates_hub = updates_hub
        self.discoverer = discoverer or Discoverer()
        self.parser = parser or FileParser()
        self.summarizer = summarizer or AutoSummarizer()
        self.embedder = embedder or AutoEmbedder()
    
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
        """
        from cosma_backend.queue import QueueAction

        self._publish_update(Update.directory_processing_started(str(path)))

        started_processing = datetime.now(timezone.utc)
        enqueued = 0

        for file in self.discoverer.files_in(path, filter_config=filter_config):
            try:
                # Touch the timestamp so stale-file cleanup knows this file still exists
                await self.db.update_file_timestamp(file.file_path)

                should_skip, _saved = await self._should_skip_file(file)
                if should_skip:
                    self._publish_update(Update.file_skipped(
                        file.file_path, file.filename, reason="already processed"
                    ))
                    continue

                await indexing_queue.enqueue(
                    file.file_path, QueueAction.INDEX,
                    cooldown_seconds=indexing_queue.initial_cooldown_seconds,
                )
                enqueued += 1
            except DatabaseClosingError:
                logger.debug("Skipping enqueue (DB closing during shutdown)")
                return enqueued
            except Exception:
                logger.exception("Error enqueueing file", file_path=file.file_path)
                continue

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
            # Stage 1: Parse
            self._publish_update(Update.file_parsing(file.file_path, file.filename))
            await self.parser.parse_file(file)
            self._publish_update(Update.file_parsed(file.file_path, file.filename))

            # Check if file hash is different before proceeding
            if not await self._has_file_changed(file):
                logger.info("Skipping processing file, hashed not changed", file=file)
                self._publish_update(Update.file_skipped(
                    file.file_path,
                    file.filename,
                    reason="content not changed"
                ))
                return

            # Delete any existing partial data so a crash-recovery retry
            # starts from a clean slate (cascades to embeddings, keywords, FTS).
            await self.db.delete_file(file.file_path)

            # Stage 2: Summarize
            self._publish_update(Update.file_summarizing(file.file_path, file.filename))
            await self.summarizer.summarize(file)
            await self._save_to_db(file)
            self._publish_update(Update.file_summarized(file.file_path, file.filename))
            
            # Stage 3: Embed — non-fatal; file is still FTS-searchable without embeddings
            try:
                self._publish_update(Update.file_embedding(file.file_path, file.filename))
                await self.embedder.embed(file)
                await self._save_embeddings(file)
                self._publish_update(Update.file_embedded(file.file_path, file.filename))
            except Exception as embed_err:
                logger.warning("Embedding failed, file saved without embeddings",
                               file=file.file_path, error=str(embed_err))
                # File is still COMPLETE for FTS — just missing semantic search
                file.status = ProcessingStatus.COMPLETE

            # Mark as complete
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
        
    async def _save_embeddings(self, file: File) -> None:
        """Save file embeddings to database."""
        await self._save_to_db(file)
        await self.db.upsert_file_embeddings(file)
            
    def _publish_update(self, update: Any):
        if self.updates_hub:
            self.updates_hub.publish(update)
