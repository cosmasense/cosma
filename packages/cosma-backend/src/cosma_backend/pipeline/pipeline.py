from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional
import logging

from cosma_backend.db import Database
from cosma_backend.logging import sm
from cosma_backend.models import File
from cosma_backend.models.status import ProcessingStatus
from cosma_backend.models.update import Update
from cosma_backend.discoverer import Discoverer
from cosma_backend.parser import FileParser
from cosma_backend.summarizer import AutoSummarizer
from cosma_backend.embedder import AutoEmbedder
from cosma_backend.utils.pubsub import Hub

logger = logging.getLogger(__name__)


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
    
    async def process_directory(self, path: str | Path):
        """
        Process all files in a directory through the full pipeline.
        After processing, deletes any files from the database that weren't seen
        (i.e., files that no longer exist in the filesystem).
        
        Args:
            path: Root directory to process
            
        Returns:
            PipelineResult with statistics
        """
        # result = PipelineResult()
        
        # Publish directory processing started
        self._publish_update(Update.directory_processing_started(str(path)))
        
        started_processing = datetime.now(timezone.utc)
        
        # Stage 1: Discovery
        logger.info(f"Discovering files in {path}")
        for file in self.discoverer.files_in(path):
            # result.discovered += 1
            
            try:
                # Update the timestamp to mark this file as still present in the filesystem
                await self.db.update_file_timestamp(file.file_path)

                # Check if file needs processing
                if await self._should_skip_file(file):
                    logger.info(sm("Skipping processing file", file=file))
                    self._publish_update(Update.file_skipped(
                        file.file_path, 
                        file.filename, 
                        reason="already processed"
                    ))
                    # result.skipped += 1
                    continue
                
                # Process the file through the pipeline
                await self.process_file(file)
                
            except Exception:
                continue
        
        try:
            logger.info(sm("Deleting files no longer present in filesystem", started_processing=started_processing, path=str(path)))
            rows = await self.db.delete_files_not_updated_since(started_processing, str(path))
            logger.info(sm("Deleted unused files", count=len(rows)))
        except Exception as e:
            logger.error(sm("Error while deleting unused files", error=str(e)))

        logger.info(sm("Completed processing directory", directory=str(path)))
        self._publish_update(Update.directory_processing_completed(str(path)))

    
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
                logger.info(sm("Skipping processing file, hashed not changed", file=file))
                self._publish_update(Update.file_skipped(
                    file.file_path, 
                    file.filename, 
                    reason="content not changed"
                ))
                return
            
            # Stage 2: Summarize
            self._publish_update(Update.file_summarizing(file.file_path, file.filename))
            await self.summarizer.summarize(file)
            await self._save_to_db(file)
            self._publish_update(Update.file_summarized(file.file_path, file.filename))
            
            # Stage 3: Embed (if embedder is available)
            self._publish_update(Update.file_embedding(file.file_path, file.filename))
            await self.embedder.embed(file)
            # embeddings need special care when saving
            await self._save_embeddings(file)
            self._publish_update(Update.file_embedded(file.file_path, file.filename))
            
            # Mark as complete
            self._publish_update(Update.file_complete(file.file_path, file.filename))
            
        except Exception as e:
            # result.failed += 1
            # result.errors.append((str(file_path), str(e)))
            logger.error(sm("Pipeline failed for file", file=file, error=e))
            
            # Publish failure update
            self._publish_update(Update.file_failed(
                file.file_path, 
                file.filename, 
                error=str(e)
            ))
            
            # Save failed state to DB if we have file_data
            file.status = ProcessingStatus.FAILED
            file.processing_error = str(e)
            await self._save_to_db(file)
                
            raise e
            
    async def is_supported(self, file: File) -> bool:
        """Check if a file is supported for processing"""
        return self.parser.is_supported(file)
    
    async def _should_skip_file(self, file: File) -> bool:
        """Check if file should be skipped based on DB state."""
        if not await self.is_supported(file):
            return False
        
        saved_file = await self.db.get_file_by_path(file.file_path)
        
        if not saved_file or saved_file.status not in (ProcessingStatus.COMPLETE, ProcessingStatus.FAILED):
            logger.info(sm("Should skip", file=file, status=saved_file.status if saved_file else "No saved file"))
            return False
            
        saved_modified = saved_file.modified.replace(microsecond=0)
        current_modified = file.modified.replace(microsecond=0)
        
        logger.info(sm("Should skip", file=file, saved_modified=saved_modified, current_modified=current_modified))
            
        return saved_modified == current_modified



    async def _has_file_changed(self, file: File) -> bool:
        """Check if file has been changed based on hash."""
        saved_file = await self.db.get_file_by_path(file.file_path)
        
        logger.info(sm("Saved file", saved_file=saved_file, status=saved_file.status if saved_file else "N/A"))
        
        if not saved_file or saved_file.status is not ProcessingStatus.COMPLETE:
            return True
            
        return saved_file.content_hash != file.content_hash
    
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
