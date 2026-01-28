import asyncio
import datetime
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from watchdog.events import (
    FileSystemEvent,
    FileCreatedEvent,
    FileModifiedEvent,
    FileDeletedEvent,
    FileMovedEvent,
    DirCreatedEvent,
    DirModifiedEvent,
    DirDeletedEvent,
    DirMovedEvent,
)
from watchdog.observers.api import BaseObserver

from cosma_backend.db import Database
from cosma_backend.logging import sm
from cosma_backend.models import File
from cosma_backend.models.watch import WatchedDirectory
from cosma_backend.models.update import Update
from cosma_backend.pipeline import Pipeline
from cosma_backend.utils.pubsub import Hub
from cosma_backend.watcher.awatchdog import watch

if TYPE_CHECKING:
    from cosma_backend.filter import FilterConfigManager

logger = logging.getLogger(__name__)

# Per-folder config file name to watch for changes
LOCAL_CONFIG_FILENAME = ".cosmaconfig"


class WatcherJob:
    pipeline: Pipeline
    db: Database
    watched_dir: WatchedDirectory
    queue: asyncio.Queue[FileSystemEvent]
    observer: Optional[BaseObserver]
    task: Optional[asyncio.Task]
    closed: bool
    filter_manager: Optional["FilterConfigManager"]

    def __init__(
        self,
        watched_dir: WatchedDirectory,
        pipeline: Pipeline,
        db: Database,
        filter_manager: Optional["FilterConfigManager"] = None
    ):
        self.watched_dir = watched_dir
        self.pipeline = pipeline
        self.db = db
        self.filter_manager = filter_manager
        self.queue = asyncio.Queue()
        self.observer = None
        self.closed = False
        self.task = None
    
    def _publish_update(self, update: Update):
        """Publish an update through the pipeline's updates hub."""
        if self.pipeline.updates_hub:
            self.pipeline.updates_hub.publish(update)
            
    async def do_initial_processing(self):
        # Get filter config for this directory
        filter_config = None
        if self.filter_manager is not None:
            filter_config = self.filter_manager.get_config_for_directory(self.watched_dir.path)

        await self.pipeline.process_directory(self.watched_dir.path, filter_config=filter_config)
    
    async def start(self):
        logger.info(sm("Starting watchdog observer", watched_dir=self.watched_dir))
        
        # Publish watch started update
        from cosma_backend.models.update import UpdateOpcode
        self._publish_update(Update.create(
            UpdateOpcode.WATCH_STARTED,
            path=str(self.watched_dir.path),
            recursive=self.watched_dir.recursive,
            file_pattern=self.watched_dir.file_pattern
        ))
        
        self.observer = await watch(self.watched_dir.path, self.queue, recursive=self.watched_dir.recursive)
        self.task = asyncio.create_task(self.consumer_task())
        asyncio.create_task(self.do_initial_processing())
        
    async def stop(self):
        self.closed = True
        if self.task is not None:
            self.task.cancel()
        if self.observer is not None:
            self.observer.unschedule_all()
        
    def _should_include_file(self, file_path: Path) -> bool:
        """Check if file should be included based on filter config."""
        if self.filter_manager is None:
            return True

        base_path = self.watched_dir.path
        return self.filter_manager.should_include_file(file_path, base_path)

    def _is_config_file(self, file_path: Path) -> bool:
        """Check if the file is a .cosmaconfig file."""
        return file_path.name == LOCAL_CONFIG_FILENAME

    def _handle_config_change(self, file_path: Path):
        """Handle changes to .cosmaconfig file."""
        if self.filter_manager is None:
            return

        # Check if this is the config file for our watched directory
        if file_path.parent == self.watched_dir.path:
            logger.info(sm("Config file changed, reloading filter config",
                          config_path=str(file_path),
                          watched_dir=str(self.watched_dir.path)))
            self.filter_manager.reload_directory(self.watched_dir.path)

    async def consumer_task(self):
        while not self.closed:
            event = await self.queue.get()

            # Skip directory created and modified events - we only care about deletion/movement
            if isinstance(event, (DirCreatedEvent, DirModifiedEvent)):
                logger.debug(sm("Skipping directory event", event_type=type(event).__name__, path=event.src_path))
                continue

            try:
                # Handle directory deletion
                if isinstance(event, DirDeletedEvent):
                    logger.info(sm("Directory deleted", path=event.src_path))
                    path = Path(str(event.src_path)).resolve()

                    self._publish_update(Update.directory_deleted(str(path)))
                    deleted_files = await self.db.delete_files_in_directory(str(path))
                    logger.info(sm("Deleted files from deleted directory", directory=str(path), count=len(deleted_files)))

                # Handle directory movement
                elif isinstance(event, DirMovedEvent):
                    logger.info(sm("Directory moved", src=event.src_path, dest=event.dest_path))
                    src_path = Path(str(event.src_path)).resolve()
                    dest_path = Path(str(event.dest_path)).resolve()

                    self._publish_update(Update.directory_moved(str(src_path), str(dest_path)))

                    # Delete all files at the old location
                    deleted_files = await self.db.delete_files_in_directory(str(src_path))
                    logger.info(sm("Deleted files from moved directory", old_directory=str(src_path), count=len(deleted_files)))

                    # Process the new directory location if it exists
                    if dest_path.exists() and dest_path.is_dir():
                        logger.info(sm("Processing files in new directory location", directory=str(dest_path)))
                        await self.pipeline.process_directory(dest_path)

                # Handle different event types
                elif isinstance(event, FileDeletedEvent):
                    logger.info(sm("File deleted", path=event.src_path))
                    path = Path(str(event.src_path)).resolve()

                    # Check if config file was deleted
                    if self._is_config_file(path):
                        self._handle_config_change(path)

                    self._publish_update(Update.file_deleted(str(path)))
                    await self.db.delete_file(str(path))

                elif isinstance(event, FileMovedEvent):
                    # Handle moved files as delete old + create new
                    logger.info(sm("File moved", src=event.src_path, dest=event.dest_path))
                    src_path = Path(str(event.src_path)).resolve()
                    dest_path = Path(str(event.dest_path)).resolve()

                    self._publish_update(Update.file_moved(str(src_path), str(dest_path)))
                    await self.db.delete_file(str(src_path))

                    # Check if config file was moved
                    if self._is_config_file(dest_path):
                        self._handle_config_change(dest_path)
                        continue  # Don't index config files

                    # Check filter before processing destination
                    if not self._should_include_file(dest_path):
                        logger.debug(sm("Skipping excluded file", path=str(dest_path)))
                        continue

                    # Check if destination file type is supported before processing
                    dest_file = File.from_path(dest_path)
                    if await self.pipeline.is_supported(dest_file):
                        await self.pipeline.process_file(dest_file)
                    else:
                        logger.debug(sm("Skipping unsupported file type", path=str(dest_path)))

                elif isinstance(event, (FileCreatedEvent, FileModifiedEvent)):
                    # Handle created and modified files the same way - process them
                    event_type = "created" if isinstance(event, FileCreatedEvent) else "modified"
                    logger.info(sm(f"File {event_type}", path=event.src_path))
                    path = Path(str(event.src_path)).resolve()

                    # Check if config file changed
                    if self._is_config_file(path):
                        self._handle_config_change(path)
                        continue  # Don't index config files

                    # Check filter before processing
                    if not self._should_include_file(path):
                        logger.debug(sm("Skipping excluded file", path=str(path)))
                        continue

                    # Publish file system event update
                    if isinstance(event, FileCreatedEvent):
                        self._publish_update(Update.file_created(str(path)))
                    else:
                        self._publish_update(Update.file_modified(str(path)))

                    # Check if file type is supported before processing
                    file = File.from_path(path)
                    if await self.pipeline.is_supported(file):
                        await self.pipeline.process_file(file)
                    else:
                        logger.debug(sm("Skipping unsupported file type", path=str(path)))

                else:
                    logger.warning(sm("Unknown event type", event_type=type(event).__name__, path=event.src_path))

            except Exception as e:
                logger.error(sm("Error processing file system event", event_type=type(event).__name__, path=event.src_path, error=e))
    

class Watcher:
    jobs: set[WatcherJob]
    filter_manager: Optional["FilterConfigManager"]

    def __init__(
        self,
        db: Database,
        pipeline: Pipeline,
        updates_hub: Optional[Hub] = None,
        filter_manager: Optional["FilterConfigManager"] = None,
    ):
        self.db = db
        self.pipeline = pipeline
        self.updates_hub = updates_hub
        self.filter_manager = filter_manager

        self.jobs = set()

    async def create_job(self, watched_dir: WatchedDirectory):
        """
        Create and start a watcher job for a watched directory.

        Args:
            watched_dir: WatchedDirectory instance to watch
        """
        job = WatcherJob(watched_dir, self.pipeline, self.db, self.filter_manager)
        self.jobs.add(job)
        await job.start()

    async def stop_watching(self, path: Path | str) -> bool:
        """
        Stop watching a directory.

        Args:
            path: Path to the directory to stop watching

        Returns:
            True if a job was found and stopped, False otherwise
        """
        path = Path(path).resolve()

        # Find the job for this path
        job_to_stop = None
        for job in self.jobs:
            if job.watched_dir.path == path:
                job_to_stop = job
                break

        if job_to_stop is None:
            logger.warning(sm("No watcher job found for path", path=str(path)))
            return False

        # Stop the job
        await job_to_stop.stop()
        self.jobs.discard(job_to_stop)

        logger.info(sm("Stopped watching directory", path=str(path)))
        return True

    async def stop_watching_by_id(self, job_id: int) -> bool:
        """
        Stop watching a directory by its database ID.

        Args:
            job_id: Database ID of the watched directory

        Returns:
            True if a job was found and stopped, False otherwise
        """
        # Find the job for this ID
        job_to_stop = None
        for job in self.jobs:
            if job.watched_dir.id == job_id:
                job_to_stop = job
                break

        if job_to_stop is None:
            logger.warning(sm("No watcher job found for ID", job_id=job_id))
            return False

        # Stop the job
        await job_to_stop.stop()
        self.jobs.discard(job_to_stop)

        logger.info(sm("Stopped watching directory by ID", job_id=job_id, path=str(job_to_stop.watched_dir.path)))
        return True

    async def start_watching(self, path: str | Path, recursive: bool = True, file_pattern: Optional[str] = None):
        """
        Start watching a directory for file changes.
        
        Args:
            path: Path to the directory to watch
            recursive: Whether to watch subdirectories recursively
            file_pattern: Optional glob pattern for filtering files (e.g., "*.pdf")
            
        Raises:
            ValueError: If the directory is already being watched or a parent directory is being watched
        """
        # Create WatchedDirectory model
        path = Path(path).resolve()
        
        # Check if this directory or any parent is already being watched
        watched_dirs = await self.db.get_watched_directories(active_only=True)
        
        for existing_dir in watched_dirs:
            # Check if exact path is already being watched
            if existing_dir.path == path:
                logger.warning(sm("Directory already being watched", path=str(path)))
                raise ValueError(f"Directory '{path}' is already being watched")
            
            # Check if a parent directory is being watched with recursive=True
            # A parent is watching this path if:
            # 1. The parent is recursive
            # 2. This path starts with the parent path
            if existing_dir.recursive:
                try:
                    # Use relative_to to check if path is a subdirectory of existing_dir.path
                    path.relative_to(existing_dir.path)
                    # If we get here, path is a subdirectory of existing_dir.path
                    logger.warning(sm("Parent directory already being watched", 
                                     path=str(path), 
                                     parent=str(existing_dir.path)))
                    raise ValueError(f"Parent directory '{existing_dir.path}' is already watching '{path}' recursively")
                except ValueError:
                    # relative_to raises ValueError if path is not relative to existing_dir.path
                    # This means it's not a subdirectory, so continue checking
                    pass
        
        watched_dir = WatchedDirectory.from_path(path, recursive=recursive, file_pattern=file_pattern)
        
        # Add directory to watched_directories table in database
        await self.db.add_watched_directory(watched_dir)
        
        # Create and start the watcher job
        await self.create_job(watched_dir)

    async def initialize_from_database(self):
        """
        Create jobs for all active watched directories from the database.
        This should be called on startup to restore watching state.
        """
        logger.info(sm("Initializing watcher from database"))
        
        # Get all active watched directories
        watched_dirs = await self.db.get_watched_directories(active_only=True)
        
        if not watched_dirs:
            logger.info(sm("No watched directories found in database"))
            return
        
        logger.info(sm("Found watched directories", count=len(watched_dirs)))
        
        # Create jobs for each watched directory
        for watched_dir in watched_dirs:
            # Check if path still exists
            if not watched_dir.path.exists():
                logger.warning(sm("Watched directory no longer exists", path=watched_dir.path_str, id=watched_dir.id))
                continue
            
            if not watched_dir.path.is_dir():
                logger.warning(sm("Watched path is not a directory", path=watched_dir.path_str, id=watched_dir.id))
                continue
            
            logger.info(sm("Creating job for watched directory", 
                          path=watched_dir.path_str, 
                          id=watched_dir.id,
                          recursive=watched_dir.recursive,
                          file_pattern=watched_dir.file_pattern))
            try:
                await self.create_job(watched_dir)
            except Exception as e:
                logger.error(sm("Failed to create job for watched directory", 
                               path=watched_dir.path_str, 
                               id=watched_dir.id,
                               error=str(e)))
        
        logger.info(sm("Watcher initialization complete", active_jobs=len(self.jobs)))
