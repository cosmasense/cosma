"""
Cosma Backend Application

Main Quart web application that orchestrates all backend services:
- Database: SQLite with FTS5 for full-text search
- Pipeline: File discovery → parsing → summarization → embedding
- Searcher: Hybrid search combining semantic + full-text search
- Watcher: File system monitoring for automatic re-indexing
- Queue: Background processing with configurable scheduling
- Model Lifecycle: Automatic unloading of idle AI models

Application lifecycle:
1. App instance created, config loaded from env vars + TOML
2. `before_serving`: Initialize DB, services, start background tasks
3. Request handling via API blueprints
4. `after_serving`: Graceful shutdown of all services
"""

import asyncio
import datetime
from pathlib import Path
from typing import Coroutine

from dotenv import load_dotenv
from platformdirs import PlatformDirs
from quart import Quart, request
from quart_schema import QuartSchema

from cosma_backend import db
from cosma_backend.api import api_blueprint
from cosma_backend.db.database import Database
from cosma_backend.discoverer import Discoverer
from cosma_backend.embedder import AutoEmbedder
from cosma_backend.filter import FilterConfigManager
from cosma_backend.logging import get_logger, configure_logging
from cosma_backend.model_lifecycle import ModelLifecycleManager
from cosma_backend.models.update import Update
from cosma_backend.parser import FileParser
from cosma_backend.pipeline import Pipeline
from cosma_backend.queue import IndexingQueue
from cosma_backend.queue.scheduler import Scheduler
from cosma_backend.searcher import HybridSearcher
from cosma_backend.settings import SettingsManager
from cosma_backend.summarizer import AutoSummarizer
from cosma_backend.utils.pubsub import Hub
from cosma_backend.watcher import Watcher

load_dotenv()

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

configure_logging()
logger = get_logger(__name__)

class App(Quart):
    """
    Extended Quart application with typed service attributes.

    All services are initialized in `before_serving` hook after config is loaded.
    Services are available as instance attributes for easy access in route handlers.
    """

    # Database connection
    db: Database

    # Pub/sub hub for real-time updates (SSE)
    updates_hub: Hub[Update]

    # Background tasks tracked for graceful shutdown
    jobs: set[asyncio.Task]

    # Core processing services
    pipeline: Pipeline          # File processing pipeline
    searcher: HybridSearcher    # Hybrid semantic + FTS search
    watcher: Watcher            # File system change monitoring

    # Configuration
    filter_manager: FilterConfigManager  # Include/exclude patterns
    settings_manager: SettingsManager    # Persistent TOML settings
    dirs: PlatformDirs                   # Platform-specific directories

    # Background processing
    indexing_queue: IndexingQueue        # Async file processing queue
    scheduler: Scheduler                 # Conditional queue processing
    model_lifecycle: ModelLifecycleManager  # Auto-unload idle models

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.updates_hub = Hub()
        self.jobs = set()
        self.filter_manager = FilterConfigManager()

    def initialize_config(self) -> None:
        """
        Load configuration from environment variables and TOML file.

        Config priority: env vars (COSMA_*) > TOML file > defaults
        Must be called before `before_serving` hook.
        """
        logger.info("Loading config")
        self.config.from_prefixed_env("COSMA")

        # Bootstrap settings (env vars only, needed before app starts)
        self.config.setdefault("APP_NAME", "cosma")
        self.dirs = PlatformDirs(self.config["APP_NAME"], ensure_exists=True)
        log_path = Path(self.dirs.user_log_dir) / "cosma-backend.log"
        configure_logging(log_path=log_path)
        global logger
        logger = get_logger(__name__)
        logger.info("Loading config")
        self.config.setdefault("HOST", '127.0.0.1')
        self.config.setdefault("PORT", 60534)
        self.config.setdefault("DATABASE_PATH", Path(self.dirs.user_data_dir) / "app.db")

        # Load persistent settings from TOML (model configs, queue settings, etc.)
        self.settings_manager = SettingsManager(self.dirs)
        self.settings_manager.load()

        logger.debug("Config loaded")

    def submit_job(self, coro: Coroutine) -> asyncio.Task:
        """
        Submit a background coroutine as a tracked task.

        Tasks are automatically removed from the job set when complete.
        Used for fire-and-forget operations that should be tracked.
        """
        def remove_task_callback(task: asyncio.Task) -> None:
            self.jobs.discard(task)  # discard to avoid KeyError if already removed
            # Log any unhandled exceptions
            if not task.cancelled():
                exc = task.exception()
                if exc:
                    logger.exception("Background job failed", exc_info=exc)

        task = asyncio.create_task(coro)
        self.jobs.add(task)
        task.add_done_callback(remove_task_callback)

        return task
        

app = App(__name__)
app.initialize_config()
QuartSchema(app)

# Register API blueprints
app.register_blueprint(api_blueprint, url_prefix='/api')

@app.before_serving
async def initialize_services():
    logger.info("Initializing database")
    app.db = await db.connect(app.config['DATABASE_PATH'])

    logger.info("Initializing filter configuration")
    # Load global filter config (creates default if not exists)
    global_config = app.filter_manager.global_config
    logger.info("Filter config loaded",
                  mode=global_config.mode.value,
                  exclude_count=len(global_config.exclude),
                  include_count=len(global_config.include),
                  config_path=str(global_config.config_path))

    logger.info("Initializing services")
    settings = app.settings_manager.settings
    discoverer = Discoverer()
    parser = FileParser(config=settings.parser)
    summarizer = AutoSummarizer(config=settings.summarizer)
    embedder = AutoEmbedder(config=settings.embedder)

    app.pipeline = Pipeline(
        db=app.db,
        updates_hub=app.updates_hub,
        parser=parser,
        discoverer=discoverer,
        summarizer=summarizer,
        embedder=embedder,
    )

    app.searcher = HybridSearcher(
        db=app.db,
        embedder=embedder,
    )

    app.indexing_queue = IndexingQueue(
        pipeline=app.pipeline,
        updates_hub=app.updates_hub,
        config=settings.queue,
        db=app.db,
    )
    await app.indexing_queue.start()

    app.scheduler = Scheduler(
        queue=app.indexing_queue,
        updates_hub=app.updates_hub,
        config=settings.scheduler,
    )
    app.scheduler.start()

    app.model_lifecycle = ModelLifecycleManager(
        summarizer=summarizer,
        idle_unload_seconds=settings.summarizer.idle_unload_seconds,
        embedder=embedder,
    )
    app.model_lifecycle.start()

    app.watcher = Watcher(
        db=app.db,
        pipeline=app.pipeline,
        filter_manager=app.filter_manager,
        indexing_queue=app.indexing_queue,
    )

    # Run startup cleanup to remove excluded files from database
    logger.info("Running startup cleanup for excluded files")
    removed_count = await cleanup_excluded_files_on_startup(app.db, app.filter_manager)
    if removed_count > 0:
        logger.info("Startup cleanup complete", removed_files=removed_count)

    await app.watcher.initialize_from_database()

    logger.info("Initialized services")


async def cleanup_excluded_files_on_startup(db: Database, filter_manager: FilterConfigManager) -> int:
    """
    Remove files from the database that:
    1. Match current exclusion patterns (or don't match whitelist in whitelist mode)
    2. Are not under any active watched directory (orphaned files)
    Called on server startup.

    Returns:
        Number of files removed
    """
    watched_dirs = await db.get_watched_directories(active_only=True)
    removed_count = 0

    # Step 1: Clean up orphaned files (not under any active watched directory)
    async with db.acquire() as conn:
        all_files = await conn.fetchall("SELECT id, file_path FROM files")

        for row in all_files:
            file_path = row["file_path"]
            is_under_watched = False

            for watched_dir in watched_dirs:
                watched_path = str(watched_dir.path)
                if file_path.startswith(watched_path + "/") or file_path == watched_path:
                    is_under_watched = True
                    break

            if not is_under_watched:
                await conn.execute("DELETE FROM files WHERE id = ?", (row["id"],))
                removed_count += 1
                logger.info("Removed orphaned file from index (not under any watched directory)",
                              file_path=file_path)

    if not watched_dirs:
        return removed_count

    # Step 2: Clean up excluded files under watched directories
    for watched_dir in watched_dirs:
        base_path = watched_dir.path
        config = filter_manager.get_config_for_directory(base_path)

        # Get all files under this directory from database
        async with db.acquire() as conn:
            rows = await conn.fetchall(
                "SELECT id, file_path FROM files WHERE file_path LIKE ? || '/%' OR file_path = ?",
                (str(base_path), str(base_path))
            )

            for row in rows:
                file_path = Path(row["file_path"])

                if not config.should_include(file_path, base_path):
                    # File should be excluded, delete it
                    await conn.execute("DELETE FROM files WHERE id = ?", (row["id"],))
                    removed_count += 1
                    logger.info("Removed excluded file from index",
                                  file_path=str(file_path))

    return removed_count


@app.after_serving
async def handle_shutdown():
    logger.info("Stopping model lifecycle manager, scheduler, and indexing queue")
    await app.model_lifecycle.stop()
    await app.scheduler.stop()
    await app.indexing_queue.stop()
    logger.info("Closing DB")
    await app.db.close()
    

@app.before_request
async def log_request():
    request.start_time = datetime.datetime.now()
    logger.info(
        "Incoming request",
        method=request.method,
        path=request.path,
        remote_addr=request.remote_addr,
        user_agent=request.headers.get('User-Agent')
    )

@app.after_request
async def log_response(response):
    if hasattr(request, 'start_time'):
        duration = (datetime.datetime.now() - request.start_time).total_seconds()
        logger.info(
            "Request completed",
            method=request.method,
            path=request.path,
            status_code=response.status_code,
            duration_seconds=duration
        )
    return response


# ====== Application Entry Point ======

def run() -> None:
    app.run(
        host=app.config['HOST'],
        port=app.config['PORT'],
        use_reloader=False,
    )
