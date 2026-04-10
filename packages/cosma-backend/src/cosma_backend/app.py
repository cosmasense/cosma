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
# Limit PyTorch MPS memory usage (set before torch is imported)
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.75")

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
        global logger
        logger.info("Loading config")
        self.config.from_prefixed_env("COSMA")

        # Bootstrap settings (env vars only, needed before app starts)
        self.config.setdefault("APP_NAME", "cosma")
        self.dirs = PlatformDirs(self.config["APP_NAME"], ensure_exists=True)
        log_path = Path(self.dirs.user_log_dir) / "cosma-backend.log"
        configure_logging(log_path=log_path)
        logger = get_logger(__name__)
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
    """Two-phase startup: API-ready first, heavy models load in background.

    Phase 1 (blocking): DB + filter + services created (no model loading)
           → API is reachable, frontend can connect
    Phase 2 (background): Load embedding model + start indexing + watcher
           → Search becomes available, then indexing starts
    """
    # ── Phase 1: Fast startup — get the API responding ──
    logger.info("Phase 1: Initializing database and core services")
    app.db = await db.connect(app.config['DATABASE_PATH'])

    global_config = app.filter_manager.global_config
    logger.info("Filter config loaded",
                  mode=global_config.mode.value,
                  include_count=len(global_config.include),
                  config_path=str(global_config.config_path))

    settings = app.settings_manager.settings

    # Apply GPU memory cap from settings.
    #
    # IMPORTANT: the cap affects different backends differently:
    #   * PyTorch / MPS (embedder, local whisper) — honors
    #     PYTORCH_MPS_HIGH_WATERMARK_RATIO directly.
    #   * llama.cpp (summarizer via llama-cpp-python) — uses its own Metal
    #     backend. Memory usage is controlled by `summarizer.llamacpp.n_gpu_layers`
    #     (negative = all layers on GPU, positive = N layers only). Metal
    #     respects system memory pressure automatically.
    #   * Ollama — runs as a separate process. Configure Ollama itself via
    #     OLLAMA_NUM_GPU / OLLAMA_KEEP_ALIVE env vars before launching it;
    #     this setting has no direct effect on an already-running Ollama server.
    gpu_cap = settings.queue.gpu_memory_cap
    if not (0.0 < gpu_cap <= 1.0):
        logger.warning("Invalid gpu_memory_cap, falling back to 0.75",
                      configured=gpu_cap)
        gpu_cap = 0.75
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = str(gpu_cap)

    # Hint for any Ollama instance spawned in this environment
    os.environ.setdefault("OLLAMA_KEEP_ALIVE", "5m")

    logger.info(
        "GPU memory cap applied",
        gpu_cap=f"{gpu_cap:.0%}",
        affects="torch (embedder, local whisper)",
        note="llama.cpp uses n_gpu_layers; Ollama uses its own env config",
    )

    discoverer = Discoverer()
    parser = FileParser(config=settings.parser)
    summarizer = AutoSummarizer(config=settings.summarizer)

    # Create embedder WITHOUT eager model loading — stays lazy
    embedder = AutoEmbedder(config=settings.embedder, preferred_provider="lazy_local")

    app.pipeline = Pipeline(
        db=app.db, updates_hub=app.updates_hub,
        parser=parser, discoverer=discoverer,
        summarizer=summarizer, embedder=embedder,
    )
    app.searcher = HybridSearcher(db=app.db, embedder=embedder)

    app.indexing_queue = IndexingQueue(
        pipeline=app.pipeline, updates_hub=app.updates_hub,
        config=settings.queue, db=app.db,
    )
    app.model_lifecycle = ModelLifecycleManager(
        summarizer=summarizer,
        idle_unload_seconds=settings.summarizer.idle_unload_seconds,
        embedder=embedder,
    )

    logger.info("Phase 1 complete — API is ready")

    # ── Phase 2: Background — load models and start indexing ──
    # Start indexing services immediately (non-blocking async)
    await app.indexing_queue.start()
    app.scheduler = Scheduler(
        queue=app.indexing_queue, updates_hub=app.updates_hub,
        config=settings.scheduler,
    )
    app.indexing_queue.set_pre_task_hook(app.scheduler.evaluate_and_apply)
    app.scheduler.start()
    app.model_lifecycle.start()

    app.watcher = Watcher(
        db=app.db, pipeline=app.pipeline,
        filter_manager=app.filter_manager,
        indexing_queue=app.indexing_queue,
    )

    # Progress tracking for Phase 2 (0.0 → 1.0)
    app._deferred_init_progress: float = 0.0
    app._model_loading_done = asyncio.Event()

    async def _model_loading_progress_updater():
        """Smoothly increment progress from 10% → 78% over ~12 s while
        the embedding model loads on a background thread."""
        progress = 0.10
        while not app._model_loading_done.is_set():
            remaining = 0.78 - progress
            if remaining > 0.01:
                progress += remaining * 0.06
                app._deferred_init_progress = progress
            try:
                await asyncio.wait_for(
                    app._model_loading_done.wait(), timeout=0.3,
                )
                break
            except asyncio.TimeoutError:
                pass

    async def _deferred_heavy_init():
        """Load embedding model in a thread so it doesn't block the event loop.

        This lets Uvicorn start accepting connections immediately after Phase 1
        instead of waiting ~8s for the SentenceTransformer model to load.
        Cancellation-safe: if SIGTERM arrives mid-load, the task is cancelled.
        """
        try:
            app._deferred_init_progress = 0.05
            logger.info("Phase 2: Loading embedding model...")
            if embedder.preferred_provider in ("local", "lazy_local"):
                embedder.preferred_provider = "local"
                app._deferred_init_progress = 0.10
                progress_task = asyncio.ensure_future(
                    _model_loading_progress_updater(),
                )
                await asyncio.to_thread(embedder._eagerly_initialize_models)
                app._model_loading_done.set()
                await progress_task
            app._deferred_init_progress = 0.80
            logger.info("Embedding model loaded — search is ready")

            logger.info("Running startup cleanup")
            removed = await cleanup_excluded_files_on_startup(app.db, app.filter_manager)
            if removed > 0:
                logger.info("Cleanup complete", removed_files=removed)
            app._deferred_init_progress = 0.90

            await app.watcher.initialize_from_database()
            app._deferred_init_progress = 1.0
            logger.info("Phase 2 complete — indexing is ready")
        except asyncio.CancelledError:
            logger.info("Phase 2 cancelled (shutdown in progress)")
        except Exception:
            logger.exception("Error in deferred startup")

    app._deferred_init_task = asyncio.ensure_future(_deferred_heavy_init())


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

    # Step 1: Clean up orphaned files (not under any active watched directory).
    # Uses SQL-level DELETE to avoid loading all file rows into Python memory.
    async with db.acquire() as conn:
        if not watched_dirs:
            cursor = await conn.execute("SELECT COUNT(*) FROM files")
            row = await cursor.fetchone()
            removed_count = row[0] if row else 0
            if removed_count:
                await conn.execute("DELETE FROM files")
        else:
            # Build a single DELETE that removes files not under any watched dir
            conditions = " AND ".join(
                "(file_path NOT LIKE ? || '/' || '%' AND file_path != ?)"
                for _ in watched_dirs
            )
            params: list[str] = []
            for wd in watched_dirs:
                wp = str(wd.path)
                params.extend([wp, wp])
            # Count first, then delete (asqlite Cursor has no rowcount)
            count_cursor = await conn.execute(
                f"SELECT COUNT(*) FROM files WHERE {conditions}", tuple(params)
            )
            count_row = await count_cursor.fetchone()
            removed_count = count_row[0] if count_row else 0
            if removed_count:
                await conn.execute(
                    f"DELETE FROM files WHERE {conditions}", tuple(params)
                )
                logger.info("Removed orphaned files not under any watched directory",
                            count=removed_count)

    if not watched_dirs:
        return removed_count

    # Step 2: Clean up excluded files under watched directories.
    # Uses cursor iteration instead of fetchall to keep memory bounded.
    for watched_dir in watched_dirs:
        base_path = watched_dir.path
        config = filter_manager.get_config_for_directory(base_path)

        ids_to_delete: list = []
        async with db.acquire() as conn:
            cursor = await conn.execute(
                "SELECT id, file_path FROM files WHERE file_path LIKE ? || '/' || '%' OR file_path = ?",
                (str(base_path), str(base_path))
            )
            rows = await cursor.fetchall()
            for row in rows:
                file_path = Path(row["file_path"])
                if not config.should_include(file_path, base_path):
                    ids_to_delete.append(row["id"])

            # Batch delete collected IDs
            for file_id in ids_to_delete:
                await conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
            removed_count += len(ids_to_delete)
            if ids_to_delete:
                logger.info("Removed excluded files under directory",
                            directory=str(base_path), count=len(ids_to_delete))

    return removed_count


@app.after_serving
async def handle_shutdown():
    # Cancel Phase 2 if it's still running (e.g. model loading in thread).
    # This prevents the shutdown from waiting on a slow model download.
    deferred = getattr(app, "_deferred_init_task", None)
    if deferred and not deferred.done():
        logger.info("Shutdown: cancelling deferred startup")
        deferred.cancel()
        try:
            await deferred
        except (asyncio.CancelledError, Exception):
            pass

    logger.info("Shutdown: stopping lifecycle manager, scheduler, and indexing queue")
    try:
        await app.model_lifecycle.stop()
    except Exception:
        logger.exception("Error stopping model lifecycle manager")
    try:
        await app.scheduler.stop()
    except Exception:
        logger.exception("Error stopping scheduler")
    try:
        await app.indexing_queue.stop()
    except Exception:
        logger.exception("Error stopping indexing queue")

    # Unload summarizer models — this is important for Ollama, which runs in a
    # separate process and keeps the model pinned in GPU until explicitly told
    # to release (via a request with keep_alive=0). Without this call, Ollama
    # can hold 100% of GPU memory even after the backend quits.
    logger.info("Shutdown: unloading summarizer models")
    try:
        summarizer = getattr(app.pipeline, "summarizer", None)
        if summarizer is not None and hasattr(summarizer, "unload_models"):
            await asyncio.wait_for(summarizer.unload_models(), timeout=3.0)
    except asyncio.TimeoutError:
        logger.warning("Summarizer unload timed out after 3s")
    except Exception:
        logger.exception("Error unloading summarizer models")

    # Unload Whisper if it was loaded
    try:
        from cosma_backend.parser.media import unload_whisper_model
        unload_whisper_model()
    except Exception:
        logger.exception("Error unloading whisper model")

    logger.info("Shutdown: closing DB")
    try:
        await app.db.close()
    except Exception:
        logger.exception("Error closing DB")

    logger.info("Shutdown complete")
    

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
