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
1. `create_app()` builds an App, loads config, registers blueprints + hooks
2. `before_serving`: Initialize DB, services, start background tasks
3. Request handling via API blueprints
4. `after_serving`: Graceful shutdown of all services
"""

import asyncio
import datetime
import os
from pathlib import Path
from typing import Coroutine, Optional, Protocol

from dotenv import load_dotenv
from platformdirs import PlatformDirs
from quart import Quart, request
from quart_schema import QuartSchema

from cosma_backend import bootstrap as _bootstrap_core
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

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Limit PyTorch MPS memory usage (set before torch is imported).
# BOTH watermarks must be set explicitly. When only HIGH is set,
# PyTorch's auto-derivation for LOW lands at 1.4 (HIGH × 1.86) which
# is invalid (>1.0), and SentenceTransformer's MPS load aborts with
# "invalid low watermark ratio 1.4". Forcing LOW < HIGH < 1.0 fixes
# it. See `embedder/providers.py` MPS-load fallback.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.75")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")

configure_logging()
logger = get_logger(__name__)


class AppDirs(Protocol):
    """Minimal subset of `platformdirs.PlatformDirs` used by the backend.

    Any object exposing these three properties is accepted — this lets tests
    pass a temp-directory shim without touching real user dirs.
    """

    @property
    def user_config_dir(self) -> str: ...
    @property
    def user_data_dir(self) -> str: ...
    @property
    def user_log_dir(self) -> str: ...


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
    dirs: AppDirs                        # Platform-specific directories

    # Background processing
    indexing_queue: IndexingQueue        # Async file processing queue
    scheduler: Scheduler                 # Conditional queue processing
    model_lifecycle: ModelLifecycleManager  # Auto-unload idle models

    def __init__(self, *args, dirs: Optional[AppDirs] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.updates_hub = Hub()
        self.jobs = set()
        # Caller-supplied dirs take priority over the default PlatformDirs.
        # Tests set this to a temp-directory shim so nothing touches
        # ~/Library/Application Support/cosma.
        self._dirs_override: Optional[AppDirs] = dirs

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
        if self._dirs_override is not None:
            self.dirs = self._dirs_override
        else:
            self.dirs = PlatformDirs(self.config["APP_NAME"], ensure_exists=True)
        log_path = Path(self.dirs.user_log_dir) / "cosma-backend.log"
        configure_logging(log_path=log_path)
        logger = get_logger(__name__)
        self.config.setdefault("HOST", '127.0.0.1')
        self.config.setdefault("PORT", 60534)
        self.config.setdefault("DATABASE_PATH", Path(self.dirs.user_data_dir) / "app.db")

        # Filter and settings managers are bound to whichever dirs the app
        # ended up with — real PlatformDirs in prod, temp shim in tests.
        self.filter_manager = FilterConfigManager(dirs=self.dirs)
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


async def _wait_for_bootstrap_and_resume(app: Quart) -> None:
    """Subscribe to bootstrap progress and clear the queue's bootstrap pause
    when a terminal "complete" event arrives (success or soft-failure).

    Uses the same pub/sub that the /api/bootstrap/events SSE endpoint uses,
    so a completion that happens before the user even clicks anything
    (when models are already on disk but the initial status check missed)
    still unblocks the queue.
    """
    q = _bootstrap_core.runner.subscribe()
    try:
        while True:
            evt = await q.get()
            if evt.done and evt.stage == "complete":
                break
            if evt.done and evt.stage == "error":
                # Soft-fail: unblock the queue anyway so any components that
                # did land are usable. The status endpoint still reports
                # ready=False so the frontend can prompt the user.
                break
    finally:
        _bootstrap_core.runner.unsubscribe(q)
        app.indexing_queue.bootstrap_resume()


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


def create_app(dirs: Optional[AppDirs] = None) -> App:
    """Build a fully-wired backend App.

    Args:
        dirs: Optional override for platform directories. Production leaves this
            unset and gets the default `PlatformDirs("cosma")`. Tests pass a
            temp-directory shim so settings/logs/db stay isolated.
    """
    app = App(__name__, dirs=dirs)
    app.initialize_config()
    QuartSchema(app)
    app.register_blueprint(api_blueprint, url_prefix='/api')

    # Shutdown signal observed by long-lived SSE streams (see api/updates.py,
    # api/bootstrap.py). Setting this in after_serving lets them bail out
    # cleanly instead of sitting in queue.get() until uvicorn force-cancels.
    app.shutdown_event = asyncio.Event()

    @app.before_serving
    async def initialize_services():
        """Two-phase startup: API-ready first, heavy models load in background.

        Phase 1 (blocking): DB + filter + services created (no model loading)
               → API is reachable, frontend can connect
        Phase 2 (background): Load embedding model + start indexing + watcher
               → Search becomes available, then indexing starts
        """
        # ── Startup banner ──
        # Single, easy-to-grep line that lands at the top of every
        # backend launch's log so the user can confirm which build is
        # running without reading the whole startup spam.
        import platform as _platform
        from cosma_backend import __version__, __api_version__
        try:
            torch_mod = __import__("torch")
            mps_ok = bool(torch_mod.backends.mps.is_available())
        except Exception:
            mps_ok = False
        logger.info(
            "cosma-backend starting",
            version=__version__,
            api_version=__api_version__,
            python=_platform.python_version(),
            platform=f"{_platform.system()} {_platform.machine()}",
            mps=mps_ok,
            host=app.config.get("HOST"),
            port=app.config.get("PORT"),
        )

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

        # Size the shared pipeline thread pool to cover parse + embed
        # in flight at the same time. The pool is what MarkItDown,
        # Spotlight extraction, and the embedder all use to offload
        # blocking work; if it's too small, parse_concurrency is
        # silently capped by it. See pipeline_executor.py.
        from cosma_backend.pipeline_executor import configure_pipeline_executor
        configure_pipeline_executor(
            max_workers=(
                settings.queue.parse_concurrency
                + settings.queue.embed_concurrency
                + 1   # whisper headroom
            ),
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
            parse_concurrency=settings.queue.parse_concurrency,
            summarize_concurrency=settings.queue.summarize_concurrency,
            embed_concurrency=settings.queue.embed_concurrency,
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

        # ── Phase 2: Background — load models, then start indexing ──
        # IMPORTANT: do NOT start the indexing queue yet. Indexing competes
        # with both the embedder load and (via summarize) the llama.cpp
        # load for GPU/CPU. Starting it eagerly turned the demo's first
        # search into a 5–10 s freeze. We start the queue only after
        # _deferred_heavy_init has loaded the embedder, pre-warmed the
        # LLM, and observed an `indexing_start_grace_seconds` quiet
        # window. See cosma/docs/STARTUP_PROFILE_FINDINGS.md.
        app.scheduler = Scheduler(
            queue=app.indexing_queue, updates_hub=app.updates_hub,
            config=settings.scheduler,
        )
        app.indexing_queue.set_pre_task_hook(app.scheduler.evaluate_and_apply)
        app.scheduler.start()
        app.model_lifecycle.start()

        # If required models aren't on disk yet, gate the queue until the
        # bootstrap runner finishes. Watcher discovery runs normally so items
        # queue up and begin processing the instant models arrive.
        bootstrap_status = await _bootstrap_core.get_status(
            llama_cfg=settings.summarizer.llamacpp,
            whisper_cfg=settings.parser.whisper,
            summ_provider=settings.summarizer.provider,
        )
        if not bootstrap_status.ready:
            app.indexing_queue.bootstrap_pause()
            app._bootstrap_gate_task = asyncio.ensure_future(
                _wait_for_bootstrap_and_resume(app)
            )
        else:
            app._bootstrap_gate_task = None

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
            """Load models, pre-warm the LLM, then start indexing.

            Order is intentional: embedder first so semantic search becomes
            usable as soon as possible, LLM second to pay its load cost
            behind the loading indicator (otherwise it lands on the first
            file's summarize call and blocks the event loop), then a short
            grace window before starting the indexing queue so any user
            search at app launch has the GPU to itself.

            Cancellation-safe: SIGTERM mid-load cancels this task.
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
                app._deferred_init_progress = 0.60
                logger.info("Embedding model loaded — search is ready")

                # Pre-warm the LLM so the first user-triggered summarize
                # doesn't pay the multi-second Llama.from_pretrained cost
                # on the event loop.  AutoSummarizer._get_llamacpp_summarizer
                # already runs the constructor in a thread (see auto.py),
                # so awaiting it here is non-blocking for the loop.
                logger.info("Phase 2: Pre-warming LLM (summarizer)...")
                try:
                    if hasattr(app.pipeline.summarizer, "_get_llamacpp_summarizer"):
                        await app.pipeline.summarizer._get_llamacpp_summarizer()
                    logger.info("LLM pre-warmed")
                except Exception:
                    # LLM pre-warm is best-effort — if it fails the queue
                    # still starts and the user gets a real error on the
                    # first summarize. Don't block search on this.
                    logger.exception("LLM pre-warm failed (non-fatal)")
                app._deferred_init_progress = 0.80

                logger.info("Running startup cleanup")
                removed = await cleanup_excluded_files_on_startup(app.db, app.filter_manager)
                if removed > 0:
                    logger.info("Cleanup complete", removed_files=removed)
                app._deferred_init_progress = 0.90

                await app.watcher.initialize_from_database()

                # Grace window: let any in-flight user search finish on a
                # quiet GPU before indexing kicks in. Configurable via
                # settings.queue.indexing_start_grace_seconds (default 5s).
                grace = float(getattr(
                    settings.queue, "indexing_start_grace_seconds", 5.0,
                ))
                if grace > 0:
                    logger.info(
                        "Phase 2: Holding indexing for grace window",
                        grace_seconds=grace,
                    )
                    await asyncio.sleep(grace)

                logger.info("Phase 2: Starting indexing queue")
                await app.indexing_queue.start()
                app._deferred_init_progress = 1.0
                logger.info("Phase 2 complete — indexing is running")
            except asyncio.CancelledError:
                logger.info("Phase 2 cancelled (shutdown in progress)")
            except Exception:
                logger.exception("Error in deferred startup")

        app._deferred_init_task = asyncio.ensure_future(_deferred_heavy_init())

    @app.after_serving
    async def handle_shutdown():
        # Signal SSE streams first so they stop blocking on queue.get().
        # Without this step uvicorn's graceful-shutdown window gets burned
        # waiting for these long-lived handlers, leading to force-cancels
        # and ugly CancelledError tracebacks.
        app.shutdown_event.set()
        try:
            from cosma_backend.models.update import Update
            app.updates_hub.publish(Update.shutting_down())
        except Exception:
            logger.exception("Error publishing SHUTTING_DOWN update")
        try:
            from cosma_backend import bootstrap as _bootstrap_core
            _bootstrap_core.runner.broadcast_shutdown()
        except Exception:
            logger.exception("Error broadcasting bootstrap shutdown")

        # Cancel Phase 2 if it's still running (e.g. model loading in thread).
        deferred = getattr(app, "_deferred_init_task", None)
        if deferred and not deferred.done():
            logger.info("Shutdown: cancelling deferred startup")
            deferred.cancel()
            try:
                await deferred
            except (asyncio.CancelledError, Exception):
                pass

        gate = getattr(app, "_bootstrap_gate_task", None)
        if gate and not gate.done():
            gate.cancel()
            try:
                await gate
            except (asyncio.CancelledError, Exception):
                pass

        logger.info("Shutdown: stopping lifecycle manager, scheduler, and indexing queue")

        async def _safe(label: str, coro):
            try:
                await coro
            except Exception:
                logger.exception(f"Error stopping {label}")

        # Parallelize independent teardown: these three don't share state,
        # so gather cuts wall-clock shutdown time roughly in thirds.
        await asyncio.gather(
            _safe("model lifecycle", app.model_lifecycle.stop()),
            _safe("scheduler",       app.scheduler.stop()),
            _safe("indexing queue",  app.indexing_queue.stop()),
        )

        # Unload summarizer — important for Ollama (holds GPU until told to
        # release via keep_alive=0). Short bounded timeout so we don't run
        # out the uvicorn grace window.
        logger.info("Shutdown: unloading summarizer models")
        try:
            summarizer = getattr(app.pipeline, "summarizer", None)
            if summarizer is not None and hasattr(summarizer, "unload_models"):
                await asyncio.wait_for(summarizer.unload_models(), timeout=3.0)
        except asyncio.TimeoutError:
            logger.warning("Summarizer unload timed out after 3s")
        except Exception:
            logger.exception("Error unloading summarizer models")

        try:
            from cosma_backend.parser.media import unload_whisper_model
            unload_whisper_model()
        except Exception:
            logger.exception("Error unloading whisper model")

        # Kill joblib/loky pool workers. sentence-transformers uses them
        # indirectly; their daemon threads outlive shutdown and keep the
        # interpreter alive, which the macOS supervisor then SIGKILLs (exit
        # code 9). kill_workers=True releases the semaphores cleanly.
        try:
            from joblib.externals.loky import get_reusable_executor
            get_reusable_executor().shutdown(wait=False, kill_workers=True)
        except Exception:
            # joblib may not be importable in slim deployments; best-effort.
            pass

        logger.info("Shutdown: closing DB")
        try:
            await app.db.close()
        except Exception:
            logger.exception("Error closing DB")

        logger.info("Shutdown complete")

    @app.before_request
    async def log_request():
        request.start_time = datetime.datetime.now()
        # Per-request log lines are demoted to DEBUG. They were the
        # single biggest source of noise in production logs (~200
        # lines/min on a busy frontend) and the slow-request probe
        # below already surfaces anything interesting at WARNING.
        logger.debug(
            "Incoming request",
            method=request.method,
            path=request.path,
        )

    @app.after_request
    async def log_response(response):
        if hasattr(request, 'start_time'):
            duration = (datetime.datetime.now() - request.start_time).total_seconds()
            logger.debug(
                "Request completed",
                method=request.method,
                path=request.path,
                status_code=response.status_code,
                duration_seconds=duration,
            )
            # Slow-request probe: when a request exceeds 1 s, capture what the
            # pipeline was doing so we can correlate latency spikes with work.
            if duration > 1.0:
                try:
                    q = getattr(app, "indexing_queue", None)
                    queue_ctx = {
                        "processing": sum(
                            1 for i in q._items.values()
                            if i.status.name == "PROCESSING"
                        ) if q is not None else None,
                        "waiting": sum(
                            1 for i in q._items.values()
                            if i.status.name == "WAITING"
                        ) if q is not None else None,
                        "total_items": len(q._items) if q is not None else None,
                        "is_paused": q.is_paused if q is not None else None,
                    }
                except Exception:
                    queue_ctx = {}
                logger.warning(
                    "Slow request",
                    method=request.method,
                    path=request.path,
                    duration_seconds=duration,
                    **queue_ctx,
                )
        return response

    return app


# ====== Application Entry Point ======

def run() -> None:
    """Production entrypoint: build the default app and serve it with Quart's
    dev server. The uvicorn-based path lives in `cosma_backend.serve`."""
    app = create_app()
    app.run(
        host=app.config['HOST'],
        port=app.config['PORT'],
        use_reloader=False,
    )
