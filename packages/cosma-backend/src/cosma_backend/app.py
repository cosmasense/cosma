import asyncio
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Coroutine

from dotenv import load_dotenv
from platformdirs import PlatformDirs, user_config_dir
from quart import Quart, request
from quart_schema import QuartSchema, validate_request, validate_response

from cosma_backend import db
from cosma_backend.api import api_blueprint
from cosma_backend.db.database import Database
from cosma_backend.logging import get_logger, configure_logging
from cosma_backend.models.update import Update
from cosma_backend.settings import SettingsManager
from cosma_backend.utils.pubsub import Hub
from cosma_backend.pipeline import Pipeline
from cosma_backend.searcher import HybridSearcher
from cosma_backend.discoverer import Discoverer
from cosma_backend.parser import FileParser
from cosma_backend.summarizer import AutoSummarizer
from cosma_backend.embedder import AutoEmbedder
from cosma_backend.watcher import Watcher
from cosma_backend.filter import FilterConfigManager

load_dotenv()

logger = get_logger(__name__)

class App(Quart):
    db: Database
    updates_hub: Hub[Update]
    jobs: set[asyncio.Task]
    pipeline: Pipeline
    searcher: HybridSearcher
    watcher: Watcher
    filter_manager: FilterConfigManager
    settings_manager: SettingsManager
    dirs: PlatformDirs

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.updates_hub = Hub()
        self.jobs = set()
        self.filter_manager = FilterConfigManager()      

    def initialize_config(self):
        self.config.from_prefixed_env("COSMA")

        # Bootstrap-only settings (env vars only, needed before app starts)
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

        # Load persistent settings from TOML
        self.settings_manager = SettingsManager(self.dirs)
        self.settings_manager.load()

        logger.debug("Config loaded")
        
    def submit_job(self, coro: Coroutine) -> asyncio.Task:
        def remove_task_callback(task: asyncio.Task):
            self.jobs.remove(task)
        
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

    app.watcher = Watcher(
        db=app.db,
        pipeline=app.pipeline,
        filter_manager=app.filter_manager,
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

@app.post("/echo")
async def echo():
    data = await request.get_json()
    return {"input": data, "extra": True}

# ====== Sample Database Usage ======

@app.get("/get")
async def get():
    # I haven't implemented a get_files function yet for the db,
    # but I can if/when we need it.
    # For now I'm just running a SQL query directly
    async with app.db.acquire() as conn:
        files = await conn.fetchall("SELECT * FROM files;")

    return [dict(file) for file in files]

# ====== Main Indexing Route ======
# Note: Indexing routes have been moved to backend/api/index.py
# This endpoint remains for backward compatibility but will be deprecated

@dataclass
class IndexIn:
    directory_path: str

@dataclass
class IndexOut:
    success: bool

@app.post("/index")  # type: ignore[return-value]
@validate_request(IndexIn)
@validate_response(IndexOut, 201)
async def index(data: IndexIn) -> tuple[IndexOut, int]:
    # TODO: extract, summarize, and db
    # something like:
    # for file in extract_files():
    #    parsed_file = parse_file(file)
    #    summarized_file = app.summarizer.summarize_file(parsed_file)
    #    await app.db.insert_file(summarized_file)
    
    # Note: Use /api/index/directory instead (this route kept for compatibility)

    return IndexOut(success=True), 201


def run() -> None:
    app.run(
        host=app.config['HOST'],
        port=app.config['PORT'],
        use_reloader=False,
    )
