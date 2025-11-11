import asyncio
import datetime
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Coroutine

from dotenv import load_dotenv
from rich.logging import RichHandler
from platformdirs import PlatformDirs, user_config_dir
from quart import Quart, request
from quart_schema import QuartSchema, validate_request, validate_response

from cosma_backend import db
from cosma_backend.api import api_blueprint
from cosma_backend.db.database import Database
from cosma_backend.logging import sm
from cosma_backend.models.update import Update
from cosma_backend.utils.pubsub import Hub
from cosma_backend.pipeline import Pipeline
from cosma_backend.searcher import HybridSearcher
from cosma_backend.discoverer import Discoverer
from cosma_backend.parser import FileParser
from cosma_backend.summarizer import AutoSummarizer
from cosma_backend.embedder import AutoEmbedder
from cosma_backend.watcher import Watcher

load_dotenv()

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

logger = logging.getLogger(__name__)

class App(Quart):
    db: Database
    updates_hub: Hub[Update]
    jobs: set[asyncio.Task]
    pipeline: Pipeline
    searcher: HybridSearcher
    watcher: Watcher
    dirs: PlatformDirs
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.updates_hub = Hub()
        self.jobs = set()      

    def initialize_config(self):
        logger.info("Loading config")
        self.config.from_prefixed_env("COSMA")

        self.config.setdefault("APP_NAME", "cosma")
        self.dirs = PlatformDirs(self.config["APP_NAME"], ensure_exists=True)

        # add new config variable defaults here (if there should be a default)
        self.config.setdefault("HOST", '127.0.0.1')
        self.config.setdefault("PORT", 60534)
        self.config.setdefault("DATABASE_PATH", Path(self.dirs.user_data_dir) / "app.db")
        
        # ===== Embedder Configuration =====
        self.config.setdefault("EMBEDDING_MODEL", "text-embedding-3-small")
        self.config.setdefault("EMBEDDING_DIMENSIONS", 512)
        self.config.setdefault("LOCAL_EMBEDDING_MODEL", "intfloat/e5-base-v2")
        self.config.setdefault("LOCAL_EMBEDDING_DIMENSIONS", 768)
        self.config.setdefault("EMBEDDING_PROVIDER", "local")
        
        # ===== Summarizer Configuration =====
        self.config.setdefault("MAX_TOKENS_PER_REQUEST", 100000)
        self.config.setdefault("CHUNK_OVERLAP_TOKENS", 1000)
        self.config.setdefault("OLLAMA_MODEL", "qwen3-vl:2b-instruct")
        self.config.setdefault("OLLAMA_HOST", "http://localhost:11434")
        self.config.setdefault("OLLAMA_MODEL_CONTEXT_LENGTH", 10000)
        self.config.setdefault("ONLINE_MODEL", "openai/gpt-4.1-nano-2025-04-14")
        self.config.setdefault("ONLINE_MODEL_CONTEXT_LENGTH", 128000)
        self.config.setdefault("LLAMACPP_MODEL_CONTEXT_LENGTH", 8192)
        self.config.setdefault("LLAMACPP_N_CTX", 8192)
        self.config.setdefault("LLAMACPP_N_THREADS", 4)
        self.config.setdefault("LLAMACPP_N_GPU_LAYERS", 0)
        self.config.setdefault("LLAMACPP_VERBOSE", False)
        self.config.setdefault("AI_PROVIDER", "auto")
        
        # ===== Parser Configuration =====
        self.config.setdefault("EXTRACTION_STRATEGY", "spotlight_first")
        self.config.setdefault("SPOTLIGHT_ENABLED", True)
        self.config.setdefault("SPOTLIGHT_TIMEOUT_SECONDS", 5)
        self.config.setdefault("WHISPER_PROVIDER", "online")
        self.config.setdefault("ONLINE_WHISPER_MODEL", "whisper-1")
        self.config.setdefault("LOCAL_WHISPER_MODEL", "turbo")
        
        # Convert string boolean values to actual booleans
        if isinstance(self.config.get("LLAMACPP_VERBOSE"), str):
            self.config["LLAMACPP_VERBOSE"] = self.config["LLAMACPP_VERBOSE"].lower() == "true"
        if isinstance(self.config.get("SPOTLIGHT_ENABLED"), str):
            self.config["SPOTLIGHT_ENABLED"] = self.config["SPOTLIGHT_ENABLED"].lower() == "true"
        
        # Convert string numeric values to actual integers
        for key in [
            "EMBEDDING_DIMENSIONS", "LOCAL_EMBEDDING_DIMENSIONS", 
            "MAX_TOKENS_PER_REQUEST", "CHUNK_OVERLAP_TOKENS",
            "OLLAMA_MODEL_CONTEXT_LENGTH", "ONLINE_MODEL_CONTEXT_LENGTH",
            "LLAMACPP_MODEL_CONTEXT_LENGTH", "LLAMACPP_N_CTX", 
            "LLAMACPP_N_THREADS", "LLAMACPP_N_GPU_LAYERS", "SPOTLIGHT_TIMEOUT_SECONDS"
        ]:
            if isinstance(self.config.get(key), str):
                try:
                    self.config[key] = int(self.config[key])
                except ValueError:
                    pass  # Keep default if conversion fails
        
        logger.debug(sm("Config loaded", config=self.config))
        
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
    logger.info(sm("Initializing database"))
    app.db = await db.connect(app.config['DATABASE_PATH'])
    
    logger.info(sm("Initializing services"))
    discoverer = Discoverer()
    parser = FileParser(config=app.config)
    summarizer = AutoSummarizer(config=app.config)
    embedder = AutoEmbedder(config=app.config)
    
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
    )
    await app.watcher.initialize_from_database()
    
    logger.info(sm("Initialized services"))


@app.after_serving
async def handle_shutdown():
    logger.info(sm("Closing DB"))
    await app.db.close()
    

@app.before_request
async def log_request():
    request.start_time = datetime.datetime.now()
    logger.info(sm(
        "Incoming request",
        method=request.method,
        path=request.path,
        remote_addr=request.remote_addr,
        user_agent=request.headers.get('User-Agent')
    ))

@app.after_request
async def log_response(response):
    if hasattr(request, 'start_time'):
        duration = (datetime.datetime.now() - request.start_time).total_seconds()
        logger.info(sm(
            "Request completed",
            method=request.method,
            path=request.path,
            status_code=response.status_code,
            duration_seconds=duration
        ))
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
