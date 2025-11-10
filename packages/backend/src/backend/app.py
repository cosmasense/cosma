import asyncio
import datetime
from dataclasses import dataclass
import logging
from typing import Coroutine

from dotenv import load_dotenv
from rich.logging import RichHandler
from quart import Quart, request
from quart_schema import QuartSchema, validate_request, validate_response

from backend import db
from backend.api import api_blueprint
from backend.db.database import Database
from backend.logging import sm
from backend.models.update import Update
from backend.utils.pubsub import Hub
from backend.pipeline import Pipeline
from backend.searcher import HybridSearcher
from backend.discoverer import Discoverer
from backend.parser import FileParser
from backend.summarizer import AutoSummarizer
from backend.embedder import AutoEmbedder
from backend.watcher import Watcher

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
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.updates_hub = Hub()
        self.jobs = set()        

    def initialize_config(self):
        logger.info("Loading config")
        self.config.from_prefixed_env("BACKEND")

        # add new config variable defaults here (if there should be a default)
        self.config.setdefault("DATABASE_PATH", './app.db')
        self.config.setdefault("HOST", '127.0.0.1')
        self.config.setdefault("PORT", 8080)
        
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
    parser = FileParser()
    summarizer = AutoSummarizer()
    embedder = AutoEmbedder()
    
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
