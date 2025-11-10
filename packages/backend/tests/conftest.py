"""Pytest fixtures and configuration for backend testing."""

import asyncio
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, Generator
import pytest
import pytest_asyncio
import asqlite
from unittest.mock import AsyncMock, MagicMock

from backend.db.database import Database
from backend.models.file import File
from backend.models.status import ProcessingStatus
from backend.app import App
from backend.discoverer import Discoverer
from backend.parser import FileParser
from backend.summarizer import AutoSummarizer
from backend.embedder import AutoEmbedder
from backend.pipeline import Pipeline
from backend.searcher import HybridSearcher
from backend.watcher import Watcher
from backend.utils.pubsub import Hub
from backend.models.update import Update


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def temp_db() -> AsyncGenerator[Database, None]:
    """Create a temporary in-memory database for testing."""
    # Use file-based database for better SQLite FTS support
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
        db_path = temp_file.name
    
    try:
        db = await Database.from_path(db_path)
        yield db
    finally:
        await db.close()
        # Clean up temp file
        Path(db_path).unlink(missing_ok=True)


@pytest_asyncio.fixture
async def temp_file_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_text_file(temp_file_dir: Path) -> Path:
    """Create a sample text file for testing."""
    file_path = temp_file_dir / "sample.txt"
    file_path.write_text("This is a sample text file for testing.")
    return file_path


@pytest.fixture
def sample_markdown_file(temp_file_dir: Path) -> Path:
    """Create a sample markdown file for testing."""
    content = """# Sample Document

This is a sample markdown document with some content.

## Features

- Feature 1
- Feature 2
- Feature 3

The document contains various elements for testing the parsing and summarization pipeline.
"""
    file_path = temp_file_dir / "sample.md"
    file_path.write_text(content)
    return file_path


@pytest.fixture
def sample_file_data() -> dict:
    """Sample file data for creating test File objects."""
    now = datetime.now(timezone.utc)
    return {
        "path": Path("/test/sample.txt"),
        "file_path": "/test/sample.txt",
        "filename": "sample.txt",
        "extension": ".txt",
        "file_size": 1024,
        "created": now,
        "modified": now,
        "accessed": now,
        "content_type": "text/plain",
        "content": "Sample content for testing",
        "content_hash": "abc123",
        "summary": "A test file with sample content",
        "title": "Test File",
        "keywords": ["test", "sample", "content"],
        "status": ProcessingStatus.COMPLETE,
    }


@pytest_asyncio.fixture
async def app_instance() -> AsyncGenerator[App, None]:
    """Create a test instance of the Quart app."""
    from quart_schema import QuartSchema
    
    app = App(__name__)
    app.initialize_config()
    
    # Override config for testing
    app.config["TESTING"] = True
    app.config["DATABASE_PATH"] = ":memory:"  # Will be overridden in test
    
    QuartSchema(app)
    
    yield app


@pytest_asyncio.fixture
async def test_client(app_instance: App, temp_db: Database) -> AsyncGenerator[App, None]:
    """Create a test client with a temporary database."""
    app_instance.db = temp_db
    
    # Register API blueprints
    from backend.api import api_blueprint
    app_instance.register_blueprint(api_blueprint, url_prefix='/api')
    
    yield app_instance


@pytest.fixture
def mock_updates_hub() -> Hub[Update]:
    """Create a mock updates hub for testing."""
    return Hub()


@pytest.fixture
def mock_parser() -> FileParser:
    """Create a mock parser for testing to avoid real model loading."""
    from unittest.mock import AsyncMock
    
    parser = FileParser()
    
    # Mock the parse_file method to avoid real parsing
    async def mock_parse_file(file: File):
        # Set basic mock parsing results
        file.content = f"Mock content for {file.filename}"
        file.content_type = "text/plain"
        file.content_hash = "mock_hash_123"
        file.parsed_at = datetime.now(timezone.utc)
        file.status = ProcessingStatus.PARSED
    
    parser.parse_file = AsyncMock(side_effect=mock_parse_file)
    return parser


@pytest.fixture
def mock_discoverer() -> Discoverer:
    """Create a mock discoverer for testing."""
    from unittest.mock import MagicMock
    discoverer = MagicMock()
    # No need to mock methods yet since it just uses filesystem
    return discoverer


@pytest_asyncio.fixture
async def test_pipeline(temp_db: Database, mock_updates_hub: Hub[Update], mock_summarizer: AutoSummarizer, mock_embedder: AutoEmbedder, mock_parser: FileParser, mock_discoverer: Discoverer) -> Pipeline:
    """Create a test pipeline with fully mocked components."""
    return Pipeline(
        db=temp_db,
        updates_hub=mock_updates_hub,
        discoverer=mock_discoverer,
        parser=mock_parser,
        summarizer=mock_summarizer,
        embedder=mock_embedder,
    )


@pytest.fixture
def mock_searcher(temp_db: Database) -> HybridSearcher:
    """Create a mock searcher for testing."""
    from backend.searcher import HybridSearcher
    import numpy as np
    
    # Mock the embedder
    mock_embedder_instance = mock_embedder()
    
    searcher = HybridSearcher(db=temp_db, embedder=mock_embedder_instance)
    
    # Mock search methods to avoid real embeddings during testing
    async def mock_search(query: str, directory: str = None, limit: int = 50):
        # Return mock search results
        from backend.searcher.searcher import SearchResult
        
        mock_results = []
        # Add a few mock results
        for i in range(3):
            mock_file = FileFactory()
            mock_file.id = i + 1
            mock_file.file_path = f"/mock/path/{i}.txt"
            mock_file.filename = f"mock_file_{i}.txt"
            mock_file.summary = f"Mock summary for file {i}"
            
            result = MagicMock()
            result.file_metadata = mock_file
            result.combined_score = 0.9 - (i * 0.1)  # Decreasing scores
            mock_results.append(result)
        
        return mock_results[:limit]
    
    searcher.search = AsyncMock(side_effect=mock_search)
    return searcher


@pytest.fixture
def mock_watcher(temp_db: Database, test_pipeline: Pipeline) -> Watcher:
    """Create a mock watcher for testing."""
    watcher = Watcher(db=temp_db, pipeline=test_pipeline)
    
    # Mock the watcher initialization to avoid real file system watching
    async def mock_initialize():
        return True
    
    watcher.initialize_from_database = AsyncMock(side_effect=mock_initialize)
    return watcher


@pytest.fixture 
def mocked_env_vars(monkeypatch):
    """Set up mocked environment variables for testing."""
    monkeypatch.setenv("BACKEND_DATABASE_PATH", ":memory:")
    monkeypatch.setenv("BACKEND_HOST", "127.0.0.1")
    monkeypatch.setenv("BACKEND_PORT", "8080")
    # Mock any LLM API keys to avoid real calls
    monkeypatch.setenv("OPENAI_API_KEY", "mock-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "mock-key")


@pytest.fixture
def mock_litellm():
    """Mock litellm to avoid real LLM calls."""
    from unittest.mock import patch, MagicMock
    
    mock_response = {
        "choices": [{
            "message": {
                "content": "Mock LLM response for testing"
            }
        }]
    }
    
    with patch('litellm.acompletion') as mock_completion:
        mock_completion.return_value = mock_response
        yield mock_completion


@pytest.fixture(scope="session")
def mock_real_services():
    """Mock all real services at import level to prevent model loading."""
    from unittest.mock import patch, AsyncMock, MagicMock
    import numpy as np
    
    # Mock AutoSummarizer to prevent real initialization
    with patch('backend.summarizer.AutoSummarizer') as mock_summarizer_class:
        mock_summarizer = MagicMock()
        mock_summarizer.summarize = AsyncMock()
        mock_summarizer_class.return_value = mock_summarizer
        
        # Mock AutoEmbedder to prevent real model loading
        with patch('backend.embedder.AutoEmbedder') as mock_embedder_class:
            mock_embedder = MagicMock()
            
            async def mock_embed_func(file):
                file.embedding = np.random.rand(384).astype(np.float32)
                file.embedding_model = "mock-model"
                file.embedding_dimensions = 384
                file.embedded_at = datetime.now(timezone.utc)
            
            mock_embedder.embed = AsyncMock(side_effect=mock_embed_func)
            mock_embedder_class.return_value = mock_embedder
            
            # Mock SentenceTransformer to prevent model loading
            with patch('sentence_transformers.SentenceTransformer') as mock_st:
                mock_model = MagicMock()
                mock_model.encode = MagicMock(return_value=np.random.rand(384))
                mock_st.return_value = mock_model
                
                yield {
                    'summarizer': mock_summarizer,
                    'embedder': mock_embedder,
                    'sentence_transformer': mock_st
                }


@pytest.fixture
def discoverer() -> Discoverer:
    """Create a discoverer instance for testing."""
    return Discoverer()


@pytest.fixture
def parser() -> FileParser:
    """Create a parser instance for testing."""
    return FileParser()


@pytest.fixture
def mock_summarizer() -> AutoSummarizer:
    """Create a mock summarizer for testing."""
    import pytest
    from unittest.mock import AsyncMock
    
    summarizer = AutoSummarizer()
    
    # Mock the summarize method to avoid real LLM calls
    async def mock_summarize(file: File):
        # Set realistic mock values
        file.summary = f"Mock summary for {file.filename}"
        file.title = f"Mock Title: {file.filename}"
        file.keywords = ["keyword1", "keyword2", "keyword3"]
        file.summarized_at = datetime.now(timezone.utc)
    
    summarizer.summarize = AsyncMock(side_effect=mock_summarize)
    return summarizer


@pytest.fixture
def mock_embedder() -> AutoEmbedder:
    """Create a mock embedder for testing."""
    import pytest
    from unittest.mock import AsyncMock
    import numpy as np
    
    embedder = AutoEmbedder()
    
    # Mock the embed method
    async def mock_embed(file: File):
        # Create a realistic mock embedding
        mock_embedding = np.random.rand(384).astype(np.float32)
        file.embedding = mock_embedding
        file.embedding_model = "mock-model"
        file.embedding_dimensions = 384
        file.embedded_at = datetime.now(timezone.utc)
    
    embedder.embed = AsyncMock(side_effect=mock_embed)
    
    # Mock get_embedding method if it exists
    if hasattr(embedder, 'get_embedding'):
        embedder.get_embedding = AsyncMock(
            return_value=np.random.rand(384).astype(np.float32)
        )
    
    return embedder


@pytest_asyncio.fixture
async def sample_file_in_db(temp_db: Database, sample_file_data: dict) -> File:
    """Create a sample file in the database for testing."""
    file_obj = File(**sample_file_data)
    file_id = await temp_db.upsert_file(file_obj)
    file_obj.id = file_id
    return file_obj


@pytest.fixture
def temp_env_vars(monkeypatch):
    """Set temporary environment variables for testing."""
    monkeypatch.setenv("BACKEND_DATABASE_PATH", ":memory:")
    monkeypatch.setenv("BACKEND_HOST", "127.0.0.1")
    monkeypatch.setenv("BACKEND_PORT", "8080")


# Factory for creating test File objects
@pytest.fixture
def file_factory():
    """Factory fixture for creating test File objects."""
    from tests.factories import FileFactory
    return FileFactory