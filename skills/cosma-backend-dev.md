# CosmaSense Backend Development Guide for AI Agents

Use this skill when modifying the CosmaSense backend codebase, adding new API endpoints, or extending the processing pipeline.

## Project Structure

```
cosma/packages/cosma-backend/src/cosma_backend/
├── app.py                  # App class (Quart), service wiring, startup/shutdown
├── settings.py             # SettingsManager, all *Config dataclasses
├── api/                    # REST API blueprints
│   ├── __init__.py         # Blueprint registration (api_blueprint)
│   ├── models.py           # Shared response dataclasses (FileResponse, etc.)
│   ├── files.py            # /api/files/*
│   ├── index.py            # /api/index/*
│   ├── search.py           # /api/search/*
│   ├── status.py           # /api/status/
│   ├── watch.py            # /api/watch/*
│   ├── queue.py            # /api/queue/*
│   ├── filters.py          # /api/filters/*
│   ├── settings.py         # /api/settings/*
│   └── updates.py          # /api/updates/ (SSE)
├── models/                 # Core data models
│   ├── file.py             # File dataclass (all pipeline stages)
│   ├── update.py           # UpdateOpcode enum, Update dataclass (SSE events)
│   ├── status.py           # ProcessingStatus enum
│   └── watch.py            # WatchedDirectory dataclass
├── pipeline/               # File processing pipeline
│   ├── pipeline.py         # Pipeline class (process_file, process_directory)
│   ├── discoverer.py       # File discovery and metadata extraction
│   ├── parser.py           # FileParser (MarkItDown, Spotlight)
│   ├── summarizer.py       # AutoSummarizer (Ollama/LiteLLM)
│   └── embedder.py         # AutoEmbedder (local/OpenAI)
├── queue/                  # Indexing queue with debounce
│   ├── __init__.py
│   ├── indexing_queue.py   # IndexingQueue class
│   ├── scheduler.py        # Scheduler + rule engine
│   └── metrics.py          # SystemMetricsCollector (psutil + macOS CLIs)
├── watcher/                # Filesystem monitoring
│   └── watcher.py          # Watcher, WatcherJob (watchdog integration)
├── db/                     # Database layer
│   ├── database.py         # Database class (async SQLite + sqlite-vec)
│   └── schema.sql          # DDL for tables, FTS5, vec0
├── search/                 # Search engine
│   └── searcher.py         # HybridSearcher (semantic + FTS5)
└── hub.py                  # Pub/sub for SSE events
```

## How to Add a New API Endpoint

### Step 1: Create (or extend) a blueprint file

File: `api/myfeature.py`

```python
from dataclasses import dataclass
from quart import Blueprint, current_app
from quart_schema import validate_request, validate_response

myfeature_bp = Blueprint("myfeature", __name__)


@dataclass
class MyRequest:
    some_field: str


@dataclass
class MyResponse:
    success: bool
    message: str


@myfeature_bp.post("/do-something")
@validate_request(MyRequest)
@validate_response(MyResponse, 200)
async def do_something(data: MyRequest) -> tuple[MyResponse, int]:
    """POST /api/myfeature/do-something"""
    # Access services via current_app:
    db = current_app.db
    pipeline = current_app.pipeline
    queue = current_app.indexing_queue
    hub = current_app.updates_hub

    return MyResponse(success=True, message="Done"), 200
```

### Step 2: Register the blueprint

In `api/__init__.py`, add:

```python
from .myfeature import myfeature_bp

# Inside the existing registration block:
api_blueprint.register_blueprint(myfeature_bp, url_prefix="/myfeature")
```

This makes it available at `/api/myfeature/do-something`.

### Step 3: Add SSE events (if needed)

In `models/update.py`, add new opcodes:

```python
class UpdateOpcode(enum.Enum):
    # ... existing opcodes ...
    MY_EVENT = "my_event"
```

Publish from your code:

```python
from cosma_backend.models.update import Update, UpdateOpcode

current_app.updates_hub.publish(
    Update.create(UpdateOpcode.MY_EVENT, key="value")
)
```

### Step 4: Write tests

```python
# tests/unit/test_myfeature.py
import pytest
from cosma_backend.app import app

@pytest.fixture
def client():
    return app.test_client()

@pytest.mark.asyncio
async def test_do_something(client):
    response = await client.post(
        "/api/myfeature/do-something",
        json={"some_field": "test"},
    )
    assert response.status_code == 200
    data = await response.get_json()
    assert data["success"] is True
```

Run tests:

```bash
cd packages/cosma-backend && uv run --group test pytest --no-cov
```

---

## Available Services on `current_app`

| Attribute | Type | Description |
|-----------|------|-------------|
| `db` | `Database` | Async SQLite with vector search |
| `pipeline` | `Pipeline` | File processing (parse/summarize/embed) |
| `searcher` | `HybridSearcher` | Semantic + keyword search |
| `watcher` | `Watcher` | Filesystem monitoring |
| `indexing_queue` | `IndexingQueue` | Debounced queue |
| `scheduler` | `Scheduler` | System-aware pause/resume |
| `updates_hub` | `Hub[Update]` | Pub/sub for SSE events |
| `filter_manager` | `FilterConfigManager` | File inclusion/exclusion patterns |
| `settings_manager` | `SettingsManager` | App settings (TOML-backed) |
| `submit_job(coro)` | method | Submit a background asyncio task |

---

## Key Patterns

### Structured Logging

Always use `sm()` for structured log messages:

```python
from cosma_backend.log import sm
logger.info(sm("Processed file", path=file.path, status="complete"))
```

No emojis in logs.

### Database Operations

```python
# Read
file = await current_app.db.get_file_by_path("/path/to/file")
dirs = await current_app.db.get_watched_directories(active_only=True)

# Write
file_id = await current_app.db.upsert_file(file)
await current_app.db.delete_file("/path/to/file")

# Search
results = await current_app.db.keyword_search("query", limit=20)
similar = await current_app.db.search_similar_files(embedding, limit=10)
```

### Queue Integration

```python
from cosma_backend.queue.indexing_queue import QueueAction

# Enqueue a file for processing
await current_app.indexing_queue.enqueue("/path/to/file", QueueAction.INDEX)

# Enqueue a deletion
await current_app.indexing_queue.enqueue("/path/to/file", QueueAction.DELETE)

# Enqueue a move
await current_app.indexing_queue.enqueue(
    "/old/path", QueueAction.MOVE, dest_path="/new/path"
)
```

### Settings Access

```python
settings = current_app.settings_manager.settings

# Read
cooldown = settings.queue.cooldown_seconds
provider = settings.summarizer.provider

# Update
current_app.settings_manager.set_by_path("queue.cooldown_seconds", 30)
current_app.settings_manager.save()
```

### File Model Lifecycle

A `File` object progresses through stages:

```
Discovery → Parsing → Summarization → Embedding
(status)  DISCOVERED → PARSED → SUMMARIZED → COMPLETE
```

Each stage populates more fields:
- **Discovery**: path, filename, extension, file_size, timestamps
- **Parsing**: content, content_hash, content_type, parsed_at
- **Summarization**: title, summary, keywords, summarized_at
- **Embedding**: embedding (numpy), embedding_model, embedded_at

---

## Adding a New Settings Section

1. Define a config dataclass in `settings.py`:

```python
@dataclass
class MyFeatureConfig:
    enabled: bool = False
    threshold: int = 100
```

2. Add it to the `Settings` dataclass:

```python
@dataclass
class Settings:
    # ... existing fields ...
    my_feature: MyFeatureConfig = field(default_factory=MyFeatureConfig)
```

3. Access it in your code:

```python
config = current_app.settings_manager.settings.my_feature
if config.enabled:
    ...
```

4. Users can update via API:

```bash
curl -X PUT http://localhost:60534/api/settings/ \
  -H "Content-Type: application/json" \
  -d '{"my_feature.enabled": true, "my_feature.threshold": 50}'
```

---

## Adding a New Filter Pattern Type

Filters use glob patterns. To add new default patterns:

1. Edit `FilterConfigManager` in the filters module
2. Add patterns to the appropriate list in the default config
3. Test with: `POST /api/filters/test`

---

## SwiftUI Frontend Sync

When adding/modifying backend endpoints, update the SwiftUI frontend:

1. **APIModels.swift**: Add Codable request/response structs with `CodingKeys` mapping snake_case to camelCase
2. **APIClient.swift**: Add async method calling the endpoint
3. **APIModels.swift** `EventOpcode` enum: Add any new SSE opcodes
4. **AppModel.swift** `handleBackend(event:)`: Handle new SSE events

Backend uses snake_case JSON keys. Swift uses camelCase properties.
