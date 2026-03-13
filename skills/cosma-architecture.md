# CosmaSense Architecture Overview for AI Agents

Use this skill to understand the overall system architecture before making changes.

## System Overview

CosmaSense is a local file search engine with AI-powered indexing. It runs entirely on-device.

```
┌─────────────┐    ┌─────────────┐    ┌──────────────┐
│  macOS App   │    │     TUI     │    │     CLI      │
│  (SwiftUI)   │    │  (Textual)  │    │   (Click)    │
└──────┬───────┘    └──────┬──────┘    └──────┬───────┘
       │                   │                  │
       └───────────────────┼──────────────────┘
                           │ HTTP + SSE
                    ┌──────┴──────┐
                    │   Backend   │
                    │   (Quart)   │
                    └──────┬──────┘
                           │
       ┌───────────┬───────┼───────┬───────────┐
       │           │       │       │           │
  ┌────┴───┐  ┌───┴───┐ ┌─┴──┐ ┌──┴──┐  ┌────┴────┐
  │Watcher │  │ Queue │ │ DB │ │Search│  │Pipeline │
  │watchdog│  │debounce│ │SQLite│ │hybrid│  │parse/AI │
  └────────┘  └───────┘ └────┘ └─────┘  └─────────┘
```

## Data Flow

```
File Change → Watcher → Queue (debounce 60s) → Pipeline → DB
                                                  │
                                          ┌───────┼───────┐
                                          │       │       │
                                        Parse  Summarize Embed
                                      (MarkItDown)(LLM) (e5-base-v2)
```

## Monorepo Structure

```
cosma/                          # Root package (CLI orchestrator)
├── packages/
│   ├── cosma-backend/          # Quart async web server
│   ├── cosma-tui/              # Textual terminal UI
│   └── cosma-client/           # Shared Python client lib
├── pyproject.toml              # uv workspace config
└── postman.json                # API collection for testing
```

Frontend (separate repo):
```
fileSearchForntend/             # macOS SwiftUI app
```

Documentation:
```
docs/                           # Mintlify documentation site
```

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Web Framework | Quart (async Flask) + Uvicorn |
| Database | SQLite + sqlite-vec (vectors) + FTS5 (full-text) |
| File Watching | watchdog (FSEvents on macOS) |
| File Parsing | MarkItDown (20+ formats) + macOS Spotlight |
| Summarization | LiteLLM / Ollama (local LLMs) |
| Embeddings | sentence-transformers (e5-base-v2) or OpenAI |
| Search | Hybrid: cosine similarity + BM25 |
| Frontend | SwiftUI (macOS 26) + Textual (TUI) |
| Package Manager | uv (Python), Xcode (Swift) |

## Port

Backend always runs on `localhost:60534`.

## Testing

```bash
# All tests
cd cosma && uv run pytest

# Backend only
cd cosma/packages/cosma-backend && uv run --group test pytest --no-cov

# Unit only
uv run --group test pytest tests/unit/ -m unit --no-cov

# Integration only
uv run --group test pytest tests/integration/ -m integration --no-cov
```

## Running the Backend

```bash
uv run cosma serve          # Start on port 60534
DEBUG=1 uv run cosma serve  # With debug logging
```
