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

## Key Design Decisions

1. **Queue with debounce**: File changes are debounced (60s cooldown) to batch rapid edits. The queue sits between the watcher and pipeline.

2. **Dual pause**: Queue can be paused manually (user) OR by scheduler (system conditions). Queue is paused if EITHER is true.

3. **Scheduler rules**: Battery level, power source, CPU idle time, temperature, fan speed, time window. Combine with ALL (AND) or ANY (OR).

4. **Hybrid search**: Final score = semantic_score (0-0.5) + keyword_score (0-0.5). Both methods run in parallel.

5. **SSE for real-time**: All state changes published to a Hub. Clients subscribe via `/api/updates/`. Keep-alive every 15s.

6. **Filter modes**: Blacklist (exclude matching) or Whitelist (only include matching). Mode-specific pattern storage prevents data loss when switching.

7. **Processing pipeline stages**: Discovery → Parsing → Summarization → Embedding. Each stage is skipped if the file hasn't changed (hash check).

## Data Storage Locations

| Data | Path |
|------|------|
| Database | `~/Library/Application Support/cosma/cosma.db` |
| Settings | `~/Library/Application Support/cosma/settings.toml` |
| Filters | `~/Library/Application Support/cosma/filter.json` |
| Logs | stderr (structured JSON via structlog) |

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
cd cosma
uv run cosma serve          # Start on port 60534
DEBUG=1 uv run cosma serve  # With debug logging
```
