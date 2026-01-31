# AGENTS.md

## Skills

Detailed guides for AI agents are in the `skills/` directory (one level up from this repo):

- **`skills/cosma-architecture.md`** - System overview, data flow, tech stack, storage locations
- **`skills/cosma-api.md`** - Complete API reference (25 endpoints), workflows, SSE opcodes
- **`skills/cosma-backend-dev.md`** - Backend development: adding endpoints, models, queue integration, settings, tests
- **`skills/cosma-swiftui-dev.md`** - SwiftUI frontend: adding API calls, SSE events, CodingKeys patterns

Read the relevant skill file before making changes.

## Codebases

There are four codebases:

- `cosma-backend` (packages/cosma-backend) - Quart async web server + SQLite + AI pipeline
- `cosma-client` (packages/cosma-client) - Shared Python client library
- `cosma-tui` (packages/cosma-tui) - Textual terminal UI
- `cosma` (root) - Click CLI orchestrator

Frontend (separate repo):
- `fileSearchForntend/` - macOS SwiftUI app

## Testing
- Run all tests: `uv run pytest`
- For cosma-backend specifically: `cd packages/cosma-backend && uv run --group test pytest --no-cov`
- Run unit tests only: `uv run --group test pytest tests/unit/ -m unit --no-cov`
- Run integration tests only: `uv run --group test pytest tests/integration/ -m integration --no-cov`

## Development

- In cosma-backend: use structured logging via `sm`
  Example: `logger.info(sm("Processed file", path=file.path))`
- Do not use emojis in logs or otherwise unless explicitly stated
- Backend runs on `localhost:60534`
- API collection for manual testing: `postman.json` in this repo
