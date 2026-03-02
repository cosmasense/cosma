# AGENTS.md

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

- Do not use emojis in logs or otherwise unless explicitly stated
- Backend runs on `localhost:60534`
- API collection for manual testing: `postman.json` in this repo
