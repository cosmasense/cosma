# AGENTS.md

## Codebases

There are four codebases:

- `cosma-backend` (packages/cosma-backend) - Backend service with vector search, file watching, document parsing, and LLM integration
- `cosma-client` (packages/cosma-client) - API client library for communicating with the backend
- `cosma-tui` (packages/cosma-tui) - Terminal user interface built with Textual
- `cosma` (root) - Main CLI with commands for indexing, searching, and managing the backend

## Testing
- Run all tests: `uv run pytest`
- For cosma-backend specifically: `cd packages/cosma-backend && uv run --group test pytest --no-cov`
- Run unit tests only: `uv run --group test pytest tests/unit/ -m unit --no-cov`
- Run integration tests only: `uv run --group test pytest tests/integration/ -m integration --no-cov`

## Development

- In cosma-backend: use structured logging via `sm`
  Example: `logger.info(sm("Processed file", path=file.path))`
- Do not use emojis in logs or otherwise unless explicitly stated
