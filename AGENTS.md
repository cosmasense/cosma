# AGENTS.md

## Codebases

There are three codebases:

- `cosma-backend` (packages/cosma-backend)
- `cosma-tui` (packages/cosma-tui)
- `cosma` (root)

## Testing
- Run all tests: `uv run pytest`
- For cosma-backend specifically: `cd packages/cosma-backend && uv run --group test pytest --no-cov`
- Run unit tests only: `uv run --group test pytest tests/unit/ -m unit --no-cov`
- Run integration tests only: `uv run --group test pytest tests/integration/ -m integration --no-cov`

## Development

- In cosma-backend: use structured logging via `sm`
  Example: `logger.info(sm("Processed file", path=file.path))`
- Do not use emojis in logs or otherwise unless explicitly stated
