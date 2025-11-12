# AGENTS.md

## Codebases

There are three codebases:

- `cosma-backend` (packages/cosma-backend)
- `cosma-tui` (packages/cosma-tui)
- `cosma` (root)

## Testing
- Run tests with `uv run pytest`

## Development

- In cosma-backend: use structured logging via `sm`
  Example: `logger.info(sm("Processed file", path=file.path))`
- Do not use emojis in logs or otherwise unless explicitly stated
