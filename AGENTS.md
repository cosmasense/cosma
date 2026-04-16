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
- Tests must pass before committing or pushing. If a test fails, fix the
  underlying code rather than the test — tests encode the API/behavior
  contract. Only update test assertions when the contract itself is
  intentionally changing, and mention that contract change in the commit.

## Development

- Do not use emojis in logs or otherwise unless explicitly stated
- Backend runs on `localhost:60534`
- API collection for manual testing: `postman.json` in this repo

## Releasing

Release + PyPI workflow lives in `docs/RELEASING.md`. Read it when tagging
a new version or debugging a publish. Short version: bump versions →
`git tag v*` → push tags → GitHub Actions does the rest; clients
auto-upgrade on next launch.
