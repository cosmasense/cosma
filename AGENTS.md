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

## Releasing to PyPI

`.github/workflows/release.yml` builds and publishes every package in the
monorepo to PyPI on any `v*` tag push. The Swift frontend runs `uv tool
upgrade cosma` on every launch, so once a tag lands on PyPI every user
picks up the new backend automatically on their next app start.

Steps:

1. Bump `version` in every `pyproject.toml` that changed (the root
   orchestrator plus any package under `packages/*`). Keep versions in
   lockstep — `cosma` root depends on the others and mixed versions will
   confuse `uv tool`.
2. Commit the bump: `git commit -m "v0.8.0"`.
3. Tag the commit: `git tag v0.8.0`.
4. Push both: `git push && git push --tags`.
5. Watch the `release` workflow on GitHub Actions. It builds wheels,
   publishes to PyPI via trusted publishing, and drafts a release on the
   matching tag.

Notes:
- PyPI trusted publishing requires the `pypi` environment (already
  configured). No API tokens needed.
- `skip-existing: true` is set on the publish step, so re-running the
  workflow after a partial failure is safe.
- Never bump a tag without bumping the version first — PyPI rejects
  duplicate versions, and the upload step will fail silently (skipped).
