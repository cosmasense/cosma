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

## Dev loop for pipeline failures (COSMA_DEV=1)

When `COSMA_DEV=1` is set in the backend's environment (the frontend
propagates it when the same var is set via `launchctl`), every file that
lands in `FAILED` status gets appended as a JSON line to:

```
~/Library/Logs/cosma/failed-items.jsonl
```

Each record has `timestamp, file_path, extension, phase, error` — enough
to cluster failures by extension or error message.

Iteration loop:

1. Backend runs, some files fail → lines appear in `failed-items.jsonl`.
2. Tail or `jq` the file to find the next error class worth fixing:
   `jq -r '.error' ~/Library/Logs/cosma/failed-items.jsonl | sort | uniq -c | sort -rn`
3. Fix the code in the backend.
4. Restart the backend (dev mode picks up source changes on restart — no rebuild).
5. Re-enqueue every FAILED file and wipe the log for a clean slate:
   `curl -X POST http://localhost:60534/api/queue/retry_all_failed`
6. Tail `failed-items.jsonl` again; repeat until empty.

The `/api/queue/retry_all_failed` endpoint returns 403 when COSMA_DEV is
unset, so production users can never hit it.
