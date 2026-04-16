#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

uv sync --all-packages

# uv creates .pth files with macOS UF_HIDDEN flag; Python 3.13 skips hidden .pth files
chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null || true

exec uv run cosma serve "$@"
