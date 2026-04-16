"""
Shared filesystem paths for on-disk models (llama.cpp, whisper.cpp).

Why a dedicated module:
- Both llama.cpp and whisper.cpp need large GGUF/bin files (~400 MB – 2 GB).
- huggingface_hub and pywhispercpp each default to their own cache locations
  (~/.cache/huggingface, ~/.local/share/pywhispercpp). That scatters models
  across the disk, makes disk-usage opaque to the user, and means the files
  don't survive when a user blows away a cache directory.
- Putting everything under the backend's user_data_dir ("~/Library/Application
  Support/cosma/models/") means: one place to inspect, one place to wipe,
  and the models persist across app reinstalls the same way the venv does.

App-name note: we intentionally reuse the existing "cosma" platformdirs
namespace (not "cosmasense") to avoid stranding the venv, database, and
settings files that already live there.
"""

from __future__ import annotations

from pathlib import Path
from platformdirs import PlatformDirs

APP_NAME = "cosma"

# Single PlatformDirs instance. `ensure_exists=False` here because we only
# materialize subdirs on demand (see ensure_models_dir). The app startup code
# in app.py uses ensure_exists=True for the top-level dir.
_dirs = PlatformDirs(APP_NAME, ensure_exists=False)


def user_data_dir() -> Path:
    """Root user data dir, e.g. ~/Library/Application Support/cosma/."""
    return Path(_dirs.user_data_dir)


def models_root() -> Path:
    """Parent of all on-disk model directories."""
    return user_data_dir() / "models"


def llama_models_dir() -> Path:
    """Where llama.cpp GGUF + mmproj files live."""
    return models_root() / "llama"


def whisper_models_dir() -> Path:
    """Where whisper.cpp ggml-*.bin files live."""
    return models_root() / "whisper"


def ensure_models_dir(kind: str) -> Path:
    """Create-and-return a model subdir (`llama` or `whisper`)."""
    mapping = {
        "llama": llama_models_dir(),
        "whisper": whisper_models_dir(),
    }
    path = mapping[kind]
    path.mkdir(parents=True, exist_ok=True)
    return path
