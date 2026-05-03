"""Locate ffmpeg / ffprobe at runtime.

Why this module exists:

macOS apps launched from Finder (or anywhere outside a shell) get a
minimal `$PATH` — usually just `/usr/bin:/bin:/usr/sbin:/sbin`. That
means a Homebrew-installed `ffmpeg` at `/opt/homebrew/bin/ffmpeg` is
invisible to a `subprocess.run(["ffmpeg", ...])` call from the cosma
backend, even though it works fine when the user runs `ffmpeg` from
Terminal. The symptom: every video file landed in the "codec
unsupported, no transcript or frames extracted" branch because both
audio extraction and frame extraction silently 127'd out.

Resolution order:
1.  Cached result (these probes hit disk and we'd rather not).
2.  ``shutil.which(...)`` — honors any PATH the parent process did
    inherit, including dev environments that did set PATH explicitly.
3.  Known macOS Homebrew locations (``/opt/homebrew/bin`` for Apple
    Silicon, ``/usr/local/bin`` for Intel) plus ``/usr/bin``.
4.  ``imageio_ffmpeg.get_ffmpeg_exe()`` — bundles a static binary in
    the backend's wheel, so this branch always succeeds even if the
    user has no system ffmpeg installed at all.

ffprobe gets the same treatment, but imageio-ffmpeg only ships
ffmpeg, not ffprobe. For ffprobe we fall back to using ffmpeg with
``-f null`` and parsing the duration from its stderr — implemented
in media.py rather than here so this module stays focused.
"""

from __future__ import annotations

import os
import shutil
from typing import Optional

from cosma_backend.logging import get_logger

logger = get_logger(__name__)

_FFMPEG_CACHE: dict[str, Optional[str]] = {}

# macOS-friendly fallback locations checked in order. Apple Silicon
# Homebrew is `/opt/homebrew`; Intel Homebrew is `/usr/local`. Both
# coexist on Apple Silicon machines that have run Rosetta-targeted
# casks in the past, so we list both.
_MAC_FALLBACK_DIRS = [
    "/opt/homebrew/bin",
    "/usr/local/bin",
    "/usr/bin",
]


def _which_in_dirs(name: str, dirs: list[str]) -> Optional[str]:
    for directory in dirs:
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _bundled_ffmpeg() -> Optional[str]:
    """Return the path to imageio-ffmpeg's bundled binary, or None if
    the package isn't installed."""
    try:
        import imageio_ffmpeg  # type: ignore[import-not-found]
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        logger.debug("imageio-ffmpeg unavailable", error=str(exc))
        return None


def ffmpeg_path() -> str:
    """Best-effort path to a usable ffmpeg binary. Always returns a
    string; if nothing on disk works, returns the literal "ffmpeg"
    so subprocess will at least produce a clear FileNotFoundError
    that the caller can surface to the user."""
    if "ffmpeg" in _FFMPEG_CACHE:
        cached = _FFMPEG_CACHE["ffmpeg"]
        if cached is not None:
            return cached

    found = (
        shutil.which("ffmpeg")
        or _which_in_dirs("ffmpeg", _MAC_FALLBACK_DIRS)
        or _bundled_ffmpeg()
    )
    if found:
        logger.info("Resolved ffmpeg binary", path=found)
    else:
        logger.warning("No ffmpeg binary found; falling back to literal 'ffmpeg'")

    _FFMPEG_CACHE["ffmpeg"] = found
    return found or "ffmpeg"


def ffprobe_path() -> str:
    """Best-effort path to ffprobe. imageio-ffmpeg doesn't ship
    ffprobe, so when no system ffprobe is found we return the
    literal name; callers should treat ffprobe as optional and fall
    back to ffmpeg-based duration detection."""
    if "ffprobe" in _FFMPEG_CACHE:
        cached = _FFMPEG_CACHE["ffprobe"]
        if cached is not None:
            return cached

    found = shutil.which("ffprobe") or _which_in_dirs("ffprobe", _MAC_FALLBACK_DIRS)
    if found:
        logger.info("Resolved ffprobe binary", path=found)
    else:
        logger.info("ffprobe not found on disk; will fall back to ffmpeg for duration")

    _FFMPEG_CACHE["ffprobe"] = found
    return found or "ffprobe"


def have_system_ffmpeg() -> bool:
    """True iff a system ffmpeg (not the bundled fallback) is on disk.
    Used in diagnostics to tell the user whether they're running on
    the bundled binary or their own."""
    return bool(shutil.which("ffmpeg") or _which_in_dirs("ffmpeg", _MAC_FALLBACK_DIRS))
