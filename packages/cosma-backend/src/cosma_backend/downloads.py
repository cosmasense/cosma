"""
Progress-aware model downloads for llama.cpp and whisper.cpp.

Why this module exists separately from providers.py / media.py:
- Both the summarizer and the parser need to download models, but the actual
  provider code should stay focused on *using* the models, not on the
  cache/download plumbing.
- The bootstrap API streams progress over SSE. Centralizing downloads here
  gives a single choke-point to hook progress callbacks, so the status
  endpoint can report "{bytes_done, bytes_total}" without every caller having
  to implement its own tqdm shim.
- Downloads are idempotent: if the target file already exists on disk with
  non-zero size, we skip the network round-trip entirely. That's why the
  setup wizard is fast on reinstall.

Downloads resolve HuggingFace glob patterns (e.g. "*Q4_K_M.gguf") to a
concrete filename on the remote repo, then pin to local_dir=<models_dir>
so files land in the shared user_data_dir, not the HF cache.
"""

from __future__ import annotations

import contextlib
import fnmatch
import io
import os
import threading

# Silence huggingface_hub's tqdm progress bar. When the backend is launched
# by the Swift app its stdout is piped; tqdm's writes can SIGPIPE mid-download
# (observed as "[Errno 32] Broken pipe" aborting the whisper model pull).
# Our own filesystem poller drives the UI progress, so we don't need tqdm.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from cosma_backend.logging import get_logger
from cosma_backend.paths import ensure_models_dir

logger = get_logger(__name__)

# Progress callback signature: (bytes_done, bytes_total, stage_label).
# bytes_total may be 0 if unknown (rare for HF downloads).
ProgressCb = Callable[[int, int, str], None]


@dataclass
class DownloadResult:
    path: Path
    skipped: bool  # True if file already existed; no bytes transferred


def _resolve_hf_filename(repo_id: str, filename: str) -> str:
    """Resolve a glob pattern like '*Q4_K_M.gguf' to a concrete repo filename.

    llama-cpp-python's Llama.from_pretrained accepts globs, but hf_hub_download
    does not — so we list the repo once and pick the first match. Prefer an
    exact match; fall back to glob.
    """
    if "*" not in filename and "?" not in filename:
        return filename

    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(repo_id)
    matches = [f for f in files if fnmatch.fnmatch(f, filename)]
    if not matches:
        raise FileNotFoundError(f"No files in {repo_id} match pattern {filename!r}")
    # Prefer the shortest match — avoids accidentally picking a "-of-00002"
    # shard when the user intended the single-file quant.
    matches.sort(key=len)
    return matches[0]


def download_hf_file(
    *,
    repo_id: str,
    filename: str,
    kind: str,  # "llama" | "whisper"
    progress: Optional[ProgressCb] = None,
    stage_label: str = "",
    subdir: Optional[str] = None,
) -> DownloadResult:
    """Download a single file from HuggingFace into our managed models dir.

    Idempotent: returns DownloadResult(skipped=True) if the resolved file
    already exists. Progress is reported by polling on-disk bytes.

    Why the `subdir` parameter:
    When multiple downloads run in parallel with the same local_dir, HF
    writes each incomplete blob to <local_dir>/.cache/huggingface/download/
    <hash>.incomplete — a shared cache dir where the filenames are opaque
    content hashes. That makes it ambiguous which .incomplete belongs to
    which caller. Giving each stage its own subdir means exactly one
    .incomplete exists under each dir's .cache — trivial to watch.
    """
    base = ensure_models_dir(kind)
    target_dir = base / subdir if subdir else base
    target_dir.mkdir(parents=True, exist_ok=True)
    concrete = _resolve_hf_filename(repo_id, filename)
    dest = target_dir / Path(concrete).name

    if dest.exists() and dest.stat().st_size > 0:
        logger.info("model already present, skipping download",
                    repo_id=repo_id, filename=concrete, path=str(dest))
        if progress:
            size = dest.stat().st_size
            progress(size, size, stage_label or concrete)
        return DownloadResult(path=dest, skipped=True)

    from huggingface_hub import hf_hub_download, HfApi

    # Pre-fetch the expected total size so our filesystem poller can
    # report a real percentage. Falls back to 0 (indeterminate) if the
    # metadata call fails — the poller still emits byte counts.
    total_bytes = 0
    try:
        info = HfApi().get_paths_info(repo_id=repo_id, paths=[concrete])
        if info and getattr(info[0], "size", None):
            total_bytes = int(info[0].size)
    except Exception as e:
        logger.debug("Could not pre-fetch size metadata", error=str(e))

    # HF writes to <dest>.incomplete during transfer. We watch both
    # paths because older HF versions write directly to dest, newer
    # versions use the .incomplete suffix, and we don't want to miss
    # progress by guessing wrong.
    label = stage_label or concrete
    stop = threading.Event()
    poller: Optional[threading.Thread] = None

    if progress is not None:
        poller = threading.Thread(
            target=_poll_size,
            args=(dest, stop, progress, label, total_bytes),
            daemon=True,
        )
        poller.start()

    try:
        # local_dir pins the file to our managed directory; without this,
        # HF stores via content-hash symlinks under ~/.cache/huggingface,
        # which defeats the "one place to inspect" design.
        #
        # Redirect stdout/stderr to a discard buffer for the duration of the
        # HF call: when the backend runs as a child of the Swift app, its
        # stdout is piped, and any stray write from deep inside huggingface_hub
        # / urllib3 / tqdm can SIGPIPE (seen as "[Errno 32] Broken pipe" that
        # aborted whisper model downloads). We already surface progress via
        # the filesystem poller, so suppressing library chatter is safe.
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            path = hf_hub_download(
                repo_id=repo_id,
                filename=concrete,
                local_dir=str(target_dir),
            )
        logger.info("model downloaded",
                    repo_id=repo_id, filename=concrete, path=path)
        return DownloadResult(path=Path(path), skipped=False)
    finally:
        stop.set()
        if poller is not None:
            poller.join(timeout=1.0)
        # Final 100%-done event so the frontend's fraction hits 1.0 even
        # if the last poll undershoots.
        if progress is not None and dest.exists():
            size = dest.stat().st_size
            progress(size, total_bytes or size, label)


def _poll_size(
    dest: Path,
    stop: "threading.Event",
    cb: ProgressCb,
    label: str,
    total: int,
) -> None:
    """Emit progress events based on the growing file size on disk.

    Runs in a background thread. Checks, in order:
      1. The final dest path (once HF moves it into place).
      2. Any `*.incomplete` in <dest_dir>/.cache/huggingface/download/
         — HF's canonical in-progress location. With one-download-per-subdir
         there is at most one such file per stage, so a naive max() wins.

    Polling interval is ~200 ms: slow enough to be invisible in the I/O
    budget, fast enough to feel responsive in the UI.
    """
    cache_dir = dest.parent / ".cache" / "huggingface" / "download"
    while not stop.is_set():
        done = 0
        try:
            done = dest.stat().st_size
        except FileNotFoundError:
            pass
        if done == 0 and cache_dir.exists():
            # Pick the largest .incomplete — with one download per subdir
            # there's effectively only one, but max() is cheap and robust
            # to HF re-resolving hashes between runs.
            try:
                for child in cache_dir.iterdir():
                    if child.name.endswith(".incomplete"):
                        s = child.stat().st_size
                        if s > done:
                            done = s
            except FileNotFoundError:
                pass
        try:
            cb(done, total, label)
        except Exception:
            pass  # never let a progress exception kill the download
        stop.wait(0.2)


