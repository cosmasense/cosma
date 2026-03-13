"""
Memory release utilities.

After unloading ML models (SentenceTransformer, llama.cpp, whisper),
Python's allocator keeps freed pages mapped. On macOS this shows up
as high "Memory" in Activity Monitor even though the model is gone.

``release_memory()`` forces the OS to reclaim those pages by:
1. Running multiple GC passes to break reference cycles.
2. Clearing PyTorch's internal caches (CUDA + MPS + CPU allocator).
3. Calling macOS ``malloc_zone_pressure_relief`` via ctypes to tell
   the system malloc zones to return pages to the kernel.
"""

from __future__ import annotations

import ctypes
import gc
import platform
import sys

from cosma_backend.logging import get_logger

logger = get_logger(__name__)


def release_memory() -> None:
    """Aggressively release freed memory back to the OS."""
    # 1. Multiple GC passes — first pass may reveal new cycles
    for _ in range(3):
        gc.collect()

    # 2. Clear PyTorch caches if torch was imported
    if "torch" in sys.modules:
        try:
            import torch

            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            # Clear the CPU caching allocator (PyTorch ≥2.0)
            if hasattr(torch, "clear_autocast_cache"):
                torch.clear_autocast_cache()
        except Exception as exc:
            logger.debug("torch cache clear failed", error=str(exc))

    # 3. macOS: ask malloc zones to release free pages to the kernel
    if platform.system() == "Darwin":
        try:
            libc = ctypes.CDLL("libSystem.B.dylib")
            # int malloc_zone_pressure_relief(malloc_zone_t *zone, size_t goal)
            # zone=NULL (0) means all zones, goal=0 means release as much as possible
            libc.malloc_zone_pressure_relief(0, 0)
        except Exception as exc:
            logger.debug("malloc_zone_pressure_relief failed", error=str(exc))

    # 4. Linux: malloc_trim returns free heap pages to the OS
    elif platform.system() == "Linux":
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception as exc:
            logger.debug("malloc_trim failed", error=str(exc))
