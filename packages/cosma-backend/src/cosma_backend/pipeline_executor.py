"""Dedicated thread pool for pipeline CPU work.

Why a separate pool instead of asyncio's default executor:

- The default executor is shared with anything else that calls
  ``asyncio.to_thread``. If the pipeline ever saturates those workers
  (e.g. a burst of concurrent parses), unrelated ``to_thread`` calls
  would queue behind them. A dedicated pool walls off pipeline work.

- We set a nice value inside each worker so the OS scheduler prefers the
  event-loop thread when CPU is contended. This doesn't bypass Python's
  GIL — if pipeline code holds the GIL heavily, API handlers still wait —
  but it helps whenever the pipeline is blocked on C-extension work
  (numpy / torch MPS / tokenizer tasks that release the GIL).

- Pool size is intentionally small (2) so the pipeline can't spawn more
  concurrent heavy work than ``queue.max_concurrency`` anyway.

Usage:
    from cosma_backend.pipeline_executor import run_in_pipeline
    result = await run_in_pipeline(blocking_func, arg1, arg2)
"""
from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def _lower_priority() -> None:
    """Called once per worker thread at startup to deprioritize it."""
    try:
        os.nice(10)
    except OSError:
        pass


_PIPELINE_EXECUTOR: ThreadPoolExecutor | None = None


def get_pipeline_executor() -> ThreadPoolExecutor:
    """Return the process-wide pipeline executor, creating it on first use."""
    global _PIPELINE_EXECUTOR
    if _PIPELINE_EXECUTOR is None:
        _PIPELINE_EXECUTOR = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="pipeline-",
            initializer=_lower_priority,
        )
    return _PIPELINE_EXECUTOR


async def run_in_pipeline(func: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """Run a blocking function on the pipeline executor.

    Drop-in replacement for ``asyncio.to_thread`` for pipeline CPU work.
    API handlers should keep using ``asyncio.to_thread`` so they never
    share a pool with heavy pipeline work.
    """
    loop = asyncio.get_running_loop()
    if kwargs:
        from functools import partial
        return await loop.run_in_executor(get_pipeline_executor(), partial(func, *args, **kwargs))
    return await loop.run_in_executor(get_pipeline_executor(), func, *args)


def shutdown(wait: bool = False) -> None:
    """Tear down the pipeline executor. Called from app shutdown."""
    global _PIPELINE_EXECUTOR
    if _PIPELINE_EXECUTOR is not None:
        _PIPELINE_EXECUTOR.shutdown(wait=wait)
        _PIPELINE_EXECUTOR = None
