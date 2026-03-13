"""
Async Utility Decorators

Helpers for running blocking code in async contexts.
"""

import asyncio
from functools import wraps
from typing import TypeVar, ParamSpec, Callable, Awaitable

P = ParamSpec('P')
T = TypeVar('T')


def async_wrap(func: Callable[P, T]) -> Callable[P, Awaitable[T]]:
    """
    Decorator that wraps a synchronous function to run in a thread pool.

    Use this for CPU-bound or blocking I/O operations that would otherwise
    block the event loop. The wrapped function becomes awaitable.

    Example:
        @async_wrap
        def blocking_io():
            return Path(file).read_text()

        content = await blocking_io()  # Runs in thread pool
    """
    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        return await asyncio.to_thread(func, *args, **kwargs)
    return wrapper
