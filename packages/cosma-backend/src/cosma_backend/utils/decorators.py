import asyncio
from functools import wraps
from typing import TypeVar, ParamSpec, Callable, Awaitable

P = ParamSpec('P')
T = TypeVar('T')

def async_wrap(func: Callable[P, T]) -> Callable[P, Awaitable[T]]:
    """One decorator for all blocking functions"""
    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        return await asyncio.to_thread(func, *args, **kwargs)
    return wrapper
