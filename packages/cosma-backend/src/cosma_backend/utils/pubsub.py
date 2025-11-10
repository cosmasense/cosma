# adapted from https://gist.github.com/appeltel/fd3ddeeed6c330c7208502462639d2c9

import asyncio
from contextlib import contextmanager
import logging
from typing import TypeVar, Generic

logger = logging.getLogger(__name__)

T = TypeVar('T')


class Hub(Generic[T]):
    subscriptions: set[asyncio.Queue[T]]
    
    def __init__(self):
        self.subscriptions = set()

    def publish(self, message: T):
        for queue in self.subscriptions:
            queue.put_nowait(message)


@contextmanager
def subscribe(hub: Hub[T]):
    queue: asyncio.Queue[T] = asyncio.Queue()
    hub.subscriptions.add(queue)
    try:
        yield queue
    finally:
        hub.subscriptions.remove(queue)
