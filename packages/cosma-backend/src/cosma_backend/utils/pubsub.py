"""
Pub/Sub Hub for Real-time Updates

Simple publish-subscribe pattern for broadcasting events to multiple
subscribers. Used primarily for SSE updates to connected clients.

Usage:
    hub = Hub[Update]()

    # Publisher (in pipeline, queue, etc.)
    hub.publish(Update.file_complete(path, filename))

    # Subscriber (in SSE endpoint)
    with subscribe(hub) as queue:
        while True:
            update = await queue.get()
            yield update.to_sse().encode()

Adapted from: https://gist.github.com/appeltel/fd3ddeeed6c330c7208502462639d2c9
"""

import asyncio
from contextlib import contextmanager
from typing import TypeVar, Generic

from cosma_backend.logging import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class Hub(Generic[T]):
    """
    Generic pub/sub hub that broadcasts messages to all subscribers.

    Thread-safe for publishing; subscribers receive messages via asyncio.Queue.
    """

    subscriptions: set[asyncio.Queue[T]]

    def __init__(self) -> None:
        self.subscriptions = set()

    def publish(self, message: T) -> None:
        """Broadcast message to all subscribed queues."""
        for queue in self.subscriptions:
            queue.put_nowait(message)


@contextmanager
def subscribe(hub: Hub[T]):
    """
    Context manager that subscribes to a hub and yields a queue.

    The queue receives all messages published to the hub while subscribed.
    Automatically unsubscribes on context exit.
    """
    queue: asyncio.Queue[T] = asyncio.Queue()
    hub.subscriptions.add(queue)
    try:
        yield queue
    finally:
        hub.subscriptions.remove(queue)
