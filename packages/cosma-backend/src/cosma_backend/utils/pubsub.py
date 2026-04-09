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
    Includes backpressure: subscriber queues are capped to prevent unbounded
    memory growth from slow or disconnected clients.
    """

    MAX_QUEUE_SIZE = 1000  # Maximum buffered messages per subscriber

    subscriptions: set[asyncio.Queue[T]]

    def __init__(self) -> None:
        self.subscriptions = set()

    def publish(self, message: T) -> None:
        """Broadcast message to all subscribed queues.

        Drops subscribers whose queue is full to prevent memory exhaustion.
        """
        dead_queues: list[asyncio.Queue[T]] = []
        for queue in self.subscriptions:
            if queue.qsize() >= self.MAX_QUEUE_SIZE:
                logger.warning("Subscriber queue full, dropping subscriber",
                              queue_size=queue.qsize())
                dead_queues.append(queue)
            else:
                queue.put_nowait(message)

        for queue in dead_queues:
            self.subscriptions.discard(queue)


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
