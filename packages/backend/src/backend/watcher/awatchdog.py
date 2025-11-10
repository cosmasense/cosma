import asyncio
import functools
from pathlib import Path
from typing import Optional

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver


class _EventHandler(FileSystemEventHandler):
    def __init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop,
                 *args, **kwargs):
        self._loop = loop
        self._queue = queue
        super().__init__(*args, **kwargs)

    def on_created(self, event: FileSystemEvent) -> None:
        self._loop.call_soon_threadsafe(self._queue.put_nowait, event)
    
    def on_modified(self, event: FileSystemEvent) -> None:
        self._loop.call_soon_threadsafe(self._queue.put_nowait, event)
    
    def on_deleted(self, event: FileSystemEvent) -> None:
        self._loop.call_soon_threadsafe(self._queue.put_nowait, event)
    
    def on_moved(self, event: FileSystemEvent) -> None:
        self._loop.call_soon_threadsafe(self._queue.put_nowait, event)


class EventIterator:
    def __init__(self, queue: asyncio.Queue,
                 loop: Optional[asyncio.BaseEventLoop] = None):
        self.queue = queue

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self.queue.get()

        if item is None:
            raise StopAsyncIteration

        return item


def _watch(path: Path, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop,
          recursive: bool = False) -> BaseObserver:
    """Watch a directory for changes."""

    handler = _EventHandler(queue, loop)

    observer = Observer()
    observer.schedule(handler, str(path), recursive=recursive)
    observer.start()
    return observer
    

async def watch(path: Path, queue: asyncio.Queue, recursive: bool = False) -> BaseObserver:
    loop = asyncio.get_running_loop()
    partial = functools.partial(_watch, path=path, queue=queue, loop=loop, recursive=recursive)
    return await asyncio.to_thread(partial)


# async def consume(queue: asyncio.Queue) -> None:
#     async for event in EventIterator(queue):
#         print("Got an event!", event)


# if __name__ == "__main__":
#     loop = asyncio.get_event_loop()
#     queue = asyncio.Queue(loop=loop)

#     futures = [
#         loop.run_in_executor(None, watch, Path("."), queue, loop, False),
#         consume(queue),
#     ]

#     loop.run_until_complete(asyncio.gather(*futures))
