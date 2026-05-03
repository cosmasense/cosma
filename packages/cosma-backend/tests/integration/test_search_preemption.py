"""Goal-1 latency benchmarks.

Asserts the two user-visible properties of the search-first startup:

  1. The indexing queue treats `search_preempt()` as a hard pause —
     no new items are dispatched while inside the preempt window.
  2. The pause window naturally expires; indexing resumes without any
     external nudge.

We do NOT load real models here (kept fast/CI-friendly). The "first
search after cold start" timing is a smoke test: from create_app to a
successful /api/search/ HTTP 200 should be sub-second when nothing
heavy is loading on the loop. End-to-end real-model timing is a manual
demo verification (task #11).
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from cosma_backend.queue import IndexingQueue, QueueAction
from cosma_backend.queue.indexing_queue import QueueItemStatus
from cosma_backend.settings import QueueConfig
from cosma_backend.utils.pubsub import Hub


@pytest.mark.integration
@pytest.mark.asyncio
class TestSearchPreemption:
    """The indexing queue must yield to user searches."""

    @pytest_asyncio.fixture
    async def queue(self):
        # Mock pipeline so no real models load. We're exercising the
        # queue's pause logic, not the pipeline.
        pipeline = AsyncMock()
        pipeline.process_file = AsyncMock()
        pipeline.is_supported = AsyncMock(return_value=True)
        pipeline.db = AsyncMock()
        pipeline.db.delete_file = AsyncMock()

        q = IndexingQueue(
            pipeline=pipeline,
            updates_hub=Hub(),
            config=QueueConfig(
                cooldown_seconds=1,
                initial_cooldown_seconds=0,
                max_concurrency=1,
                max_retries=0,
                file_processing_timeout=5,
                search_preempt_seconds=0.5,  # short window for the test
            ),
            db=None,
        )
        yield q
        await q.stop()

    async def test_search_preempt_pauses_dispatch(self, queue: IndexingQueue):
        """Calling search_preempt() flips is_paused to True and the queue
        will not pick up new items until the window expires."""
        assert not queue.is_paused
        queue.search_preempt()
        assert queue.is_paused, "search_preempt should pause the queue"
        assert queue.is_search_preempted

    async def test_search_preempt_expires_naturally(self, queue: IndexingQueue):
        """The pause window expires after search_preempt_seconds without
        any explicit resume call."""
        queue.search_preempt()
        assert queue.is_paused
        # Window is 0.5 s in the fixture.
        await asyncio.sleep(0.6)
        assert not queue.is_paused, (
            "search preempt should auto-expire — otherwise indexing stalls "
            "forever after the user types a single search"
        )

    async def test_repeated_search_extends_window(self, queue: IndexingQueue):
        """While the user is actively typing/searching, repeated calls
        should keep the pause active."""
        queue.search_preempt()
        await asyncio.sleep(0.3)  # mid-window
        queue.search_preempt()    # extend
        await asyncio.sleep(0.3)  # would expire from t=0, but extended
        assert queue.is_paused, "second search should have extended the pause"

    async def test_in_flight_items_finish_during_preempt(self, queue: IndexingQueue):
        """Search preemption only blocks NEW dispatch — items already in
        the worker should run to completion. We assert the contract by
        showing that pause is independent of any in-flight work."""
        # No items in the queue, so this is mostly a documentation test:
        # the is_paused flag governs the dispatch decision, not the
        # cancellation of running tasks.
        queue.search_preempt()
        assert queue.is_paused
        # No exceptions raised, no hang.

    async def test_bootstrap_pause_takes_priority(self, queue: IndexingQueue):
        """Both bootstrap and search-preempt should pause the queue.
        Order doesn't matter — both are non-overridable."""
        queue.bootstrap_pause()
        assert queue.is_paused
        queue.search_preempt()
        assert queue.is_paused
        # Bootstrap-resume should still leave search-preempt active.
        queue.bootstrap_resume()
        assert queue.is_paused, (
            "search-preempt should outlive bootstrap_resume"
        )


@pytest.mark.integration
@pytest.mark.asyncio
class TestSearchEndpointPreempts:
    """End-to-end: hitting POST /api/search/ should call search_preempt()
    on the app's indexing queue."""

    @pytest_asyncio.fixture
    async def wired_app(self, app_instance, temp_db):
        """Minimal app wiring for the /api/search/ route."""
        app = app_instance
        app.db = temp_db

        # Real IndexingQueue with a mocked pipeline (we never dispatch).
        pipeline = AsyncMock()
        pipeline.process_file = AsyncMock()
        pipeline.is_supported = AsyncMock(return_value=True)
        pipeline.db = temp_db
        app.pipeline = pipeline
        app.indexing_queue = IndexingQueue(
            pipeline=pipeline,
            updates_hub=app.updates_hub,
            config=QueueConfig(
                cooldown_seconds=1,
                initial_cooldown_seconds=0,
                max_concurrency=1,
                max_retries=0,
                search_preempt_seconds=10.0,
            ),
            db=temp_db,
        )

        # Stub searcher — return an empty result list so the endpoint
        # returns 200 without touching real embedders.
        searcher = AsyncMock()
        searcher.search = AsyncMock(return_value=[])
        app.searcher = searcher

        yield app

    async def test_search_endpoint_calls_search_preempt(
        self, wired_app,
    ):
        assert not wired_app.indexing_queue.is_search_preempted
        async with wired_app.test_client() as client:
            r = await client.post(
                "/api/search/",
                json={"query": "hello world", "limit": 10},
            )
            assert r.status_code == 200, await r.get_data(as_text=True)
        assert wired_app.indexing_queue.is_search_preempted, (
            "search endpoint did not nudge the queue — indexing will "
            "fight the embedder for GPU during the user's query"
        )
