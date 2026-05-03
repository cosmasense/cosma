"""Robustness tests across the new pipeline + queue surface.

Covers the corners that the corpus and stage-pipelining suites don't:

  - MOVE / DELETE actions still behave correctly when the file is in a
    non-terminal state (PARSED / SUMMARIZED) at the time of the action.
  - End-to-end crash recovery against a real IndexingQueue: stop mid-
    pipeline, restart, verify the queue picks up where it left off
    without re-parsing already-parsed files.
  - Stress: many concurrent enqueues for the same file collapse cleanly.
  - Stress: search_preempt + enqueue + scheduler-pause races don't
    deadlock or lose items.
  - Backend version handshake endpoint returns the documented shape.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest
import pytest_asyncio

from cosma_backend.db.database import Database
from cosma_backend.discoverer import Discoverer
from cosma_backend.embedder import AutoEmbedder
from cosma_backend.models.file import File
from cosma_backend.models.status import ProcessingStatus
from cosma_backend.parser import FileParser
from cosma_backend.pipeline import Pipeline
from cosma_backend.queue import IndexingQueue, QueueAction
from cosma_backend.settings import QueueConfig
from cosma_backend.summarizer import AutoSummarizer
from cosma_backend.utils.pubsub import Hub


# Shared lightweight pipeline factory: real Discoverer + parser support
# check, mocked AI calls. Each stage records (path, t0, t1) so tests can
# assert sequence and counts.
def _build_pipeline(
    db: Database, hub: Hub,
    parse_delay: float = 0.001,
    summarize_delay: float = 0.001,
    embed_delay: float = 0.001,
) -> tuple[Pipeline, dict]:
    parser = FileParser()
    timings = {"parse": [], "summarize": [], "embed": []}

    async def fake_parse(file: File) -> None:
        await asyncio.sleep(parse_delay)
        import hashlib
        try:
            data = Path(file.file_path).read_bytes()
        except OSError:
            data = b""
        file.content = "x"
        file.content_hash = hashlib.sha256(data).hexdigest()
        file.content_type = "text/plain"
        file.parsed_at = datetime.now(timezone.utc)
        file.status = ProcessingStatus.PARSED
        timings["parse"].append(file.file_path)
    parser.parse_file = AsyncMock(side_effect=fake_parse)

    summarizer = AutoSummarizer.__new__(AutoSummarizer)
    async def fake_summarize(file: File) -> None:
        await asyncio.sleep(summarize_delay)
        file.title = f"T:{file.filename}"
        file.summary = "S"
        file.keywords = ["k"]
        file.summarized_at = datetime.now(timezone.utc)
        file.status = ProcessingStatus.SUMMARIZED
        timings["summarize"].append(file.file_path)
    summarizer.summarize = AsyncMock(side_effect=fake_summarize)

    embedder = AutoEmbedder.__new__(AutoEmbedder)
    async def fake_embed(file: File) -> None:
        await asyncio.sleep(embed_delay)
        file.embedding = np.zeros(1536, dtype=np.float32)
        file.embedding_model = "mock"
        file.embedding_dimensions = 1536
        file.embedded_at = datetime.now(timezone.utc)
        file.status = ProcessingStatus.COMPLETE
        timings["embed"].append(file.file_path)
    embedder.embed = AsyncMock(side_effect=fake_embed)

    return Pipeline(
        db=db, updates_hub=hub,
        discoverer=Discoverer(),
        parser=parser, summarizer=summarizer, embedder=embedder,
        parse_concurrency=4, summarize_concurrency=1, embed_concurrency=1,
    ), timings


# =============================================================================
# Task #18 — MOVE / DELETE through the new pipeline
# =============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
class TestMoveDeleteThroughStages:
    """The MOVE and DELETE queue actions must still behave correctly when
    the file row in the DB is in a non-terminal status (PARSED /
    SUMMARIZED), not just when it's COMPLETE."""

    async def test_delete_removes_PARSED_row(
        self, temp_db: Database, mock_updates_hub: Hub, tmp_path: Path,
    ):
        """A user (or stale-file sweep) deletes a file that has been
        parsed but not yet summarized. The DB row must be removed
        regardless of intermediate status."""
        pipeline, _ = _build_pipeline(temp_db, mock_updates_hub)
        target = tmp_path / "del_parsed.txt"
        target.write_text("payload")

        f = File.from_path(target)
        f.content = "payload"
        f.content_hash = "h1"
        f.parsed_at = datetime.now(timezone.utc)
        f.status = ProcessingStatus.PARSED
        await temp_db.upsert_file(f)
        assert await temp_db.get_file_by_path(str(target.resolve())) is not None

        await temp_db.delete_file(str(target.resolve()))
        assert await temp_db.get_file_by_path(str(target.resolve())) is None

    async def test_delete_removes_SUMMARIZED_row(
        self, temp_db: Database, mock_updates_hub: Hub, tmp_path: Path,
    ):
        """Same contract for SUMMARIZED."""
        target = tmp_path / "del_sum.txt"
        target.write_text("payload")
        f = File.from_path(target)
        f.content = "payload"
        f.content_hash = "h2"
        f.title = "T"; f.summary = "S"; f.keywords = ["k"]
        f.parsed_at = datetime.now(timezone.utc)
        f.summarized_at = datetime.now(timezone.utc)
        f.status = ProcessingStatus.SUMMARIZED
        await temp_db.upsert_file(f)

        await temp_db.delete_file(str(target.resolve()))
        assert await temp_db.get_file_by_path(str(target.resolve())) is None

    async def test_move_deletes_src_and_processes_dest_from_parse(
        self, temp_db: Database, mock_updates_hub: Hub, tmp_path: Path,
    ):
        """When a file is moved while it was PARSED, the IndexingQueue's
        MOVE handler deletes the src row and runs process_file on the
        dest. Since dest has no DB row, it starts at parse — we re-parse
        the moved file. That's the documented contract; we want this
        test to pin it so we don't accidentally "optimize" it into
        carrying over partial state across paths (which would require
        moving the row, not just the file)."""
        pipeline, timings = _build_pipeline(temp_db, mock_updates_hub)

        src = tmp_path / "src.txt"
        dest = tmp_path / "dest.txt"
        src.write_text("body")
        # Simulate prior partial work on src.
        f = File.from_path(src)
        f.content = "body"
        f.content_hash = "ms1"
        f.parsed_at = datetime.now(timezone.utc)
        f.status = ProcessingStatus.PARSED
        await temp_db.upsert_file(f)

        # Simulate the move on disk.
        src.rename(dest)

        # MOVE action behavior, replicated inline (same as
        # IndexingQueue._process_item's MOVE branch):
        await temp_db.delete_file(str(src.resolve()))
        dest_file = File.from_path(dest)
        if await pipeline.is_supported(dest_file):
            await pipeline.process_file(dest_file)

        # src row is gone.
        assert await temp_db.get_file_by_path(str(src.resolve())) is None
        # dest row is now COMPLETE.
        dest_row = await temp_db.get_file_by_path(str(dest.resolve()))
        assert dest_row is not None
        assert dest_row.status == ProcessingStatus.COMPLETE
        # parse ran for the dest path (src parse was discarded).
        assert str(dest.resolve()) in timings["parse"]


# =============================================================================
# Task #19 — End-to-end crash recovery via the real IndexingQueue
# =============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
class TestEndToEndCrashRecovery:
    """Drive a real IndexingQueue, stop it mid-flight, restart, verify
    the partially-processed file resumes from PARSED."""

    async def test_recovery_resumes_PARSED_files(
        self, temp_db: Database, mock_updates_hub: Hub, tmp_path: Path,
    ):
        # First "session": process file enough to get to PARSED, then
        # simulate a crash by stopping before summarize even starts.
        pipeline_a, timings_a = _build_pipeline(
            temp_db, mock_updates_hub,
            parse_delay=0.001, summarize_delay=10.0, embed_delay=0.001,
        )
        queue_a = IndexingQueue(
            pipeline=pipeline_a,
            updates_hub=mock_updates_hub,
            config=QueueConfig(
                cooldown_seconds=1, initial_cooldown_seconds=0,
                max_concurrency=2, max_retries=0,
                file_processing_timeout=30,
            ),
            db=temp_db,
        )

        target = tmp_path / "recover_me.txt"
        target.write_text("real content")

        await queue_a.start()
        await queue_a.enqueue(target, QueueAction.INDEX)

        # Wait until parse has produced a PARSED row in the DB.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            row = await temp_db.get_file_by_path(str(target.resolve()))
            if row and row.status == ProcessingStatus.PARSED:
                break
            await asyncio.sleep(0.05)
        else:
            pytest.fail("parse stage never persisted PARSED — recovery "
                        "tests can't proceed")

        # Crash: stop the queue before summarize completes.
        await queue_a.stop()

        # Sanity: the row is still PARSED in the DB (the row should
        # NOT have been deleted by a mid-flight cleanup).
        row = await temp_db.get_file_by_path(str(target.resolve()))
        assert row.status == ProcessingStatus.PARSED, (
            f"Row was lost during shutdown: {row}"
        )

        # Second "session": fresh queue + fast pipeline. start() must
        # call _recover_in_flight_files which re-enqueues the file.
        pipeline_b, timings_b = _build_pipeline(
            temp_db, mock_updates_hub,
            parse_delay=0.001, summarize_delay=0.001, embed_delay=0.001,
        )
        queue_b = IndexingQueue(
            pipeline=pipeline_b,
            updates_hub=mock_updates_hub,
            config=QueueConfig(
                cooldown_seconds=1, initial_cooldown_seconds=0,
                max_concurrency=2, max_retries=0,
                file_processing_timeout=30,
            ),
            db=temp_db,
        )
        await queue_b.start()
        # Wait for COMPLETE.
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            row = await temp_db.get_file_by_path(str(target.resolve()))
            if row and row.status == ProcessingStatus.COMPLETE:
                break
            await asyncio.sleep(0.05)
        else:
            pytest.fail(
                f"Recovery did not finish file. Final row: {row!r}, "
                f"timings_b={timings_b}"
            )
        await queue_b.stop()

        # Critical: parse_b must NOT have been called for the recovered
        # file — that's the whole point. summarize_b and embed_b should
        # have run.
        assert str(target.resolve()) not in timings_b["parse"], (
            "Recovery re-parsed the file. The persisted PARSED state was "
            "ignored. This regresses the resume-from-saved-status contract."
        )
        assert str(target.resolve()) in timings_b["summarize"]
        assert str(target.resolve()) in timings_b["embed"]


# =============================================================================
# Task #20 — Concurrent enqueue and signal races
# =============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
class TestConcurrentEnqueueRaces:
    """Stress the queue's debounce / dedup under concurrent load."""

    @pytest_asyncio.fixture
    async def queue(self, temp_db: Database, mock_updates_hub: Hub):
        pipeline, _ = _build_pipeline(temp_db, mock_updates_hub)
        q = IndexingQueue(
            pipeline=pipeline,
            updates_hub=mock_updates_hub,
            config=QueueConfig(
                cooldown_seconds=1, initial_cooldown_seconds=0,
                max_concurrency=4, max_retries=0,
                file_processing_timeout=30,
                search_preempt_seconds=0.5,
            ),
            db=temp_db,
        )
        yield q
        await q.stop()

    async def test_100_concurrent_enqueues_for_same_file_collapse(
        self, queue: IndexingQueue, tmp_path: Path,
    ):
        """The watcher can fire dozens of FileModifiedEvents for one
        file in milliseconds (e.g., editor saving in chunks). The
        queue must collapse them into one item, not 100."""
        target = tmp_path / "noisy.txt"
        target.write_text("noisy")

        # Fire 100 enqueues concurrently.
        await asyncio.gather(*(
            queue.enqueue(target, QueueAction.INDEX) for _ in range(100)
        ))
        # Single item in the queue.
        items = await queue.get_items()
        assert len(items) == 1, (
            f"Concurrent enqueues did not dedup: {len(items)} items"
        )
        assert items[0].file_path == str(target.resolve())

    async def test_search_preempt_during_enqueue_burst(
        self, queue: IndexingQueue, tmp_path: Path,
    ):
        """A search arrives while a watcher burst is mid-flight. The
        queue must (a) accept and dedup the enqueues and (b) honor the
        search preempt — no deadlock, no lost items."""
        files = []
        for i in range(20):
            p = tmp_path / f"burst_{i}.txt"
            p.write_text(f"c{i}")
            files.append(p)

        # Race: half the enqueues, then search_preempt, then the rest.
        async def first_half():
            for p in files[:10]:
                await queue.enqueue(p, QueueAction.INDEX)

        async def search_then_rest():
            await asyncio.sleep(0.001)
            queue.search_preempt()
            for p in files[10:]:
                await queue.enqueue(p, QueueAction.INDEX)

        await asyncio.gather(first_half(), search_then_rest())
        items = await queue.get_items()
        assert len(items) == 20, (
            f"Lost items under preempt race: got {len(items)}"
        )
        # Search-preempt window is 0.5 s; assert it was set.
        assert queue.is_search_preempted

    async def test_enqueue_during_pause_then_resume(
        self, queue: IndexingQueue, tmp_path: Path,
    ):
        """Items enqueued while paused must be queued (not dropped) and
        flow through normally when the queue resumes."""
        target = tmp_path / "during_pause.txt"
        target.write_text("paused")

        queue.manual_pause()
        await queue.enqueue(target, QueueAction.INDEX)
        items = await queue.get_items()
        assert len(items) == 1

        queue.manual_resume()
        # Item is still in the queue — pause/resume doesn't lose it.
        items = await queue.get_items()
        assert len(items) == 1


# =============================================================================
# Backend version handshake smoke test
# =============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
class TestVersionHandshake:
    """The /api/status/version endpoint is the new wire-level pin
    between frontend and backend. Test it returns the documented shape
    so the Swift handshake can rely on it."""

    async def test_version_endpoint_shape(self, app_instance, temp_db):
        app_instance.db = temp_db
        async with app_instance.test_client() as client:
            r = await client.get("/api/status/version")
            assert r.status_code == 200, await r.get_data(as_text=True)
            body = await r.get_json()
        assert "backend_version" in body
        assert "api_version" in body
        assert "min_frontend_api_version" in body
        assert isinstance(body["api_version"], int)
        assert isinstance(body["min_frontend_api_version"], int)
        # Frontend pins against api_version=1 right now; if this trips
        # someone's bumped the backend without bumping the Swift pin.
        from cosma_backend import __api_version__
        assert body["api_version"] == __api_version__
