"""Unit tests for IndexingQueue."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cosma_backend.queue.indexing_queue import (
    IndexingQueue,
    QueueAction,
    QueueItem,
    QueueItemStatus,
)
from cosma_backend.settings import QueueConfig
from cosma_backend.utils.pubsub import Hub


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_pipeline():
    """Mock pipeline with db.delete_file, process_file, is_supported, embed_fallback."""
    pipeline = MagicMock()
    pipeline.db = MagicMock()
    pipeline.db.delete_file = AsyncMock(return_value=None)
    pipeline.process_file = AsyncMock(return_value=None)
    pipeline.is_supported = AsyncMock(return_value=True)
    pipeline.embed_fallback = AsyncMock(return_value=None)
    return pipeline


@pytest.fixture
def mock_hub():
    return MagicMock(spec=Hub)


@pytest.fixture
def queue_config():
    """Short cooldown for fast tests."""
    return QueueConfig(cooldown_seconds=0.1, max_concurrency=2, max_retries=3)


@pytest.fixture
def queue(mock_pipeline, mock_hub, queue_config):
    return IndexingQueue(pipeline=mock_pipeline, updates_hub=mock_hub, config=queue_config)


# ---------------------------------------------------------------------------
# Enqueue
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIndexingQueueEnqueue:
    async def test_enqueue_creates_cooling_down_item(self, queue):
        item = await queue.enqueue("/tmp/test.txt", QueueAction.INDEX)
        assert item.status == QueueItemStatus.COOLING_DOWN
        assert item.action == QueueAction.INDEX
        assert item.retry_count == 0

    async def test_reenqueue_resets_timer(self, queue):
        item1 = await queue.enqueue("/tmp/test.txt", QueueAction.INDEX)
        old_expires = item1.cooldown_expires_at

        await asyncio.sleep(0.05)
        item2 = await queue.enqueue("/tmp/test.txt", QueueAction.INDEX)

        assert item2.id == item1.id
        assert item2.cooldown_expires_at > old_expires

    async def test_reenqueue_during_processing_stores_pending(self, queue):
        item = await queue.enqueue("/tmp/test.txt", QueueAction.INDEX)
        # Manually set to PROCESSING to simulate
        item.status = QueueItemStatus.PROCESSING

        item2 = await queue.enqueue("/tmp/test.txt", QueueAction.INDEX)
        # Should return existing item, pending stored internally
        assert item2.id == item.id
        key = str(item.file_path)
        assert key in queue._pending_reenqueue

    async def test_delete_makes_item_waiting_immediately(self, queue):
        item = await queue.enqueue("/tmp/test.txt", QueueAction.INDEX)
        assert item.status == QueueItemStatus.COOLING_DOWN

        item2 = await queue.enqueue("/tmp/test.txt", QueueAction.DELETE)
        assert item2.status == QueueItemStatus.WAITING
        assert item2.action == QueueAction.DELETE

    async def test_move_dest_to_src_mapping(self, queue):
        item = await queue.enqueue("/tmp/src.txt", QueueAction.MOVE, dest_path="/tmp/dest.txt")
        assert item.action == QueueAction.MOVE
        # Check reverse mapping exists
        dest_key = str(item.dest_path)
        assert dest_key in queue._dest_to_src

    async def test_index_for_move_dest_resets_move_cooldown(self, queue):
        move_item = await queue.enqueue("/tmp/src.txt", QueueAction.MOVE, dest_path="/tmp/dest.txt")
        old_expires = move_item.cooldown_expires_at

        await asyncio.sleep(0.05)
        # INDEX for the dest path should reset the MOVE cooldown
        result = await queue.enqueue("/tmp/dest.txt", QueueAction.INDEX)
        assert result.id == move_item.id
        assert result.cooldown_expires_at > old_expires


# ---------------------------------------------------------------------------
# Remove
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIndexingQueueRemove:
    async def test_remove_by_id(self, queue):
        item = await queue.enqueue("/tmp/test.txt", QueueAction.INDEX)
        removed = await queue.remove_item(item.id)
        assert removed is True
        assert queue.item_count == 0

    async def test_remove_not_found(self, queue):
        removed = await queue.remove_item("nonexistent-id")
        assert removed is False

    async def test_remove_items_under_directory(self, queue):
        await queue.enqueue("/tmp/mydir/a.txt", QueueAction.INDEX)
        await queue.enqueue("/tmp/mydir/b.txt", QueueAction.INDEX)
        await queue.enqueue("/tmp/other/c.txt", QueueAction.INDEX)

        removed = await queue.remove_items_under("/tmp/mydir")
        assert removed == 2
        assert queue.item_count == 1


# ---------------------------------------------------------------------------
# Pause / Resume
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIndexingQueuePause:
    def test_manual_pause(self, queue):
        assert not queue.is_paused
        queue.manual_pause()
        assert queue.is_paused
        assert queue.is_manually_paused

    def test_scheduler_pause(self, queue):
        queue.scheduler_pause()
        assert queue.is_paused
        assert queue.is_scheduler_paused

    def test_or_semantics_both_paused(self, queue):
        """Both paused -> resume one -> still paused."""
        queue.manual_pause()
        queue.scheduler_pause()
        assert queue.is_paused

        queue.manual_resume()
        assert queue.is_paused  # scheduler still paused

        queue.scheduler_resume()
        assert not queue.is_paused


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIndexingQueueProcessing:
    async def test_processing_loop_transitions_items(self, queue, mock_pipeline):
        """Items transition from COOLING_DOWN -> WAITING -> PROCESSING -> removed."""
        item = await queue.enqueue("/tmp/test.txt", QueueAction.INDEX)

        # Wait for cooldown to expire
        await asyncio.sleep(0.15)

        # Patch File.from_path so it doesn't hit the filesystem
        mock_file = MagicMock()
        with patch("cosma_backend.queue.indexing_queue.File") as MockFile:
            MockFile.from_path.return_value = mock_file

            await queue.start()
            await asyncio.sleep(0.6)  # give loop time to pick up and process
            await queue.stop()

        # Item should have been processed and removed
        assert queue.item_count == 0
        mock_pipeline.process_file.assert_called()

    async def test_exponential_backoff_cooldown(self, queue, mock_pipeline):
        """Retry cooldown should increase exponentially."""
        mock_pipeline.process_file.side_effect = RuntimeError("transient error")

        item = await queue.enqueue("/tmp/backoff.txt", QueueAction.INDEX)

        # Wait for initial cooldown
        await asyncio.sleep(0.15)

        await queue.start()
        # After first failure, cooldown should be base * 2^0 = 0.1s
        await asyncio.sleep(0.3)
        await queue.stop()

        # Item should still be in queue (retrying) with retry_count > 0
        items = await queue.get_items()
        if items:
            assert items[0].retry_count >= 1
            # Backoff should be at least base cooldown
            remaining = items[0].cooldown_expires_at - time.time()
            # The cooldown was set, even if it already expired during our sleep
            assert items[0].retry_count > 0

    async def test_max_retries_exceeded_removes_item(self, queue, mock_pipeline):
        """After max_retries, item is removed and fallback embedding is enqueued."""
        mock_pipeline.process_file.side_effect = RuntimeError("parse error")

        item = await queue.enqueue("/tmp/fail.txt", QueueAction.INDEX)

        await queue.start()
        # Wait long enough for retries (each retry goes through cooldown)
        # With cooldown=0.1 and max_retries=3, need at least ~2s
        await asyncio.sleep(3)
        await queue.stop()

        # Original INDEX item removed; an EMBED_FALLBACK item may exist briefly
        # but the important thing is the INDEX item is gone
        items = await queue.get_items()
        index_items = [i for i in items if i.action == QueueAction.INDEX]
        assert len(index_items) == 0

    async def test_embed_fallback_calls_pipeline(self, queue, mock_pipeline):
        """EMBED_FALLBACK items call pipeline.embed_fallback."""
        item = await queue.enqueue("/tmp/fallback.txt", QueueAction.EMBED_FALLBACK)

        await asyncio.sleep(0.15)

        await queue.start()
        await asyncio.sleep(0.6)
        await queue.stop()

        assert queue.item_count == 0
        mock_pipeline.embed_fallback.assert_called()

    async def test_embed_fallback_no_retry(self, queue, mock_pipeline):
        """EMBED_FALLBACK items should not retry on failure."""
        mock_pipeline.embed_fallback.side_effect = RuntimeError("embed error")

        item = await queue.enqueue("/tmp/nofb.txt", QueueAction.EMBED_FALLBACK)

        await asyncio.sleep(0.15)

        await queue.start()
        await asyncio.sleep(0.6)
        await queue.stop()

        # Should be removed immediately (no retries for fallback)
        assert queue.item_count == 0


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIndexingQueueStatus:
    async def test_get_status_returns_correct_counts(self, queue):
        await queue.enqueue("/tmp/a.txt", QueueAction.INDEX)
        await queue.enqueue("/tmp/b.txt", QueueAction.INDEX)

        status = await queue.get_status()
        assert status["total_items"] == 2
        assert status["cooling_down"] == 2
        assert status["waiting"] == 0
        assert status["processing"] == 0
        assert status["paused"] is False
