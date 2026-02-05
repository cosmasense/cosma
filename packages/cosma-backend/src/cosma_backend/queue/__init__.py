"""
Indexing Queue Module

Background processing queue with debouncing, scheduling, and retry logic.

Components:
- IndexingQueue: Core queue with cooldown/debounce, pause/resume
- Scheduler: Conditional processing based on system metrics
- SystemMetricsCollector: Reads battery, GPU, CPU, etc.

Queue Flow:
1. File event → enqueue with cooldown
2. Cooldown expires → status becomes WAITING
3. Scheduler conditions met → dispatch to pipeline
4. On failure → exponential backoff retry or fallback indexing

Persistence:
- Queue items are persisted to SQLite for crash recovery
- Items are restored on startup with their original cooldowns
"""

from cosma_backend.queue.indexing_queue import IndexingQueue, QueueItem, QueueAction, QueueItemStatus

__all__ = ["IndexingQueue", "QueueItem", "QueueAction", "QueueItemStatus"]
