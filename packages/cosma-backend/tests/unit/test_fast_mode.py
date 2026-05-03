"""Tests for SummarizerConfig.fast_mode.

When fast_mode is True, the summarizer caps every file at the first
chunk regardless of document length. The output keeps a
partial-coverage note so search results stay honest about what was
actually summarized.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest

from cosma_backend.models.file import File
from cosma_backend.models.status import ProcessingStatus
from cosma_backend.settings import SummarizerConfig
from cosma_backend.summarizer.base import BaseSummarizer


class _RecordingSummarizer(BaseSummarizer):
    """Summarizer that records how many chunks it actually saw, so
    tests can assert the cap took effect."""

    def __init__(self, config: SummarizerConfig, n_synthetic_chunks: int):
        super().__init__(config=config, max_tokens=1000, model="fake")
        self._n = n_synthetic_chunks
        self.chunks_dispatched: list[int] = []

    async def is_available(self) -> bool:
        return True

    async def _prepare_content(self, content):
        # Bypass the real chunker: pretend the content split into N
        # chunks. We then route through the same _prepare_content that
        # the production code calls, but the parent class's full
        # behavior (including the max_chunks cap) lives in BaseSummarizer.
        # To keep this test honest we still call super so the cap path
        # is exercised — feed it a payload designed to chunk.
        return await super()._prepare_content(content)

    async def _get_ai_response(self, chunk, chunk_num, total_chunks, images, filename):
        self.chunks_dispatched.append(chunk_num)
        return f'{{"summary": "stub {chunk_num}", "keywords": ["k{chunk_num}"]}}'


def _make_file(content_text: str) -> File:
    now = datetime.now(timezone.utc)
    return File(
        path=Path("/tmp/x.txt"),
        file_path="/tmp/x.txt",
        filename="x.txt",
        extension=".txt",
        file_size=len(content_text),
        created=now, modified=now, accessed=now,
        content_type="text/plain",
        content=content_text,
        content_hash="h",
        status=ProcessingStatus.PARSED,
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fast_mode_caps_at_one_chunk():
    """A document that would naturally split into multiple chunks gets
    truncated to 1 when fast_mode is on."""
    # Build content long enough that the chunker produces 2+ chunks.
    long_content = ". ".join(f"sentence number {i} with some words" for i in range(2000)) + "."
    cfg = SummarizerConfig(fast_mode=True, summarize_budget_seconds=0)
    summ = _RecordingSummarizer(cfg, n_synthetic_chunks=8)
    f = _make_file(long_content)

    out = await summ.summarize(f)

    assert out.status == ProcessingStatus.SUMMARIZED
    # Exactly 1 chunk dispatched.
    assert len(summ.chunks_dispatched) == 1, (
        f"fast_mode should cap at 1 chunk, dispatched {summ.chunks_dispatched}"
    )
    # Partial-coverage note present (because the content was long
    # enough to chunk further).
    assert "partial" in (out.summary or "").lower()
    assert "fast mode" in (out.summary or "").lower()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fast_mode_off_processes_all_chunks_under_cap():
    """When fast_mode is off, normal behavior — process up to
    max_chunks (default 10)."""
    long_content = ". ".join(f"sentence number {i} with some words" for i in range(2000)) + "."
    cfg = SummarizerConfig(fast_mode=False, summarize_budget_seconds=0)
    summ = _RecordingSummarizer(cfg, n_synthetic_chunks=8)
    f = _make_file(long_content)

    out = await summ.summarize(f)

    assert out.status == ProcessingStatus.SUMMARIZED
    assert len(summ.chunks_dispatched) > 1, (
        "non-fast mode should process more than one chunk on long content"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fast_mode_short_document_no_partial_note():
    """A short document that's just one chunk anyway shouldn't get a
    misleading 'partial' note even with fast_mode on."""
    short_content = "Just one short paragraph with a title-like sentence."
    cfg = SummarizerConfig(fast_mode=True, summarize_budget_seconds=0)
    summ = _RecordingSummarizer(cfg, n_synthetic_chunks=1)
    f = _make_file(short_content)

    out = await summ.summarize(f)

    assert out.status == ProcessingStatus.SUMMARIZED
    assert len(summ.chunks_dispatched) == 1
    assert "partial" not in (out.summary or "").lower(), (
        f"short doc shouldn't have partial note: {out.summary!r}"
    )
