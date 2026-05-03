"""Tests for the per-file summarize-budget cap.

When a file's chunks would exceed the configured wall-clock budget,
we stop dispatching new chunks and finalize with whatever chunk
summaries we already have. The file ends as SUMMARIZED with a
partial-coverage note in the summary, NOT FAILED.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
import pytest_asyncio

from cosma_backend.models.file import File
from cosma_backend.models.status import ProcessingStatus
from cosma_backend.settings import SummarizerConfig
from cosma_backend.summarizer.base import BaseSummarizer


class _FakeSlowSummarizer(BaseSummarizer):
    """Minimal concrete BaseSummarizer where each chunk takes a
    configurable wall-clock duration."""

    def __init__(
        self, config: SummarizerConfig, chunk_seconds: float, n_chunks: int,
    ):
        super().__init__(config=config, max_tokens=1000, model="fake")
        self._chunk_seconds = chunk_seconds
        self._n_chunks = n_chunks

    async def is_available(self) -> bool:
        return True

    async def _prepare_content(self, content):
        # Pretend we split into N predictable chunks regardless of
        # input — we only care about the loop, not the splitter.
        return [f"chunk {i}" for i in range(self._n_chunks)]

    async def _get_ai_response(self, chunk, chunk_num, total_chunks, images, filename):
        await asyncio.sleep(self._chunk_seconds)
        # Hand back a parseable JSON response.
        title = "Stubbed Title" if chunk_num == 0 else None
        if title:
            return f'{{"title": "{title}", "summary": "stub {chunk_num}", "keywords": ["k{chunk_num}"]}}'
        return f'{{"summary": "stub {chunk_num}", "keywords": ["k{chunk_num}"]}}'


def _file_with_content(text: str = "x" * 5000) -> File:
    now = datetime.now(timezone.utc)
    return File(
        path=Path("/tmp/bench.txt"),
        file_path="/tmp/bench.txt",
        filename="bench.txt",
        extension=".txt",
        file_size=len(text),
        created=now, modified=now, accessed=now,
        content_type="text/plain",
        content=text,
        content_hash="abc",
        status=ProcessingStatus.PARSED,
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_budget_stops_dispatch_on_long_file():
    """Five chunks × 0.6 s each = 3 s of work; 1.5 s budget means we
    expect roughly 2-3 chunks done, partial-coverage note appended."""
    cfg = SummarizerConfig(summarize_budget_seconds=1.5)
    summ = _FakeSlowSummarizer(cfg, chunk_seconds=0.6, n_chunks=5)
    f = _file_with_content()

    t0 = time.monotonic()
    out = await summ.summarize(f)
    elapsed = time.monotonic() - t0

    assert out.status == ProcessingStatus.SUMMARIZED, (
        f"file should COMPLETE not FAIL — got status={out.status} "
        f"summary={out.summary!r}"
    )
    # Wall time should be under (budget + 2 chunks of slack):
    assert elapsed < 1.5 + 1.5, f"summarize hung: {elapsed:.2f}s"
    # Partial-coverage note should be in summary:
    assert "partial" in (out.summary or "").lower(), (
        f"summary missing partial note: {out.summary!r}"
    )
    assert "budget" in (out.summary or ""), (
        f"summary missing budget note: {out.summary!r}"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_budget_lets_short_file_finish_normally():
    """One chunk that fits in the budget — no partial note, full
    summary."""
    cfg = SummarizerConfig(summarize_budget_seconds=10.0)
    summ = _FakeSlowSummarizer(cfg, chunk_seconds=0.05, n_chunks=1)
    f = _file_with_content()

    out = await summ.summarize(f)

    assert out.status == ProcessingStatus.SUMMARIZED
    assert "partial" not in (out.summary or "").lower(), out.summary
    assert out.title == "Stubbed Title"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_budget_zero_disables_cap():
    """Budget of 0 means "no cap" — even a slow multi-chunk file
    runs to completion."""
    cfg = SummarizerConfig(summarize_budget_seconds=0)
    summ = _FakeSlowSummarizer(cfg, chunk_seconds=0.05, n_chunks=4)
    f = _file_with_content()

    out = await summ.summarize(f)

    assert out.status == ProcessingStatus.SUMMARIZED
    assert "partial" not in (out.summary or "").lower(), out.summary


@pytest.mark.unit
@pytest.mark.asyncio
async def test_first_chunk_always_runs_even_if_budget_already_blown():
    """A user with a very tight budget (0.1 s) and a slow model still
    gets the first chunk attempted, since one summary > zero summary."""
    cfg = SummarizerConfig(summarize_budget_seconds=0.1)
    summ = _FakeSlowSummarizer(cfg, chunk_seconds=0.5, n_chunks=3)
    f = _file_with_content()

    out = await summ.summarize(f)

    assert out.status == ProcessingStatus.SUMMARIZED
    # Got at least the first chunk's summary.
    assert out.summary
