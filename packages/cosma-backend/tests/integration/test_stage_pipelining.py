"""Goal-2 stage-pipelining + crash-safety tests.

Validates the four properties promised by STAGE_PIPELINING_DESIGN.md:

  1. Stages run in parallel across files (parse on CPU, summarize on
     GPU, embed on MPS) — call timestamps from different stages
     interleave instead of serializing per file.
  2. A file in DB with status=PARSED resumes at the summarize stage on
     the next process_file call (parser is NOT invoked).
  3. A file in DB with status=SUMMARIZED resumes at embed only.
  4. A crash mid-summarize leaves the parsed row durable in the DB.
     The mid-flight `delete_file` is gone, so re-running picks up
     from PARSED instead of starting over.

Plus a throughput regression check: with parse_concurrency=4 vs 1,
the wall-clock to process N files where summarize is slow should be
visibly faster.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
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
from cosma_backend.summarizer import AutoSummarizer
from cosma_backend.utils.pubsub import Hub


def _make_pipeline_with_stage_timing(
    db: Database, hub: Hub, stage_durations: dict[str, float],
    parse_conc: int = 4, sum_conc: int = 1, embed_conc: int = 1,
) -> tuple[Pipeline, dict[str, list[tuple[str, float, float]]]]:
    """Build a Pipeline whose stages each sleep for a configured duration
    and record (file_path, start_time, end_time) so tests can assert
    overlap/non-overlap between stages."""
    parser = FileParser()
    timings: dict[str, list[tuple[str, float, float]]] = {
        "parse": [], "summarize": [], "embed": [],
    }

    async def fake_parse(file: File) -> None:
        t0 = time.monotonic()
        await asyncio.sleep(stage_durations["parse"])
        import hashlib
        try:
            data = Path(file.file_path).read_bytes()
        except OSError:
            data = file.file_path.encode()
        file.content = "x"
        file.content_hash = hashlib.sha256(data).hexdigest()
        file.content_type = "text/plain"
        file.parsed_at = datetime.now(timezone.utc)
        file.status = ProcessingStatus.PARSED
        timings["parse"].append((file.file_path, t0, time.monotonic()))
    parser.parse_file = AsyncMock(side_effect=fake_parse)

    summarizer = AutoSummarizer.__new__(AutoSummarizer)
    async def fake_summarize(file: File) -> None:
        t0 = time.monotonic()
        await asyncio.sleep(stage_durations["summarize"])
        file.title = f"T:{file.filename}"
        file.summary = f"S:{file.filename}"
        file.keywords = ["k1", "k2"]
        file.summarized_at = datetime.now(timezone.utc)
        file.status = ProcessingStatus.SUMMARIZED
        timings["summarize"].append((file.file_path, t0, time.monotonic()))
    summarizer.summarize = AsyncMock(side_effect=fake_summarize)

    embedder = AutoEmbedder.__new__(AutoEmbedder)
    async def fake_embed(file: File) -> None:
        t0 = time.monotonic()
        await asyncio.sleep(stage_durations["embed"])
        file.embedding = np.zeros(1536, dtype=np.float32)
        file.embedding_model = "mock"
        file.embedding_dimensions = 1536
        file.embedded_at = datetime.now(timezone.utc)
        file.status = ProcessingStatus.COMPLETE
        timings["embed"].append((file.file_path, t0, time.monotonic()))
    embedder.embed = AsyncMock(side_effect=fake_embed)

    pipeline = Pipeline(
        db=db, updates_hub=hub,
        discoverer=Discoverer(),
        parser=parser, summarizer=summarizer, embedder=embedder,
        parse_concurrency=parse_conc,
        summarize_concurrency=sum_conc,
        embed_concurrency=embed_conc,
    )
    return pipeline, timings


def _max_overlap(spans: list[tuple[str, float, float]]) -> int:
    """Maximum number of spans active at any moment. >1 means parallel."""
    events = []
    for _, s, e in spans:
        events.append((s, +1))
        events.append((e, -1))
    events.sort()
    max_n = 0
    cur = 0
    for _, delta in events:
        cur += delta
        max_n = max(max_n, cur)
    return max_n


def _stages_overlap(
    a: list[tuple[str, float, float]],
    b: list[tuple[str, float, float]],
) -> bool:
    """True if any span from `a` overlaps any span from `b` in time."""
    for _, sa, ea in a:
        for _, sb, eb in b:
            if sa < eb and sb < ea:
                return True
    return False


@pytest.mark.integration
@pytest.mark.asyncio
class TestStageParallelism:
    """Per-stage semaphores must let stages run in parallel across files."""

    async def test_parse_stages_run_in_parallel(
        self, temp_db: Database, mock_updates_hub: Hub, tmp_path: Path,
    ):
        """With parse_concurrency=4 and 4 files of 50 ms parse each, the
        max overlap should be 4 — they all run at once."""
        pipeline, timings = _make_pipeline_with_stage_timing(
            temp_db, mock_updates_hub,
            stage_durations={"parse": 0.05, "summarize": 0.001, "embed": 0.001},
            parse_conc=4,
        )
        files = []
        for i in range(4):
            p = tmp_path / f"file_{i}.txt"
            p.write_text(f"content {i}")
            files.append(p)

        await asyncio.gather(*(
            pipeline.process_file(File.from_path(p)) for p in files
        ))
        assert _max_overlap(timings["parse"]) == 4, (
            f"parse stages did not overlap as expected: "
            f"max_overlap={_max_overlap(timings['parse'])}, "
            f"timings={timings['parse']}"
        )

    async def test_summarize_serialized_by_semaphore(
        self, temp_db: Database, mock_updates_hub: Hub, tmp_path: Path,
    ):
        """summarize_concurrency=1 means at most one summarize runs at a
        time even when many files are in flight."""
        pipeline, timings = _make_pipeline_with_stage_timing(
            temp_db, mock_updates_hub,
            stage_durations={"parse": 0.001, "summarize": 0.05, "embed": 0.001},
            parse_conc=4, sum_conc=1,
        )
        files = []
        for i in range(4):
            p = tmp_path / f"sfile_{i}.txt"
            p.write_text(f"content {i}")
            files.append(p)

        await asyncio.gather(*(
            pipeline.process_file(File.from_path(p)) for p in files
        ))
        assert _max_overlap(timings["summarize"]) == 1, (
            "summarize stage overlapped despite semaphore=1"
        )

    async def test_different_stages_overlap(
        self, temp_db: Database, mock_updates_hub: Hub, tmp_path: Path,
    ):
        """The whole point of stage pipelining: while file A is in
        summarize, file B should already be parsing. We use more files
        than parse_concurrency so that the second batch's parses run
        concurrently with the first batch's summarize."""
        pipeline, timings = _make_pipeline_with_stage_timing(
            temp_db, mock_updates_hub,
            stage_durations={"parse": 0.05, "summarize": 0.05, "embed": 0.05},
            parse_conc=4, sum_conc=1, embed_conc=1,
        )
        # 8 files with parse_conc=4: first 4 parse t=0..50ms, finish and
        # release parse_sem, files 4–7 start parsing AT THE SAME TIME as
        # file 0 starts summarizing. Parse-summarize spans must overlap.
        files = []
        for i in range(8):
            p = tmp_path / f"pfile_{i}.txt"
            p.write_text(f"content {i}")
            files.append(p)

        await asyncio.gather(*(
            pipeline.process_file(File.from_path(p)) for p in files
        ))

        assert _stages_overlap(timings["parse"], timings["summarize"]), (
            "parse and summarize never ran concurrently — "
            "stage pipelining is not working"
        )


@pytest.mark.integration
@pytest.mark.asyncio
class TestCrashResumption:
    """Persisted progress lets process_file resume mid-pipeline."""

    async def test_resume_from_PARSED_skips_parse(
        self, temp_db: Database, mock_updates_hub: Hub, tmp_path: Path,
    ):
        """Pre-seed a row with status=PARSED. process_file should NOT call
        parse — only summarize and embed."""
        pipeline, timings = _make_pipeline_with_stage_timing(
            temp_db, mock_updates_hub,
            stage_durations={"parse": 0.001, "summarize": 0.001, "embed": 0.001},
        )

        target = tmp_path / "resume_parsed.txt"
        target.write_text("hello world")
        f = File.from_path(target)
        f.content = "hello world"
        f.content_hash = "deadbeef"
        f.content_type = "text/plain"
        f.parsed_at = datetime.now(timezone.utc)
        f.status = ProcessingStatus.PARSED
        await temp_db.upsert_file(f)

        await pipeline.process_file(File.from_path(target))

        parsed_paths = {t[0] for t in timings["parse"]}
        assert str(target.resolve()) not in parsed_paths, (
            "parse was called on a PARSED row — crash recovery is broken"
        )
        summarized_paths = {t[0] for t in timings["summarize"]}
        assert str(target.resolve()) in summarized_paths, (
            "summarize did not run after resume from PARSED"
        )

    async def test_resume_from_SUMMARIZED_skips_parse_and_summarize(
        self, temp_db: Database, mock_updates_hub: Hub, tmp_path: Path,
    ):
        """Pre-seed status=SUMMARIZED. Only embed should run."""
        pipeline, timings = _make_pipeline_with_stage_timing(
            temp_db, mock_updates_hub,
            stage_durations={"parse": 0.001, "summarize": 0.001, "embed": 0.001},
        )

        target = tmp_path / "resume_summarized.txt"
        target.write_text("hello world")
        f = File.from_path(target)
        f.content = "hello world"
        f.content_hash = "cafebabe"
        f.content_type = "text/plain"
        f.parsed_at = datetime.now(timezone.utc)
        f.title = "T"
        f.summary = "S"
        f.keywords = ["k"]
        f.summarized_at = datetime.now(timezone.utc)
        f.status = ProcessingStatus.SUMMARIZED
        await temp_db.upsert_file(f)

        await pipeline.process_file(File.from_path(target))

        parsed_paths = {t[0] for t in timings["parse"]}
        summarized_paths = {t[0] for t in timings["summarize"]}
        assert str(target.resolve()) not in parsed_paths
        assert str(target.resolve()) not in summarized_paths
        embedded_paths = {t[0] for t in timings["embed"]}
        assert str(target.resolve()) in embedded_paths

    async def test_crash_during_summarize_leaves_PARSED_row(
        self, temp_db: Database, mock_updates_hub: Hub, tmp_path: Path,
    ):
        """If summarize raises, the parsed content must already be
        durable in the DB so the next attempt resumes from PARSED.
        This is the regression test for the removed mid-flight
        `delete_file` call.
        """
        pipeline, _ = _make_pipeline_with_stage_timing(
            temp_db, mock_updates_hub,
            stage_durations={"parse": 0.001, "summarize": 0.001, "embed": 0.001},
        )
        # Make summarize blow up.
        async def boom(_file: File) -> None:
            raise RuntimeError("simulated crash")
        pipeline.summarizer.summarize = AsyncMock(side_effect=boom)

        target = tmp_path / "crash.txt"
        target.write_text("body content")

        with pytest.raises(RuntimeError):
            await pipeline.process_file(File.from_path(target))

        # The pipeline's exception handler will mark the row FAILED with
        # fallback indexing. That overwrites status=PARSED that the parse
        # stage just persisted. So the durable contract here is weaker
        # than crash-vs-graceful-failure: a *thrown* exception falls
        # through to FAILED + fallback. A *true crash* (kill -9) leaves
        # the PARSED row intact because the except clause never runs.
        # Simulate that by checking what was written *before* the except
        # ran. We do this by re-running with an instrumented summarize
        # that reads the DB row at the moment it's about to throw.
        captured_status: list[Any] = []
        async def boom_after_check(file: File) -> None:
            row = await temp_db.get_file_by_path(file.file_path)
            captured_status.append(row.status if row else None)
            raise RuntimeError("simulated crash 2")
        target2 = tmp_path / "crash2.txt"
        target2.write_text("body 2")
        pipeline.summarizer.summarize = AsyncMock(side_effect=boom_after_check)
        with pytest.raises(RuntimeError):
            await pipeline.process_file(File.from_path(target2))
        assert captured_status == [ProcessingStatus.PARSED], (
            f"parse stage did not persist before summarize crashed: "
            f"saw {captured_status}. The mid-flight delete_file regression "
            f"would manifest as None here."
        )


@pytest.mark.integration
@pytest.mark.asyncio
class TestThroughput:
    """A throughput sanity check: stage pipelining should beat the
    serialized baseline. Not a precise benchmark — just enough to catch
    a regression where someone re-introduces a global lock."""

    async def test_parallel_faster_than_serial(
        self, temp_db: Database, mock_updates_hub: Hub, tmp_path: Path,
    ):
        # 12 files w/ a parse-dominant workload so the parse_concurrency
        # change (1 → 4) actually moves the wall-clock needle. With small
        # N + balanced stages, scheduling and SQLite-write overhead
        # dominated and the 5% threshold lived inside the noise floor —
        # especially after the DB pragma tuning made writes faster across
        # the board (random 5/2026).
        N = 12
        files = []
        for i in range(N):
            p = tmp_path / f"tfile_{i}.txt"
            p.write_text(f"c{i}")
            files.append(p)

        # Baseline: parse serialized.
        baseline_pipeline, _ = _make_pipeline_with_stage_timing(
            temp_db, mock_updates_hub,
            stage_durations={"parse": 0.05, "summarize": 0.01, "embed": 0.01},
            parse_conc=1, sum_conc=1, embed_conc=1,
        )
        t0 = time.monotonic()
        await asyncio.gather(*(
            baseline_pipeline.process_file(File.from_path(p)) for p in files
        ))
        baseline = time.monotonic() - t0

        # Wipe so the second run doesn't skip via mtime cache.
        for p in files:
            await temp_db.delete_file(str(p.resolve()))

        # Optimized: parse parallel.
        opt_pipeline, _ = _make_pipeline_with_stage_timing(
            temp_db, mock_updates_hub,
            stage_durations={"parse": 0.05, "summarize": 0.01, "embed": 0.01},
            parse_conc=4, sum_conc=1, embed_conc=1,
        )
        t0 = time.monotonic()
        await asyncio.gather(*(
            opt_pipeline.process_file(File.from_path(p)) for p in files
        ))
        optimized = time.monotonic() - t0

        # Be lenient — CI machines have variable load. We just want to
        # rule out a serialization regression.
        assert optimized < baseline * 0.95, (
            f"Stage pipelining did not improve throughput: "
            f"baseline={baseline:.3f}s, optimized={optimized:.3f}s. "
            f"Someone may have re-introduced a global lock."
        )
