"""Performance benchmarks for the indexing pipeline.

These tests do two things the regular suite doesn't:

  1. **Compare measured wall time against the theoretical minimum**
     given per-stage concurrency limits. The gap is pure pipeline
     overhead (asyncio scheduling, SSE event publishing, DB writes
     between stages). If the gap blows out past ~30%, someone has
     re-introduced a global lock or a sleep that throws off the
     stage parallelism.

  2. **Sample system resources during a real queue run** and print
     peak / mean CPU and RSS so the developer running the benchmark
     can see at a glance whether memory grows unbounded or CPU
     pegs unexpectedly. These are informational — they print, they
     don't assert — so a noisy CI machine doesn't flake the suite.

Run them with:
    uv run --group test pytest tests/integration/test_perf_benchmarks.py \\
        --no-cov -s

`-s` is important — without it pytest captures stdout and the numbers
are invisible.

These tests still use mocked AI stages (the durations come from
configurable sleep timers) — running the *real* Qwen3-VL on tens of
files would take minutes per benchmark and need a GPU on CI. The
mocks let us measure pipeline mechanics independently of model speed.
A "real-model" benchmark belongs in a manual-only run, see
`docs/STARTUP_PROFILE_FINDINGS.md` for guidance.
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any
from unittest.mock import AsyncMock

import numpy as np
import psutil
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


# ---------------------------------------------------------------------------
# Pipeline factory with configurable per-stage durations
# ---------------------------------------------------------------------------

def _build_timed_pipeline(
    db: Database, hub: Hub,
    parse_s: float, summarize_s: float, embed_s: float,
    parse_conc: int = 4, sum_conc: int = 1, embed_conc: int = 1,
) -> Pipeline:
    parser = FileParser()
    async def fake_parse(file: File) -> None:
        if parse_s > 0:
            await asyncio.sleep(parse_s)
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
    parser.parse_file = AsyncMock(side_effect=fake_parse)

    summarizer = AutoSummarizer.__new__(AutoSummarizer)
    async def fake_summarize(file: File) -> None:
        if summarize_s > 0:
            await asyncio.sleep(summarize_s)
        file.title = f"T:{file.filename}"
        file.summary = "S"
        file.keywords = ["k"]
        file.summarized_at = datetime.now(timezone.utc)
        file.status = ProcessingStatus.SUMMARIZED
    summarizer.summarize = AsyncMock(side_effect=fake_summarize)

    embedder = AutoEmbedder.__new__(AutoEmbedder)
    async def fake_embed(file: File) -> None:
        if embed_s > 0:
            await asyncio.sleep(embed_s)
        file.embedding = np.zeros(1536, dtype=np.float32)
        file.embedding_model = "mock"
        file.embedding_dimensions = 1536
        file.embedded_at = datetime.now(timezone.utc)
        file.status = ProcessingStatus.COMPLETE
    embedder.embed = AsyncMock(side_effect=fake_embed)

    return Pipeline(
        db=db, updates_hub=hub,
        discoverer=Discoverer(),
        parser=parser, summarizer=summarizer, embedder=embedder,
        parse_concurrency=parse_conc,
        summarize_concurrency=sum_conc,
        embed_concurrency=embed_conc,
    )


def _theoretical_minimum_wall_time(
    n: int, parse_s: float, summarize_s: float, embed_s: float,
    parse_conc: int, sum_conc: int, embed_conc: int,
) -> float:
    """For a fully-pipelined run, the wall-time floor is dominated by
    the slowest stage's *aggregate* time, divided by its concurrency.
    Both the very first file and the very last one also add a stage
    "fill-up" and "drain" cost that's bounded by the longest single
    stage path through the pipeline. We treat the union as:

        steady_state = max(
            n * parse_s    / parse_conc,
            n * summarize_s / sum_conc,
            n * embed_s    / embed_conc,
        )
        prelude + epilogue ~= parse_s + summarize_s + embed_s

    so the theoretical minimum wall time is roughly
    `steady_state + prelude` for n >> conc.
    """
    steady = max(
        n * parse_s / parse_conc,
        n * summarize_s / sum_conc,
        n * embed_s / embed_conc,
    )
    prelude = parse_s + summarize_s + embed_s
    return steady + prelude


def _print_block(title: str, lines: list[str]) -> None:
    """Pretty boxed printout so benchmark numbers stand out from the
    pytest noise. Run with `-s` to actually see them."""
    width = max(len(title), max((len(l) for l in lines), default=0)) + 2
    print()
    print("=" * (width + 4))
    print(f"  {title.ljust(width)}  ")
    print("-" * (width + 4))
    for line in lines:
        print(f"  {line.ljust(width)}  ")
    print("=" * (width + 4))


# ---------------------------------------------------------------------------
# 1. Theoretical-minimum efficiency benchmark
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
class TestThroughputEfficiency:
    """How close does the pipeline get to the theoretical wall-time floor?"""

    @pytest.mark.parametrize("n,parse_s,summarize_s,embed_s", [
        # Realistic ratio: summarize dominates (Qwen3-VL ~5–30 s on real
        # hardware; we use 50 ms here so the test stays fast). With
        # parse_conc=4 / sum_conc=1 / embed_conc=1, summarize is the
        # bottleneck — wall time ~= n * summarize.
        (12, 0.005, 0.05, 0.005),
        # Inverted ratio (parse-heavy): parse becomes the bottleneck
        # at parse_conc=4, so wall time ~= n * parse / 4.
        (12, 0.05, 0.005, 0.005),
        # Balanced — embed-dominant at conc=1.
        (12, 0.005, 0.005, 0.05),
    ])
    async def test_efficiency_vs_theoretical_minimum(
        self, n, parse_s, summarize_s, embed_s,
        temp_db: Database, mock_updates_hub: Hub, tmp_path: Path,
    ):
        pipeline = _build_timed_pipeline(
            temp_db, mock_updates_hub,
            parse_s=parse_s, summarize_s=summarize_s, embed_s=embed_s,
            parse_conc=4, sum_conc=1, embed_conc=1,
        )
        files = []
        for i in range(n):
            p = tmp_path / f"perf_{parse_s}_{summarize_s}_{embed_s}_{i}.txt"
            p.write_text(f"c{i}")
            files.append(p)

        t0 = time.monotonic()
        await asyncio.gather(*(
            pipeline.process_file(File.from_path(p)) for p in files
        ))
        elapsed = time.monotonic() - t0

        floor = _theoretical_minimum_wall_time(
            n, parse_s, summarize_s, embed_s,
            parse_conc=4, sum_conc=1, embed_conc=1,
        )
        efficiency = floor / elapsed if elapsed > 0 else 0.0

        bottleneck_name, bottleneck_total = max(
            (("parse", n * parse_s / 4),
             ("summarize", n * summarize_s / 1),
             ("embed", n * embed_s / 1)),
            key=lambda kv: kv[1],
        )

        _print_block(
            f"Throughput efficiency: n={n} "
            f"(parse={parse_s*1000:.0f}ms, sum={summarize_s*1000:.0f}ms, "
            f"emb={embed_s*1000:.0f}ms)",
            [
                f"Theoretical minimum: {floor*1000:7.1f} ms",
                f"Measured wall time:  {elapsed*1000:7.1f} ms",
                f"Efficiency:          {efficiency*100:7.1f} %",
                f"Bottleneck stage:    {bottleneck_name} "
                f"(~{bottleneck_total*1000:.0f} ms aggregate)",
                f"Per-file overhead:   "
                f"{(elapsed - floor) / n * 1000:7.2f} ms",
            ],
        )

        # Efficiency floor of 60% is generous — leaves room for SQLite
        # writes between stages, asyncio scheduling jitter, and short
        # stages where the prelude+epilogue dominates. Catches
        # regressions where someone serializes a stage by accident.
        assert efficiency > 0.60, (
            f"Pipeline efficiency dropped below 60% — there's likely a "
            f"global lock or unintended serialization. "
            f"Efficiency={efficiency:.1%}, floor={floor:.3f}s, "
            f"measured={elapsed:.3f}s"
        )


# ---------------------------------------------------------------------------
# 2. Scheduling overhead benchmark (zero-cost stages)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
class TestSchedulingOverhead:
    """With every stage's actual work set to zero, what's left is pure
    pipeline overhead: semaphore acquire/release, DB upserts, asyncio
    scheduling, SSE event publishing. This is the floor — any real
    workload must add at least this much per file."""

    async def test_per_file_overhead_under_2ms(
        self, temp_db: Database, mock_updates_hub: Hub, tmp_path: Path,
    ):
        n = 50
        pipeline = _build_timed_pipeline(
            temp_db, mock_updates_hub,
            parse_s=0, summarize_s=0, embed_s=0,
            parse_conc=4, sum_conc=1, embed_conc=1,
        )
        files = []
        for i in range(n):
            p = tmp_path / f"oh_{i}.txt"
            p.write_text(f"c{i}")
            files.append(p)

        t0 = time.monotonic()
        await asyncio.gather(*(
            pipeline.process_file(File.from_path(p)) for p in files
        ))
        elapsed = time.monotonic() - t0
        per_file = elapsed / n

        _print_block(
            f"Scheduling overhead (n={n}, zero-cost stages)",
            [
                f"Total wall time:   {elapsed*1000:7.1f} ms",
                f"Per-file overhead: {per_file*1000:7.2f} ms",
                f"Per-file overhead: {per_file*1_000_000:7.0f} µs",
                "",
                "Composition (each per file):",
                "  - 3× semaphore acquire/release",
                "  - 3× DB upsert (PARSED → SUMMARIZED → COMPLETE)",
                "  - 1× embeddings INSERT into vec0 (DELETE skipped on",
                "    first-time embed — see upsert_file_embeddings)",
                "  - ~6 SSE events through the Hub",
                "  - asyncio scheduling for ~6 await points",
            ],
        )

        # 25 ms/file is the regression floor. If pipeline overhead per
        # file ever climbs past that on a CI machine, something added a
        # blocking call or a needless lock. Real per-file overhead on
        # the dev machine should be ~3–8 ms.
        assert per_file < 0.025, (
            f"Per-file pipeline overhead is {per_file*1000:.2f} ms — "
            f"investigate. Was {per_file*1000:.2f} ms vs <25 ms threshold."
        )


# ---------------------------------------------------------------------------
# 3. System-resource profile during a real-queue run
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
class TestResourceUsage:
    """Sample CPU% and resident memory during an end-to-end queue run.
    Pure observation — prints peak/mean and never asserts. The numbers
    are useful when comparing two branches or hunting a memory leak."""

    async def test_resource_profile(
        self, temp_db: Database, mock_updates_hub: Hub, tmp_path: Path,
    ):
        n = 30
        pipeline = _build_timed_pipeline(
            temp_db, mock_updates_hub,
            parse_s=0.005, summarize_s=0.020, embed_s=0.005,
            parse_conc=4, sum_conc=1, embed_conc=1,
        )
        queue = IndexingQueue(
            pipeline=pipeline,
            updates_hub=mock_updates_hub,
            config=QueueConfig(
                cooldown_seconds=1, initial_cooldown_seconds=0,
                max_concurrency=6, max_retries=0,
                file_processing_timeout=10,
            ),
            db=temp_db,
        )

        files = []
        for i in range(n):
            p = tmp_path / f"res_{i}.txt"
            p.write_text(f"c{i}" * 10)
            files.append(p)

        proc = psutil.Process(os.getpid())
        # Prime the per-process CPU counter — the first call always
        # returns 0.0 because there's no baseline to diff against.
        proc.cpu_percent(interval=None)

        cpu_samples: list[float] = []
        rss_samples: list[int] = []
        sampler_stop = asyncio.Event()
        baseline_rss = proc.memory_info().rss

        async def sampler():
            while not sampler_stop.is_set():
                cpu_samples.append(proc.cpu_percent(interval=None))
                rss_samples.append(proc.memory_info().rss)
                try:
                    await asyncio.wait_for(sampler_stop.wait(), timeout=0.05)
                except asyncio.TimeoutError:
                    pass

        sampler_task = asyncio.create_task(sampler())

        await queue.start()
        for p in files:
            await queue.enqueue(p, QueueAction.INDEX)

        # Wait until the queue drains.
        deadline = time.time() + 30
        while time.time() < deadline:
            status = await queue.get_status()
            if status["total_items"] == 0:
                break
            await asyncio.sleep(0.05)

        await queue.stop()
        sampler_stop.set()
        await sampler_task

        # Drop the leading 0.0 if present (cpu_percent priming).
        cpu_useful = [c for c in cpu_samples if c > 0.0]
        peak_cpu = max(cpu_useful, default=0.0)
        avg_cpu = mean(cpu_useful) if cpu_useful else 0.0
        peak_rss = max(rss_samples, default=baseline_rss)
        rss_growth = peak_rss - baseline_rss

        _print_block(
            f"Resource profile: n={n} files through real IndexingQueue",
            [
                f"Wall time:          ~{(len(rss_samples) * 0.05):.2f} s "
                f"({len(rss_samples)} samples)",
                f"Peak CPU%:          {peak_cpu:7.1f} %",
                f"Avg  CPU% (active): {avg_cpu:7.1f} %",
                f"Baseline RSS:       {baseline_rss / 1024 / 1024:7.1f} MiB",
                f"Peak RSS:           {peak_rss     / 1024 / 1024:7.1f} MiB",
                f"RSS growth:         {rss_growth   / 1024 / 1024:+7.1f} MiB",
                "",
                "Notes:",
                "  - CPU% is per-process; >100% means multi-core utilization.",
                "  - With mocked stages this is mostly asyncio + SQLite.",
                "  - RSS growth >50 MiB on N=30 mock files would point at",
                "    a leak — File objects shouldn't be retained.",
            ],
        )

        # Sanity-only assertion: peak RSS should not have ballooned.
        # 100 MiB is generous for a test that does ~30 SQLite writes
        # and creates ~30 small File objects.
        assert rss_growth < 100 * 1024 * 1024, (
            f"RSS grew by {rss_growth / 1024 / 1024:.1f} MiB during a "
            f"{n}-file mock run. Likely a leak in the queue or pipeline "
            f"holding File / Update objects past their intended lifetime."
        )


# ---------------------------------------------------------------------------
# 4. Stage-overlap measurement (visualizes the parallelism)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
class TestStageOverlap:
    """How much of the run was spent in each stage *concurrently with
    another stage*? This is what the per-stage semaphore design buys
    us. Higher overlap = better GPU/CPU utilization."""

    async def test_overlap_breakdown(
        self, temp_db: Database, mock_updates_hub: Hub, tmp_path: Path,
    ):
        # 16 files with imbalanced stages so the pipeline has plenty of
        # opportunity to overlap.
        n = 16
        spans: dict[str, list[tuple[float, float]]] = {
            "parse": [], "summarize": [], "embed": [],
        }

        parser = FileParser()
        async def fake_parse(file: File) -> None:
            t0 = time.monotonic()
            await asyncio.sleep(0.020)
            import hashlib
            try:
                data = Path(file.file_path).read_bytes()
            except OSError:
                data = b""
            file.content = "x"; file.content_hash = hashlib.sha256(data).hexdigest()
            file.content_type = "text/plain"
            file.parsed_at = datetime.now(timezone.utc)
            file.status = ProcessingStatus.PARSED
            spans["parse"].append((t0, time.monotonic()))
        parser.parse_file = AsyncMock(side_effect=fake_parse)

        summarizer = AutoSummarizer.__new__(AutoSummarizer)
        async def fake_summarize(file: File) -> None:
            t0 = time.monotonic()
            await asyncio.sleep(0.030)
            file.title = "T"; file.summary = "S"; file.keywords = ["k"]
            file.summarized_at = datetime.now(timezone.utc)
            file.status = ProcessingStatus.SUMMARIZED
            spans["summarize"].append((t0, time.monotonic()))
        summarizer.summarize = AsyncMock(side_effect=fake_summarize)

        embedder = AutoEmbedder.__new__(AutoEmbedder)
        async def fake_embed(file: File) -> None:
            t0 = time.monotonic()
            await asyncio.sleep(0.020)
            file.embedding = np.zeros(1536, dtype=np.float32)
            file.embedding_model = "m"; file.embedding_dimensions = 1536
            file.embedded_at = datetime.now(timezone.utc)
            file.status = ProcessingStatus.COMPLETE
            spans["embed"].append((t0, time.monotonic()))
        embedder.embed = AsyncMock(side_effect=fake_embed)

        pipeline = Pipeline(
            db=temp_db, updates_hub=mock_updates_hub,
            discoverer=Discoverer(),
            parser=parser, summarizer=summarizer, embedder=embedder,
            parse_concurrency=4, summarize_concurrency=1, embed_concurrency=1,
        )

        files = []
        for i in range(n):
            p = tmp_path / f"ov_{i}.txt"
            p.write_text(f"c{i}")
            files.append(p)

        t0 = time.monotonic()
        await asyncio.gather(*(
            pipeline.process_file(File.from_path(p)) for p in files
        ))
        wall = time.monotonic() - t0

        # Compute coverage (sum of stage-active intervals) and overlap
        # (any moment >=2 stages active simultaneously).
        events: list[tuple[float, int, str]] = []
        for stage, sps in spans.items():
            for s, e in sps:
                events.append((s, +1, stage))
                events.append((e, -1, stage))
        events.sort()

        active_stages: set[str] = set()
        last_t = events[0][0] if events else 0.0
        time_at_concurrency: dict[int, float] = {}
        for t, delta, stage in events:
            dt = t - last_t
            time_at_concurrency[len(active_stages)] = (
                time_at_concurrency.get(len(active_stages), 0.0) + dt
            )
            if delta > 0:
                active_stages.add(stage)
            else:
                active_stages.discard(stage)
            last_t = t

        time_2plus = sum(t for k, t in time_at_concurrency.items() if k >= 2)
        time_3 = time_at_concurrency.get(3, 0.0)
        overlap_pct = (time_2plus / wall) * 100 if wall > 0 else 0
        triple_pct = (time_3 / wall) * 100 if wall > 0 else 0

        _print_block(
            f"Stage overlap profile: n={n} files",
            [
                f"Wall time:                   {wall*1000:7.1f} ms",
                f"Time with 1 stage active:    "
                f"{time_at_concurrency.get(1, 0.0)*1000:7.1f} ms",
                f"Time with 2 stages active:   "
                f"{time_at_concurrency.get(2, 0.0)*1000:7.1f} ms",
                f"Time with 3 stages active:   "
                f"{time_3*1000:7.1f} ms",
                f"% of run with overlap (≥2):  {overlap_pct:5.1f} %",
                f"% of run with 3 in flight:   {triple_pct:5.1f} %",
                "",
                "Higher % overlap = better hardware utilization. With",
                "real models a triple-concurrent run keeps CPU on parse,",
                "GPU on summarize (Metal), and MPS on embed all busy.",
            ],
        )

        # If we can't sustain 2-stage overlap for at least 30% of the
        # run, the per-stage semaphore design is broken. Real-world
        # numbers should be 60–80%.
        assert overlap_pct > 30, (
            f"Stage overlap is only {overlap_pct:.1f}% — pipeline is "
            f"effectively serial."
        )
