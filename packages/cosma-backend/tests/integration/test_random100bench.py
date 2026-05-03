"""random100bench — full-pipeline reproducible benchmark.

Picks N random files from ~/Downloads (configurable), copies them
into an isolated temp tree, runs the real cosma_backend pipeline
against an isolated SQLite database, fires 10 search queries
equally spaced through the indexing window, samples CPU/RSS
throughout, and writes a Markdown + JSON report under
`benchmarks/random100bench/reports/<timestamp>/`.

Reproducibility: the chosen file list is hashed and saved to
`benchmarks/random100bench/manifests/<seed>.json`. Re-running with the
same seed picks the same files as long as they still exist on disk
(missing files are dropped + a warning is printed; if too many are
gone the benchmark refuses to run).

What's real, what's mocked:

  - REAL: Pipeline, IndexingQueue, Watcher, Discoverer, FileParser
    (MarkItDown), Database (SQLite + FTS5 + vec0), Searcher
    (HybridSearcher), filter manager, settings, all the queue's
    debounce/scheduling/preempt logic.
  - MOCKED: AutoSummarizer.summarize (returns title/summary/keywords
    derived from the filename), AutoEmbedder.embed (returns a
    deterministic random vector).
    Reason: real Qwen3-VL summarize is 5–30 s per file, so 100 files
    is 30+ minutes per run. The benchmark is here to catch
    *pipeline* bugs (queue, parser, DB, search wire), not to
    benchmark the LLM. Set COSMA_BENCH_REAL=1 to force real models.

Run:
    uv run --group test pytest \\
        tests/integration/test_random100bench.py \\
        -m benchmark --no-cov -s

By default this benchmark is OFF — the `benchmark` mark must be
selected explicitly. CI does not run it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import random
import shutil
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import numpy as np
import psutil
import pytest
import pytest_asyncio

import cosma_backend
from cosma_backend.db.database import Database, EMBEDDING_STORAGE_DIMENSIONS
from cosma_backend.discoverer import Discoverer
from cosma_backend.embedder import AutoEmbedder
from cosma_backend.filter import FilterConfigManager
from cosma_backend.models.file import File
from cosma_backend.models.status import ProcessingStatus
from cosma_backend.parser import FileParser
from cosma_backend.pipeline import Pipeline
from cosma_backend.queue import IndexingQueue, QueueAction
from cosma_backend.searcher import HybridSearcher
from cosma_backend.settings import QueueConfig
from cosma_backend.summarizer import AutoSummarizer
from cosma_backend.utils.pubsub import Hub


# ---------------------------------------------------------------------------
# Configuration knobs (env-overridable)
# ---------------------------------------------------------------------------

DEFAULT_SOURCE_DIR = Path.home() / "Downloads"
DEFAULT_N = 100
DEFAULT_SEED = 42
DEFAULT_QUERIES = 10
# Skip files larger than this (avoids the benchmark spending all its
# time parsing one giant ISO).
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MiB

# We deliberately exclude media (audio + video) extensions from the
# corpus even though the parser supports them, because real audio/video
# parsing requires whisper.cpp + a vision LLM that the test environment
# may not have bootstrapped. The benchmark would otherwise hang for
# 30+ minutes on a 5-minute mp4 trying to pull whisper at runtime. To
# benchmark media specifically, set COSMA_BENCH_INCLUDE_MEDIA=1.
MEDIA_EXTS = {".mp3", ".wav", ".aac", ".mp4", ".mov", ".mkv", ".avi"}

# Queries we'll fire during indexing. Generic enough to land hits on
# almost any download corpus.
DEFAULT_QUERY_BANK = [
    "report",
    "invoice",
    "screenshot",
    "data",
    "config",
    "image",
    "document",
    "code",
    "log",
    "summary",
    "presentation",
    "video",
    "music",
    "spreadsheet",
    "archive",
]

# Where the benchmark stores its manifests + reports.
BENCH_ROOT = (
    Path(__file__).resolve().parent.parent.parent
    / "benchmarks" / "random100bench"
)
MANIFEST_DIR = BENCH_ROOT / "manifests"
REPORTS_DIR = BENCH_ROOT / "reports"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _system_info() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cosma_backend_version": getattr(
            cosma_backend, "__version__", "unknown",
        ),
        "cosma_backend_api_version": getattr(
            cosma_backend, "__api_version__", "unknown",
        ),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_freq_mhz": (
            psutil.cpu_freq().max if psutil.cpu_freq() else None
        ),
        "ram_total_gib": round(vm.total / 1024 ** 3, 2),
        "ram_available_gib": round(vm.available / 1024 ** 3, 2),
    }


def _select_random_files(
    source: Path, n: int, seed: int,
    supported_exts: set[str],
) -> tuple[list[Path], dict[str, int]]:
    """Walk `source` once, filter to supported + size-bounded files,
    randomly sample `n` with the given seed.

    Returns (sampled_paths, stats_dict).
    """
    candidates: list[Path] = []
    stats = {
        "examined": 0, "unsupported": 0, "too_large": 0,
        "unreadable": 0, "supported": 0,
    }
    for root, dirs, files in os.walk(source):
        # Skip hidden directories — most macOS Downloads bookkeeping
        # lives in `.DS_Store` and similar; not interesting.
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fname in files:
            if fname.startswith("."):
                continue
            stats["examined"] += 1
            p = Path(root) / fname
            ext = p.suffix.lower()
            if ext not in supported_exts:
                stats["unsupported"] += 1
                continue
            try:
                size = p.stat().st_size
            except OSError:
                stats["unreadable"] += 1
                continue
            if size > MAX_FILE_SIZE:
                stats["too_large"] += 1
                continue
            stats["supported"] += 1
            candidates.append(p)
    rng = random.Random(seed)
    sampled = rng.sample(candidates, min(n, len(candidates)))
    return sampled, stats


def _load_or_create_manifest(
    seed: int, source: Path, n: int, supported_exts: set[str],
) -> tuple[list[Path], dict[str, Any]]:
    """Return the seeded random sample, persisting the path list to a
    manifest so the same seed reproduces the same files on later runs
    (when they still exist)."""
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = MANIFEST_DIR / f"seed_{seed}.json"

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        cached_paths = [Path(p) for p in manifest["paths"]]
        existing = [p for p in cached_paths if p.exists()]
        missing_n = len(cached_paths) - len(existing)
        manifest["existing_at_run"] = len(existing)
        manifest["missing_at_run"] = missing_n
        if len(existing) >= max(1, int(0.5 * len(cached_paths))):
            return existing, manifest
        # Too many gone; rebuild.
        print(
            f"[random100bench] manifest seed={seed}: "
            f"{missing_n}/{len(cached_paths)} files vanished — rebuilding"
        )

    sampled, stats = _select_random_files(source, n, seed, supported_exts)
    manifest = {
        "seed": seed,
        "n_requested": n,
        "n_actual": len(sampled),
        "source": str(source),
        "selection_stats": stats,
        "paths": [str(p) for p in sampled],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return sampled, manifest


def _copy_corpus(sources: list[Path], dest: Path) -> list[Path]:
    """Flatten the source files into `dest/<ix>_<basename>` to avoid
    name collisions across different source directories. Returns the
    new paths inside `dest`."""
    dest.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for i, src in enumerate(sources):
        # Keep extension (parser uses it). Prefix with ordinal so two
        # files with the same name from different folders don't clash.
        target = dest / f"{i:03d}_{src.name}"
        try:
            shutil.copy2(src, target)
            out.append(target)
        except OSError as e:
            print(f"[random100bench] copy failed {src}: {e}")
    return out


class _MockEmbedder:
    """Duck-typed stand-in for AutoEmbedder used by both the pipeline
    (for `embed`) and the searcher (for `embed_text_async`).

    Inheriting from AutoEmbedder via __new__ left attributes like
    `preferred_provider` unset and the searcher crashed reading them.
    A plain class with the methods the rest of the system actually
    calls is simpler and more honest."""

    preferred_provider = "local"
    model_name = "mock"
    embedding_dimensions = EMBEDDING_STORAGE_DIMENSIONS

    def __init__(self) -> None:
        self.embedded_files: int = 0

    def is_model_loaded(self) -> bool:
        return True

    @staticmethod
    def _vector_for(seed_text: str) -> np.ndarray:
        seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(EMBEDDING_STORAGE_DIMENSIONS).astype(np.float32)
        v /= max(1e-8, float(np.linalg.norm(v)))
        return v

    async def embed(self, file: File) -> None:
        v = self._vector_for(file.content_hash or file.file_path)
        file.embedding = v
        file.embedding_model = self.model_name
        file.embedding_dimensions = EMBEDDING_STORAGE_DIMENSIONS
        file.embedded_at = datetime.now(timezone.utc)
        file.status = ProcessingStatus.COMPLETE
        self.embedded_files += 1

    async def embed_text_async(self, text):
        if isinstance(text, list):
            return np.stack([self._vector_for(t) for t in text])
        return self._vector_for(text)


def _build_mocked_pipeline(
    db: Database, hub: Hub, *,
    parse_concurrency: int = 4,
    summarize_concurrency: int = 1,
    embed_concurrency: int = 1,
) -> tuple[Pipeline, dict[str, list[float]], _MockEmbedder]:
    """Real parser + DB; mocked summarizer + embedder. Records per-stage
    timings in the returned `stage_durations` dict."""
    from cosma_backend.settings import ParserConfig
    parser = FileParser(config=ParserConfig())

    durations: dict[str, list[float]] = {
        "parse": [], "summarize": [], "embed": [],
    }
    # Wrap the real parser to time each call.
    real_parse = parser.parse_file

    async def timed_parse(file: File) -> None:
        t0 = time.monotonic()
        try:
            await real_parse(file)
        finally:
            durations["parse"].append(time.monotonic() - t0)

    parser.parse_file = timed_parse  # type: ignore[assignment]

    summarizer = AutoSummarizer.__new__(AutoSummarizer)
    async def fake_summarize(file: File) -> None:
        t0 = time.monotonic()
        stem = Path(file.filename).stem
        file.title = stem.replace("_", " ").replace("-", " ")[:80] or "Untitled"
        file.summary = f"Mock summary of {file.filename}"
        file.keywords = [w.lower() for w in stem.split("_")[:5]] or [stem.lower()]
        file.summarized_at = datetime.now(timezone.utc)
        file.status = ProcessingStatus.SUMMARIZED
        durations["summarize"].append(time.monotonic() - t0)
    summarizer.summarize = AsyncMock(side_effect=fake_summarize)

    mock_embedder = _MockEmbedder()
    real_embed = mock_embedder.embed
    async def timed_embed(file: File) -> None:
        t0 = time.monotonic()
        try:
            await real_embed(file)
        finally:
            durations["embed"].append(time.monotonic() - t0)
    mock_embedder.embed = timed_embed  # type: ignore[assignment]

    pipeline = Pipeline(
        db=db, updates_hub=hub,
        discoverer=Discoverer(),
        parser=parser, summarizer=summarizer, embedder=mock_embedder,
        parse_concurrency=parse_concurrency,
        summarize_concurrency=summarize_concurrency,
        embed_concurrency=embed_concurrency,
    )
    return pipeline, durations, mock_embedder


async def _resource_sampler(
    proc: psutil.Process, stop: asyncio.Event,
    samples: dict[str, list[float]],
) -> None:
    """Sample CPU% and RSS every 100ms until `stop` fires."""
    proc.cpu_percent(interval=None)  # prime the counter
    while not stop.is_set():
        try:
            samples["cpu"].append(proc.cpu_percent(interval=None))
            samples["rss_mib"].append(
                proc.memory_info().rss / 1024 / 1024
            )
            samples["t"].append(time.monotonic())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.1)
        except asyncio.TimeoutError:
            pass


# ---------------------------------------------------------------------------
# The benchmark
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_random100bench(tmp_path: Path):
    """Drive the full pipeline against 100 random Downloads files."""
    n = int(os.environ.get("COSMA_BENCH_N", DEFAULT_N))
    seed = int(os.environ.get("COSMA_BENCH_SEED", DEFAULT_SEED))
    source = Path(os.environ.get(
        "COSMA_BENCH_SOURCE", str(DEFAULT_SOURCE_DIR),
    )).expanduser()
    n_queries = int(os.environ.get("COSMA_BENCH_QUERIES", DEFAULT_QUERIES))

    if not source.exists():
        pytest.skip(f"benchmark source dir does not exist: {source}")

    print(f"\n[random100bench] source={source} n={n} seed={seed} "
          f"queries={n_queries}")

    sysinfo = _system_info()
    print(f"[random100bench] system: {sysinfo['system']} "
          f"{sysinfo['machine']} | {sysinfo['cpu_count_logical']} logical "
          f"CPUs | {sysinfo['ram_total_gib']} GiB RAM | "
          f"cosma {sysinfo['cosma_backend_version']} "
          f"(api v{sysinfo['cosma_backend_api_version']}) | "
          f"Python {sysinfo['python']}")

    parser_for_exts = FileParser()
    supported_exts = set(parser_for_exts.get_supported_extensions())
    if not os.environ.get("COSMA_BENCH_INCLUDE_MEDIA"):
        supported_exts -= MEDIA_EXTS
        print(f"[random100bench] excluding media extensions: "
              f"{sorted(MEDIA_EXTS)} (set COSMA_BENCH_INCLUDE_MEDIA=1 to include)")

    sampled, manifest = _load_or_create_manifest(seed, source, n, supported_exts)
    if not sampled:
        pytest.skip(f"no supported files found under {source}")
    print(f"[random100bench] selected {len(sampled)} files "
          f"(supported={manifest.get('selection_stats', {}).get('supported','?')}, "
          f"too_large={manifest.get('selection_stats', {}).get('too_large','?')})")

    # Copy into an isolated test corpus directory.
    corpus_dir = tmp_path / "corpus"
    copied = _copy_corpus(sampled, corpus_dir)
    print(f"[random100bench] copied {len(copied)} files into {corpus_dir}")

    # The shared pipeline thread pool is built lazily on first use.
    # Configure it BEFORE the parser fires its first MarkItDown call,
    # otherwise we get the default 6 workers (still better than the
    # old 2, but the bench wants explicit control). Match what app.py
    # does at production startup.
    from cosma_backend.pipeline_executor import configure_pipeline_executor
    configure_pipeline_executor(max_workers=6)  # parse=4 + embed=1 + 1

    # Build the backend with isolated DB + hub.
    hub: Hub = Hub()
    db_path = tmp_path / "bench.db"
    db = await Database.from_path(str(db_path))

    pipeline, stage_durations, mock_embedder = _build_mocked_pipeline(db, hub)
    searcher = HybridSearcher(db=db, embedder=mock_embedder)

    # Indexing queue with realistic config. file_processing_timeout=30s
    # is tighter than production (300s) so a hung parse on one weird
    # Downloads file (a corrupt PDF, a "looks like a docx but isn't"
    # rename) doesn't stall the whole benchmark.
    #
    # search_preempt_seconds=1.0 (vs production default 10.0): the
    # bench fires queries in tight succession to exercise indexing
    # *while* search is happening. With the production 10s window,
    # 10 queries × 10s = 100s of the run would be queue-paused, which
    # would crowd out the actual indexing measurement. 1s is enough to
    # exercise the preempt code path without dominating wall time.
    queue = IndexingQueue(
        pipeline=pipeline, updates_hub=hub,
        config=QueueConfig(
            cooldown_seconds=1, initial_cooldown_seconds=0,
            max_concurrency=6, max_retries=0,
            file_processing_timeout=30,
            search_preempt_seconds=1.0,
        ),
        db=db,
    )

    # Resource sampling (separate task so the indexing path is unmodified).
    proc = psutil.Process(os.getpid())
    res_samples: dict[str, list[float]] = {"cpu": [], "rss_mib": [], "t": []}
    res_stop = asyncio.Event()
    sampler_task = asyncio.create_task(
        _resource_sampler(proc, res_stop, res_samples),
    )

    # Per-file outcomes are tallied from the DB at the end (source of
    # truth). We previously also tried to listen on the SSE hub for
    # real-time tracking, but the hub uses a synchronous context
    # manager and the extra plumbing wasn't worth its cost — the DB
    # row's final status tells the same story.

    # Search query schedule: fire one every (estimated_total / n_queries)
    # seconds, but kick off immediately and re-fire on a cadence — we
    # don't know the indexing duration in advance, so we just space
    # them across whatever the actual run takes.
    query_results: list[dict[str, Any]] = []
    queries_done = asyncio.Event()

    async def _query_loop(start_t: float, total_queries: int, total_files: int):
        # Fire queries at progress milestones, NOT at fixed wall-clock
        # offsets. The previous version assumed indexing would take
        # ~15 s and packed all queries into the first 15 s — when a
        # bigger corpus took 80 s, 65 s of indexing had no queries
        # AND the early queries hit a near-empty DB and returned 0.
        #
        # Strategy: fire query i when ~i/N of the work has drained.
        # Concretely: when (total_files - completed) <= total_files * (1 - (i+1)/(N+1)).
        # Also wait until the queue exists and has drained at least
        # one item, so the first query has SOMETHING to find.
        bank = DEFAULT_QUERY_BANK[:]
        rng = random.Random(seed + 1)
        rng.shuffle(bank)

        async def _completed_count() -> int:
            st = await queue.get_status()
            return total_files - st["total_items"]

        # Wait until at least 1 file is done before firing query 0,
        # so we don't measure cold-cache empty-DB latency.
        for _ in range(300):  # ~30 s safety cap
            if await _completed_count() >= 1:
                break
            await asyncio.sleep(0.1)

        for i in range(total_queries):
            # Trigger threshold: fire when at least ((i+1)/(N+1)) of
            # the corpus is indexed. Spaces queries from "just started"
            # to "almost done."
            target_completed = max(1, int((i + 1) * total_files / (total_queries + 1)))
            for _ in range(3000):  # ~5 min hard cap per query
                done = await _completed_count()
                if done >= target_completed:
                    break
                await asyncio.sleep(0.1)

            q_text = bank[i % len(bank)]
            queue.search_preempt()
            t0 = time.monotonic()
            try:
                results = await searcher.search(q_text, limit=10)
                latency = time.monotonic() - t0
                query_results.append({
                    "i": i, "t_offset": round(t0 - start_t, 3),
                    "query": q_text, "latency_ms": round(latency * 1000, 1),
                    "results": len(results),
                    "completed_at_query_time": await _completed_count(),
                    "preempted": True,
                    "ok": True,
                })
            except Exception as e:
                query_results.append({
                    "i": i, "t_offset": round(time.monotonic() - start_t, 3),
                    "query": q_text, "latency_ms": None,
                    "results": 0, "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                })
        queries_done.set()

    # Start the queue and enqueue everything.
    await queue.start()
    bench_start = time.monotonic()

    enqueue_t0 = time.monotonic()
    for p in copied:
        await queue.enqueue(p, QueueAction.INDEX)
    enqueue_elapsed = time.monotonic() - enqueue_t0

    query_task = asyncio.create_task(
        _query_loop(bench_start, n_queries, len(copied)),
    )

    # Wait for the queue to fully drain. Tight cap (5 min) — if it
    # doesn't finish 100 mocked-AI files in that time, something is
    # wedged and we want to know early. Print periodic progress so a
    # hung run is observable.
    drain_deadline = time.monotonic() + 300
    last_print = 0.0
    while time.monotonic() < drain_deadline:
        st = await queue.get_status()
        if (st["total_items"] == 0
                and st["processing"] == 0
                and st["waiting"] == 0
                and st["cooling_down"] == 0):
            break
        now = time.monotonic()
        if now - last_print > 5.0:
            elapsed_s = now - bench_start
            print(f"[random100bench] t+{elapsed_s:5.1f}s queue: "
                  f"total={st['total_items']} "
                  f"processing={st['processing']} "
                  f"waiting={st['waiting']} "
                  f"cooling_down={st['cooling_down']}")
            last_print = now
        await asyncio.sleep(0.25)

    indexing_elapsed = time.monotonic() - bench_start

    # Wait for queries to finish (in case indexing was so fast they
    # didn't all fire yet — fire any remaining now).
    try:
        await asyncio.wait_for(queries_done.wait(), timeout=20)
    except asyncio.TimeoutError:
        query_task.cancel()
        try:
            await query_task
        except asyncio.CancelledError:
            pass

    # Tear down.
    await queue.stop()
    res_stop.set()
    await sampler_task

    # Tally DB outcomes (the SSE listener may miss events under
    # backpressure; the DB is the source of truth).
    db_status_counts: dict[str, int] = {}
    failed_rows: list[dict[str, Any]] = []
    for p in copied:
        row = await db.get_file_by_path(str(p.resolve()))
        if row is None:
            db_status_counts["NOT_IN_DB"] = db_status_counts.get("NOT_IN_DB", 0) + 1
            continue
        sn = row.status.name if hasattr(row.status, "name") else str(row.status)
        db_status_counts[sn] = db_status_counts.get(sn, 0) + 1
        if sn == "FAILED":
            failed_rows.append({
                "file_path": row.file_path,
                "extension": row.extension,
                "processing_error": row.processing_error,
            })

    await db.close()

    # ---------- Compute statistics ----------
    def _stats(xs: list[float]) -> dict[str, float]:
        if not xs:
            return {"n": 0}
        return {
            "n": len(xs),
            "min": round(min(xs), 4),
            "p50": round(statistics.median(xs), 4),
            "p99": round(sorted(xs)[max(0, int(len(xs) * 0.99) - 1)], 4),
            "max": round(max(xs), 4),
            "mean": round(statistics.fmean(xs), 4),
            "total": round(sum(xs), 4),
        }

    parse_stats = _stats(stage_durations["parse"])
    sum_stats = _stats(stage_durations["summarize"])
    emb_stats = _stats(stage_durations["embed"])

    # Theoretical minimum given measured stage totals + concurrency.
    theoretical_min = max(
        parse_stats.get("total", 0) / 4,
        sum_stats.get("total", 0) / 1,
        emb_stats.get("total", 0) / 1,
    ) + (parse_stats.get("mean", 0)
         + sum_stats.get("mean", 0)
         + emb_stats.get("mean", 0))
    efficiency = (
        theoretical_min / indexing_elapsed if indexing_elapsed > 0 else 0
    )

    cpu_active = [c for c in res_samples["cpu"] if c > 0]
    resource_summary = {
        "samples": len(res_samples["cpu"]),
        "peak_cpu_pct": round(max(res_samples["cpu"], default=0.0), 1),
        "mean_cpu_pct_active": round(
            statistics.fmean(cpu_active), 1) if cpu_active else 0.0,
        "baseline_rss_mib": round(min(res_samples["rss_mib"], default=0), 1),
        "peak_rss_mib": round(max(res_samples["rss_mib"], default=0), 1),
        "rss_growth_mib": round(
            (max(res_samples["rss_mib"], default=0)
             - min(res_samples["rss_mib"], default=0)),
            1,
        ),
    }

    query_latencies = [
        q["latency_ms"] for q in query_results if q.get("latency_ms") is not None
    ]
    query_summary = {
        "total": len(query_results),
        "ok": sum(1 for q in query_results if q.get("ok")),
        "failed": sum(1 for q in query_results if not q.get("ok")),
        "latency_ms": _stats(query_latencies),
    }

    report = {
        "system_info": sysinfo,
        "config": {
            "n_requested": n, "n_actual": len(copied),
            "seed": seed, "source": str(source),
            "n_queries": n_queries,
            "ai_mode": "mocked (real parser, mock summarize+embed)",
            "manifest_seed": seed,
        },
        "selection": manifest.get("selection_stats", {}),
        "manifest_existing_at_run": manifest.get("existing_at_run"),
        "manifest_missing_at_run": manifest.get("missing_at_run"),
        "timings": {
            "enqueue_total_s": round(enqueue_elapsed, 3),
            "indexing_wall_s": round(indexing_elapsed, 3),
            "theoretical_min_s": round(theoretical_min, 3),
            "efficiency_pct": round(efficiency * 100, 1),
            "parse": parse_stats,
            "summarize": sum_stats,
            "embed": emb_stats,
        },
        "resource_usage": resource_summary,
        "outcomes_by_db_status": db_status_counts,
        "failed_files": failed_rows[:50],  # cap to keep report readable
        "failed_files_total": len(failed_rows),
        "queries": query_results,
        "query_summary": query_summary,
    }

    # ---------- Write report ----------
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPORTS_DIR / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    (out_dir / "report.md").write_text(_render_markdown(report))

    print(f"\n[random100bench] report written to {out_dir}")
    print(_render_markdown(report))

    # The benchmark always passes — its purpose is to produce a report.
    # We do, however, fail loudly on a few invariants that would make
    # the report meaningless.
    assert len(copied) > 0, "no files were copied — selection failed"
    assert indexing_elapsed > 0, "no indexing time recorded"
    # If more than half the files ended up NOT_IN_DB, something
    # systemic is broken (e.g. queue silently drops items).
    not_in_db = db_status_counts.get("NOT_IN_DB", 0)
    assert not_in_db < len(copied) // 2, (
        f"{not_in_db}/{len(copied)} files missing from DB after run — "
        f"investigate the queue + pipeline."
    )


def _render_markdown(report: dict[str, Any]) -> str:
    """Pretty-print the report as a single Markdown blob."""
    si = report["system_info"]
    cfg = report["config"]
    t = report["timings"]
    r = report["resource_usage"]
    qs = report["query_summary"]

    def _fmt_stats(label: str, s: dict[str, Any]) -> str:
        if not s or s.get("n", 0) == 0:
            return f"- **{label}**: (no samples)"
        return (
            f"- **{label}**: n={s['n']}, total={s['total']}s, "
            f"mean={s['mean']*1000:.1f}ms, p50={s['p50']*1000:.1f}ms, "
            f"p99={s['p99']*1000:.1f}ms, max={s['max']*1000:.1f}ms"
        )

    lines = [
        f"# random100bench report",
        "",
        f"_Generated {si['timestamp']}_",
        "",
        "## System info",
        "",
        f"- Platform: `{si['platform']}`",
        f"- Machine: `{si['machine']}`",
        f"- Python: `{si['python']}`",
        f"- cosma_backend: `{si['cosma_backend_version']}` "
        f"(api v{si['cosma_backend_api_version']})",
        f"- CPUs: {si['cpu_count_logical']} logical "
        f"({si['cpu_count_physical']} physical)",
        f"- RAM: {si['ram_total_gib']} GiB total, "
        f"{si['ram_available_gib']} GiB free",
        "",
        "## Configuration",
        "",
        f"- Source: `{cfg['source']}`",
        f"- Files: {cfg['n_actual']}/{cfg['n_requested']} requested",
        f"- Seed: {cfg['seed']}",
        f"- Queries: {cfg['n_queries']}",
        f"- AI mode: {cfg['ai_mode']}",
        "",
        "## Selection",
        "",
    ]
    for k, v in report.get("selection", {}).items():
        lines.append(f"- {k}: {v}")
    if report.get("manifest_existing_at_run") is not None:
        lines.append(
            f"- Manifest cache: "
            f"{report['manifest_existing_at_run']} found / "
            f"{report['manifest_missing_at_run']} missing"
        )

    lines += [
        "",
        "## Timings",
        "",
        f"- Enqueue all files: {t['enqueue_total_s']}s",
        f"- Indexing wall time: **{t['indexing_wall_s']}s**",
        f"- Theoretical minimum: {t['theoretical_min_s']}s",
        f"- Efficiency vs theoretical: **{t['efficiency_pct']}%**",
        "",
        "### Per-stage durations",
        "",
        _fmt_stats("parse",     t["parse"]),
        _fmt_stats("summarize", t["summarize"]),
        _fmt_stats("embed",     t["embed"]),
        "",
        "## Resource usage",
        "",
        f"- Samples: {r['samples']} (100 ms cadence)",
        f"- Peak CPU%: **{r['peak_cpu_pct']}%** "
        "(>100% = multi-core utilization)",
        f"- Mean CPU% (active samples): {r['mean_cpu_pct_active']}%",
        f"- Baseline RSS: {r['baseline_rss_mib']} MiB",
        f"- Peak RSS: {r['peak_rss_mib']} MiB",
        f"- RSS growth: {r['rss_growth_mib']} MiB",
        "",
        "## Outcomes (DB-truth)",
        "",
    ]
    for k, v in report.get("outcomes_by_db_status", {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    if report.get("failed_files"):
        lines += [
            f"### Failed files ({report['failed_files_total']} total, "
            f"showing first {len(report['failed_files'])})",
            "",
        ]
        for f in report["failed_files"]:
            err = f.get("processing_error", "(no message)") or "(no message)"
            lines.append(
                f"- `{Path(f['file_path']).name}` "
                f"(.{f.get('extension', '?')}): {err[:200]}"
            )
        lines.append("")

    lines += [
        "## Search queries fired during indexing",
        "",
        f"- Total: {qs['total']}, OK: {qs['ok']}, Failed: {qs['failed']}",
    ]
    if qs.get("latency_ms", {}).get("n", 0) > 0:
        lat = qs["latency_ms"]
        lines.append(
            f"- Latency: mean={lat['mean']:.1f}ms, p50={lat['p50']:.1f}ms, "
            f"p99={lat['p99']:.1f}ms, max={lat['max']:.1f}ms"
        )

    lines += ["", "### Query timeline", "", "| # | t+ (s) | query | latency (ms) | results |", "|---|---|---|---|---|"]
    for q in report.get("queries", []):
        lat = q.get("latency_ms")
        lat_str = f"{lat:.1f}" if isinstance(lat, (int, float)) else "FAIL"
        lines.append(
            f"| {q['i']} | {q.get('t_offset','?')} | "
            f"`{q['query']}` | {lat_str} | {q.get('results',0)} |"
        )

    return "\n".join(lines)
