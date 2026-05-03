"""random10bench_real — same shape as random100bench, but with REAL
AI: Qwen3-VL-2B-Instruct via llama.cpp + Qwen3-Embedding-0.6B via
SentenceTransformer. N=10 by default because each file takes 5–30 s
of summarize, so 100 files would be 30+ minutes per run.

This is the bench that tells you whether the GPU is actually being
exercised end-to-end. Watch Activity Monitor's GPU pane while it
runs — if Metal usage stays at 0%, the LLM isn't paged into the GPU
backend correctly.

Run:
    uv run --group test pytest \\
        tests/integration/test_random10bench_real.py \\
        -m benchmark --no-cov -s

Set COSMA_BENCH_REAL_N=20 to do more files. Plan for ~3 min/file
on M-series with 2B-Q4 GGUF.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import random
import shutil
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
import pytest
import pytest_asyncio

import cosma_backend
from cosma_backend.db.database import Database
from cosma_backend.discoverer import Discoverer
from cosma_backend.embedder import AutoEmbedder
from cosma_backend.models.file import File
from cosma_backend.models.status import ProcessingStatus
from cosma_backend.parser import FileParser
from cosma_backend.pipeline import Pipeline
from cosma_backend.queue import IndexingQueue, QueueAction
from cosma_backend.searcher import HybridSearcher
from cosma_backend.settings import (
    EmbedderConfig, ParserConfig, QueueConfig, SummarizerConfig,
)
from cosma_backend.summarizer import AutoSummarizer
from cosma_backend.utils.pubsub import Hub


DEFAULT_N = 10
SEED = 42
QUERY_BANK = [
    "report", "invoice", "screenshot", "summary", "data",
    "image", "document", "code", "log", "presentation",
]
MEDIA_EXTS = {".mp3", ".wav", ".aac", ".mp4", ".mov", ".mkv", ".avi"}
MAX_FILE_SIZE = 50 * 1024 * 1024


def _system_info() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cosma_backend_version": getattr(cosma_backend, "__version__", "?"),
        "cosma_backend_api_version": getattr(cosma_backend, "__api_version__", "?"),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "ram_total_gib": round(vm.total / 1024 ** 3, 2),
        "ram_available_gib": round(vm.available / 1024 ** 3, 2),
    }


def _check_models_present() -> tuple[bool, str]:
    """Return (ok, message). ok=False → skip the bench cleanly."""
    llama_dir = Path.home() / "Library/Application Support/cosma/models/llama/gguf"
    if not llama_dir.exists():
        return False, f"missing llama models dir: {llama_dir}"
    ggufs = list(llama_dir.glob("*.gguf"))
    if not ggufs:
        return False, f"no .gguf models under {llama_dir}"
    return True, f"found GGUF: {ggufs[0].name}"


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_random10bench_real(tmp_path: Path):
    """Real-AI smoke benchmark."""
    n = int(os.environ.get("COSMA_BENCH_REAL_N", DEFAULT_N))
    source = Path(os.environ.get(
        "COSMA_BENCH_SOURCE", str(Path.home() / "Downloads"),
    )).expanduser()

    ok, msg = _check_models_present()
    if not ok:
        pytest.skip(f"real-AI bench skipped: {msg}")
    print(f"\n[random10bench_real] {msg}")

    if not source.exists():
        pytest.skip(f"source dir does not exist: {source}")

    sys_info = _system_info()
    print(f"[random10bench_real] system: {sys_info['platform']} | "
          f"{sys_info['cpu_count_logical']} CPUs | "
          f"{sys_info['ram_total_gib']} GiB RAM | "
          f"cosma {sys_info['cosma_backend_version']}")

    parser_for_exts = FileParser()
    supported_exts = set(parser_for_exts.get_supported_extensions()) - MEDIA_EXTS

    # Use the same manifest path as random100bench so seed_42 picks
    # the SAME 10 files (a subset of the 100).
    bench_root = (Path(__file__).resolve().parent.parent.parent
                  / "benchmarks/random100bench/manifests")
    manifest_path = bench_root / f"seed_{SEED}.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        cached = [Path(p) for p in manifest["paths"] if Path(p).exists()]
        rng = random.Random(SEED)
        rng.shuffle(cached)
        sampled = cached[:n]
        print(f"[random10bench_real] reusing manifest seed_{SEED}: "
              f"{len(sampled)} of {len(cached)} cached files")
    else:
        # Fallback: walk source ourselves.
        candidates = []
        for root, dirs, files in os.walk(source):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for fn in files:
                if fn.startswith("."):
                    continue
                p = Path(root) / fn
                if p.suffix.lower() not in supported_exts:
                    continue
                try:
                    if p.stat().st_size > MAX_FILE_SIZE:
                        continue
                except OSError:
                    continue
                candidates.append(p)
        rng = random.Random(SEED)
        sampled = rng.sample(candidates, min(n, len(candidates)))

    if not sampled:
        pytest.skip("no candidate files")

    # Copy into tmp.
    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for i, src in enumerate(sampled):
        dst = corpus / f"{i:03d}_{src.name}"
        try:
            shutil.copy2(src, dst)
            copied.append(dst)
        except OSError as e:
            print(f"[random10bench_real] copy failed {src}: {e}")
    print(f"[random10bench_real] copied {len(copied)} files")

    # Configure pipeline executor for the bench. Real models add load.
    from cosma_backend.pipeline_executor import configure_pipeline_executor
    configure_pipeline_executor(max_workers=6)

    # Build the REAL pipeline. Default settings → llama.cpp summarizer
    # + local SentenceTransformer embedder.
    settings_summarizer = SummarizerConfig()
    settings_embedder = EmbedderConfig()
    settings_parser = ParserConfig()

    hub: Hub = Hub()
    db_path = tmp_path / "bench_real.db"
    db = await Database.from_path(str(db_path))

    # Pre-load embedder + LLM in threads — same as production Phase 2.
    print("[random10bench_real] loading embedder model (this is the moment GPU should spin up)...")
    embedder_load_t0 = time.monotonic()
    embedder = AutoEmbedder(config=settings_embedder, preferred_provider="local")
    await asyncio.to_thread(embedder._eagerly_initialize_models)
    embedder_load_s = time.monotonic() - embedder_load_t0
    print(f"[random10bench_real] embedder loaded in {embedder_load_s:.2f}s")

    print("[random10bench_real] loading LLM (Qwen3-VL via llama.cpp)...")
    llm_load_t0 = time.monotonic()
    summarizer = AutoSummarizer(config=settings_summarizer)
    if hasattr(summarizer, "_get_llamacpp_summarizer"):
        await summarizer._get_llamacpp_summarizer()
    llm_load_s = time.monotonic() - llm_load_t0
    print(f"[random10bench_real] LLM loaded in {llm_load_s:.2f}s")

    pipeline = Pipeline(
        db=db, updates_hub=hub,
        discoverer=Discoverer(),
        parser=FileParser(config=settings_parser),
        summarizer=summarizer, embedder=embedder,
        parse_concurrency=4, summarize_concurrency=1, embed_concurrency=1,
    )
    searcher = HybridSearcher(db=db, embedder=embedder)

    queue = IndexingQueue(
        pipeline=pipeline, updates_hub=hub,
        config=QueueConfig(
            cooldown_seconds=1, initial_cooldown_seconds=0,
            max_concurrency=6, max_retries=0,
            file_processing_timeout=180,  # generous: real summarize can be slow
            search_preempt_seconds=2.0,
        ),
        db=db,
    )

    # Resource sampling.
    proc = psutil.Process(os.getpid())
    res_samples: dict[str, list[float]] = {"cpu": [], "rss_mib": [], "t": []}
    res_stop = asyncio.Event()
    proc.cpu_percent(interval=None)
    async def sampler():
        while not res_stop.is_set():
            res_samples["cpu"].append(proc.cpu_percent(interval=None))
            res_samples["rss_mib"].append(proc.memory_info().rss / 1024 / 1024)
            res_samples["t"].append(time.monotonic())
            try:
                await asyncio.wait_for(res_stop.wait(), timeout=0.25)
            except asyncio.TimeoutError:
                pass
    sampler_task = asyncio.create_task(sampler())

    await queue.start()
    bench_start = time.monotonic()
    for p in copied:
        await queue.enqueue(p, QueueAction.INDEX)

    # Single mid-run query — fire when ~30% of files are done.
    query_results: list[dict] = []
    async def query_loop():
        target = max(1, int(0.3 * len(copied)))
        for _ in range(600):
            st = await queue.get_status()
            if (len(copied) - st["total_items"]) >= target:
                break
            await asyncio.sleep(0.5)
        # Fire the query — exercises preempt + cancel + real semantic search.
        for q in QUERY_BANK[:3]:
            queue.search_preempt()
            queue.cancel_in_flight()
            t0 = time.monotonic()
            try:
                results = await searcher.search(q, limit=10)
                lat = time.monotonic() - t0
                query_results.append({
                    "query": q, "latency_ms": round(lat * 1000, 1),
                    "results": len(results),
                    "t_offset": round(t0 - bench_start, 2),
                })
            except Exception as e:
                query_results.append({
                    "query": q, "error": f"{type(e).__name__}: {e}",
                    "t_offset": round(time.monotonic() - bench_start, 2),
                })
            await asyncio.sleep(2.0)

    qtask = asyncio.create_task(query_loop())

    # Drain. Hard cap at 30 minutes.
    drain_deadline = time.monotonic() + 30 * 60
    last_print = 0.0
    while time.monotonic() < drain_deadline:
        st = await queue.get_status()
        if (st["total_items"] == 0 and st["processing"] == 0
                and st["waiting"] == 0 and st["cooling_down"] == 0):
            break
        now = time.monotonic()
        if now - last_print > 15.0:
            done = len(copied) - st["total_items"]
            print(f"[random10bench_real] t+{now - bench_start:.0f}s "
                  f"{done}/{len(copied)} done | "
                  f"queue={st['total_items']} proc={st['processing']} "
                  f"CPU={(res_samples['cpu'][-1] if res_samples['cpu'] else 0):.0f}% "
                  f"RSS={(res_samples['rss_mib'][-1] if res_samples['rss_mib'] else 0):.0f}MiB")
            last_print = now
        await asyncio.sleep(1.0)

    indexing_elapsed = time.monotonic() - bench_start
    try:
        await asyncio.wait_for(qtask, timeout=30)
    except asyncio.TimeoutError:
        qtask.cancel()
        try:
            await qtask
        except asyncio.CancelledError:
            pass

    await queue.stop()
    res_stop.set()
    await sampler_task

    # Outcomes from DB.
    db_status: dict[str, int] = {}
    failed: list[dict] = []
    for p in copied:
        row = await db.get_file_by_path(str(p.resolve()))
        if row is None:
            db_status["NOT_IN_DB"] = db_status.get("NOT_IN_DB", 0) + 1
            continue
        sn = row.status.name if hasattr(row.status, "name") else str(row.status)
        db_status[sn] = db_status.get(sn, 0) + 1
        if sn == "FAILED":
            failed.append({
                "filename": Path(row.file_path).name,
                "extension": row.extension,
                "error": (row.processing_error or "")[:200],
            })

    await db.close()

    cpu_active = [c for c in res_samples["cpu"] if c > 0]
    peak_cpu = max(cpu_active, default=0.0)
    avg_cpu = statistics.fmean(cpu_active) if cpu_active else 0.0
    peak_rss = max(res_samples["rss_mib"], default=0.0)
    base_rss = min(res_samples["rss_mib"], default=0.0)

    # Report.
    print()
    print("=" * 70)
    print(f"  random10bench_real — {len(copied)} files, real Qwen3-VL + Qwen3-Embedding")
    print("=" * 70)
    print(f"  Embedder load:        {embedder_load_s:7.2f}s")
    print(f"  LLM load:             {llm_load_s:7.2f}s")
    print(f"  Indexing wall time:   {indexing_elapsed:7.2f}s "
          f"({indexing_elapsed/max(1, len(copied)):.2f}s/file mean)")
    print()
    print(f"  Outcomes (DB-truth):  {db_status}")
    if failed:
        print(f"  Failed files ({len(failed)}):")
        for f in failed:
            print(f"    - {f['filename']} (.{f['extension']}): {f['error']}")
    print()
    print(f"  Peak CPU%:            {peak_cpu:7.1f} %")
    print(f"  Mean CPU% (active):   {avg_cpu:7.1f} %")
    print(f"  Baseline RSS:         {base_rss:7.1f} MiB")
    print(f"  Peak RSS:             {peak_rss:7.1f} MiB "
          f"(+{peak_rss-base_rss:.1f} growth)")
    print()
    print(f"  Search queries during indexing:")
    for q in query_results:
        if "error" in q:
            print(f"    t+{q['t_offset']}s  '{q['query']}'  ERROR: {q['error']}")
        else:
            print(f"    t+{q['t_offset']}s  '{q['query']}'  "
                  f"latency={q['latency_ms']}ms  results={q['results']}")
    print()
    if peak_rss > 2000:
        print(f"  GPU likely engaged: RSS peaked at {peak_rss:.0f} MiB — "
              "consistent with Qwen3-VL-2B + embedder loaded.")
    else:
        print(f"  Suspicious: peak RSS only {peak_rss:.0f} MiB. "
              "Models may not have actually loaded into the inference path.")
    print("=" * 70)

    # Save report.
    out_dir = (Path(__file__).resolve().parent.parent.parent
               / "benchmarks/random100bench/reports"
               / f"real_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps({
        "system_info": sys_info,
        "n_files": len(copied),
        "embedder_load_s": embedder_load_s,
        "llm_load_s": llm_load_s,
        "indexing_wall_s": indexing_elapsed,
        "db_status": db_status,
        "failed": failed,
        "peak_cpu": peak_cpu, "avg_cpu_active": avg_cpu,
        "baseline_rss_mib": base_rss, "peak_rss_mib": peak_rss,
        "queries": query_results,
    }, indent=2))
    print(f"[random10bench_real] report → {out_dir}")

    # Sanity: at least half should COMPLETE.
    complete = db_status.get("COMPLETE", 0)
    assert complete >= len(copied) // 2, (
        f"Only {complete}/{len(copied)} files COMPLETE — real models "
        f"are misbehaving. Failed: {failed}"
    )
