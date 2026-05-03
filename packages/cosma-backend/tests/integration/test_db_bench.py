"""Database microbenchmark.

Measures throughput and latency for the SQLite operations the indexing
pipeline and search path actually use. Pure DB — no parser, no AI,
no queue. The point is to know which DB ops are the floor (fast,
no need to optimize) vs ceiling (slow, worth investigating).

Run:
    uv run --group test pytest tests/integration/test_db_bench.py \\
        -m benchmark --no-cov -s

Numbers we care about:

  - upsert_file (single)                  — every file index does this 3x
  - upsert_file (100x in tight loop)      — bulk discovery
  - get_file_by_path                      — every watcher event reads this
  - get_files_under_directory_summary     — bulk skip-check at startup
  - upsert_file_embeddings                — every successful embed
  - HybridSearcher-shaped FTS5 + vec0     — every search query
  - get_files_by_status                   — failed-tab list query

We seed N=1000 file rows up front; some metrics scale with corpus size
so we want a representative-but-fast amount of state.
"""

from __future__ import annotations

import asyncio
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import pytest_asyncio

from cosma_backend.db.database import (
    Database, EMBEDDING_STORAGE_DIMENSIONS,
)
from cosma_backend.models.file import File
from cosma_backend.models.status import ProcessingStatus


def _stats(xs: list[float]) -> str:
    if not xs:
        return "(no samples)"
    s = sorted(xs)
    return (
        f"n={len(xs)} min={s[0]*1000:6.2f}ms "
        f"p50={s[len(s)//2]*1000:6.2f}ms "
        f"p99={s[max(0, int(len(s)*0.99)-1)]*1000:6.2f}ms "
        f"max={s[-1]*1000:6.2f}ms"
    )


def _make_file(i: int, base_dir: str = "/tmp/bench") -> File:
    now = datetime.now(timezone.utc)
    return File(
        path=Path(f"{base_dir}/file_{i:05d}.txt"),
        file_path=f"{base_dir}/file_{i:05d}.txt",
        filename=f"file_{i:05d}.txt",
        extension=".txt",
        file_size=4096 + i,
        created=now, modified=now, accessed=now,
        content_type="text/plain",
        content=f"This is bench file number {i} with some realistic body text "
                f"about quarterly reports invoices presentations and design assets.",
        content_hash=f"hash_{i:05d}",
        title=f"Title of file {i}",
        summary=f"Summary text describing the bench file at index {i}.",
        keywords=[f"kw{i}", "bench", "test", "report", "data"],
        status=ProcessingStatus.COMPLETE,
        parsed_at=now, summarized_at=now, embedded_at=now,
        embedding=np.random.default_rng(i).standard_normal(
            EMBEDDING_STORAGE_DIMENSIONS,
        ).astype(np.float32),
        embedding_model="bench",
        embedding_dimensions=EMBEDDING_STORAGE_DIMENSIONS,
    )


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_db_bench(tmp_path: Path):
    db_path = tmp_path / "bench.db"
    db = await Database.from_path(str(db_path))

    print("\n" + "=" * 70)
    print(f"  DB benchmark — {db_path}")
    print("=" * 70)

    # -----------------------------------------------------------------
    # 1. Single-row upsert (cold)
    # -----------------------------------------------------------------
    n_cold = 50
    cold_lat: list[float] = []
    files = [_make_file(i) for i in range(n_cold)]
    for f in files:
        t0 = time.monotonic()
        f.id = await db.upsert_file(f)
        cold_lat.append(time.monotonic() - t0)
    print(f"\nupsert_file (cold, no embeddings):    {_stats(cold_lat)}")

    # -----------------------------------------------------------------
    # 2. Single-row upsert with embedding (the real production path)
    # -----------------------------------------------------------------
    n_embed = 50
    embed_lat: list[float] = []
    files_with_embed = [_make_file(n_cold + i) for i in range(n_embed)]
    for f in files_with_embed:
        t0 = time.monotonic()
        f.id = await db.upsert_file(f)
        await db.upsert_file_embeddings(f, first_embed=True)
        embed_lat.append(time.monotonic() - t0)
    print(f"upsert_file + upsert_embeddings:      {_stats(embed_lat)}")

    # -----------------------------------------------------------------
    # 3. Bulk seed (1000 files) so subsequent reads see realistic state
    # -----------------------------------------------------------------
    target = 1000
    bulk_t0 = time.monotonic()
    for i in range(n_cold + n_embed, target):
        f = _make_file(i)
        f.id = await db.upsert_file(f)
        await db.upsert_file_embeddings(f, first_embed=True)
    bulk_elapsed = time.monotonic() - bulk_t0
    print(f"\nBulk seed to {target} rows:            "
          f"{bulk_elapsed:.2f}s "
          f"({(target - n_cold - n_embed) / bulk_elapsed:.1f} files/sec)")

    # -----------------------------------------------------------------
    # 4. get_file_by_path — every watcher event reads this
    # -----------------------------------------------------------------
    paths = [f"/tmp/bench/file_{i:05d}.txt" for i in range(0, target, 7)]
    read_lat: list[float] = []
    for p in paths:
        t0 = time.monotonic()
        await db.get_file_by_path(p)
        read_lat.append(time.monotonic() - t0)
    print(f"\nget_file_by_path:                     {_stats(read_lat)}")

    # -----------------------------------------------------------------
    # 5. Bulk skip-check (the optimization that turned 10k roundtrips → 1)
    # -----------------------------------------------------------------
    bulk_skip_lat: list[float] = []
    for _ in range(20):
        t0 = time.monotonic()
        summary = await db.get_files_under_directory_summary("/tmp/bench")
        bulk_skip_lat.append(time.monotonic() - t0)
        assert len(summary) == target
    print(f"get_files_under_directory_summary({target}): "
          f"{_stats(bulk_skip_lat)}")

    # -----------------------------------------------------------------
    # 6. get_files_by_status — Failed/Recent tabs use this
    # -----------------------------------------------------------------
    status_lat: list[float] = []
    for _ in range(20):
        t0 = time.monotonic()
        await db.get_files_by_status("COMPLETE", limit=50, offset=0)
        status_lat.append(time.monotonic() - t0)
    print(f"get_files_by_status (limit=50):       {_stats(status_lat)}")

    # -----------------------------------------------------------------
    # 7. FTS5 search (the keyword half of HybridSearcher)
    # -----------------------------------------------------------------
    queries = ["bench", "report", "invoice", "data", "presentation",
               "test report", "kw5"]
    fts_lat: list[float] = []
    fts_results: dict[str, int] = {}
    async with db.acquire() as conn:
        for q in queries * 5:  # 5x for stable percentiles
            t0 = time.monotonic()
            cur = await conn.execute(
                "SELECT rowid FROM files_fts WHERE files_fts MATCH ? LIMIT 50",
                (q,),
            )
            rows = await cur.fetchall()
            fts_lat.append(time.monotonic() - t0)
            fts_results[q] = len(rows)
    print(f"\nFTS5 MATCH (limit=50):                {_stats(fts_lat)}")
    print(f"  hits: {fts_results}")

    # -----------------------------------------------------------------
    # 8. vec0 nearest-neighbor search (the semantic half)
    # -----------------------------------------------------------------
    vec_lat: list[float] = []
    rng = np.random.default_rng(0)
    for _ in range(35):
        q_vec = rng.standard_normal(EMBEDDING_STORAGE_DIMENSIONS).astype(np.float32)
        # Normalize for cosine
        q_vec /= max(1e-8, float(np.linalg.norm(q_vec)))
        t0 = time.monotonic()
        results = await db.search_similar_files(
            query_embedding=q_vec, limit=20, threshold=10.0,
        )
        vec_lat.append(time.monotonic() - t0)
    print(f"vec0 nearest-neighbor (k=20):         {_stats(vec_lat)}")

    # -----------------------------------------------------------------
    # 9. update_file_timestamp — called by every directory sweep entry
    # -----------------------------------------------------------------
    touch_lat: list[float] = []
    for i in range(0, 100):
        t0 = time.monotonic()
        await db.update_file_timestamp(f"/tmp/bench/file_{i:05d}.txt")
        touch_lat.append(time.monotonic() - t0)
    print(f"\nupdate_file_timestamp:                {_stats(touch_lat)}")

    # -----------------------------------------------------------------
    # 10. Concurrent reads while writes are happening (simulates the
    #     real fight: search/queue-status calls during heavy indexing)
    # -----------------------------------------------------------------
    async def writer(n: int) -> float:
        t0 = time.monotonic()
        for i in range(n):
            f = _make_file(target + i)
            f.id = await db.upsert_file(f)
            await db.upsert_file_embeddings(f, first_embed=True)
        return time.monotonic() - t0

    async def reader_loop(stop_t: float, results: list[float]) -> None:
        # Mimic the search endpoint: bulk skip-check + a couple of
        # FTS lookups every iteration.
        while time.monotonic() < stop_t:
            t0 = time.monotonic()
            await db.get_files_under_directory_summary("/tmp/bench")
            async with db.acquire() as conn:
                cur = await conn.execute(
                    "SELECT rowid FROM files_fts WHERE files_fts MATCH ? LIMIT 20",
                    ("report",),
                )
                await cur.fetchall()
            results.append(time.monotonic() - t0)
            await asyncio.sleep(0.02)

    write_n = 200
    reader_lat_under_load: list[float] = []
    contention_t0 = time.monotonic()
    writer_task = asyncio.create_task(writer(write_n))
    reader_task = asyncio.create_task(
        reader_loop(time.monotonic() + 30, reader_lat_under_load),
    )
    write_elapsed = await writer_task
    reader_task.cancel()
    try:
        await reader_task
    except asyncio.CancelledError:
        pass
    contention_total = time.monotonic() - contention_t0

    print()
    print("Concurrent reader+writer (the real production fight):")
    print(f"  Writer: {write_n} files in {write_elapsed:.2f}s "
          f"({write_n/write_elapsed:.1f}/sec)")
    print(f"  Reader during writes: {_stats(reader_lat_under_load)}")
    print(f"  Total elapsed: {contention_total:.2f}s")

    # -----------------------------------------------------------------
    # 11. SQLite pragma snapshot — what does the running DB say?
    # -----------------------------------------------------------------
    print("\nActive SQLite settings:")
    for pragma in ("journal_mode", "synchronous", "cache_size",
                   "page_size", "mmap_size", "temp_store",
                   "wal_autocheckpoint"):
        async with db.acquire() as conn:
            cur = await conn.execute(f"PRAGMA {pragma}")
            row = await cur.fetchone()
            val = row[0] if row else "(none)"
            print(f"  {pragma:24s} = {val}")

    # -----------------------------------------------------------------
    # 12. DB file size
    # -----------------------------------------------------------------
    db_size = db_path.stat().st_size
    print(f"\nDB file size after {target + write_n} rows: "
          f"{db_size / 1024 / 1024:.2f} MiB "
          f"({db_size / (target + write_n):.0f} bytes/row average)")

    print("=" * 70)
    await db.close()
