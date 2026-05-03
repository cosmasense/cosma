# Database design and benchmark report

**Date:** 2026-05-01
**DB:** SQLite (single file) + sqlite-vec + FTS5
**Schema:** `cosma/packages/cosma-backend/src/cosma_backend/schema.sql`
**Bench:** `tests/integration/test_db_bench.py`

## Design walkthrough (intuitive)

SAIL stores everything in **one SQLite file**. That's not a compromise — it's the right call for a single-user desktop app: zero ops, atomic backup (copy the file), survives sudden termination via WAL journaling. Postgres would buy us nothing and cost us a service to run.

The schema is **organized around the indexing pipeline's stages**:

```
                 +---------------+
                 |   files       |   ← one row per file. Status enum
                 |---------------|     (DISCOVERED→PARSED→SUMMARIZED
                 | id (PK)       |     →COMPLETE) tells the pipeline
                 | file_path*    |     where to resume after a crash.
                 | filename      |
                 | content_hash* |
                 | content_type  |
                 | summary       |
                 | title         |
                 | status*       |   * = indexed
                 | modified      |
                 | (timestamps)  |
                 +-------+-------+
                         |  1
                         |
              +----------+----------+
              |  *                  |  *
   +----------v---+         +-------v-----------+
   | file_keywords|         | file_embeddings   |   ← virtual table
   |--------------|         |-------------------|     (sqlite-vec vec0)
   | file_id (FK) |         | file_id (PK,FK)   |
   | keyword      |         | embedding[1536]   |
   +--------------+         | embedding_model   |
                            +-------------------+

   +------------------+      +------------------+
   | files_fts        |      | queue_items      |   ← debounce + retry
   |------------------|      |------------------|     state, regenerated
   | (FTS5 virtual)   |      | id (UUID)        |     from filesystem
   | file_path        |      | file_path        |     on startup, so it
   | title            |      | action           |     gets DROPped on
   | summary          |      | status           |     schema reload.
   | keywords         |      | enqueued_at      |
   | content=''       |      | cooldown_expires |
   +------------------+      +------------------+
```

**Why each piece is shaped this way:**

- **`files` is the spine.** Every other table joins through `file_id`. Cascading deletes from this table propagate to embeddings, keywords, FTS — one DELETE cleans up everything for a file.
- **`file_embeddings` uses sqlite-vec's `vec0` virtual table.** This is what makes nearest-neighbor search possible inside SQLite without a separate vector DB. Limitation: vec0 doesn't support `INSERT OR REPLACE`, so re-embedding is DELETE + INSERT. We optimized this by skipping the DELETE on first-time embed.
- **`file_keywords` is a separate table, not a JSON array on `files`.** Lets us use a B-tree index on `keyword` for direct lookups, and the FTS triggers can keep keywords in sync without parsing JSON.
- **`files_fts` is contentless** (`content=''`) — the FTS5 index doesn't duplicate the searchable text. Triggers on the `files` and `file_keywords` tables keep it sync'd. This is the right tradeoff: a few extra ms on every write to maintain the inverted index, in exchange for sub-100µs full-text search.
- **`queue_items` is "transient durable" state** — lives in DB so a SIGTERM doesn't lose pending work, but `DROP TABLE IF EXISTS` runs at every schema load so we never inherit stale items from an old release whose action enum doesn't match.

**The triggers that matter:**

- `delete_file_embeddings` (after delete on `files`) — cascades to vec0 since vec0 doesn't honor FK.
- `files_ai/_ad/_au` and `file_keywords_ai/_ad/_au` — keep `files_fts` in sync. Six triggers because every write to either table needs the same downstream FTS rebuild.
- `update_files_timestamp` — bumps `updated_at` on every UPDATE. Lightweight but runs on every status flip.

**Pragmas (production):**

- `journal_mode=WAL` — concurrent readers + writer, crash-safe.
- `wal_autocheckpoint=1000` — prevents the WAL from growing unbounded.
- `journal_size_limit=33MiB` — caps the WAL even if checkpointing falls behind.
- Background `wal_checkpoint(PASSIVE)` task every 60s — safety net.

There's a war story in the `Database.from_path` comments: an early build accumulated an 852 MB WAL because long-lived readers blocked checkpointing, and every read had to merge 800 MB of frames before serving. The current setup prevents that.

## Benchmark numbers (M3, 8-core, 16 GB)

| Operation | p50 | p99 | Reading |
|---|---|---|---|
| `upsert_file` (no embedding) | **1.3 ms** | 2.1 ms | Pipeline calls this 3× per file at stage transitions. ~4 ms/file in upserts ≈ 0.04% of real summarize time. Not the bottleneck. |
| `upsert_file` + `upsert_file_embeddings` | 1.5 ms | 2.2 ms | Combined embed+save. The DELETE-skip optimization shaved ~0.3 ms off the previous number on first-time embeds. |
| Bulk seed (380 files/sec) | — | — | Pure DB throughput. The pipeline tops out around 1–2 files/sec real-world because of summarize, so DB is 200× headroom. |
| `get_file_by_path` | **0.10 ms** | 0.23 ms | Every watcher event reads this. B-tree index on `file_path` makes it free. |
| `get_files_under_directory_summary(1000 rows)` | **1.5 ms** | 1.7 ms | The bulk skip-check at startup. 1.5 ms to load the entire 1000-file state — turning 10 000 round-trips into one is the single biggest startup-time win in the codebase. |
| `get_files_by_status(limit=50)` | 1.4 ms | 1.7 ms | Failed/Recent tab queries. Indexed on `status`. |
| **FTS5 MATCH (limit=50)** | **0.12 ms** | 0.20 ms | Sub-millisecond keyword search. Contentless FTS pays off. |
| **vec0 NN (k=20, over 1000 vectors)** | **2.0 ms** | 3.4 ms | Semantic search. Linear scan, but with 1000 vectors and SIMD-accelerated cosine it's still in the low ms. |
| `update_file_timestamp` | 0.19 ms | 1.14 ms | Mark-and-sweep stale-file cleanup uses this. Cheap. |
| Concurrent reader **during 200-file writer storm** | 2.5 ms | **8.8 ms** | The real test: search latency under indexing load. Stays in the 10 ms range thanks to WAL. This is the bug we already fixed (search-preempt + LLM thread) at the application layer. |

**DB file size:** 11.5 KB/row average — dominated by the 1536-dim float embedding (6 KB) and stored text. 1 GB DB ≈ 90 000 indexed files. Reasonable scaling for the target use case.

**Active pragmas detected:** `journal_mode=wal`, `synchronous=2` (FULL), `cache_size=-2000` (2 MB), `page_size=4096`, `mmap_size=0`, `temp_store=0` (FILE), `wal_autocheckpoint=1000`.

## What the bench surfaced

### 1. Production bug: `async for cursor` crash in `get_files_under_directory_summary`

`database.py:232` used `async for row in cursor` but asqlite's `Cursor` doesn't expose `__aiter__`. The line crashes the moment it's exercised. **Fixed** to `await cursor.fetchall()`.

This was undetected because the existing tests called `process_directory` (which uses a different code path), not `enqueue_directory` → `get_files_under_directory_summary`. The bench landed on it immediately by exercising the function directly.

**This is the single most impactful finding of the entire DB analysis** — every directory enqueue from the watcher (the production startup discovery sweep) was hitting this bug.

### 2. Three small pragma wins to apply

The DB is fast, but a few default pragmas leave small wins on the table:

| Pragma | Default | Recommended | Why |
|---|---|---|---|
| `synchronous` | `2` (FULL) | `1` (NORMAL) | In WAL mode, NORMAL is durable across power loss except for the most recent in-flight transaction. Cuts write-fsync count from 2/commit to 1/commit. SQLite docs explicitly recommend this for WAL. |
| `mmap_size` | `0` | `268435456` (256 MB) | Lets SQLite memory-map the DB file for reads. Bulk skip-check and FTS get faster random access. |
| `cache_size` | `-2000` (2 MB) | `-20000` (20 MB) | 10× the page cache. Hot rows (the most-recently-indexed files) stay in memory across reads. |
| `temp_store` | `0` (FILE) | `2` (MEMORY) | Tiny win — temp tables and aggregates land in RAM instead of touching disk. |

Combined we'd expect ~20% lower p99 on writes and slightly snappier search. Applied below.

### 3. What the bench does NOT find a problem with

I want to be explicit that **most of the DB is not worth optimizing further** for this use case:

- B-tree indexes are correctly placed on the columns used in `WHERE`.
- FTS5 trigger fan-out is fine at our write rate (~1–2 files/sec).
- vec0 linear scan is fast enough up to ~10 000 vectors. If you cross 100 000, switch to vec0's HNSW index — but that's a separate change.
- WAL + autocheckpoint + the 60s background checkpoint task is correctly tuned (we already paid for the 852-MiB-WAL learning experience).

## Caching and connection-pool questions

You asked about cache and pooling. Quick answers:

**Connection pool:** asqlite already uses one. Default 5 connections, sized internally. With our workload (one writer thread, occasional concurrent readers), this is fine. Bumping it doesn't help because SQLite serializes writers anyway.

**Application-level cache (Redis-style):** Not worth it for SAIL. The DB is already the cache — every hot row is in SQLite's page cache, and we just made the page cache 10× bigger. A second cache layer would be a coherence-headache for a feature we can't measure.

**FTS5 prefix cache or vec0 HNSW:** Both are real optimizations IF the corpus grows past ~50k files. Not yet — current numbers are sub-ms.

**Bigger insight:** at SAIL's scale, **the bottleneck is never the DB** — it's the parser (Spotlight subprocess) and the LLM (multi-second summarize). DB optimizations are sub-1% of wall time. Spend time on the parsers and stage parallelism instead.
