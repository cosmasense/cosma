# Stage-pipelined indexing — design

**Date:** 2026-05-01
**Status:** Implementation in progress (Goal 2 of the demo-prep work)
**See also:** `STARTUP_PROFILE_FINDINGS.md` for why this matters.

## Goals

1. **GPU never idle.** While file A is being embedded on MPS, file B
   should be summarizing on Metal, file C should be parsing on the CPU.
2. **Crash-safe pickup.** Force-quitting mid-summarize must leave the
   parsed text on disk so the next launch resumes from PARSED rather
   than re-parsing from scratch.
3. **No public-API change.** The frontend, scheduler, watcher,
   `/api/queue/*` endpoints all continue to work unchanged.

## Non-goals

- Sub-millisecond schedulability or work-stealing across machines.
- Replacing the existing IndexingQueue. We're adding to the pipeline,
  not rebuilding the queue.
- A new dependency. Per-stage parallelism is asyncio + semaphores.

## Mechanism

### Per-stage semaphores in `Pipeline`

```python
self._parse_sem      = asyncio.Semaphore(parse_concurrency)      # default 4
self._summarize_sem  = asyncio.Semaphore(summarize_concurrency)  # default 1
self._embed_sem      = asyncio.Semaphore(embed_concurrency)      # default 1
```

`process_file` becomes a sequence of three semaphore-guarded stages.
A file holds at most one semaphore at a time, so a file that's parsing
does not also block the embed slot. Across files, the three stages
run truly in parallel — bounded only by their own semaphores.

Backpressure is implicit: if summarize is slow, parse-finishers pile
up at `await summarize_sem.acquire()`. They consume zero CPU while
waiting (asyncio condition variable). No separate queue, no `maxsize`
to tune, no risk of unbounded memory growth from queued File objects.

### Stage methods on `Pipeline`

| Method | Holds | Persists on success | Reads from DB on resume |
|---|---|---|---|
| `_run_parse_stage(file)` | `_parse_sem` | `status=PARSED`, `content_hash`, `content`, `parsed_at` | n/a (always runs) |
| `_run_summarize_stage(file)` | `_summarize_sem` | `status=SUMMARIZED`, `title`, `summary`, `keywords`, `summarized_at` | `content`, `content_hash` |
| `_run_embed_stage(file)` | `_embed_sem` | `status=COMPLETE`, embeddings table row, `embedded_at` | `summary`, `keywords` |

Each stage calls `_save_to_db(file)` on success. If a crash interrupts
us, the row is durable at the last completed stage.

### `process_file` orchestration with resumption

```
1. Skip-check (mtime + status in COMPLETE/FAILED): existing logic.
2. Look up saved_file in DB.
3. Decide entry stage:
     saved_file is None or status == DISCOVERED  → start at parse
     saved_file.status == PARSED                 → load content, jump to summarize
     saved_file.status == SUMMARIZED             → load summary/keywords, jump to embed
     saved_file.status == FAILED                 → re-run from parse (full retry)
4. Run from entry stage onward. Each stage persists on completion.
5. After embed: status=COMPLETE, embeddings stored.
```

### Removal of mid-flight `delete_file`

The current code calls `db.delete_file(path)` between parse and
summarize "so a crash-recovery retry starts from a clean slate."
That's exactly the wrong tradeoff for Goal 2 — the deletion is what
makes crash recovery impossible. We remove it. Stages overwrite their
own fields on retry; the embeddings table uses DELETE-then-INSERT
inside `upsert_file_embeddings`, so stale vectors can't outlive a
re-embed.

The original concern (partial keyword/FTS data) is addressed by
deleting only those rows that the current stage is about to rewrite,
not the whole `files` row.

### Crash recovery at startup

`IndexingQueue.start()` already calls `_load_from_db()` which restores
queue state. We extend it: after that, query
`files WHERE status IN (DISCOVERED, PARSED, SUMMARIZED)` and re-enqueue
each as `QueueAction.INDEX`. The pipeline's resume-aware `process_file`
will pick up where each one left off. This must run BEFORE the queue
starts accepting watcher events to prevent double-processing.

## Concurrency tuning

```toml
[queue]
parse_concurrency = 4       # CPU + disk; safe to overlap heavily
summarize_concurrency = 1   # llama.cpp via Metal — single GPU context
embed_concurrency = 1       # SentenceTransformer on MPS — same constraint
max_concurrency = 6         # ceiling on total in-flight files
```

Old `max_concurrency` (default 2) becomes the **upper bound** on
in-flight Files. With per-stage semaphores 4/1/1, you can sustain
4 in parse + 1 in summarize + 1 in embed = 6. Setting it lower than
4+1+1 just throttles parse.

## Throughput claim

Old (max_concurrency=2, no per-stage):
- Two files run their full pipelines side-by-side. While file A is in
  summarize, file B is *also* in summarize (or also fighting for GPU
  in embed). GPU contention; no benefit from parallelism.

New (max_concurrency=6, per-stage 4/1/1):
- 4 files parsing on CPU
- 1 file on the GPU summarize (uninterrupted)
- 1 file on MPS embed (uninterrupted)

Wall-time gain on a corpus of N files where
`T_parse << T_summarize ≈ T_embed`:
~30–50% throughput improvement, dominated by removing the GPU idle
gaps between `summarize(file_k)` finishing and `embed(file_k+1)`
starting.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Memory: 6 in-flight Files holding `content` strings | content is bounded (parser truncates ≥4 KB chunks). 6 × ~50 KB negligible. |
| Embedder cold-start blocks first batch | Already pre-warmed in Phase 2 (Goal 1, A3). |
| Race: two crash-recovery items + a new watcher event for same path | Existing `_items` dict keyed by file_path already de-dupes; second enqueue collapses into the first. |
| Migration: existing `_save_to_db` called before stage 3 changes meaning of "row in DB" | All-existing rows are status=COMPLETE or FAILED. New PARSED/SUMMARIZED rows only appear after this ships. No data migration required. |
| Public API drift | All `IndexingQueue` methods unchanged; this is a Pipeline-level refactor. |

## Test coverage

1. **Stage parallelism**: process 6 files, assert that parse, summarize,
   and embed call timestamps interleave (not sequential per file).
2. **Crash resumption from PARSED**: pre-populate a DB row with status
   PARSED + content; call `process_file`; assert parser is NOT called,
   summarize and embed ARE.
3. **Crash resumption from SUMMARIZED**: same with status SUMMARIZED;
   assert only embed runs.
4. **No mid-flight delete**: simulate crash by raising in summarize
   after parse; assert the parsed row survives in DB.
5. **Throughput**: corpus end-to-end wall time with concurrency 4/1/1
   should be lower than with 1/1/1 by a measurable margin.
6. **Existing tests unchanged**: the corpus suite from Goal 1 must
   keep passing.
