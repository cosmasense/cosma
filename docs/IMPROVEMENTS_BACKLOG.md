# SAIL improvements backlog

Updated 2026-05-01 after the LLM/DB/perf optimization sessions.

The items below are ranked by **value-per-effort**, not by what's
"interesting." Tier 1 is what I'd actually pick up next; Tier 4 and
"Skip" exist so future-me doesn't re-evaluate the same tradeoffs.

## Tier 1 — high value, not done

### 1. Speculative decoding via Qwen3-0.5B draft model

**Estimated win:** ~2× decode speedup on the LLM (per-file mean
~25 s → ~12 s real-AI). Biggest single perf number left on the table.

**How:** `llama-cpp-python`'s `Llama(draft_model=...)` parameter. Pair
Qwen3-0.5B-Instruct GGUF as the draft, Qwen3-VL-2B-Instruct as the
target. Apple's "Recurrent Drafter" research achieved 2.3× on Metal.

**Risk:** model pairing is finicky — the draft must share the target's
tokenizer. Need to test that Qwen3 0.5B and Qwen3-VL-2B agree at the
token level, or pick a different draft.

**Effort:** ~3 days (download/wire + bench + handle the stalled-draft
case).

### 2. Filename-match boost in search ranking

**Win:** big perceived-quality bump. Today, querying `tax2024.pdf`
matches files that *mention* "tax 2024" once before files literally
named that. Power users hit this in the first 5 minutes.

**How:** add a third signal to `HybridSearcher.search` alongside FTS
and semantic — exact-token filename match. RRF-fuse all three, with
filename signal weighted the highest.

**Effort:** ~2 hours.

### 3. Background re-indexing for fast-mode files

**Win:** `fast_mode=True` indexes everything quickly with 1-chunk
summaries; users want the full multi-chunk version eventually.
Background pass revisits fast-mode-indexed files when the queue is
otherwise idle. Lazy upgrade, no user action needed.

**How:** add a `coverage` field to the `files` table (`"head_only"` |
`"full"`) so the scheduler can find candidates. Scheduler rule: when
queue is empty + on-charger + idle for N minutes, enqueue oldest
`coverage="head_only"` row.

**Effort:** ~1 day.

### 4. Pre-extracted text caching by content_hash

**Win:** files that share `content_hash` (duplicates, same file in
different folders, screenshots saved twice) currently re-parse. Most
users have 10–20 % near-dup files.

**How:** check `SELECT content FROM files WHERE content_hash = ?`
before invoking the parser. If a row exists with the parsed text,
copy it. Already crash-safe via the persisted-progress pipeline.

**Effort:** ~1 day.

## Tier 2 — quality of life, worth doing eventually

### 5. Smart chunking for code / markdown

Sentence-based splitter is wrong for source code (split on function
/ class boundaries) and markdown (split on `##` headings). Per-type
chunkers produce dramatically better summaries for those file types.
~1 day per file type.

### 6. Filename + path search column

Separate from #2: index filename and path tokens in their own FTS
column with a higher BM25 weight. Lets `bench` query find
`random100bench.py` even if the file body never mentions "bench".
~2 hours.

### 7. `/api/status/health` with deep probes

Current `/api/status/` returns surface info. A proper health endpoint
that runs a DB write probe + tiny embedding + DB integrity check
gives the frontend a clean "is something wrong" signal. ~3 hours.

### 8. Hard path-allowlist guard

Configurable allowlist (default: `~/`), refuse to watch outside it.
Turns "I pointed the watcher at /System" support tickets into "you
can't do that." ~1 hour.

### 9. vec0 HNSW index past 10k files

Current linear scan is sub-2 ms at 1k vectors but scales O(N).
Switch to `vec0`'s HNSW index past a threshold. Stays fast at 100k+.
~2 days (schema migration).

### 10. DB content compression

`files.content` stores extracted text. zstd compression via SQLite's
`application_id` halves DB size for text-heavy corpora. ~1 day.

## Tier 3 — robustness, half-day each

### 11. Watcher coalescing for bursts

`git checkout` triggers ~1000 file events in 100 ms. Currently each
becomes a queue item. Coalesce into a single "directory revisit"
item that the discovery sweep handles in bulk.

### 12. Settings hot-reload

Non-disruptive settings (queue concurrency, fast_mode, model swap)
currently need a backend restart. Watch the TOML, apply at safe
points. Per-setting policy table needed.

### 13. Auto-recovery from SQLite corruption

`PRAGMA integrity_check` on startup. If not "ok", copy aside and
rebuild from the watched-folder file list. Rare but real.

### 14. Streaming summaries via SSE

Today: `file_summarizing → file_summarized`. Add intermediate
"chunk 2 of 5" events so the queue UI can show per-file progress.

### 15. Memory-pressure-aware queue throttling

Already have CPU + battery scheduler rules. Add "free RAM < 2 GB →
pause" so we don't fight the user's other apps when they're
memory-stressed.

## Tier 4 — marginal, would skip

- **Embedder model swap.** `e5-base-v2` is fine. Without a measured
  quality regression, not worth the migration.
- **`executemany` cross-file DB batching.** At our 1-2 file/sec
  real-world rate, per-row overhead is sub-1 % of wall time.
- **SSE event coalescing.** Already deferred; with cancel-in-flight
  the SSE storm is naturally truncated.
- **Frontend optimistic UI.** SSE roundtrip is < 100 ms; users don't
  perceive the wait.
- **Telemetry.** Privacy/policy can-of-worms for a single-user app.

## Categorically skip

- **MLX migration** — kept llama-cpp-python for the
  single-installation property.
- **Fine-tuning / LoRA** — high cost, marginal return for a generic
  file indexer.
- **MoE / mixture-of-experts** — overkill at our scale.
- **Beating Spotlight/MarkItDown's macOS-level concurrency limits**
  — already hit; the 63 % efficiency gap in random100bench is OS
  serialization, not our pipeline.

---

## My recommendation if asked

- **5 hours available:** #2 + #4 + #8. Three independent wins.
- **2 days available:** add #5 to the above. Biggest summary-quality
  lift for the file types users actually care about (code repos,
  tech notes).
- **3 days, want to chase the LLM ceiling:** #1 (speculative
  decoding). Single biggest perf number left.
- **Avoid right now:** anything in Tier 4 or "categorically skip" —
  diminishing returns territory.
