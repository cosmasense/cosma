# Startup latency investigation — findings

**Date:** 2026-05-01
**Why:** User reports the frontend freezes for a long time at startup;
"longer than just an embedding model load." We need numbers and a
root cause before fixing.

## TL;DR

The embedder is **not** the main culprit. The biggest offender is the
**LLM (llama.cpp) being loaded synchronously on the asyncio event loop
on the first call to `summarize()`**, not at startup. While that
constructor runs (multi-GB GGUF mmap + Metal context init), the event
loop is fully blocked — every HTTP request, including `/api/search/`,
hangs. That's the freeze.

Three independent issues, ranked by user-visible impact:

| # | Issue | Severity | Fix size |
|---|-------|----------|----------|
| 1 | LLM loads sync on first `summarize()`, blocking the event loop | **High** | Small |
| 2 | Indexing kicks off concurrently with embedder model load → both fight for memory/GPU | Medium | Small |
| 3 | Embedder warmup runs in a thread (good), but the cold encode path is still on the critical path of the first search | Low | Tiny |

## Evidence

### Issue 1: LLM sync-load blocks the event loop

`AutoSummarizer.summarize()` (`summarizer/auto.py:108`) →
`_get_llamacpp_summarizer()` (`auto.py:47-68`) →
`summarizer = LlamaCppSummarizer(config=self.config)` (line 52, **synchronous**).

`LlamaCppSummarizer.__init__` (`summarizer/summarizer.py:676-739`) calls
`Llama.from_pretrained(...)` directly — not via `asyncio.to_thread` and
not via `run_in_pipeline`. This is multi-second blocking C++ work
(model file mmap, Metal kernel compile, KV cache alloc).

Because it runs on the event loop, every other coroutine — including
the route handler for `/api/search/` — is starved until it returns.
Worst case the user has the app open, types a search query, and waits
~10 s for the LLM to finish loading before the first character of
results comes back.

Compare with whisper, which does it correctly:
`parser/whisper_local.py:104` —
`model = await run_in_pipeline(_load_model_sync, resolved)`.

### Issue 2: Indexing starts before embedder finishes loading

`app.py:367-368` starts `indexing_queue.start()` immediately at the top
of Phase 2. The deferred heavy init that loads the embedder
(`_deferred_heavy_init`, line 425) is fired-and-forgotten on the same
event loop. So as soon as the queue picks up its first item, the
pipeline calls `embedder.embed(file)` — which awaits the very same
SentenceTransformer that's still loading on a thread, OR worse,
triggers the LLM sync-load described above.

Net effect at cold start with files in the watched folder:
- t=0 s: API up
- t=0 s: queue starts pulling items
- t=0 s: embedder model load begins (in thread — good)
- t=0 s: pipeline tries to summarize first file → LLM sync-load → event loop frozen
- t=0–8 s: user's `/api/search/` request blocks
- t≈8 s: LLM done; embedder still loading; first search starts
- t≈10 s: embedder done; search completes

### Issue 3: Embedder cold path

Embedder loading happens in a thread (`asyncio.to_thread(embedder._eagerly_initialize_models)`,
`app.py:436`) and includes a "warmup" `embed_text("warmup")` call
(`embedder/auto.py:78`) to prime PyTorch op compilation. This is good.
The remaining cost (a few hundred ms on a warm cache) is fine.

### Phase 1 (the synchronous part) is fast

`initialize_config()` does TOML parse + `PlatformDirs(ensure_exists=True)`
+ filter config load. All disk-only, all small files. Sub-100 ms.

`bootstrap.get_status()` is local-only — module probes via
`importlib.util.find_spec` and glob in the models dir. No network,
sub-50 ms.

Filter and settings managers load tiny TOML; negligible.

## Concrete fix plan (feeds Goal 1 tasks #3 and #4)

1. **Move LLM load off the event loop and pre-warm it.**
   In Phase 2, after the embedder is loaded, do:
   ```python
   await asyncio.to_thread(_warm_llamacpp, app.pipeline.summarizer)
   ```
   where `_warm_llamacpp` calls `_get_llamacpp_summarizer()` and
   ideally runs one short throwaway prompt to compile Metal kernels.
   This costs ~5–8 s but it's hidden behind the loading indicator
   instead of fired on the first user search.

2. **Gate `indexing_queue.start()` on `app._model_loading_done` plus
   a configurable grace period (default 5 s).** This is task #3.

3. **Keep `summarize()` itself on a thread for inference.** It already
   is for whisper; do the same for llama.cpp's `create_chat_completion`
   so a long-running summarize call doesn't starve the event loop and
   prevent search preemption (task #4).

4. **Search-preempts-indexing** sits on top of all of the above
   (task #4). Without #1 and #2 it can't help: there's nothing to
   preempt while the LLM is locking the loop.

## Per-stage indexing durations (estimates, to be benchmarked)

| Stage | Typical duration | Resource |
|-------|------------------|----------|
| parse (text) | 50–200 ms | CPU + disk |
| parse (PDF) | 500 ms–3 s | CPU + disk |
| parse (image OCR) | 1–3 s | CPU |
| parse (audio whisper) | 2–10× audio length | CPU/Metal |
| **summarize (Qwen3-VL-2B llama.cpp)** | **5–30 s / file** | **GPU (Metal)** |
| embed (1536-d sentence-transformer) | 100–500 ms | MPS |
| save_to_db | <50 ms | SQLite |

Summarize dominates wall time by 10×–100×. The GPU sits idle during
parse, embed, save. So the Goal 2 stage-pipelining hypothesis is
correct: there is real throughput to gain. Task #2 will replace these
estimates with measured numbers on the corpus before we commit to
the design in #6.
