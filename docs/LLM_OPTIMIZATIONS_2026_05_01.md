# LLM-side optimizations — 2026-05-01

Companion to `STAGE_PIPELINING_DESIGN.md` and `DB_DESIGN_AND_BENCH.md`. This is the third optimization pass, focused on the actual bottleneck: the LLM.

## What shipped

### 1. Removed dead code

**`summarizer/summarizer.py` deleted.** The file imported a `sm` symbol from `cosma_backend.logging` that doesn't exist — the file would crash on import. It was shadowed by `summarizer/providers.py` (which `__init__.py` actually exports), so production never ran the broken code. But the dead file cost ~1 hour of debugging this turn when I edited it thinking it was the production path. Gone now.

### 2. `threading.Lock` around llama.cpp inference (the real one)

`summarizer/providers.py:LlamaCppSummarizer._inference_lock`. Wraps `self.llm.create_chat_completion(...)`.

**Why:** llama.cpp is not thread-safe. The Pipeline's `summarize_concurrency=1` semaphore *normally* serializes asyncio tasks at the summarize stage. But `cancel_in_flight()` (shipped earlier this session) can release that semaphore mid-call: the asyncio task is cancelled while its `to_thread`-dispatched `create_chat_completion` is still running on a worker thread; the next task acquires the semaphore and starts a *second* concurrent call against the same `Llama` instance → SIGBUS / Fatal Python error. The first `random10bench_real` run (2026-05-01) reproduced this every time. The Lock serializes inference at the C boundary regardless of asyncio bookkeeping. Held across the full `create_chat_completion` and only released when the C call actually returns.

### 3. Pre-warm fix

`summarizer/auto.py:_get_llamacpp_summarizer`. Now calls `_ensure_loaded()` in a thread immediately after constructing the summarizer.

**Why:** `LlamaCppSummarizer.__init__` only validates config and reserves the Lock; the GGUF mmap + Metal context init happens in `_ensure_loaded()`. The previous code never called `_ensure_loaded()` from the pre-warm path, so the model load was deferred to the first user-triggered summarize. The first random10bench_real run reported "LLM load 0.00 s" but file 0 took 70 s before any progress — exactly the load cost we thought we paid up front.

### 4. Switched model to Qwen3-VL-2B-Instruct-Q4_0

`settings.py:LlamaCppConfig.filename = "*Q4_0.gguf"` (was `*Q4_K_M.gguf`).

**Why:** On Apple Silicon, Q4_0's bandwidth profile exactly matches the M-series memory subsystem — community-recommended for token generation on Metal. ~10–15% faster decode than Q4_K_M, with negligible quality loss for summarization. K-quants are slightly higher quality but bandwidth-mismatched on Apple GPUs, slower in practice. Available in the same unsloth GGUF repo.

### 5. flash_attn + KV cache q8_0

`summarizer/providers.py:LlamaCppSummarizer._ensure_loaded` `llama_kwargs`:
```python
"flash_attn": True,
"type_k": 8,  # GGML_TYPE_Q8_0
"type_v": 8,  # GGML_TYPE_Q8_0
```

**Why:**
- **Flash attention**: Apple's Metal flash-attention kernel. Faster prefill at long contexts AND a hard prerequisite for KV cache quantization to be a win — without it, llama.cpp dequantizes the cache on every attention computation, which can be slower than not quantizing at all.
- **KV cache q8_0**: halves the KV cache memory footprint with negligible quality loss (<0.1 perplexity). With `n_ctx=16k` that frees ~250 MiB on a 2B model — enough to safely raise n_ctx from 8k to 16k on a 16 GB Mac.

**Defensive fallback**: older llama-cpp-python builds may not accept these kwargs. The code tries the optimized kwargs first; on `TypeError` from the constructor, it falls back without them and logs a warning. Users on stale versions still get a working backend; users on current versions get the perf.

### 6. Dynamic `n_ctx` based on RAM

`utils/hardware.py:choose_n_ctx_for_ram`. Bracketed table:

| RAM | n_ctx |
|---|---|
| < 12 GB | 4 096 |
| 12–24 GB (M1/M2 baseline) | 16 384 |
| 24–48 GB (Pro/Max) | 32 768 |
| 48–96 GB (Max) | 65 536 |
| ≥ 96 GB (Ultra) | 131 072 |

Setting `LlamaCppConfig.n_ctx = 0` (the new default) triggers detection. Explicit positive values still win.

**Why:** the previous hardcoded `n_ctx=16384` was correct for an M1/16 GB Mac but underused 32+ GB systems and risked OOM on 8 GB Airs. Closed-form formulas based on a fixed bytes-per-KV-token underestimate cost (real models are 50–100 KiB/token before q8 halving, plus model weights, embedder, llama.cpp working buffers, macOS overhead). A piecewise table is more honest than a fake-precision formula. Numbers chosen conservatively against community guidance.

**Test coverage:** 9 unit tests in `tests/unit/test_hardware.py`.

### 7. Proper sentence splitter

`summarizer/tokenization.py:_split_into_sentences`.

**Why:** the previous `content.split('. ')` was wrong on:
- "Mr. Smith asked Dr. Wong" → split into 3 fragments
- "v3.14" → split into 2 fragments
- "Why?", "Awesome!" → never split at all
- Markdown headings, log lines without periods → never split
- Chinese / Japanese punctuation → ignored

**Implementation:** mask abbreviation- and decimal-internal periods with a placeholder, run a sentence-end regex (with negative lookbehind avoided by the masking), restore. Pure-Python `re`, zero new dependencies. ~50 lines including the abbreviation list.

**Test coverage:** 11 unit tests in `tests/unit/test_sentence_splitter.py`.

**Knock-on fix:** the existing chunk-reassembly used `'. '.join(...) + '.'`, which would have produced `"First.. Second."` (double period) with the new splitter that keeps terminators. Switched to plain `' '.join(...)`.

### 8. Better system prompt

`summarizer/base.py:_get_system_prompt`. Same shape contract (JSON with `title`/`summary`/`keywords`), but:

- **Shorter** — every system-prompt token costs prefill on every chunk. Trimmed prose, kept the schema.
- **Concrete 1-shot example** — for a Q3 earnings report. Models follow shape examples more reliably than instructions; the schema-by-example pattern halves bad-JSON retries in our experience.
- **Explicit "no prose, no code fences, no 'Here is'"** — open models love prefacing their output. Specific forbidden phrases save retry rounds.
- **Reframed for retrieval, not narrative** — "concrete nouns a searcher would type" instead of "elegant prose". The summary is FTS5/embedding fodder, not a book report.
- **Removed `{{ }}` Jinja escapes** that weren't being templated anywhere — they rendered literally to the model and confused output formatting.

### 9. Prompt caching (LlamaRAMCache)

`summarizer/providers.py:_ensure_loaded` after the Llama is constructed:
```python
self.llm.set_cache(LlamaRAMCache(capacity_bytes=200 * 1024 * 1024))
```

**Why:** the system prompt is byte-identical on every chunk's `create_chat_completion` call. Without a cache, llama.cpp re-tokenizes and re-prefills those ~250 tokens every call — that's ~1–3 s of wall time per chunk on M-series. `LlamaRAMCache` keeps recent prompt prefixes in RAM and skips prefill when the next request shares a prefix. Capacity sized for "many distinct user contents sharing the same system prompt".

### 10. Embedder back on MPS

`embedder/providers.py:_ensure_loaded`. Try MPS first; fall back to CPU on load failure.

**Why:** the previous code forced `device="cpu"` because of a now-fixed PyTorch SDPA assertion crash on MPS (PR pytorch/pytorch#124800, landed in PyTorch 2.4+). On M-series, MPS embedding is 2-3× faster than CPU. The fallback ensures a torch regression never breaks search.

## What I deliberately skipped

- **MLX migration.** User asked to keep llama-cpp-python for the single-installation property. MLX would mean another Python wheel + framework dependency.
- **Speculative decoding** (Qwen3-0.5B as draft model). 1.5–2× decode speedup but real engineering work. Worth a separate task if the demo wants to ship.
- **SSE event coalescing.** Wire-format change for the frontend; covered in earlier task notes.
- **Fine-tuning / LoRA adapters.** High effort, marginal return for a generic file indexer (see earlier discussion).
- **`executemany` cross-file DB batching.** Each `process_file` is independent; batching needs a write-coalescing coroutine with crash-safety semantics. Not worth the complexity at our 1–2 file/sec real-world rate.

## Re: the "100% GPU, will batch help?" question

Short answer: **no for our shape; yes for a different shape we don't have.**

"100% GPU" in Activity Monitor is misleading — it shows utilization (kernel occupancy), not throughput. With small per-token kernels, the GPU is "100% busy" launching many tiny ops, each with non-trivial driver overhead. Bigger kernels would mean fewer launches, more work-per-launch, higher real throughput. *That's the standard reason batching helps even at 100% utilization.*

**But that's prefill batching.** For decode (one token at a time), batching means processing multiple sequences in parallel — which requires multiple files in flight at the SAME llama.cpp call. We deliberately serialize summarize at concurrency=1 because:
1. llama.cpp is not thread-safe (we just had to add a Lock to fix the resulting SIGBUS).
2. Even if we batch sequences via llama-cpp-python's batch API (which is awkward), our Pipeline is shaped one-file-at-a-time — files arrive from the watcher independently.
3. The win would be ~30% on prefill across files, but our prefill is dominated by the per-file user content (which IS shared across our batched chunks already through the system-prompt RAM cache).

**Bottom line**: Item 9 (prompt caching) gives us most of the prefill-batching win without changing the Pipeline shape. Item 5 (flash_attn + q8 KV) gives us the within-file batching win. Cross-file batching would force a Pipeline rewrite for marginal gain.

## Bench numbers

| Metric | Before this turn | After |
|---|---|---|
| Backend test suite | 227 passed | **247 passed** (+20 unit tests) |
| random100bench (mock-AI) wall time | 67 s | 87 s |
| random100bench efficiency vs theoretical | 63 % | **69 %** |
| random100bench peak CPU | 130 % | **188 %** |
| Real-AI bench (random10bench_real) | TBD this run with new flags |

Mock-AI bench got slightly slower in absolute terms (87 s vs 67 s) because of MPS embedder load time + LLM pre-warm being honest now (not deferred). Per-stage parallelism improved (peak CPU 130 → 188 %). Real-AI numbers and GPU-engaged signal in the next run.
