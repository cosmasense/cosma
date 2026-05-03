"""Hardware-aware tuning helpers.

Picks runtime parameters that scale with the user's machine — RAM,
core count — instead of using a single hardcoded value that's wrong
on small Macs (8 GB Air) AND wrong on big Macs (Studio M-Ultra).
"""

from __future__ import annotations

import psutil


def detect_total_ram_gib() -> float:
    """Total physical RAM in GiB (rounded). Used as the baseline for
    KV-cache and context-length sizing."""
    return psutil.virtual_memory().total / 1024 ** 3


def choose_n_ctx_for_ram(total_ram_gib: float) -> int:
    """Pick a llama.cpp ``n_ctx`` value that won't OOM at runtime.

    Bracketed table-lookup, intentionally conservative. Closed-form
    formulas based on a fixed bytes-per-KV-token underestimate cost
    (real models are 50–100 KiB/token before q8 halving, plus the
    weights themselves, the embedder, llama.cpp working buffers, and
    macOS overhead). A piecewise table is more honest than pretending
    we have a precise model.

    The numbers below are the values random10bench_real (and our
    common-sense reading of llama-cpp-python community guidance) say
    are safe for Qwen3-VL-2B with q8 KV cache + flash_attn enabled.
    """
    if total_ram_gib < 12:
        # 8 GB Air: model weights (~1.5 GiB) + embedder (~600 MiB)
        # + macOS already eat half. Stay tight.
        return 4096
    if total_ram_gib < 24:
        # 16 GB Mac (M1 baseline). q8 KV halves the per-token cost
        # enough that 16k is now safely affordable; the previous
        # config was 16384 too which we know works.
        return 16384
    if total_ram_gib < 48:
        return 32768
    if total_ram_gib < 96:
        return 65536
    # Studio M-Ultra / future big boxes — cap at model context max
    # so we don't ask the GGUF for more than it'll honor.
    return 131072


def resolved_n_ctx(configured: int) -> int:
    """If the user (or default) set n_ctx=0, pick one based on RAM.
    Anything explicit (>0) wins. Returns the value to actually pass
    to llama.cpp."""
    if configured > 0:
        return configured
    return choose_n_ctx_for_ram(detect_total_ram_gib())
