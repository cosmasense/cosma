"""
Embedder provider implementations: Online (LiteLLM/OpenAI) and Local (SentenceTransformers).
"""

from __future__ import annotations

import asyncio
import gc
import os
import threading
import time
from typing import TYPE_CHECKING

import litellm
import numpy as np

from cosma_backend.logging import get_logger

from .base import BaseEmbedder, EmbeddingProviderError

if TYPE_CHECKING:
    from cosma_backend.settings import EmbedderConfig

logger = get_logger(__name__)


class OnlineEmbedder(BaseEmbedder):
    """Embedder using online models via LiteLLM (OpenAI API)."""

    def __init__(self, config: EmbedderConfig | None = None, model: str | None = None, api_key: str | None = None, dimensions: int | None = None) -> None:
        """
        Initialize online embedder.

        Args:
            config: EmbedderConfig instance
            model: Model name override
            api_key: API key (default from env)
            dimensions: Embedding dimensions override
        """
        from cosma_backend.settings import EmbedderConfig as _EmbedderConfig
        self.config = config or _EmbedderConfig()
        self.model = model or self.config.model
        self.configured_dimensions = dimensions or self.config.dimensions

        # Initialize base class
        super().__init__(model_name=self.model, dimensions=self.configured_dimensions)

        # Set API key if provided
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        # Validate model compatibility
        if self.model == "text-embedding-3-small":
            if not (512 <= self.configured_dimensions <= 1536):
                msg = f"Dimensions must be between 512 and 1536 for {self.model}"
                raise ValueError(msg)

        logger.info("Online embedder initialized",
                   model=self.model,
                   dimensions=self.configured_dimensions)

    def is_available(self) -> bool:
        """Check if online embedder is available."""
        return bool(os.getenv("OPENAI_API_KEY"))

    def embed_text(self, text: str | list[str]) -> np.ndarray:
        """Generate embeddings using online model."""
        texts = self._validate_text(text)

        logger.debug("Generating embeddings",
                    model=self.model,
                    num_texts=len(texts),
                    dimensions=self.configured_dimensions)

        try:
            # Call litellm embedding endpoint
            response = litellm.embedding(
                model=self.model,
                input=texts,
                dimensions=self.configured_dimensions,
                timeout=60,
                max_retries=2
            )

            # Extract embeddings from response
            embeddings = []
            for item in response.data:
                embeddings.append(item["embedding"])

            # Convert to numpy array
            embeddings_array = np.array(embeddings, dtype=np.float32)

            # Return single vector if input was single text
            if isinstance(text, str):
                return embeddings_array[0]

            return embeddings_array

        except Exception as e:
            error_msg = f"Online embedding generation failed: {e!s}"
            logger.exception(error_msg, model=self.model)
            raise EmbeddingProviderError(error_msg)

    async def _embed_text_async(
        self, text: str | list[str], *, priority: bool = False,
    ) -> np.ndarray:
        """Truly async embedding generation using litellm async API."""
        del priority  # Online API has no GPU contention; priority is a no-op here.
        texts = self._validate_text(text)

        logger.debug("Generating embeddings async",
                    model=self.model,
                    num_texts=len(texts),
                    dimensions=self.configured_dimensions)

        try:
            # Call async litellm embedding endpoint
            response = await litellm.aembedding(
                model=self.model,
                input=texts,
                dimensions=self.configured_dimensions,
                timeout=60,
                max_retries=2
            )

            # Extract embeddings from response
            embeddings = []
            for item in response.data:
                embeddings.append(item["embedding"])

            # Convert to numpy array
            embeddings_array = np.array(embeddings, dtype=np.float32)

            # Return single vector if input was single text
            if isinstance(text, str):
                return embeddings_array[0]

            return embeddings_array

        except Exception as e:
            error_msg = f"Async online embedding generation failed: {e!s}"
            logger.exception(error_msg, model=self.model)
            raise EmbeddingProviderError(error_msg)


class LocalEmbedder(BaseEmbedder):
    """Embedder using local models via sentence-transformers.

    The SentenceTransformer model is loaded lazily on first use and can be
    unloaded to free memory via ``unload_model()``.
    """

    def __init__(self, config: EmbedderConfig | None = None, model_name: str | None = None, dimensions: int | None = None) -> None:
        from cosma_backend.settings import EmbedderConfig as _EmbedderConfig
        self.config = config or _EmbedderConfig()
        # Check availability without importing the heavy model
        try:
            import importlib.util
            self.sentence_transformers_available = importlib.util.find_spec("sentence_transformers") is not None
        except Exception:
            self.sentence_transformers_available = False
        if not self.sentence_transformers_available:
            logger.warning("sentence-transformers not installed, local embeddings unavailable")

        self.model_name = model_name or self.config.local_model
        self.configured_dimensions = dimensions or self.config.local_dimensions

        super().__init__(model_name=self.model_name, dimensions=self.configured_dimensions)

        if "Qwen3-Embedding" in self.model_name:
            if not (32 <= self.configured_dimensions <= 1024):
                msg = f"Dimensions must be between 32 and 1024 for {self.model_name}"
                raise ValueError(msg)

        # Model is loaded lazily on first embed call
        self.model = None
        self.last_used_at: float = 0.0

        # Search-priority gate. Two parts:
        #
        # 1. ``_encode_lock`` serializes every actual model.encode call.
        #    PyTorch's MPS backend serializes Metal command buffers
        #    internally too, but submitting from two Python threads at
        #    once still costs latency and can fight for the GIL during
        #    the pre/post-processing around the C call.
        #
        # 2. ``_search_waiters`` + ``_encode_cond`` let a search-query
        #    encode jump ahead of any indexing-side encode that hasn't
        #    started yet. Indexing encodes block at the gate while
        #    ``_search_waiters > 0``; search encodes increment the
        #    counter, run, then notify. We keep the gate cheap so the
        #    common no-search-pending case is just a counter read.
        #
        # The encode lock + the priority gate together mean:
        #   - Search query embeds never wait behind unrelated indexing
        #     batches (priority).
        #   - Two concurrent encodes never hit MPS at once (lock).
        #   - In-flight encodes still finish naturally — Python can't
        #     interrupt the C call — but the next encode picked up
        #     after release is whichever the priority rule says.
        self._encode_cond = threading.Condition()
        self._encode_lock = threading.Lock()
        self._search_waiters = 0

        logger.info("Local embedder created (model will load on first use)",
                     model=self.model_name, dimensions=self.configured_dimensions)

    @staticmethod
    def _mps_tripwire_path() -> "Path":
        """Marker file written before the first MPS encode and deleted
        if it succeeds. If a SIGABRT (uncatchable) takes the process
        down mid-encode, this file survives — the next launch detects
        it and skips MPS automatically. Lives next to the user data
        dir so it persists across process boundaries but doesn't
        pollute the user's home."""
        from pathlib import Path
        from platformdirs import PlatformDirs
        d = PlatformDirs("cosma", ensure_exists=False)
        return Path(d.user_data_dir) / ".mps-tripwire"

    def _ensure_loaded(self) -> None:
        """Load the SentenceTransformer model if not already loaded."""
        if self.model is not None:
            return
        if not self.sentence_transformers_available:
            raise EmbeddingProviderError("sentence-transformers not installed")
        from sentence_transformers import SentenceTransformer
        logger.info("Loading local embedding model on first use",
                     model=self.model_name, dimensions=self.configured_dimensions)
        # MPS embedder strategy — three-layer defense.
        #
        # Layer 1 — watermarks: PyTorch's MPS allocator auto-derives
        #   LOW from HIGH; HIGH=0.75 alone yields LOW=1.4 which is
        #   invalid and aborts MPS init. Set both explicitly.
        #
        # Layer 2 — MPS_FALLBACK: PyTorch's official escape hatch.
        #   Unsupported MPS ops automatically fall back to CPU
        #   instead of asserting. Catches both the "meta tensor" load
        #   error AND any sub-op that's not implemented for MPS.
        #
        # Layer 3 — eager attention: SentenceTransformer's underlying
        #   transformer model uses SDPA by default. PyTorch's MPS
        #   SDPA kernel has a known SIGABRT (C-level, can't be caught
        #   in Python) on certain model shapes — including
        #   intfloat/e5-base-v2 in our build. Forcing
        #   `attn_implementation="eager"` bypasses SDPA entirely.
        #   Slightly slower than SDPA on MPS but no crash. Per the
        #   transformers docs this is the recommended workaround
        #   while the MPS SDPA backend stabilizes.
        #
        # Layer 4 (post-load) — sanity-check encode that runs ONLY
        #   after the model is on MPS but before we hand it to
        #   anyone. If it still crashes the process, exit cleanly
        #   from the parent — the next launch goes CPU.
        #
        # If despite all this MPS still crashes a user's process,
        # they can flip COSMA_FORCE_CPU_EMBEDDER=1 and we skip MPS
        # entirely.
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.75")
        os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

        force_cpu = os.environ.get("COSMA_FORCE_CPU_EMBEDDER") == "1"
        # Tripwire check: if the previous launch crashed during the
        # MPS sanity-encode, the marker file is still present and we
        # skip MPS this run. Self-healing — user just needs to relaunch
        # once and we route around the broken op.
        if not force_cpu and self._mps_tripwire_path().exists():
            logger.warning(
                "MPS tripwire present from a previous crashed launch — "
                "forcing CPU embedder this session. Delete "
                f"{self._mps_tripwire_path()} to retry MPS.",
            )
            force_cpu = True
        device = "cpu"
        if not force_cpu:
            try:
                import torch
                if torch.backends.mps.is_available():
                    device = "mps"
            except Exception:
                pass

        # Pass model_kwargs through so transformers loads with
        # eager attention on MPS. Sentence-transformers >=3.0 supports
        # `model_kwargs`; older versions silently ignore it.
        model_kwargs: dict = {}
        if device == "mps":
            model_kwargs["attn_implementation"] = "eager"

        def _load(target_device: str, kw: dict) -> "SentenceTransformer":
            try:
                return SentenceTransformer(
                    self.model_name, device=target_device,
                    model_kwargs=kw if kw else None,
                )
            except TypeError:
                # Older sentence-transformers without model_kwargs.
                return SentenceTransformer(self.model_name, device=target_device)

        try:
            self.model = _load(device, model_kwargs)
            actual_device = device
        except Exception as e:
            if device == "mps":
                logger.warning(
                    "MPS load failed, falling back to CPU. "
                    "Search will be 2-3× slower than ideal.",
                    error=str(e),
                )
                self.model = _load("cpu", {})
                actual_device = "cpu"
            else:
                raise

        logger.info(
            "Local embedding model loaded",
            model=self.model_name,
            device=actual_device,
            attn_impl=model_kwargs.get("attn_implementation", "default"),
        )

        # Tripwire: write a marker file BEFORE the first encode on MPS.
        # If the encode SIGABRTs the process (uncatchable in Python),
        # the marker survives. The next launch sees the unfinished
        # marker and skips MPS — the user gets CPU automatically
        # instead of a restart loop.
        if actual_device == "mps":
            tripwire_path = self._mps_tripwire_path()
            try:
                tripwire_path.parent.mkdir(parents=True, exist_ok=True)
                tripwire_path.write_text("warmup-in-progress")
            except OSError:
                pass  # tripwire is best-effort
            try:
                # Tiny encode to provoke any MPS crash NOW, before
                # the model serves real work. If we survive this,
                # MPS is good for the rest of the session.
                _ = self.model.encode(
                    ["sanity"], normalize_embeddings=True, show_progress_bar=False,
                )
                # Only clear tripwire if we got past the encode cleanly.
                try:
                    tripwire_path.unlink(missing_ok=True)
                except OSError:
                    pass
                logger.info("MPS sanity-encode passed", model=self.model_name)
            except Exception as e:
                # Python-catchable path: log and fall back gracefully.
                logger.warning(
                    "MPS sanity-encode failed, switching to CPU mid-load",
                    error=str(e),
                )
                self.model = _load("cpu", {})
                actual_device = "cpu"
                try:
                    tripwire_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self.last_used_at = 0.0
            from cosma_backend.utils.memory import release_memory
            release_memory()
            logger.info("Local embedding model unloaded", model=self.model_name)

    def is_available(self) -> bool:
        """Check if local embedder can be used (model loads lazily)."""
        return self.sentence_transformers_available

    def _encode_with_priority(
        self, texts: list[str], *, priority: bool,
    ) -> np.ndarray:
        """Run model.encode while honoring the search-priority gate.

        priority=True (search path): increment the waiter count BEFORE
        we acquire the encode lock so any indexing thread parked at
        the gate sees us and stays parked. Then take the encode lock
        and run.

        priority=False (indexing path): wait at the gate until the
        waiter count is 0, then acquire the encode lock and run.

        Both paths run model.encode under ``_encode_lock`` so two
        threads never submit to MPS at once.
        """
        if priority:
            with self._encode_cond:
                self._search_waiters += 1
        else:
            with self._encode_cond:
                while self._search_waiters > 0:
                    self._encode_cond.wait()
        try:
            with self._encode_lock:
                return self.model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
        finally:
            if priority:
                with self._encode_cond:
                    self._search_waiters -= 1
                    if self._search_waiters == 0:
                        self._encode_cond.notify_all()

    def embed_text(
        self, text: str | list[str], *, priority: bool = False,
    ) -> np.ndarray:
        """Generate embeddings using local model."""
        self._ensure_loaded()
        self.last_used_at = time.time()

        texts = self._validate_text(text)

        logger.debug("Generating local embeddings",
                    model=self.model_name,
                    num_texts=len(texts),
                    dimensions=self.configured_dimensions)

        try:
            embeddings = self._encode_with_priority(texts, priority=priority)

            # Truncate to configured dimensions if needed
            if embeddings.shape[1] > self.configured_dimensions:
                embeddings = embeddings[:, :self.configured_dimensions]

            # Convert to float32
            embeddings = embeddings.astype(np.float32)

            # Return single vector if input was single text
            if isinstance(text, str):
                return embeddings[0]

            return embeddings

        except Exception as e:
            error_msg = f"Local embedding generation failed: {e!s}"
            logger.exception(error_msg, model=self.model_name)
            raise EmbeddingProviderError(error_msg)

    async def _embed_text_async(
        self, text: str | list[str], *, priority: bool = False,
    ) -> np.ndarray:
        """Generate embeddings using local model with async support.

        ``priority=True`` is set by the search path so a user query
        jumps ahead of any indexing-side encode parked at the gate.
        See ``_encode_with_priority``.
        """
        # Model load is moved off the event loop. Phase 2 of app
        # startup pre-loads the embedder in a thread so the typical
        # path here is a no-op (``self.model`` is already set).
        if self.model is None:
            await asyncio.to_thread(self._ensure_loaded)
        self.last_used_at = time.time()

        texts = self._validate_text(text)

        logger.debug("Generating local embeddings async",
                    model=self.model_name,
                    num_texts=len(texts),
                    dimensions=self.configured_dimensions)

        try:
            # Run on the default asyncio executor (asyncio.to_thread).
            # Crucially NOT the pipeline executor — that pool is shared
            # with markitdown parses, and a search query embed used to
            # queue behind a multi-second PDF parse. The encode itself
            # is cheap; sharing the pipeline pool was the bottleneck.
            embeddings = await asyncio.to_thread(
                self._encode_with_priority, texts, priority=priority,
            )

            # Truncate to configured dimensions if needed
            if embeddings.shape[1] > self.configured_dimensions:
                embeddings = embeddings[:, :self.configured_dimensions]

            # Convert to float32
            embeddings = embeddings.astype(np.float32)

            # Return single vector if input was single text
            if isinstance(text, str):
                return embeddings[0]

            return embeddings

        except Exception as e:
            error_msg = f"Async local embedding generation failed: {e!s}"
            logger.exception(error_msg, model=self.model_name)
            raise EmbeddingProviderError(error_msg)
