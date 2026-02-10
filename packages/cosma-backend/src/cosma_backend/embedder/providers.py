"""
Embedder provider implementations: Online (LiteLLM/OpenAI) and Local (SentenceTransformers).
"""

from __future__ import annotations

import asyncio
import gc
import os
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

    async def _embed_text_async(self, text: str | list[str]) -> np.ndarray:
        """Truly async embedding generation using litellm async API."""
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

        logger.info("Local embedder created (model will load on first use)",
                     model=self.model_name, dimensions=self.configured_dimensions)

    def _ensure_loaded(self) -> None:
        """Load the SentenceTransformer model if not already loaded."""
        if self.model is not None:
            return
        if not self.sentence_transformers_available:
            raise EmbeddingProviderError("sentence-transformers not installed")
        from sentence_transformers import SentenceTransformer
        logger.info("Loading local embedding model on first use",
                     model=self.model_name, dimensions=self.configured_dimensions)
        self.model = SentenceTransformer(self.model_name)
        logger.info("Local embedding model loaded", model=self.model_name)

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

    def embed_text(self, text: str | list[str]) -> np.ndarray:
        """Generate embeddings using local model."""
        self._ensure_loaded()
        self.last_used_at = time.time()

        texts = self._validate_text(text)

        logger.debug("Generating local embeddings",
                    model=self.model_name,
                    num_texts=len(texts),
                    dimensions=self.configured_dimensions)

        try:
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False
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
            error_msg = f"Local embedding generation failed: {e!s}"
            logger.exception(error_msg, model=self.model_name)
            raise EmbeddingProviderError(error_msg)

    async def _embed_text_async(self, text: str | list[str]) -> np.ndarray:
        """Generate embeddings using local model with async support."""
        self._ensure_loaded()
        self.last_used_at = time.time()

        texts = self._validate_text(text)

        logger.debug("Generating local embeddings async",
                    model=self.model_name,
                    num_texts=len(texts),
                    dimensions=self.configured_dimensions)

        try:
            # Generate embeddings asynchronously using asyncio.to_thread
            embeddings = await asyncio.to_thread(
                self.model.encode,
                texts,
                normalize_embeddings=True,
                show_progress_bar=False
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
