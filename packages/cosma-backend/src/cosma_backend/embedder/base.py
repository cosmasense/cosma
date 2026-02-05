"""
Base embedder class and exceptions.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

import numpy as np

from cosma_backend.logging import get_logger

logger = get_logger(__name__)


class EmbedderError(Exception):
    """Base exception for embedder errors."""


class EmbeddingProviderError(EmbedderError):
    """Exception for embedding provider-specific errors."""


class BaseEmbedder(ABC):
    """Abstract base class for text embedders."""

    def __init__(self, model_name: str, dimensions: int) -> None:
        """
        Initialize embedder with model specifications.

        Args:
            model_name: Name of the embedding model
            dimensions: Dimension of the output embeddings
        """
        self.model_name = model_name
        self.dimensions = dimensions
        logger.info("Initializing embedder", model=model_name, dimensions=dimensions)

    @abstractmethod
    def embed_text(self, text: str | list[str]) -> np.ndarray:
        """
        Generate embeddings for text input.

        Args:
            text: Single text or list of texts to embed

        Returns:
            Numpy array of embeddings (single vector or matrix)
        """

    async def embed_text_async(self, text: str | list[str]) -> np.ndarray:
        """
        Async version of embed_text that runs in a thread pool.

        Args:
            text: Single text or list of texts to embed

        Returns:
            Numpy array of embeddings (single vector or matrix)
        """
        # Check if this is an OnlineEmbedder with direct async support
        if hasattr(self, '_embed_text_async'):
            return await self._embed_text_async(text)

        # Fallback to thread pool for legacy implementations
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_text, text)

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this embedder is available for use."""

    def _validate_text(self, text: str | list[str]) -> list[str]:
        """Validate and normalize text input."""
        texts = [text] if isinstance(text, str) else text

        # Filter out empty texts
        valid_texts = [t for t in texts if t and t.strip()]

        if not valid_texts:
            msg = "No valid text provided for embedding"
            raise ValueError(msg)

        return valid_texts
