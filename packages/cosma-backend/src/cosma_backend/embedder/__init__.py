"""
Embedder module for generating text embeddings for semantic search.

This module provides multiple embedder implementations:
- OnlineEmbedder: Online models via LiteLLM/OpenAI API
- LocalEmbedder: Local models via sentence-transformers
- AutoEmbedder: Automatic provider selection with fallback
"""

from .base import (
    BaseEmbedder,
    EmbedderError,
    EmbeddingProviderError,
)
from .providers import (
    LocalEmbedder,
    OnlineEmbedder,
)
from .auto import (
    AutoEmbedder,
    generate_embedding,
    get_available_embedders,
    get_embedder_info,
)

__all__ = [
    # Base classes and exceptions
    "BaseEmbedder",
    "EmbedderError",
    "EmbeddingProviderError",
    # Provider implementations
    "OnlineEmbedder",
    "LocalEmbedder",
    # Auto embedder
    "AutoEmbedder",
    # Convenience functions
    "generate_embedding",
    "get_available_embedders",
    "get_embedder_info",
]
