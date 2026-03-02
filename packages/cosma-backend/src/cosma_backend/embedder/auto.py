"""
AutoEmbedder - automatic provider selection with fallback support.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

import numpy as np

from cosma_backend.logging import get_logger
from cosma_backend.models import File
from cosma_backend.models.status import ProcessingStatus

from .base import EmbedderError
from .providers import LocalEmbedder, OnlineEmbedder

if TYPE_CHECKING:
    from cosma_backend.settings import EmbedderConfig

logger = get_logger(__name__)


class AutoEmbedder:
    """
    Automatic embedder that selects the best available provider.

    Tries providers in order of preference:
    1. User-specified provider
    2. Online models (OpenAI)
    3. Local models (fallback)
    """

    def __init__(self, config: EmbedderConfig | None = None, preferred_provider: str | None = None) -> None:
        """
        Initialize auto embedder.

        Args:
            config: EmbedderConfig instance
            preferred_provider: Preferred provider ('local', 'online')
        """
        from cosma_backend.settings import EmbedderConfig as _EmbedderConfig
        self.config = config or _EmbedderConfig()
        self.preferred_provider = preferred_provider or self.config.provider
        logger.debug("Preferred provider", og=preferred_provider, provider=self.preferred_provider)
        self.embedders: dict = {}

        logger.debug("AutoEmbedder initializing",
                    preferred_provider=self.preferred_provider)

        # Eagerly initialize models based on preferred provider
        self._eagerly_initialize_models()

    def _eagerly_initialize_models(self) -> None:
        """Initialize embedding models based on provider preference - eager for local, lazy for online."""
        logger.debug("Initializing embedding models")

        if self.preferred_provider == "local":
            # Create local embedder and load model immediately for instant search
            logger.debug("Initializing local embedding provider (eager load)")
            local_embedder = self._get_local_embedder()
            if local_embedder:
                local_embedder._ensure_loaded()
                logger.debug("Local embedder loaded",
                            model=local_embedder.model_name,
                            dimensions=local_embedder.dimensions)
            else:
                logger.warning("Local embedder failed to initialize")

            # Check online availability but don't initialize (lazy loading)
            logger.debug("Checking online embedding provider availability (lazy loading)")
            online_available = self._check_online_availability()
            if online_available:
                logger.debug("Online embedder available as fallback (will load on first use)")
            else:
                logger.warning("Online embedder not available - check API keys")
        else:
            # For online preference, check availability but don't initialize (lazy loading)
            logger.debug("Checking online embedding provider availability (lazy loading)")
            online_available = self._check_online_availability()
            if online_available:
                logger.debug("Online embedder ready (will load on first use)",
                            provider="online")
            else:
                logger.warning("Online embedder not available - check API keys")

            # Skip local model initialization when user explicitly chose online
            logger.debug("Skipping local embedding model initialization (online provider preferred)")
            logger.info("To use local models as fallback, set EMBEDDING_PROVIDER=local")

        # Summary of initialization strategy
        if self.preferred_provider == "local":
            logger.info("AutoEmbedder configured: LOCAL model loaded, ONLINE models lazy-loaded")
        else:
            logger.info("AutoEmbedder configured: ONLINE models only (LOCAL models skipped)")

    def _check_online_availability(self) -> bool:
        """Check if online embedder is available without initializing it."""
        try:
            return bool(os.getenv("OPENAI_API_KEY"))
        except Exception:
            return False

    def _check_local_availability(self) -> bool:
        """Check if local embedder is available without initializing it."""
        try:
            import importlib.util
            return importlib.util.find_spec("sentence_transformers") is not None
        except Exception:
            return False

    def _get_online_embedder(self) -> OnlineEmbedder | None:
        """Get or create online embedder if available."""
        if "online" not in self.embedders:
            try:
                embedder = OnlineEmbedder(config=self.config)
                if embedder.is_available():
                    self.embedders["online"] = embedder
                    logger.info("Online embedder available")
                else:
                    logger.debug("Online embedder not available")
                    return None
            except Exception as e:
                logger.debug("Failed to create online embedder", error=str(e))
                return None

        return self.embedders.get("online")

    def _get_local_embedder(self) -> LocalEmbedder | None:
        """Get or create local embedder if available."""
        if "local" not in self.embedders:
            try:
                embedder = LocalEmbedder(config=self.config)
                if embedder.is_available():
                    self.embedders["local"] = embedder
                    logger.info("Local embedder available")
                else:
                    logger.debug("Local embedder not available")
                    return None
            except Exception as e:
                logger.warning("Failed to create local embedder", error=str(e))
                return None

        return self.embedders.get("local")

    def embed_text(self, text: str | list[str]) -> np.ndarray:
        """
        Generate embeddings using the best available provider.

        Args:
            text: Text or list of texts to embed

        Returns:
            Numpy array of embeddings

        Raises:
            EmbedderError: If no embedders are available or all fail
        """
        providers = []

        # Build provider list based on preference
        if self.preferred_provider == "online":
            # When online is explicitly preferred, don't initialize local models as fallback
            providers = [self._get_online_embedder()]
        elif self.preferred_provider == "local":
            providers = [self._get_local_embedder(), self._get_online_embedder()]
        else:  # default: local first
            providers = [self._get_local_embedder(), self._get_online_embedder()]

        logger.debug("All available providers", providers=providers)

        # Try each provider
        for embedder in providers:
            if embedder:
                try:
                    logger.debug("Attempting embedding generation",
                                provider=type(embedder).__name__)
                    return embedder.embed_text(text)
                except Exception as e:
                    logger.warning("Embedder failed, trying next provider",
                                 provider=type(embedder).__name__,
                                 error=str(e))
                    continue

        error_msg = "All embedding providers failed or are unavailable"
        logger.error(error_msg, preferred_provider=self.preferred_provider)
        raise EmbedderError(error_msg)

    async def embed_text_async(self, text: str | list[str]) -> np.ndarray:
        """
        Async version of embed_text with fallback providers.

        Args:
            text: Text or list of texts to embed

        Returns:
            Numpy array of embeddings

        Raises:
            EmbedderError: If no embedders are available or all fail
        """
        providers = []

        # Build provider list based on preference
        if self.preferred_provider == "online":
            providers = [self._get_online_embedder()]
        elif self.preferred_provider == "local":
            providers = [self._get_local_embedder(), self._get_online_embedder()]
        else:  # default: local first
            providers = [self._get_local_embedder(), self._get_online_embedder()]

        # Try each provider
        for embedder in providers:
            if embedder:
                try:
                    logger.debug("Attempting async embedding generation",
                                provider=type(embedder).__name__)
                    return await embedder.embed_text_async(text)
                except Exception as e:
                    logger.warning("Async embedder failed, trying next provider",
                                 provider=type(embedder).__name__,
                                 error=str(e))
                    continue

        error_msg = "All embedding providers failed or are unavailable"
        logger.error(error_msg, preferred_provider=self.preferred_provider)
        raise EmbedderError(error_msg)

    @property
    def last_used_at(self) -> float:
        """Return the most recent last_used_at across local embedders."""
        local = self.embedders.get("local")
        if local and hasattr(local, "last_used_at"):
            return local.last_used_at
        return 0.0

    def is_model_loaded(self) -> bool:
        """Check if the local embedding model is currently loaded in memory."""
        local = self.embedders.get("local")
        return local is not None and hasattr(local, "model") and local.model is not None

    async def unload_models(self) -> None:
        """Unload local embedding model to free memory."""
        local = self.embedders.get("local")
        if local and hasattr(local, "unload_model"):
            local.unload_model()

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the currently available model."""
        if self.preferred_provider == "online" and self._get_online_embedder():
            embedder = self._get_online_embedder()
            return {
                "provider": "online",
                "model": embedder.model_name,
                "dimensions": embedder.dimensions
            }
        if self.preferred_provider == "local" and self._get_local_embedder():
            embedder = self._get_local_embedder()
            return {
                "provider": "local",
                "model": embedder.model_name,
                "dimensions": embedder.dimensions
            }
        # Auto mode - return first available (respecting online-only preference)
        if self._get_online_embedder():
            embedder = self._get_online_embedder()
            return {
                "provider": "online",
                "model": embedder.model_name,
                "dimensions": embedder.dimensions
            }
        # Only try local if not explicitly using online-only
        if self.preferred_provider != "online" and self._get_local_embedder():
            embedder = self._get_local_embedder()
            return {
                "provider": "local",
                "model": embedder.model_name,
                "dimensions": embedder.dimensions
            }

        return {
            "provider": None,
            "model": None,
            "dimensions": None
        }

    def get_available_providers(self) -> list[str]:
        """Get list of available providers (respects online-only preference)."""
        providers = []

        if self._get_online_embedder():
            providers.append("online")

        # Only check local if not explicitly using online-only
        if self.preferred_provider != "online" and self._get_local_embedder():
            providers.append("local")

        return providers

    def _prepare_embedding_text(self, file: File) -> str:
        """
        Prepare text for embedding generation.

        Args:
            file: File metadata to prepare text from

        Returns:
            Text prepared for embedding
        """
        parts = []

        # Add title
        if file.title:
            parts.append(f"Title: {file.title}")

        # Add summary
        if file.summary:
            parts.append(f"Summary: {file.summary}")

        # Add keywords
        if file.keywords:
            parts.append(f"Keywords: {', '.join(file.keywords)}")

        return " ".join(parts)

    async def embed(self, file: File) -> None:
        """Generate and attach embeddings to a file."""
        embedding_text = self._prepare_embedding_text(file)
        embedding = await self.embed_text_async(embedding_text)

        model_info = self.get_model_info()

        file.embedding = embedding
        file.embedding_model = model_info["model"]
        file.embedding_dimensions = model_info["dimensions"]
        file.embedded_at = datetime.now(timezone.utc)

        file.status = ProcessingStatus.COMPLETE


# Convenience functions for easier usage
def generate_embedding(text: str | list[str], provider: str | None = None) -> np.ndarray:
    """
    Convenience function to generate embeddings.

    Args:
        text: Text or list of texts to embed
        provider: Preferred embedding provider

    Returns:
        Numpy array of embeddings
    """
    embedder = AutoEmbedder(preferred_provider=provider)
    return embedder.embed_text(text)


def get_available_embedders() -> list[str]:
    """Get list of available embedding providers."""
    embedder = AutoEmbedder()
    return embedder.get_available_providers()


def get_embedder_info() -> dict[str, Any]:
    """Get information about the current embedder configuration."""
    embedder = AutoEmbedder()
    return embedder.get_model_info()
