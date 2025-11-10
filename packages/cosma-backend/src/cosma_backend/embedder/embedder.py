#!/usr/bin/env python
"""
@File    :   embedder.py
@Time    :   2025/07/14
@Author  :
@Version :   1.0
@Contact :
@License :
@Desc    :   Embedding generation for semantic search
"""

from datetime import datetime, timezone
import os
import asyncio
from abc import ABC, abstractmethod
from typing import Any

# Import AI libraries
import litellm
import numpy as np
from cosma_backend.models import File
import logging
from cosma_backend.logging import sm
from cosma_backend.models.status import ProcessingStatus

# Configure structured logger
logger = logging.getLogger(__name__)


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
        logger.info(sm("Initializing embedder", model=model_name, dimensions=dimensions))

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


class OnlineEmbedder(BaseEmbedder):
    """Embedder using online models via LiteLLM (OpenAI API)."""

    def __init__(self, model: str | None = None, api_key: str | None = None, dimensions: int | None = None) -> None:
        """
        Initialize online embedder.

        Args:
            model: Model name (default: text-embedding-3-small)
            api_key: API key (default from env)
            dimensions: Embedding dimensions (default: 512 for efficiency)
        """
        # Default to text-embedding-3-small with 512 dimensions for efficiency
        self.model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.configured_dimensions = dimensions or int(os.getenv("EMBEDDING_DIMENSIONS", "512"))

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

        logger.info(sm("Online embedder initialized",
                   model=self.model,
                   dimensions=self.configured_dimensions))

    def is_available(self) -> bool:
        """Check if online embedder is available."""
        return bool(os.getenv("OPENAI_API_KEY"))

    def embed_text(self, text: str | list[str]) -> np.ndarray:
        """Generate embeddings using online model."""
        texts = self._validate_text(text)

        logger.debug(sm("Generating embeddings",
                    model=self.model,
                    num_texts=len(texts),
                    dimensions=self.configured_dimensions))

        try:
            # Call litellm embedding endpoint
            response = litellm.embedding(
                model=self.model,
                input=texts,
                dimensions=self.configured_dimensions,  # Only for models that support it
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
            logger.exception(sm(error_msg, model=self.model))
            raise EmbeddingProviderError(error_msg)
    
    async def _embed_text_async(self, text: str | list[str]) -> np.ndarray:
        """Truly async embedding generation using litellm async API."""
        texts = self._validate_text(text)

        logger.debug(sm("Generating embeddings async",
                    model=self.model,
                    num_texts=len(texts),
                    dimensions=self.configured_dimensions))

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
            logger.exception(sm(error_msg, model=self.model))
            raise EmbeddingProviderError(error_msg)


class LocalEmbedder(BaseEmbedder):
    """Embedder using local models via sentence-transformers."""

    def __init__(self, model_name: str | None = None, dimensions: int | None = None) -> None:
        """
        Initialize local embedder.

        Args:
            model_name: Model name (default: Qwen/Qwen3-Embedding-0.6B)
            dimensions: Embedding dimensions (default: 768 for efficiency)
        """
        # Import sentence-transformers lazily
        try:
            from sentence_transformers import SentenceTransformer
            self.sentence_transformers_available = True
        except ImportError:
            self.sentence_transformers_available = False
            logger.warning(sm("sentence-transformers not installed, local embeddings unavailable"))

        # Default to Qwen model with 768 dimensions for efficiency
        self.model_name = model_name or os.getenv("LOCAL_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
        self.configured_dimensions = dimensions or int(os.getenv("LOCAL_EMBEDDING_DIMENSIONS", "768"))

        # Initialize base class
        super().__init__(model_name=self.model_name, dimensions=self.configured_dimensions)

        # Validate model compatibility
        if "Qwen3-Embedding" in self.model_name:
            if not (32 <= self.configured_dimensions <= 1024):
                msg = f"Dimensions must be between 32 and 1024 for {self.model_name}"
                raise ValueError(msg)

        # Initialize model if available
        self.model = None
        if self.sentence_transformers_available:
            try:
                logger.info(sm("Loading local embedding model",
                           model=self.model_name,
                           dimensions=self.configured_dimensions))
                # This will automatically download from HuggingFace if model doesn't exist locally
                self.model = SentenceTransformer(self.model_name)
                logger.info(sm("Local embedder initialized",
                           model=self.model_name,
                           dimensions=self.configured_dimensions))
            except Exception as e:
                logger.exception(sm("Failed to load local model",
                            model=self.model_name,
                            error=str(e)))
                logger.info(sm("Model will be downloaded from HuggingFace on first use"))

    def is_available(self) -> bool:
        """Check if local embedder is available."""
        return self.sentence_transformers_available and self.model is not None

    def embed_text(self, text: str | list[str]) -> np.ndarray:
        """Generate embeddings using local model."""
        if not self.is_available():
            msg = "Local embedder not available"
            raise EmbeddingProviderError(msg)

        texts = self._validate_text(text)

        logger.debug(sm("Generating local embeddings",
                    model=self.model_name,
                    num_texts=len(texts),
                    dimensions=self.configured_dimensions))

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
            logger.exception(sm(error_msg, model=self.model_name))
            raise EmbeddingProviderError(error_msg)


class AutoEmbedder:
    """
    Automatic embedder that selects the best available provider.

    Tries providers in order of preference:
    1. User-specified provider
    2. Online models (OpenAI)
    3. Local models (fallback)
    """

    def __init__(self, preferred_provider: str | None = None) -> None:
        """
        Initialize auto embedder.

        Args:
            preferred_provider: Preferred provider ('local', 'online')
        """
        self.preferred_provider = preferred_provider or os.getenv("EMBEDDING_PROVIDER", "local")
        logger.debug(sm("Preferred provider", og=preferred_provider, provider=self.preferred_provider))
        self.embedders = {}

        logger.info(sm("AutoEmbedder initializing",
                   preferred_provider=self.preferred_provider))
        
        # Eagerly initialize models based on preferred provider
        self._eagerly_initialize_models()

    def _eagerly_initialize_models(self) -> None:
        """Initialize embedding models based on provider preference - eager for local, lazy for online."""
        logger.info(sm("Initializing embedding models"))
        
        if self.preferred_provider == "local":
            # Eagerly initialize local embedder for local preference
            logger.info(sm("Eagerly loading local embedding models"))
            local_embedder = self._get_local_embedder()
            if local_embedder:
                logger.info(sm("Local embedder ready", 
                           model=local_embedder.model_name,
                           dimensions=local_embedder.dimensions))
            else:
                logger.warning(sm("Local embedder failed to initialize"))
                
            # Check online availability but don't initialize (lazy loading)
            logger.info(sm("Checking online embedding provider availability (lazy loading)"))
            online_available = self._check_online_availability()
            if online_available:
                logger.info(sm("Online embedder available as fallback (will load on first use)"))
            else:
                logger.warning(sm("Online embedder not available - check API keys"))
        else:
            # For online preference, check availability but don't initialize (lazy loading)
            logger.info(sm("Checking online embedding provider availability (lazy loading)"))
            online_available = self._check_online_availability()
            if online_available:
                logger.info(sm("Online embedder ready (will load on first use)", 
                           provider="online"))
            else:
                logger.warning(sm("Online embedder not available - check API keys"))
                
            # Skip local model initialization when user explicitly chose online
            logger.info(sm("Skipping local embedding model initialization (online provider preferred)"))
            logger.info(sm("To use local models as fallback, set EMBEDDING_PROVIDER=local"))
        
        # Summary of initialization strategy
        if self.preferred_provider == "local":
            logger.info(sm("AutoEmbedder configured: LOCAL models preloaded, ONLINE models lazy-loaded"))
        else:
            logger.info(sm("AutoEmbedder configured: ONLINE models only (LOCAL models skipped)"))

        logger.info(sm("AutoEmbedder initialization complete",
                   preferred_provider=self.preferred_provider,
                   strategy="eager_local_lazy_online" if self.preferred_provider == "local" else "online_only"))

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
                embedder = OnlineEmbedder()
                if embedder.is_available():
                    self.embedders["online"] = embedder
                    logger.info(sm("Online embedder available"))
                else:
                    logger.debug(sm("Online embedder not available"))
                    return None
            except Exception as e:
                logger.debug(sm("Failed to create online embedder", error=str(e)))
                return None

        return self.embedders.get("online")

    def _get_local_embedder(self) -> LocalEmbedder | None:
        """Get or create local embedder if available."""
        if "local" not in self.embedders:
            try:
                embedder = LocalEmbedder()
                if embedder.is_available():
                    self.embedders["local"] = embedder
                    logger.info(sm("Local embedder available"))
                else:
                    logger.debug(sm("Local embedder not available"))
                    return None
            except Exception as e:
                logger.warning(sm("Failed to create local embedder", error=str(e)))
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
            
        logger.debug(sm("All available providers", providers=providers))

        # Try each provider
        for embedder in providers:
            if embedder:
                try:
                    logger.info(sm("Attempting embedding generation",
                               provider=type(embedder).__name__))
                    return embedder.embed_text(text)
                except Exception as e:
                    logger.warning(sm("Embedder failed, trying next provider",
                                 provider=type(embedder).__name__,
                                 error=str(e)))
                    continue

        error_msg = "All embedding providers failed or are unavailable"
        logger.error(sm(error_msg, preferred_provider=self.preferred_provider))
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
                    logger.info(sm("Attempting async embedding generation",
                               provider=type(embedder).__name__))
                    return await embedder.embed_text_async(text)
                except Exception as e:
                    logger.warning(sm("Async embedder failed, trying next provider",
                                 provider=type(embedder).__name__,
                                 error=str(e)))
                    continue

        error_msg = "All embedding providers failed or are unavailable"
        logger.error(sm(error_msg, preferred_provider=self.preferred_provider))
        raise EmbedderError(error_msg)

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
    
        # Add content (truncated)
        # if file.content:
        #     content = file.content[:1000]  # Limit content length
        #     parts.append(f"Content: {content}")
    
        return " ".join(parts)
        
    async def embed(self, file: File):
        embedding_text = self._prepare_embedding_text(file)
        embedding = await self.embed_text_async(embedding_text)
        
        model_info = self.get_model_info()
        
        file.embedding = embedding
        file.embedding_model = model_info["model"]
        file.embedding_dimensions = model_info["dimensions"]
        file.embedded_at = datetime.now(timezone.utc)
        
        file.status = ProcessingStatus.COMPLETE


# Convenience functions for easier usage
def generate_embedding(text: str | list[str],
                      provider: str | None = None) -> np.ndarray:
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
