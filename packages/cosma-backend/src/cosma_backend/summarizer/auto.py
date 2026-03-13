"""
AutoSummarizer - automatic provider selection with fallback support.
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

from cosma_backend.logging import get_logger
from cosma_backend.models.file import File

from .base import SummarizerError
from .providers import LlamaCppSummarizer, OllamaSummarizer, OnlineSummarizer

if TYPE_CHECKING:
    from cosma_backend.settings import SummarizerConfig

logger = get_logger(__name__)


class AutoSummarizer:
    """
    Automatic summarizer that selects the best available provider.

    Tries providers in order of preference:
    1. User-specified provider
    2. Local llama.cpp (fastest local inference)
    3. Local Ollama (privacy-focused)
    4. Online models (fallback)
    """

    def __init__(self, config: SummarizerConfig | None = None, preferred_provider: Optional[str] = None):
        """
        Initialize auto summarizer.

        Args:
            config: SummarizerConfig instance
            preferred_provider: Preferred provider ('llamacpp', 'ollama', 'online', 'auto')
        """
        from cosma_backend.settings import SummarizerConfig as _SummarizerConfig
        self.config = config or _SummarizerConfig()
        self.preferred_provider = preferred_provider or self.config.provider
        self.summarizers: dict = {}

        logger.info("AutoSummarizer initialized", preferred_provider=self.preferred_provider)

    async def _get_llamacpp_summarizer(self) -> Optional[LlamaCppSummarizer]:
        """Get or create llama.cpp summarizer if available."""
        if "llamacpp" not in self.summarizers:
            try:
                logger.info("llama.cpp summarizer initializing")
                summarizer = LlamaCppSummarizer(config=self.config)
                if await summarizer.is_available():
                    self.summarizers["llamacpp"] = summarizer
                    logger.info("llama.cpp summarizer available")
                else:
                    self.summarizers["llamacpp"] = None
                    logger.warning(
                        "llama.cpp summarizer not available: llama-cpp-python is not installed. "
                        "Install with: uv pip install -e \"packages/cosma-backend[llamacpp]\""
                    )
                    return None
            except Exception as e:
                logger.warning("Failed to create llama.cpp summarizer", error=str(e))
                self.summarizers["llamacpp"] = None
                return None

        return self.summarizers.get("llamacpp")

    async def _get_ollama_summarizer(self) -> Optional[OllamaSummarizer]:
        """Get or create Ollama summarizer if available."""
        if "ollama" not in self.summarizers:
            try:
                summarizer = OllamaSummarizer(config=self.config)
                if await summarizer.is_available():
                    self.summarizers["ollama"] = summarizer
                    logger.info("Ollama summarizer available")
                else:
                    logger.debug("Ollama summarizer not available")
                    self.summarizers["ollama"] = None
                    return None
            except Exception as e:
                logger.debug(f"Failed to create Ollama summarizer - error: {str(e)}")
                self.summarizers["ollama"] = None
                return None

        return self.summarizers.get("ollama")

    async def _get_online_summarizer(self) -> Optional[OnlineSummarizer]:
        """Get or create online summarizer if available."""
        if "online" not in self.summarizers:
            try:
                summarizer = OnlineSummarizer(config=self.config)
                if await summarizer.is_available():
                    self.summarizers["online"] = summarizer
                    logger.info("Online summarizer available")
                else:
                    logger.debug("Online summarizer not available")
                    self.summarizers["online"] = None
                    return None
            except Exception as e:
                logger.debug(f"Failed to create online summarizer - error: {str(e)}")
                self.summarizers["online"] = None
                return None

        return self.summarizers.get("online")

    async def summarize(self, file_metadata: File) -> File:
        """
        Summarize using the best available provider with fallback.

        Args:
            file_metadata: File metadata to summarize

        Returns:
            Enhanced file metadata with summary and keywords

        Raises:
            SummarizerError: If no summarizers are available or all fail
        """
        # When a specific provider is requested (not "auto"), try only that provider
        _provider_getters = {
            "llamacpp": self._get_llamacpp_summarizer,
            "ollama": self._get_ollama_summarizer,
            "online": self._get_online_summarizer,
        }

        if self.preferred_provider in _provider_getters:
            provider = await _provider_getters[self.preferred_provider]()
            if not provider or not await provider.is_available():
                raise SummarizerError(
                    f"Summarizer provider '{self.preferred_provider}' is not available. "
                    f"Check that the required package is installed and configured."
                )
            logger.info("Attempting summarization", provider=type(provider).__name__)
            return await provider.summarize(file_metadata)

        # "auto" mode: try all providers in priority order with fallback
        providers = [
            await self._get_llamacpp_summarizer(),
            await self._get_ollama_summarizer(),
            await self._get_online_summarizer()
        ]

        for provider in providers:
            if provider and await provider.is_available():
                try:
                    logger.info("Attempting summarization", provider=type(provider).__name__)
                    return await provider.summarize(file_metadata)
                except Exception as e:
                    logger.warning("Summarizer failed, trying next provider", provider=type(provider).__name__, error=str(e))
                    continue

        error_msg = "All AI summarizers failed or are unavailable"
        logger.error("All AI summarizers failed or are unavailable", preferred_provider=self.preferred_provider)
        raise SummarizerError(error_msg)

    async def get_available_providers(self) -> List[str]:
        """Get list of available providers."""
        providers = []

        if await self._get_llamacpp_summarizer():
            providers.append("llamacpp")

        if await self._get_ollama_summarizer():
            providers.append("ollama")

        if await self._get_online_summarizer():
            providers.append("online")

        return providers

    @property
    def last_used_at(self) -> float:
        """Return the most recent last_used_at across all active summarizers."""
        latest = 0.0
        for summarizer in self.summarizers.values():
            if summarizer is not None and hasattr(summarizer, "last_used_at"):
                latest = max(latest, summarizer.last_used_at)
        return latest

    def is_any_model_loaded(self) -> bool:
        """Check if any summarizer model is currently loaded."""
        for summarizer in self.summarizers.values():
            if summarizer is None:
                continue
            if hasattr(summarizer, "_model_loaded") and summarizer._model_loaded:
                return True
            if hasattr(summarizer, "llm") and summarizer.llm is not None:
                return True
        return False

    async def unload_models(self) -> None:
        """Unload models from memory for all providers that support it."""
        for name, summarizer in self.summarizers.items():
            if summarizer is not None and hasattr(summarizer, "unload"):
                await summarizer.unload()
                logger.info("Unloaded model", provider=name)


# Convenience functions for easier usage
async def summarize_file(file_metadata: File, provider: Optional[str] = None) -> File:
    """
    Convenience function to summarize a file.

    Args:
        file_metadata: File metadata to summarize
        provider: Preferred AI provider

    Returns:
        Enhanced file metadata with summary and keywords
    """
    summarizer = AutoSummarizer(preferred_provider=provider)
    return await summarizer.summarize(file_metadata)


async def get_available_providers() -> List[str]:
    """Get list of available AI providers."""
    summarizer = AutoSummarizer()
    return await summarizer.get_available_providers()


async def is_summarizer_available() -> bool:
    """Check if any summarizer is available."""
    try:
        providers = await get_available_providers()
        return len(providers) > 0
    except Exception:
        return False
