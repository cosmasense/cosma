"""
Summarizer module for AI-powered file summarization.

This module provides multiple summarizer implementations:
- OllamaSummarizer: Local Ollama models
- OnlineSummarizer: Online AI models via LiteLLM
- LlamaCppSummarizer: Local llama.cpp models
- AutoSummarizer: Automatic provider selection with fallback
"""

from .base import (
    AIProviderError,
    BaseSummarizer,
    SummarizerError,
)
from .providers import (
    LlamaCppSummarizer,
    OllamaSummarizer,
    OnlineSummarizer,
)
from .auto import (
    AutoSummarizer,
    get_available_providers,
    is_summarizer_available,
    summarize_file,
)
from .tokenization import (
    chunk_content,
    estimate_tokens,
    estimate_tokens_fast,
    extract_json_from_response,
    get_encoding_for_model,
)

__all__ = [
    # Base classes and exceptions
    "BaseSummarizer",
    "SummarizerError",
    "AIProviderError",
    # Provider implementations
    "OllamaSummarizer",
    "OnlineSummarizer",
    "LlamaCppSummarizer",
    # Auto summarizer
    "AutoSummarizer",
    # Convenience functions
    "summarize_file",
    "get_available_providers",
    "is_summarizer_available",
    # Token utilities
    "estimate_tokens",
    "estimate_tokens_fast",
    "chunk_content",
    "get_encoding_for_model",
    "extract_json_from_response",
]
