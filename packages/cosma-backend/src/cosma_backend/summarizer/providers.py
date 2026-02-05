"""
Summarizer provider implementations: Ollama, Online (LiteLLM), and LlamaCpp.
"""

from __future__ import annotations

import gc
import os
import time
from typing import Any, Optional, TYPE_CHECKING

import litellm
import ollama

from cosma_backend.logging import get_logger

from .base import AIProviderError, BaseSummarizer
from .tokenization import extract_json_from_response

if TYPE_CHECKING:
    from cosma_backend.settings import SummarizerConfig

logger = get_logger(__name__)


class OllamaSummarizer(BaseSummarizer):
    """Summarizer using local Ollama models."""

    def __init__(self, config: SummarizerConfig | None = None, host: Optional[str] = None, model: Optional[str] = None, max_tokens: Optional[int] = None):
        """
        Initialize Ollama summarizer.

        Args:
            config: SummarizerConfig instance
            host: Ollama host URL override
            model: Model name override
            max_tokens: Maximum context tokens override
        """
        from cosma_backend.settings import SummarizerConfig as _SummarizerConfig
        self.config = config or _SummarizerConfig()

        # Get model name before initializing base class
        model_name = model or self.config.ollama.model

        # Initialize base class with context length and model
        context_length = max_tokens or self.config.ollama.context_length
        super().__init__(config=self.config, max_tokens=context_length, model=model_name)

        try:
            import ollama
            self.ollama_available = True
        except ImportError:
            self.ollama_available = False
            raise ImportError("ollama package is not installed")

        self.host = host or self.config.ollama.host

        self._last_used_at: float = 0.0
        self._model_loaded: bool = False

        try:
            self.client = ollama.AsyncClient(host=self.host)
            logger.info("Ollama summarizer initialized", host=self.host, model=self.model, max_tokens=self.max_tokens)
        except Exception as e:
            logger.error("Failed to initialize Ollama client", host=self.host, error=str(e))
            raise AIProviderError(f"Failed to initialize Ollama: {str(e)}")

    @property
    def last_used_at(self) -> float:
        return self._last_used_at

    async def unload(self) -> None:
        """Tell Ollama to unload the model from GPU/memory."""
        try:
            await self.client.generate(model=self.model, keep_alive="0")
            self._model_loaded = False
            logger.info("Ollama model unloaded", model=self.model)
        except Exception as e:
            logger.warning("Failed to unload Ollama model", model=self.model, error=str(e))

    async def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            # Try to list models to check if Ollama is running
            list = await self.client.list()
            if self.model and self.model not in (m.model for m in list.models):
                logger.info("Ollama model not found, pulling", model=self.model)
                await self.client.pull(self.model)
            self._model_loaded = True
            return True
        except Exception as e:
            logger.debug(f"Ollama not available - error: {str(e)}")
            return False

    async def _get_ai_response(self, chunk: str, chunk_num: int, total_chunks: int, images: list[str], filename: str) -> str | None:
        content = self._format_content_with_context(chunk, chunk_num, total_chunks, filename)
        user_message: dict[str, Any] = {"role": "user", "content": content}
        if images:
            user_message["images"] = images

        raw_response = await self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self._get_system_prompt(include_title=(chunk_num == 0))},
                user_message,
            ],
            think=False,
            options=ollama.Options(
                num_predict=500,
                num_ctx=16_000,
            )
        )

        self._last_used_at = time.time()
        self._model_loaded = True
        logger.info("AI response", summarizer=self.__class__.__name__, response=raw_response)
        return extract_json_from_response(raw_response['message']['content'])


class OnlineSummarizer(BaseSummarizer):
    """Summarizer using online AI models via LiteLLM."""

    def __init__(self, config: SummarizerConfig | None = None, model: Optional[str] = None, api_key: Optional[str] = None, max_tokens: Optional[int] = None):
        """
        Initialize online summarizer.

        Args:
            config: SummarizerConfig instance
            model: Model name override
            api_key: API key override
            max_tokens: Maximum context tokens override
        """
        from cosma_backend.settings import SummarizerConfig as _SummarizerConfig
        self.config = config or _SummarizerConfig()

        # Get model name before initializing base class
        model_name = model or self.config.online.model

        # Initialize base class with context length and model
        context_length = max_tokens or self.config.online.context_length
        super().__init__(config=self.config, max_tokens=context_length, model=model_name)

        try:
            import litellm
            self.litellm_available = True
        except ImportError:
            self.litellm_available = False
            raise ImportError("litellm package is not installed")

        # Set API key if provided
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        logger.info("Online summarizer initialized", model=self.model, max_tokens=self.max_tokens)

    async def is_available(self) -> bool:
        """Check if online models are available."""
        # Check for required API keys based on model
        if self.model.startswith("gpt-") or self.model.startswith("o1-"):
            return bool(os.getenv("OPENAI_API_KEY"))
        elif self.model.startswith("claude-"):
            return bool(os.getenv("ANTHROPIC_API_KEY"))
        elif self.model.startswith("gemini-"):
            return bool(os.getenv("GOOGLE_API_KEY"))
        else:
            # Assume OpenAI by default
            return bool(os.getenv("OPENAI_API_KEY"))

    async def _get_ai_response(self, chunk: str, chunk_num: int, total_chunks: int, images: list[str], filename: str) -> str | None:
        content = self._format_content_with_context(chunk, chunk_num, total_chunks, filename)
        user_message = {"role": "user", "content": content}
        if images:
            user_message["images"] = images

        response = litellm.completion(
            model=self.model,
            messages=[
                {"role": "system", "content": self._get_system_prompt(include_title=(chunk_num == 0))},
                user_message,
            ],
            temperature=0.1,
            max_tokens=300,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            response_format={"type": "json_object"},
            timeout=120,
            max_retries=2,
        )

        return response.choices[0].message.content


class LlamaCppSummarizer(BaseSummarizer):
    """Summarizer using local llama.cpp models with lazy loading."""

    def __init__(self, config: SummarizerConfig | None = None, model_path: Optional[str] = None, max_tokens: Optional[int] = None, n_ctx: Optional[int] = None):
        """
        Initialize llama.cpp summarizer (lazy - model loaded on first use).

        Args:
            config: SummarizerConfig instance
            model_path: Path to GGUF model file override
            max_tokens: Maximum context tokens override
            n_ctx: Context window size override
        """
        from cosma_backend.settings import SummarizerConfig as _SummarizerConfig
        self.config = config or _SummarizerConfig()

        # Get model path from config if not provided
        self.model_path = model_path or self.config.llamacpp.model_path
        self.repo_id = self.config.llamacpp.repo_id
        self.filename = self.config.llamacpp.filename
        if not (self.model_path or all((self.repo_id, self.filename))):
            raise ValueError("LLAMACPP_MODEL_PATH environment variable must be set or model_path must be provided")

        # Initialize base class with context length
        context_length = max_tokens or self.config.llamacpp.context_length
        super().__init__(config=self.config, max_tokens=context_length, model="llama.cpp")

        self.n_ctx = n_ctx or self.config.llamacpp.n_ctx

        # Check if llama_cpp is importable (don't load model yet)
        try:
            import importlib.util
            self.llamacpp_available = importlib.util.find_spec("llama_cpp") is not None
        except ImportError:
            self.llamacpp_available = False

        # Model will be loaded lazily on first use
        self.llm = None

    def _ensure_loaded(self) -> None:
        """Load the model if not already loaded (lazy loading)."""
        if self.llm is not None:
            return

        if not self.llamacpp_available:
            raise ImportError("llama-cpp-python package is not installed. Install with: pip install llama-cpp-python")

        from llama_cpp import Llama

        try:
            if self.repo_id and self.filename:
                self.llm = Llama.from_pretrained(
                    repo_id=self.repo_id,
                    filename=self.filename,
                    n_ctx=self.n_ctx,
                    n_threads=self.config.llamacpp.n_threads,
                    n_gpu_layers=self.config.llamacpp.n_gpu_layers,
                    verbose=self.config.llamacpp.verbose,
                )
                logger.info("llama.cpp model loaded (lazy)",
                            repo_id=self.repo_id,
                            filename=self.filename,
                            n_ctx=self.n_ctx,
                            max_tokens=self.max_tokens)
            else:
                self.llm = Llama(
                    model_path=self.model_path,
                    n_ctx=self.n_ctx,
                    n_threads=self.config.llamacpp.n_threads,
                    n_gpu_layers=self.config.llamacpp.n_gpu_layers,
                    verbose=self.config.llamacpp.verbose,
                )
                logger.info("llama.cpp model loaded (lazy)",
                            model_path=self.model_path,
                            n_ctx=self.n_ctx,
                            max_tokens=self.max_tokens)
        except Exception as e:
            logger.error("Failed to load llama.cpp model", model_path=self.model_path, error=str(e))
            raise AIProviderError(f"Failed to load llama.cpp: {str(e)}")

    async def unload(self) -> None:
        """Unload the model to free memory."""
        if self.llm is not None:
            logger.info("Unloading llama.cpp model")
            del self.llm
            self.llm = None
            gc.collect()

    async def is_available(self) -> bool:
        """Check if llama.cpp is available (doesn't load model)."""
        return self.llamacpp_available

    async def _get_ai_response(self, chunk: str, chunk_num: int, total_chunks: int, images: list[str], filename: str) -> str | None:
        # Ensure model is loaded before use
        self._ensure_loaded()
        content = self._format_content_with_context(chunk, chunk_num, total_chunks, filename)
        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": self._get_system_prompt(include_title=(chunk_num == 0))},
                {"role": "user", "content": content},
            ],
            max_tokens=500,
            temperature=0.1,
            top_p=0.95,
            stream=False,
        )

        response_content = response['choices'][0]['message']['content'].strip()
        logger.info("llama.cpp raw response", response=response_content)
        return extract_json_from_response(response_content)
