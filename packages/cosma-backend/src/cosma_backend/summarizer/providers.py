"""
Summarizer provider implementations: Ollama, Online (LiteLLM), and LlamaCpp.
"""

from __future__ import annotations

import asyncio
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
        if not self._model_loaded:
            return
        try:
            await self.client.generate(model=self.model, keep_alive="0")
            self._model_loaded = False
            self._last_used_at = 0.0
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

        response = await litellm.acompletion(
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
    """Summarizer using local llama.cpp models with lazy loading and vision support."""

    # Map config chat_handler names to llama_cpp chat handler classes
    _CHAT_HANDLERS: dict[str, str] = {
        "qwen3.5": "Qwen35ChatHandler",
        "qwen3-vl": "Qwen3VLChatHandler",
        "qwen2.5-vl": "Qwen25VLChatHandler",
        "minicpm-v-2.6": "MiniCPMv26ChatHandler",
        "gemma3": "Gemma3ChatHandler",
    }

    def __init__(self, config: SummarizerConfig | None = None, model_path: Optional[str] = None, max_tokens: Optional[int] = None, n_ctx: Optional[int] = None):
        from cosma_backend.settings import SummarizerConfig as _SummarizerConfig
        self.config = config or _SummarizerConfig()

        self.model_path = model_path or self.config.llamacpp.model_path
        self.repo_id = self.config.llamacpp.repo_id
        self.filename = self.config.llamacpp.filename
        self.clip_model_path = self.config.llamacpp.clip_model_path
        self.clip_repo_id = self.config.llamacpp.clip_repo_id
        self.clip_filename = self.config.llamacpp.clip_filename
        self.chat_handler_name = self.config.llamacpp.chat_handler
        self.enable_thinking = self.config.llamacpp.enable_thinking

        if not (self.model_path or all((self.repo_id, self.filename))):
            raise ValueError("llamacpp.model_path or llamacpp.repo_id + llamacpp.filename must be configured")

        context_length = max_tokens or self.config.llamacpp.context_length
        super().__init__(config=self.config, max_tokens=context_length, model="llama.cpp")

        self.n_ctx = n_ctx or self.config.llamacpp.n_ctx

        try:
            import importlib.util
            self.llamacpp_available = importlib.util.find_spec("llama_cpp") is not None
        except ImportError:
            self.llamacpp_available = False

        self.llm = None
        self.chat_handler = None
        self._last_used_at: float = 0.0

    @property
    def last_used_at(self) -> float:
        return self._last_used_at

    def _resolve_clip_model_path(self) -> Optional[str]:
        """Download or locate the clip (mmproj) model for vision support."""
        if self.clip_model_path:
            return self.clip_model_path

        if not (self.clip_repo_id and self.clip_filename):
            return None

        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id=self.clip_repo_id,
                filename=self.clip_filename,
            )
            logger.info("clip model downloaded", repo_id=self.clip_repo_id, filename=self.clip_filename, path=path)
            return path
        except Exception as e:
            logger.warning("Failed to download clip model, vision will be unavailable", error=str(e))
            return None

    def _create_chat_handler(self, clip_path: str) -> Any:
        """Create the appropriate chat handler for the configured model."""
        from llama_cpp import llama_chat_format

        class_name = self._CHAT_HANDLERS.get(self.chat_handler_name)
        if not class_name:
            logger.warning("Unknown chat handler, falling back to no handler", handler=self.chat_handler_name)
            return None

        handler_cls = getattr(llama_chat_format, class_name, None)
        if handler_cls is None:
            logger.warning("Chat handler class not found in llama_cpp", class_name=class_name)
            return None

        kwargs: dict[str, Any] = {"clip_model_path": clip_path}
        if self.chat_handler_name == "qwen3.5":
            kwargs["enable_thinking"] = self.enable_thinking
        if self.config.llamacpp.image_min_tokens > 0:
            kwargs["image_min_tokens"] = self.config.llamacpp.image_min_tokens
        kwargs["verbose"] = self.config.llamacpp.verbose

        return handler_cls(**kwargs)

    def _ensure_loaded(self) -> None:
        """Load the model if not already loaded (lazy loading)."""
        if self.llm is not None:
            return

        if not self.llamacpp_available:
            raise ImportError("llama-cpp-python package is not installed. Install with: pip install llama-cpp-python")

        from llama_cpp import Llama

        # Set up vision chat handler if clip model is available
        clip_path = self._resolve_clip_model_path()
        if clip_path:
            self.chat_handler = self._create_chat_handler(clip_path)
            if self.chat_handler:
                logger.info("Vision chat handler created", handler=self.chat_handler_name)

        try:
            llama_kwargs: dict[str, Any] = {
                "n_ctx": self.n_ctx,
                "n_threads": self.config.llamacpp.n_threads,
                "n_gpu_layers": self.config.llamacpp.n_gpu_layers,
                "verbose": self.config.llamacpp.verbose,
            }
            if self.chat_handler:
                llama_kwargs["chat_handler"] = self.chat_handler

            if self.repo_id and self.filename:
                self.llm = Llama.from_pretrained(
                    repo_id=self.repo_id,
                    filename=self.filename,
                    **llama_kwargs,
                )
                logger.info("llama.cpp model loaded",
                            repo_id=self.repo_id,
                            filename=self.filename,
                            n_ctx=self.n_ctx,
                            vision=self.chat_handler is not None)
            else:
                self.llm = Llama(
                    model_path=self.model_path,
                    **llama_kwargs,
                )
                logger.info("llama.cpp model loaded",
                            model_path=self.model_path,
                            n_ctx=self.n_ctx,
                            vision=self.chat_handler is not None)
        except Exception as e:
            logger.error("Failed to load llama.cpp model", error=str(e))
            raise AIProviderError(f"Failed to load llama.cpp: {str(e)}")

    async def unload(self) -> None:
        """Unload the model and chat handler to free memory."""
        if self.llm is not None:
            logger.info("Unloading llama.cpp model")
            del self.llm
            self.llm = None
        if self.chat_handler is not None:
            del self.chat_handler
            self.chat_handler = None
        self._last_used_at = 0.0
        from cosma_backend.utils.memory import release_memory
        release_memory()

    async def is_available(self) -> bool:
        """Check if llama.cpp is available (doesn't load model)."""
        return self.llamacpp_available

    def _build_user_content(self, text: str, images: list[str]) -> str | list[dict[str, Any]]:
        """Build user message content, using OpenAI multimodal format for images."""
        if not images or not self.chat_handler:
            return text

        content: list[dict[str, Any]] = []
        for img_b64 in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
            })
        content.append({"type": "text", "text": text})
        return content

    async def _get_ai_response(self, chunk: str, chunk_num: int, total_chunks: int, images: list[str], filename: str) -> str | None:
        self._ensure_loaded()
        text = self._format_content_with_context(chunk, chunk_num, total_chunks, filename)
        user_content = self._build_user_content(text, images)

        messages = [
            {"role": "system", "content": self._get_system_prompt(include_title=(chunk_num == 0))},
            {"role": "user", "content": user_content},
        ]

        # Offload synchronous llama.cpp inference to a thread so it doesn't
        # block the async event loop (inference can take several seconds).
        response = await asyncio.to_thread(
            self.llm.create_chat_completion,
            messages=messages,
            max_tokens=500,
            temperature=0.1,
            top_p=0.95,
            stream=False,
        )

        self._last_used_at = time.time()
        response_content = response['choices'][0]['message']['content'].strip()
        logger.info("llama.cpp raw response", response=response_content)
        return extract_json_from_response(response_content)
