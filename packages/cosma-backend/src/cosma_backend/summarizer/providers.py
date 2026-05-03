"""
Summarizer provider implementations: Ollama, Online (LiteLLM), and LlamaCpp.
"""

from __future__ import annotations

import asyncio
import os
import threading
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
                {"role": "system", "content": self._get_system_prompt(
                    include_title=(chunk_num == 0),
                    is_visual=bool(images),
                )},
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
                {"role": "system", "content": self._get_system_prompt(
                    include_title=(chunk_num == 0),
                    is_visual=bool(images),
                )},
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

        # n_ctx scales with the user's machine: a 16 GB Mac gets a
        # smaller window than a 64 GB Studio. resolved_n_ctx returns
        # the explicit value if set, else picks one from RAM. Avoids
        # OOM on small machines AND under-using big ones.
        from cosma_backend.utils.hardware import resolved_n_ctx
        self.n_ctx = n_ctx or resolved_n_ctx(self.config.llamacpp.n_ctx)
        # Reserve headroom in every chunk for the system prompt, user scaffolding
        # (filename, part-of-N header), image tokens, and generation budget. When
        # n_ctx is small (e.g. 8192) and a chunk filled it entirely, llama.cpp
        # rejected the prefill with "Requested tokens (N) exceed context window".
        # Budget: ~1200 for system+scaffolding+generation, plus image_min_tokens
        # when vision is enabled.
        prompt_headroom = 1200
        if self.chat_handler_name and self.config.llamacpp.image_min_tokens > 0:
            prompt_headroom += self.config.llamacpp.image_min_tokens
        raw_context_length = max_tokens or self.config.llamacpp.context_length
        # Cap chunk size at n_ctx - headroom so a chunk + prompt always fits.
        context_length = max(512, min(raw_context_length, self.n_ctx - prompt_headroom))
        if context_length < raw_context_length:
            logger.info("Reducing chunk size to fit n_ctx with prompt headroom",
                        configured=raw_context_length, effective=context_length,
                        n_ctx=self.n_ctx, headroom=prompt_headroom)
        # Pass the HF tokenizer repo as the model identifier so token counting
        # uses the real Qwen tokenizer rather than the cl100k_base fallback.
        # Mismatched counts caused chunks to exceed n_ctx and trip llama.cpp
        # with "token out of bounds" at prefill.
        tokenizer_repo = self.config.llamacpp.tokenizer_repo or "llama.cpp"
        super().__init__(config=self.config, max_tokens=context_length, model=tokenizer_repo)

        try:
            import importlib.util
            self.llamacpp_available = importlib.util.find_spec("llama_cpp") is not None
        except ImportError:
            self.llamacpp_available = False

        self.llm = None
        self.chat_handler = None
        self._last_used_at: float = 0.0

        # llama.cpp is NOT thread-safe. The Pipeline's summarize
        # semaphore (concurrency=1) normally guarantees one asyncio
        # task in summarize at a time, but cancel_in_flight() can
        # release that semaphore mid-call: the asyncio task is
        # cancelled while its `to_thread`-dispatched
        # create_chat_completion is still running on a worker thread,
        # and the next task can race in for a SECOND concurrent call
        # against the same Llama instance → SIGBUS / Fatal Python
        # error. The first random10bench_real run (2026-05-01)
        # reproduced this every time. This Lock serializes inference
        # calls at the C boundary regardless of asyncio bookkeeping.
        self._inference_lock = threading.Lock()

    @property
    def last_used_at(self) -> float:
        return self._last_used_at

    def _resolve_clip_model_path(self) -> Optional[str]:
        """Download or locate the clip (mmproj) model for vision support.

        Uses the shared cosma_backend.downloads helper so the file lands in
        ~/Library/Application Support/cosma/models/llama/ instead of the
        HuggingFace cache. This keeps all model blobs in one inspectable
        location and lets the bootstrap API report them consistently.
        """
        if self.clip_model_path:
            logger.info(
                "Vision: using explicit clip_model_path from config",
                clip_model_path=self.clip_model_path,
            )
            return self.clip_model_path

        if not (self.clip_repo_id and self.clip_filename):
            # Hard-disable vision if the user blanked these out. Loud
            # because it's almost always a misconfiguration: an image-
            # capable model with no clip projector means every image
            # gets summarized text-only.
            logger.error(
                "Vision DISABLED: clip_repo_id or clip_filename is empty in "
                "settings.toml. Image files will be summarized text-only "
                "(usually just the parser metadata placeholder).",
                clip_repo_id=self.clip_repo_id,
                clip_filename=self.clip_filename,
            )
            return None

        try:
            from cosma_backend.downloads import download_hf_file
            result = download_hf_file(
                repo_id=self.clip_repo_id,
                filename=self.clip_filename,
                kind="llama",
                subdir="mmproj",
                stage_label="mmproj",
            )
            logger.info(
                "Vision: clip (mmproj) projector resolved",
                clip_repo_id=self.clip_repo_id,
                clip_filename=self.clip_filename,
                clip_path=str(result.path),
            )
            return str(result.path)
        except Exception as e:
            # Promoted warning → error. This is the single most common
            # reason image hits look broken (offline at first install,
            # mmproj-F16.gguf never landed) and it used to show up at
            # warning level which the user could easily miss.
            logger.error(
                "Vision DISABLED: failed to download clip (mmproj) model. "
                "Image files will be summarized text-only — usually just "
                "the parser metadata placeholder. Re-run when online or "
                "manually drop the file at "
                "~/Library/Application Support/cosma/models/llama/mmproj/.",
                clip_repo_id=self.clip_repo_id,
                clip_filename=self.clip_filename,
                error=str(e),
                error_type=type(e).__name__,
            )
            return None

    # Fallback order when the configured handler isn't present in the
    # installed llama-cpp-python build. Tries the user's preferred
    # handler first, then walks through other handlers from the same
    # family (any *VL handler can drive a multimodal call). Stops on
    # the first one that exists. The Qwen3VLChatHandler ONLY exists in
    # the cosmasense fork's v0.3.32-metal build; users who installed
    # via `uv tool install cosma` get upstream PyPI's llama-cpp-python
    # which has Qwen2.5-VL but not Qwen3-VL — without a fallback they
    # silently lose vision.
    _HANDLER_FALLBACK_ORDER: list[str] = [
        "Qwen3VLChatHandler",
        "Qwen25VLChatHandler",
        "MiniCPMv26ChatHandler",
        "Gemma3ChatHandler",
    ]

    def _resolve_handler_class(self) -> tuple[str | None, Any | None]:
        """Pick the best chat handler class available in this
        llama-cpp-python build. Returns (class_name, handler_class) or
        (None, None) if nothing usable exists. Logs the fallback so the
        user can see when they're running on a non-preferred handler.
        """
        from llama_cpp import llama_chat_format

        preferred_name = self._CHAT_HANDLERS.get(self.chat_handler_name)
        if preferred_name is None:
            logger.warning(
                "Unknown chat_handler in settings; will pick first "
                "available from fallback order.",
                handler=self.chat_handler_name,
                known_handlers=list(self._CHAT_HANDLERS.keys()),
            )

        # Try preferred first, then walk the fallback list.
        candidates = [preferred_name] if preferred_name else []
        for name in self._HANDLER_FALLBACK_ORDER:
            if name not in candidates:
                candidates.append(name)

        tried: list[str] = []
        for class_name in candidates:
            if class_name is None:
                continue
            cls = getattr(llama_chat_format, class_name, None)
            if cls is not None:
                if preferred_name and class_name != preferred_name:
                    logger.warning(
                        "Preferred chat handler not in this llama-cpp-python "
                        "build — falling back. Vision will work but with the "
                        "fallback model. To get the preferred handler, "
                        "install the cosmasense Metal fork: "
                        "`uv tool install --reinstall cosma --with "
                        "https://github.com/cosmasense/llama-cpp-python/"
                        "releases/download/v0.3.32-metal/"
                        "llama_cpp_python-0.3.32-cp313-cp313-"
                        "macosx_11_0_arm64.whl` (swap cp313 for your Python "
                        "minor version).",
                        preferred=preferred_name,
                        fell_back_to=class_name,
                    )
                return class_name, cls
            tried.append(class_name)

        logger.error(
            "Vision DISABLED: NONE of the chat handler classes are present "
            "in this llama-cpp-python build. Reinstall llama-cpp-python "
            "(stock PyPI build has at least Qwen25VLChatHandler since "
            "0.3.16): `uv tool install --reinstall --upgrade cosma`. "
            "For Qwen3-VL specifically, install the cosmasense Metal fork.",
            tried=tried,
        )
        return None, None

    def _create_chat_handler(self, clip_path: str) -> Any:
        """Create the appropriate chat handler for the configured model.

        Falls back to the next available handler in
        `_HANDLER_FALLBACK_ORDER` if the preferred handler class isn't
        present in this llama-cpp-python build — common for users who
        install via PyPI rather than the cosmasense fork.
        """
        class_name, handler_cls = self._resolve_handler_class()
        if handler_cls is None:
            return None

        kwargs: dict[str, Any] = {"clip_model_path": clip_path}
        # `enable_thinking` is only a kwarg on Qwen3.5's handler.
        if class_name == "Qwen35ChatHandler":
            kwargs["enable_thinking"] = self.enable_thinking
        if self.config.llamacpp.image_min_tokens > 0:
            kwargs["image_min_tokens"] = self.config.llamacpp.image_min_tokens
        kwargs["verbose"] = self.config.llamacpp.verbose

        try:
            handler = handler_cls(**kwargs)
        except Exception as e:
            logger.error(
                "Vision DISABLED: chat handler constructor raised. The "
                "clip path is on disk but llama-cpp-python rejected it "
                "(usually mmproj/model architecture mismatch).",
                handler=self.chat_handler_name,
                class_name=class_name,
                clip_path=clip_path,
                error=str(e),
                error_type=type(e).__name__,
            )
            return None
        logger.info(
            "Vision ENABLED: chat handler created",
            requested_handler=self.chat_handler_name,
            actual_class_name=class_name,
            clip_path=clip_path,
            image_min_tokens=self.config.llamacpp.image_min_tokens,
        )
        return handler

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
        if self.chat_handler is None:
            # Final summary line so the operator sees ONE definitive
            # statement of the vision state in the log instead of having
            # to assemble it from two warnings + a guess.
            logger.error(
                "Vision DISABLED for this LlamaCppSummarizer instance. "
                "Image summaries will fall back to text-only inference "
                "against the parser placeholder — expect garbage on photos."
            )

        try:
            llama_kwargs: dict[str, Any] = {
                "n_ctx": self.n_ctx,
                "n_threads": self.config.llamacpp.n_threads,
                "n_gpu_layers": self.config.llamacpp.n_gpu_layers,
                "verbose": self.config.llamacpp.verbose,
            }
            # Optional perf flags. SIGABRT-on-incompatible-build is a
            # real risk (e.g. flash_attn=True + Qwen3VLChatHandler in
            # cosmasense v0.3.32-metal crashed at 2026-05-01). We
            # default these OFF and let users opt in via settings.toml
            # once verified against their own llama-cpp-python build.
            if getattr(self.config.llamacpp, "flash_attn", False):
                llama_kwargs["flash_attn"] = True
            if getattr(self.config.llamacpp, "kv_cache_quant", False):
                # GGML_TYPE_Q8_0 = 8. Quality loss <0.1 perplexity.
                # Only meaningful with flash_attn=True; without FA,
                # llama.cpp dequantizes the cache every attention
                # call, which is *slower* than not quantizing.
                llama_kwargs["type_k"] = 8
                llama_kwargs["type_v"] = 8
            if self.chat_handler:
                llama_kwargs["chat_handler"] = self.chat_handler

            # Resolve the model file path once.
            if self.repo_id and self.filename:
                from cosma_backend.downloads import download_hf_file
                result = download_hf_file(
                    repo_id=self.repo_id,
                    filename=self.filename,
                    kind="llama",
                    subdir="gguf",
                    stage_label="llama-gguf",
                )
                model_path_resolved = str(result.path)
            else:
                model_path_resolved = self.model_path

            # Older llama-cpp-python builds may not accept flash_attn /
            # type_k / type_v. Try the optimized kwargs first; fall
            # back gracefully if the Llama constructor rejects them so
            # we never crash a user who's behind on llama-cpp-python
            # versions. Logged so the regression is visible.
            try:
                self.llm = Llama(model_path=model_path_resolved, **llama_kwargs)
                logger.info(
                    "llama.cpp model loaded",
                    model_path=model_path_resolved,
                    n_ctx=self.n_ctx,
                    vision=self.chat_handler is not None,
                    flash_attn=llama_kwargs.get("flash_attn", False),
                    kv_quant=("type_k" in llama_kwargs),
                )
            except TypeError as e:
                logger.warning(
                    "llama-cpp-python rejected one of the optional perf "
                    "kwargs — retrying without them.",
                    error=str(e),
                )
                for k in ("flash_attn", "type_k", "type_v"):
                    llama_kwargs.pop(k, None)
                self.llm = Llama(model_path=model_path_resolved, **llama_kwargs)
                logger.info("llama.cpp model loaded (legacy kwargs)",
                            model_path=model_path_resolved,
                            n_ctx=self.n_ctx,
                            vision=self.chat_handler is not None)

            # Prompt caching. The system prompt is byte-identical on
            # every chunk's create_chat_completion call (see
            # base._get_system_prompt). Without a cache, llama.cpp
            # re-tokenizes and re-prefills those ~250 tokens every
            # call. LlamaRAMCache skips prefill when the next request
            # shares a prefix. Gated by config so it can be turned off
            # if a build interaction surfaces.
            if getattr(self.config.llamacpp, "prompt_cache_enabled", True):
                try:
                    from llama_cpp import LlamaRAMCache
                    capacity_mib = int(getattr(
                        self.config.llamacpp,
                        "prompt_cache_capacity_mib",
                        200,
                    ))
                    self.llm.set_cache(LlamaRAMCache(
                        capacity_bytes=capacity_mib * 1024 * 1024,
                    ))
                    logger.info(
                        "LlamaRAMCache enabled "
                        "(system-prompt prefill cached)",
                        capacity_mib=capacity_mib,
                    )
                except Exception:
                    logger.exception(
                        "Failed to enable LlamaRAMCache (non-fatal)",
                    )
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

    def _build_user_content(self, text: str, images: list[str], filename: str = "") -> str | list[dict[str, Any]]:
        """Build user message content, using OpenAI multimodal format for images."""
        if not images:
            return text
        if not self.chat_handler:
            # Caller had images to attach but vision is dead — most
            # common cause of "image summary is just the file metadata
            # placeholder". Loud per-call so the operator sees it
            # alongside the actual offending filename.
            logger.error(
                "Image summarize call dropped vision: chat_handler is None "
                "(see _ensure_loaded errors above for why). Calling LLM "
                "text-only — expect a garbage summary that just echoes "
                "the parser placeholder.",
                filename=filename,
                num_images=len(images),
            )
            return text

        content: list[dict[str, Any]] = []
        for img_b64 in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
            })
        content.append({"type": "text", "text": text})
        logger.debug(
            "Image summarize call going out with vision",
            filename=filename,
            num_images=len(images),
            text_len=len(text),
        )
        return content

    async def _get_ai_response(self, chunk: str, chunk_num: int, total_chunks: int, images: list[str], filename: str) -> str | None:
        self._ensure_loaded()
        text = self._format_content_with_context(chunk, chunk_num, total_chunks, filename)
        # is_visual = images actually reach the model. If chat_handler
        # is None the multimodal payload gets stripped in
        # _build_user_content and the model only sees text — using a
        # visual prompt in that case would just waste prefill on
        # instructions about pictures the model can't see.
        is_visual = bool(images) and self.chat_handler is not None
        sys_prompt = self._get_system_prompt(
            include_title=(chunk_num == 0),
            is_visual=is_visual,
        )

        # Hard safety net against context-window overflow.
        #
        # Upstream chunking uses an HF tokenizer (or a tiktoken / 4-chars
        # heuristic fallback) whose token counts can disagree with the
        # GGUF model's real tokenizer by 20-50% for CJK / code / dense
        # technical text. When the disagreement goes the wrong way, a
        # chunk that "fits" by upstream count blows past n_ctx at prefill
        # and llama.cpp aborts with "Requested tokens (N) exceed context
        # window of M" — which is what the user's failure report showed
        # across ~17 PDFs.
        #
        # We measure the prompt with the *actual* model tokenizer here
        # and trim the user content from the tail until it fits, so the
        # call always succeeds with whatever fraction of the chunk we
        # can carry. Better a partial summary than a hard FAILED.
        generation_budget = 500
        chat_template_overhead = 200  # role markers, BOS/EOS, format tokens
        try:
            sys_token_count = len(self.llm.tokenize(sys_prompt.encode("utf-8"), add_bos=False, special=True))
        except Exception:
            sys_token_count = max(1, len(sys_prompt) // 3)
        image_tokens = self.config.llamacpp.image_min_tokens if (images and self.chat_handler) else 0
        available_for_user = (
            self.n_ctx
            - sys_token_count
            - generation_budget
            - chat_template_overhead
            - image_tokens
        )
        # Floor so impossibly tight n_ctx values still produce *something*.
        available_for_user = max(256, available_for_user)

        try:
            text_tokens = self.llm.tokenize(text.encode("utf-8"), add_bos=False, special=False)
        except Exception:
            text_tokens = None

        if text_tokens is not None and len(text_tokens) > available_for_user:
            logger.warning(
                "User content exceeds n_ctx budget at prefill; truncating to fit",
                original_tokens=len(text_tokens),
                kept_tokens=available_for_user,
                n_ctx=self.n_ctx,
                sys_tokens=sys_token_count,
                image_tokens=image_tokens,
                generation_budget=generation_budget,
            )
            try:
                text = self.llm.detokenize(text_tokens[:available_for_user]).decode("utf-8", errors="ignore")
            except Exception:
                # detokenize() can fail on partial multi-byte sequences; fall
                # back to a character-based trim with the same total budget.
                approx_chars = available_for_user * 3
                text = text[:approx_chars]
            text = text + "\n\n[content truncated to fit model context window]"

        user_content = self._build_user_content(text, images, filename=filename)
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ]

        # Offload synchronous llama.cpp inference to a thread so it doesn't
        # block the async event loop (inference can take several seconds).
        # The threading.Lock serializes the actual C call; see
        # `_inference_lock` doc in __init__ for why it has to be at the
        # C boundary, not the asyncio one.
        def _run_inference() -> Any:
            with self._inference_lock:
                return self.llm.create_chat_completion(
                    messages=messages,
                    max_tokens=generation_budget,
                    temperature=0.1,
                    top_p=0.95,
                    stream=False,
                )
        response = await asyncio.to_thread(_run_inference)

        self._last_used_at = time.time()
        response_content = response['choices'][0]['message']['content'].strip()
        logger.info("llama.cpp raw response", response=response_content)
        return extract_json_from_response(response_content)
