"""
Reproduction tests for the Qwen3-VL chat handler load path.

These tests catch the failure mode reported in production: stock PyPI
llama-cpp-python is missing Qwen3VLChatHandler, the previous fallback
chain quietly tried Qwen25VLChatHandler instead, and that handler
crashed with `TypeError: Llava15ChatHandler.__init__() got an unexpected
keyword argument 'image_min_tokens'` — leaving vision permanently off
without a clear signal at install time.

If these tests fail, the build does not have the cosmasense fork
installed. See the install hint logged by `_resolve_handler_class`.
"""

from unittest.mock import patch

import pytest

from cosma_backend.summarizer.providers import LlamaCppSummarizer


@pytest.mark.unit
class TestQwen3VLHandlerAvailable:
    """The cosmasense fork must be installed; stock PyPI doesn't ship Qwen3VLChatHandler."""

    def test_handler_class_present_in_llama_cpp(self):
        from llama_cpp import llama_chat_format

        cls = getattr(llama_chat_format, LlamaCppSummarizer._REQUIRED_HANDLER_CLASS, None)
        assert cls is not None, (
            f"{LlamaCppSummarizer._REQUIRED_HANDLER_CLASS} is missing from "
            "llama_cpp.llama_chat_format. The current llama-cpp-python "
            "build is stock PyPI rather than the cosmasense fork. "
            f"{LlamaCppSummarizer._FORK_INSTALL_HINT}"
        )

    def test_handler_signature_accepts_clip_and_image_kwargs(self):
        """The MRO walk in `_create_chat_handler` must see clip_model_path
        and image_min_tokens. Qwen3VLChatHandler itself only declares
        `force_reasoning, add_vision_id, **kwargs`, so a leaf-only
        signature scan would silently skip both. If this assertion ever
        flips, the kwargs gating in `_create_chat_handler` will start
        dropping required arguments.
        """
        import inspect
        from llama_cpp import llama_chat_format

        cls = getattr(llama_chat_format, LlamaCppSummarizer._REQUIRED_HANDLER_CLASS)
        accepted: set[str] = set()
        for parent_cls in inspect.getmro(cls):
            try:
                accepted.update(inspect.signature(parent_cls.__init__).parameters.keys())
            except (TypeError, ValueError):
                continue
        assert "clip_model_path" in accepted
        assert "image_min_tokens" in accepted
        assert "verbose" in accepted


@pytest.mark.unit
class TestResolveHandlerClass:
    def test_returns_qwen3vl_when_present(self):
        summarizer = LlamaCppSummarizer.__new__(LlamaCppSummarizer)
        summarizer.chat_handler_name = "qwen3-vl"

        name, cls = summarizer._resolve_handler_class()
        assert name == "Qwen3VLChatHandler"
        assert cls is not None

    def test_returns_none_with_loud_log_when_handler_absent(self, caplog):
        """Reproduce the production failure mode: stock PyPI build with
        no Qwen3VLChatHandler. Confirm we DON'T silently fall back to
        another class — instead we log an actionable error and return
        (None, None) so the caller treats vision as unavailable.
        """
        summarizer = LlamaCppSummarizer.__new__(LlamaCppSummarizer)
        summarizer.chat_handler_name = "qwen3-vl"

        from llama_cpp import llama_chat_format

        with patch.object(llama_chat_format, LlamaCppSummarizer._REQUIRED_HANDLER_CLASS, create=False) as _:
            pass  # noop, just confirms the attr exists

        # Strip the attribute and re-resolve.
        original = getattr(llama_chat_format, LlamaCppSummarizer._REQUIRED_HANDLER_CLASS)
        try:
            delattr(llama_chat_format, LlamaCppSummarizer._REQUIRED_HANDLER_CLASS)
            with caplog.at_level("ERROR"):
                name, cls = summarizer._resolve_handler_class()
        finally:
            setattr(llama_chat_format, LlamaCppSummarizer._REQUIRED_HANDLER_CLASS, original)

        assert name is None
        assert cls is None
        # The error must include the install command so the user knows
        # exactly how to fix it. structlog renders fields into the log
        # record's getMessage(), so check the joined text.
        joined = " ".join(rec.getMessage() for rec in caplog.records)
        assert "cosmasense" in joined
        assert "Qwen3VLChatHandler" in joined


@pytest.mark.unit
class TestCreateChatHandlerKwargs:
    """Guard the kwargs construction. If the handler signature changes
    upstream and stops accepting one of these kwargs, surface it as a
    test failure rather than as silently-broken vision in production.
    """

    def test_construct_with_real_kwargs_does_not_raise_typeerror(self, tmp_path):
        """Construct the handler with our actual kwargs against a dummy
        clip path. The constructor may succeed (cosmasense fork defers
        gguf validation to first use) or raise a runtime error from
        llama.cpp on file format — both are fine. The forbidden outcome
        is TypeError, which would mean the kwargs we're passing are
        unrecognized. That's the exact bug class the user hit on stock
        Qwen25VLChatHandler.
        """
        from llama_cpp import llama_chat_format

        cls = llama_chat_format.Qwen3VLChatHandler
        dummy_clip = tmp_path / "fake_mmproj.gguf"
        dummy_clip.write_bytes(b"GGUF\x00" * 16)

        try:
            cls(
                clip_model_path=str(dummy_clip),
                image_min_tokens=512,
                verbose=False,
            )
        except TypeError as e:
            pytest.fail(
                "Qwen3VLChatHandler rejected one of our kwargs — the "
                "signature has drifted. Check `_create_chat_handler` and "
                f"update the kwargs gating. Error: {e}"
            )
        except Exception:
            # Runtime/load error from llama.cpp on the dummy file is
            # expected and not what we are guarding against here.
            pass
