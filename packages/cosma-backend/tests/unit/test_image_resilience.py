"""Unit tests for the summarizer's image-prep resilience.

Real production failures we hit:

  LlamaCppSummarizer summarization failed: cannot identify image file
  <_io.BytesIO object at 0x...>

These were unusually-encoded / corrupted JPEGs (manga page scans saved
by a scraper) handed straight into llama-cpp-python's mtmd handler,
which internally calls `PIL.Image.open()` and dies with
`UnidentifiedImageError` on bad bytes — crashing the entire summarize
call and bubbling up as a hard-FAILED file.

The fix is to round-trip every image through Pillow BEFORE base64-
encoding so we catch the decode failure on OUR side and fall back to
a text-only summary (still produces a useful index entry from filename
and content_type) instead of crashing.

LOAD_TRUNCATED_IMAGES lets us recover lightly-truncated JPEGs (the
last few percent missing); files truncated more heavily, or in a
format Pillow can't identify, are skipped via the None-return contract.
"""

import base64
import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from cosma_backend.summarizer.base import BaseSummarizer


class _ConcreteSummarizer(BaseSummarizer):
    """Test-only concrete subclass. BaseSummarizer is abstract — we
    need *something* that satisfies the ABC to instantiate via __new__
    without arguments. The abstract methods are never called."""

    async def is_available(self) -> bool:
        return True

    async def _get_ai_response(self, content, file_metadata, images=None):
        return ""


def _png_bytes(width: int = 8, height: int = 8) -> bytes:
    img = Image.new("RGB", (width, height), color=(123, 200, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_bytes(width: int = 8, height: int = 8) -> bytes:
    img = Image.new("RGB", (width, height), color=(200, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


@pytest.mark.unit
class TestTranscodeToJpeg:
    def test_valid_png_transcodes_to_jpeg_b64(self, tmp_path):
        p = tmp_path / "ok.png"
        p.write_bytes(_png_bytes())

        result = BaseSummarizer._transcode_to_jpeg_b64(p)

        assert result is not None
        decoded = base64.b64decode(result)
        assert decoded[:3] == b"\xff\xd8\xff"  # JPEG SOI marker

    def test_lightly_truncated_jpeg_decodes_via_load_truncated(self, tmp_path):
        """A JPEG missing only the last few bytes (e.g., scraper cut
        the connection right before EOI) — LOAD_TRUNCATED_IMAGES lets
        Pillow yield the partially-decoded image instead of erroring.
        Severely-truncated JPEGs are handled by the None-return path
        (see test_completely_unidentifiable_returns_none_without_raising).
        """
        full = _jpeg_bytes(width=128, height=128)
        truncated = full[:-2]  # drop EOI marker
        p = tmp_path / "truncated.jpg"
        p.write_bytes(truncated)

        result = BaseSummarizer._transcode_to_jpeg_b64(p)

        assert result is not None
        decoded = base64.b64decode(result)
        assert decoded[:3] == b"\xff\xd8\xff"

    def test_completely_unidentifiable_returns_none_without_raising(self, tmp_path):
        """The actual production failure: a .jpg file Pillow can't
        identify at all (severely truncated, wrong magic bytes, or some
        oddball encoding). The contract is: log a warning and return
        None so the caller falls back to text-only summarization.
        Never raise — that's what was crashing the summarize call.
        """
        p = tmp_path / "garbage.jpg"
        p.write_bytes(b"this is not an image, not even close, just text")

        result = BaseSummarizer._transcode_to_jpeg_b64(p)
        assert result is None

    def test_missing_file_returns_none(self, tmp_path):
        result = BaseSummarizer._transcode_to_jpeg_b64(tmp_path / "nope.png")
        assert result is None

    def test_alpha_channel_is_flattened_to_rgb(self, tmp_path):
        """JPEG can't store alpha; the .convert("RGB") guard prevents
        a "cannot write mode RGBA as JPEG" save error from leaking out."""
        p = tmp_path / "rgba.png"
        img = Image.new("RGBA", (16, 16), color=(50, 50, 200, 128))
        img.save(str(p), format="PNG")

        result = BaseSummarizer._transcode_to_jpeg_b64(p)
        assert result is not None


@pytest.mark.unit
class TestPrepareImagesGracefulDegrade:
    """End-to-end: _prepare_images is what llama-cpp-python eventually
    sees. A bad image must NOT raise — it must just yield no images so
    the summarizer falls back to text-only.
    """

    @pytest.mark.asyncio
    async def test_bad_image_yields_empty_list_not_exception(self, tmp_path):
        p = tmp_path / "manga_scan.jpg"
        p.write_bytes(b"truncated downloaded by scraper " * 4)

        file_metadata = SimpleNamespace(
            content_type="image/jpeg",
            path=p,
            extra_images=None,
        )
        summarizer = _ConcreteSummarizer()
        result = await summarizer._prepare_images(file_metadata)

        assert result == []

    @pytest.mark.asyncio
    async def test_good_image_round_trips_through_pillow(self, tmp_path):
        """Important: even a perfectly fine JPEG now round-trips through
        Pillow (so the bytes the vision model receives are guaranteed
        decodable). Verify the output is a non-empty base64 JPEG.
        """
        p = tmp_path / "fine.jpg"
        p.write_bytes(_jpeg_bytes(width=32, height=32))

        file_metadata = SimpleNamespace(
            content_type="image/jpeg",
            path=p,
            extra_images=None,
        )
        summarizer = _ConcreteSummarizer()
        result = await summarizer._prepare_images(file_metadata)

        assert len(result) == 1
        decoded = base64.b64decode(result[0])
        assert decoded[:3] == b"\xff\xd8\xff"

    @pytest.mark.asyncio
    async def test_video_frames_still_pass_through(self, tmp_path):
        """extra_images (already in-memory JPEG-encoded video frames)
        must pass through untouched — they've already been validated by
        the video extractor.
        """
        frame_bytes = _jpeg_bytes()
        file_metadata = SimpleNamespace(
            content_type=None,
            path=Path("/dev/null"),
            extra_images=[frame_bytes, frame_bytes],
        )
        summarizer = _ConcreteSummarizer()
        result = await summarizer._prepare_images(file_metadata)

        assert len(result) == 2
        for b64 in result:
            assert base64.b64decode(b64) == frame_bytes
