"""Unit tests for the partial_kind classification + INDEXED_PARTIAL routing.

The original failure report had four files mislabelled as FAILED that
weren't actually failures:

  * 20 GB Blu-ray remux exceeding parser.max_file_size_mb
    → was tagged partial_kind="parser_failed" → status=FAILED
    → now tagged partial_kind="oversize" → status=INDEXED_PARTIAL
  * Blank PDF where every extractor returned empty
    → was tagged partial_kind="parser_failed" → status=FAILED
    → now tagged partial_kind="no_content" → status=INDEXED_PARTIAL
  * XLSX with no MarkItDown converter / openpyxl fallback exhausted
    → same: now "no_content" → INDEXED_PARTIAL

Genuine parser failures (codec unreadable even after H.264 transcode,
ffmpeg unavailable) still map to "parser_failed" → FAILED. The Failed
tab now means "something actually went wrong", not "we didn't bother."

These tests don't spin up the full Pipeline (slow, needs mocks for
embedder + summarizer); they pin the parser-side classification and
the pipeline's routing table separately.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from cosma_backend.models.file import File
from cosma_backend.parser.parser import FileParser


@pytest.mark.unit
class TestParserPartialKindClassification:
    """The parser is the authority on WHY a file is partial. Pin which
    `partial_kind` value each failure mode emits."""

    @pytest.mark.asyncio
    async def test_oversize_file_classified_as_oversize(self, tmp_path):
        p = tmp_path / "huge_movie.mkv"
        p.write_bytes(b"\x00" * 32)
        file_obj = File.from_path(p)
        # Lie about the size to trip the cap without writing 20 GB to disk.
        file_obj.file_size = (20480 + 1) * 1024 * 1024

        parser = FileParser()
        out = await parser.parse_file(file_obj)

        assert out.metadata_only is True
        assert out.partial_kind == "oversize"
        assert "parse limit" in (out.metadata_only_reason or "")

    @pytest.mark.asyncio
    async def test_empty_extraction_classified_as_no_content(self, tmp_path):
        # An XLSX whose every extraction method comes back empty. We
        # patch the cascade to skip MarkItDown / openpyxl / media so
        # we land in the "every parser returned empty" branch
        # deterministically — without this we'd be at the mercy of
        # MarkItDown happening to extract something from a real blank
        # workbook.
        p = tmp_path / "blank.pdf"
        p.write_bytes(b"%PDF-1.4 fake header for is_supported gate")
        file_obj = File.from_path(p)

        parser = FileParser()

        async def _empty(*_a, **_kw):
            return None

        with patch.object(parser, "_try_spotlight_extraction", _empty), \
             patch.object(parser, "_try_media_extraction", _empty), \
             patch.object(parser, "_try_markitdown_extraction", _empty), \
             patch.object(parser, "_try_pypdf_extraction", _empty):
            out = await parser.parse_file(file_obj)

        assert out.metadata_only is True
        assert out.partial_kind == "no_content"
        assert "appears empty" in (out.metadata_only_reason or "")

    @pytest.mark.asyncio
    async def test_user_elected_partial_kind_is_preserved(self, tmp_path):
        """The discoverer marks user-elected metadata-only files BEFORE
        the parser runs. If the parser then hits its oversize/empty
        branch, the user's choice must win — overwriting it would change
        the file's final status from INDEXED_PARTIAL to FAILED."""
        p = tmp_path / "huge_user_elected.mkv"
        p.write_bytes(b"\x00" * 32)
        file_obj = File.from_path(p)
        file_obj.file_size = (20480 + 1) * 1024 * 1024
        file_obj.partial_kind = "user_elected"

        parser = FileParser()
        out = await parser.parse_file(file_obj)

        assert out.partial_kind == "user_elected"


@pytest.mark.unit
class TestPipelinePartialKindRouting:
    """The pipeline turns partial_kind into the final ProcessingStatus.
    Pin the routing table so a future refactor can't silently re-stuff
    by-design partials back into the Failed tab.
    """

    def test_routing_table_is_explicit(self):
        """Reflects the contract in pipeline.py: only "parser_failed"
        (and any unknown value, defensively) ends up FAILED. The other
        three values are by-design partials.
        """
        from cosma_backend.models.status import ProcessingStatus
        from cosma_backend.pipeline import pipeline as pipeline_module
        import inspect

        # Inline-source assertion is intentionally crude: a behavior test
        # would require spinning up a real Pipeline. This catches the
        # most likely regression — someone adding a new partial_kind and
        # forgetting to add it to the _BY_DESIGN_PARTIAL set.
        src = inspect.getsource(pipeline_module)
        assert '_BY_DESIGN_PARTIAL = {"user_elected", "oversize", "no_content"}' in src, (
            "If you added or renamed a partial_kind, update the "
            "_BY_DESIGN_PARTIAL set in pipeline.py and this test."
        )
        # Sanity-check the status enum hasn't been renamed out from under us.
        assert ProcessingStatus.INDEXED_PARTIAL is not None
        assert ProcessingStatus.FAILED is not None
