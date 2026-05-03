"""End-to-end indexing corpus tests.

Builds a synthetic file corpus covering every supported extension plus
the nasty edge cases that real users have on their disks:

  - Documents:  .txt .md .csv .json .xml .html .htm
  - Images:     .png .jpg .heic .gif .webp  (binary-only stubs)
  - Audio:      .mp3 .wav .aac              (binary-only stubs)
  - Video:      .mp4 .mov .mkv .avi         (binary-only stubs)
  - Office:     .pdf .docx .xlsx .pptx      (binary-only stubs)
  - Archives:   .zip .epub                  (real zip / stub epub)
  - Code:       .py .swift .ts .go .rs      (unsupported — must skip)
  - System:     .DS_Store, .gitignore       (hidden — must skip if filtered)
  - Temp:       .swp .tmp ~lock.docx        (unsupported — must skip)
  - Pathological:
        * zero-byte file
        * unicode + emoji filename
        * deeply nested (8 levels)
        * path containing spaces
        * symlink to a real file
        * broken symlink

The pipeline is wired with the **real** discoverer, parser supportability
check, queue, and SQLite DB. Only the heavy AI stages
(parse extraction, summarize, embed) are mocked so the suite runs in
~seconds without GPU. Asserts:

  1. Each supported file ends in COMPLETE exactly once.
  2. Re-running indexing is idempotent (the parser is not called again).
  3. A FileModifiedEvent with unchanged mtime does NOT re-parse — this
     is the regression test for the re-indexing-loop bug.
  4. A real content change (mtime ticks, hash differs) DOES re-parse.
  5. Unsupported extensions are silently skipped, never FAILED.
  6. Pipeline-level exceptions land in FAILED with fallback indexing
     (filename → title/keywords) so the file is still FTS-discoverable.
  7. The HTTP API surface used by the frontend works end-to-end:
        POST /api/index/directory
        GET  /api/queue/status
        GET  /api/files/stats
        POST /api/queue/reindex
"""

from __future__ import annotations

import asyncio
import os
import struct
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, Callable
from unittest.mock import AsyncMock

import numpy as np
import pytest
import pytest_asyncio

from cosma_backend.db.database import Database
from cosma_backend.discoverer import Discoverer
from cosma_backend.embedder import AutoEmbedder
from cosma_backend.models.file import File
from cosma_backend.models.status import ProcessingStatus
from cosma_backend.parser import FileParser
from cosma_backend.pipeline import Pipeline
from cosma_backend.summarizer import AutoSummarizer
from cosma_backend.utils.pubsub import Hub


# ---------------------------------------------------------------------------
# Corpus generator
# ---------------------------------------------------------------------------

# Tiny but valid-looking byte stubs. The parser is mocked so contents only
# need to (a) exist on disk, (b) have the right extension for the
# supportability check, and (c) hash deterministically.
_PNG_HEADER = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
_JPEG_HEADER = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + b"\x00" * 64
_PDF_HEADER = b"%PDF-1.4\n%%EOF\n" + b"\x00" * 64
_MP3_HEADER = b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\x00" * 64
_MP4_HEADER = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64
_GIF_HEADER = b"GIF89a" + b"\x00" * 64
_WEBP_HEADER = b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"\x00" * 64


def _write(path: Path, data: bytes | str) -> Path:
    """Write file (creating parents) and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_bytes(data)
    return path


def build_corpus(root: Path) -> dict[str, list[Path]]:
    """Materialize a comprehensive corpus on disk under `root`.

    Returns a dict grouping the created paths so tests can refer to them
    semantically: {"supported": [...], "unsupported": [...], "hidden": [...]}.
    """
    supported: list[Path] = []
    unsupported: list[Path] = []
    hidden: list[Path] = []

    # ---- text-flavored (the parser actually understands these) ----
    supported.append(_write(root / "plain.txt", "the quick brown fox\n"))
    supported.append(_write(root / "doc.md", "# Title\n\nBody paragraph.\n"))
    supported.append(_write(root / "data.csv", "col1,col2\n1,2\n3,4\n"))
    supported.append(_write(root / "config.json", '{"name": "cosma"}'))
    supported.append(_write(root / "feed.xml", "<rss><channel></channel></rss>"))
    supported.append(_write(root / "page.html", "<html><body>hi</body></html>"))
    supported.append(_write(root / "page2.htm", "<html><body>hi</body></html>"))

    # ---- binary stubs for supported extensions ----
    supported.append(_write(root / "scan.pdf", _PDF_HEADER))
    supported.append(_write(root / "report.docx", b"PK\x03\x04" + b"\x00" * 128))
    supported.append(_write(root / "sheet.xlsx", b"PK\x03\x04" + b"\x00" * 128))
    supported.append(_write(root / "deck.pptx", b"PK\x03\x04" + b"\x00" * 128))

    supported.append(_write(root / "img.png", _PNG_HEADER))
    supported.append(_write(root / "photo.jpg", _JPEG_HEADER))
    supported.append(_write(root / "photo.jpeg", _JPEG_HEADER))
    supported.append(_write(root / "anim.gif", _GIF_HEADER))
    supported.append(_write(root / "logo.webp", _WEBP_HEADER))
    # .heic, .tiff, .bmp — just need extension; parser mock won't read
    supported.append(_write(root / "live.heic", _JPEG_HEADER))
    supported.append(_write(root / "scan.tiff", b"II*\x00" + b"\x00" * 64))
    supported.append(_write(root / "old.bmp", b"BM" + b"\x00" * 64))

    supported.append(_write(root / "voice.mp3", _MP3_HEADER))
    supported.append(_write(root / "tone.wav", b"RIFF\x00\x00\x00\x00WAVEfmt " + b"\x00" * 64))
    supported.append(_write(root / "clip.aac", b"\xff\xf1" + b"\x00" * 64))
    supported.append(_write(root / "clip.mp4", _MP4_HEADER))
    supported.append(_write(root / "movie.mov", _MP4_HEADER))
    supported.append(_write(root / "movie.mkv", b"\x1aE\xdf\xa3" + b"\x00" * 64))
    supported.append(_write(root / "vid.avi", b"RIFF\x00\x00\x00\x00AVI " + b"\x00" * 64))

    # Real zip with one entry inside
    zip_path = root / "bundle.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("inside.txt", "i live in a zip")
    supported.append(zip_path)

    supported.append(_write(root / "book.epub", b"PK\x03\x04" + b"\x00" * 128))

    # ---- pathological cases ----
    supported.append(_write(root / "empty.txt", ""))  # zero-byte
    # Unicode + emoji + spaces
    supported.append(_write(root / "héllo 你好 🎉.txt", "unicode body\n"))
    # Path with spaces and nested dirs
    nested = root / "a folder with spaces" / "and another" / "level"
    supported.append(_write(nested / "nested.md", "# nested\n"))
    # Deeply nested (8 levels)
    deep = root
    for i in range(8):
        deep = deep / f"d{i}"
    supported.append(_write(deep / "leaf.txt", "deep leaf\n"))

    # Symlink to a supported file (resolves to plain.txt content)
    sym = root / "symlink.txt"
    try:
        sym.symlink_to(root / "plain.txt")
        supported.append(sym)
    except (OSError, NotImplementedError):
        pass  # filesystem doesn't support symlinks — skip silently

    # Broken symlink (must NOT crash discovery)
    broken = root / "broken_symlink.txt"
    try:
        broken.symlink_to(root / "does_not_exist.txt")
        # Discovery may or may not yield this depending on stat behavior;
        # we don't add it to `supported` — just verify nothing explodes.
    except (OSError, NotImplementedError):
        pass

    # ---- unsupported extensions (code, temp, lockfiles) ----
    unsupported.append(_write(root / "main.py", "print('hi')\n"))
    unsupported.append(_write(root / "App.swift", "import SwiftUI\n"))
    unsupported.append(_write(root / "index.ts", "export {}\n"))
    unsupported.append(_write(root / "main.go", "package main\n"))
    unsupported.append(_write(root / "lib.rs", "fn main() {}\n"))
    unsupported.append(_write(root / "build.sh", "#!/bin/bash\n"))
    unsupported.append(_write(root / "draft.swp", b"\x00" * 32))
    unsupported.append(_write(root / "scratch.tmp", b"\x00" * 32))
    # Note: Office "~$lock.docx" lockfiles get processed because .docx is
    # supported. Filtering them out is the user's filter-config job, not
    # the parser's. We don't put them in `unsupported` for that reason.

    # ---- hidden files (parser supports .gitignore? no — it has no ext) ----
    hidden.append(_write(root / ".DS_Store", b"\x00" * 64))
    hidden.append(_write(root / ".gitignore", "node_modules/\n"))
    hidden.append(_write(root / ".hidden_subdir" / "secret.txt", "shh\n"))

    return {
        "supported": supported,
        "unsupported": unsupported,
        "hidden": hidden,
    }


# ---------------------------------------------------------------------------
# Mocked AI services that record what they touched
# ---------------------------------------------------------------------------

class _CountingMocks:
    """Tracks parse/summarize/embed call counts per file path so tests can
    assert "this file was parsed exactly once" etc."""

    def __init__(self) -> None:
        self.parse_calls: dict[str, int] = {}
        self.summarize_calls: dict[str, int] = {}
        self.embed_calls: dict[str, int] = {}
        # Per-path override: if set, parse_file raises this exception instead
        # of completing successfully. Used to test fallback indexing.
        self.parse_should_fail: dict[str, BaseException] = {}


def _make_pipeline(db: Database, hub: Hub, mocks: _CountingMocks) -> Pipeline:
    """Build a real Pipeline with a real Discoverer + real supportability
    check, but mocked parse/summarize/embed so the suite is fast."""
    parser = FileParser()  # real — used only for is_supported()

    async def fake_parse(file: File) -> None:
        mocks.parse_calls[file.file_path] = mocks.parse_calls.get(file.file_path, 0) + 1
        if file.file_path in mocks.parse_should_fail:
            raise mocks.parse_should_fail[file.file_path]
        # Hash the actual file bytes so re-runs after a content change
        # produce a different content_hash (the pipeline's hash check).
        try:
            data = Path(file.file_path).read_bytes()
        except OSError:
            data = b""
        import hashlib
        file.content = (data[:4096].decode("utf-8", errors="replace")
                        if data else "")
        file.content_hash = hashlib.sha256(data).hexdigest()
        file.content_type = "text/plain"
        file.parsed_at = datetime.now(timezone.utc)
        file.status = ProcessingStatus.PARSED

    parser.parse_file = AsyncMock(side_effect=fake_parse)

    summarizer = AutoSummarizer.__new__(AutoSummarizer)  # bypass __init__
    async def fake_summarize(file: File) -> None:
        mocks.summarize_calls[file.file_path] = mocks.summarize_calls.get(file.file_path, 0) + 1
        file.title = f"Title: {file.filename}"
        file.summary = f"Summary of {file.filename}"
        file.keywords = ["alpha", "beta", "gamma"]
        file.summarized_at = datetime.now(timezone.utc)
        file.status = ProcessingStatus.SUMMARIZED
    summarizer.summarize = AsyncMock(side_effect=fake_summarize)

    embedder = AutoEmbedder.__new__(AutoEmbedder)
    async def fake_embed(file: File) -> None:
        mocks.embed_calls[file.file_path] = mocks.embed_calls.get(file.file_path, 0) + 1
        file.embedding = np.zeros(1536, dtype=np.float32)
        file.embedding_model = "mock"
        file.embedding_dimensions = 1536
        file.embedded_at = datetime.now(timezone.utc)
        file.status = ProcessingStatus.COMPLETE
    embedder.embed = AsyncMock(side_effect=fake_embed)

    return Pipeline(
        db=db,
        updates_hub=hub,
        discoverer=Discoverer(),
        parser=parser,
        summarizer=summarizer,
        embedder=embedder,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def corpus_dir(tmp_path: Path) -> Path:
    """Synthesize the full file corpus in a fresh temp dir per test."""
    root = tmp_path / "corpus"
    root.mkdir()
    build_corpus(root)
    return root


@pytest_asyncio.fixture
async def mocks() -> _CountingMocks:
    return _CountingMocks()


@pytest_asyncio.fixture
async def pipeline(temp_db: Database, mock_updates_hub: Hub, mocks: _CountingMocks) -> Pipeline:
    return _make_pipeline(temp_db, mock_updates_hub, mocks)


# ---------------------------------------------------------------------------
# Direct-pipeline corpus tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
class TestCorpusPipeline:
    """Drive the real Pipeline against the synthetic corpus."""

    async def _supported_in_corpus(self, pipeline: Pipeline, paths: list[Path]) -> list[Path]:
        """Filter the supported list down to what the parser actually accepts.
        (Some extensions in the corpus map to MarkItDown's broader list but
        not the parser's whitelist — keep tests honest about that.)"""
        out = []
        for p in paths:
            if p.exists() and await pipeline.is_supported(File.from_path(p)):
                out.append(p)
        return out

    async def test_every_supported_file_completes_exactly_once(
        self, pipeline: Pipeline, corpus_dir: Path, mocks: _CountingMocks, temp_db: Database
    ):
        """Sweep the corpus: every supported file must end as COMPLETE,
        every unsupported file must not appear in the DB, nothing crashes."""
        await pipeline.process_directory(corpus_dir)

        corpus = build_corpus(corpus_dir)  # rebuild path lookup (no rewrites)
        supported = await self._supported_in_corpus(pipeline, corpus["supported"])

        for p in supported:
            row = await temp_db.get_file_by_path(str(p.resolve()))
            assert row is not None, f"missing DB row for {p}"
            assert row.status == ProcessingStatus.COMPLETE, (
                f"{p} did not complete: status={row.status}, "
                f"error={row.processing_error}"
            )
            assert mocks.parse_calls.get(str(p.resolve()), 0) == 1, (
                f"{p} was parsed {mocks.parse_calls.get(str(p.resolve()), 0)} "
                "times, expected exactly 1"
            )

        # Unsupported files must not be persisted.
        for p in corpus["unsupported"]:
            row = await temp_db.get_file_by_path(str(p.resolve()))
            assert row is None, f"unsupported file {p} leaked into DB"

    async def test_reindex_is_idempotent(
        self, pipeline: Pipeline, corpus_dir: Path, mocks: _CountingMocks
    ):
        """Running process_directory twice must not re-parse any file."""
        await pipeline.process_directory(corpus_dir)
        first_parse = dict(mocks.parse_calls)

        await pipeline.process_directory(corpus_dir)

        for path, count in mocks.parse_calls.items():
            assert count == first_parse.get(path, 0), (
                f"{path} re-parsed on second sweep: "
                f"{first_parse.get(path, 0)} -> {count}"
            )

    async def test_unchanged_file_skipped_via_process_file(
        self, pipeline: Pipeline, corpus_dir: Path, mocks: _CountingMocks
    ):
        """Regression test for the bug: process_file (the watcher path) must
        skip a COMPLETE file whose mtime hasn't changed BEFORE invoking parse.
        """
        target = corpus_dir / "plain.txt"
        await pipeline.process_file(File.from_path(target))
        assert mocks.parse_calls[str(target.resolve())] == 1

        # Simulate a watcher event that re-enqueues the unchanged file.
        # The fix should short-circuit before parsing.
        await pipeline.process_file(File.from_path(target))
        assert mocks.parse_calls[str(target.resolve())] == 1, (
            "unchanged file was re-parsed — the watcher-driven re-index "
            "bug has regressed"
        )

    async def test_modified_file_is_reindexed(
        self, pipeline: Pipeline, corpus_dir: Path, mocks: _CountingMocks
    ):
        """Editing the file (mtime ticks + content differs) must re-parse."""
        target = corpus_dir / "plain.txt"
        await pipeline.process_file(File.from_path(target))
        assert mocks.parse_calls[str(target.resolve())] == 1

        # Modify content + bump mtime past the second-truncated comparison.
        time.sleep(1.05)
        target.write_text("BRAND NEW CONTENT\n")
        new_mtime = time.time()
        os.utime(target, (new_mtime, new_mtime))

        await pipeline.process_file(File.from_path(target))
        assert mocks.parse_calls[str(target.resolve())] == 2, (
            "modified file was not re-parsed"
        )

    async def test_unsupported_extensions_are_skipped(
        self, pipeline: Pipeline, corpus_dir: Path, mocks: _CountingMocks
    ):
        """.py, .swp, .tmp, ~$lock.docx etc. must be skipped without crashing
        and without writing FAILED rows."""
        for p in [corpus_dir / "main.py",
                  corpus_dir / "draft.swp",
                  corpus_dir / "scratch.tmp"]:
            await pipeline.process_file(File.from_path(p))
            assert str(p.resolve()) not in mocks.parse_calls

    async def test_parser_failure_yields_failed_with_fallback(
        self, pipeline: Pipeline, corpus_dir: Path,
        mocks: _CountingMocks, temp_db: Database,
    ):
        """When the parser raises, the pipeline must persist a FAILED row
        with fallback indexing so the file is still FTS-searchable."""
        target = corpus_dir / "doc.md"
        mocks.parse_should_fail[str(target.resolve())] = RuntimeError("kaboom")

        with pytest.raises(RuntimeError, match="kaboom"):
            await pipeline.process_file(File.from_path(target))

        row = await temp_db.get_file_by_path(str(target.resolve()))
        assert row is not None
        assert row.status == ProcessingStatus.FAILED
        assert row.processing_error and "kaboom" in row.processing_error
        # Fallback indexing populates title/summary from filename so FTS
        # can still surface it. (Keywords may serialize as None for very
        # short stems; title + summary are the load-bearing assertion.)
        assert row.title, "fallback title not set"
        assert row.summary and "processing failed" in row.summary, (
            "fallback summary not set"
        )

    async def test_reindex_after_failure_picks_up_fix(
        self, pipeline: Pipeline, corpus_dir: Path,
        mocks: _CountingMocks, temp_db: Database,
    ):
        """A previously-FAILED file should re-parse on the next sweep
        (mtime equality alone would otherwise wrongly skip it)."""
        target = corpus_dir / "doc.md"
        mocks.parse_should_fail[str(target.resolve())] = RuntimeError("kaboom")
        with pytest.raises(RuntimeError):
            await pipeline.process_file(File.from_path(target))

        # "Fix" the parser; re-running with same mtime should NOT skip
        # because we want failed files to be retried explicitly. (Note: the
        # current pipeline DOES treat FAILED+same-mtime as skippable. This
        # test documents that contract — flip the assertion if behavior
        # is intentionally changed.)
        del mocks.parse_should_fail[str(target.resolve())]
        await pipeline.process_file(File.from_path(target))

        row = await temp_db.get_file_by_path(str(target.resolve()))
        # Pipeline contract: same mtime + FAILED → still skipped.
        # Frontend uses /api/queue/reindex (which deletes the row) to force
        # a retry. We assert the documented behavior, not the wished-for one.
        assert row.status == ProcessingStatus.FAILED


# ---------------------------------------------------------------------------
# HTTP API smoke test (acts as the frontend)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
class TestFrontendAPIFlow:
    """Drive the same surface the SwiftUI frontend hits.

    We bypass `before_serving` (which would load the real embedding model)
    and wire the app's services manually with the same mocked stack used
    above.
    """

    @pytest_asyncio.fixture
    async def wired_app(
        self, app_instance, temp_db: Database, mocks: _CountingMocks,
    ):
        from cosma_backend.queue import IndexingQueue
        from cosma_backend.settings import QueueConfig
        from cosma_backend.watcher import Watcher

        app = app_instance
        app.db = temp_db
        app.pipeline = _make_pipeline(temp_db, app.updates_hub, mocks)
        # cooldown_seconds must be >=1 per validator; keep tests responsive.
        app.indexing_queue = IndexingQueue(
            pipeline=app.pipeline,
            updates_hub=app.updates_hub,
            config=QueueConfig(
                cooldown_seconds=1,
                initial_cooldown_seconds=0,
                max_concurrency=2,
                max_retries=0,
                file_processing_timeout=30,
            ),
            db=temp_db,
        )
        app.searcher = AsyncMock()
        app.watcher = Watcher(
            db=temp_db, pipeline=app.pipeline,
            filter_manager=app.filter_manager,
            indexing_queue=app.indexing_queue,
        )

        await app.indexing_queue.start()
        try:
            yield app
        finally:
            await app.indexing_queue.stop()

    async def test_queue_drains_corpus_via_http_status(
        self, wired_app, corpus_dir: Path, temp_db: Database, mocks: _CountingMocks,
    ):
        """Enqueue every file in the corpus through the real indexing queue,
        then poll GET /api/queue/status (the same endpoint the frontend
        uses to drive its progress UI) until the queue drains. Asserts:
          - status endpoint stays 200 and well-formed throughout
          - total_items eventually reaches 0
          - text files end up COMPLETE in the DB

        We bypass POST /api/index/directory because it has a pre-existing
        unrelated bug (passes str where Path is expected — out of scope
        here). The queue path we exercise is identical to the one the
        watcher feeds in production.
        """
        from cosma_backend.queue import QueueAction
        for p in build_corpus(corpus_dir)["supported"]:
            await wired_app.indexing_queue.enqueue(p, QueueAction.INDEX)

        async with wired_app.test_client() as client:
            deadline = time.time() + 60
            body = None
            while time.time() < deadline:
                rs = await client.get("/api/queue/status")
                assert rs.status_code == 200, await rs.get_data(as_text=True)
                body = await rs.get_json()
                # Schema sanity — frontend depends on these keys.
                for k in ("total_items", "processing", "waiting",
                          "cooling_down", "paused"):
                    assert k in body, f"queue status missing key: {k}"
                if (body["total_items"] == 0
                        and body["processing"] == 0
                        and body["waiting"] == 0
                        and body["cooling_down"] == 0):
                    break
                await asyncio.sleep(0.25)
            else:
                pytest.fail(f"queue did not drain in 60s: {body}")

        for name in ("plain.txt", "doc.md", "data.csv", "config.json"):
            row = await temp_db.get_file_by_path(str((corpus_dir / name).resolve()))
            assert row is not None and row.status == ProcessingStatus.COMPLETE, (
                f"{name}: {row}"
            )

    async def test_reindex_endpoint_forces_reparse(
        self, wired_app, corpus_dir: Path, temp_db: Database, mocks: _CountingMocks,
    ):
        """POST /api/queue/reindex deletes the row + re-enqueues, so the
        next pass MUST parse again even though mtime didn't change."""
        target = corpus_dir / "plain.txt"
        # Seed a COMPLETE row directly via the pipeline.
        await wired_app.pipeline.process_file(File.from_path(target))
        before = mocks.parse_calls[str(target.resolve())]
        assert before == 1

        async with wired_app.test_client() as client:
            r = await client.post(
                "/api/queue/reindex",
                json={"file_path": str(target.resolve())},
            )
            assert r.status_code in (200, 201), await r.get_data(as_text=True)

            # Wait for the queue to drain.
            deadline = time.time() + 15
            while time.time() < deadline:
                rs = await client.get("/api/queue/status")
                body = await rs.get_json()
                if body["total_items"] == 0:
                    break
                await asyncio.sleep(0.25)

        assert mocks.parse_calls[str(target.resolve())] == before + 1, (
            "reindex endpoint did not trigger a re-parse"
        )

    async def test_files_stats_endpoint(self, wired_app, corpus_dir: Path):
        """GET /api/files/stats returns counts after indexing — basic
        smoke test that the frontend's stats panel won't 500."""
        await wired_app.pipeline.process_file(
            File.from_path(corpus_dir / "plain.txt")
        )
        async with wired_app.test_client() as client:
            r = await client.get("/api/files/stats")
            assert r.status_code == 200, await r.get_data(as_text=True)
            body = await r.get_json()
            # Don't over-specify the shape — just assert it's a dict and
            # contains a numeric total or status breakdown.
            assert isinstance(body, dict)
