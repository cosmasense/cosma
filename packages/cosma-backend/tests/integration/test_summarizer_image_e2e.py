"""
End-to-end image summarization test.

Spins up the real LlamaCppSummarizer (downloads + loads Qwen3-VL +
mmproj on first run, ~2 GB), feeds it a synthetic test image with a
bright red square plus the text 'HELLO COSMA', and asserts the
returned summary actually describes visual content from the pixels —
not the parser-metadata placeholder you get when vision is off.

Marked `slow` so it stays out of default unit/integration runs. Invoke
explicitly:

    uv run --group test pytest tests/integration/test_summarizer_image_e2e.py \
        -m slow --no-cov -v -s
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

import pytest

from cosma_backend.models import ProcessingStatus
from cosma_backend.models.file import File
from cosma_backend.summarizer.providers import LlamaCppSummarizer


def _render_test_image(target: Path) -> None:
    """Render a small distinctive test image: red square on white +
    bold 'HELLO COSMA' text. Both signals are easy for Qwen3-VL to
    pick up. We assert later that the summary mentions at least one
    of the words we can only know from looking at the pixels.
    """
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (640, 360), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([(60, 60), (260, 260)], fill=(220, 30, 30))

    text = "HELLO COSMA"
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 56)
    except OSError:
        font = ImageFont.load_default()
    draw.text((300, 140), text, fill=(0, 0, 0), font=font)

    img.save(target, format="JPEG", quality=92)


@pytest.fixture(scope="module")
def test_image(tmp_path_factory) -> Path:
    target_dir = tmp_path_factory.mktemp("vision_e2e")
    target = target_dir / "hello_cosma.jpg"
    _render_test_image(target)
    return target


def _file_for(image_path: Path) -> File:
    stat = image_path.stat()
    return File(
        path=image_path,
        file_path=str(image_path),
        filename=image_path.name,
        extension=image_path.suffix,
        file_size=stat.st_size,
        created=datetime.fromtimestamp(stat.st_ctime),
        modified=datetime.fromtimestamp(stat.st_mtime),
        accessed=datetime.fromtimestamp(stat.st_atime),
        content_type="image/jpeg",
        # Parser placeholder for image files. Vision should let the
        # summarizer ignore this and describe the actual pixels.
        content=(
            f"JPEG image file: 640x360 pixels, {stat.st_size} bytes, RGB mode"
        ),
        status=ProcessingStatus.PARSED,
    )


@pytest.mark.slow
@pytest.mark.asyncio
async def test_qwen3vl_summarizes_real_image_pixels(test_image: Path):
    """Production failure mode reproduced end-to-end: when vision is
    off, the summarizer parrots the parser-metadata placeholder
    ('JPEG image file: NxN pixels, ... RGB mode'). When vision works,
    the summary describes the actual pixels — text, color, or shape
    we baked into the image. This test fails loudly in either of the
    two ways the bug manifested: handler not loaded, or handler loaded
    but never sees the image bytes.
    """
    summarizer = LlamaCppSummarizer()

    # Sanity gate — skip cleanly if the local llama.cpp build doesn't
    # ship Qwen3VLChatHandler (i.e. you're on stock PyPI without the
    # cosmasense fork). The unit-level test_summarizer_vision.py
    # covers that case explicitly; this test is about end-to-end.
    name, cls = summarizer._resolve_handler_class()
    if cls is None:
        pytest.skip(
            f"{LlamaCppSummarizer._REQUIRED_HANDLER_CLASS} not present in "
            "this llama-cpp-python build — install the cosmasense fork "
            "to run this end-to-end test."
        )

    file_meta = _file_for(test_image)
    result = await summarizer.summarize(file_meta)

    summary = (result.summary or "").lower()
    title = (result.title or "").lower()
    keywords = " ".join(result.keywords or []).lower()
    blob = " ".join([summary, title, keywords])

    # The placeholder text must NOT be the only thing the summary
    # parrots back. If vision is silently off, the model has nothing
    # else to go on, and the parser placeholder ('JPEG image file:
    # 640x360 pixels, ...') leaks through verbatim.
    placeholder = "jpeg image file:"
    assert placeholder not in summary, (
        "Summary contains the parser-metadata placeholder verbatim "
        "— vision is off and the model is parroting the text content. "
        f"Full summary: {result.summary!r}"
    )

    # Positive evidence: the model must mention something it could
    # only know from looking at the pixels. Any one of these is fine:
    # OCR'd text or visual color/shape vocabulary.
    visual_signals = [
        "hello", "cosma",                       # text we rendered
        "red", "crimson",                       # the square's color
        "square", "rectangle", "shape",         # the square's shape
        "text", "letters", "writing", "word",   # OCR'able content
    ]
    matches = [w for w in visual_signals if w in blob]
    assert matches, (
        "Summary contains no visual vocabulary — model likely never "
        "received the image bytes. Expected one of "
        f"{visual_signals}. Got: title={result.title!r} "
        f"summary={result.summary!r} keywords={result.keywords!r}"
    )
    print(
        f"\n[vision-e2e] matched visual signals: {matches}\n"
        f"  title:    {result.title!r}\n"
        f"  summary:  {result.summary!r}\n"
        f"  keywords: {result.keywords!r}"
    )
