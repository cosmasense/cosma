"""
Apple Vision framework OCR for scanned PDFs (and raster images).

Why this module exists:
- The original failure report included a PDF (`wechat-channels-en.pdf`)
  that MarkItDown extracted as empty. That's a scanned PDF — pypdf/pdf2image
  can rasterize but there's no text layer to pull, so every extraction
  method returned empty content.
- macOS ships `VNRecognizeTextRequest` (Vision framework) which is fast,
  high-quality, free, and works offline. Calling it through pyobjc adds
  a thin dependency but avoids shipping Tesseract (~60 MB + language
  files) or needing a separate Java runtime for Tika.
- This runs *last* in the parser fallback chain — only when earlier
  methods have given up. That means the expensive rasterize+OCR cost
  is only paid for files that would otherwise fail outright.

Graceful degradation: if pyobjc isn't installed (e.g. non-macOS dev
machine), `is_available()` returns False and the caller skips OCR.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from typing import Optional

from cosma_backend.logging import get_logger

logger = get_logger(__name__)


def is_available() -> bool:
    """True iff Vision-based OCR can run on this machine.

    Checks for macOS + pyobjc-framework-Vision + pyobjc-framework-Quartz
    without actually importing them (import is slow — ~200 ms — and we
    want /api/bootstrap/status to stay cheap).
    """
    if sys.platform != "darwin":
        return False
    for mod in ("Vision", "Quartz"):
        if importlib.util.find_spec(mod) is None:
            return False
    return True


async def ocr_pdf(path: Path, max_pages: int = 20, max_chars: int = 200_000) -> Optional[str]:
    """Run Vision OCR over a scanned PDF. Returns joined text or None.

    `max_pages` / `max_chars` guard against pathological inputs — a
    50-page scanned book would still OCR fast (~1-2 s/page on M-series)
    but we'd rather return enough text for a useful summary than block
    the pipeline. Same reasoning as the Spotlight 4 MB cap.
    """
    if not is_available():
        return None
    try:
        return await asyncio.to_thread(_ocr_pdf_sync, path, max_pages, max_chars)
    except Exception as e:
        logger.warning("Vision OCR failed", path=str(path), error=str(e))
        return None


def _ocr_pdf_sync(path: Path, max_pages: int, max_chars: int) -> Optional[str]:
    """Synchronous OCR — runs in a worker thread via asyncio.to_thread.

    The pyobjc Vision/Quartz APIs are all sync and release the GIL on
    C-code calls, so this is the cheapest wrapping.
    """
    # Imports are deferred to call time so non-macOS dev machines (and
    # test collection) don't pay the ~200 ms pyobjc import cost.
    import Quartz
    import Vision
    from Foundation import NSURL

    url = NSURL.fileURLWithPath_(str(path))
    document = Quartz.CGPDFDocumentCreateWithURL(url)
    if document is None:
        return None

    page_count = min(Quartz.CGPDFDocumentGetNumberOfPages(document), max_pages)
    if page_count == 0:
        return None

    chunks: list[str] = []
    total_chars = 0

    for i in range(1, page_count + 1):
        page = Quartz.CGPDFDocumentGetPage(document, i)
        if page is None:
            continue

        # Render the PDF page to a CGImage at ~200 DPI so text is crisp
        # enough for OCR without using absurd amounts of memory.
        box = Quartz.CGPDFPageGetBoxRect(page, Quartz.kCGPDFCropBox)
        scale = 2.0
        width = int(box.size.width * scale)
        height = int(box.size.height * scale)
        if width < 10 or height < 10:
            continue

        color_space = Quartz.CGColorSpaceCreateDeviceRGB()
        ctx = Quartz.CGBitmapContextCreate(
            None, width, height, 8, 0, color_space,
            Quartz.kCGImageAlphaPremultipliedLast,
        )
        if ctx is None:
            continue
        # White background so anti-aliased black text stays high-contrast.
        Quartz.CGContextSetRGBFillColor(ctx, 1, 1, 1, 1)
        Quartz.CGContextFillRect(ctx, Quartz.CGRectMake(0, 0, width, height))
        Quartz.CGContextScaleCTM(ctx, scale, scale)
        Quartz.CGContextDrawPDFPage(ctx, page)
        image = Quartz.CGBitmapContextCreateImage(ctx)
        if image is None:
            continue

        text = _recognize_text(image, Vision)
        if text:
            chunks.append(text)
            total_chars += len(text)
            if total_chars >= max_chars:
                logger.debug("OCR hit char cap, stopping", pages_done=i, chars=total_chars)
                break

    if not chunks:
        return None
    result = "\n\n".join(chunks).strip()
    logger.info("Vision OCR completed", path=str(path),
                pages=len(chunks), chars=len(result))
    return result or None


def _recognize_text(cg_image, Vision) -> Optional[str]:
    """Run a single VNRecognizeTextRequest on a CGImage. Returns str or None.

    `recognitionLevel=0` is Fast, `1` is Accurate. Accurate is ~2x slower
    but dramatically better on small or rotated text; worth it for the
    "this extraction failed entirely" fallback path.
    """
    import Quartz  # re-import at call time to keep this function self-contained
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(1)   # accurate
    request.setUsesLanguageCorrection_(True)

    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
        cg_image, None,
    )
    ok, err = handler.performRequests_error_([request], None)
    if not ok:
        return None

    observations = request.results()
    if not observations:
        return None

    lines: list[str] = []
    for obs in observations:
        candidates = obs.topCandidates_(1)
        if candidates and candidates.count() > 0:
            s = str(candidates.objectAtIndex_(0).string())
            if s:
                lines.append(s)
    return "\n".join(lines) if lines else None
