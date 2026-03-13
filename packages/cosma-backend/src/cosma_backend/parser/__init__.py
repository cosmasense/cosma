"""
Parser Module

Extracts text content from various file formats using a hybrid strategy:

1. macOS Spotlight (fast, pre-indexed content if available)
2. Media extraction (audio transcription, video frames)
3. MarkItDown fallback (supports 20+ formats)

Supported formats:
- Documents: PDF, DOCX, PPTX, XLSX, EPUB
- Images: PNG, JPG, HEIC, etc. (metadata + vision analysis)
- Audio: MP3, WAV, AAC (Whisper transcription)
- Video: MP4, AVI, MOV (transcript + frame extraction)
- Web: HTML, XML, JSON
- Archives: ZIP (extracts and processes contents)

Usage:
    parser = FileParser()
    file = await parser.parse_file(file_metadata)
"""

from .parser import (
    FileParser,
    get_supported_extensions,
    is_supported_file,
)

__all__ = [
    "FileParser",
    "get_supported_extensions",
    "is_supported_file",
]
