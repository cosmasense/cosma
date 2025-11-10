import hashlib
import logging
import mimetypes
import os
from pathlib import Path

from markitdown import MarkItDown

from backend.logging import sm
from backend.models import File, ProcessingStatus
from backend.parser.spotlight import spotlight_to_text
from backend.parser.media import (
    extract_audio_transcript, 
    extract_video_transcript, 
    extract_image_info,
    is_supported_media_file
)

# Configure structured logger
logger = logging.getLogger(__name__)


class FileParser:
    """
    High-performance file parser that converts various file formats to markdown.

    Uses Microsoft's MarkItDown library to support 20+ file formats including:
    - Documents: PDF, DOCX, PPTX, XLSX
    - Images: PNG, JPG, JPEG, TIFF, BMP, HEIC (with OCR)
    - Audio: MP3, WAV, AAC (with transcription)
    - Web: HTML pages
    - Text: CSV, JSON, XML, TXT
    - Archives: ZIP (processes contents)
    - And more formats...
    """

    # Supported file extensions (MarkItDown supports many more)
    SUPPORTED_EXTENSIONS = {
        # Documents
        ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
        # Images
        ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".heic", ".gif", ".webp",
        # Audio/Video
        ".mp3", ".wav", ".aac", ".mp4", ".avi", ".mov", ".mkv",
        # Web & Text
        ".html", ".htm", ".txt", ".csv", ".json", ".xml", ".md",
        # Archives
        ".zip", ".epub"
    }

    def __init__(self) -> None:
        """Initialize the parser with MarkItDown."""
        try:
            self.markitdown = MarkItDown()
            
            # Phase 2: Extraction strategy configuration
            self.extraction_strategy = os.getenv("EXTRACTION_STRATEGY", "spotlight_first")
            self.spotlight_enabled = os.getenv("SPOTLIGHT_ENABLED", "true").lower() == "true"
            
            # Statistics tracking
            self.extraction_stats = {
                "spotlight_success": 0,
                "spotlight_failed": 0,
                "markitdown_success": 0,
                "markitdown_failed": 0,
                "media_success": 0,
                "media_failed": 0
            }
            
            logger.info(sm("FileParser initialized successfully",
                          extraction_strategy=self.extraction_strategy,
                          spotlight_enabled=self.spotlight_enabled))
        except Exception as e:
            logger.exception(sm("Failed to initialize MarkItDown", error=str(e)))
            raise

    def is_supported(self, file: File) -> bool:
        """
        Check if a file format is supported by the parser.

        Args:
            file_path: Path to the file to check

        Returns:
            True if the file format is supported, False otherwise
        """
        extension = file.path.suffix.lower()
        return extension in self.SUPPORTED_EXTENSIONS

    def get_supported_extensions(self) -> set[str]:
        """Get the set of supported file extensions."""
        return self.SUPPORTED_EXTENSIONS.copy()

    def _calculate_content_hash(self, content: str) -> str:
        """Calculate SHA-256 hash of content for deduplication."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _detect_mime_type(self, file_path: Path) -> str | None:
        """Detect MIME type of a file."""
        try:
            mime_type, _ = mimetypes.guess_type(str(file_path))
            return mime_type
        except Exception:
            return None

    async def parse_file(self, file: File) -> File:
        """
        Parse a single file using hybrid extraction strategy (Phase 2).
        
        Extraction order:
        1. Spotlight extraction (macOS) - if enabled and available
        2. Media processing (audio/video/image) - if media file
        3. MarkItDown fallback - for all other supported formats

        Args:
            file_path: Path to the file to parse
            title: Optional custom title for the file

        Returns:
            File object with parsed content

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is not supported
            Exception: For other parsing errors
        """
        logger.info(sm("Starting hybrid file parsing", 
                       file_path=str(file.file_path), 
                       title=file.title,
                       strategy=self.extraction_strategy))
        
        path = Path(file.file_path)

        # Validate file exists
        if not path.exists():
            error_msg = f"File not found: {file.file_path}"
            logger.error(sm(error_msg))
            raise FileNotFoundError(error_msg)

        # Check if format is supported
        if not self.is_supported(file):
            error_msg = f"Unsupported file format: {path.suffix}"
            logger.error(sm(error_msg, extension=path.suffix))
            raise ValueError(error_msg)
            
        MB_50 = 50 * 1024 * 1024  # 50MB in bytes (52,428,800 bytes)
        if file.file_size > MB_50:
            error_msg = "File size too large (over 50MB)"
            logger.error(sm(error_msg, size=file.file_size, max=MB_50))
            raise ValueError(error_msg)


        try:
            # Detect and set MIME type
            file.content_type = self._detect_mime_type(path)

            # Phase 2: Hybrid extraction strategy
            extracted_content = None
            extraction_method = None
            
            # Strategy 1: Try Spotlight extraction first (if enabled and strategy allows)
            if (self.spotlight_enabled and 
                self.extraction_strategy in ["spotlight_first"]):
                
                extracted_content = self._try_spotlight_extraction(path)
                if extracted_content:
                    extraction_method = "spotlight"
                    self.extraction_stats["spotlight_success"] += 1
                    logger.info(sm("EXTRACTION: Spotlight successful", 
                                   filename=path.name,
                                   method="spotlight", 
                                   content_length=len(extracted_content)))
                else:
                    self.extraction_stats["spotlight_failed"] += 1
                    logger.info(sm("EXTRACTION: Spotlight failed, falling back",
                                   filename=path.name,
                                   method="spotlight_failed"))

            # Strategy 2: Try media processing for audio/video/image files
            if not extracted_content:
                is_media, media_type = is_supported_media_file(path)
                if is_media:
                    extracted_content = self._try_media_extraction(path, media_type)
                    if extracted_content:
                        extraction_method = f"media_{media_type}"
                        self.extraction_stats["media_success"] += 1
                        logger.info(sm("EXTRACTION: Media successful",
                                       filename=path.name,
                                       media_type=media_type,
                                       method=f"media_{media_type}", 
                                       content_length=len(extracted_content)))
                    else:
                        self.extraction_stats["media_failed"] += 1
                        logger.info(sm("EXTRACTION: Media failed",
                                       filename=path.name,
                                       media_type=media_type,
                                       method=f"media_{media_type}_failed"))

            # Strategy 3: Fallback to MarkItDown
            if not extracted_content and self.extraction_strategy != "spotlight_only":
                extracted_content = self._try_markitdown_extraction(path)
                if extracted_content:
                    extraction_method = "markitdown"
                    self.extraction_stats["markitdown_success"] += 1
                    logger.info(sm("EXTRACTION: MarkItDown successful",
                                   filename=path.name,
                                   method="markitdown", 
                                   content_length=len(extracted_content)))
                else:
                    self.extraction_stats["markitdown_failed"] += 1
                    logger.info(sm("EXTRACTION: MarkItDown failed",
                                   filename=path.name,
                                   method="markitdown_failed"))

            # Process results
            if extracted_content and len(extracted_content.strip()) > 10:
                file.content = extracted_content
                
                # Calculate content hash for deduplication
                content_hash = self._calculate_content_hash(extracted_content)
                file.content_hash = content_hash

                # Mark as successfully processed
                file.status = ProcessingStatus.PARSED

                logger.info(sm("File parsed successfully",
                               filename=file.filename,
                               content_length=len(extracted_content),
                               extraction_method=extraction_method,
                               content_hash=content_hash[:12]))

                return file
            else:
                error_msg = f"All extraction methods failed or returned empty content"
                logger.warning(sm(error_msg, file_path=str(path)))
                file.status = ProcessingStatus.FAILED
                file.processing_error = error_msg
                raise ValueError(error_msg)

        except Exception as e:
            error_msg = f"Failed to parse file: {e!s}"
            logger.exception(sm(error_msg, file_path=str(path), error=str(e)))

            raise e

    def _try_spotlight_extraction(self, path: Path) -> str | None:
        """
        Try extracting text using macOS Spotlight.
        
        Args:
            path: Path to file
            
        Returns:
            Extracted text or None if failed
        """
        try:
            content = spotlight_to_text(path)
            if content and len(content.strip()) > 50:  # Minimum content threshold
                logger.debug(sm("Spotlight extraction successful", 
                                path=str(path), 
                                length=len(content)))
                return content
            else:
                logger.debug(sm("Spotlight extraction returned insufficient content", 
                                path=str(path)))
                return None
        except Exception as e:
            logger.debug(sm("Spotlight extraction failed", 
                            path=str(path), 
                            error=str(e)))
            return None

    def _try_media_extraction(self, path: Path, media_type: str) -> str | None:
        """
        Try extracting content from media files.
        
        Args:
            path: Path to media file
            media_type: Type of media ('audio', 'video', 'image')
            
        Returns:
            Extracted content or None if failed
        """
        try:
            if media_type == "audio":
                content = extract_audio_transcript(path)
            elif media_type == "video":
                content = extract_video_transcript(path)
            elif media_type == "image":
                # For images, we'll let the summarization process handle vision analysis
                # Just extract basic image info here
                image_info = extract_image_info(path)
                if image_info:
                    dimensions_str = ""
                    if image_info.get("dimensions"):
                        dims = image_info["dimensions"]
                        dimensions_str = f" ({dims['width']}x{dims['height']}, {dims.get('mode', 'unknown')} mode)"
                    
                    content = f"Image file: {path.name}{dimensions_str}, Size: {image_info['file_size']} bytes, Type: {image_info['file_type']}"
                else:
                    content = f"Image file: {path.name}"
            else:
                logger.warning(sm("Unknown media type", media_type=media_type))
                return None
                
            if content and len(content.strip()) > 10:
                logger.debug(sm("Media extraction successful", 
                                path=str(path), 
                                media_type=media_type,
                                length=len(content)))
                return content
            else:
                logger.debug(sm("Media extraction returned insufficient content", 
                                path=str(path), 
                                media_type=media_type))
                return None
                
        except Exception as e:
            logger.debug(sm("Media extraction failed", 
                            path=str(path), 
                            media_type=media_type,
                            error=str(e)))
            return None

    def _try_markitdown_extraction(self, path: Path) -> str | None:
        """
        Try extracting content using MarkItDown.
        
        Args:
            path: Path to file
            
        Returns:
            Extracted markdown content or None if failed
        """
        try:
            logger.debug(sm("Trying MarkItDown extraction", path=str(path)))
            
            result = self.markitdown.convert(str(path))
            
            if result and result.text_content:
                content = result.text_content.strip()
                if len(content) > 0:
                    logger.debug(sm("MarkItDown extraction successful", 
                                    path=str(path), 
                                    length=len(content)))
                    return content
                    
            logger.debug(sm("MarkItDown extraction returned empty content", path=str(path)))
            return None
            
        except Exception as e:
            logger.debug(sm("MarkItDown extraction failed", 
                            path=str(path), 
                            error=str(e)))
            return None

    def set_extraction_strategy(self, strategy: str) -> None:
        """
        Set extraction strategy.
        
        Args:
            strategy: Strategy to use ('spotlight_first', 'markitdown_only', 'spotlight_only')
        """
        valid_strategies = ["spotlight_first", "markitdown_only", "spotlight_only"]
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy. Must be one of: {valid_strategies}")
            
        self.extraction_strategy = strategy
        logger.info(sm("Extraction strategy updated", strategy=strategy))

    def get_extraction_stats(self) -> dict[str, int]:
        """
        Get statistics on which extraction method was used.
        
        Returns:
            Dictionary with extraction statistics
        """
        return self.extraction_stats.copy()

    def reset_extraction_stats(self) -> None:
        """Reset extraction statistics."""
        for key in self.extraction_stats:
            self.extraction_stats[key] = 0
        logger.info(sm("Extraction statistics reset"))


def get_supported_extensions() -> set[str]:
    """Get the set of supported file extensions."""
    parser = FileParser()
    return parser.get_supported_extensions()


def is_supported_file(file: File) -> bool:
    """Check if a file format is supported."""
    parser = FileParser()
    return parser.is_supported(file)
