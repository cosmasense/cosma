#!/usr/bin/env python
"""
@File    :   spotlight.py
@Time    :   2025/08/04
@Author  :   Phase 2 Implementation  
@Version :   2.0
@Desc    :   macOS Spotlight text extraction with CLI and PyObjC support
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

from ..logging import get_logger

if TYPE_CHECKING:
    from cosma_backend.settings import ParserConfig

# Configure structured logger
logger = get_logger(__name__)


async def spotlight_to_text(path: Path, config: ParserConfig | None = None) -> str | None:
    """
    Extract text content from a file using macOS Spotlight.

    Why `mdimport -t -d3` instead of `mdls -name kMDItemTextContent`:
    Apple restricts `kMDItemTextContent` from non-system callers — `mdls`
    returns "(null)" for that attribute on every file even though Spotlight
    DID index the text. The older code took that `(null)` at face value and
    gave up, which is why Spotlight "never worked" in practice. The
    `mdimport -t -d3 <path>` invocation asks the Spotlight importer to
    test-import the file and dump all attributes (including
    kMDItemTextContent) to stdout, which is the documented way to see
    indexed text from userland tools.

    Returns None if Spotlight has no text importer for the file type
    (images, binaries, etc.) — the caller should fall back to MarkItDown.
    """
    if sys.platform != "darwin":
        logger.warning("Spotlight extraction only available on macOS", platform=sys.platform)
        return None

    if not path.exists():
        logger.warning("File not found for Spotlight extraction", path=str(path))
        return None

    from cosma_backend.settings import ParserConfig as _ParserConfig
    cfg = config or _ParserConfig()
    timeout = cfg.spotlight_timeout_seconds

    try:
        logger.debug("Extracting text via Spotlight (mdimport)", path=str(path))

        # -t: test-import (don't mutate the real index)
        # -d3: print summary + ALL imported attributes (vs -d2 which omits
        #      kMDItemTextContent, which is exactly what we need).
        process = await asyncio.create_subprocess_exec(
            "mdimport", "-t", "-d3", str(path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            logger.warning("Spotlight extraction timed out", path=str(path), timeout=timeout)
            return None

        if process.returncode != 0:
            logger.debug("mdimport failed",
                         path=str(path),
                         returncode=process.returncode,
                         stderr=stderr.decode(errors="replace").strip()[:200])
            return None

        output = stdout.decode(errors="replace")

        # mdimport can dump several MB for large docs. Cap early so we don't
        # blow memory on a 200 MB transcript embedded in a weird file.
        MAX_SPOTLIGHT_OUTPUT = 4 * 1024 * 1024  # 4MB — Spotlight's own text cap
        if len(output) > MAX_SPOTLIGHT_OUTPUT:
            output = output[:MAX_SPOTLIGHT_OUTPUT]

        content = _parse_mditem_text_content(output)
        if not content:
            logger.debug("No kMDItemTextContent in mdimport output", path=str(path))
            return None

        # Short content usually means the importer only knows the filename
        # (e.g. binary files with no extractable text). Not worth sending to
        # the summarizer — let the caller fall through to MarkItDown.
        if len(content.strip()) < 10:
            logger.debug("Spotlight content too short", path=str(path), length=len(content))
            return None

        logger.info("Extracted text via Spotlight",
                    path=str(path), content_length=len(content))
        return content

    except FileNotFoundError:
        logger.warning("mdimport not found (non-macOS?)")
        return None
    except Exception as e:
        logger.exception("Unexpected error in Spotlight extraction",
                         path=str(path), error=str(e))
        return None


# Match `    kMDItemTextContent = "...contents..."` where `...contents...`
# may span multiple lines and contain escaped quotes.
# mdimport terminates each attribute line with `;` and uses doubled-quotes
# to escape a literal `"` inside the string — the regex is anchored so we
# don't accidentally match inside some other attribute's value.
import re  # local import — keeps the public API at the top of the file

_TEXT_CONTENT_RE = re.compile(
    r'^\s*kMDItemTextContent\s*=\s*"(?P<body>(?:[^"\\]|\\.|"")*)"\s*;',
    re.MULTILINE | re.DOTALL,
)


def _parse_mditem_text_content(output: str) -> str | None:
    """Pull kMDItemTextContent's value out of `mdimport -t -d3` stdout.

    Split out for testability — the parsing is the fiddly part and doing
    it inline with the subprocess wrangling made spotlight_to_text hard
    to eyeball.
    """
    m = _TEXT_CONTENT_RE.search(output)
    if not m:
        return None
    body = m.group("body")
    # macOS escapes: \n, \t, \", and "" (double-quote pair). Order matters:
    # unescape `""` first so we don't turn a literal escaped quote into the
    # start of a new string.
    body = body.replace('""', '"')
    body = body.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace('\\\\', '\\')
    return body


async def spotlight_metadata(path: Path) -> dict[str, Any]:
    """
    Get comprehensive metadata for a file using Spotlight.
    
    Args:
        path: Path to the file
        
    Returns:
        Dictionary of metadata attributes
    """
    if sys.platform != "darwin":
        logger.warning("Spotlight metadata only available on macOS")
        return {}
        
    if not path.exists():
        logger.warning("File not found for metadata extraction", path=str(path))
        return {}
        
    try:
        # Get commonly useful attributes
        attributes = [
            "kMDItemDisplayName",
            "kMDItemContentType", 
            "kMDItemKind",
            "kMDItemContentCreationDate",
            "kMDItemContentModificationDate",
            "kMDItemFSSize",
            "kMDItemTextContent",  # Full text if available
            "kMDItemTitle",
            "kMDItemAuthors",
            "kMDItemKeywords",
            "kMDItemSubject",
            "kMDItemDescription"
        ]
        
        metadata = {}
        
        # Create tasks for all attributes to run concurrently
        tasks = []
        for attr in attributes:
            task = _get_metadata_attribute(path, attr)
            tasks.append(task)
        
        # Wait for all tasks with individual timeout handling
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                continue  # Skip attributes that failed
            if result:  # Non-None result means we got metadata
                metadata[attributes[i]] = result
                
        logger.debug("Retrieved Spotlight metadata", 
                    path=str(path), 
                    attributes_found=len(metadata))
        return metadata
        
    except Exception as e:
        logger.exception("Error retrieving Spotlight metadata", 
                        path=str(path), 
                        error=str(e))
        return {}


async def _get_metadata_attribute(path: Path, attr: str) -> Optional[str]:
    """
    Helper function to get a single metadata attribute from Spotlight.
    
    Args:
        path: Path to the file
        attr: Spotlight attribute name
        
    Returns:
        Clean attribute value or None
    """
    try:
        process = await asyncio.create_subprocess_exec(
            "mdls", "-name", attr, str(path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=3)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return None
        
        if process.returncode == 0:
            output = stdout.decode().strip()
            if "= " in output:
                _, value = output.split("= ", 1)
                if value.strip() != "(null)":
                    # Clean up the value
                    value = value.strip()
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    return value
                    
        return None
            
    except Exception:
        return None  # Skip attributes that fail


async def is_spotlight_indexed(path: Path) -> bool:
    """
    Check if a file has been indexed by Spotlight.
    
    Args:
        path: Path to the file to check
        
    Returns:
        True if the file appears to be indexed by Spotlight
    """
    if sys.platform != "darwin":
        return False
        
    if not path.exists():
        return False
        
    try:
        # Try to get basic metadata - if this works, file is indexed
        process = await asyncio.create_subprocess_exec(
            "mdls", "-name", "kMDItemDisplayName", str(path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=2)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return False
        
        if process.returncode != 0:
            return False
            
        output = stdout.decode().strip()
        # If we get actual metadata (not null), file is indexed
        return "= " in output and "(null)" not in output
        
    except Exception:
        return False


async def validate_spotlight_availability() -> bool:
    """
    Validate that Spotlight/mdls is available on the system.
    
    Returns:
        True if Spotlight tools are available
    """
    if sys.platform != "darwin":
        logger.info("Spotlight not available - not running on macOS")
        return False
        
    try:
        process = await asyncio.create_subprocess_exec(
            "which", "mdls",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=5)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            available = False
        else:
            available = process.returncode == 0
        
        if available:
            logger.info("Spotlight/mdls tools are available")
        else:
            logger.warning("Spotlight/mdls tools not found in PATH")
            
        return available
        
    except Exception as e:
        logger.exception("Error checking Spotlight availability", error=str(e))
        return False


# Convenience function for backward compatibility
async def extract_text_with_spotlight(file_path: str) -> str | None:
    """
    Convenience function to extract text using Spotlight.
    
    Args:
        file_path: String path to the file
        
    Returns:
        Extracted text or None
    """
    return await spotlight_to_text(Path(file_path))
