#!/usr/bin/env python
"""
@File    :   spotlight.py
@Time    :   2025/08/04
@Author  :   Phase 2 Implementation  
@Version :   2.0
@Desc    :   macOS Spotlight text extraction with CLI and PyObjC support
"""

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from ..logging import sm

# Configure structured logger
logger = logging.getLogger(__name__)


def spotlight_to_text(path: Path) -> str | None:
    """
    Extract text content from a file using macOS Spotlight indexing.
    
    Args:
        path: Path to the file to extract text from
        
    Returns:
        Extracted text content or None if unavailable
        
    Raises:
        OSError: If not running on macOS
    """
    if sys.platform != "darwin":
        logger.warning(sm("Spotlight extraction only available on macOS", platform=sys.platform))
        return None
        
    if not path.exists():
        logger.warning(sm("File not found for Spotlight extraction", path=str(path)))
        return None
        
    # Get timeout from environment
    timeout = int(os.getenv("SPOTLIGHT_TIMEOUT_SECONDS", "5"))
    
    try:
        logger.debug(sm("Extracting text via Spotlight", path=str(path)))
        
        # Use mdls to get kMDItemTextContent
        result = subprocess.run(
            ["mdls", "-name", "kMDItemTextContent", str(path)],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode != 0:
            logger.debug(sm("mdls command failed", 
                        path=str(path), 
                        returncode=result.returncode,
                        stderr=result.stderr.strip()))
            return None
            
        output = result.stdout.strip()
        
        # Parse mdls output format: kMDItemTextContent = "content here"
        if "= " not in output:
            logger.debug(sm("No text content found in Spotlight", path=str(path)))
            return None
            
        # Extract content after "= "
        _, content_part = output.split("= ", 1)
        
        # Handle null value
        if content_part.strip() == "(null)":
            logger.debug(sm("Spotlight returned null content", path=str(path)))
            return None
            
        # Remove surrounding quotes and clean up
        content = content_part.strip()
        if content.startswith('"') and content.endswith('"'):
            content = content[1:-1]
            
        # Unescape common escape sequences
        content = content.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"')
        
        if len(content.strip()) < 10:  # Too short to be useful
            logger.debug(sm("Spotlight content too short", path=str(path), length=len(content)))
            return None
            
        logger.info(sm("Successfully extracted text via Spotlight", 
                   path=str(path), 
                   content_length=len(content)))
        return content
        
    except subprocess.TimeoutExpired:
        logger.warning(sm("Spotlight extraction timed out", path=str(path), timeout=timeout))
        return None
    except subprocess.CalledProcessError as e:
        logger.warning(sm("Spotlight extraction process error", 
                      path=str(path), 
                      error=str(e)))
        return None
    except Exception as e:
        logger.exception(sm("Unexpected error in Spotlight extraction", 
                        path=str(path), 
                        error=str(e)))
        return None


def spotlight_metadata(path: Path) -> dict[str, Any]:
    """
    Get comprehensive metadata for a file using Spotlight.
    
    Args:
        path: Path to the file
        
    Returns:
        Dictionary of metadata attributes
    """
    if sys.platform != "darwin":
        logger.warning(sm("Spotlight metadata only available on macOS"))
        return {}
        
    if not path.exists():
        logger.warning(sm("File not found for metadata extraction", path=str(path)))
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
        
        for attr in attributes:
            try:
                result = subprocess.run(
                    ["mdls", "-name", attr, str(path)],
                    capture_output=True,
                    text=True,
                    timeout=3
                )
                
                if result.returncode == 0:
                    output = result.stdout.strip()
                    if "= " in output:
                        _, value = output.split("= ", 1)
                        if value.strip() != "(null)":
                            # Clean up the value
                            value = value.strip()
                            if value.startswith('"') and value.endswith('"'):
                                value = value[1:-1]
                            metadata[attr] = value
                            
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                continue  # Skip attributes that fail
                
        logger.debug(sm("Retrieved Spotlight metadata", 
                    path=str(path), 
                    attributes_found=len(metadata)))
        return metadata
        
    except Exception as e:
        logger.exception(sm("Error retrieving Spotlight metadata", 
                        path=str(path), 
                        error=str(e)))
        return {}


def is_spotlight_indexed(path: Path) -> bool:
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
        result = subprocess.run(
            ["mdls", "-name", "kMDItemDisplayName", str(path)],
            capture_output=True,
            text=True,
            timeout=2
        )
        
        if result.returncode != 0:
            return False
            
        output = result.stdout.strip()
        # If we get actual metadata (not null), file is indexed
        return "= " in output and "(null)" not in output
        
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, Exception):
        return False


def validate_spotlight_availability() -> bool:
    """
    Validate that Spotlight/mdls is available on the system.
    
    Returns:
        True if Spotlight tools are available
    """
    if sys.platform != "darwin":
        logger.info("Spotlight not available - not running on macOS")
        return False
        
    try:
        result = subprocess.run(
            ["which", "mdls"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        available = result.returncode == 0
        
        if available:
            logger.info(sm("Spotlight/mdls tools are available"))
        else:
            logger.warning(sm("Spotlight/mdls tools not found in PATH"))
            
        return available
        
    except Exception as e:
        logger.exception(sm("Error checking Spotlight availability", error=str(e)))
        return False


# Convenience function for backward compatibility
def extract_text_with_spotlight(file_path: str) -> str | None:
    """
    Convenience function to extract text using Spotlight.
    
    Args:
        file_path: String path to the file
        
    Returns:
        Extracted text or None
    """
    return spotlight_to_text(Path(file_path))
