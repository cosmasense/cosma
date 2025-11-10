#!/usr/bin/env python
"""
@File    :   media.py
@Time    :   2025/08/04
@Author  :   Phase 2 Implementation  
@Version :   2.0
@Desc    :   Audio/video transcription and image captioning with multi-provider support
"""

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ..logging import sm

# Configure structured logger
logger = logging.getLogger(__name__)


def extract_audio_transcript(path: Path, backend: str | None = None) -> str | None:
    """
    Extract transcript from audio file using configurable backends.
    
    Args:
        path: Path to audio file
        backend: Backend to use ('openai' | 'local' | None for auto)
        
    Returns:
        Transcript text or None if extraction failed
    """
    # Get provider from env (defaults to online)
    provider = backend or os.getenv("WHISPER_PROVIDER", "online")
    
    # Map provider names to backend names
    backend_map = {"online": "openai", "local": "local", "openai": "openai"}
    backend = backend_map.get(provider, "openai")
    
    if not path.exists():
        logger.warning(sm("Audio file not found", path=str(path)))
        return None
        
    logger.info(sm("Extracting audio transcript", 
               path=str(path), 
               backend=backend,
               size=path.stat().st_size))
    
    try:
        if backend == "openai":
            return _transcribe_with_openai(path)
        elif backend == "local":
            return _transcribe_with_local_whisper(path)
        else:
            logger.error(sm("Unknown Whisper backend", backend=backend))
            return None
            
    except Exception as e:
        logger.exception(sm("Audio transcription failed", 
                        path=str(path), 
                        backend=backend,
                        error=str(e)))
        return None


def extract_video_transcript(path: Path, backend: str | None = None) -> str | None:
    """
    Extract transcript from video file by extracting audio first.
    
    Args:
        path: Path to video file
        backend: Backend to use ('openai' | 'local' | None for auto)
        
    Returns:
        Transcript text or None if extraction failed
    """
    if not path.exists():
        logger.warning(sm("Video file not found", path=str(path)))
        return None
        
    logger.info(sm("Extracting video transcript", 
               path=str(path), 
               backend=backend))
    
    try:
        # Extract audio from video using ffmpeg
        audio_path = _extract_audio_from_video(path)
        if not audio_path:
            return None
            
        # Transcribe the extracted audio
        transcript = extract_audio_transcript(audio_path, backend)
        
        # Clean up temporary audio file
        audio_path.unlink(missing_ok=True)
        
        return transcript
        
    except Exception as e:
        logger.exception(sm("Video transcription failed", 
                        path=str(path), 
                        error=str(e)))
        return None


def extract_image_info(path: Path) -> dict[str, Any] | None:
    """
    Extract basic information about an image file.
    Note: Image analysis is now integrated into the summarization process.
    
    Args:
        path: Path to image file
        
    Returns:
        Dictionary with basic image info or None if failed
    """
    if not path.exists():
        logger.warning(sm("Image file not found", path=str(path)))
        return None
    
    try:
        # Get basic file info
        file_stats = path.stat()
        
        # Try to get image dimensions if PIL is available
        dimensions = None
        try:
            from PIL import Image
            with Image.open(path) as img:
                dimensions = {"width": img.width, "height": img.height, "mode": img.mode}
        except ImportError:
            logger.debug(sm("PIL not available, skipping image dimension extraction"))
        except Exception as e:
            logger.debug(sm("Failed to extract image dimensions", error=str(e)))
        
        info = {
            "file_size": file_stats.st_size,
            "file_type": path.suffix.lower(),
            "dimensions": dimensions
        }
        
        logger.info(sm("Image info extracted", 
                   path=str(path),
                   **info))
        
        return info
            
    except Exception as e:
        logger.exception(sm("Image info extraction failed", 
                        path=str(path),
                        error=str(e)))
        return None


# =============================================================================
# OpenAI API Implementations
# =============================================================================

def _transcribe_with_openai(audio_path: Path) -> str | None:
    """Transcribe audio using OpenAI Whisper API."""
    try:
        import openai
        
        # Get API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error(sm("OpenAI API key not configured"))
            return None
            
        client = openai.OpenAI(api_key=api_key)
        
        # Check file size (OpenAI has 25MB limit)
        file_size = audio_path.stat().st_size
        max_size = 25 * 1024 * 1024  # 25MB
        
        if file_size > max_size:
            logger.warning(sm("Audio file too large for OpenAI", 
                          size=file_size, 
                          max_size=max_size))
            # TODO: Could split large files into chunks
            return None
        
        # Get model from env
        model = os.getenv("ONLINE_WHISPER_MODEL", "whisper-1")
        
        # Open file and transcribe
        with open(audio_path, 'rb') as audio_file:
            transcript = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                response_format="text"
            )
        
        logger.info(sm("OpenAI transcription completed", 
                   path=str(audio_path),
                   length=len(transcript)))
        
        return transcript.strip() if transcript else None
        
    except ImportError:
        logger.error(sm("OpenAI library not installed - install with: pip install openai"))
        return None
    except Exception as e:
        logger.exception(sm("OpenAI transcription error", error=str(e)))
        return None


# Vision processing is now integrated into the summarization system
# See main.summarizer.summarizer.AutoSummarizer.summarize_with_image()


# =============================================================================
# Local Model Implementations
# =============================================================================

def _transcribe_with_local_whisper(audio_path: Path) -> str | None:
    """Transcribe audio using local Whisper model."""
    try:
        # Try whisper-cpp first (faster), then whisper
        if _check_whisper_cpp_available():
            return _transcribe_with_whisper_cpp(audio_path)
        else:
            return _transcribe_with_whisper_python(audio_path)
            
    except Exception as e:
        logger.exception(sm("Local Whisper transcription error", error=str(e)))
        return None


def _transcribe_with_whisper_cpp(audio_path: Path) -> str | None:
    """Transcribe using whisper.cpp (faster native implementation)."""
    try:
        model_name = os.getenv("LOCAL_WHISPER_MODEL", "turbo")
        
        # Run whisper.cpp main command
        result = subprocess.run([
            "whisper", 
            str(audio_path),
            "-m", f"models/ggml-{model_name}.bin",
            "-t", "4",  # 4 threads
            "-np",      # no print
            "-nt",      # no timestamps
        ], capture_output=True, text=True, timeout=300)  # 5 min timeout
        
        if result.returncode == 0:
            transcript = result.stdout.strip()
            logger.info(sm("whisper.cpp transcription completed", 
                       model=model_name,
                       length=len(transcript)))
            return transcript
        else:
            logger.warning(sm("whisper.cpp failed", stderr=result.stderr))
            return None
            
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(sm("whisper.cpp not available or failed", error=str(e)))
        return None


def _transcribe_with_whisper_python(audio_path: Path) -> str | None:
    """Transcribe using OpenAI Whisper Python library."""
    try:
        import whisper
        
        model_name = os.getenv("LOCAL_WHISPER_MODEL", "turbo")
        
        # Load model (cached after first use)
        model = whisper.load_model(model_name)
        
        # Transcribe
        result = model.transcribe(str(audio_path))
        transcript = result["text"]
        
        logger.info(sm("Whisper Python transcription completed", 
                   model=model_name,
                   length=len(transcript)))
        
        return transcript.strip()
        
    except ImportError:
        logger.error(sm("Whisper library not installed - install with: pip install openai-whisper"))
        return None
    except Exception as e:
        logger.exception(sm("Whisper Python error", error=str(e)))
        return None


# Local vision processing is now integrated into the summarization system
# See main.summarizer.summarizer.AutoSummarizer._summarize_with_vision_local()


# =============================================================================
# Utility Functions
# =============================================================================

def _extract_audio_from_video(video_path: Path) -> Path | None:
    """Extract audio track from video file using ffmpeg."""
    try:
        # Check if ffmpeg is available
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error(sm("ffmpeg not available - required for video processing"))
        return None
    
    try:
        # Create temporary audio file
        temp_audio = Path(tempfile.mktemp(suffix='.wav'))
        
        # Extract audio with ffmpeg
        result = subprocess.run([
            "ffmpeg",
            "-i", str(video_path),
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # PCM 16-bit
            "-ar", "16000",  # 16kHz sample rate (good for speech)
            "-ac", "1",  # Mono
            "-y",  # Overwrite output
            str(temp_audio)
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0 and temp_audio.exists():
            logger.info(sm("Audio extracted from video", 
                       video=str(video_path),
                       audio=str(temp_audio),
                       size=temp_audio.stat().st_size))
            return temp_audio
        else:
            logger.warning(sm("Audio extraction failed", 
                          video=str(video_path),
                          stderr=result.stderr))
            return None
            
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        logger.exception(sm("ffmpeg audio extraction error", error=str(e)))
        return None


def _check_whisper_cpp_available() -> bool:
    """Check if whisper.cpp is available."""
    try:
        result = subprocess.run(["whisper", "--help"], 
                              capture_output=True, 
                              timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, PermissionError):
        return False


def get_supported_media_extensions() -> dict[str, list[str]]:
    """
    Get supported media file extensions by category.
    
    Returns:
        Dictionary mapping categories to extension lists
    """
    return {
        "audio": [".mp3", ".wav", ".aac", ".m4a", ".flac", ".ogg"],
        "video": [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"],
        "image": [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".heic", ".gif", ".webp"]
    }


def is_supported_media_file(path: Path) -> tuple[bool, str | None]:
    """
    Check if a file is a supported media type.
    
    Args:
        path: Path to file to check
        
    Returns:
        Tuple of (is_supported, media_type)
    """
    extension = path.suffix.lower()
    supported = get_supported_media_extensions()
    
    for media_type, extensions in supported.items():
        if extension in extensions:
            return True, media_type
    
    return False, None


# Test/validation functions
def validate_media_backends() -> dict[str, bool]:
    """
    Validate which media processing backends are available.
    
    Returns:
        Dictionary of backend availability
    """
    results = {}
    
    # Test OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    results["openai"] = bool(api_key)
    
    # Test local Whisper
    try:
        import importlib.util
        whisper_spec = importlib.util.find_spec("whisper")
        results["whisper_python"] = whisper_spec is not None
    except ImportError:
        results["whisper_python"] = False
    
    # Test whisper.cpp
    results["whisper_cpp"] = _check_whisper_cpp_available()
    
    # Test ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        results["ffmpeg"] = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        results["ffmpeg"] = False
    
    logger.info(sm("Media backend availability", **results))
    return results
