"""
Media Processing Module

Handles audio transcription, video frame extraction, and image metadata.

Audio Transcription:
- OpenAI Whisper API (online, fast, requires API key)
- Local Whisper Python model (offline, uses GPU/CPU)
- Supports partial transcript recovery on timeout

Video Processing:
- Extracts audio track via FFmpeg for transcription
- Extracts evenly-spaced frames for vision analysis
- Parallel transcript + frame extraction

Image Processing:
- Basic metadata extraction (dimensions, format)
- Images are passed to summarizer for vision analysis

Model Lifecycle:
- Whisper model is lazy-loaded on first use
- Tracks last_used_at for automatic unloading
- Use unload_whisper_model() to free memory
"""

from __future__ import annotations

import asyncio
import gc
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Optional, Dict, TYPE_CHECKING

from ..logging import get_logger

if TYPE_CHECKING:
    from cosma_backend.settings import ParserConfig

# Configure structured logger
logger = get_logger(__name__)

# Timeout settings (seconds)
WHISPER_TIMEOUT_SECONDS = 30  # Max time for transcription
FFMPEG_EXTRACT_TIMEOUT_SECONDS = 60  # Max time for audio extraction from video
FRAME_EXTRACT_TIMEOUT_SECONDS = 30  # Max time for frame extraction

# Video frame extraction settings
DEFAULT_NUM_FRAMES = 4  # Number of frames to extract from videos
MAX_FRAME_DIMENSION = 512  # Max width/height for extracted frames (saves memory)

# Global whisper model cache with lifecycle tracking
_whisper_model: Any = None
_whisper_model_name: str | None = None
_whisper_last_used_at: float | None = None


async def extract_audio_transcript(path: Path, config: ParserConfig | None = None, backend: str | None = None) -> str | None:
    """
    Extract transcript from audio file using configurable backends.

    Includes automatic fallback: if the preferred backend is unavailable
    (e.g., local openai-whisper not installed), tries the other available
    backend instead of silently failing.

    Args:
        path: Path to audio file
        config: ParserConfig instance
        backend: Backend to use ('openai' | 'local' | None for auto)

    Returns:
        Transcript text or None if extraction failed
    """
    from cosma_backend.settings import ParserConfig as _ParserConfig
    cfg = config or _ParserConfig()
    provider = backend or cfg.whisper.provider

    # Map provider names to backend names
    backend_map = {"online": "openai", "local": "local", "openai": "openai"}
    preferred = backend_map.get(provider, "openai")

    if not path.exists():
        logger.warning("Audio file not found", path=str(path))
        return None

    # Build fallback order: preferred first, then the other backend as fallback
    backends_to_try: list[str] = [preferred]
    if preferred == "local":
        backends_to_try.append("openai")
    elif preferred == "openai":
        backends_to_try.append("local")

    logger.info("Extracting audio transcript",
               path=str(path),
               preferred=preferred,
               size=path.stat().st_size)

    last_error: str | None = None
    for attempt_backend in backends_to_try:
        try:
            if attempt_backend == "openai":
                # Skip openai if no API key configured — save time and give a
                # clearer error path.
                if not os.getenv("OPENAI_API_KEY"):
                    last_error = "OpenAI API key not set (OPENAI_API_KEY)"
                    logger.info("Skipping OpenAI transcription",
                               path=str(path), reason=last_error)
                    continue
                result = await _transcribe_with_openai(path)
            elif attempt_backend == "local":
                # Quick pre-check: is the local whisper library importable?
                import importlib.util
                if importlib.util.find_spec("whisper") is None and not await _check_whisper_cpp_available():
                    last_error = "Neither openai-whisper nor whisper.cpp is installed"
                    logger.warning("Local whisper unavailable, will try fallback",
                                   path=str(path), reason=last_error)
                    continue
                result = await _transcribe_with_local_whisper(path)
            else:
                logger.error("Unknown Whisper backend", backend=attempt_backend)
                continue

            if result:
                if attempt_backend != preferred:
                    logger.info("Fell back to alternate backend successfully",
                               path=str(path),
                               preferred=preferred,
                               used=attempt_backend)
                return result

            # None returned but no exception — treat as soft failure, try next
            last_error = f"{attempt_backend} backend returned no transcript"

        except Exception as e:
            last_error = f"{attempt_backend} raised: {e}"
            logger.exception("Audio transcription backend failed, trying fallback",
                            path=str(path),
                            backend=attempt_backend,
                            error=str(e))

    logger.warning("All audio transcription backends failed",
                   path=str(path),
                   last_error=last_error)
    return None


from dataclasses import dataclass, field


@dataclass
class VideoContent:
    """Content extracted from a video file."""
    transcript: str | None = None
    frames: list[bytes] = field(default_factory=list)  # JPEG-encoded frames


async def extract_video_content(
    path: Path,
    backend: str | None = None,
    num_frames: int = DEFAULT_NUM_FRAMES,
    extract_frames: bool = True,
) -> VideoContent:
    """
    Extract transcript and key frames from video file.

    Args:
        path: Path to video file
        backend: Backend to use ('openai' | 'local' | None for auto)
        num_frames: Number of frames to extract (evenly spaced)
        extract_frames: Whether to extract frames for vision analysis

    Returns:
        VideoContent with transcript and frames
    """
    result = VideoContent()

    if not path.exists():
        logger.warning("Video file not found", path=str(path))
        return result

    logger.info("Extracting video content",
               path=str(path),
               backend=backend,
               num_frames=num_frames if extract_frames else 0)

    # Extract frames and audio in parallel
    tasks = []

    if extract_frames:
        tasks.append(asyncio.create_task(_extract_video_frames(path, num_frames)))

    tasks.append(asyncio.create_task(_extract_and_transcribe_audio(path, backend)))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    if extract_frames:
        frames_result = results[0]
        transcript_result = results[1]
        if isinstance(frames_result, list):
            result.frames = frames_result
        elif isinstance(frames_result, Exception):
            logger.warning("Frame extraction failed", error=str(frames_result))
    else:
        transcript_result = results[0]

    if isinstance(transcript_result, str):
        result.transcript = transcript_result
    elif isinstance(transcript_result, Exception):
        logger.warning("Transcript extraction failed", error=str(transcript_result))

    logger.info("Video content extracted",
               path=str(path),
               has_transcript=result.transcript is not None,
               num_frames=len(result.frames))

    return result


async def _extract_and_transcribe_audio(path: Path, backend: str | None = None) -> str | None:
    """Extract audio and transcribe it."""
    audio_path = await _extract_audio_from_video(path)
    if not audio_path:
        return None

    try:
        transcript = await extract_audio_transcript(audio_path, backend=backend)
        return transcript
    finally:
        # Clean up temporary audio file
        audio_path.unlink(missing_ok=True)


async def _extract_video_frames(path: Path, num_frames: int = DEFAULT_NUM_FRAMES) -> list[bytes]:
    """
    Extract evenly-spaced frames from a video using ffmpeg.

    Args:
        path: Path to video file
        num_frames: Number of frames to extract

    Returns:
        List of JPEG-encoded frame bytes
    """
    frames: list[bytes] = []
    MAX_TOTAL_FRAME_SIZE = 50 * 1024 * 1024  # 50MB total limit for all frames

    try:
        # First, get video duration
        duration = await _get_video_duration(path)
        if duration is None or duration <= 0:
            logger.warning("Could not determine video duration", path=str(path))
            return frames

        # Calculate timestamps for evenly-spaced frames
        # Avoid very start and end (often black frames)
        start_offset = min(0.5, duration * 0.05)
        end_offset = min(0.5, duration * 0.05)
        usable_duration = duration - start_offset - end_offset

        if usable_duration <= 0:
            timestamps = [duration / 2]  # Just get middle frame
        else:
            timestamps = [
                start_offset + (usable_duration * i / (num_frames - 1)) if num_frames > 1
                else start_offset + usable_duration / 2
                for i in range(num_frames)
            ]

        # Extract each frame with total size tracking
        total_frame_size = 0
        for i, ts in enumerate(timestamps):
            frame_data = await _extract_single_frame(path, ts)
            if frame_data:
                if total_frame_size + len(frame_data) > MAX_TOTAL_FRAME_SIZE:
                    logger.warning("Frame extraction size limit reached",
                                  frames_extracted=len(frames),
                                  total_size=total_frame_size)
                    break
                frames.append(frame_data)
                total_frame_size += len(frame_data)
                logger.debug("Extracted frame", index=i, timestamp=ts)

        logger.info("Frames extracted from video",
                   path=str(path),
                   requested=num_frames,
                   extracted=len(frames))

    except Exception as e:
        logger.exception("Frame extraction failed", path=str(path), error=str(e))

    return frames


async def _get_video_duration(path: Path) -> float | None:
    """Get video duration in seconds using ffprobe."""
    try:
        process = await asyncio.create_subprocess_exec(
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=10)

        if process.returncode == 0:
            duration_str = stdout.decode().strip()
            return float(duration_str)

    except (asyncio.TimeoutError, ValueError, FileNotFoundError) as e:
        logger.warning("Failed to get video duration", path=str(path), error=str(e))

    return None


async def _extract_single_frame(path: Path, timestamp: float) -> bytes | None:
    """Extract a single frame at the given timestamp."""
    try:
        # Use ffmpeg to extract frame as JPEG
        process = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-ss", str(timestamp),
            "-i", str(path),
            "-vframes", "1",
            "-vf", f"scale='min({MAX_FRAME_DIMENSION},iw)':'min({MAX_FRAME_DIMENSION},ih)':force_original_aspect_ratio=decrease",
            "-f", "image2",
            "-c:v", "mjpeg",
            "-q:v", "3",  # Quality (2-31, lower is better)
            "pipe:1",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=FRAME_EXTRACT_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            logger.warning("Frame extraction timed out", timestamp=timestamp)
            return None

        if process.returncode == 0 and stdout:
            return stdout

        logger.debug("Failed to extract frame", timestamp=timestamp, stderr=stderr.decode()[:200])
        return None

    except FileNotFoundError:
        logger.warning("ffmpeg not available for frame extraction")
        return None
    except Exception as e:
        logger.warning("Frame extraction error", timestamp=timestamp, error=str(e))
        return None


async def extract_video_transcript(path: Path, backend: str | None = None) -> str | None:
    """
    Extract transcript from video file by extracting audio first.

    Args:
        path: Path to video file
        backend: Backend to use ('openai' | 'local' | None for auto)

    Returns:
        Transcript text or None if extraction failed

    Note: Use extract_video_content() to also get frames for vision analysis.
    """
    if not path.exists():
        logger.warning("Video file not found", path=str(path))
        return None

    logger.info("Extracting video transcript",
               path=str(path),
               backend=backend)

    try:
        # Extract audio from video using ffmpeg
        audio_path = await _extract_audio_from_video(path)
        if not audio_path:
            return None

        # Transcribe the extracted audio
        transcript = await extract_audio_transcript(audio_path, backend=backend)

        # Clean up temporary audio file
        audio_path.unlink(missing_ok=True)

        return transcript

    except Exception as e:
        logger.exception("Video transcription failed",
                        path=str(path),
                        error=str(e))
        return None


async def extract_image_info(path: Path) -> dict[str, Any] | None:
    """
    Extract basic information about an image file.
    Note: Image analysis is now integrated into the summarization process.
    
    Args:
        path: Path to image file
        
    Returns:
        Dictionary with basic image info or None if failed
    """
    if not path.exists():
        logger.warning("Image file not found", path=str(path))
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
            logger.debug("PIL not available, skipping image dimension extraction")
        except Exception as e:
            logger.debug("Failed to extract image dimensions", error=str(e))
        
        info = {
            "file_size": file_stats.st_size,
            "file_type": path.suffix.lower(),
            "dimensions": dimensions
        }
        
        logger.info("Image info extracted", 
                   path=str(path),
                   **info)
        
        return info
            
    except Exception as e:
        logger.exception("Image info extraction failed", 
                        path=str(path),
                        error=str(e))
        return None


# =============================================================================
# OpenAI API Implementations
# =============================================================================

async def _transcribe_with_openai(audio_path: Path, config: Optional[Dict[str, Any]] = None) -> str | None:
    """Transcribe audio using OpenAI Whisper API."""
    try:
        import openai
        
        # Get API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OpenAI API key not configured")
            return None
            
        client = openai.OpenAI(api_key=api_key)
        
        # Check file size (OpenAI has 25MB limit)
        file_size = audio_path.stat().st_size
        max_size = 25 * 1024 * 1024  # 25MB
        
        if file_size > max_size:
            logger.warning("Audio file too large for OpenAI", 
                          size=file_size, 
                          max_size=max_size)
            # TODO: Could split large files into chunks
            return None
        
        # Get model from env
        model = os.getenv("ONLINE_WHISPER_MODEL", "whisper-1")
        
        # Open file and transcribe (OpenAI client is synchronous but the overall function is async)
        with open(audio_path, 'rb') as audio_file:
            transcript = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                response_format="text"
            )
        
        logger.info("OpenAI transcription completed", 
                   path=str(audio_path),
                   length=len(transcript))
        
        return transcript.strip() if transcript else None
        
    except ImportError:
        logger.error("OpenAI library not installed - install with: pip install openai")
        return None
    except Exception as e:
        logger.exception("OpenAI transcription error", error=str(e))
        return None


# Vision processing is now integrated into the summarization system
# See main.summarizer.summarizer.AutoSummarizer.summarize_with_image()


# =============================================================================
# Local Model Implementations
# =============================================================================

async def _transcribe_with_local_whisper(audio_path: Path) -> str | None:
    """Transcribe audio using local Whisper model.

    Preference order:
      1. whisper.cpp via pywhispercpp — fastest, no subprocess, bundled binary.
      2. openai-whisper (PyTorch) — slower but supports more model variants.

    The previous implementation shelled out to a `whisper` binary that was
    ambiguous across installs (see whisper_local.py for the rationale).
    """
    try:
        from cosma_backend.parser import whisper_local
        if whisper_local.is_available():
            result = await whisper_local.transcribe(audio_path)
            if result:
                return result
            # Fall through to python whisper on transcription failure.
            logger.info("pywhispercpp returned no text; trying whisper python fallback")
        return await _transcribe_with_whisper_python(audio_path)
    except Exception as e:
        logger.exception("Local Whisper transcription error", error=str(e))
        return None


async def _transcribe_with_whisper_cpp(audio_path: Path) -> str | None:
    """Transcribe using whisper.cpp (faster native implementation).

    If timeout is reached, returns partial transcript collected so far.
    """
    try:
        model_name = os.getenv("LOCAL_WHISPER_MODEL", "turbo")

        # Run whisper.cpp main command
        process = await asyncio.create_subprocess_exec(
            "whisper",
            str(audio_path),
            "-m", f"models/ggml-{model_name}.bin",
            "-t", "4",  # 4 threads
            "-np",      # no print
            "-nt",      # no timestamps
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # Read stdout incrementally to capture partial output on timeout
        partial_output: list[bytes] = []
        start_time = time.time()
        timed_out = False

        try:
            while True:
                # Check timeout
                elapsed = time.time() - start_time
                remaining = WHISPER_TIMEOUT_SECONDS - elapsed
                if remaining <= 0:
                    timed_out = True
                    break

                # Try to read a chunk with remaining timeout
                try:
                    chunk = await asyncio.wait_for(
                        process.stdout.read(4096),
                        timeout=min(remaining, 1.0)  # Check every second
                    )
                    if not chunk:
                        # EOF reached - process finished
                        break
                    partial_output.append(chunk)
                except asyncio.TimeoutError:
                    # Just a read timeout, continue if we have time left
                    continue

        except Exception as e:
            logger.warning("Error reading whisper.cpp output", error=str(e))

        if timed_out:
            process.kill()
            await process.wait()
            # Return partial transcript if we have any
            if partial_output:
                partial_transcript = b"".join(partial_output).decode().strip()
                logger.warning("whisper.cpp timed out, returning partial transcript",
                              timeout=WHISPER_TIMEOUT_SECONDS,
                              path=str(audio_path),
                              partial_length=len(partial_transcript))
                return partial_transcript if partial_transcript else None
            logger.warning("whisper.cpp timed out with no output",
                          timeout=WHISPER_TIMEOUT_SECONDS,
                          path=str(audio_path))
            return None

        # Wait for process to finish
        await process.wait()

        if process.returncode == 0:
            transcript = b"".join(partial_output).decode().strip()
            logger.info("whisper.cpp transcription completed",
                       model=model_name,
                       length=len(transcript))
            return transcript
        else:
            stderr = await process.stderr.read()
            logger.warning("whisper.cpp failed", stderr=stderr.decode())
            return None

    except FileNotFoundError:
        logger.warning("whisper.cpp not available or failed", error="Command not found")
        return None
    except Exception as e:
        logger.warning("whisper.cpp not available or failed", error=str(e))
        return None


async def _transcribe_with_whisper_python(audio_path: Path) -> str | None:
    """Transcribe using OpenAI Whisper Python library with lazy loading.

    If timeout is reached, returns partial transcript collected so far.
    """
    global _whisper_model, _whisper_model_name, _whisper_last_used_at

    try:
        import whisper

        model_name = os.getenv("LOCAL_WHISPER_MODEL", "turbo")

        # Lazy load model (only when needed)
        if _whisper_model is None or _whisper_model_name != model_name:
            logger.info("Loading Whisper model", model=model_name)
            _whisper_model = whisper.load_model(model_name)
            _whisper_model_name = model_name

        # Update last used timestamp
        _whisper_last_used_at = time.time()

        # Use a shared list to collect partial results from the thread
        partial_segments: list[str] = []
        timed_out = False

        def _do_transcribe_with_segments():
            """Transcribe and collect segments incrementally."""
            nonlocal timed_out
            start_time = time.time()

            # Load audio
            audio = whisper.load_audio(str(audio_path))
            audio = whisper.pad_or_trim(audio)

            # Make log-Mel spectrogram
            mel = whisper.log_mel_spectrogram(audio, n_mels=_whisper_model.dims.n_mels).to(_whisper_model.device)

            # Detect language (use first 30 seconds)
            _, probs = _whisper_model.detect_language(mel)
            detected_lang = max(probs, key=probs.get)

            # Decode with segment callback
            options = whisper.DecodingOptions(language=detected_lang)

            # For full transcription with segments, use transcribe which handles chunking
            result = _whisper_model.transcribe(
                str(audio_path),
                language=detected_lang,
                verbose=False,
            )

            # Collect segments, checking timeout periodically
            for segment in result.get("segments", []):
                if time.time() - start_time > WHISPER_TIMEOUT_SECONDS:
                    timed_out = True
                    logger.warning("Whisper timeout reached, keeping partial transcript",
                                  segments_collected=len(partial_segments),
                                  timeout=WHISPER_TIMEOUT_SECONDS)
                    break
                partial_segments.append(segment["text"])

            return result

        try:
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, _do_transcribe_with_segments),
                timeout=WHISPER_TIMEOUT_SECONDS + 5  # Allow a bit more for cleanup
            )

            if not timed_out:
                transcript = result["text"]
                logger.info("Whisper Python transcription completed",
                           model=model_name,
                           length=len(transcript))
                return transcript.strip()

        except asyncio.TimeoutError:
            timed_out = True
            logger.warning("Whisper Python transcription timed out (hard limit)",
                          timeout=WHISPER_TIMEOUT_SECONDS,
                          path=str(audio_path))

        # Return partial transcript if we have any segments
        if partial_segments:
            partial_transcript = " ".join(partial_segments).strip()
            logger.info("Returning partial Whisper transcript",
                       model=model_name,
                       segments=len(partial_segments),
                       length=len(partial_transcript))
            return partial_transcript

        return None

    except ImportError:
        logger.warning(
            "openai-whisper package not installed — "
            "install with `uv pip install openai-whisper` or configure "
            "Whisper provider to 'online' in Settings"
        )
        return None
    except Exception as e:
        logger.exception("Whisper Python error", error=str(e))
        return None


def get_whisper_last_used_at() -> float | None:
    """Get the timestamp when whisper model was last used."""
    return _whisper_last_used_at


def unload_whisper_model() -> None:
    """Unload the whisper model to free memory."""
    global _whisper_model, _whisper_model_name, _whisper_last_used_at

    if _whisper_model is not None:
        logger.info("Unloading Whisper model", model=_whisper_model_name)
        del _whisper_model
        _whisper_model = None
        _whisper_model_name = None
        _whisper_last_used_at = None
        from cosma_backend.utils.memory import release_memory
        release_memory()


# Local vision processing is now integrated into the summarization system
# See main.summarizer.summarizer.AutoSummarizer._summarize_with_vision_local()


# =============================================================================
# Utility Functions
# =============================================================================

async def _extract_audio_from_video(video_path: Path) -> Path | None:
    """Extract audio track from video file using ffmpeg."""
    try:
        # Check if ffmpeg is available
        process = await asyncio.create_subprocess_exec(
            "ffmpeg", "-version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await process.communicate()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, "ffmpeg")
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("ffmpeg not available - required for video processing")
        return None
    
    try:
        # Create temporary audio file
        temp_audio = Path(tempfile.mktemp(suffix='.wav'))
        
        # Extract audio with ffmpeg
        process = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-i", str(video_path),
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # PCM 16-bit
            "-ar", "16000",  # 16kHz sample rate (good for speech)
            "-ac", "1",  # Mono
            "-y",  # Overwrite output
            str(temp_audio),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=FFMPEG_EXTRACT_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            logger.warning("ffmpeg audio extraction timed out",
                          timeout=FFMPEG_EXTRACT_TIMEOUT_SECONDS,
                          path=str(video_path))
            return None
        
        if process.returncode == 0 and temp_audio.exists():
            logger.info("Audio extracted from video", 
                       video=str(video_path),
                       audio=str(temp_audio),
                       size=temp_audio.stat().st_size)
            return temp_audio
        else:
            logger.warning("Audio extraction failed", 
                          video=str(video_path),
                          stderr=stderr.decode())
            return None
            
    except Exception as e:
        logger.exception("ffmpeg audio extraction error", error=str(e))
        return None


async def _check_whisper_cpp_available() -> bool:
    """Availability probe for whisper.cpp.

    Now backed by pywhispercpp (Python binding); the old subprocess probe
    for a bare `whisper` binary was unreliable because that binary name
    collides with openai-whisper's CLI. Kept as a function for source
    compatibility with callers like validate_media_backends().
    """
    from cosma_backend.parser import whisper_local
    return whisper_local.is_available()


async def get_supported_media_extensions() -> dict[str, list[str]]:
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


async def is_supported_media_file(path: Path) -> tuple[bool, str | None]:
    """
    Check if a file is a supported media type.
    
    Args:
        path: Path to file to check
        
    Returns:
        Tuple of (is_supported, media_type)
    """
    extension = path.suffix.lower()
    supported = await get_supported_media_extensions()
    
    for media_type, extensions in supported.items():
        if extension in extensions:
            return True, media_type
    
    return False, None


# Test/validation functions
async def validate_media_backends() -> dict[str, bool]:
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
    results["whisper_cpp"] = await _check_whisper_cpp_available()
    
    # Test ffmpeg
    try:
        process = await asyncio.create_subprocess_exec(
            "ffmpeg", "-version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await process.communicate()
        results["ffmpeg"] = process.returncode == 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        results["ffmpeg"] = False
    
    logger.info("Media backend availability", **results)
    return results
