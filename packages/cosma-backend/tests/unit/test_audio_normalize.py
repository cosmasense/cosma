"""Unit tests for audio-format support and the whisper PCM normalizer.

The transcription pipeline now decodes any audio container/codec to a
16 kHz mono 16-bit PCM WAV before handing it to whisper.cpp (which only
ingests PCM). These tests cover the cheap, ffmpeg-free pieces of that
path: the canonical-WAV header sniff and the supported-extension lists.
The ffmpeg transcode itself is exercised by the integration suite.
"""

import struct

import pytest

from cosma_backend.parser.media import _is_canonical_whisper_wav, get_supported_media_extensions
from cosma_backend.parser.parser import FileParser


def _wav_header(audio_format: int, channels: int, sample_rate: int, bits: int) -> bytes:
    """Build a minimal 44-byte RIFF/WAVE/fmt header for the given params."""
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    return (
        b"RIFF" + struct.pack("<I", 36) + b"WAVE"
        + b"fmt " + struct.pack("<I", 16)
        + struct.pack("<HHIIHH", audio_format, channels, sample_rate, byte_rate, block_align, bits)
        + b"data" + struct.pack("<I", 0)
    )


@pytest.mark.unit
class TestCanonicalWhisperWav:
    def test_canonical_16k_mono_pcm_is_recognized(self, tmp_path):
        p = tmp_path / "ok.wav"
        p.write_bytes(_wav_header(audio_format=1, channels=1, sample_rate=16000, bits=16))
        assert _is_canonical_whisper_wav(p) is True

    @pytest.mark.parametrize("af,ch,sr,bits", [
        (1, 2, 16000, 16),   # stereo
        (1, 1, 44100, 16),   # wrong sample rate
        (1, 1, 16000, 24),   # 24-bit
        (3, 1, 16000, 32),   # IEEE float
    ])
    def test_non_canonical_wav_is_rejected(self, tmp_path, af, ch, sr, bits):
        p = tmp_path / "x.wav"
        p.write_bytes(_wav_header(audio_format=af, channels=ch, sample_rate=sr, bits=bits))
        assert _is_canonical_whisper_wav(p) is False

    def test_non_wav_extension_is_rejected(self, tmp_path):
        p = tmp_path / "song.mp3"
        p.write_bytes(b"\xff\xfb\x90\x00" + b"\x00" * 100)
        assert _is_canonical_whisper_wav(p) is False

    def test_truncated_or_garbage_file_is_rejected(self, tmp_path):
        p = tmp_path / "tiny.wav"
        p.write_bytes(b"RIFF")  # nowhere near a full header
        assert _is_canonical_whisper_wav(p) is False


@pytest.mark.unit
class TestAudioExtensionSupport:
    @pytest.mark.parametrize("ext", [".mp3", ".wav", ".aac", ".m4a", ".flac",
                                      ".ogg", ".opus", ".wma", ".aiff", ".aif"])
    def test_common_audio_formats_are_supported_for_parsing(self, ext):
        parser = FileParser()
        assert ext in parser.get_supported_extensions(), f"{ext} should be parseable"

    @pytest.mark.asyncio
    async def test_media_and_parser_audio_lists_agree(self):
        """Every audio ext the media router recognizes must also pass the
        parser's is_supported gate — otherwise the file is rejected before
        it ever reaches the audio path."""
        parser_exts = FileParser().get_supported_extensions()
        media_audio = set((await get_supported_media_extensions())["audio"])
        missing = media_audio - parser_exts
        assert not missing, f"audio exts known to media but rejected by parser: {sorted(missing)}"
