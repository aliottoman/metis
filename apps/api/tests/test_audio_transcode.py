"""The dictation container fix.

The bug this pins: Safari and the native app record MP4 (m4a), Chrome records
WebM, and Cohere Transcribe rejects both with an HTTP 400 naming its accepted
list (flac, mp3, mpeg, mpga, ogg, wav). The clip has to be converted on the
host before it is sent. These tests run the REAL converters on real audio —
this repo only runs on the Mac that ships afconvert, so there is nothing to
mock and a mocked pass would prove nothing about the 400.
"""

from __future__ import annotations

import io
import math
import struct
import subprocess
import wave
from pathlib import Path

import pytest

from waqil_api.audio_transcode import (
    COHERE_NATIVE_SUFFIXES,
    TranscodeError,
    needs_transcoding,
    to_wav,
)


def _sine_wav(seconds: float = 0.4, rate: int = 16_000) -> bytes:
    """A real little WAV: a 440 Hz tone, mono, 16-bit."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        for index in range(int(seconds * rate)):
            sample = int(12_000 * math.sin(2 * math.pi * 440 * index / rate))
            out.writeframes(struct.pack("<h", sample))
    return buffer.getvalue()


def _as_m4a(wav: bytes, tmp_path: Path) -> bytes:
    """What Safari actually uploads: the same audio in an MP4/AAC container."""
    source = tmp_path / "in.wav"
    target = tmp_path / "out.m4a"
    source.write_bytes(wav)
    subprocess.run(
        ["/usr/bin/afconvert", "-f", "m4af", "-d", "aac", str(source), str(target)],
        check=True,
        capture_output=True,
    )
    return target.read_bytes()


def test_the_decision_matches_coheres_accepted_list() -> None:
    for suffix in COHERE_NATIVE_SUFFIXES:
        assert not needs_transcoding(f"clip.{suffix}", "audio/whatever")
    assert needs_transcoding("dictation.m4a", "audio/mp4")
    assert needs_transcoding("dictation.webm", "audio/webm")
    # No extension: the media type decides.
    assert needs_transcoding("dictation", "audio/mp4")
    assert not needs_transcoding("dictation", "audio/wav")


@pytest.mark.asyncio
async def test_a_safari_style_m4a_becomes_a_real_wav(tmp_path) -> None:
    m4a = _as_m4a(_sine_wav(), tmp_path)
    converted = await to_wav(m4a, "dictation.m4a", "audio/mp4")
    assert converted[:4] == b"RIFF" and converted[8:12] == b"WAVE"
    with wave.open(io.BytesIO(converted)) as result:
        # The transcriber's native diet: 16 kHz mono 16-bit.
        assert result.getframerate() == 16_000
        assert result.getnchannels() == 1
        assert result.getsampwidth() == 2
        assert result.getnframes() > 0


@pytest.mark.asyncio
async def test_garbage_bytes_fail_with_a_named_reason(tmp_path) -> None:
    with pytest.raises(TranscodeError, match="could not be converted"):
        await to_wav(b"this is not audio at all", "dictation.m4a", "audio/mp4")
