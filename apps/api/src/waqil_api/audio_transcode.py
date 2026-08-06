"""Recordings into a container Cohere Transcribe accepts.

The browser picks the recording container, not us: Safari and the native
Metis.app (WKWebView) produce MP4/AAC, Chrome produces WebM/Opus. Cohere
accepts neither — its 400 lists flac, mp3, mpeg, mpga, ogg, wav — so the clip
is converted here, on the host, before it ever leaves the machine.

Two converters, in order of preference:
  afconvert   ships with macOS, decodes everything CoreAudio does (mp4, m4a,
              aac, caf, aiff) — covers Safari and the native app with zero
              dependencies.
  ffmpeg      the fallback for WebM/Opus, which CoreAudio does not read.
              Optional; without it, Chrome dictation fails with a message
              that says so instead of a mystery 400.

Output is deliberately WAV LEI16 @ 16 kHz mono: the transcription model's
native diet, and small enough that transcoding never bloats a clip past the
upload ceiling except when the recording was genuinely enormous.
"""
from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
from pathlib import Path

# Verbatim from Cohere's own error message. A clip already in one of these
# containers is sent untouched — no decode, no generation loss.
COHERE_NATIVE_SUFFIXES = frozenset({"flac", "mp3", "mpeg", "mpga", "ogg", "wav"})

_AFCONVERT = "/usr/bin/afconvert"


class TranscodeError(RuntimeError):
    """The clip could not be converted; the message is user-facing."""


def needs_transcoding(filename: str, media_type: str) -> bool:
    """Whether this clip must be converted before Cohere will take it."""
    return _suffix_of(filename, media_type) not in COHERE_NATIVE_SUFFIXES


def _suffix_of(filename: str, media_type: str) -> str:
    name = Path(filename).name
    if "." in name:
        return name.rsplit(".", 1)[-1].lower()
    return media_type.rsplit("/", 1)[-1].lower()


async def to_wav(
    audio: bytes, filename: str, media_type: str, *, timeout_seconds: float = 30.0
) -> bytes:
    """One clip to WAV LEI16/16k/mono, via a temp file that never outlives the call."""
    suffix = _suffix_of(filename, media_type) or "bin"
    return await asyncio.to_thread(_to_wav_blocking, audio, suffix, timeout_seconds)


def _to_wav_blocking(audio: bytes, suffix: str, timeout_seconds: float) -> bytes:
    with tempfile.TemporaryDirectory(prefix="metis-dictation-") as scratch:
        source = Path(scratch) / f"clip.{suffix}"
        target = Path(scratch) / "clip.wav"
        source.write_bytes(audio)

        attempts: list[list[str]] = []
        if Path(_AFCONVERT).exists():
            attempts.append(
                [_AFCONVERT, "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                 str(source), str(target)]
            )
        if ffmpeg := shutil.which("ffmpeg"):
            attempts.append(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                 "-i", str(source), "-ar", "16000", "-ac", "1", "-f", "wav",
                 str(target)]
            )
        if not attempts:
            raise TranscodeError(
                "no audio converter is available on this machine"
            )

        failures: list[str] = []
        for command in attempts:
            try:
                completed = subprocess.run(
                    command, capture_output=True, timeout=timeout_seconds
                )
            except subprocess.TimeoutExpired:
                failures.append(f"{Path(command[0]).name}: timed out")
                continue
            if completed.returncode == 0 and target.is_file() and target.stat().st_size:
                return target.read_bytes()
            detail = (completed.stderr or completed.stdout or b"").decode(
                "utf-8", "replace"
            ).strip()
            failures.append(f"{Path(command[0]).name}: {detail[:200] or 'failed'}")

        hint = (
            " WebM needs ffmpeg (brew install ffmpeg), or dictate from Safari "
            "or the Metis app instead."
            if suffix == "webm" and not shutil.which("ffmpeg")
            else ""
        )
        raise TranscodeError(
            f"this {suffix} recording could not be converted ({'; '.join(failures)}).{hint}"
        )
