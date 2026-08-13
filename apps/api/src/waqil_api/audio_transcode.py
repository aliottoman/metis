"""Recordings into a container the selected transcriber accepts.

The browser picks the recording container, not us: Safari and the native
Metis.app (WKWebView) produce MP4/AAC, Chrome produces WebM/Opus. Cohere
accepts neither — its 400 lists flac, mp3, mpeg, mpga, ogg, wav — so the clip
is converted here, on the host, before it ever leaves the machine.

What counts as acceptable is per provider, and deliberately not a union.
ElevenLabs takes the browser's own containers, so a clip bound for Scribe
skips the decode entirely; Cohere's allowlist stays exactly as narrow as
Cohere's own error message says it is, because the cost of widening it by
accident is a 400 the user reads as dictation being broken.

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

# Scribe reads what the browser records. The two containers that cost Cohere a
# transcode — WebM from Chrome, MP4/M4A from Safari and the native app — are
# the whole reason this list exists.
ELEVENLABS_NATIVE_SUFFIXES = COHERE_NATIVE_SUFFIXES | frozenset(
    {"aac", "m4a", "mp4", "oga", "opus", "webm"}
)

NATIVE_SUFFIXES_BY_PROVIDER = {
    "cohere": COHERE_NATIVE_SUFFIXES,
    "elevenlabs": ELEVENLABS_NATIVE_SUFFIXES,
}

_AFCONVERT = "/usr/bin/afconvert"


class TranscodeError(RuntimeError):
    """The clip could not be converted; the message is user-facing."""


def needs_transcoding(
    filename: str, media_type: str, *, provider: str = "cohere"
) -> bool:
    """Whether this clip must be converted before `provider` will take it.

    Cohere is the default because it was the only transcriber: an unnamed
    provider gets the narrow allowlist, so a caller that has not been taught
    about the choice can never accidentally send a container to a service that
    refuses it.
    """
    accepted = NATIVE_SUFFIXES_BY_PROVIDER.get(provider, COHERE_NATIVE_SUFFIXES)
    return _suffix_of(filename, media_type) not in accepted


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
                [
                    _AFCONVERT,
                    "-f",
                    "WAVE",
                    "-d",
                    "LEI16@16000",
                    "-c",
                    "1",
                    str(source),
                    str(target),
                ]
            )
        if ffmpeg := shutil.which("ffmpeg"):
            attempts.append(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(source),
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-f",
                    "wav",
                    str(target),
                ]
            )
        if not attempts:
            raise TranscodeError("no audio converter is available on this machine")

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
            detail = (
                (completed.stderr or completed.stdout or b"")
                .decode("utf-8", "replace")
                .strip()
            )
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
