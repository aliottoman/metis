"""The ingress's own configuration, read straight from the environment.

Deliberately not `waqil_api.config.Settings`. That class is the right one for
the trusted process and the wrong one here: it carries the Cohere, Cline,
ElevenLabs and Notion credentials, and this process sits on the public side of
a tunnel. What it needs is a shared secret to check, an alias to compare, a
loopback address to forward to, and four numbers that bound what one caller
can do. Reading those directly keeps the blast radius of this process to
exactly those values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse


class IngressConfigError(RuntimeError):
    """The ingress cannot start with the configuration it was given."""


def _int(name: str, default: int, *, low: int, high: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise IngressConfigError(f"{name} must be a whole number") from exc
    if not low <= value <= high:
        raise IngressConfigError(f"{name} must be between {low} and {high}")
    return value


@dataclass(frozen=True, slots=True)
class IngressConfig:
    # The bearer ElevenLabs must present. No default: an ingress with a
    # guessable secret is an open door, so an absent value is a refusal to
    # start rather than a weak default nobody notices.
    shared_secret: str
    # The model name ElevenLabs is configured to send. It is a label to check,
    # never a routing instruction — which model reasons is decided on :8000
    # from the owner's stored preference.
    public_model_alias: str
    loopback_url: str
    host: str
    port: int
    max_body_bytes: int
    max_concurrency: int
    request_timeout_seconds: float
    rate_per_minute: int
    # ElevenLabs signs post-call webhooks with this. Absent, the webhook route
    # rejects everything: an unverified payload is not evidence, it is input.
    webhook_secret: str

    @classmethod
    def from_environment(cls) -> IngressConfig:
        secret = (os.environ.get("WAQIL_VOICE_SHARED_SECRET") or "").strip()
        if len(secret) < 32:
            raise IngressConfigError(
                "WAQIL_VOICE_SHARED_SECRET must be set to at least 32 characters"
            )
        loopback = (
            (os.environ.get("WAQIL_VOICE_LOOPBACK_URL") or "http://127.0.0.1:8000")
            .strip()
            .rstrip("/")
        )
        parsed = urlparse(loopback)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
            # The one direction this process may talk in. A configuration
            # mistake here would turn the adapter into a proxy for anything.
            raise IngressConfigError(
                "WAQIL_VOICE_LOOPBACK_URL must be an http loopback address"
            )
        return cls(
            shared_secret=secret,
            public_model_alias=(
                os.environ.get("WAQIL_VOICE_PUBLIC_MODEL_ALIAS") or "metis-voice"
            ).strip(),
            loopback_url=loopback,
            host=(os.environ.get("WAQIL_VOICE_INGRESS_HOST") or "127.0.0.1").strip(),
            port=_int("WAQIL_VOICE_INGRESS_PORT", 8788, low=1, high=65_535),
            # A spoken turn is a sentence and a few bounded prior ones. A body
            # larger than this is not a conversation.
            max_body_bytes=_int(
                "WAQIL_VOICE_MAX_BODY_BYTES",
                128 * 1024,
                low=1_024,
                high=4 * 1024 * 1024,
            ),
            max_concurrency=_int("WAQIL_VOICE_MAX_CONCURRENCY", 4, low=1, high=32),
            request_timeout_seconds=float(
                _int("WAQIL_VOICE_REQUEST_TIMEOUT_SECONDS", 60, low=5, high=300)
            ),
            rate_per_minute=_int("WAQIL_VOICE_RATE_PER_MINUTE", 60, low=1, high=600),
            webhook_secret=(
                os.environ.get("WAQIL_ELEVENLABS_WEBHOOK_SECRET") or ""
            ).strip(),
        )
