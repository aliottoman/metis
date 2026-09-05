"""Three routes, and a hard rule about what happens on each.

The whole security posture of this file is that it decides nothing. It checks
that a caller is who it claims to be, that the request is the shape it must
be, and that it is not arriving too fast or too large — and then it hands the
question to port 8000 and streams back whatever comes out. There is no branch
here that answers a user, resolves an account, retrieves a document, or writes
a record, because there is nothing here to do any of that with.

Two details are load-bearing and easy to lose in a refactor:

* The bearer token is checked **before the body is read**. An unauthenticated
  caller must not be able to make this process allocate memory for a payload.
* Caller-supplied system messages are dropped rather than forwarded. They are
  the obvious injection surface — anyone who can reach this route could
  otherwise write Metis's instructions — and voice's actual system prompt
  lives on the trusted side where nothing external can reach it.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import IngressConfig

# The complete public surface. The isolation test compares the app's registered
# routes against this set, so a fourth route cannot appear without a
# deliberate, reviewed change here.
PUBLIC_ROUTES = frozenset(
    {
        ("GET", "/health"),
        ("POST", "/v1/chat/completions"),
        ("POST", "/v1/elevenlabs/post-call"),
    }
)

# The only tools ElevenLabs may name. Its own system tools are conversational
# controls — the agent ending or transferring a call — and everything else is
# refused rather than forwarded, because a tool this process does not
# recognize is a capability nobody reviewed.
ALLOWED_TOOLS = frozenset({"end_call", "language_detection", "skip_turn"})

# How many prior turns travel with a question, and how much of each.
#
# Sized for what a spoken turn actually is: one person saying one sentence.
# The previous ceilings (12 messages of 4,000 characters) allowed 48,000
# characters of history — roughly 12,000 tokens — on a request whose answer is
# two to four sentences, and every turn paid for it again. Six turns of 300
# characters is a conversation you can still follow back, at a twenty-fifth of
# the cost.
#
# None of it is authority in any case: it is a record of what was said, and
# the trusted side labels it as one.
MAX_HISTORY_MESSAGES = 6
MAX_MESSAGE_CHARS = 300
# What someone just said keeps room to be a long sentence. This is the one
# place in the request that must not be squeezed for tokens.
MAX_UTTERANCE_CHARS = 2_000

# Webhook signatures older than this are refused. It bounds replay to the
# window in which a captured request is still useful to an attacker.
WEBHOOK_TOLERANCE_SECONDS = 300


class RateLimiter:
    """A sliding window per session, so one conversation cannot flood :8000."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._seen: dict[str, deque[float]] = {}

    def allow(self, key: str, *, now: float) -> bool:
        # Retire inactive buckets globally, not only the bucket making this
        # request. Otherwise forged conversation ids accumulate forever.
        for stale_key, stale_window in list(self._seen.items()):
            while stale_window and now - stale_window[0] > 60.0:
                stale_window.popleft()
            if not stale_window:
                self._seen.pop(stale_key, None)
        window = self._seen.setdefault(key, deque())
        while window and now - window[0] > 60.0:
            window.popleft()
        if len(window) >= self.per_minute:
            return False
        window.append(now)
        return True


class Seen:
    """Conversation id + event type, remembered just long enough to dedupe."""

    def __init__(self, limit: int = 512) -> None:
        self._order: deque[str] = deque()
        self._keys: set[str] = set()
        self._limit = limit

    def add(self, key: str) -> bool:
        """True when this is new; False when it has already been handled."""
        if key in self._keys:
            return False
        self._keys.add(key)
        self._order.append(key)
        if len(self._order) > self._limit:
            self._keys.discard(self._order.popleft())
        return True

    def __contains__(self, key: str) -> bool:
        return key in self._keys


def _unauthorized() -> JSONResponse:
    # Deliberately uniform: a caller learns that it failed, never which check.
    return JSONResponse({"error": "unauthorized"}, status_code=401)


def create_app(config: IngressConfig) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.client = httpx.AsyncClient(
            base_url=config.loopback_url,
            timeout=config.request_timeout_seconds,
        )
        try:
            yield
        finally:
            if application.state.client is not None:
                await application.state.client.aclose()
                application.state.client = None

    app = FastAPI(
        title="Metis voice ingress",
        # No docs, no schema. A public process publishing its own API surface
        # is an invitation, and the three routes are documented in the repo
        # where the people who need them can read them.
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    # Everything this app holds. No database, no blob store, no corpus, no
    # customer service, no model broker, no settings object.
    app.state.config = config
    app.state.gate = asyncio.Semaphore(config.max_concurrency)
    app.state.limiter = RateLimiter(config.rate_per_minute)
    app.state.seen = Seen()
    app.state.client = None

    @app.get("/health")
    async def health() -> dict[str, Any]:
        # No secret, no loopback address, no configuration detail: this route
        # exists so a supervisor can see the process is up.
        return {"status": "ok", "service": "metis-voice-ingress"}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        if not _authorized(request, config):
            return _unauthorized()

        body = await _read_bounded(request, config.max_body_bytes)
        if body is None:
            return JSONResponse({"error": "request too large"}, status_code=413)
        try:
            payload = json.loads(body)
        except ValueError:
            return JSONResponse({"error": "malformed request"}, status_code=400)
        if not isinstance(payload, dict):
            return JSONResponse({"error": "malformed request"}, status_code=400)

        alias = str(payload.get("model") or "").strip()
        if alias != config.public_model_alias:
            # The alias is checked, never used. Which model reasons is the
            # owner's stored preference on :8000, and a caller naming one here
            # is naming something this process will not act on.
            return JSONResponse({"error": "unknown model"}, status_code=422)
        if not payload.get("stream"):
            return JSONResponse(
                {"error": "this endpoint streams only"}, status_code=422
            )
        tools = payload.get("tools") or []
        if not _tools_allowed(tools):
            return JSONResponse({"error": "unsupported tool"}, status_code=422)

        session = _session_key(request, payload)
        metis_session = _metis_session(payload)
        if not app.state.limiter.allow(session, now=time.monotonic()):
            return JSONResponse({"error": "too many requests"}, status_code=429)

        transcript, history = _conversation(payload)
        if not transcript:
            return JSONResponse({"error": "no utterance"}, status_code=422)

        sentences = _ask_metis(
            app, config, session, metis_session, transcript, history
        )
        return StreamingResponse(_sse(sentences, alias), media_type="text/event-stream")

    @app.post("/v1/elevenlabs/post-call")
    async def post_call(request: Request) -> Any:
        body = await _read_bounded(request, config.max_body_bytes)
        if body is None:
            return JSONResponse({"error": "request too large"}, status_code=413)
        # Verified against the raw bytes, before anything parses them: a
        # signature checked after parsing is a signature over a different
        # document than the one that was sent.
        if not _webhook_verified(request, body, config):
            return _unauthorized()
        try:
            payload = json.loads(body)
        except ValueError:
            return JSONResponse({"error": "malformed request"}, status_code=400)
        if not isinstance(payload, dict):
            return JSONResponse({"error": "malformed request"}, status_code=400)

        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        conversation = str(data.get("conversation_id") or "").strip()
        event_type = str(payload.get("type") or "post_call").strip()
        seen_key = f"{conversation}:{event_type}"
        if seen_key in app.state.seen:
            return {"status": "duplicate"}

        client: httpx.AsyncClient | None = app.state.client
        if client is None:
            return JSONResponse({"error": "unavailable"}, status_code=503)
        try:
            response = await client.post(
                "/api/v1/voice/post-call",
                json={
                    "event_type": event_type,
                    "provider_conversation_id": conversation,
                    "payload": payload,
                    "body_sha256": hashlib.sha256(body).hexdigest(),
                },
                headers={"authorization": f"Bearer {config.shared_secret}"},
            )
        except httpx.HTTPError:
            # A non-2xx asks ElevenLabs to retry. Acknowledging before the
            # trusted process has stored the evidence silently loses calls.
            return JSONResponse({"error": "storage unavailable"}, status_code=503)
        if response.status_code >= 400:
            return JSONResponse({"error": "storage unavailable"}, status_code=503)
        app.state.seen.add(seen_key)
        return {"status": "accepted"}

    return app


def _authorized(request: Request, config: IngressConfig) -> bool:
    header = request.headers.get("authorization") or ""
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return False
    return hmac.compare_digest(token.strip(), config.shared_secret)


async def _read_bounded(request: Request, limit: int) -> bytes | None:
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        return None
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def _tools_allowed(tools: Any) -> bool:
    if not isinstance(tools, list):
        return False
    for tool in tools:
        if not isinstance(tool, dict):
            return False
        function = (
            tool.get("function") if isinstance(tool.get("function"), dict) else {}
        )
        name = str(function.get("name") or tool.get("name") or "").strip()
        if name not in ALLOWED_TOOLS:
            return False
    return True


def _session_key(request: Request, payload: dict[str, Any]) -> str:
    """A stable per-conversation key for rate limiting and correlation.

    ElevenLabs identifies a conversation in more than one place depending on
    how the agent is configured, so all of them are read and none are trusted
    for anything but bucketing — the worst a forged value can do is share a
    rate-limit window with someone else's.
    """
    extra = payload.get("elevenlabs_extra_body")
    if isinstance(extra, dict):
        for key in ("conversation_id", "session_id"):
            value = str(extra.get(key) or "").strip()
            if value:
                return value[:120]
    for header in ("elevenlabs-conversation-id", "x-conversation-id"):
        value = (request.headers.get(header) or "").strip()
        if value:
            return value[:120]
    user = str(payload.get("user") or "").strip()
    return user[:120] if user else "anonymous"


def _metis_session(payload: dict[str, Any]) -> str:
    """The tab identity supplied as Custom LLM extra body, never authority."""
    extra = payload.get("elevenlabs_extra_body")
    if not isinstance(extra, dict):
        return ""
    value = str(extra.get("metis_session_id") or "").strip()
    return value[:120] if value.startswith("vs_") else ""


def _conversation(payload: dict[str, Any]) -> tuple[str, list[str]]:
    """The latest utterance and bounded prior turns — never a system message.

    A system message from this side of the tunnel is somebody writing Metis's
    instructions from the internet. They are dropped, not sanitized: there is
    no version of "you are now in developer mode" that becomes safe by being
    escaped, and the real system prompt is on the trusted side.
    """
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return "", []
    spoken: list[tuple[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip().lower()
        if role not in ("user", "assistant"):
            continue
        content = _text_of(message.get("content"))
        if content:
            spoken.append((role, content))
    if not spoken:
        return "", []
    # The question itself keeps a generous ceiling; only the history behind it
    # is squeezed. Truncating what someone just asked to save tokens would be
    # saving them in the one place they cannot be spared.
    latest = next(
        (
            text[:MAX_UTTERANCE_CHARS]
            for role, text in reversed(spoken)
            if role == "user"
        ),
        "",
    )
    history = [
        f"{'You' if role == 'assistant' else 'They'}: {text[:MAX_MESSAGE_CHARS]}"
        for role, text in spoken[:-1][-MAX_HISTORY_MESSAGES:]
    ]
    return latest, history


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            str(block.get("text") or "")
            for block in content
            if isinstance(block, dict) and block.get("type") in (None, "text")
        ]
        return " ".join(part for part in parts if part).strip()
    return ""


# What this process says when the trusted side could not. Fixed and vague on
# purpose: whatever went wrong is diagnosed on :8000, and its message is not
# for the far end of a tunnel.
UNREACHABLE = "I couldn't reach Metis just then. Try me again."
FAILED = "Something went wrong on my side. Try me again."
NO_ANSWER = "I don't have an answer for that one."


async def _ask_metis(
    app: FastAPI,
    config: IngressConfig,
    session: str,
    metis_session: str,
    transcript: str,
    history: list[str],
) -> AsyncIterator[str]:
    """One question to the trusted process; its spoken sentences back, as they
    are decided.

    The written answer, the citations and any receipt travel to the browser
    over :8000's own loopback stream. Nothing but the words to be spoken
    crosses back through here — the tunnel carries speech, not state. A
    failure after the first sentence is left to end the answer where it
    stopped: an apology tacked onto half an answer is worse than the half.
    """
    client: httpx.AsyncClient | None = app.state.client
    if client is None:
        yield UNREACHABLE
        return
    spoke = False
    async with app.state.gate:
        try:
            async with client.stream(
                "POST",
                "/api/v1/voice/turn",
                json={
                    "provider_conversation_id": session,
                    "metis_session_id": metis_session,
                    "transcript": transcript,
                    "history": history,
                },
                headers={"authorization": f"Bearer {config.shared_secret}"},
            ) as response:
                if response.status_code >= 400:
                    yield FAILED
                    return
                async for line in response.aiter_lines():
                    frame = _frame(line)
                    if frame is None:
                        continue
                    if frame["type"] == "spoken":
                        spoke = True
                        yield frame["text"]
                    elif frame["type"] == "error":
                        if not spoke:
                            yield FAILED
                        return
                    elif frame["type"] == "done":
                        break
        except httpx.HTTPError:
            if not spoke:
                yield UNREACHABLE
            return
    if not spoke:
        yield NO_ANSWER


def _frame(line: str) -> dict[str, Any] | None:
    """One line of the trusted process's reply, or None for anything else."""
    try:
        frame = json.loads(line)
    except ValueError:
        return None
    if not isinstance(frame, dict):
        return None
    kind = str(frame.get("type") or "")
    text = str(frame.get("text") or "").strip()
    if kind == "spoken" and text:
        return {"type": "spoken", "text": text}
    if kind in ("done", "error"):
        return {"type": kind}
    return None


async def _sse(sentences: AsyncIterator[str], alias: str):
    """The OpenAI streaming shape ElevenLabs expects, terminated properly.

    One content chunk per sentence, sent the moment the trusted side decided
    it, so the voice starts on the first sentence while the rest is still
    being written.
    """
    created = int(time.time())
    identifier = f"chatcmpl-{created}"

    def frame(delta: dict[str, Any], finish: str | None) -> bytes:
        return (
            b"data: "
            + json.dumps(
                {
                    "id": identifier,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": alias,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }
            ).encode("utf-8")
            + b"\n\n"
        )

    yield frame({"role": "assistant"}, None)
    first = True
    async for sentence in sentences:
        # Chunks are concatenated on the far side, so each sentence after the
        # first carries the space that separates it from the one before.
        yield frame({"content": sentence if first else f" {sentence}"}, None)
        first = False
    yield frame({}, "stop")
    yield b"data: [DONE]\n\n"


def _webhook_verified(request: Request, body: bytes, config: IngressConfig) -> bool:
    """ElevenLabs' HMAC over `timestamp.body`, with a bounded age.

    No secret configured means no verification is possible, which means the
    payload is not evidence — so it is refused rather than trusted.
    """
    if not config.webhook_secret:
        return False
    header = request.headers.get("elevenlabs-signature") or ""
    timestamp = ""
    signature = ""
    for part in header.split(","):
        key, _, value = part.strip().partition("=")
        if key == "t":
            timestamp = value
        elif key in ("v0", "v1"):
            signature = value
    if not timestamp or not signature:
        return False
    try:
        age = time.time() - float(timestamp)
    except ValueError:
        return False
    if not -WEBHOOK_TOLERANCE_SECONDS <= age <= WEBHOOK_TOLERANCE_SECONDS:
        return False
    expected = hmac.new(
        config.webhook_secret.encode("utf-8"),
        f"{timestamp}.".encode("utf-8") + body,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)
