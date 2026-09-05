"""Who is talking, for how long, and what is running because of it.

The trusted process owns the whole lifecycle, and that is the point rather
than an implementation detail. The browser asks for a session and renews a
short lease; it does not start the tunnel, does not hold the API key, and does
not get to decide when the connector stops. Client cleanup is helpful and
never trusted — a crashed tab cannot tell anyone it crashed, so the lease it
stopped renewing is what closes the tunnel behind it.

Two things are deliberately refused rather than arranged automatically:

* The connector will not start unless the owner has explicitly said this
  machine may run one. The device is Oracle-managed, and a recorded
  authorization in a document is not the same as a go-ahead at the moment a
  process first opens an outbound tunnel.
* The conversation token is minted here, server-side, and returned once. The
  ElevenLabs API key never reaches the browser at all.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import secrets
import shutil
import sys
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, AsyncIterator

from .config import Settings
from .contracts import VoiceRenditionV1, VoiceSessionStartV1, VoiceSessionV1
from .model_provider import ModelProviderError
from .voice_graph import OnSpoken, VoiceGraph, VoiceTurn
from .voice_intents import classify_refusal

logger = logging.getLogger("waqil.voice.session")

# A conversation token is valid for ten minutes. Keep a shorter two-minute
# window so opening Voice can hide the remote token/tunnel startup without
# leaving either process warm after someone simply looked at the screen.
VOICE_PREWARM_SECONDS = 120
# A live audio session should not remain billable simply because its browser is
# still healthy enough to renew a lease. Two quiet minutes is long enough to
# think or read a response, while still making an abandoned tab inexpensive.
VOICE_IDLE_TIMEOUT_SECONDS = 120
VOICE_TURN_CACHE_LIMIT = 64


def _turn_fingerprint(
    provider_conversation_id: str, transcript: str, history: tuple[str, ...]
) -> str:
    """Stable identity for one Custom LLM request, including its position.

    The utterance alone is not an idempotency key: "yes" said after a later
    assistant turn is a new request. Provider conversation history supplies
    that position and is identical when ElevenLabs retries the same payload.
    """
    canonical = json.dumps(
        {
            "provider_conversation_id": provider_conversation_id.strip(),
            "transcript": transcript.strip(),
            "history": list(history),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class VoiceUnavailable(RuntimeError):
    """Voice cannot start, and the message says exactly what is missing."""


class VoiceSessionExpired(RuntimeError):
    """The named session is not open."""


@dataclass
class _TurnAttempt:
    """One provider request, retained so a transport retry is the same turn."""

    turn_id: str
    run_id: str | None = None
    rendition: VoiceRenditionV1 | None = None


@dataclass
class _Session:
    id: str
    conversation_id: str
    started_at: datetime
    lease_expires_at: datetime
    last_activity_at: datetime
    voice_model: str
    spoken_confirmation: bool
    provider_conversation_id: str = ""
    state: str = "starting"
    # Set as soon as an End request arrives. The active turn is allowed to
    # finish under ``turn_lock`` before the session becomes ended, while any
    # turn queued behind it is refused rather than slipping in ahead of close.
    closing: bool = field(default=False, repr=False)
    ended_at: datetime | None = None
    reason: str = ""
    account_id: str | None = None
    turns: int = 0
    # What was said, bounded. Held in memory for the length of the session and
    # nowhere else: durable history is the conversation's own messages.
    history: list[str] = field(default_factory=list)
    # ElevenLabs may retry a Custom LLM request after losing the response. A
    # per-session lock makes the lookup-and-run atomic, and the fingerprint
    # includes the provider's preceding history so somebody genuinely saying
    # the same thing twice still creates two turns.
    turn_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    turn_attempts: dict[str, _TurnAttempt] = field(default_factory=dict, repr=False)
    # The live channel to the browser. Durability is not this queue's job —
    # written answers land in the conversation, receipts land in their own
    # records, and a reconnecting browser re-reads both.
    listeners: list[asyncio.Queue] = field(default_factory=list)

    @property
    def live(self) -> bool:
        return self.state in ("starting", "live")

    def elapsed(self, now: datetime) -> float:
        return ((self.ended_at or now) - self.started_at).total_seconds()

    def idle_remaining(self, now: datetime) -> float:
        return max(
            0.0,
            VOICE_IDLE_TIMEOUT_SECONDS - (now - self.last_activity_at).total_seconds(),
        )

    def public(self, now: datetime) -> VoiceSessionV1:
        return VoiceSessionV1(
            id=self.id,
            conversation_id=self.conversation_id,
            state=self.state,  # type: ignore[arg-type]
            provider_conversation_id=self.provider_conversation_id,
            started_at=self.started_at,
            lease_expires_at=self.lease_expires_at,
            ended_at=self.ended_at,
            reason=self.reason,
            spoken_confirmation=self.spoken_confirmation,
            voice_model=self.voice_model,
            elapsed_seconds=round(self.elapsed(now), 1),
            idle_timeout_seconds=VOICE_IDLE_TIMEOUT_SECONDS,
            idle_seconds_remaining=round(self.idle_remaining(now), 1),
        )


class VoiceSessionService:
    """Sessions, leases, and the two processes a live session keeps alive."""

    def __init__(
        self,
        settings: Settings,
        *,
        graph: VoiceGraph,
        speech_preference: Any,
        model: Any,
        database: Any = None,
    ) -> None:
        self.settings = settings
        self.graph = graph
        self.speech_preference = speech_preference
        self.model = model
        self.database = database
        self._sessions: dict[str, _Session] = {}
        self._lock = asyncio.Lock()
        self._ingress: asyncio.subprocess.Process | None = None
        self._connector: asyncio.subprocess.Process | None = None
        self._secret: str = ""
        self._process_errors: dict[int, str] = {}
        self._prepared_token = ""
        self._prepared_until: datetime | None = None

    # -- readiness --------------------------------------------------------

    def unavailable_reasons(self) -> list[str]:
        """Every unmet prerequisite, in the order a person should fix it."""
        reasons: list[str] = []
        if not self.settings.voice_enabled:
            return ["Voice mode is switched off in this Metis configuration."]
        speech = getattr(self.model, "elevenlabs", None)
        if speech is None or not speech.available:
            reasons.append("Voice needs WAQIL_ELEVENLABS_API_KEY.")
        if not self.settings.elevenlabs_agent_id.strip():
            reasons.append(
                "Voice needs an ElevenLabs agent. Create one, then set "
                "WAQIL_ELEVENLABS_AGENT_ID."
            )
        if not self.settings.voice_tunnel_hostname.strip():
            reasons.append(
                "Voice needs the named tunnel's hostname in "
                "WAQIL_VOICE_TUNNEL_HOSTNAME."
            )
        if not self.settings.voice_tunnel_authorized:
            # A.3: the appendix records that the owner authorized a connector
            # in principle, and explicitly asks for one further go-ahead at the
            # moment one is first started on this managed device. This flag is
            # that go-ahead, and its absence is not a bug to route around.
            reasons.append(
                "Starting a tunnel on this managed device needs an explicit "
                "go-ahead: set WAQIL_VOICE_TUNNEL_AUTHORIZED=true."
            )
        if shutil.which(self.settings.cloudflared_path) is None:
            reasons.append(f"`{self.settings.cloudflared_path}` is not installed.")
        return reasons

    def unavailable_reason(self) -> str:
        reasons = self.unavailable_reasons()
        return reasons[0] if reasons else ""

    # -- lifecycle --------------------------------------------------------

    async def prewarm(self) -> None:
        """Prepare the tunnel and short-lived provider token, but no call.

        This creates no Metis conversation and opens no WebRTC/audio session.
        It is safe to run when the Voice surface appears, then consume once
        when the owner actually presses Start.
        """
        reason = self.unavailable_reason()
        if reason:
            raise VoiceUnavailable(reason)
        async with self._lock:
            now = datetime.now(UTC)
            if self._prepared_valid(now):
                return
            self._clear_prepared()
            try:
                # Token creation and the local connector are independent until
                # the first finalized utterance, so pay both waits in parallel.
                _, token = await asyncio.gather(
                    self._ensure_processes(), self._conversation_token()
                )
            except ModelProviderError as error:
                await self._release_if_idle()
                raise VoiceUnavailable(str(error)) from error
            except Exception:
                await self._release_if_idle()
                raise
            self._prepared_token = token
            self._prepared_until = now + timedelta(seconds=VOICE_PREWARM_SECONDS)
            logger.info("voice transport prepared")

    async def start(self) -> VoiceSessionStartV1:
        reason = self.unavailable_reason()
        if reason:
            raise VoiceUnavailable(reason)
        async with self._lock:
            await self._ensure_processes()
            preference = self.speech_preference.load()
            now = datetime.now(UTC)
            conversation = await self._conversation()
            session = _Session(
                id=f"vs_{uuid.uuid4().hex[:20]}",
                conversation_id=conversation,
                started_at=now,
                lease_expires_at=now
                + timedelta(seconds=self.settings.voice_lease_seconds),
                last_activity_at=now,
                voice_model=preference.voice_model,
                spoken_confirmation=preference.spoken_confirmation,
            )
            self._sessions[session.id] = session
            try:
                token = self._take_prepared(datetime.now(UTC))
                if not token:
                    token = await self._conversation_token()
            except ModelProviderError as error:
                session.state, session.reason = "failed", str(error)
                await self._release_if_idle()
                raise VoiceUnavailable(str(error)) from error
            session.state = "live"
            logger.info("voice session %s open", session.id)
            return VoiceSessionStartV1(
                session=session.public(now),
                conversation_token=token,
                lease_seconds=self.settings.voice_lease_seconds,
            )

    async def renew(self, session_id: str) -> VoiceSessionV1:
        session = self._require(session_id)
        now = datetime.now(UTC)
        if session.elapsed(now) > self.settings.voice_session_max_seconds:
            return await self.end(session_id, reason="it reached its time limit")
        if session.idle_remaining(now) <= 0:
            return await self.end(
                session_id, reason="there was no voice activity for two minutes"
            )
        session.lease_expires_at = now + timedelta(
            seconds=self.settings.voice_lease_seconds
        )
        return session.public(now)

    async def end(
        self, session_id: str, *, reason: str = "you stopped it"
    ) -> VoiceSessionV1:
        session = self._sessions.get(session_id)
        if session is None:
            raise VoiceSessionExpired("that voice session is not open")
        if session.live:
            session.closing = True
        # End and turns share the same per-session fence. An answer already in
        # progress completes before End returns; once closing is set, queued or
        # later turns are refused. This also ensures no customer append can land
        # after the session has reported itself ended.
        async with session.turn_lock:
            now = datetime.now(UTC)
            if session.live:
                session.state, session.ended_at, session.reason = "ended", now, reason
                # A read-back waiting for a yes does not outlive the conversation
                # it was asked in.
                self.graph.forget(session.id)
                await self._publish(session, {"type": "voice.ended", "reason": reason})
                for listener in session.listeners:
                    # None is the stream's own end-of-stream marker.
                    listener.put_nowait(None)
                logger.info(
                    "voice session %s closed after %.1fs (%s)",
                    session.id,
                    session.elapsed(now),
                    reason,
                )
            session.closing = False
        async with self._lock:
            await self._release_if_idle()
        return session.public(now)

    async def status(self, session_id: str) -> VoiceSessionV1:
        return self._require(session_id).public(datetime.now(UTC))

    async def bind(
        self, session_id: str, provider_conversation_id: str
    ) -> VoiceSessionV1:
        """Attach the browser SDK's conversation to exactly one live lease.

        This is intentionally a loopback call from the browser after WebRTC
        connects. It avoids enabling provider-controlled Custom LLM body
        overrides merely to carry a Metis session id through the tunnel.
        """
        conversation = (provider_conversation_id or "").strip()
        if not conversation:
            raise VoiceSessionExpired("the provider conversation id is missing")
        async with self._lock:
            session = self._require(session_id)
            if not session.live or session.closing:
                raise VoiceSessionExpired("that Metis voice session is not open")
            if (
                session.provider_conversation_id
                and session.provider_conversation_id != conversation
            ):
                raise VoiceSessionExpired(
                    "that Metis voice session is already bound to another conversation"
                )
            owner = next(
                (
                    candidate
                    for candidate in self._sessions.values()
                    if candidate.id != session.id
                    and candidate.live
                    and candidate.provider_conversation_id == conversation
                ),
                None,
            )
            if owner is not None:
                raise VoiceSessionExpired(
                    "that provider conversation belongs to another session"
                )
            session.provider_conversation_id = conversation
            return session.public(datetime.now(UTC))

    async def sweep(self) -> None:
        """Expire leases nobody renewed. The backstop for a crashed browser."""
        now = datetime.now(UTC)
        stale = [
            (
                session.id,
                "there was no voice activity for two minutes"
                if session.idle_remaining(now) <= 0
                else "the connection went quiet",
            )
            for session in self._sessions.values()
            if session.live
            and (session.lease_expires_at <= now or session.idle_remaining(now) <= 0)
        ]
        for session_id, reason in stale:
            await self.end(session_id, reason=reason)
        # Ended sessions are kept briefly so a late status poll gets an answer
        # rather than a 404, then dropped.
        cutoff = now - timedelta(minutes=10)
        for session_id, session in list(self._sessions.items()):
            if session.ended_at is not None and session.ended_at < cutoff:
                self._sessions.pop(session_id, None)
        if self._prepared_until is not None and self._prepared_until <= now:
            async with self._lock:
                if self._prepared_until is not None and self._prepared_until <= now:
                    self._clear_prepared()
                    await self._release_if_idle()

    async def shutdown(self) -> None:
        self._clear_prepared()
        for session_id in list(self._sessions):
            with contextlib.suppress(Exception):
                await self.end(session_id, reason="Metis is shutting down")
        async with self._lock:
            await self._stop_processes()

    # -- one spoken turn --------------------------------------------------

    async def turn(
        self,
        *,
        provider_conversation_id: str,
        metis_session_id: str = "",
        transcript: str,
        history: list[str] | None = None,
        on_spoken: OnSpoken | None = None,
    ) -> VoiceRenditionV1:
        """The ingress asking the trusted process a question.

        Binding is first-come: a provider conversation that has not been seen
        binds to the oldest live session waiting for one. Metis is a
        single-user application, so this is a queue of at most a few tabs, and
        an utterance arriving for a session that does not exist is refused
        rather than answered on a guess.

        `on_spoken` hears each sentence the moment it may be said, ahead of
        the rendition this returns once the turn is complete.
        """
        session = self._bind(provider_conversation_id, metis_session_id)

        # The lock is deliberately per session rather than global: retries for
        # this conversation serialize, while two independent tabs can still
        # answer concurrently.
        async with session.turn_lock:
            if session.closing or not session.live:
                raise VoiceSessionExpired("that voice session is not open")
            request_history = (
                tuple(history) if history is not None else tuple(session.history)
            )
            fingerprint = _turn_fingerprint(
                provider_conversation_id, transcript, request_history
            )
            session.last_activity_at = datetime.now(UTC)
            attempt = session.turn_attempts.get(fingerprint)
            if attempt is not None and attempt.rendition is not None:
                # A retry still has to be heard: the provider lost the reply,
                # so the cached words are said again without a second turn.
                if on_spoken is not None:
                    await on_spoken(attempt.rendition.spoken)
                return attempt.rendition

            if attempt is None:
                session.turns += 1
                attempt = _TurnAttempt(
                    turn_id=f"t_{session.turns:04d}_{fingerprint[:8]}"
                )
                session.turn_attempts[fingerprint] = attempt
                while len(session.turn_attempts) > VOICE_TURN_CACHE_LIMIT:
                    session.turn_attempts.pop(next(iter(session.turn_attempts)))

            if attempt.run_id is None:
                attempt.run_id = await self._record_question(
                    session, transcript, attempt.turn_id
                )

            rendition = await self.graph.answer(
                VoiceTurn(
                    transcript=transcript,
                    voice_session_id=session.id,
                    turn_id=attempt.turn_id,
                    run_id=attempt.run_id,
                    account_id=session.account_id,
                    history=tuple(history or session.history),
                ),
                on_spoken=on_spoken,
            )
            # Cache before the non-critical twin-write and live publication.
            # A retry after the provider received the answer must never run the
            # model or commit the user's request again.
            attempt.rendition = rendition
            session.history.append(f"They: {transcript}")
            session.history.append(f"You: {rendition.spoken}")
            del session.history[:-12]
            session.last_activity_at = datetime.now(UTC)

            await self._record_answer(session, rendition)
            await self._publish(
                session,
                {"type": "voice.turn", "rendition": rendition.model_dump(mode="json")},
            )
            if rendition.intent in ("refuse_build", "refuse_protected"):
                refusal = classify_refusal(transcript)
                await self._publish(
                    session,
                    {
                        "type": "voice.build_deferred"
                        if rendition.intent == "refuse_build"
                        else "voice.refused",
                        # The verbatim utterance, so the composer is prefilled with
                        # what was actually said rather than a paraphrase of it.
                        "transcript": transcript,
                        "hand_off": bool(refusal and refusal.hand_off),
                        "voice_session_id": session.id,
                        "turn_id": attempt.turn_id,
                        "run_id": attempt.run_id,
                    },
                )
            return rendition

    async def turn_stream(self, **turn_arguments: Any) -> AsyncIterator[dict[str, Any]]:
        """The turn as the ingress consumes it: one frame per spoken sentence.

        Frames are `{"type": "spoken", "text": ...}` as each sentence clears
        the gate, then `{"type": "done"}`; a failure becomes one
        `{"type": "error", "detail": ...}` frame. The turn runs as its own
        task so a slow model never blocks the frames already decided.
        """
        frames: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        async def say(sentence: str) -> None:
            await frames.put({"type": "spoken", "text": sentence})

        async def run() -> None:
            try:
                await self.turn(**turn_arguments, on_spoken=say)
                await frames.put({"type": "done"})
            except VoiceSessionExpired as error:
                await frames.put({"type": "error", "detail": str(error)})
            except Exception as error:  # noqa: BLE001 - the caller hears one frame
                logger.warning("voice turn failed: %s", str(error)[:200])
                await frames.put({"type": "error", "detail": "the turn failed"})
            finally:
                await frames.put(None)

        task = asyncio.create_task(run())
        try:
            while (frame := await frames.get()) is not None:
                yield frame
        finally:
            if not task.done():
                task.cancel()

    async def post_call(self, payload: dict[str, Any]) -> dict[str, Any]:
        """A verified provider webhook, stored as evidence and nothing more.

        Nothing here becomes a customer fact, an action, a memory or an
        account link. The provider's own analysis is recorded as what it is —
        untrusted output from a service that was listening — and every
        structured thing anyone wants out of it stays a proposal a person
        accepts.
        """
        conversation = str(payload.get("provider_conversation_id") or "").strip()
        session = next(
            (
                item
                for item in self._sessions.values()
                if item.provider_conversation_id == conversation
            ),
            None,
        )
        body = (
            payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
        )
        data = body.get("data") if isinstance(body.get("data"), dict) else {}
        transcript = data.get("transcript")
        record_id = ""
        if self.database is not None:
            try:
                record_id = await self.database.record_voice_post_call(
                    provider_conversation_id=conversation,
                    event_type=str(payload.get("event_type") or "post_call"),
                    voice_session_id=session.id if session else "",
                    conversation_id=session.conversation_id if session else None,
                    body_sha256=str(payload.get("body_sha256") or ""),
                    transcript=transcript if isinstance(transcript, list) else [],
                    # Stored under a name that says what it is. A transcription
                    # service's opinion about what was agreed is not a decision
                    # this host made, and nothing downstream may read it as one.
                    provider_analysis=(
                        data.get("analysis")
                        if isinstance(data.get("analysis"), dict)
                        else {}
                    ),
                )
            except Exception as error:  # noqa: BLE001 - caller must retry persistence
                logger.warning("voice post-call not stored: %s", str(error)[:200])
                raise
        logger.info(
            "voice post-call received for %s (session %s)",
            conversation or "unknown",
            session.id if session else "none",
        )
        if session is not None:
            await self._publish(
                session,
                {"type": "voice.post_call", "provider_conversation_id": conversation},
            )
        return {
            "status": "stored" if record_id else "received",
            "record_id": record_id,
            "provider_conversation_id": conversation,
            "voice_session_id": session.id if session else "",
        }

    # -- the browser's live channel ---------------------------------------

    async def events(self, session_id: str) -> AsyncIterator[dict[str, Any]]:
        session = self._require(session_id)
        queue: asyncio.Queue = asyncio.Queue(maxsize=64)
        session.listeners.append(queue)
        try:
            while True:
                event = await queue.get()
                if event is None:
                    return
                yield event
        finally:
            with contextlib.suppress(ValueError):
                session.listeners.remove(queue)

    async def _publish(self, session: _Session, event: dict[str, Any]) -> None:
        for listener in list(session.listeners):
            try:
                listener.put_nowait(event)
            except asyncio.QueueFull:
                # A browser that stopped reading loses events rather than
                # stalling the turn that produced them.
                logger.info("voice listener fell behind on %s", session.id)

    # -- persistence ------------------------------------------------------

    async def _conversation(self) -> str:
        if self.database is None:
            return f"conv_voice_{uuid.uuid4().hex[:12]}"
        created = await self.database.create_conversation("Voice conversation")
        return created.id

    async def _record_question(
        self, session: _Session, transcript: str, turn_id: str
    ) -> str:
        """The utterance, in the conversation, as a message the user sent."""
        if self.database is None:
            return ""
        try:
            message = await self.database.add_message(
                session.conversation_id, "user", transcript
            )
            return str(getattr(message, "id", "") or "")
        except Exception as error:  # noqa: BLE001 - a turn survives a write failure
            logger.info("voice question not recorded: %s", str(error)[:200])
            return ""

    async def _record_answer(
        self, session: _Session, rendition: VoiceRenditionV1
    ) -> None:
        if self.database is None or not rendition.written.strip():
            return
        try:
            await self.database.add_message(
                session.conversation_id, "assistant", rendition.written
            )
        except Exception as error:  # noqa: BLE001 - the answer was already spoken
            logger.info("voice answer not recorded: %s", str(error)[:200])

    # -- processes --------------------------------------------------------

    def shared_secret(self) -> str:
        """The bearer the ingress checks, minted locally on first use.

        The environment value wins when it is set. Otherwise one is generated
        and written 0600 beside the rest of the local state, because the
        alternative — a default — is an open door, and asking someone to
        invent 32 random characters before they can try voice is how a feature
        goes unused.
        """
        if self._secret:
            return self._secret
        configured = self.settings.voice_shared_secret.strip()
        if configured:
            self._secret = configured
            return self._secret
        path = self.settings.voice_secret_path
        try:
            if path.is_file():
                stored = path.read_text(encoding="utf-8").strip()
                if len(stored) >= 32:
                    self._secret = stored
                    return self._secret
        except OSError:
            pass
        minted = secrets.token_urlsafe(48)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(minted, encoding="utf-8")
            path.chmod(0o600)
        except OSError as error:
            logger.info("voice secret not persisted: %s", str(error)[:200])
        self._secret = minted
        return self._secret

    async def _ensure_processes(self) -> None:
        if self._ingress is None or self._ingress.returncode is not None:
            self._ingress = await self._spawn_ingress()
            await self._wait_for_ingress()
        if self._connector is None or self._connector.returncode is not None:
            self._connector = await self._spawn_connector()
            # A bad tunnel name or missing credentials exits immediately. Give
            # it one scheduler turn and report that at Start, not after WebRTC
            # has connected to a dead endpoint.
            await asyncio.sleep(0.15)
            if self._connector.returncode is not None:
                detail = await self._process_error(self._connector)
                await self._stop_processes()
                raise VoiceUnavailable(
                    "The Cloudflare tunnel could not start"
                    + (f": {detail}" if detail else ".")
                )

    async def _spawn_ingress(self) -> asyncio.subprocess.Process:
        environment = {
            **os.environ,
            "WAQIL_VOICE_SHARED_SECRET": self.shared_secret(),
            "WAQIL_VOICE_PUBLIC_MODEL_ALIAS": self.settings.voice_public_model_alias,
            "WAQIL_VOICE_INGRESS_HOST": self.settings.voice_ingress_host,
            "WAQIL_VOICE_INGRESS_PORT": str(self.settings.voice_ingress_port),
            "WAQIL_VOICE_LOOPBACK_URL": f"http://{self.settings.host}:{self.settings.port}",
        }
        logger.info(
            "starting voice ingress on port %s", self.settings.voice_ingress_port
        )
        process = await asyncio.create_subprocess_exec(
            self.settings.voice_ingress_command.strip() or sys.executable,
            "-m",
            "waqil_voice_ingress.main",
            env=environment,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        self._watch_stderr(process, "voice ingress")
        return process

    async def _spawn_connector(self) -> asyncio.subprocess.Process:
        """The named tunnel, pointed at the ingress port and nothing else.

        `--url` names the loopback address of the ingress. Port 8000 is never
        an argument here, and there is no code path that could make it one.
        """
        logger.info(
            "starting cloudflared for tunnel %s", self.settings.voice_tunnel_name
        )
        process = await asyncio.create_subprocess_exec(
            self.settings.cloudflared_path,
            "tunnel",
            "run",
            "--url",
            f"http://{self.settings.voice_ingress_host}:{self.settings.voice_ingress_port}",
            self.settings.voice_tunnel_name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        self._watch_stderr(process, "cloudflared")
        return process

    def _watch_stderr(self, process: asyncio.subprocess.Process, name: str) -> None:
        """Drain child stderr continuously so a full pipe cannot stall voice."""

        async def drain() -> None:
            stream = process.stderr
            if stream is None:
                return
            try:
                while line := await stream.readline():
                    text = line.decode("utf-8", errors="replace").strip()
                    if text:
                        self._process_errors[id(process)] = text[-400:]
                        logger.info("%s: %s", name, text[:400])
            except (OSError, RuntimeError):
                return

        asyncio.create_task(drain(), name=f"{name}-stderr")

    async def _wait_for_ingress(self) -> None:
        """Do not call a session live until the isolated adapter answers."""
        import urllib.error
        import urllib.request

        url = (
            f"http://{self.settings.voice_ingress_host}:"
            f"{self.settings.voice_ingress_port}/health"
        )
        for _ in range(30):
            process = self._ingress
            if process is None or process.returncode is not None:
                detail = await self._process_error(process)
                await self._stop_processes()
                raise VoiceUnavailable(
                    "The isolated voice service could not start"
                    + (f": {detail}" if detail else ".")
                )
            try:
                response = await asyncio.to_thread(
                    urllib.request.urlopen, url, timeout=0.25
                )
                with response:
                    if response.status == 200:
                        return
            except (OSError, urllib.error.URLError):
                await asyncio.sleep(0.1)
        await self._stop_processes()
        raise VoiceUnavailable("The isolated voice service did not become ready.")

    async def _process_error(self, process: asyncio.subprocess.Process | None) -> str:
        if process is None:
            return ""
        # Let the drainer consume any final line written immediately before the
        # process exited, then return only its bounded tail.
        await asyncio.sleep(0)
        return self._process_errors.get(id(process), "")

    async def _release_if_idle(self) -> None:
        """Stop both processes once no live lease needs them.

        Checked across every session rather than per tab: two windows can each
        hold a session, and closing one must not take the connector out from
        under the other.
        """
        if any(session.live for session in self._sessions.values()):
            return
        if self._prepared_valid(datetime.now(UTC)):
            return
        await self._stop_processes()

    def _prepared_valid(self, now: datetime) -> bool:
        return bool(
            self._prepared_token
            and self._prepared_until is not None
            and self._prepared_until > now
        )

    def _take_prepared(self, now: datetime) -> str:
        if not self._prepared_valid(now):
            self._clear_prepared()
            return ""
        token = self._prepared_token
        self._clear_prepared()
        return token

    def _clear_prepared(self) -> None:
        self._prepared_token = ""
        self._prepared_until = None

    async def _stop_processes(self) -> None:
        for name, process in (
            ("cloudflared", self._connector),
            ("voice ingress", self._ingress),
        ):
            if process is None or process.returncode is not None:
                continue
            logger.info("stopping %s", name)
            with contextlib.suppress(ProcessLookupError):
                process.terminate()
            with contextlib.suppress(TimeoutError, ProcessLookupError):
                await asyncio.wait_for(process.wait(), timeout=5)
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
        self._connector = None
        self._ingress = None

    # -- the provider -----------------------------------------------------

    async def _conversation_token(self) -> str:
        """A short-lived WebRTC conversation token, minted here, returned once.

        The WebRTC endpoint, not the signed-URL one: the SDK's session config
        accepts `conversationToken` or `signedUrl` and refuses both together,
        and those are different provider endpoints. Transport is WebRTC, so
        this is the token endpoint.

        Server-side because the alternative is handing the browser an API key
        that opens the whole ElevenLabs account. What the browser gets expires
        on its own and authorizes one conversation.
        """
        speech = getattr(self.model, "elevenlabs", None)
        if speech is None or not speech.available:
            raise ModelProviderError("Voice needs WAQIL_ELEVENLABS_API_KEY")
        client = await speech._client()
        try:
            response = await client.get(
                "/v1/convai/conversation/token",
                params={"agent_id": self.settings.elevenlabs_agent_id.strip()},
            )
        except Exception as exc:  # noqa: BLE001 - network errors become model errors
            raise ModelProviderError(
                f"Could not reach ElevenLabs: {str(exc)[:200]}"
            ) from exc
        if response.status_code >= 400:
            raise ModelProviderError(
                f"ElevenLabs refused the conversation: HTTP {response.status_code}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise ModelProviderError("ElevenLabs returned a non-JSON reply") from exc
        token = str((payload or {}).get("token") or "").strip()
        if not token:
            raise ModelProviderError("ElevenLabs returned no conversation token")
        return token

    # -- internals --------------------------------------------------------

    def _require(self, session_id: str) -> _Session:
        session = self._sessions.get(session_id)
        if session is None:
            raise VoiceSessionExpired("that voice session is not open")
        return session

    def _bind(
        self, provider_conversation_id: str, metis_session_id: str = ""
    ) -> _Session:
        conversation = (provider_conversation_id or "").strip()
        explicit = (metis_session_id or "").strip()
        if explicit:
            session = self._sessions.get(explicit)
            if session is None or not session.live:
                raise VoiceSessionExpired("that Metis voice session is not open")
            if (
                session.provider_conversation_id
                and session.provider_conversation_id != conversation
            ):
                raise VoiceSessionExpired(
                    "that provider conversation belongs to another session"
                )
            session.provider_conversation_id = conversation
            return session
        for session in self._sessions.values():
            if session.live and session.provider_conversation_id == conversation:
                return session
        candidate = next(
            (
                session
                for session in sorted(
                    self._sessions.values(), key=lambda item: item.started_at
                )
                if session.live and not session.provider_conversation_id
            ),
            None,
        )
        if candidate is None:
            raise VoiceSessionExpired("no voice session is open for that conversation")
        candidate.provider_conversation_id = conversation
        return candidate
