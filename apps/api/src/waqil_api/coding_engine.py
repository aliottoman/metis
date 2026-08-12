"""Local, versioned client for a replaceable project coding engine.

The ClineCore SDK runs in a separate Node process.  This module owns the small
NDJSON boundary between that process and the Python application: it never uses
a shell, never replays an interrupted mutation, rejects oversized or malformed
frames, correlates concurrent calls, and exposes only bounded progress events.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, TypeVar, cast

from pydantic import BaseModel, ValidationError

from .coding_contracts import (
    AbortSliceV1,
    ContinueSliceV1,
    CodingEngineInfoV1,
    CodingEventV1,
    CodingCleanupFailureV1,
    CodingCleanupHoldV1,
    CodingCleanupPlanV1,
    CodingProviderV1,
    CodingSessionCreateV1,
    CodingSessionUpdateV1,
    CodingSessionV1,
    CodingUsageV1,
    DeleteSessionResultV1,
    HOST_CHECKS,
    HostCallRequestV1,
    HostCheck,
    HostCheckResultV1,
    RpcEventNotificationV1,
    RpcMethod,
    RpcRequestV1,
    RpcResponseV1,
    RestartWithModelV1,
    RestoreSliceV1,
    SessionReferenceV1,
    ShutdownResultV1,
    SliceResultV1,
    StartSliceV1,
    SubscribeV1,
    SubscribeResultV1,
)

if TYPE_CHECKING:
    from .database import Database


DEFAULT_MAX_FRAME_BYTES = 1024 * 1024
_SAFE_ENVIRONMENT_KEYS = frozenset(
    {"HOME", "LANG", "LC_ALL", "PATH", "TEMP", "TMP", "TMPDIR"}
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_ -]?key|access[_ -]?token|password|passwd|secret)\b"
    r"(\s*[:=]\s*)([^\s,;]+)"
)
_KNOWN_TOKEN = re.compile(
    r"(?i)\b(?:bearer\s+|sk-|gh[pousr]_|xox[abprs]-)[A-Za-z0-9._-]{12,}"
)
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SAFE_ERROR_DATA = frozenset(
    {"modelId", "providerId", "retryAfterSeconds", "sessionId", "state"}
)

# How long one drained frame may take to arrive once the journal's end cursor
# is already known. It bounds a dead sidecar, not the trace: the drain stops
# when it reaches the stated cursor, so a healthy session never waits here.
_EVENT_DRAIN_FRAME_SECONDS = 10.0


# Answers one run_check request: (session_id, check) -> result.
CheckHandler: TypeAlias = Callable[[str, "HostCheck"], Awaitable["HostCheckResultV1"]]


class CodingEngineError(RuntimeError):
    """Base error for the local coding-engine boundary."""


class CodingEngineUnavailable(CodingEngineError):
    """The sidecar exited or could not be started."""


class CodingEngineTimeout(CodingEngineError):
    """A request crossed its wall-clock deadline."""


class CodingEngineProtocolError(CodingEngineError):
    """The sidecar emitted a frame that violates the pinned protocol."""


class CodingEngineRemoteError(CodingEngineError):
    """A typed failure returned by the sidecar."""

    def __init__(
        self, code: str, message: str, data: Mapping[str, Any] | None = None
    ) -> None:
        super().__init__(f"{code}: {_safe_message(message, 1_000)}")
        self.code = code
        self.message = _safe_message(message, 1_000)
        self.data = _safe_error_data(data)


class CodingEngine(Protocol):
    async def get_info(self) -> CodingEngineInfoV1: ...

    async def start_slice(self, request: StartSliceV1) -> SliceResultV1: ...

    async def continue_slice(self, request: ContinueSliceV1) -> SliceResultV1: ...

    async def abort_slice(
        self, session_id: str, *, reason: str | None = None
    ) -> SliceResultV1: ...

    async def restore_slice(
        self, session_id: str, *, provider: CodingProviderV1 | None = None
    ) -> SliceResultV1: ...

    async def restart_with_model(
        self, request: RestartWithModelV1
    ) -> SliceResultV1: ...

    def subscribe(
        self, session_id: str, *, after_cursor: int = 0
    ) -> AsyncIterator[CodingEventV1]: ...

    async def drain_events(
        self, session_id: str, *, after_cursor: int = 0, limit: int = 256
    ) -> list[CodingEventV1]: ...

    async def get_usage(self, session_id: str) -> CodingUsageV1: ...

    async def delete_session(self, session_id: str) -> DeleteSessionResultV1: ...

    async def close(self) -> None: ...


def _safe_message(value: Any, limit: int) -> str:
    text = _CONTROL_CHARACTERS.sub("", str(value or ""))
    text = _SECRET_ASSIGNMENT.sub(r"\1\2[REDACTED]", text)
    text = _KNOWN_TOKEN.sub("[REDACTED]", text)
    return text[:limit]


def _safe_error_data(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if not value:
        return {}
    result: dict[str, Any] = {}
    for key in _SAFE_ERROR_DATA:
        item = value.get(key)
        if isinstance(item, bool | int | float):
            result[key] = item
        elif isinstance(item, str):
            result[key] = _safe_message(item, 240)
    return result


def _safe_relative_path(value: str) -> str:
    cleaned = value.replace("\\", "/").strip()
    path = PurePosixPath(cleaned)
    if (
        not cleaned
        or cleaned.startswith("/")
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\x00" in cleaned
    ):
        return ""
    return str(path)[:512]


def _sanitize_event(event: CodingEventV1) -> CodingEventV1:
    return event.model_copy(
        update={
            "status": _safe_message(event.status, 80),
            "tool": re.sub(r"[^A-Za-z0-9_.:-]", "", event.tool)[:80],
            "path": _safe_relative_path(event.path),
            "message": _safe_message(event.message, 1_000),
        }
    )


class _StreamClosed:
    def __init__(self, error: CodingEngineError) -> None:
        self.error = error


_QueueItem = CodingEventV1 | _StreamClosed
ResultT = TypeVar("ResultT", bound=BaseModel)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


class SidecarCodingEngine:
    """Supervised NDJSON-RPC client for the local ClineCore sidecar.

    A process is started lazily and restarted for the *next* call after a crash.
    An interrupted call is failed rather than replayed: start/continue can edit
    files, so blind retry would violate at-most-once semantics.  The caller can
    reconnect with ``restore_slice`` using its persisted sidecar session ID.
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: Path | None = None,
        environment: Mapping[str, str] | None = None,
        start_timeout_seconds: float = 10.0,
        request_timeout_seconds: float = 900.0,
        shutdown_timeout_seconds: float = 5.0,
        max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
        event_queue_size: int = 256,
    ) -> None:
        if not command or any(not str(part) for part in command):
            raise ValueError(
                "sidecar command must contain at least one non-empty argument"
            )
        if start_timeout_seconds <= 0 or request_timeout_seconds <= 0:
            raise ValueError("sidecar timeouts must be positive")
        if shutdown_timeout_seconds <= 0:
            raise ValueError("shutdown timeout must be positive")
        if not 4_096 <= max_frame_bytes <= 16 * 1024 * 1024:
            raise ValueError("max_frame_bytes must be between 4 KiB and 16 MiB")
        if not 1 <= event_queue_size <= 10_000:
            raise ValueError("event_queue_size must be between 1 and 10000")

        self._command = tuple(str(part) for part in command)
        self._cwd = cwd
        self._environment = self._build_environment(environment)
        self._start_timeout = start_timeout_seconds
        self._request_timeout = request_timeout_seconds
        self._shutdown_timeout = shutdown_timeout_seconds
        self._max_frame_bytes = max_frame_bytes
        self._event_queue_size = event_queue_size

        self._process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._process_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._handshake_lock = asyncio.Lock()
        self._handshaken_process: asyncio.subprocess.Process | None = None
        self._handshake_info: CodingEngineInfoV1 | None = None
        self._pending: dict[str, asyncio.Future[RpcResponseV1]] = {}
        self._expired_ids: deque[str] = deque(maxlen=1_024)
        self._subscribers: defaultdict[str, set[asyncio.Queue[_QueueItem]]] = (
            defaultdict(set)
        )
        self._closed = False
        # Answers a model's run_check request. Installed per turn by the
        # coordinator, which knows the mirror the check must run against.
        self._check_handler: CheckHandler | None = None
        self._host_call_tasks: set[asyncio.Task[None]] = set()

    @staticmethod
    def _build_environment(overrides: Mapping[str, str] | None) -> dict[str, str]:
        environment = {
            key: value
            for key, value in os.environ.items()
            if key in _SAFE_ENVIRONMENT_KEYS
        }
        if overrides:
            for key, value in overrides.items():
                if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,79}", key):
                    raise ValueError(f"invalid sidecar environment key: {key!r}")
                environment[key] = str(value)
        return environment

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def get_info(self) -> CodingEngineInfoV1:
        """Prove the child is the exact pinned engine before granting it work."""

        process = await self._ensure_process()
        return await self._handshake_process(process)

    async def _handshake_process(
        self, process: asyncio.subprocess.Process
    ) -> CodingEngineInfoV1:
        """Bind the pinned protocol proof to one exact child generation."""

        if self._handshaken_process is process and self._handshake_info is not None:
            return self._handshake_info
        async with self._handshake_lock:
            if self._handshaken_process is process and self._handshake_info is not None:
                return self._handshake_info
            try:
                raw = await self._rpc("getInfo", {}, timeout=self._start_timeout)
                info = self._validate_result(CodingEngineInfoV1, raw, "getInfo")
                if process is not self._process or process.returncode is not None:
                    raise CodingEngineUnavailable(
                        "coding-engine child changed during its startup handshake"
                    )
            except CodingEngineError as error:
                await self._break_connection(process, error)
                raise
            self._handshaken_process = process
            self._handshake_info = info
            return info

    async def start_slice(self, request: StartSliceV1) -> SliceResultV1:
        raw = await self._rpc("startSlice", request.wire_dict())
        return self._validate_result(SliceResultV1, raw, "startSlice")

    async def continue_slice(self, request: ContinueSliceV1) -> SliceResultV1:
        raw = await self._rpc("continueSlice", request.wire_dict())
        return self._validate_result(SliceResultV1, raw, "continueSlice")

    async def abort_slice(
        self, session_id: str, *, reason: str | None = None
    ) -> SliceResultV1:
        reference = AbortSliceV1(sessionId=session_id, reason=reason)
        raw = await self._rpc(
            "abortSlice",
            reference.model_dump(mode="json", by_alias=True, exclude_none=True),
        )
        return self._validate_result(SliceResultV1, raw, "abortSlice")

    async def restore_slice(
        self, session_id: str, *, provider: CodingProviderV1 | None = None
    ) -> SliceResultV1:
        reference = RestoreSliceV1(sessionId=session_id, provider=provider)
        raw = await self._rpc("restoreSlice", reference.wire_dict())
        return self._validate_result(SliceResultV1, raw, "restoreSlice")

    async def restart_with_model(self, request: RestartWithModelV1) -> SliceResultV1:
        raw = await self._rpc("restartWithModel", request.wire_dict())
        return self._validate_result(SliceResultV1, raw, "restartWithModel")

    async def get_usage(self, session_id: str) -> CodingUsageV1:
        reference = SessionReferenceV1(sessionId=session_id)
        raw = await self._rpc(
            "getUsage", reference.model_dump(mode="json", by_alias=True)
        )
        return self._validate_result(CodingUsageV1, raw, "getUsage")

    async def delete_session(self, session_id: str) -> DeleteSessionResultV1:
        reference = SessionReferenceV1(sessionId=session_id)
        raw = await self._rpc(
            "deleteSession", reference.model_dump(mode="json", by_alias=True)
        )
        return self._validate_result(DeleteSessionResultV1, raw, "deleteSession")

    async def subscribe(
        self, session_id: str, *, after_cursor: int = 0
    ) -> AsyncIterator[CodingEventV1]:
        if after_cursor < 0:
            raise ValueError("after_cursor must be non-negative")
        reference = SubscribeV1(sessionId=session_id, afterCursor=after_cursor)
        queue: asyncio.Queue[_QueueItem] = asyncio.Queue(maxsize=self._event_queue_size)
        subscribers = self._subscribers[session_id]
        subscribers.add(queue)
        cursor = after_cursor
        try:
            raw = await self._rpc(
                "subscribe", reference.model_dump(mode="json", by_alias=True)
            )
            subscribed = self._validate_result(SubscribeResultV1, raw, "subscribe")
            if subscribed.session_id != session_id:
                raise CodingEngineProtocolError(
                    "subscribe response named a different session"
                )
            while True:
                item = await queue.get()
                if isinstance(item, _StreamClosed):
                    raise item.error
                if item.cursor <= cursor:
                    continue
                cursor = item.cursor
                yield item
        finally:
            subscribers.discard(queue)
            if not subscribers:
                self._subscribers.pop(session_id, None)

    async def drain_events(
        self, session_id: str, *, after_cursor: int = 0, limit: int = 256
    ) -> list[CodingEventV1]:
        """Every journaled event after ``after_cursor``, through the last one.

        The subscribe response states the journal's current end cursor, so the
        drain has an exact target instead of a guess: it reads until that
        cursor arrives, the journal is empty, or ``limit`` is reached. This
        replaced an idle-timeout read whose 150 ms first-event window could
        expire before the sidecar's backlog frames had even been parsed off
        stdout -- and losing that race discarded the whole trace, including
        the terminal event, leaving a failed round with nothing to explain it.

        The per-frame wait stays generous and bounded, so a sidecar that dies
        mid-drain still returns what it already delivered rather than hanging.
        """

        if after_cursor < 0:
            raise ValueError("after_cursor must be non-negative")
        if limit <= 0:
            return []
        reference = SubscribeV1(sessionId=session_id, afterCursor=after_cursor)
        queue: asyncio.Queue[_QueueItem] = asyncio.Queue(maxsize=self._event_queue_size)
        subscribers = self._subscribers[session_id]
        subscribers.add(queue)
        drained: list[CodingEventV1] = []
        cursor = after_cursor
        try:
            raw = await self._rpc(
                "subscribe", reference.model_dump(mode="json", by_alias=True)
            )
            subscribed = self._validate_result(SubscribeResultV1, raw, "subscribe")
            if subscribed.session_id != session_id:
                raise CodingEngineProtocolError(
                    "subscribe response named a different session"
                )
            target = int(subscribed.cursor)
            # Nothing journaled past the caller's cursor: not a failure, and
            # not something to wait on.
            while cursor < target and len(drained) < limit:
                try:
                    item = await asyncio.wait_for(
                        queue.get(), timeout=_EVENT_DRAIN_FRAME_SECONDS
                    )
                except TimeoutError:
                    break
                if isinstance(item, _StreamClosed):
                    break
                if item.cursor <= cursor:
                    continue
                cursor = item.cursor
                drained.append(item)
        finally:
            subscribers.discard(queue)
            if not subscribers:
                self._subscribers.pop(session_id, None)
        return drained

    async def shutdown(self) -> ShutdownResultV1:
        if not self.running:
            return ShutdownResultV1(state="shutting_down")
        raw = await self._rpc("shutdown", {}, timeout=self._shutdown_timeout)
        result = self._validate_result(ShutdownResultV1, raw, "shutdown")
        await self._stop_process()
        return result

    async def close(self) -> None:
        if self._closed:
            return
        try:
            await self.shutdown()
        except CodingEngineError:
            await self._stop_process()
        finally:
            self._closed = True

    async def _ensure_process(self) -> asyncio.subprocess.Process:
        if self._closed:
            raise CodingEngineUnavailable("coding-engine client is closed")
        async with self._process_lock:
            process = self._process
            if process is not None and process.returncode is None:
                return process
            try:
                process = await asyncio.wait_for(
                    asyncio.create_subprocess_exec(
                        *self._command,
                        cwd=str(self._cwd) if self._cwd is not None else None,
                        env=self._environment,
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        limit=self._max_frame_bytes + 1,
                    ),
                    timeout=self._start_timeout,
                )
            except TimeoutError as error:
                raise CodingEngineUnavailable(
                    "coding-engine sidecar did not start"
                ) from error
            except OSError as error:
                raise CodingEngineUnavailable(
                    f"coding-engine sidecar could not start: {_safe_message(error, 240)}"
                ) from error
            self._process = process
            self._reader_task = asyncio.create_task(
                self._read_stdout(process), name="coding-sidecar-stdout"
            )
            self._stderr_task = asyncio.create_task(
                self._drain_stderr(process), name="coding-sidecar-stderr"
            )
            return process

    async def _rpc(
        self,
        method: RpcMethod,
        params: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        process = await self._ensure_process()
        if method not in {"getInfo", "shutdown"}:
            await self._handshake_process(process)
        request_id = f"rpc_{uuid.uuid4().hex}"
        try:
            request = RpcRequestV1(id=request_id, method=method, params=params)
        except ValidationError as error:
            raise CodingEngineProtocolError(f"invalid RPC request: {error}") from error
        try:
            frame = (
                json.dumps(
                    request.model_dump(mode="json", by_alias=True),
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )
        except (TypeError, ValueError) as error:
            raise CodingEngineProtocolError(
                "RPC request is not JSON serializable"
            ) from error
        if len(frame) > self._max_frame_bytes:
            raise CodingEngineProtocolError(
                f"RPC request exceeds {self._max_frame_bytes} byte frame limit"
            )

        loop = asyncio.get_running_loop()
        future: asyncio.Future[RpcResponseV1] = loop.create_future()
        self._pending[request_id] = future
        try:
            async with self._write_lock:
                if process is not self._process or process.returncode is not None:
                    raise CodingEngineUnavailable("coding-engine sidecar disconnected")
                if process.stdin is None:
                    raise CodingEngineUnavailable("coding-engine stdin is unavailable")
                process.stdin.write(frame)
                await asyncio.wait_for(
                    process.stdin.drain(),
                    timeout=min(timeout or self._request_timeout, 30.0),
                )
            response = await asyncio.wait_for(
                asyncio.shield(future), timeout=timeout or self._request_timeout
            )
        except TimeoutError as error:
            self._expired_ids.append(request_id)
            future.cancel()
            await self._break_connection(
                process, CodingEngineTimeout(f"{method} exceeded its deadline")
            )
            raise CodingEngineTimeout(f"{method} exceeded its deadline") from error
        except asyncio.CancelledError:
            # The request may already be mutating the mirror. Stop the process
            # and force an explicit restore; leaving it alive would let work
            # continue after its owning Metis task was cancelled.
            self._expired_ids.append(request_id)
            future.cancel()
            await self._break_connection(
                process, CodingEngineUnavailable(f"{method} was cancelled")
            )
            raise
        except (BrokenPipeError, ConnectionResetError, OSError) as error:
            unavailable = CodingEngineUnavailable(
                f"coding-engine transport failed during {method}"
            )
            await self._break_connection(process, unavailable)
            raise unavailable from error
        finally:
            self._pending.pop(request_id, None)

        if response.error is not None:
            raise CodingEngineRemoteError(
                response.error.code, response.error.message, response.error.data
            )
        if response.result is None:  # model validation already enforces this
            raise CodingEngineProtocolError("RPC response contained no result")
        return response.result

    @staticmethod
    def _validate_result(
        contract: type[ResultT], raw: dict[str, Any], method: str
    ) -> ResultT:
        try:
            return contract.model_validate(raw)
        except ValidationError as error:
            raise CodingEngineProtocolError(
                f"{method} returned an invalid result"
            ) from error

    async def _read_stdout(self, process: asyncio.subprocess.Process) -> None:
        error: CodingEngineError = CodingEngineUnavailable(
            "coding-engine sidecar disconnected"
        )
        try:
            if process.stdout is None:
                raise CodingEngineUnavailable("coding-engine stdout is unavailable")
            while True:
                try:
                    frame = await process.stdout.readline()
                except (ValueError, asyncio.LimitOverrunError) as read_error:
                    raise CodingEngineProtocolError(
                        "coding-engine emitted an oversized frame"
                    ) from read_error
                if not frame:
                    break
                if len(frame) > self._max_frame_bytes or not frame.endswith(b"\n"):
                    raise CodingEngineProtocolError(
                        "coding-engine emitted an oversized or unterminated frame"
                    )
                self._handle_frame(frame[:-1])
        except asyncio.CancelledError:
            return
        except CodingEngineError as caught:
            error = caught
        except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as caught:
            error = CodingEngineProtocolError(
                f"coding-engine emitted an invalid frame: {type(caught).__name__}"
            )
        except Exception as caught:  # noqa: BLE001 - process boundary
            error = CodingEngineUnavailable(
                f"coding-engine reader failed: {_safe_message(caught, 240)}"
            )
        finally:
            await self._break_connection(process, error)

    async def _answer_host_call(self, request: HostCallRequestV1) -> None:
        """Run one host-owned check and write its result back to the sidecar.

        Never raises into the reader: a check that fails, times out, or has no
        handler installed still has to produce a frame, because the sidecar is
        holding a model turn open waiting for one.
        """

        result: HostCheckResultV1 | None = None
        error: str = ""
        try:
            params = dict(request.params or {})
            session_id = str(params.get("sessionId") or "")
            check = str(params.get("check") or "")
            handler = self._check_handler
            if handler is None:
                error = "this Metis run does not offer host checks"
            elif check not in HOST_CHECKS:
                error = f"unknown check {check!r}"
            else:
                result = await handler(session_id, cast(HostCheck, check))
        except asyncio.CancelledError:
            raise
        except Exception as caught:  # noqa: BLE001 - the sidecar must get a frame
            error = _safe_message(caught, 240)
        payload: dict[str, Any] = {"version": "1", "id": request.id}
        if result is not None:
            payload["hostResult"] = result.model_dump(mode="json", by_alias=True)
        else:
            payload["hostError"] = {"message": error or "the check produced no result"}
        try:
            await self._write_frame(payload)
        except CodingEngineError:
            # The pipe died while the check ran. The sidecar's own bridge
            # fails its pending calls when that happens, so the model is not
            # left waiting.
            pass

    async def _write_frame(self, payload: dict[str, Any]) -> None:
        frame = (
            json.dumps(
                payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            ).encode("utf-8")
            + b"\n"
        )
        if len(frame) > self._max_frame_bytes:
            raise CodingEngineProtocolError("host frame exceeds the frame limit")
        process = self._process
        async with self._write_lock:
            if process is None or process.returncode is not None:
                raise CodingEngineUnavailable("coding-engine sidecar disconnected")
            if process.stdin is None:
                raise CodingEngineUnavailable("coding-engine stdin is unavailable")
            process.stdin.write(frame)
            await process.stdin.drain()

    def set_check_handler(self, handler: CheckHandler | None) -> None:
        """Install what answers a model's run_check request, or remove it."""

        self._check_handler = handler

    def _handle_frame(self, frame: bytes) -> None:
        decoded = json.loads(
            frame.decode("utf-8"), parse_constant=_reject_json_constant
        )
        if not isinstance(decoded, dict):
            raise CodingEngineProtocolError("coding-engine frame must be a JSON object")
        if decoded.get("method") == "hostCall":
            # The one call that travels sidecar -> host. Answered on a task so
            # the reader loop keeps draining frames while a check runs; a
            # check that blocked this loop would stall every event behind it.
            request = HostCallRequestV1.model_validate(decoded)
            task = asyncio.create_task(
                self._answer_host_call(request), name="coding-sidecar-host-call"
            )
            self._host_call_tasks.add(task)
            task.add_done_callback(self._host_call_tasks.discard)
            return
        if decoded.get("method") == "event":
            notification = RpcEventNotificationV1.model_validate(decoded)
            event = _sanitize_event(notification.params)
            subscribers = tuple(self._subscribers.get(event.session_id, ()))
            for queue in subscribers:
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull as caught:
                    raise CodingEngineProtocolError(
                        "coding-engine event subscriber fell behind"
                    ) from caught
            return
        response = RpcResponseV1.model_validate(decoded)
        future = self._pending.get(response.id)
        if future is None:
            if response.id in self._expired_ids:
                return
            raise CodingEngineProtocolError("coding-engine returned an unknown RPC id")
        if future.done():
            raise CodingEngineProtocolError(
                "coding-engine returned a duplicate RPC response"
            )
        future.set_result(response)

    async def _drain_stderr(self, process: asyncio.subprocess.Process) -> None:
        """Drain without retaining output, which may contain source or credentials."""
        try:
            if process.stderr is None:
                return
            while await process.stderr.read(64 * 1024):
                pass
        except asyncio.CancelledError:
            return

    async def _break_connection(
        self,
        process: asyncio.subprocess.Process,
        error: CodingEngineError,
        *,
        terminate: bool = True,
    ) -> None:
        reader_task: asyncio.Task[None] | None
        stderr_task: asyncio.Task[None] | None
        async with self._process_lock:
            if process is not self._process:
                return
            self._process = None
            self._handshaken_process = None
            self._handshake_info = None
            reader_task = self._reader_task
            stderr_task = self._stderr_task
            if terminate and process.returncode is None:
                process.terminate()
            for future in tuple(self._pending.values()):
                if not future.done():
                    future.set_exception(error)
            for queues in tuple(self._subscribers.values()):
                for queue in tuple(queues):
                    try:
                        queue.put_nowait(_StreamClosed(error))
                    except asyncio.QueueFull:
                        # Preserve the fail-closed signal even for a slow consumer.
                        queue.get_nowait()
                        queue.put_nowait(_StreamClosed(error))
        if process.returncode is None:
            try:
                await asyncio.wait_for(process.wait(), self._shutdown_timeout)
            except TimeoutError:
                process.kill()
                await process.wait()
        tasks = [
            task
            for task in (reader_task, stderr_task)
            if task is not None and task is not asyncio.current_task()
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._reader_task is reader_task:
            self._reader_task = None
        if self._stderr_task is stderr_task:
            self._stderr_task = None

    async def _stop_process(self) -> None:
        async with self._process_lock:
            process, self._process = self._process, None
            self._handshaken_process = None
            self._handshake_info = None
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
        if process.returncode is None:
            try:
                await asyncio.wait_for(process.wait(), self._shutdown_timeout)
            except TimeoutError:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), self._shutdown_timeout)
                except TimeoutError:
                    process.kill()
                    await process.wait()
        for task in (self._reader_task, self._stderr_task):
            if task is not None and task is not asyncio.current_task():
                task.cancel()
        tasks = [
            task
            for task in (self._reader_task, self._stderr_task)
            if task is not None and task is not asyncio.current_task()
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._reader_task = None
        self._stderr_task = None


class CodingSessionStore:
    """Narrow persistence facade used by future engine integrations."""

    def __init__(self, database: Database) -> None:
        self._database = database

    async def create(self, value: CodingSessionCreateV1) -> CodingSessionV1:
        return await self._database.create_coding_session(value)

    async def get(self, session_id: str) -> CodingSessionV1 | None:
        return await self._database.get_coding_session(session_id)

    async def for_run(self, run_id: str) -> list[CodingSessionV1]:
        return await self._database.list_coding_sessions(run_id=run_id)

    async def resumable(self) -> list[CodingSessionV1]:
        return await self._database.list_coding_sessions(resumable_only=True)

    async def begin_cleanup(
        self,
        session_id: str,
        value: CodingCleanupPlanV1,
        *,
        lease_until: datetime,
    ) -> CodingSessionV1:
        return await self._database.begin_coding_cleanup(
            session_id,
            value,
            lease_until=lease_until,
        )

    async def record_ancestry(
        self, session_id: str, sidecar_ids: Sequence[str]
    ) -> None:
        """Durably own a sidecar identity the moment it is minted."""

        await self._database.record_coding_sidecar_ancestry(session_id, sidecar_ids)

    async def sidecar_identities(self) -> set[str]:
        """Every identity a durable session still answers for."""

        return await self._database.list_coding_sidecar_identities()

    async def fail_cleanup(
        self,
        session_id: str,
        value: CodingCleanupFailureV1,
    ) -> CodingSessionV1:
        return await self._database.fail_coding_cleanup(session_id, value)

    async def complete_cleanup(
        self,
        session_id: str,
        *,
        released_at: datetime,
    ) -> CodingSessionV1:
        return await self._database.complete_coding_cleanup(
            session_id,
            released_at=released_at,
        )

    async def hold_cleanup(
        self,
        session_id: str,
        value: CodingCleanupHoldV1,
    ) -> CodingSessionV1:
        return await self._database.hold_coding_cleanup(session_id, value)

    async def cleanup_candidates(
        self,
        *,
        now: datetime,
        limit: int = 100,
    ) -> list[CodingSessionV1]:
        return await self._database.list_coding_cleanup_candidates(
            now=now,
            limit=limit,
        )

    async def workspace_paths(self) -> set[Path]:
        return await self._database.list_coding_workspace_paths()

    async def update(
        self, session_id: str, value: CodingSessionUpdateV1
    ) -> CodingSessionV1:
        return await self._database.update_coding_session(session_id, value)
