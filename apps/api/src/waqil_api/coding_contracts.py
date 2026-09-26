"""Versioned contracts for Metis's local coding-engine boundary.

The coding engine is deliberately a replaceable implementation detail.  These
contracts are owned by Metis, not by ClineCore, so SDK types never leak into the
database, control plane, or web API.  They also keep credentials out of durable
records: ``CodingProviderV1`` is used only for a live request, while
``CodingModelRouteV1`` is the secret-free value that may be persisted.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
import hashlib
from pathlib import Path
import re
from typing import Any, Literal, TypeAlias
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)


PROTOCOL_VERSION = "1"
CLINE_SDK_VERSION = "0.0.86"
CODING_TOOL_POLICY_VERSION = "1"
# Both parties must name the same tools or the handshake fails. `run_check` is
# a real custom tool the sidecar registers through the SDK's `extraTools`, with
# its own closed `{check: enum}` schema; the SDK's `run_commands` shell is
# disabled and never advertised. Metis runs the named check itself in its
# pinned networkless verifier, and nothing executes in the sidecar either way.
CODING_ALLOWED_TOOLS = frozenset(
    {"read_files", "search_codebase", "editor", "run_check"}
)
MAX_PROMPT_CHARACTERS = 400_000
MAX_PROMPT_BYTES = 256 * 1024
MAX_SYSTEM_PROMPT_BYTES = 64 * 1024
MAX_WORKSPACE_SNAPSHOT_BYTES = 4 * 1024 * 1024
MAX_SAFE_INTEGER = 9_007_199_254_740_991
# Mirrors project_slices.MAX_SLICE_FILES. Duplicated rather than imported: this
# module is imported by project_coding_engine, which project_slices is not, so
# importing it here would risk a cycle for one shared bound.
MAX_ALLOWED_PATHS = 8
# Files a session may read but never change. A broad-scope session cannot be
# expressed as an 8-entry allowlist, so the direct path supplies a deny-list
# instead. Mirrors protocol.ts MAX_PROTECTED_PATHS.
MAX_PROTECTED_PATHS = 256
SHA256_PATTERN = r"^[a-f0-9]{64}$"
IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$"
EVENT_TYPE_PATTERN = r"^[a-z][a-z0-9_.-]{0,79}$"
ToolDenialReasonCode: TypeAlias = Literal[
    "tool_not_allowed",
    "invalid_search_scope",
    "missing_path",
    "invalid_path",
    "outside_workspace",
    "protected_path",
    "symlink_path",
    "outside_slice",
    "stuck_inspection_loop",
]
RpcMethod: TypeAlias = Literal[
    "startSlice",
    "continueSlice",
    "abortSlice",
    "restoreSlice",
    "subscribe",
    "getUsage",
    "restartWithModel",
    "deleteSession",
    "getInfo",
    "shutdown",
]
CodingFinishReason: TypeAlias = Literal[
    "completed",
    "error",
    "aborted",
    "max_iterations",
    "mistake_limit",
]
ControlledStopReason: TypeAlias = Literal["max_iterations"]


class CodingContract(BaseModel):
    """Strict base model used on both sides of the process boundary."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        serialize_by_alias=True,
    )


class CodingSessionState(StrEnum):
    STARTING = "starting"
    RUNNING = "running"
    IDLE = "idle"
    RESTORING = "restoring"
    ABORTING = "aborting"
    ABORTED = "aborted"
    COMPLETED = "completed"
    FAILED = "failed"


class CodingCleanupStatus(StrEnum):
    """Durable ownership state for host and sidecar session artifacts."""

    ACTIVE = "active"
    PENDING = "pending"
    RETRY = "retry"
    HELD = "held"
    CLEAN = "clean"


TERMINAL_CODING_STATES = frozenset(
    {
        CodingSessionState.ABORTED,
        CodingSessionState.COMPLETED,
        CodingSessionState.FAILED,
    }
)


def _sanitized_error(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", value)
    cleaned = re.sub(
        r"(?i)\b(api[_ -]?key|access[_ -]?token|password|passwd|secret)\b"
        r"(\s*[:=]\s*)([^\s,;]+)",
        r"\1\2[REDACTED]",
        cleaned,
    )
    cleaned = re.sub(
        r"(?i)\b(?:bearer\s+|sk-|gh[pousr]_|xox[abprs]-)[A-Za-z0-9._-]{12,}",
        "[REDACTED]",
        cleaned,
    )
    return cleaned[:2_000]


class CodingModelRouteV1(CodingContract):
    provider_id: str = Field(alias="providerId", pattern=IDENTIFIER_PATTERN)
    model_id: str = Field(alias="modelId", min_length=1, max_length=240)

    @field_validator("model_id")
    @classmethod
    def clean_model_id(cls, value: str) -> str:
        value = value.strip()
        if (
            not value
            or len(value.encode("utf-8")) > 256
            or any(character in value for character in "\r\n\x00")
        ):
            raise ValueError("model_id must be a non-empty single line")
        return value


class CodingProviderV1(CodingModelRouteV1):
    """Ephemeral provider details. The API key must never be persisted."""

    api_key: SecretStr | None = Field(
        default=None, alias="apiKey", max_length=16 * 1024
    )
    base_url: str | None = Field(default=None, alias="baseUrl", max_length=2_048)
    headers: dict[str, SecretStr] = Field(default_factory=dict, max_length=32)

    @field_validator("api_key")
    @classmethod
    def bounded_api_key(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None:
            secret = value.get_secret_value()
            if (
                not secret
                or "\x00" in secret
                or len(secret.encode("utf-8")) > 16 * 1024
            ):
                raise ValueError("api_key must be 1 through 16 KiB without NUL bytes")
        return value

    @field_validator("base_url")
    @classmethod
    def valid_base_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if len(value.encode("utf-8")) > 2_048 or "\x00" in value:
            raise ValueError("base_url exceeds the protocol limit")
        parsed = urlparse(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "base_url must be an HTTP(S) origin/path without credentials, query, or fragment"
            )
        return value.rstrip("/")

    @field_validator("headers")
    @classmethod
    def valid_headers(cls, value: dict[str, SecretStr]) -> dict[str, SecretStr]:
        for name, wrapped in value.items():
            secret = wrapped.get_secret_value()
            if not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", name):
                raise ValueError("provider header names must be valid HTTP tokens")
            if (
                not secret
                or "\x00" in secret
                or len(secret.encode("utf-8")) > 16 * 1024
            ):
                raise ValueError("provider header values must be 1 through 16 KiB")
        return value

    def wire_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "providerId": self.provider_id,
            "modelId": self.model_id,
        }
        if self.api_key is not None:
            result["apiKey"] = self.api_key.get_secret_value()
        if self.base_url is not None:
            result["baseUrl"] = self.base_url
        if self.headers:
            result["headers"] = {
                name: value.get_secret_value() for name, value in self.headers.items()
            }
        return result


class SliceLimitsV1(CodingContract):
    max_iterations: int = Field(default=24, alias="maxIterations", ge=1, le=200)
    timeout_ms: int = Field(default=900_000, alias="timeoutMs", ge=1_000, le=3_600_000)
    max_tokens_per_turn: int | None = Field(
        default=None, alias="maxTokensPerTurn", ge=256, le=262_144
    )


class StartSliceV1(CodingContract):
    session_id: str | None = Field(
        default=None, alias="sessionId", pattern=IDENTIFIER_PATTERN
    )
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARACTERS)
    system_prompt: str | None = Field(
        default=None, alias="systemPrompt", max_length=MAX_SYSTEM_PROMPT_BYTES
    )
    workspace_root: Path = Field(alias="workspaceRoot")
    provider: CodingProviderV1
    limits: SliceLimitsV1 = Field(default_factory=SliceLimitsV1)
    # The exact files this slice may create or edit. Absent means the sidecar
    # applies no slice-scoped write restriction of its own; Metis's independent
    # host-side diff check remains mandatory either way. Present-and-empty
    # means a read-only round: no editor write is permitted at all.
    allowed_paths: list[str] | None = Field(
        default=None, alias="allowedPaths", max_length=MAX_ALLOWED_PATHS
    )
    protected_paths: list[str] | None = Field(
        default=None, alias="protectedPaths", max_length=MAX_PROTECTED_PATHS
    )
    # The `cline_direct` path: one session owns the whole authorized scope, so
    # the sliced path's fixed inspection-count stop does not apply to it. Sent
    # explicitly rather than inferred from an absent allowlist, because those
    # two facts happen to coincide today and would drift apart silently.
    broad_scope: bool | None = Field(default=None, alias="broadScope")

    @field_validator("allowed_paths")
    @classmethod
    def safe_allowed_paths(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        return [_relative_path(item) for item in value]

    @field_validator("prompt")
    @classmethod
    def non_blank_prompt(cls, value: str) -> str:
        if not value.strip() or "\x00" in value:
            raise ValueError("prompt must not be blank")
        if len(value.encode("utf-8")) > MAX_PROMPT_BYTES:
            raise ValueError("prompt exceeds the 256 KiB protocol limit")
        return value

    @field_validator("system_prompt")
    @classmethod
    def bounded_system_prompt(cls, value: str | None) -> str | None:
        if value is not None and (
            not value
            or "\x00" in value
            or len(value.encode("utf-8")) > MAX_SYSTEM_PROMPT_BYTES
        ):
            raise ValueError("system_prompt must be 1 through 64 KiB without NUL bytes")
        return value

    @field_validator("workspace_root")
    @classmethod
    def absolute_workspace(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("workspace_root must be absolute")
        resolved = value.resolve()
        if (
            resolved == Path(resolved.anchor)
            or len(str(resolved).encode("utf-8")) > 4_096
        ):
            raise ValueError("workspace_root must be a bounded non-root path")
        return resolved

    def wire_dict(self) -> dict[str, Any]:
        result = self.model_dump(
            mode="json", by_alias=True, exclude={"provider"}, exclude_none=True
        )
        result["provider"] = self.provider.wire_dict()
        return result


class ContinueSliceV1(CodingContract):
    session_id: str = Field(alias="sessionId", pattern=IDENTIFIER_PATTERN)
    recovery_session_id: str | None = Field(
        default=None, alias="recoverySessionId", pattern=IDENTIFIER_PATTERN
    )
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARACTERS)
    timeout_ms: int | None = Field(
        default=None, alias="timeoutMs", ge=1_000, le=3_600_000
    )
    provider: CodingProviderV1 | None = None

    @field_validator("prompt")
    @classmethod
    def non_blank_prompt(cls, value: str) -> str:
        if not value.strip() or "\x00" in value:
            raise ValueError("prompt must not be blank")
        if len(value.encode("utf-8")) > MAX_PROMPT_BYTES:
            raise ValueError("prompt exceeds the 256 KiB protocol limit")
        return value

    def wire_dict(self) -> dict[str, Any]:
        result = self.model_dump(
            mode="json", by_alias=True, exclude={"provider"}, exclude_none=True
        )
        if self.provider is not None:
            result["provider"] = self.provider.wire_dict()
        return result


class SessionReferenceV1(CodingContract):
    session_id: str = Field(alias="sessionId", pattern=IDENTIFIER_PATTERN)


class RestoreSliceV1(SessionReferenceV1):
    provider: CodingProviderV1 | None = None

    def wire_dict(self) -> dict[str, Any]:
        result = self.model_dump(
            mode="json", by_alias=True, exclude={"provider"}, exclude_none=True
        )
        if self.provider is not None:
            result["provider"] = self.provider.wire_dict()
        return result


class AbortSliceV1(SessionReferenceV1):
    reason: str | None = Field(default=None, max_length=2_048)

    @field_validator("reason")
    @classmethod
    def valid_reason(cls, value: str | None) -> str | None:
        if value is not None and (
            not value or len(value.encode("utf-8")) > 2_048 or "\x00" in value
        ):
            raise ValueError("reason must be 1 through 2048 UTF-8 bytes")
        return value


class SubscribeV1(SessionReferenceV1):
    after_cursor: int = Field(default=0, alias="afterCursor", ge=0, le=MAX_SAFE_INTEGER)


class RestartWithModelV1(CodingContract):
    session_id: str = Field(alias="sessionId", pattern=IDENTIFIER_PATTERN)
    new_session_id: str | None = Field(
        default=None, alias="newSessionId", pattern=IDENTIFIER_PATTERN
    )
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARACTERS)
    provider: CodingProviderV1
    limits: SliceLimitsV1 | None = None
    # Carried forward from the parent slice: a repair fork must never widen
    # its write scope beyond the failing slice it was forked from.
    allowed_paths: list[str] | None = Field(
        default=None, alias="allowedPaths", max_length=MAX_ALLOWED_PATHS
    )

    @field_validator("allowed_paths")
    @classmethod
    def safe_allowed_paths(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        return [_relative_path(item) for item in value]

    @field_validator("prompt")
    @classmethod
    def non_blank_prompt(cls, value: str) -> str:
        if not value.strip() or "\x00" in value:
            raise ValueError("prompt must not be blank")
        if len(value.encode("utf-8")) > MAX_PROMPT_BYTES:
            raise ValueError("prompt exceeds the 256 KiB protocol limit")
        return value

    def wire_dict(self) -> dict[str, Any]:
        result = self.model_dump(
            mode="json", by_alias=True, exclude={"provider"}, exclude_none=True
        )
        result["provider"] = self.provider.wire_dict()
        return result


class CodingUsageV1(CodingContract):
    input_tokens: int = Field(default=0, alias="inputTokens", ge=0, le=MAX_SAFE_INTEGER)
    output_tokens: int = Field(
        default=0, alias="outputTokens", ge=0, le=MAX_SAFE_INTEGER
    )
    total_tokens: int = Field(default=0, alias="totalTokens", ge=0, le=MAX_SAFE_INTEGER)
    requests: int = Field(default=0, ge=0, le=MAX_SAFE_INTEGER)
    cost_usd: float | None = Field(default=None, alias="costUsd", ge=0)
    # Prompt-cache counters. None means the provider did not report them,
    # which is not the same as zero: the difference is what says whether a
    # large input count was re-sent context or a cache read.
    cache_read_tokens: int | None = Field(
        default=None, alias="cacheReadTokens", ge=0, le=MAX_SAFE_INTEGER
    )
    cache_write_tokens: int | None = Field(
        default=None, alias="cacheWriteTokens", ge=0, le=MAX_SAFE_INTEGER
    )

    @model_validator(mode="after")
    def total_is_not_smaller_than_parts(self) -> CodingUsageV1:
        if self.total_tokens < self.input_tokens + self.output_tokens:
            raise ValueError("total_tokens cannot be smaller than its token parts")
        return self


class SliceResultV1(CodingContract):
    session_id: str = Field(alias="sessionId", pattern=IDENTIFIER_PATTERN)
    state: Literal["starting", "running", "idle", "completed", "aborted", "failed"]
    finish_reason: CodingFinishReason | None = Field(default=None, alias="finishReason")
    controlled_stop_reason: ControlledStopReason | None = Field(
        default=None, alias="controlledStopReason"
    )
    iterations: int = Field(default=0, ge=0, le=MAX_SAFE_INTEGER)
    tool_call_count: int = Field(
        default=0, alias="toolCallCount", ge=0, le=MAX_SAFE_INTEGER
    )
    summary: str = Field(default="", max_length=4_100)
    usage: CodingUsageV1 = Field(default_factory=CodingUsageV1)
    model: CodingModelRouteV1 | None = None
    parent_session_id: str | None = Field(
        default=None, alias="parentSessionId", pattern=IDENTIFIER_PATTERN
    )


class SubscribeResultV1(CodingContract):
    session_id: str = Field(alias="sessionId", pattern=IDENTIFIER_PATTERN)
    cursor: int = Field(ge=0, le=MAX_SAFE_INTEGER)
    oldest_cursor: int = Field(alias="oldestCursor", ge=0, le=MAX_SAFE_INTEGER)
    replayed: int = Field(ge=0, le=MAX_SAFE_INTEGER)
    overflowed: bool

    @model_validator(mode="after")
    def valid_cursor_window(self) -> SubscribeResultV1:
        if self.cursor == 0 and self.oldest_cursor != 0:
            raise ValueError("an empty event stream must have oldest_cursor=0")
        if self.cursor > 0 and not 1 <= self.oldest_cursor <= self.cursor:
            raise ValueError("oldest_cursor must fall within the retained event window")
        return self


class DeleteSessionResultV1(CodingContract):
    session_id: str = Field(alias="sessionId", pattern=IDENTIFIER_PATTERN)
    deleted: bool
    deleted_session_ids: list[str] = Field(
        default_factory=list, alias="deletedSessionIds", max_length=64
    )

    @field_validator("deleted_session_ids")
    @classmethod
    def valid_deleted_session_ids(cls, value: list[str]) -> list[str]:
        if any(not re.fullmatch(IDENTIFIER_PATTERN, item) for item in value):
            raise ValueError("deleted session identities are invalid")
        if len(value) != len(set(value)):
            raise ValueError("deleted session identities must be unique")
        return value


class ShutdownResultV1(CodingContract):
    state: Literal["shutting_down"]


class CodingEngineInfoV1(CodingContract):
    protocol_version: Literal["1"] = Field(alias="protocolVersion")
    engine: Literal["clinecore"]
    runtime: Literal["cline", "fake"]
    sdk_version: Literal["0.0.86"] = Field(alias="sdkVersion")
    policy_version: Literal["1"] = Field(alias="policyVersion")
    allowed_tools: list[str] = Field(alias="allowedTools", min_length=1, max_length=16)

    @field_validator("allowed_tools")
    @classmethod
    def exact_tool_policy(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)) or frozenset(value) != CODING_ALLOWED_TOOLS:
            raise ValueError("coding engine allowed-tools policy does not match Metis")
        return value


class CodingEventV1(CodingContract):
    """The intentionally small, safe event shape exposed outside the sidecar.

    Source text, tool arguments, command output, and provider payloads are not
    represented.  A UI can show progress from these fields without gaining a
    second route to credentials or unbounded model-authored content.
    """

    session_id: str = Field(alias="sessionId", pattern=IDENTIFIER_PATTERN)
    cursor: int = Field(ge=1, le=MAX_SAFE_INTEGER)
    type: str = Field(pattern=EVENT_TYPE_PATTERN)
    status: str = Field(default="", max_length=4_100)
    tool: str = Field(default="", max_length=4_100)
    reason_code: ToolDenialReasonCode | None = Field(default=None, alias="reasonCode")
    path: str = Field(default="", max_length=4_100)
    message: str = Field(default="", max_length=4_100)
    usage: CodingUsageV1 | None = None
    # Whether `usage` counts this event alone or the session so far. Cline
    # reports session totals, so summing snapshots would multiply the bill.
    usage_scope: Literal["cumulative", "incremental"] | None = Field(
        default=None, alias="usageScope"
    )
    # Coarse SDK failure label (an error name or code), never free text.
    failure_class: str = Field(default="", alias="failureClass", max_length=120)
    # 1-based agent iteration, when the sidecar could observe one.
    iteration: int | None = Field(default=None, ge=0, le=MAX_SAFE_INTEGER)
    occurred_at: datetime = Field(alias="occurredAt")

    @field_validator("occurred_at")
    @classmethod
    def timestamp_has_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must include a timezone")
        return value


class RpcErrorV1(CodingContract):
    code: str = Field(pattern=r"^[A-Z][A-Z0-9_]{0,63}$")
    message: str = Field(min_length=1, max_length=1_000)
    data: dict[str, Any] | None = None


class RpcRequestV1(CodingContract):
    version: Literal["1"] = "1"
    id: str = Field(pattern=IDENTIFIER_PATTERN)
    method: RpcMethod
    params: dict[str, Any] = Field(default_factory=dict)


class RpcResponseV1(CodingContract):
    version: Literal["1"] = "1"
    id: str = Field(pattern=IDENTIFIER_PATTERN)
    result: dict[str, Any] | None = None
    error: RpcErrorV1 | None = None

    @model_validator(mode="after")
    def exactly_one_outcome(self) -> RpcResponseV1:
        if (self.result is None) == (self.error is None):
            raise ValueError("RPC response must contain exactly one of result or error")
        return self


class RpcEventNotificationV1(CodingContract):
    version: Literal["1"] = "1"
    method: Literal["event"]
    params: CodingEventV1


# The complete set of checks a coding session may ask Metis to run. The model
# selects a NAME; Metis owns the argv and runs it in the pinned networkless
# verifier. Mirrors protocol.ts HOST_CHECKS.
HostCheck: TypeAlias = Literal["imports", "pytest", "ruff", "acceptance", "full"]
HOST_CHECKS: tuple[HostCheck, ...] = (
    "imports",
    "pytest",
    "ruff",
    "acceptance",
    "full",
)


class HostCheckFindingV1(CodingContract):
    path: str = Field(default="", max_length=1_000)
    severity: Literal["error", "warning"] = "error"
    detail: str = Field(default="", max_length=2_000)


class HostCheckResultV1(CodingContract):
    check: HostCheck
    ok: bool = False
    # Why the check could not run at all, as opposed to running and failing.
    unavailable: str | None = Field(default=None, max_length=500)
    errors: int = Field(default=0, ge=0, le=MAX_SAFE_INTEGER)
    warnings: int = Field(default=0, ge=0, le=MAX_SAFE_INTEGER)
    findings: list[HostCheckFindingV1] = Field(default_factory=list, max_length=50)
    duration_ms: int | None = Field(default=None, alias="durationMs", ge=0)
    truncated: bool = False


class HostCallRequestV1(CodingContract):
    """A request travelling sidecar -> host on the same stdio pipe."""

    version: Literal["1"] = "1"
    method: Literal["hostCall"]
    id: str = Field(pattern=IDENTIFIER_PATTERN)
    params: dict[str, Any] = Field(default_factory=dict)


class HostCallResponseV1(CodingContract):
    version: Literal["1"] = "1"
    id: str = Field(pattern=IDENTIFIER_PATTERN)
    host_result: HostCheckResultV1 | None = Field(default=None, alias="hostResult")
    host_error: dict[str, str] | None = Field(default=None, alias="hostError")


def _relative_path(value: str, *, directory_allowed: bool = False) -> str:
    normalized = value.replace("\\", "/")
    if directory_allowed:
        normalized = normalized.rstrip("/")
    path = Path(normalized)
    if (
        not normalized
        or normalized.startswith("/")
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\x00" in normalized
    ):
        raise ValueError("workspace manifest paths must be safe relative paths")
    result = path.as_posix()
    return result + "/" if directory_allowed and value.endswith(("/", "\\")) else result


class CodingWorkspaceFileV1(CodingContract):
    path: str = Field(min_length=1, max_length=1_024)
    sha256: str = Field(pattern=SHA256_PATTERN)
    bytes: int = Field(ge=0, le=128_000_000)
    source: Literal["disk", "staged"]
    disk_sha256: str = Field(default="", max_length=64)

    @field_validator("path")
    @classmethod
    def safe_path(cls, value: str) -> str:
        return _relative_path(value)

    @field_validator("disk_sha256")
    @classmethod
    def valid_optional_digest(cls, value: str) -> str:
        if value and not re.fullmatch(SHA256_PATTERN, value):
            raise ValueError("disk_sha256 must be empty or a SHA-256 digest")
        return value


class CodingWorkspaceSnapshotV1(CodingContract):
    schema_version: Literal["1"] = "1"
    mirror_id: str = Field(pattern=IDENTIFIER_PATTERN)
    asset_id: str = Field(pattern=IDENTIFIER_PATTERN)
    source_root: Path
    project_root: Path
    tree_sha256: str = Field(pattern=SHA256_PATTERN)
    overlay_sha256: str = Field(pattern=SHA256_PATTERN)
    files: list[CodingWorkspaceFileV1] = Field(max_length=50_000)
    excluded_count: int = Field(ge=0)
    excluded_paths: list[str] = Field(default_factory=list, max_length=256)
    created_at: datetime

    @field_validator("source_root", "project_root")
    @classmethod
    def absolute_roots(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("workspace snapshot roots must be absolute")
        resolved = value.resolve()
        if (
            resolved == Path(resolved.anchor)
            or len(str(resolved).encode("utf-8")) > 4_096
        ):
            raise ValueError("workspace snapshot roots must be bounded non-root paths")
        return resolved

    @field_validator("excluded_paths")
    @classmethod
    def safe_excluded_paths(cls, values: list[str]) -> list[str]:
        return [_relative_path(value, directory_allowed=True) for value in values]

    @model_validator(mode="after")
    def unique_bounded_manifest(self) -> CodingWorkspaceSnapshotV1:
        paths = [item.path for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("workspace snapshot contains duplicate file paths")
        digest = hashlib.sha256()
        for item in sorted(self.files, key=lambda candidate: candidate.path):
            for value in (
                item.path,
                str(item.bytes),
                item.sha256,
                item.source,
                item.disk_sha256,
            ):
                digest.update(value.encode("utf-8"))
                digest.update(b"\x00")
        if self.tree_sha256 != digest.hexdigest():
            raise ValueError("tree_sha256 does not match workspace file manifest")
        if self.excluded_count < len(self.excluded_paths):
            raise ValueError("excluded_count cannot be smaller than excluded_paths")
        if (
            len(self.model_dump_json(by_alias=True).encode("utf-8"))
            > MAX_WORKSPACE_SNAPSHOT_BYTES
        ):
            raise ValueError("workspace snapshot exceeds the 4 MiB persistence limit")
        return self


class CodingSessionCreateV1(CodingContract):
    run_id: str = Field(pattern=IDENTIFIER_PATTERN)
    conversation_id: str = Field(pattern=IDENTIFIER_PATTERN)
    project_id: str = Field(pattern=IDENTIFIER_PATTERN)
    workspace_path: Path
    baseline_digest: str = Field(pattern=SHA256_PATTERN)
    overlay_digest: str = Field(pattern=SHA256_PATTERN)
    workspace_snapshot: CodingWorkspaceSnapshotV1
    model_route: CodingModelRouteV1
    state: CodingSessionState = CodingSessionState.STARTING

    @field_validator("workspace_path")
    @classmethod
    def workspace_is_absolute(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("workspace_path must be absolute")
        resolved = value.resolve()
        if resolved == Path(resolved.anchor):
            raise ValueError("workspace_path must not be a filesystem root")
        return resolved

    @model_validator(mode="after")
    def snapshot_matches_session(self) -> CodingSessionCreateV1:
        snapshot = self.workspace_snapshot
        if self.workspace_path != snapshot.project_root:
            raise ValueError("workspace_path does not match workspace snapshot")
        if self.baseline_digest != snapshot.tree_sha256:
            raise ValueError("baseline_digest does not match workspace snapshot")
        if self.overlay_digest != snapshot.overlay_sha256:
            raise ValueError("overlay_digest does not match workspace snapshot")
        if self.project_id != snapshot.asset_id:
            raise ValueError("project_id does not match workspace snapshot")
        return self


class CodingSessionUpdateV1(CodingContract):
    sidecar_session_id: str | None = Field(default=None, pattern=IDENTIFIER_PATTERN)
    event_cursor: int | None = Field(default=None, ge=0, le=MAX_SAFE_INTEGER)
    baseline_digest: str | None = Field(default=None, pattern=SHA256_PATTERN)
    overlay_digest: str | None = Field(default=None, pattern=SHA256_PATTERN)
    workspace_snapshot: CodingWorkspaceSnapshotV1 | None = None
    model_route: CodingModelRouteV1 | None = None
    state: CodingSessionState | None = None
    last_error: str | None = Field(default=None, max_length=2_000)

    @field_validator("last_error")
    @classmethod
    def sanitize_last_error(cls, value: str | None) -> str | None:
        return _sanitized_error(value)

    @model_validator(mode="after")
    def atomic_snapshot_rebase(self) -> CodingSessionUpdateV1:
        rebase_fields = {
            "baseline_digest",
            "overlay_digest",
            "workspace_snapshot",
        }
        supplied = rebase_fields & self.model_fields_set
        if supplied and supplied != rebase_fields:
            raise ValueError(
                "baseline_digest, overlay_digest, and workspace_snapshot update atomically"
            )
        if supplied:
            assert self.workspace_snapshot is not None
            if self.baseline_digest != self.workspace_snapshot.tree_sha256:
                raise ValueError("baseline_digest does not match workspace snapshot")
            if self.overlay_digest != self.workspace_snapshot.overlay_sha256:
                raise ValueError("overlay_digest does not match workspace snapshot")
        return self


class CodingCleanupPlanV1(CodingContract):
    target_state: CodingSessionState
    sidecar_session_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=64,
    )

    @field_validator("target_state")
    @classmethod
    def terminal_target(cls, value: CodingSessionState) -> CodingSessionState:
        if value not in TERMINAL_CODING_STATES:
            raise ValueError("cleanup target_state must be terminal")
        return value

    @field_validator("sidecar_session_ids")
    @classmethod
    def valid_sidecar_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for session_id in value:
            if not re.fullmatch(IDENTIFIER_PATTERN, session_id):
                raise ValueError("cleanup sidecar session id is invalid")
        if len(set(value)) != len(value):
            raise ValueError("cleanup sidecar session ids must be unique")
        return value


class CodingCleanupFailureV1(CodingContract):
    error: str = Field(min_length=1)
    next_attempt_at: datetime

    @field_validator("error")
    @classmethod
    def sanitize_error(cls, value: str) -> str:
        cleaned = _sanitized_error(value)
        if not cleaned:
            raise ValueError("cleanup error must not be empty")
        return cleaned

    @field_validator("next_attempt_at")
    @classmethod
    def timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("next_attempt_at must be timezone-aware")
        return value.astimezone(UTC)


class CodingCleanupHoldV1(CodingContract):
    target_state: CodingSessionState
    reason: str = Field(min_length=1)

    @field_validator("target_state")
    @classmethod
    def terminal_target(cls, value: CodingSessionState) -> CodingSessionState:
        if value not in TERMINAL_CODING_STATES:
            raise ValueError("cleanup target_state must be terminal")
        return value

    @field_validator("reason")
    @classmethod
    def sanitize_reason(cls, value: str) -> str:
        cleaned = _sanitized_error(value)
        if not cleaned:
            raise ValueError("cleanup hold reason must not be empty")
        return cleaned


class CodingSessionV1(CodingContract):
    id: str = Field(pattern=IDENTIFIER_PATTERN)
    run_id: str = Field(pattern=IDENTIFIER_PATTERN)
    conversation_id: str = Field(pattern=IDENTIFIER_PATTERN)
    project_id: str = Field(pattern=IDENTIFIER_PATTERN)
    protocol_version: Literal["1"] = "1"
    sidecar_session_id: str | None = Field(default=None, pattern=IDENTIFIER_PATTERN)
    event_cursor: int = Field(ge=0, le=MAX_SAFE_INTEGER)
    workspace_path: Path
    baseline_digest: str = Field(pattern=SHA256_PATTERN)
    overlay_digest: str = Field(pattern=SHA256_PATTERN)
    workspace_snapshot: CodingWorkspaceSnapshotV1
    model_route: CodingModelRouteV1
    state: CodingSessionState
    last_error: str | None = None
    cleanup_status: CodingCleanupStatus = CodingCleanupStatus.ACTIVE
    cleanup_target_state: CodingSessionState | None = None
    cleanup_attempts: int = Field(default=0, ge=0)
    cleanup_next_attempt_at: datetime | None = None
    cleanup_last_error: str | None = None
    cleanup_sidecar_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=64)
    # Every sidecar identity this session has ever owned, committed when each
    # is minted. Distinct from cleanup_sidecar_ids, which records what one
    # cleanup attempt is releasing and must stay empty while cleanup is
    # active; ancestry is legitimate at any point in a session's life.
    sidecar_ancestry: tuple[str, ...] = Field(default_factory=tuple, max_length=64)
    released_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @field_validator("cleanup_sidecar_ids")
    @classmethod
    def valid_cleanup_sidecar_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for session_id in value:
            if not re.fullmatch(IDENTIFIER_PATTERN, session_id):
                raise ValueError("cleanup sidecar session id is invalid")
        return value

    @field_validator("cleanup_last_error")
    @classmethod
    def sanitized_cleanup_error(cls, value: str | None) -> str | None:
        return _sanitized_error(value)

    @field_validator("cleanup_next_attempt_at", "released_at")
    @classmethod
    def aware_cleanup_times(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("cleanup timestamps must be timezone-aware")
        return value.astimezone(UTC) if value is not None else None

    @model_validator(mode="after")
    def consistent_cleanup_state(self) -> CodingSessionV1:
        if len(set(self.cleanup_sidecar_ids)) != len(self.cleanup_sidecar_ids):
            raise ValueError("cleanup sidecar session ids must be unique")
        if self.cleanup_target_state is not None and (
            self.cleanup_target_state not in TERMINAL_CODING_STATES
        ):
            raise ValueError("cleanup target state must be terminal")
        if self.cleanup_status is CodingCleanupStatus.ACTIVE:
            if any(
                (
                    self.cleanup_target_state is not None,
                    self.cleanup_attempts,
                    self.cleanup_next_attempt_at is not None,
                    self.cleanup_last_error is not None,
                    bool(self.cleanup_sidecar_ids),
                    self.released_at is not None,
                )
            ):
                raise ValueError("active cleanup cannot contain cleanup progress")
            return self
        if self.cleanup_target_state is None:
            raise ValueError("non-active cleanup requires a target state")
        if self.cleanup_status is CodingCleanupStatus.HELD:
            if (
                not self.cleanup_last_error
                or self.cleanup_next_attempt_at is not None
                or self.cleanup_attempts != 0
                or bool(self.cleanup_sidecar_ids)
                or self.released_at is not None
            ):
                raise ValueError("held cleanup requires only a durable reason")
            return self
        if self.cleanup_attempts < 1:
            raise ValueError("started cleanup requires at least one attempt")
        if self.cleanup_status is CodingCleanupStatus.PENDING:
            if (
                self.cleanup_next_attempt_at is None
                or self.cleanup_last_error is not None
                or self.released_at is not None
            ):
                raise ValueError("pending cleanup requires a lease and no release time")
        elif self.cleanup_status is CodingCleanupStatus.RETRY:
            if (
                self.cleanup_next_attempt_at is None
                or not self.cleanup_last_error
                or self.released_at is not None
            ):
                raise ValueError("retry cleanup requires a retry time and error")
        elif self.cleanup_status is CodingCleanupStatus.CLEAN:
            if (
                self.released_at is None
                or self.cleanup_next_attempt_at is not None
                or self.cleanup_last_error is not None
                or self.state is not self.cleanup_target_state
            ):
                raise ValueError("clean cleanup must be terminal and fully released")
        return self
