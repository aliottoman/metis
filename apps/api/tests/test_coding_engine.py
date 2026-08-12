from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from waqil_api.coding_contracts import (
    MAX_ALLOWED_PATHS,
    CodingModelRouteV1,
    CodingProviderV1,
    CodingSessionCreateV1,
    CodingSessionState,
    CodingSessionUpdateV1,
    CodingWorkspaceFileV1,
    CodingWorkspaceSnapshotV1,
    ContinueSliceV1,
    RestartWithModelV1,
    SliceResultV1,
    StartSliceV1,
)
from waqil_api.coding_engine import (
    CodingEngineProtocolError,
    CodingEngineRemoteError,
    CodingEngineUnavailable,
    CodingSessionStore,
    SidecarCodingEngine,
)
from waqil_api.config import Settings
from waqil_api.database import Database
from waqil_api.runtime import AppRuntime


FAKE_SIDECAR = r"""
import json
import sys

def emit(value):
    sys.stdout.write(json.dumps(value, separators=(",", ":")) + "\n")
    sys.stdout.flush()

usage = {"inputTokens": 8, "outputTokens": 5, "totalTokens": 13, "requests": 1}
handshaken = False
for line in sys.stdin:
    request = json.loads(line)
    request_id = request["id"]
    method = request["method"]
    params = request["params"]
    if method == "getInfo":
        handshaken = True
        emit({
            "version": "1", "id": request_id,
            "result": {
                "protocolVersion": "1", "engine": "clinecore", "runtime": "fake",
                "sdkVersion": "0.0.72", "policyVersion": "1",
                "allowedTools": ["editor", "read_files", "run_check", "search_codebase"],
            },
        })
        continue
    if method != "shutdown":
        # This process-local assertion resets after every supervised restart,
        # proving no new child generation can receive work before getInfo.
        assert handshaken, "coding RPC received before this child was handshaken"
    if method == "startSlice":
        if params["prompt"] == "remote-error":
            emit({
                "version": "1",
                "id": request_id,
                "error": {
                    "code": "RUNTIME_ERROR",
                    "message": "api_key=super-secret-value",
                    "data": {"retryAfterSeconds": 3, "rawProviderBody": "do-not-expose"},
                },
            })
            continue
        assert params["provider"]["apiKey"] == "ephemeral-key"
        assert params["provider"]["baseUrl"] == "http://127.0.0.1:11434/v1"
        session_id = params.get("sessionId") or "session_one"
        emit({
            "version": "1",
            "id": request_id,
            "result": {
                "sessionId": session_id,
                "state": "idle",
                "summary": "slice complete",
                "usage": usage,
                "model": {"providerId": "ollama", "modelId": "model-a"},
            },
        })
    elif method == "continueSlice":
        if params["prompt"] == "crash-now":
            sys.exit(17)
        if "provider" in params:
            assert params["provider"]["apiKey"] == "ephemeral-key"
        emit({
            "version": "1", "id": request_id,
            "result": {"sessionId": params["sessionId"], "state": "idle", "usage": usage},
        })
    elif method == "restartWithModel":
        assert params["provider"]["apiKey"] == "ephemeral-key"
        emit({
            "version": "1", "id": request_id,
            "result": {
                "sessionId": params.get("newSessionId") or "session_two",
                "parentSessionId": params["sessionId"],
                "state": "idle", "usage": usage,
                "model": {"providerId": params["provider"]["providerId"],
                          "modelId": params["provider"]["modelId"]},
            },
        })
    elif method in {"abortSlice", "restoreSlice"}:
        if method == "restoreSlice" and "provider" in params:
            assert params["provider"]["apiKey"] == "ephemeral-key"
        state = "aborted" if method == "abortSlice" else "idle"
        emit({
            "version": "1", "id": request_id,
            "result": {"sessionId": params["sessionId"], "state": state, "usage": usage},
        })
    elif method == "getUsage":
        emit({"version": "1", "id": request_id, "result": usage})
    elif method == "subscribe":
        cursor = params.get("afterCursor", 0) + 1
        emit({
            "version": "1", "id": request_id,
            "result": {
                "sessionId": params["sessionId"], "cursor": cursor - 1,
                "oldestCursor": cursor - 1, "replayed": 0, "overflowed": False,
            },
        })
        emit({
            "version": "1", "method": "event",
            "params": {
                "sessionId": params["sessionId"], "cursor": cursor,
                "type": "tool.finished", "status": "ok", "tool": "read_file!",
                "reasonCode": "outside_workspace",
                "path": "../../private.env", "message": "api_key=super-secret-value",
                "occurredAt": "2026-08-11T00:00:00Z",
            },
        })
    elif method == "deleteSession":
        emit({
            "version": "1", "id": request_id,
            "result": {"sessionId": params["sessionId"], "deleted": True},
        })
    elif method == "shutdown":
        emit({
            "version": "1", "id": request_id,
            "result": {"state": "shutting_down"},
        })
        break
"""

REVERSE_SIDECAR = r"""
import json
import sys

pending = []
handshaken = False
for line in sys.stdin:
    request = json.loads(line)
    if request["method"] == "getInfo":
        handshaken = True
        print(json.dumps({
            "version": "1", "id": request["id"],
            "result": {
                "protocolVersion": "1", "engine": "clinecore", "runtime": "fake",
                "sdkVersion": "0.0.72", "policyVersion": "1",
                "allowedTools": ["editor", "read_files", "run_check", "search_codebase"],
            },
        }), flush=True)
    elif request["method"] == "startSlice":
        assert handshaken, "coding RPC received before this child was handshaken"
        pending.append(request)
        if len(pending) == 2:
            for item in reversed(pending):
                prompt = item["params"]["prompt"]
                result = {
                    "sessionId": "session_" + prompt,
                    "state": "idle",
                    "summary": prompt,
                    "usage": {
                        "inputTokens": 0, "outputTokens": 0,
                        "totalTokens": 0, "requests": 0,
                    },
                }
                print(json.dumps({"version": "1", "id": item["id"], "result": result}), flush=True)
            pending.clear()
    elif request["method"] == "shutdown":
        print(json.dumps({
            "version": "1", "id": request["id"],
            "result": {"state": "shutting_down"},
        }), flush=True)
        break
"""

OVERSIZED_RESPONSE_SIDECAR = r"""
import json
import sys

handshake = json.loads(sys.stdin.readline())
assert handshake["method"] == "getInfo"
print(json.dumps({
    "version": "1", "id": handshake["id"],
    "result": {
        "protocolVersion": "1", "engine": "clinecore", "runtime": "fake",
        "sdkVersion": "0.0.72", "policyVersion": "1",
        "allowedTools": ["editor", "read_files", "run_check", "search_codebase"],
    },
}), flush=True)
request = json.loads(sys.stdin.readline())
assert request["method"] == "startSlice"
result = {
    "sessionId": "session_large", "state": "idle", "summary": "x" * 5000,
    "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0, "requests": 0},
}
print(json.dumps({"version": "1", "id": request["id"], "result": result}), flush=True)
"""


def _provider() -> CodingProviderV1:
    return CodingProviderV1(
        providerId="ollama",
        modelId="model-a",
        apiKey="ephemeral-key",
        baseUrl="http://127.0.0.1:11434/v1",
    )


def _start(tmp_path: Path, prompt: str = "Build the vertical slice") -> StartSliceV1:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return StartSliceV1(
        prompt=prompt,
        workspaceRoot=workspace.resolve(),
        provider=_provider(),
    )


def test_slice_result_preserves_structured_runtime_finish_reason() -> None:
    result = SliceResultV1.model_validate(
        {
            "sessionId": "session_budget",
            "state": "failed",
            "finishReason": "error",
            "controlledStopReason": "max_iterations",
            "iterations": 24,
            "toolCallCount": 17,
        }
    )

    assert result.finish_reason == "error"
    assert result.controlled_stop_reason == "max_iterations"
    assert result.iterations == 24
    assert result.tool_call_count == 17
    with pytest.raises(ValidationError):
        SliceResultV1.model_validate(
            {
                "sessionId": "session_unknown",
                "state": "failed",
                "finishReason": "provider_quota",
            }
        )
    with pytest.raises(ValidationError):
        SliceResultV1.model_validate(
            {
                "sessionId": "session_unknown_controlled_stop",
                "state": "failed",
                "finishReason": "error",
                "controlledStopReason": "provider_quota",
            }
        )


def test_start_slice_allowed_paths_is_bounded_and_wire_optional(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    provider = _provider()

    # Absent means the sidecar applies no slice-scoped write restriction of
    # its own; the field is omitted from the wire payload entirely.
    unrestricted = StartSliceV1(
        prompt="Build it", workspaceRoot=workspace, provider=provider
    )
    assert unrestricted.allowed_paths is None
    assert "allowedPaths" not in unrestricted.wire_dict()

    # A present, canonicalized allowlist round-trips onto the wire exactly.
    scoped = StartSliceV1(
        prompt="Build it",
        workspaceRoot=workspace,
        provider=provider,
        allowedPaths=["app/planned.py", "./app/planned_test.py"],
    )
    assert scoped.allowed_paths == ["app/planned.py", "app/planned_test.py"]
    assert scoped.wire_dict()["allowedPaths"] == [
        "app/planned.py",
        "app/planned_test.py",
    ]

    # A present-but-empty allowlist is a meaningful read-only round, distinct
    # from omitting the field, and must survive onto the wire as `[]`.
    read_only = StartSliceV1(
        prompt="Build it", workspaceRoot=workspace, provider=provider, allowedPaths=[]
    )
    assert read_only.wire_dict()["allowedPaths"] == []

    # A whole synthesized slice is eight files, and the wire format has to
    # carry one: the bound below is what makes an unsliced Logivity manifest
    # buildable in a single round rather than rejected at the transport.
    assert MAX_ALLOWED_PATHS == 8
    at_bound = StartSliceV1(
        prompt="Build it",
        workspaceRoot=workspace,
        provider=provider,
        allowedPaths=[f"app/file_{index}.py" for index in range(MAX_ALLOWED_PATHS)],
    )
    assert len(at_bound.wire_dict()["allowedPaths"]) == MAX_ALLOWED_PATHS
    with pytest.raises(ValidationError):
        StartSliceV1(
            prompt="Build it",
            workspaceRoot=workspace,
            provider=provider,
            allowedPaths=[
                f"app/file_{index}.py" for index in range(MAX_ALLOWED_PATHS + 1)
            ],
        )
    with pytest.raises(ValidationError):
        StartSliceV1(
            prompt="Build it",
            workspaceRoot=workspace,
            provider=provider,
            allowedPaths=["../outside.py"],
        )
    with pytest.raises(ValidationError):
        StartSliceV1(
            prompt="Build it",
            workspaceRoot=workspace,
            provider=provider,
            allowedPaths=["/etc/passwd"],
        )


def test_restart_with_model_carries_allowed_paths_onto_the_wire(tmp_path: Path) -> None:
    request = RestartWithModelV1(
        sessionId="parent-session",
        prompt="Repair the failing slice",
        provider=_provider(),
        allowedPaths=["app/planned.py"],
    )
    assert request.allowed_paths == ["app/planned.py"]
    assert request.wire_dict()["allowedPaths"] == ["app/planned.py"]


@pytest.mark.asyncio
async def test_sidecar_round_trip_events_model_restart_and_secret_boundary(
    tmp_path,
) -> None:
    engine = SidecarCodingEngine(
        [sys.executable, "-u", "-c", FAKE_SIDECAR],
        request_timeout_seconds=2,
    )
    try:
        started = await engine.start_slice(_start(tmp_path))
        assert started.session_id == "session_one"
        assert started.model == CodingModelRouteV1(
            providerId="ollama", modelId="model-a"
        )

        stream = engine.subscribe("session_one", after_cursor=7)
        event = await anext(stream)
        await stream.aclose()
        assert event.cursor == 8
        assert event.path == ""
        assert event.tool == "read_file"
        assert event.reason_code == "outside_workspace"
        assert event.message == "api_key=[REDACTED]"

        continued = await engine.continue_slice(
            ContinueSliceV1(
                sessionId="session_one",
                prompt="Run the focused test",
                provider=_provider(),
            )
        )
        assert continued.state == "idle"
        assert (await engine.get_usage("session_one")).total_tokens == 13

        restarted = await engine.restart_with_model(
            RestartWithModelV1(
                sessionId="session_one",
                newSessionId="session_two",
                prompt="Repair the unchanged failure",
                provider=CodingProviderV1(
                    providerId="ollama",
                    modelId="model-b",
                    apiKey="ephemeral-key",
                    baseUrl="http://127.0.0.1:11434/v1",
                ),
            )
        )
        assert restarted.session_id == "session_two"
        assert restarted.parent_session_id == "session_one"
        assert restarted.model is not None
        assert restarted.model.model_id == "model-b"
        assert (
            await engine.abort_slice("session_two", reason="decision complete")
        ).state == "aborted"
        assert (await engine.delete_session("session_two")).deleted is True
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_sidecar_remote_errors_are_bounded_and_process_reconnects(
    tmp_path,
) -> None:
    engine = SidecarCodingEngine(
        [sys.executable, "-u", "-c", FAKE_SIDECAR],
        request_timeout_seconds=2,
    )
    try:
        with pytest.raises(CodingEngineRemoteError) as captured:
            await engine.start_slice(_start(tmp_path, "remote-error"))
        assert captured.value.message == "api_key=[REDACTED]"
        assert captured.value.data == {"retryAfterSeconds": 3}

        await engine.start_slice(_start(tmp_path))
        with pytest.raises(CodingEngineUnavailable):
            await engine.continue_slice(
                ContinueSliceV1(sessionId="session_one", prompt="crash-now")
            )
        # A mutation is never replayed. The next explicit restore safely starts
        # a new process and reconnects to the caller's persisted session ID.
        restored = await engine.restore_slice("session_one", provider=_provider())
        assert restored.state == "idle"
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_sidecar_refuses_oversized_outbound_frames(tmp_path) -> None:
    engine = SidecarCodingEngine(
        [sys.executable, "-u", "-c", FAKE_SIDECAR],
        request_timeout_seconds=2,
        max_frame_bytes=4_096,
    )
    try:
        with pytest.raises(CodingEngineProtocolError, match="frame limit"):
            await engine.start_slice(_start(tmp_path, "x" * 5_000))
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_sidecar_correlates_concurrent_responses_arriving_in_reverse(
    tmp_path,
) -> None:
    engine = SidecarCodingEngine(
        [sys.executable, "-u", "-c", REVERSE_SIDECAR],
        request_timeout_seconds=2,
    )
    try:
        first, second = await asyncio.gather(
            engine.start_slice(_start(tmp_path, "first")),
            engine.start_slice(_start(tmp_path, "second")),
        )
        assert (first.summary, second.summary) == ("first", "second")
        assert (first.session_id, second.session_id) == (
            "session_first",
            "session_second",
        )
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_sidecar_refuses_oversized_inbound_frames(tmp_path) -> None:
    engine = SidecarCodingEngine(
        [sys.executable, "-u", "-c", OVERSIZED_RESPONSE_SIDECAR],
        request_timeout_seconds=2,
        max_frame_bytes=4_096,
    )
    try:
        with pytest.raises(CodingEngineProtocolError, match="oversized"):
            await engine.start_slice(_start(tmp_path))
    finally:
        await engine.close()


def _tree_digest(files: list[CodingWorkspaceFileV1]) -> str:
    digest = hashlib.sha256()
    for item in sorted(files, key=lambda candidate: candidate.path):
        for value in (
            item.path,
            str(item.bytes),
            item.sha256,
            item.source,
            item.disk_sha256,
        ):
            digest.update(value.encode())
            digest.update(b"\x00")
    return digest.hexdigest()


def _snapshot(tmp_path: Path, *, overlay: str = "b" * 64) -> CodingWorkspaceSnapshotV1:
    content_digest = hashlib.sha256(b"hello\n").hexdigest()
    files = [
        CodingWorkspaceFileV1(
            path="app.py",
            sha256=content_digest,
            bytes=6,
            source="disk",
            disk_sha256=content_digest,
        )
    ]
    source = (tmp_path / "source").resolve()
    workspace = (tmp_path / "workspace").resolve()
    source.mkdir(exist_ok=True)
    workspace.mkdir(exist_ok=True)
    return CodingWorkspaceSnapshotV1(
        schema_version="1",
        mirror_id="ewm_test",
        asset_id="asset_test",
        source_root=source,
        project_root=workspace,
        tree_sha256=_tree_digest(files),
        overlay_sha256=overlay,
        files=files,
        excluded_count=1,
        excluded_paths=[".git/"],
        created_at=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_coding_session_store_persists_recovery_authority(tmp_path) -> None:
    database = Database(tmp_path / "waqil.db")
    await database.open()
    try:
        conversation = await database.create_conversation("Coding engine")
        message = await database.add_message(conversation.id, "user", "Build it")
        run = await database.create_run(
            conversation.id, message.id, graph_schema_version="1", model_aliases={}
        )
        snapshot = _snapshot(tmp_path)
        store = CodingSessionStore(database)
        created = await store.create(
            CodingSessionCreateV1(
                run_id=run.id,
                conversation_id=conversation.id,
                project_id="asset_test",
                workspace_path=snapshot.project_root,
                baseline_digest=snapshot.tree_sha256,
                overlay_digest=snapshot.overlay_sha256,
                workspace_snapshot=snapshot,
                model_route=CodingModelRouteV1(
                    providerId="ollama", modelId="deepseek-v4-pro:cloud"
                ),
            )
        )
        assert created.event_cursor == 0
        assert created.workspace_snapshot.files[0].path == "app.py"

        running = await store.update(
            created.id,
            CodingSessionUpdateV1(
                sidecar_session_id="cline_session_one",
                event_cursor=4,
                state=CodingSessionState.RUNNING,
            ),
        )
        assert running.started_at is not None
        assert [item.id for item in await store.resumable()] == [created.id]

        rebased_snapshot = _snapshot(tmp_path, overlay="c" * 64)
        with pytest.raises(ValueError, match="event_cursor=0"):
            await store.update(
                created.id,
                CodingSessionUpdateV1(
                    sidecar_session_id="cline_session_two",
                    event_cursor=5,
                    model_route=CodingModelRouteV1(
                        providerId="ollama", modelId="kimi-k2.7-code:cloud"
                    ),
                ),
            )
        rebased = await store.update(
            created.id,
            CodingSessionUpdateV1(
                baseline_digest=rebased_snapshot.tree_sha256,
                overlay_digest=rebased_snapshot.overlay_sha256,
                workspace_snapshot=rebased_snapshot,
                event_cursor=0,
                sidecar_session_id="cline_session_two",
                model_route=CodingModelRouteV1(
                    providerId="ollama", modelId="kimi-k2.7-code:cloud"
                ),
            ),
        )
        assert rebased.overlay_digest == "c" * 64
        assert rebased.sidecar_session_id == "cline_session_two"
        assert rebased.model_route.model_id == "kimi-k2.7-code:cloud"
        assert rebased.event_cursor == 0

        await store.update(created.id, CodingSessionUpdateV1(event_cursor=5))

        with pytest.raises(ValueError, match="cannot move backwards"):
            await store.update(created.id, CodingSessionUpdateV1(event_cursor=3))
        with pytest.raises(ValidationError, match="update atomically"):
            CodingSessionUpdateV1(overlay_digest="d" * 64)

        failed = await store.update(
            created.id,
            CodingSessionUpdateV1(
                state=CodingSessionState.FAILED,
                last_error="provider unavailable",
            ),
        )
        assert failed.finished_at is not None
        assert await store.resumable() == []
        assert (await store.for_run(run.id))[0].id == created.id

        raw = (
            database._connection()
            .execute(  # noqa: SLF001 - persistence assertion
                "SELECT workspace_snapshot_json, model_route_json FROM coding_sessions"
            )
            .fetchone()
        )
        assert "ephemeral-key" not in json.dumps(dict(raw))
    finally:
        await database.close()


def test_workspace_snapshot_rejects_tampered_manifest(tmp_path) -> None:
    snapshot = _snapshot(tmp_path)
    with pytest.raises(ValidationError, match="tree_sha256"):
        CodingWorkspaceSnapshotV1.model_validate(
            snapshot.model_dump(mode="json") | {"tree_sha256": "0" * 64}
        )


@pytest.mark.asyncio
async def test_runtime_keeps_lazy_sidecar_for_legacy_cleanup(tmp_path) -> None:
    common = {
        "_env_file": None,
        "data_dir": tmp_path / "data",
        "repo_root": Path(__file__).resolve().parents[3],
        "model_backend": "deterministic",
        "reference_runner_mode": "deterministic",
        "allow_test_backends": True,
    }
    legacy = AppRuntime(Settings(**common))
    assert isinstance(legacy.coding_engine, SidecarCodingEngine)
    # Construction is process-free; legacy uses this only when durable cleanup
    # debt from a prior ClineCore run must be released.
    assert legacy.coding_engine._process is None  # noqa: SLF001

    settings = Settings(**common, project_coding_engine="clinecore")
    runtime = AppRuntime(settings)
    try:
        assert isinstance(runtime.coding_engine, SidecarCodingEngine)
        assert settings.cline_sidecar_command[:2] == (
            "node",
            str(
                settings.repo_root
                / "apps"
                / "cline-sidecar"
                / "dist"
                / "src"
                / "index.js"
            ),
        )
        assert settings.cline_sidecar_command[-2:] == (
            "--max-replay-events",
            "255",
        )
        assert (
            Settings(**common, cline_sidecar_event_queue_size=3).cline_sidecar_command[
                -1
            ]
            == "2"
        )
        with pytest.raises(ValidationError):
            Settings(**common, cline_sidecar_event_queue_size=2)
        settings.prepare_directories()
        assert settings.cline_sidecar_data_dir.is_dir()
        assert settings.coding_workspace_dir.is_dir()
    finally:
        await runtime.close()
        await legacy.close()
