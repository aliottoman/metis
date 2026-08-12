from __future__ import annotations

import asyncio
import stat
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from waqil_api.asset_library import AssetManager
from waqil_api.coding_contracts import (
    CodingEngineInfoV1,
    CodingEventV1,
    CodingModelRouteV1,
    CodingProviderV1,
    CodingSessionState,
    CodingUsageV1,
    ContinueSliceV1,
    DeleteSessionResultV1,
    RestartWithModelV1,
    SliceResultV1,
    StartSliceV1,
)
from waqil_api.coding_engine import CodingEngineRemoteError, CodingSessionStore
from waqil_api.config import Settings
from waqil_api.database import Database
from waqil_api.model_provider import DeterministicModelProvider
from waqil_api.project_coding_engine import (
    ProjectCodingCoordinator,
    ProjectCodingError,
    initial_coding_prompt,
    snapshot_mirror,
)
from waqil_api.project_workspace import (
    ExternalChangeProvenance,
    ProjectWorkspaceService,
)


class RecordingEvents:
    def __init__(self) -> None:
        self.items: list[tuple[str, str, str, dict[str, Any]]] = []
        self.fail_once: set[str] = set()

    async def emit(
        self,
        run_id: str,
        thread_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        if event_type in self.fail_once:
            self.fail_once.remove(event_type)
            raise RuntimeError(f"temporary {event_type} persistence failure")
        self.items.append((run_id, thread_id, event_type, payload))


async def _empty_events() -> AsyncIterator[CodingEventV1]:
    if False:  # pragma: no cover - gives this helper its async-generator shape
        yield CodingEventV1.model_construct()


class RecordingEngine:
    def __init__(self, mutate: Callable[[Path, str], None]) -> None:
        self.mutate = mutate
        self.start_requests: list[StartSliceV1] = []
        self.continue_requests: list[ContinueSliceV1] = []
        self.restart_requests: list[RestartWithModelV1] = []
        self.restore_requests: list[tuple[str, CodingProviderV1 | None]] = []
        self.delete_requests: list[str] = []
        self.delete_failures: set[str] = set()
        self.roots: dict[str, Path] = {}
        self.result_session_id: str | None = None
        self.cancel_start = False
        self.cancel_continue = False
        self.ambiguous_continue = False
        self.result_state: Any = "completed"
        self.finish_reason: Any = None
        self.controlled_stop_reason: Any = None
        self.result_summary = ""
        self.result_iterations = 0
        self.result_tool_call_count = 0
        self.result_usage = CodingUsageV1()

    async def get_info(self) -> CodingEngineInfoV1:
        return CodingEngineInfoV1(
            protocolVersion="1",
            engine="clinecore",
            runtime="fake",
            sdkVersion="0.0.72",
            policyVersion="1",
            allowedTools=["editor", "read_files", "run_check", "search_codebase"],
        )

    async def start_slice(self, request: StartSliceV1) -> SliceResultV1:
        assert request.session_id is not None
        self.start_requests.append(request)
        self.roots[request.session_id] = request.workspace_root
        self.mutate(request.workspace_root, "start")
        if self.cancel_start:
            raise asyncio.CancelledError
        return SliceResultV1(
            sessionId=self.result_session_id or request.session_id,
            state=self.result_state,
            finishReason=self.finish_reason,
            controlledStopReason=self.controlled_stop_reason,
            summary=self.result_summary,
            iterations=self.result_iterations,
            toolCallCount=self.result_tool_call_count,
            usage=self.result_usage,
            model=CodingModelRouteV1(
                providerId=request.provider.provider_id,
                modelId=request.provider.model_id,
            ),
        )

    async def continue_slice(self, request: ContinueSliceV1) -> SliceResultV1:
        self.continue_requests.append(request)
        root = self.roots[request.session_id]
        self.mutate(root, "continue")
        if self.cancel_continue:
            raise asyncio.CancelledError
        if self.ambiguous_continue:
            assert request.recovery_session_id is not None
            # The sidecar created its host-predicted recovery child and edited
            # the mirror, but the transport lost the result.
            self.roots[request.recovery_session_id] = root
            raise CodingEngineRemoteError(
                "ENGINE_UNAVAILABLE", "the continue response was lost"
            )
        return SliceResultV1(sessionId=request.session_id, state="completed")

    async def restart_with_model(self, request: RestartWithModelV1) -> SliceResultV1:
        assert request.new_session_id is not None
        self.restart_requests.append(request)
        root = self.roots[request.session_id]
        self.roots[request.new_session_id] = root
        self.mutate(root, "restart")
        return SliceResultV1(
            sessionId=request.new_session_id,
            parentSessionId=request.session_id,
            state="completed",
            model=CodingModelRouteV1(
                providerId=request.provider.provider_id,
                modelId=request.provider.model_id,
            ),
        )

    async def abort_slice(
        self, session_id: str, *, reason: str | None = None
    ) -> SliceResultV1:
        del reason
        return SliceResultV1(sessionId=session_id, state="aborted")

    async def restore_slice(
        self, session_id: str, *, provider: CodingProviderV1 | None = None
    ) -> SliceResultV1:
        self.restore_requests.append((session_id, provider))
        if session_id not in self.roots:
            raise CodingEngineRemoteError(
                "SESSION_NOT_FOUND", "the requested session does not exist"
            )
        return SliceResultV1(sessionId=session_id, state="idle")

    def subscribe(
        self, session_id: str, *, after_cursor: int = 0
    ) -> AsyncIterator[CodingEventV1]:
        del session_id, after_cursor
        return _empty_events()

    async def drain_events(
        self, session_id: str, *, after_cursor: int = 0, limit: int = 256
    ) -> list[CodingEventV1]:
        del session_id, after_cursor, limit
        return []

    async def get_usage(self, session_id: str) -> CodingUsageV1:
        del session_id
        return CodingUsageV1()

    async def delete_session(self, session_id: str) -> DeleteSessionResultV1:
        self.delete_requests.append(session_id)
        if session_id in self.delete_failures:
            raise CodingEngineRemoteError(
                "ENGINE_UNAVAILABLE", f"could not delete {session_id}"
            )
        return DeleteSessionResultV1(
            sessionId=session_id, deleted=self.roots.pop(session_id, None) is not None
        )

    async def close(self) -> None:
        return None


async def _coordinator(
    tmp_path: Path,
    mutate: Callable[[Path, str], None],
) -> tuple[
    ProjectCodingCoordinator,
    RecordingEngine,
    CodingSessionStore,
    ProjectWorkspaceService,
    Database,
    str,
    str,
    str,
]:
    projects_root = tmp_path / "Projects"
    project = projects_root / "demo"
    (project / "app").mkdir(parents=True)
    (project / "app" / "main.py").write_text("print('original')\n", encoding="utf-8")
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        data_dir=tmp_path / "data",
        repo_root=tmp_path,
        asset_roots=[projects_root],
        model_backend="deterministic",
        allow_test_backends=True,
        max_output_tokens=4_096,
        cline_sidecar_max_iterations=7,
        cline_sidecar_request_timeout_seconds=12.5,
    )
    settings.prepare_directories()
    assets = AssetManager(
        settings.asset_roots,
        approval_path=settings.asset_approval_path,
        catalog_path=settings.asset_catalog_path,
    )
    scanned = await assets.scan()
    assert scanned
    project_id: str = next(iter(scanned)).id
    projects = ProjectWorkspaceService(settings, assets, DeterministicModelProvider())

    database = Database(tmp_path / "sessions.db")
    await database.open()
    conversation = await database.create_conversation("Coding coordinator")
    message = await database.add_message(conversation.id, "user", "Build it")
    run = await database.create_run(
        conversation.id,
        message.id,
        graph_schema_version="1",
        model_aliases={},
    )
    sessions = CodingSessionStore(database)
    engine = RecordingEngine(mutate)
    coordinator = ProjectCodingCoordinator(
        settings, engine, sessions, projects, RecordingEvents()
    )
    return (
        coordinator,
        engine,
        sessions,
        projects,
        database,
        run.id,
        conversation.id,
        project_id,
    )


def _provider(model: str = "model-a") -> CodingProviderV1:
    return CodingProviderV1(
        providerId="openai-compatible",
        modelId=model,
        apiKey=SecretStr("ephemeral-test-key"),
        baseUrl="https://models.example.test/v1",
    )


@pytest.mark.asyncio
async def test_round_uses_protocol_limits_and_reauthenticates_continuation(
    tmp_path: Path,
) -> None:
    def mutate(root: Path, operation: str) -> None:
        content = "VALUE = 1\n" if operation == "start" else "VALUE = 2\n"
        (root / "app" / "planned.py").write_text(content, encoding="utf-8")

    (
        coordinator,
        engine,
        _,
        projects,
        database,
        run_id,
        conversation_id,
        project_id,
    ) = await _coordinator(tmp_path, mutate)
    provider = _provider()
    try:
        first = await coordinator.start(
            run_id=run_id,
            conversation_id=conversation_id,
            project_id=project_id,
            prompt="Implement the planned file",
            staged={},
            provider=provider,
            allowed_paths=["app/planned.py"],
        )
        limits = engine.start_requests[0].limits
        assert limits.max_iterations == 7
        assert limits.timeout_ms == 12_500
        assert limits.max_tokens_per_turn == 4_096
        # The wire request carries the exact slice scope, so the sidecar can
        # enforce its own write allowlist independently of the host's later
        # diff check.
        assert engine.start_requests[0].allowed_paths == ["app/planned.py"]
        assert first.staged["app/planned.py"]["content"] == "VALUE = 1\n"
        assert first.session.state is CodingSessionState.IDLE

        # The settled journal belongs to round 1. It must not suppress the
        # legitimate round-2 repair after round 1 was checkpointed.
        assert (
            await coordinator.recover_for_run(
                run_id,
                project_id=project_id,
                staged=first.staged,
                provider=provider,
                allowed_paths=["app/planned.py"],
                operation_id=f"{run_id}:round:2",
            )
            is None
        )
        assert engine.restore_requests == []

        # A repair round stays confined to the slice's own planned files. The
        # scope is never widened with the accumulated staged overlay: a slice
        # may only rewrite bytes it was explicitly given permission to touch.
        second = await coordinator.continue_session(
            first.session.id,
            prompt="Repair the existing staged file",
            staged=first.staged,
            provider=provider,
            allowed_paths=["app/planned.py"],
            operation_id=f"{run_id}:round:2",
        )
        request = engine.continue_requests[0]
        assert request.timeout_ms == 12_500
        assert request.provider is not None
        assert request.provider.api_key is not None
        assert request.provider.api_key.get_secret_value() == "ephemeral-test-key"
        assert second.staged["app/planned.py"]["content"] == "VALUE = 2\n"
        assert not (tmp_path / "Projects" / "demo" / "app" / "planned.py").exists()
        await projects.discard_external_mirror(
            snapshot_mirror(second.session.workspace_snapshot)
        )
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_cancelled_round_is_recovered_without_repeating_model_mutation(
    tmp_path: Path,
) -> None:
    def mutate(root: Path, operation: str) -> None:
        del operation
        (root / "app" / "planned.py").write_text(
            "PARTIAL_BUT_VALID = True\n", encoding="utf-8"
        )

    (
        coordinator,
        engine,
        sessions,
        _,
        database,
        run_id,
        conversation_id,
        project_id,
    ) = await _coordinator(tmp_path, mutate)
    provider = _provider()
    engine.cancel_start = True
    try:
        with pytest.raises(asyncio.CancelledError):
            await coordinator.start(
                run_id=run_id,
                conversation_id=conversation_id,
                project_id=project_id,
                prompt="Implement app/planned.py",
                staged={},
                provider=provider,
                allowed_paths=["app/planned.py"],
            )

        [interrupted] = await sessions.for_run(run_id)
        assert interrupted.state is CodingSessionState.RUNNING
        assert interrupted in await sessions.resumable()
        mirror_parent = interrupted.workspace_snapshot.project_root.parent
        assert mirror_parent.is_dir()

        recovered = await coordinator.recover_for_run(
            run_id,
            project_id=project_id,
            staged={},
            provider=provider,
            allowed_paths=["app/planned.py"],
            operation_id=f"{run_id}:round:1",
        )
        assert recovered is not None
        assert recovered.recovered is True
        assert recovered.staged["app/planned.py"]["content"] == (
            "PARTIAL_BUT_VALID = True\n"
        )
        assert len(engine.start_requests) == 1
        assert engine.continue_requests == []
        assert len(engine.restore_requests) == 2
        assert engine.restore_requests[-1][0] == interrupted.sidecar_session_id

        terminal = await coordinator.release(
            recovered.session.id, CodingSessionState.COMPLETED
        )
        assert terminal is not None
        assert terminal.state is CodingSessionState.COMPLETED
        persisted = await sessions.get(recovered.session.id)
        assert persisted is not None
        assert persisted.state is CodingSessionState.COMPLETED
        assert not mirror_parent.exists()
        # Both identities this session ever owned are released, not just the
        # one it ended on: the recovery fork's SDK session and event journal
        # would otherwise outlive the run. Ancestry is committed at mint time,
        # so cleanup knows about the child even on an error path.
        assert recovered.session.sidecar_session_id in engine.delete_requests
        assert any(
            item.startswith("coding_recovery_") for item in engine.delete_requests
        )
        assert len(engine.delete_requests) == len(set(engine.delete_requests))
        assert (
            await coordinator.recover_for_run(
                run_id,
                project_id=project_id,
                staged=recovered.staged,
                provider=provider,
                allowed_paths=["app/planned.py"],
                operation_id=f"{run_id}:round:1",
            )
            is None
        )
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_initial_journal_recovers_uncheckpointed_scaffold_overlay(
    tmp_path: Path,
) -> None:
    def no_mutation(root: Path, operation: str) -> None:
        del root, operation

    (
        coordinator,
        engine,
        _,
        _,
        database,
        run_id,
        conversation_id,
        project_id,
    ) = await _coordinator(tmp_path, no_mutation)
    provider = _provider()
    content = "SCAFFOLD_READY = True\n"
    initial_staged = {
        "app/scaffold.py": {
            "content": content,
            "origin": "create",
            "base_sha256": "",
            "bytes": len(content.encode("utf-8")),
            "provenance": {"kind": "host_scaffold"},
        }
    }
    engine.cancel_start = True
    try:
        with pytest.raises(asyncio.CancelledError):
            await coordinator.start(
                run_id=run_id,
                conversation_id=conversation_id,
                project_id=project_id,
                prompt="Continue after the host scaffold",
                staged=initial_staged,
                provider=provider,
                allowed_paths=[],
            )

        # The graph checkpoint predates both scaffold staging and the cancelled
        # coding call, so recovery is deliberately given an empty overlay.
        recovered = await coordinator.recover_for_run(
            run_id,
            project_id=project_id,
            staged={},
            provider=provider,
            allowed_paths=[],
            operation_id=f"{run_id}:round:1",
        )
        assert recovered is not None
        assert recovered.recovered is True
        assert recovered.staged == initial_staged
        assert recovered.changed_paths == []
        assert len(engine.start_requests) == 1
        assert engine.continue_requests == []
        await coordinator.release(recovered.session.id, CodingSessionState.COMPLETED)
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_no_write_recovery_is_distinct_from_an_ordinary_empty_round(
    tmp_path: Path,
) -> None:
    def no_mutation(root: Path, operation: str) -> None:
        del root, operation

    (
        coordinator,
        engine,
        _,
        _,
        database,
        run_id,
        conversation_id,
        project_id,
    ) = await _coordinator(tmp_path, no_mutation)
    provider = _provider()
    engine.cancel_start = True
    try:
        with pytest.raises(asyncio.CancelledError):
            await coordinator.start(
                run_id=run_id,
                conversation_id=conversation_id,
                project_id=project_id,
                prompt="Implement app/planned.py",
                staged={},
                provider=provider,
                allowed_paths=["app/planned.py"],
            )

        recovered = await coordinator.recover_for_run(
            run_id,
            project_id=project_id,
            staged={},
            provider=provider,
            allowed_paths=["app/planned.py"],
            operation_id=f"{run_id}:round:1",
        )
        assert recovered is not None
        assert recovered.recovered is True
        assert recovered.settled_recovery is False
        assert recovered.changed_paths == []
        assert recovered.staged == {}
        await coordinator.release(recovered.session.id, CodingSessionState.COMPLETED)
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_settled_no_write_round_recovers_without_duplicate_model_call(
    tmp_path: Path,
) -> None:
    def no_mutation(root: Path, operation: str) -> None:
        del root, operation

    (
        coordinator,
        engine,
        sessions,
        _,
        database,
        run_id,
        conversation_id,
        project_id,
    ) = await _coordinator(tmp_path, no_mutation)
    provider = _provider()
    try:
        completed = await coordinator.start(
            run_id=run_id,
            conversation_id=conversation_id,
            project_id=project_id,
            prompt="Implement app/planned.py",
            staged={},
            provider=provider,
            allowed_paths=["app/planned.py"],
        )
        mirror_parent = completed.session.workspace_snapshot.project_root.parent
        assert completed.changed_paths == []
        assert completed.session.state is CodingSessionState.IDLE
        assert mirror_parent.is_dir()

        # Simulate a crash after the coordinator returned but before LangGraph
        # checkpointed the terminal no-write response.
        recovered = await coordinator.recover_for_run(
            run_id,
            project_id=project_id,
            staged={},
            provider=provider,
            allowed_paths=["app/planned.py"],
            operation_id=f"{run_id}:round:1",
        )
        assert recovered is not None
        assert recovered.recovered is True
        assert recovered.settled_recovery is True
        assert recovered.changed_paths == []
        assert len(engine.start_requests) == 1
        assert engine.continue_requests == []
        assert engine.restore_requests == []
        persisted = await sessions.get(recovered.session.id)
        assert persisted is not None
        assert persisted.state is CodingSessionState.IDLE
        assert mirror_parent.is_dir()

        await coordinator.release(recovered.session.id, CodingSessionState.COMPLETED)
        assert not mirror_parent.exists()
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_new_continuation_clears_prior_settled_marker_before_mutation(
    tmp_path: Path,
) -> None:
    def no_mutation(root: Path, operation: str) -> None:
        del root, operation

    (
        coordinator,
        engine,
        _,
        _,
        database,
        run_id,
        conversation_id,
        project_id,
    ) = await _coordinator(tmp_path, no_mutation)
    provider = _provider()
    try:
        first = await coordinator.start(
            run_id=run_id,
            conversation_id=conversation_id,
            project_id=project_id,
            prompt="Inspect the planned file",
            staged={},
            provider=provider,
            allowed_paths=["app/planned.py"],
        )
        engine.cancel_continue = True
        with pytest.raises(asyncio.CancelledError):
            await coordinator.continue_session(
                first.session.id,
                prompt="Now implement the planned file",
                staged=first.staged,
                provider=provider,
                allowed_paths=["app/planned.py"],
                operation_id=f"{run_id}:round:2",
            )

        recovered = await coordinator.recover_for_run(
            run_id,
            project_id=project_id,
            staged=first.staged,
            provider=provider,
            allowed_paths=["app/planned.py"],
            operation_id=f"{run_id}:round:2",
        )
        assert recovered is not None
        assert recovered.recovered is True
        assert recovered.settled_recovery is False
        assert len(engine.continue_requests) == 1
        await coordinator.release(recovered.session.id, CodingSessionState.COMPLETED)
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_ambiguous_continue_cleanup_deletes_parent_and_expected_child(
    tmp_path: Path,
) -> None:
    def mutate(root: Path, operation: str) -> None:
        value = 2 if operation == "continue" else 1
        (root / "app" / "planned.py").write_text(f"VALUE = {value}\n", encoding="utf-8")

    (
        coordinator,
        engine,
        _,
        _,
        database,
        run_id,
        conversation_id,
        project_id,
    ) = await _coordinator(tmp_path, mutate)
    provider = _provider()
    try:
        first = await coordinator.start(
            run_id=run_id,
            conversation_id=conversation_id,
            project_id=project_id,
            prompt="Implement app/planned.py",
            staged={},
            provider=provider,
            allowed_paths=["app/planned.py"],
        )
        parent_id = first.session.sidecar_session_id
        assert parent_id is not None

        engine.ambiguous_continue = True
        ambiguous = await coordinator.continue_session(
            first.session.id,
            prompt="Repair app/planned.py",
            staged=first.staged,
            provider=provider,
            allowed_paths=["app/planned.py"],
            operation_id=f"{run_id}:round:2",
        )
        expected_child = engine.continue_requests[-1].recovery_session_id
        assert expected_child is not None
        assert ambiguous.engine_error
        assert ambiguous.session.sidecar_session_id == parent_id
        assert ambiguous.cleanup_sidecar_ids == (parent_id, expected_child)
        assert set(engine.roots) == {parent_id, expected_child}

        engine.delete_failures.add(parent_id)
        with pytest.raises(ProjectCodingError, match="temporary artifacts"):
            await coordinator.release(
                ambiguous.session.id,
                CodingSessionState.FAILED,
                additional_sidecar_ids=ambiguous.cleanup_sidecar_ids,
            )
        assert engine.delete_requests[-2:] == [parent_id, expected_child]
        assert expected_child not in engine.roots
        # The original inference error is still available to the control plane;
        # cleanup failure is reported separately and cannot replace it.
        assert "continue response was lost" in ambiguous.engine_error
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_journal_recovers_post_import_pre_checkpoint_crash_window(
    tmp_path: Path,
) -> None:
    def mutate(root: Path, operation: str) -> None:
        del operation
        (root / "app" / "planned.py").write_text("VALUE = 7\n", encoding="utf-8")

    (
        coordinator,
        engine,
        _,
        _,
        database,
        run_id,
        conversation_id,
        project_id,
    ) = await _coordinator(tmp_path, mutate)
    provider = _provider()
    try:
        committed_session = await coordinator.start(
            run_id=run_id,
            conversation_id=conversation_id,
            project_id=project_id,
            prompt="Implement app/planned.py",
            staged={},
            provider=provider,
            allowed_paths=["app/planned.py"],
        )
        journal = (
            committed_session.session.workspace_snapshot.project_root.parent
            / ".metis-staged-overlay-v1.json"
        )
        assert journal.parent != committed_session.session.workspace_path
        info = journal.stat(follow_symlinks=False)
        assert stat.S_ISREG(info.st_mode)
        assert info.st_mode & 0o077 == 0

        # Simulate graph state that did not checkpoint the returned overlay,
        # even though the coding-session DB rebase and journal were durable.
        recovered = await coordinator.recover_for_run(
            run_id,
            project_id=project_id,
            staged={},
            provider=provider,
            allowed_paths=["app/planned.py"],
            operation_id=f"{run_id}:round:1",
        )
        assert recovered is not None
        assert recovered.recovered is True
        assert recovered.changed_paths == []
        assert recovered.staged["app/planned.py"]["content"] == "VALUE = 7\n"
        assert len(engine.start_requests) == 1
        assert engine.continue_requests == []
        await coordinator.release(recovered.session.id, CodingSessionState.COMPLETED)
        assert not journal.parent.exists()
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_max_iterations_verdict_and_exact_overlay_survive_checkpoint_crash(
    tmp_path: Path,
) -> None:
    def mutate(root: Path, operation: str) -> None:
        del operation
        (root / "app" / "planned.py").write_text("VALUE = 24\n", encoding="utf-8")

    (
        coordinator,
        engine,
        sessions,
        _,
        database,
        run_id,
        conversation_id,
        project_id,
    ) = await _coordinator(tmp_path, mutate)
    provider = _provider()
    engine.result_state = "failed"
    engine.finish_reason = "error"
    engine.controlled_stop_reason = "max_iterations"
    engine.result_summary = "Stopped at the configured iteration boundary."
    engine.result_iterations = 24
    engine.result_tool_call_count = 11
    try:
        stopped = await coordinator.start(
            run_id=run_id,
            conversation_id=conversation_id,
            project_id=project_id,
            prompt="Implement app/planned.py",
            staged={},
            provider=provider,
            allowed_paths=["app/planned.py"],
        )

        assert stopped.finish_reason == "error"
        assert stopped.controlled_stop_reason == "max_iterations"
        assert stopped.session.state is CodingSessionState.FAILED
        assert stopped.staged["app/planned.py"]["content"] == "VALUE = 24\n"
        persisted = await sessions.get(stopped.session.id)
        assert persisted is not None
        assert persisted.state is CodingSessionState.FAILED
        assert "iteration boundary" in (persisted.last_error or "")
        assert isinstance(coordinator.events, RecordingEvents)
        [coding_round] = [
            payload
            for _, _, event_type, payload in coordinator.events.items
            if event_type == "project.coding_round"
        ]
        assert coding_round["state"] == "failed"
        assert coding_round["engine_state"] == "failed"
        assert coding_round["finish_reason"] == "error"
        assert coding_round["controlled_stop_reason"] == "max_iterations"
        assert coding_round["iterations"] == 24
        assert coding_round["tool_call_count"] == 11

        # The DB row and private journal committed, but the graph checkpoint
        # did not receive the returned overlay or structured terminal reason.
        recovered = await coordinator.recover_for_run(
            run_id,
            project_id=project_id,
            staged={},
            provider=provider,
            allowed_paths=["app/planned.py"],
            operation_id=f"{run_id}:round:1",
        )

        assert recovered is not None
        assert recovered.recovered is True
        assert recovered.settled_recovery is True
        assert recovered.result == stopped.result
        assert recovered.finish_reason == "error"
        assert recovered.controlled_stop_reason == "max_iterations"
        assert recovered.staged == stopped.staged
        assert recovered.staged["app/planned.py"]["content"] == "VALUE = 24\n"
        assert len(engine.start_requests) == 1
        assert engine.continue_requests == []
        assert engine.restore_requests == []

        repaired = await coordinator.continue_session(
            recovered.session.id,
            prompt="Repair the verifier's missing planned artifact",
            staged=recovered.staged,
            provider=provider,
            allowed_paths=["app/planned.py"],
            operation_id=f"{run_id}:round:2",
        )
        [restart] = engine.restart_requests
        assert restart.session_id == recovered.session.sidecar_session_id
        assert restart.provider.model_id == provider.model_id
        assert restart.new_session_id == repaired.session.sidecar_session_id
        assert restart.new_session_id != restart.session_id
        assert engine.continue_requests == []
        await coordinator.release(repaired.session.id, CodingSessionState.COMPLETED)
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_arbitrary_failed_session_cannot_use_controlled_stop_restart(
    tmp_path: Path,
) -> None:
    def mutate(root: Path, operation: str) -> None:
        del operation
        (root / "app" / "planned.py").write_text("VALUE = 24\n", encoding="utf-8")

    (
        coordinator,
        engine,
        _,
        _,
        database,
        run_id,
        conversation_id,
        project_id,
    ) = await _coordinator(tmp_path, mutate)
    provider = _provider()
    engine.result_state = "failed"
    engine.finish_reason = "error"
    engine.result_summary = "Agent runtime exceeded maxIterations (24)"
    engine.result_iterations = 24
    try:
        failed = await coordinator.start(
            run_id=run_id,
            conversation_id=conversation_id,
            project_id=project_id,
            prompt="Implement app/planned.py",
            staged={},
            provider=provider,
            allowed_paths=["app/planned.py"],
        )

        assert failed.session.state is CodingSessionState.FAILED
        assert failed.controlled_stop_reason is None
        with pytest.raises(ProjectCodingError, match="exact controlled-stop evidence"):
            await coordinator.continue_session(
                failed.session.id,
                prompt="Try to continue the failed runtime",
                staged=failed.staged,
                provider=provider,
                allowed_paths=["app/planned.py"],
                operation_id=f"{run_id}:round:2",
            )
        assert engine.restart_requests == []
        assert engine.continue_requests == []
        await coordinator.release(failed.session.id, CodingSessionState.FAILED)
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_journal_newer_than_session_db_is_ignored_and_diff_reimported(
    tmp_path: Path,
) -> None:
    def mutate(root: Path, operation: str) -> None:
        del operation
        (root / "app" / "planned.py").write_text("VALUE = 9\n", encoding="utf-8")

    (
        coordinator,
        engine,
        sessions,
        projects,
        database,
        run_id,
        conversation_id,
        project_id,
    ) = await _coordinator(tmp_path, mutate)
    provider = _provider()
    engine.cancel_start = True
    try:
        with pytest.raises(asyncio.CancelledError):
            await coordinator.start(
                run_id=run_id,
                conversation_id=conversation_id,
                project_id=project_id,
                prompt="Implement app/planned.py",
                staged={},
                provider=provider,
                allowed_paths=["app/planned.py"],
            )
        [interrupted] = await sessions.for_run(run_id)
        old_mirror = snapshot_mirror(interrupted.workspace_snapshot)

        # Simulate the precise later half of a commit: the new journal was
        # atomically renamed, but the session DB still names the old baseline.
        _, future_staged = await projects.import_external_changes(
            project_id,
            old_mirror,
            {},
            provenance=ExternalChangeProvenance(
                engine="clinecore",
                session_id=interrupted.sidecar_session_id or interrupted.id,
                run_id=run_id,
            ),
        )
        future_mirror = await projects.rebase_external_mirror(old_mirror, future_staged)
        await coordinator._write_staged_journal(  # noqa: SLF001 - crash injection
            future_mirror, future_staged
        )

        recovered = await coordinator.recover_for_run(
            run_id,
            project_id=project_id,
            staged={},
            provider=provider,
            allowed_paths=["app/planned.py"],
            operation_id=f"{run_id}:round:1",
        )
        assert recovered is not None
        assert recovered.staged["app/planned.py"]["content"] == "VALUE = 9\n"
        assert recovered.changed_paths == ["app/planned.py"]
        assert len(engine.start_requests) == 1
        assert engine.continue_requests == []
        await coordinator.release(recovered.session.id, CodingSessionState.COMPLETED)
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_round_refuses_entire_diff_when_engine_leaves_host_plan(
    tmp_path: Path,
) -> None:
    def mutate(root: Path, operation: str) -> None:
        del operation
        (root / "app" / "unplanned.py").write_text("DRIFT = True\n", encoding="utf-8")

    (
        coordinator,
        engine,
        sessions,
        _,
        database,
        run_id,
        conversation_id,
        project_id,
    ) = await _coordinator(tmp_path, mutate)
    operation_id = f"{run_id}:round:1"
    usage = CodingUsageV1(
        inputTokens=170_336,
        outputTokens=2_185,
        totalTokens=172_521,
        requests=3,
    )
    engine.result_iterations = 3
    engine.result_tool_call_count = 5
    engine.result_usage = usage
    assert isinstance(coordinator.events, RecordingEvents)
    coordinator.events.fail_once.add("project.coding_rejected")
    try:
        # The result and refusal settle, but the first durable event write fails.
        # Ownership must remain recoverable instead of discarding billable usage.
        with pytest.raises(ProjectCodingError, match="retained for recovery"):
            await coordinator.start(
                run_id=run_id,
                conversation_id=conversation_id,
                project_id=project_id,
                prompt="Only implement app/planned.py",
                staged={},
                provider=_provider(),
                allowed_paths=["app/planned.py"],
                operation_id=operation_id,
            )

        [interrupted] = await sessions.for_run(run_id)
        assert interrupted.state is CodingSessionState.RUNNING
        assert len(engine.start_requests) == 1
        assert engine.restore_requests == []
        assert not (tmp_path / "Projects" / "demo" / "app" / "unplanned.py").exists()

        # Recovery replays the settled rejection without another model turn.
        with pytest.raises(ProjectCodingError, match="outside the host plan"):
            await coordinator.recover_for_run(
                run_id,
                project_id=project_id,
                staged={},
                provider=_provider(),
                allowed_paths=["app/planned.py"],
                operation_id=operation_id,
            )

        [session] = await sessions.for_run(run_id)
        assert session.state is CodingSessionState.FAILED
        assert "app/unplanned.py" in str(session.last_error)
        assert not (tmp_path / "Projects" / "demo" / "app" / "unplanned.py").exists()
        assert len(engine.start_requests) == 1
        assert engine.restore_requests == []
        [rejected] = [
            payload
            for _, _, event_type, payload in coordinator.events.items
            if event_type == "project.coding_rejected"
        ]
        assert rejected["operation_id"] == operation_id
        assert rejected["sidecar_session_id"] == session.sidecar_session_id
        assert rejected["model"] == "model-a"
        assert rejected["iterations"] == 3
        assert rejected["tool_call_count"] == 5
        assert rejected["usage"] == usage.model_dump(mode="json")
        assert rejected["accepted"] is False
        assert rejected["changed_paths"] == []
        assert rejected["controlled_stop"] is False
        assert not any(
            event_type == "project.coding_round"
            for _, _, event_type, _ in coordinator.events.items
        )

        mirror_parent = session.workspace_snapshot.project_root.parent
        await coordinator.release(session.id, CodingSessionState.FAILED)
        assert not mirror_parent.exists()
        assert engine.delete_requests == [session.sidecar_session_id]
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_continuation_refuses_diff_that_rewrites_an_earlier_verified_slice(
    tmp_path: Path,
) -> None:
    """A round's write scope is exactly its own allowed_paths, never widened
    with the accumulated staged overlay. Rewriting a file an earlier slice
    already staged (and had verified) must be refused like any other
    out-of-plan change, not silently accepted because the path is familiar.
    """

    def mutate(root: Path, operation: str) -> None:
        if operation == "start":
            (root / "app" / "planned.py").write_text("VALUE = 1\n", encoding="utf-8")
        else:
            (root / "app" / "planned.py").write_text("DRIFT = True\n", encoding="utf-8")

    (
        coordinator,
        engine,
        sessions,
        _,
        database,
        run_id,
        conversation_id,
        project_id,
    ) = await _coordinator(tmp_path, mutate)
    provider = _provider()
    try:
        first = await coordinator.start(
            run_id=run_id,
            conversation_id=conversation_id,
            project_id=project_id,
            prompt="Implement the first vertical slice",
            staged={},
            provider=provider,
            allowed_paths=["app/planned.py"],
        )
        assert first.staged["app/planned.py"]["content"] == "VALUE = 1\n"

        with pytest.raises(ProjectCodingError, match="outside the host plan"):
            await coordinator.continue_session(
                first.session.id,
                prompt="Implement the next vertical slice",
                staged=first.staged,
                provider=provider,
                allowed_paths=["app/other.py"],
                operation_id=f"{run_id}:round:2",
            )

        [session] = await sessions.for_run(run_id)
        assert session.state is CodingSessionState.FAILED
        assert "app/planned.py" in str(session.last_error)
        # The rejected round's bytes never reach the caller's staged overlay.
        assert first.staged["app/planned.py"]["content"] == "VALUE = 1\n"
        assert len(engine.continue_requests) == 1
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_in_plan_syntax_refusal_keeps_private_mirror_for_model_repair(
    tmp_path: Path,
) -> None:
    def mutate(root: Path, operation: str) -> None:
        content = "def broken(:\n" if operation == "start" else "VALUE = 42\n"
        (root / "app" / "planned.py").write_text(content, encoding="utf-8")

    (
        coordinator,
        engine,
        sessions,
        _,
        database,
        run_id,
        conversation_id,
        project_id,
    ) = await _coordinator(tmp_path, mutate)
    try:
        rejected = await coordinator.start(
            run_id=run_id,
            conversation_id=conversation_id,
            project_id=project_id,
            prompt="Implement app/planned.py",
            staged={},
            provider=_provider(),
            allowed_paths=["app/planned.py"],
            operation_id=f"{run_id}:round:1",
        )

        assert rejected.rejection_repairable is True
        assert rejected.rejection_path == "app/planned.py"
        assert "does not parse" in rejected.rejection_reason
        assert rejected.staged == {}
        assert rejected.session.state is CodingSessionState.FAILED
        assert not (tmp_path / "Projects" / "demo" / "app" / "planned.py").exists()
        assert engine.delete_requests == []

        repaired = await coordinator.continue_session(
            rejected.session.id,
            prompt="Repair the exact syntax error in app/planned.py",
            staged=rejected.staged,
            provider=_provider("model-b"),
            allowed_paths=["app/planned.py"],
            operation_id=f"{run_id}:round:2",
        )

        assert len(engine.restart_requests) == 1
        assert engine.restart_requests[0].provider.model_id == "model-b"
        assert repaired.staged["app/planned.py"]["content"] == "VALUE = 42\n"
        assert repaired.session.state is CodingSessionState.IDLE
        [persisted] = await sessions.for_run(run_id)
        assert repaired.result is not None
        assert persisted.sidecar_session_id == repaired.result.session_id
        assert not (tmp_path / "Projects" / "demo" / "app" / "planned.py").exists()
        await coordinator.release(repaired.session.id, CodingSessionState.COMPLETED)
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_empty_unplanned_package_marker_is_discarded_not_imported(
    tmp_path: Path,
) -> None:
    def mutate(root: Path, operation: str) -> None:
        del operation
        (root / "app" / "planned.py").write_text("VALUE = 42\n", encoding="utf-8")
        (root / "app" / "__init__.py").write_text("\n", encoding="utf-8")

    (
        coordinator,
        engine,
        _,
        _,
        database,
        run_id,
        conversation_id,
        project_id,
    ) = await _coordinator(tmp_path, mutate)
    try:
        completed = await coordinator.start(
            run_id=run_id,
            conversation_id=conversation_id,
            project_id=project_id,
            prompt="Implement app/planned.py",
            staged={},
            provider=_provider(),
            allowed_paths=["app/planned.py"],
        )

        assert completed.changed_paths == ["app/planned.py"]
        assert completed.changes["ignored_auxiliary_paths"] == ["app/__init__.py"]
        assert "app/__init__.py" not in completed.staged
        assert not (
            engine.roots[completed.result.session_id] / "app" / "__init__.py"
        ).exists()
        await coordinator.release(completed.session.id, CodingSessionState.COMPLETED)
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_nonempty_unplanned_package_marker_remains_out_of_scope(
    tmp_path: Path,
) -> None:
    def mutate(root: Path, operation: str) -> None:
        del operation
        (root / "app" / "planned.py").write_text("VALUE = 42\n", encoding="utf-8")
        (root / "app" / "__init__.py").write_text("# package\n", encoding="utf-8")

    (
        coordinator,
        _,
        _,
        _,
        database,
        run_id,
        conversation_id,
        project_id,
    ) = await _coordinator(tmp_path, mutate)
    try:
        with pytest.raises(ProjectCodingError, match="outside the host plan"):
            await coordinator.start(
                run_id=run_id,
                conversation_id=conversation_id,
                project_id=project_id,
                prompt="Implement app/planned.py",
                staged={},
                provider=_provider(),
                allowed_paths=["app/planned.py"],
            )
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_model_switch_promotes_host_chosen_child_only_after_import(
    tmp_path: Path,
) -> None:
    def mutate(root: Path, operation: str) -> None:
        value = 2 if operation == "restart" else 1
        (root / "app" / "planned.py").write_text(f"VALUE = {value}\n", encoding="utf-8")

    (
        coordinator,
        engine,
        sessions,
        projects,
        database,
        run_id,
        conversation_id,
        project_id,
    ) = await _coordinator(tmp_path, mutate)
    try:
        first = await coordinator.start(
            run_id=run_id,
            conversation_id=conversation_id,
            project_id=project_id,
            prompt="Implement app/planned.py",
            staged={},
            provider=_provider("model-a"),
            allowed_paths=["app/planned.py"],
        )
        parent_sidecar_id = first.session.sidecar_session_id
        assert parent_sidecar_id is not None

        switched = await coordinator.continue_session(
            first.session.id,
            prompt="Repair the planned file with the fallback model",
            staged=first.staged,
            provider=_provider("model-b"),
            allowed_paths=["app/planned.py"],
            operation_id=f"{run_id}:round:2",
        )
        [request] = engine.restart_requests
        assert request.session_id == parent_sidecar_id
        assert request.new_session_id is not None
        assert request.new_session_id != parent_sidecar_id
        # A model-switch repair fork carries the exact slice scope forward;
        # it must never widen write authority beyond the failing slice.
        assert request.allowed_paths == ["app/planned.py"]
        assert switched.session.sidecar_session_id == request.new_session_id
        assert switched.session.model_route.model_id == "model-b"
        assert switched.staged["app/planned.py"]["content"] == "VALUE = 2\n"
        persisted = await sessions.get(first.session.id)
        assert persisted is not None
        assert persisted.sidecar_session_id == request.new_session_id
        assert persisted.model_route.model_id == "model-b"
        assert isinstance(coordinator.events, RecordingEvents)
        coding_rounds = [
            payload
            for _, _, event_type, payload in coordinator.events.items
            if event_type == "project.coding_round"
        ]
        assert coding_rounds[-1]["sidecar_session_id"] == request.new_session_id
        assert coding_rounds[-1]["parent_sidecar_session_id"] == parent_sidecar_id
        await projects.discard_external_mirror(
            snapshot_mirror(switched.session.workspace_snapshot)
        )
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_round_rejects_sidecar_session_identity_substitution(
    tmp_path: Path,
) -> None:
    def mutate(root: Path, operation: str) -> None:
        del operation
        (root / "app" / "planned.py").write_text("VALUE = 1\n", encoding="utf-8")

    (
        coordinator,
        engine,
        sessions,
        projects,
        database,
        run_id,
        conversation_id,
        project_id,
    ) = await _coordinator(tmp_path, mutate)
    engine.result_session_id = "coding_foreign"
    try:
        with pytest.raises(ProjectCodingError, match="different session identity"):
            await coordinator.start(
                run_id=run_id,
                conversation_id=conversation_id,
                project_id=project_id,
                prompt="Implement app/planned.py",
                staged={},
                provider=_provider(),
                allowed_paths=["app/planned.py"],
            )

        [session] = await sessions.for_run(run_id)
        assert session.state is CodingSessionState.FAILED
        assert session.sidecar_session_id == session.id
        assert not (tmp_path / "Projects" / "demo" / "app" / "planned.py").exists()
        await projects.discard_external_mirror(
            snapshot_mirror(session.workspace_snapshot)
        )
    finally:
        await database.close()


def test_prompt_marks_new_authorized_files_so_the_model_never_searches_for_them() -> (
    None
):
    """A live trace burned an entire slice budget alternating denied
    read_files calls with unproductive searches, never once calling editor,
    because the model was never told which of its own authorized files
    already exist versus are new. The fix is to say so up front."""

    prompt = initial_coding_prompt(
        task="Build the app",
        planned_files=["app/main.py", "app/config.py"],
        scenarios=[],
        spec=None,
        repo_map="",
        staged={"app/config.py": {"content": "VALUE = 1\n"}},
    )

    assert (
        "- app/main.py [NEW — does not exist yet; create it directly with "
        "editor, do not call read_files on it first]" in prompt
    )
    assert "- app/config.py [EXISTS — read it before editing]" in prompt
    assert "- app/config.py" in prompt.split("READABLE CONTEXT FROM EARLIER WORK")[1]


def test_prompt_names_first_slice_as_having_nothing_staged_yet() -> None:
    prompt = initial_coding_prompt(
        task="Build the app",
        planned_files=["app/main.py"],
        scenarios=[],
        spec=None,
        repo_map="",
        staged=None,
    )

    assert (
        "- app/main.py [NEW — does not exist yet; create it directly with "
        "editor, do not call read_files on it first]" in prompt
    )
    assert (
        "(none yet — this is the first slice; nothing has been staged before it)"
        in prompt
    )
