from __future__ import annotations

import hashlib
import os
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from waqil_api.coding_contracts import (
    CodingCleanupPlanV1,
    CodingCleanupStatus,
    CodingModelRouteV1,
    CodingSessionCreateV1,
    CodingSessionState,
    CodingSessionUpdateV1,
    CodingWorkspaceSnapshotV1,
)
from waqil_api.coding_engine import CodingEngine, CodingSessionStore
from waqil_api.config import Settings
from waqil_api.contracts import RunStatus
from waqil_api.control_plane import ControlPlane
from waqil_api.database import Database, MIGRATIONS
from waqil_api.project_coding_engine import (
    ProjectCodingCoordinator,
    ProjectCodingError,
)
from waqil_api.project_workspace import ProjectWorkspaceService


class DeleteEngine:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.fail: set[str] = set()

    async def delete_session(self, session_id: str) -> None:
        self.calls.append(session_id)
        if session_id in self.fail:
            raise RuntimeError(f"delete failed for {session_id}")


class DiscardProjects:
    def __init__(self) -> None:
        self.calls: list[Path] = []
        self.orphan_sweeps: list[set[Path]] = []

    async def discard_external_mirror(self, mirror: Any) -> None:
        self.calls.append(mirror.project_root)

    async def discard_unreferenced_external_mirrors(
        self,
        referenced_workspace_paths: set[Path],
        *,
        older_than: datetime,
    ) -> tuple[Path, ...]:
        assert older_than.tzinfo is not None
        self.orphan_sweeps.append(referenced_workspace_paths)
        return ()


class NoopEvents:
    async def emit(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs


async def _owned_session(
    database: Database,
    tmp_path: Path,
) -> tuple[CodingSessionStore, Any, Any]:
    conversation = await database.create_conversation("cleanup")
    message = await database.add_message(conversation.id, "user", "build")
    run = await database.create_run(
        conversation.id,
        message.id,
        graph_schema_version="1",
        model_aliases={"_coding_engine": "clinecore"},
    )
    source = (tmp_path / "source").resolve()
    source.mkdir(exist_ok=True)
    project_root = (
        tmp_path / "coding-workspaces" / "metis-code-workspace-owned01" / "project"
    ).resolve()
    project_root.mkdir(parents=True, exist_ok=True)
    empty_digest = hashlib.sha256().hexdigest()
    snapshot = CodingWorkspaceSnapshotV1(
        mirror_id="ewm_cleanup",
        asset_id="asset_cleanup",
        source_root=source,
        project_root=project_root,
        tree_sha256=empty_digest,
        overlay_sha256="0" * 64,
        files=[],
        excluded_count=0,
        created_at=datetime.now(UTC),
    )
    store = CodingSessionStore(database)
    session = await store.create(
        CodingSessionCreateV1(
            run_id=run.id,
            conversation_id=conversation.id,
            project_id="asset_cleanup",
            workspace_path=project_root,
            baseline_digest=empty_digest,
            overlay_digest="0" * 64,
            workspace_snapshot=snapshot,
            model_route=CodingModelRouteV1(
                providerId="ollama",
                modelId="deepseek-v4-pro:cloud",
            ),
        )
    )
    session = await store.update(
        session.id,
        CodingSessionUpdateV1(
            sidecar_session_id="sidecar_parent",
            state=CodingSessionState.IDLE,
        ),
    )
    return store, session, run


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path,
        repo_root=Path(__file__).resolve().parents[3],
        model_backend="deterministic",
        reference_runner_mode="deterministic",
        allow_test_backends=True,
    )


@pytest.mark.asyncio
async def test_schema_v24_upgrades_existing_coding_session_table(
    tmp_path: Path,
) -> None:
    path = tmp_path / "v23.db"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE schema_migrations "
        "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    for version in range(1, 24):
        connection.executescript(
            "BEGIN IMMEDIATE;\n"
            + MIGRATIONS[version]
            + "\nINSERT INTO schema_migrations(version, applied_at) "
            + f"VALUES ({version}, CURRENT_TIMESTAMP);\nCOMMIT;"
        )
    connection.close()

    database = Database(path)
    await database.open()
    try:
        columns = {
            str(row["name"])
            for row in database._connection()  # noqa: SLF001
            .execute("PRAGMA table_info(coding_sessions)")
            .fetchall()
        }
        assert {
            "cleanup_status",
            "cleanup_target_state",
            "cleanup_attempts",
            "cleanup_next_attempt_at",
            "cleanup_last_error",
            "cleanup_sidecar_ids_json",
            "released_at",
        } <= columns
        versions = {
            int(row[0])
            for row in database._connection()  # noqa: SLF001
            .execute("SELECT version FROM schema_migrations")
            .fetchall()
        }
        assert 24 in versions
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_cleanup_recovers_crash_after_durable_intent(tmp_path: Path) -> None:
    database = Database(tmp_path / "waqil.db")
    await database.open()
    try:
        store, session, run = await _owned_session(database, tmp_path)
        now = datetime.now(UTC)
        pending = await store.begin_cleanup(
            session.id,
            CodingCleanupPlanV1(
                target_state=CodingSessionState.COMPLETED,
                sidecar_session_ids=("sidecar_child",),
            ),
            lease_until=now + timedelta(seconds=30),
        )

        # This is the crash boundary: intent and every child id are durable,
        # while the audit state still says IDLE and no deletion has happened.
        assert pending.cleanup_status is CodingCleanupStatus.PENDING
        assert pending.state is CodingSessionState.IDLE
        assert pending.cleanup_sidecar_ids == ("sidecar_parent", "sidecar_child")
        # Approval recovery can finish cleanup before _drive commits the run
        # verdict, but only because this session already has durable cleanup
        # intent. An ACTIVE session beside the same decision is never swept.
        database._connection().execute(  # noqa: SLF001
            """INSERT INTO approvals
            (id, run_id, proposal_id, action_id, kind, status, request_json,
             decision_json, created_at, decided_at)
            VALUES (?, ?, NULL, ?, 'project_apply_build', 'approve', '{}', '{}', ?, ?)""",
            (
                "approval_cleanup",
                run.id,
                "action_cleanup",
                now.isoformat(),
                now.isoformat(),
            ),
        )
        assert await store.cleanup_candidates(now=now) == []
        assert [
            item.id
            for item in await store.cleanup_candidates(now=now + timedelta(seconds=31))
        ] == [session.id]

        engine = DeleteEngine()
        projects = DiscardProjects()
        coordinator = ProjectCodingCoordinator(
            _settings(tmp_path),
            cast(CodingEngine, engine),
            store,
            projects,
            NoopEvents(),
        )
        clean = await coordinator.release(
            session.id,
            CodingSessionState.COMPLETED,
        )

        assert clean is not None
        assert clean.cleanup_status is CodingCleanupStatus.CLEAN
        assert clean.state is CodingSessionState.COMPLETED
        assert clean.cleanup_attempts == 2
        assert clean.released_at is not None
        assert engine.calls == ["sidecar_parent", "sidecar_child"]
        assert projects.calls == [session.workspace_path]
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_cleanup_failure_backs_off_without_changing_run_verdict(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "waqil.db")
    await database.open()
    try:
        store, session, run = await _owned_session(database, tmp_path)
        await database.set_run_status(run.id, RunStatus.COMPLETED)
        engine = DeleteEngine()
        engine.fail.add("sidecar_parent")
        projects = DiscardProjects()
        coordinator = ProjectCodingCoordinator(
            _settings(tmp_path),
            cast(CodingEngine, engine),
            store,
            projects,
            NoopEvents(),
        )

        with pytest.raises(ProjectCodingError, match="could not be released"):
            await coordinator.release(
                session.id,
                CodingSessionState.COMPLETED,
                additional_sidecar_ids=("sidecar_child",),
            )

        retry = await store.get(session.id)
        assert retry is not None
        assert retry.cleanup_status is CodingCleanupStatus.RETRY
        assert retry.state is CodingSessionState.IDLE
        assert retry.cleanup_attempts == 1
        assert retry.cleanup_next_attempt_at is not None
        assert "delete failed" in str(retry.cleanup_last_error)
        durable_run = await database.get_run(run.id)
        assert durable_run is not None
        assert durable_run.status == RunStatus.COMPLETED.value
        assert (
            await store.cleanup_candidates(
                now=retry.cleanup_next_attempt_at - timedelta(microseconds=1)
            )
            == []
        )

        engine.fail.clear()
        clean = await coordinator.release(
            session.id,
            CodingSessionState.COMPLETED,
        )
        assert clean is not None
        assert clean.cleanup_status is CodingCleanupStatus.CLEAN
        assert clean.cleanup_attempts == 2
        # The already-deleted child and mirror are retried deliberately; both
        # operations are idempotent, which closes every crash boundary.
        assert engine.calls == [
            "sidecar_parent",
            "sidecar_child",
            "sidecar_parent",
            "sidecar_child",
        ]
        assert projects.calls == [session.workspace_path, session.workspace_path]
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_maintenance_releases_terminal_no_write_orphan_without_inference(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "waqil.db")
    await database.open()
    try:
        store, session, run = await _owned_session(database, tmp_path)
        await database.set_run_status(run.id, RunStatus.COMPLETED)
        engine = DeleteEngine()
        projects = DiscardProjects()
        coordinator = ProjectCodingCoordinator(
            _settings(tmp_path),
            cast(CodingEngine, engine),
            store,
            projects,
            NoopEvents(),
        )

        class Checkpointer:
            async def aget_tuple(self, config: Any) -> Any:
                del config
                return SimpleNamespace(
                    checkpoint={
                        "channel_values": {
                            "project_coding_session_id": session.id,
                            "project_staged": {},
                        }
                    }
                )

        plane = object.__new__(ControlPlane)
        plane.settings = _settings(tmp_path)
        plane.database = database
        plane.project_coding = coordinator
        plane.projects = projects
        plane.checkpointer = Checkpointer()
        plane.events = NoopEvents()
        plane._config = lambda conversation_id, run_id: {  # type: ignore[method-assign]
            "conversation_id": conversation_id,
            "run_id": run_id,
        }

        stats = await ControlPlane.reconcile_coding_cleanup(plane)

        clean = await store.get(session.id)
        assert clean is not None
        assert clean.cleanup_status is CodingCleanupStatus.CLEAN
        assert stats == {
            "candidates": 1,
            "held": 0,
            "released": 1,
            "orphans": 0,
            # The journal sweep runs in the same pass and finds nothing:
            # every identity here is still referenced.
            "orphan_journals": 0,
        }
        assert engine.calls == ["sidecar_parent"]
        assert projects.orphan_sweeps == [{session.workspace_path}]
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_completed_no_approval_staged_bytes_are_held_and_reported() -> None:
    held: list[Any] = []
    emitted: list[tuple[str, dict[str, Any]]] = []
    released = False

    class Sessions:
        async def hold_cleanup(self, session_id: str, value: Any) -> None:
            held.append((session_id, value))

    class Coordinator:
        sessions = Sessions()

    class Events:
        async def emit(
            self,
            run_id: str,
            conversation_id: str,
            event_type: str,
            payload: dict[str, Any],
        ) -> None:
            del run_id, conversation_id
            emitted.append((event_type, payload))

    async def release(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        nonlocal released
        released = True

    plane = object.__new__(ControlPlane)
    plane.project_coding = Coordinator()
    plane.events = Events()
    plane._release_project_coding_session = release  # type: ignore[method-assign]
    await ControlPlane._cleanup_completed_no_approval_session(
        plane,
        {  # type: ignore[arg-type]
            "run_id": "run_held",
            "conversation_id": "conv_held",
            "project_staged": {"app.py": {"content": "changed"}},
        },
        "coding_held",
    )

    assert released is False
    assert held[0][0] == "coding_held"
    assert held[0][1].target_state is CodingSessionState.COMPLETED
    assert emitted[0][0] == "project.coding_cleanup_held"


def _make_workspace(container: Path, *, modified_at: datetime) -> Path:
    project = container / "project"
    project.mkdir(parents=True)
    (project / "app.py").write_text("print('safe')\n", encoding="utf-8")
    stamp = modified_at.timestamp()
    os.utime(project / "app.py", (stamp, stamp))
    os.utime(project, (stamp, stamp))
    os.utime(container, (stamp, stamp))
    return project


@pytest.mark.asyncio
async def test_orphan_mirror_gc_removes_only_old_canonical_unreferenced_dirs(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path / "data")
    settings.prepare_directories()
    parent = settings.coding_workspace_dir
    now = datetime.now(UTC)
    old = now - timedelta(days=3)
    cutoff = now - timedelta(days=1)

    removable = _make_workspace(
        parent / "metis-code-workspace-remove01",
        modified_at=old,
    )
    referenced = _make_workspace(
        parent / "metis-code-workspace-owned001",
        modified_at=old,
    )
    recent = _make_workspace(
        parent / "metis-code-workspace-recent01",
        modified_at=now,
    )
    wrong_name = _make_workspace(parent / "project-root", modified_at=old)
    linked_target = tmp_path / "real-project"
    linked_target.mkdir()
    linked = parent / "metis-code-workspace-linked01"
    linked.symlink_to(linked_target, target_is_directory=True)
    internal_link = _make_workspace(
        parent / "metis-code-workspace-linkin01",
        modified_at=old,
    )
    (internal_link / "escape").symlink_to(linked_target, target_is_directory=True)

    service = object.__new__(ProjectWorkspaceService)
    service.settings = settings
    removed = await service.discard_unreferenced_external_mirrors(
        {referenced},
        older_than=cutoff,
    )

    assert removed == (removable.parent,)
    assert not removable.parent.exists()
    assert referenced.parent.exists()
    assert recent.parent.exists()
    assert wrong_name.parent.exists()
    assert linked.is_symlink()
    assert internal_link.parent.exists()
    assert linked_target.is_dir()
