"""Event journals are released with the session that owns them.

The live leak: a continuation minted a recovery fork, the round was refused on
an error path that passed only the parent id, and the fork's journal outlived
the run -- 19,940 bytes still on disk after cleanup reported `clean`, with no
row in `coding_sessions` naming it.

Ancestry is committed when an identity is minted now, so terminal cleanup knows
every journal to release. The sweep below is only the backstop for a crash
between minting and committing, and it is deliberately timid: age-gated, one
directory, referenced identities never touched.
"""

from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from waqil_api.coding_contracts import (
    CodingCleanupPlanV1,
    CodingModelRouteV1,
    CodingSessionCreateV1,
    CodingSessionState,
    CodingSessionUpdateV1,
    CodingWorkspaceSnapshotV1,
)
from waqil_api.coding_engine import CodingSessionStore
from waqil_api.config import Settings
from waqil_api.database import Database
from waqil_api.project_coding_engine import ProjectCodingCoordinator


def _settings(tmp_path: Path) -> Settings:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        data_dir=tmp_path / "data",
        repo_root=tmp_path,
        model_backend="deterministic",
        allow_test_backends=True,
    )
    settings.prepare_directories()
    settings.cline_event_journal_dir.mkdir(parents=True, exist_ok=True)
    return settings


def _journal(settings: Settings, identity: str, *, age_days: float = 0.0) -> Path:
    path = settings.cline_event_journal_dir / f"{identity}.jsonl"
    path.write_text('{"cursor":1}\n', encoding="utf-8")
    if age_days:
        old = datetime.now(UTC).timestamp() - age_days * 86_400
        os.utime(path, (old, old))
    return path


def _coordinator(settings: Settings, database: Database) -> ProjectCodingCoordinator:
    coordinator = object.__new__(ProjectCodingCoordinator)
    coordinator.settings = settings  # type: ignore[attr-defined]
    coordinator.sessions = CodingSessionStore(database)  # type: ignore[attr-defined]
    return coordinator


async def _session(database: Database, tmp_path: Path, *, sidecar_id: str) -> Any:
    conversation = await database.create_conversation("journals")
    message = await database.add_message(conversation.id, "user", "go")
    run = await database.create_run(
        conversation.id, message.id, graph_schema_version="1", model_aliases={}
    )
    # The empty-manifest digest, so the snapshot validates.
    empty_tree = hashlib.sha256().hexdigest()
    created = await database.create_coding_session(
        CodingSessionCreateV1(
            run_id=run.id,
            conversation_id=conversation.id,
            project_id="asset_1",
            workspace_path=tmp_path / "mirror",
            workspace_snapshot=CodingWorkspaceSnapshotV1(
                mirror_id="ewm_1",
                asset_id="asset_1",
                source_root=tmp_path / "src",
                project_root=tmp_path / "mirror",
                tree_sha256=empty_tree,
                overlay_sha256=hashlib.sha256(b"").hexdigest(),
                files=[],
                excluded_count=0,
                created_at=datetime.now(UTC),
            ),
            baseline_digest=empty_tree,
            overlay_digest=hashlib.sha256(b"").hexdigest(),
            model_route=CodingModelRouteV1(providerId="ollama", modelId="glm-5.2"),
        )
    )
    # The sidecar identity is assigned once the child answers, exactly as the
    # coordinator does it.
    return await database.update_coding_session(
        created.id, CodingSessionUpdateV1(sidecar_session_id=sidecar_id)
    )


@pytest.mark.asyncio
async def test_a_minted_fork_identity_is_owned_before_it_is_used(
    tmp_path: Path,
) -> None:
    """Ancestry is durable at mint time, and terminal cleanup releases it."""

    _settings(tmp_path)
    database = Database(tmp_path / "sessions.db")
    await database.open()
    try:
        session = await _session(database, tmp_path, sidecar_id="parent-1")
        store = CodingSessionStore(database)
        await store.record_ancestry(session.id, ("parent-1", "recovery-1"))

        reloaded = await store.get(session.id)
        assert reloaded is not None
        assert set(reloaded.sidecar_ancestry) == {"parent-1", "recovery-1"}
        # Ancestry is not cleanup progress: the session is still active.
        assert reloaded.cleanup_sidecar_ids == ()

        # Terminal cleanup releases the ancestry, not just this round's id.
        pending = await store.begin_cleanup(
            session.id,
            CodingCleanupPlanV1(
                target_state=CodingSessionState.FAILED,
                sidecar_session_ids=("parent-1",),
            ),
            lease_until=datetime.now(UTC) + timedelta(seconds=30),
        )
        assert set(pending.cleanup_sidecar_ids) == {"parent-1", "recovery-1"}
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_recording_ancestry_is_idempotent(tmp_path: Path) -> None:
    _settings(tmp_path)
    database = Database(tmp_path / "sessions.db")
    await database.open()
    try:
        session = await _session(database, tmp_path, sidecar_id="parent-1")
        store = CodingSessionStore(database)
        for _ in range(3):
            await store.record_ancestry(session.id, ("parent-1", "recovery-1"))
        reloaded = await store.get(session.id)
        assert reloaded is not None
        assert sorted(reloaded.sidecar_ancestry) == ["parent-1", "recovery-1"]
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_the_sweep_removes_only_aged_unreferenced_journals(
    tmp_path: Path,
) -> None:
    """An orphan goes; an active session's journal and a fresh one stay."""

    settings = _settings(tmp_path)
    database = Database(tmp_path / "sessions.db")
    await database.open()
    try:
        session = await _session(database, tmp_path, sidecar_id="live-parent")
        await CodingSessionStore(database).record_ancestry(
            session.id, ("live-parent", "live-recovery")
        )
        # Referenced by an ACTIVE session: both must survive however old.
        live_parent = _journal(settings, "live-parent", age_days=30)
        live_fork = _journal(settings, "live-recovery", age_days=30)
        # Unreferenced but recent: the mint/commit race window.
        recent_orphan = _journal(settings, "just-minted", age_days=0)
        # Unreferenced and aged: the only thing the sweep may take.
        stale_orphan = _journal(settings, "crashed-long-ago", age_days=3)
        # Not a journal at all.
        decoy = settings.cline_event_journal_dir / "notes.txt"
        decoy.write_text("keep me\n", encoding="utf-8")

        coordinator = _coordinator(settings, database)
        removed = await coordinator.discard_unreferenced_journals(
            older_than=datetime.now(UTC) - timedelta(days=1)
        )

        assert removed == ["crashed-long-ago"]
        assert not stale_orphan.exists()
        assert live_parent.exists()
        assert live_fork.exists()
        assert recent_orphan.exists()
        assert decoy.exists()
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_a_crash_recovered_session_keeps_its_journal(tmp_path: Path) -> None:
    """A failed-but-uncleaned session is recoverable, so it is never swept."""

    settings = _settings(tmp_path)
    database = Database(tmp_path / "sessions.db")
    await database.open()
    try:
        session = await _session(database, tmp_path, sidecar_id="crashed-1")
        store = CodingSessionStore(database)
        await store.record_ancestry(session.id, ("crashed-1", "crashed-recovery"))
        await store.update(
            session.id,
            CodingSessionUpdateV1(
                state=CodingSessionState.FAILED,
                last_error="transport died mid-round",
            ),
        )
        parent = _journal(settings, "crashed-1", age_days=10)
        fork = _journal(settings, "crashed-recovery", age_days=10)

        coordinator = _coordinator(settings, database)
        removed = await coordinator.discard_unreferenced_journals(
            older_than=datetime.now(UTC) - timedelta(days=1)
        )

        assert removed == []
        assert parent.exists()
        assert fork.exists()
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_the_sweep_is_idempotent_and_stays_in_its_own_directory(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    database = Database(tmp_path / "sessions.db")
    await database.open()
    try:
        _journal(settings, "orphan-a", age_days=5)
        _journal(settings, "orphan-b", age_days=5)
        # The SDK's own store sits beside the journal directory and is never
        # a candidate: a sweep that wandered in would delete a transcript a
        # recovery still needs.
        sdk_store = settings.cline_sidecar_data_dir / "db"
        sdk_store.mkdir(parents=True, exist_ok=True)
        sdk_record = sdk_store / "sessions.db"
        sdk_record.write_text("sdk state\n", encoding="utf-8")
        old = datetime.now(UTC).timestamp() - 30 * 86_400
        os.utime(sdk_record, (old, old))

        coordinator = _coordinator(settings, database)
        cutoff = datetime.now(UTC) - timedelta(days=1)
        first = await coordinator.discard_unreferenced_journals(older_than=cutoff)
        second = await coordinator.discard_unreferenced_journals(older_than=cutoff)

        assert sorted(first) == ["orphan-a", "orphan-b"]
        assert second == []
        assert sdk_record.exists()
        assert list(settings.cline_event_journal_dir.iterdir()) == []
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_the_sweep_requires_a_timezone_aware_cutoff(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    database = Database(tmp_path / "sessions.db")
    await database.open()
    try:
        coordinator = _coordinator(settings, database)
        with pytest.raises(ValueError):
            await coordinator.discard_unreferenced_journals(
                older_than=datetime.now()  # noqa: DTZ005 - the point of the test
            )
    finally:
        await database.close()
