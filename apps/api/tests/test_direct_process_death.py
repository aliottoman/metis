"""What survives when the sidecar dies with a round in flight.

A crash between an accepted edit and its import is the case that decides
whether an agent is trustworthy: the work must not be lost, and it must not be
done twice. Metis writes the mirror and the durable session ahead of the model
call, so recovery reads what is already on disk rather than asking a model to
reconstruct it.

The frozen contract is checked separately because it lives in the graph
checkpoint rather than the coding session — a recovered round that resumed
under a re-resolved envelope would be a different run wearing the same id.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from waqil_api.coding_contracts import CodingProviderV1, CodingSessionState
from waqil_api.project_direct_contract import build_direct_contract, protected_drift

from test_project_coding_engine import _coordinator as _direct_coordinator


def _provider() -> CodingProviderV1:
    return CodingProviderV1(
        providerId="ollama", modelId="glm-5.2:cloud", baseUrl="http://x"
    )


@pytest.mark.asyncio
async def test_a_round_killed_mid_flight_keeps_its_mirror_and_session(
    tmp_path: Path,
) -> None:
    """The accepted edit is on disk in the mirror before the model returns."""

    def write_then_die(root: Path, phase: str) -> None:
        (root / "app" / "main.py").write_text("PORTED = True\n", encoding="utf-8")
        if phase == "start":
            # The process dies after the edit landed and before the result.
            raise asyncio.CancelledError

    (
        coordinator,
        engine,
        sessions,
        projects,
        database,
        run_id,
        conv,
        project_id,
    ) = await _direct_coordinator(tmp_path, write_then_die)
    try:
        engine.cancel_start = True
        with pytest.raises(asyncio.CancelledError):
            await coordinator.start(
                run_id=run_id,
                conversation_id=conv,
                project_id=project_id,
                prompt="port it",
                staged={},
                provider=_provider(),
                allowed_paths=[],
                protected_paths=[],
                broad_scope=True,
                operation_id=f"{run_id}:round:1",
            )

        # The durable session exists and is marked recoverable, and the mirror
        # it names still holds the edit: nothing was reconstructed by a model.
        rows = await sessions.for_run(run_id)
        assert len(rows) == 1, "exactly one session, not a duplicate per attempt"
        session = rows[0]
        # Deliberately NOT terminal: a cancelled round keeps its write-ahead
        # RUNNING row so recovery can still find the mirror it names. Marking
        # it failed here would strand the edit that is sitting on disk.
        assert session.state is CodingSessionState.RUNNING
        assert session.sidecar_session_id
        mirror_root = Path(session.workspace_snapshot.project_root)
        assert (mirror_root / "app" / "main.py").read_text() == "PORTED = True\n"

        # ── recovery: the same mirror is imported exactly once ─────────────
        recovered = await coordinator.recover_for_run(
            run_id,
            project_id=project_id,
            staged={},
            provider=_provider(),
            allowed_paths=[],
            operation_id=f"{run_id}:round:1",
            broad_scope=True,
        )

        assert recovered is not None
        assert "app/main.py" in dict(recovered.staged)
        assert str(recovered.staged["app/main.py"]["content"]) == "PORTED = True\n"
        # Same session ancestry, not a fresh one.
        assert recovered.session.id == session.id
        assert recovered.session.sidecar_session_id == session.sidecar_session_id

        # ── and only once: a second recovery finds nothing left to import ──
        again = await coordinator.recover_for_run(
            run_id,
            project_id=project_id,
            staged=dict(recovered.staged),
            provider=_provider(),
            allowed_paths=[],
            operation_id=f"{run_id}:round:1",
            broad_scope=True,
        )
        assert again is None, "an imported round must not be replayed"

        # No model operation was duplicated: one start request, ever.
        assert len(engine.start_requests) == 1
        assert engine.continue_requests == []
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_the_operation_identity_is_stable_across_the_crash(
    tmp_path: Path,
) -> None:
    """The same operation id, so a retry cannot be mistaken for new work."""

    def write_then_die(root: Path, phase: str) -> None:
        (root / "app" / "main.py").write_text("X = 1\n", encoding="utf-8")
        if phase == "start":
            raise asyncio.CancelledError

    (
        coordinator,
        engine,
        sessions,
        _projects,
        database,
        run_id,
        conv,
        project_id,
    ) = await _direct_coordinator(tmp_path, write_then_die)
    try:
        engine.cancel_start = True
        operation = f"{run_id}:round:1"
        with pytest.raises(asyncio.CancelledError):
            await coordinator.start(
                run_id=run_id,
                conversation_id=conv,
                project_id=project_id,
                prompt="port it",
                staged={},
                provider=_provider(),
                allowed_paths=[],
                protected_paths=[],
                broad_scope=True,
                operation_id=operation,
            )

        recovered = await coordinator.recover_for_run(
            run_id,
            project_id=project_id,
            staged={},
            provider=_provider(),
            allowed_paths=[],
            operation_id=operation,
            broad_scope=True,
        )

        assert recovered is not None
        # The journal that made recovery possible is keyed by the same
        # operation, which is what stops a replay counting as a second round.
        rows = await sessions.for_run(run_id)
        assert len(rows) == 1
        assert rows[0].id == recovered.session.id
    finally:
        await database.close()


def test_the_frozen_contract_survives_a_checkpoint_round_trip(
    tmp_path: Path,
) -> None:
    """Recovery must resume the admitted envelope, not re-resolve one.

    The contract lives in the graph checkpoint. If a crash caused it to be
    resolved again, a file edited on disk in between would silently change
    what the resumed round may touch.
    """

    project = tmp_path / "logivity"
    (project / "assets").mkdir(parents=True)
    for relative, body in (
        ("extractor.py", "OCI = 1\n"),
        ("excel_writer.py", "COL = 'A'\n"),
        ("assets/DHL_Template.xlsx", "template\n"),
        ("app.py", "APP = 1\n"),
    ):
        (project / relative).write_text(body, encoding="utf-8")

    contract, _ = build_direct_contract(
        prompt="port it. do not touch extractor.py, excel_writer.py or the DHL template",
        project=project,
        tree=[
            "extractor.py",
            "excel_writer.py",
            "assets/DHL_Template.xlsx",
            "app.py",
        ],
        max_iterations=60,
    )
    # Exactly what the checkpoint carries and gives back.
    carried = json.loads(json.dumps(contract.as_state()))

    # The crash window: a protected file changes on disk before recovery.
    (project / "extractor.py").write_text(
        "edited during the outage\n", encoding="utf-8"
    )

    assert carried["protected_files"] == list(contract.protected_files)
    assert carried["protected_hashes"] == dict(contract.protected_hashes)
    assert carried["max_iterations"] == 60
    # The drift is therefore detectable rather than absorbed into a new
    # baseline, which is the whole reason the hashes are frozen.
    assert protected_drift(project, carried["protected_hashes"]) == ["extractor.py"]


@pytest.mark.asyncio
async def test_a_crash_before_any_edit_leaves_nothing_to_import(
    tmp_path: Path,
) -> None:
    """Recovery invents nothing when there was nothing to recover."""

    def die_immediately(root: Path, phase: str) -> None:
        if phase == "start":
            raise asyncio.CancelledError

    (
        coordinator,
        engine,
        _sessions,
        _projects,
        database,
        run_id,
        conv,
        project_id,
    ) = await _direct_coordinator(tmp_path, die_immediately)
    try:
        engine.cancel_start = True
        with pytest.raises(asyncio.CancelledError):
            await coordinator.start(
                run_id=run_id,
                conversation_id=conv,
                project_id=project_id,
                prompt="port it",
                staged={},
                provider=_provider(),
                allowed_paths=[],
                protected_paths=[],
                broad_scope=True,
            )

        recovered = await coordinator.recover_for_run(
            run_id,
            project_id=project_id,
            staged={},
            provider=_provider(),
            allowed_paths=[],
            operation_id=f"{run_id}:round:1",
        )

        assert recovered is None or not dict(recovered.staged)
    finally:
        await database.close()


def _unused(value: Any) -> Any:  # pragma: no cover - keeps imports honest
    return value
