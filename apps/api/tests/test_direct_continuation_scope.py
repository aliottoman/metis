"""A direct run's repair round is admitted under the contract it started with.

The live failure this pins: GLM implemented the whole feature, ran host checks,
got a real failing test back, and repaired it in the same session until
`imports`, `pytest`, `ruff` and `full` were all clean. Metis then refused the
repair -- "the coding engine changed files outside the host plan: app/main.py,
app/store.py" -- because `_continue_session` was the one `_run_and_import` path
that did not forward `broad_scope`. A direct run is admitted with no allowlist,
so with broad scope lost, every file it touched was "outside the plan".

The approval card therefore carried round one's failing bytes while the correct
repair was discarded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from waqil_api.asset_library import AssetManager
from waqil_api.coding_contracts import CodingProviderV1
from waqil_api.coding_engine import CodingSessionStore
from waqil_api.config import Settings
from waqil_api.database import Database
from waqil_api.model_provider import DeterministicModelProvider
from waqil_api.project_coding_engine import ProjectCodingCoordinator, ProjectCodingError
from waqil_api.project_workspace import ProjectWorkspaceService

from test_project_coding_engine import RecordingEngine, RecordingEvents


async def _coordinator(tmp_path: Path, mutate: Any) -> tuple[Any, ...]:
    projects_root = tmp_path / "Projects"
    project = projects_root / "demo"
    (project / "app").mkdir(parents=True)
    (project / "app" / "main.py").write_text("VALUE = 1\n", encoding="utf-8")
    (project / "app" / "store.py").write_text("STORE = 1\n", encoding="utf-8")
    (project / "extractor.py").write_text("EXTRACT = 1\n", encoding="utf-8")
    (project / "README.md").write_text("# Demo\n", encoding="utf-8")
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        data_dir=tmp_path / "data",
        repo_root=tmp_path,
        asset_roots=[projects_root],
        model_backend="deterministic",
        allow_test_backends=True,
        max_output_tokens=4_096,
        cline_sidecar_max_iterations=7,
    )
    settings.prepare_directories()
    assets = AssetManager(
        settings.asset_roots,
        approval_path=settings.asset_approval_path,
        catalog_path=settings.asset_catalog_path,
    )
    scanned = await assets.scan()
    project_id: str = next(iter(scanned)).id
    projects = ProjectWorkspaceService(settings, assets, DeterministicModelProvider())
    database = Database(tmp_path / "sessions.db")
    await database.open()
    conversation = await database.create_conversation("Direct continuation")
    message = await database.add_message(conversation.id, "user", "Build it")
    run = await database.create_run(
        conversation.id,
        message.id,
        graph_schema_version="1",
        model_aliases={},
    )
    engine = RecordingEngine(mutate)
    coordinator = ProjectCodingCoordinator(
        settings, engine, CodingSessionStore(database), projects, RecordingEvents()
    )
    return coordinator, engine, database, run.id, conversation.id, project_id


def _provider() -> CodingProviderV1:
    return CodingProviderV1(
        providerId="openai-compatible",
        modelId="glm-5.2",
        apiKey=SecretStr("ephemeral-test-key"),
        baseUrl="https://models.example.test/v1",
    )


@pytest.mark.asyncio
async def test_a_direct_repair_round_is_imported_under_the_frozen_contract(
    tmp_path: Path,
) -> None:
    """Round one stages several files; round two repairs them and is accepted."""

    def mutate(root: Path, operation: str) -> None:
        if operation == "start":
            # A realistic vertical feature, and a test that fails.
            (root / "app" / "main.py").write_text(
                "VALUE = 1\nSTATUS_COUNTS = True\n", encoding="utf-8"
            )
            (root / "app" / "store.py").write_text(
                "STORE = 1\ndef counts():\n    return {}\n", encoding="utf-8"
            )
            (root / "tests").mkdir(exist_ok=True)
            (root / "tests" / "test_counts.py").write_text(
                "def test_counts():\n    assert False\n", encoding="utf-8"
            )
        else:
            # The same session repairing files it already owns.
            (root / "app" / "main.py").write_text(
                "VALUE = 1\nSTATUS_COUNTS = True\nFIXED = True\n", encoding="utf-8"
            )
            (root / "app" / "store.py").write_text(
                "STORE = 1\ndef counts():\n    return {'paid': 1}\n", encoding="utf-8"
            )

    (
        coordinator,
        _engine,
        database,
        run_id,
        conversation_id,
        project_id,
    ) = await _coordinator(tmp_path, mutate)
    try:
        started = await coordinator.start(
            run_id=run_id,
            conversation_id=conversation_id,
            project_id=project_id,
            prompt="Add invoice-status filtering.",
            staged={},
            provider=_provider(),
            allowed_paths=[],
            protected_paths=["extractor.py"],
            broad_scope=True,
        )
        first = sorted(started.staged)
        assert first == ["app/main.py", "app/store.py", "tests/test_counts.py"]
        assert "assert False" in started.staged["tests/test_counts.py"]["content"]

        # ── the regression: the SAME session repairs those files ───────────
        repaired = await coordinator.continue_session(
            started.session.id,
            prompt="Verification found a failing test. Repair it.",
            staged=dict(started.staged),
            provider=_provider(),
            allowed_paths=[],
            operation_id=f"{run_id}:round:2",
            protected_paths=["extractor.py"],
            broad_scope=True,
        )

        # Under the defect this raised "changed files outside the host plan".
        assert repaired.rejection_reason == ""
        assert repaired.engine_error == ""
        # Independent verification reads THESE bytes: the repair is what is
        # staged, not round one's failing version.
        assert "FIXED = True" in repaired.staged["app/main.py"]["content"]
        assert "'paid': 1" in repaired.staged["app/store.py"]["content"]
        # The earlier file is still present and untouched by the repair.
        assert "assert False" in repaired.staged["tests/test_counts.py"]["content"]
        # Nothing reached the user's project: staged only.
        project_root = await coordinator.projects.assets.project_path(project_id)
        assert (project_root / "app" / "main.py").read_text(
            encoding="utf-8"
        ) == "VALUE = 1\n"
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_a_protected_change_in_the_same_continuation_is_still_refused(
    tmp_path: Path,
) -> None:
    """Broad scope widens where writes may land, never what may be written."""

    def mutate(root: Path, operation: str) -> None:
        if operation == "start":
            (root / "app" / "main.py").write_text("VALUE = 2\n", encoding="utf-8")
        else:
            # A legitimate repair AND a protected file in one diff.
            (root / "app" / "main.py").write_text("VALUE = 3\n", encoding="utf-8")
            (root / "extractor.py").write_text("EXTRACT = 999\n", encoding="utf-8")

    (
        coordinator,
        _engine,
        database,
        run_id,
        conversation_id,
        project_id,
    ) = await _coordinator(tmp_path, mutate)
    try:
        started = await coordinator.start(
            run_id=run_id,
            conversation_id=conversation_id,
            project_id=project_id,
            prompt="Add invoice-status filtering.",
            staged={},
            provider=_provider(),
            allowed_paths=[],
            protected_paths=["extractor.py"],
            broad_scope=True,
        )
        with pytest.raises(ProjectCodingError) as refusal:
            await coordinator.continue_session(
                started.session.id,
                prompt="Repair it.",
                staged=dict(started.staged),
                provider=_provider(),
                allowed_paths=[],
                operation_id=f"{run_id}:round:2",
                protected_paths=["extractor.py"],
                broad_scope=True,
            )
        assert "extractor.py" in str(refusal.value)
        # All-or-nothing: the good half of that diff is not kept either.
        project_root = await coordinator.projects.assets.project_path(project_id)
        assert (project_root / "extractor.py").read_text(
            encoding="utf-8"
        ) == "EXTRACT = 1\n"
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_a_planner_slice_continuation_stays_allowlist_scoped(
    tmp_path: Path,
) -> None:
    """The frozen legacy path keeps exactly the restriction it always had."""

    def mutate(root: Path, operation: str) -> None:
        if operation == "start":
            (root / "app" / "main.py").write_text("VALUE = 2\n", encoding="utf-8")
        else:
            # Outside the slice's allowlist.
            (root / "app" / "store.py").write_text("STORE = 99\n", encoding="utf-8")

    (
        coordinator,
        _engine,
        database,
        run_id,
        conversation_id,
        project_id,
    ) = await _coordinator(tmp_path, mutate)
    try:
        started = await coordinator.start(
            run_id=run_id,
            conversation_id=conversation_id,
            project_id=project_id,
            prompt="Slice one.",
            staged={},
            provider=_provider(),
            allowed_paths=["app/main.py"],
            broad_scope=False,
        )
        assert sorted(started.staged) == ["app/main.py"]
        with pytest.raises(ProjectCodingError) as refusal:
            await coordinator.continue_session(
                started.session.id,
                prompt="Continue the slice.",
                staged=dict(started.staged),
                provider=_provider(),
                allowed_paths=["app/main.py"],
                operation_id=f"{run_id}:round:2",
                broad_scope=False,
            )
        assert "outside the host plan" in str(refusal.value)
        assert "app/store.py" in str(refusal.value)
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_broad_scope_is_never_inferred_from_an_empty_allowlist(
    tmp_path: Path,
) -> None:
    """Empty means a read-only sliced round -- the opposite of broad."""

    def mutate(root: Path, operation: str) -> None:
        del operation
        (root / "app" / "main.py").write_text("VALUE = 5\n", encoding="utf-8")

    (
        coordinator,
        _engine,
        database,
        run_id,
        conversation_id,
        project_id,
    ) = await _coordinator(tmp_path, mutate)
    try:
        with pytest.raises(ProjectCodingError) as refusal:
            await coordinator.start(
                run_id=run_id,
                conversation_id=conversation_id,
                project_id=project_id,
                prompt="Read-only round.",
                staged={},
                provider=_provider(),
                allowed_paths=[],
                broad_scope=False,
            )
        assert "outside the host plan" in str(refusal.value)
    finally:
        await database.close()
