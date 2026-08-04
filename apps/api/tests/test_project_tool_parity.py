"""Every advertised project tool is callable; nothing callable is hidden.

inspect_api spent its life implemented in the workspace, described in the
catalog, advertised to Grok — and impossible for any provider to actually
call, because two hand-copied name lists lagged one tool behind. These tests
make that class of drift a test failure: one canonical roster, every other
enumeration pinned to it, and a live dispatch through the workspace.
"""

from __future__ import annotations

import typing
from pathlib import Path

import pytest

from waqil_api.asset_library import AssetManager
from waqil_api.config import Settings
from waqil_api.contracts import (
    PROJECT_TOOL_OPTIONAL_ARGUMENTS,
    PROJECT_TOOL_REQUIRED_ARGUMENTS,
    ProjectAgentStepWireV1,
    ProjectBuildStepWireV1,
    ProjectToolCallV1,
    project_tool_catalog,
)
from waqil_api.model_provider import (
    _PROJECT_TOOL_NAMES,
    DeterministicModelProvider,
    OCIResponsesModelProvider,
)
from waqil_api.project_workspace import ProjectWorkspaceError, ProjectWorkspaceService

ROSTER = frozenset(PROJECT_TOOL_REQUIRED_ARGUMENTS)


def _literal_values(model: type, field: str) -> frozenset[str]:
    annotation = model.model_fields[field].annotation
    return frozenset(value for value in typing.get_args(annotation) if value != "")


def test_every_enumeration_matches_the_canonical_roster() -> None:
    assert _literal_values(ProjectToolCallV1, "name") == ROSTER
    assert _literal_values(ProjectAgentStepWireV1, "tool") == ROSTER
    assert _literal_values(ProjectBuildStepWireV1, "tool") == ROSTER
    assert frozenset(_PROJECT_TOOL_NAMES) == ROSTER
    assert frozenset(PROJECT_TOOL_OPTIONAL_ARGUMENTS) == ROSTER
    assert {entry["name"] for entry in project_tool_catalog()} == ROSTER


def test_grok_is_advertised_exactly_the_roster_plus_finish() -> None:
    provider = OCIResponsesModelProvider(
        Settings(
            _env_file=None,
            allow_oci_responses=True,
            oci_responses_project_id="ocid1.aiproject.oc1.test",
        )
    )
    advertised = {tool["name"] for tool in provider._unrestricted_project_tools()}
    # finish_project_task is the OCI completion channel, not a workspace tool.
    assert advertised == ROSTER | {"finish_project_task"}


def test_an_invented_tool_is_rejected_at_the_contract() -> None:
    with pytest.raises(Exception):
        ProjectToolCallV1(name="delete_everything", arguments={})


@pytest.mark.asyncio
async def test_every_roster_tool_dispatches_in_the_workspace(tmp_path: Path) -> None:
    """The other half of parity: names the contract accepts, the workspace runs.

    run_check is special-cased — against a staged overlay it must refuse with
    its own reason (checks run on disk), which still proves the dispatcher
    knows it; "unsupported project tool" would mean it fell off the roster.
    """
    projects_root = tmp_path / "Projects"
    project = projects_root / "demo"
    project.mkdir(parents=True)
    (project / "main.py").write_text("VALUE = 1\n", encoding="utf-8")
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        repo_root=tmp_path,
        asset_roots=[projects_root],
        model_backend="deterministic",
        allow_test_backends=True,
    )
    assets = AssetManager(
        settings.asset_roots,
        approval_path=settings.asset_approval_path,
        catalog_path=settings.asset_catalog_path,
    )
    discovered = await assets.scan()
    service = ProjectWorkspaceService(settings, assets, DeterministicModelProvider())
    asset_id = discovered[0].id

    calls: dict[str, dict[str, object]] = {
        "list_files": {},
        "search_code": {"query": "VALUE"},
        "read_file": {"path": "main.py"},
        "inspect_api": {"module": "json", "symbol": "dumps"},
        "create_file": {"path": "extra.py", "content": "EXTRA = 2\n"},
        "apply_patch": {
            "path": "main.py",
            "original": "VALUE = 1",
            "replacement": "VALUE = 2",
        },
    }
    staged: dict[str, dict[str, object]] = {}
    for name, arguments in calls.items():
        result, overlay = await service.execute_staged(
            asset_id,
            ProjectToolCallV1(name=name, arguments=arguments),  # type: ignore[arg-type]
            staged,
        )
        assert result is not None, name
        if overlay is not None:
            staged = overlay

    with pytest.raises(ProjectWorkspaceError) as refused:
        await service.execute_staged(
            asset_id,
            ProjectToolCallV1(name="run_check", arguments={"name": "tests"}),
            staged,
        )
    assert "unsupported project tool" not in str(refused.value)
