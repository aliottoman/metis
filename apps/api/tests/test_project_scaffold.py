"""The deterministic scaffold: what gets vendored, staged, refused, applied.

The invariants pinned here are the P0 contract: the vendored bytes are the
tested bytes, seeding is idempotent and disk-respecting, models cannot write
under appkit/ while the framework can, and the prompt note describes exactly
what the project actually carries.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from waqil_api.asset_library import AssetManager
from waqil_api.config import Settings
from waqil_api.contracts import ProjectToolCallV1
from waqil_api.model_provider import DeterministicModelProvider
from waqil_api.project_scaffold import (
    scaffold_prompt,
    scaffold_sources,
    wants_oci_responses,
)
from waqil_api.project_workspace import ProjectWorkspaceError, ProjectWorkspaceService


async def _service(tmp_path: Path) -> tuple[ProjectWorkspaceService, str, Path]:
    projects_root = tmp_path / "Projects"
    project = projects_root / "ledger"
    (project / "app").mkdir(parents=True)
    (project / "app" / "main.py").write_text(
        "from fastapi import FastAPI\n\napp = FastAPI()\n", encoding="utf-8"
    )
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
    return service, discovered[0].id, project


def test_scaffold_sources_gate_the_oci_module_and_env_example() -> None:
    base = scaffold_sources(frozenset())
    assert set(base) == {
        "appkit/__init__.py",
        "appkit/config.py",
        "appkit/money.py",
        "appkit/uploads.py",
    }
    with_oci = scaffold_sources({"oci_responses"})
    assert "appkit/oci_responses.py" in with_oci
    assert ".env.example" in with_oci
    # The vendored bytes are the tested bytes — and they parse.
    for path, content in with_oci.items():
        if path.endswith(".py"):
            ast.parse(content, filename=path)
    # The env example documents names only, never values.
    for line in with_oci[".env.example"].splitlines():
        if "=" in line and not line.startswith("#"):
            assert line.endswith("=")


def test_wants_oci_responses_reads_extraction_intent() -> None:
    assert wants_oci_responses(
        "Build an app where I upload supplier invoice images and Grok extracts "
        "the structured fields"
    )
    assert wants_oci_responses("Create a tool that does OCR on receipts")
    assert not wants_oci_responses("Build a todo list web app with FastAPI")
    assert not wants_oci_responses("Create a photo gallery site with albums")


@pytest.mark.asyncio
async def test_stage_scaffold_seeds_once_and_respects_disk(tmp_path: Path) -> None:
    service, asset_id, project = await _service(tmp_path)
    # A file already on disk is never re-staged: this project carries an old
    # (hand-written) money module, so seeding must leave that path alone.
    (project / "appkit").mkdir()
    (project / "appkit" / "money.py").write_text("LEGACY = True\n", encoding="utf-8")

    staged, added = await service.stage_scaffold(asset_id, {}, {"oci_responses"})
    assert "appkit/money.py" not in added
    assert "appkit/oci_responses.py" in added
    assert ".env.example" in added
    entry = staged["appkit/oci_responses.py"]
    assert entry["origin"] == "create"
    assert entry["base_sha256"] == ""
    assert entry["bytes"] > 0

    again, added_again = await service.stage_scaffold(asset_id, staged, {"oci_responses"})
    assert added_again == []
    assert again == staged


@pytest.mark.asyncio
async def test_model_writes_under_appkit_are_refused_reads_pass(tmp_path: Path) -> None:
    service, asset_id, _ = await _service(tmp_path)
    staged, _ = await service.stage_scaffold(asset_id, {}, set())

    with pytest.raises(ProjectWorkspaceError, match="Metis-owned"):
        await service.execute_staged(
            asset_id,
            ProjectToolCallV1(
                name="create_file",
                arguments={"path": "appkit/mine.py", "content": "x = 1\n"},
            ),
            staged,
        )
    with pytest.raises(ProjectWorkspaceError, match="Metis-owned"):
        await service.execute_staged(
            asset_id,
            ProjectToolCallV1(
                name="apply_patch",
                arguments={
                    "path": "appkit/money.py",
                    "original": "CENT",
                    "replacement": "PENNY",
                },
            ),
            staged,
        )
    # Reading the scaffold is how the model learns its contracts.
    result, overlay = await service.execute_staged(
        asset_id,
        ProjectToolCallV1(name="read_file", arguments={"path": "appkit/money.py"}),
        staged,
    )
    assert overlay is None
    assert "within_percent" in result["content"]


@pytest.mark.asyncio
async def test_materialize_applies_scaffold_entries(tmp_path: Path) -> None:
    service, asset_id, project = await _service(tmp_path)
    staged, added = await service.stage_scaffold(asset_id, {}, set())
    outcome = await service.materialize_staged(asset_id, staged)
    assert sorted(outcome["applied"]) == sorted(added)
    assert outcome["skipped"] == []
    assert (project / "appkit" / "money.py").is_file()
    assert (project / "appkit" / "__init__.py").is_file()


def test_scaffold_prompt_describes_what_the_project_carries() -> None:
    assert scaffold_prompt({}, {}) == ""
    base_note = scaffold_prompt({"appkit/__init__.py": {}}, {})
    assert "appkit.money" in base_note
    assert "OCI_RESPONSES_PROJECT_ID" not in base_note
    oci_note = scaffold_prompt(
        {}, {"manifest": {"file_tree": ["appkit/oci_responses.py", "app/main.py"]}}
    )
    assert "appkit.oci_responses" in oci_note
    assert "OCI_RESPONSES_PROJECT_ID" in oci_note
    assert "never" in oci_note.lower()
