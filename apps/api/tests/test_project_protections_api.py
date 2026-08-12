"""Persistent per-project protections: durable, previewable, never model-set.

Task protections come from the words in one request. These are the ones that
outlive it — the files a person has decided are off limits in this project
whatever anyone asks for next. They are stored as identities and patterns, not
bytes, so every run re-freezes the hashes and a file edited between runs
cannot inherit a stale one.

The important negative: no project tool writes here. A coding session cannot
widen its own envelope by editing settings, because the only way in is the
HTTP surface a person uses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
import pytest

from waqil_api.config import Settings
from waqil_api.main import create_app
from waqil_api.project_capability_eval import project_asset_id
from waqil_api.project_direct_contract import build_direct_contract
from waqil_api.project_tools import unrestricted_project_tools


TREE = [
    "app.py",
    "config.py",
    "extractor.py",
    "excel_writer.py",
    "assets/DHL_Template.xlsx",
]


def _seed(root: Path) -> None:
    for relative in TREE:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"# {relative}\n", encoding="utf-8")


async def _client(tmp_path: Path) -> AsyncIterator[tuple[Any, str, Path]]:
    projects_root = tmp_path / "Projects"
    project = projects_root / "demo"
    project.mkdir(parents=True)
    _seed(project)
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        data_dir=tmp_path / "data",
        repo_root=tmp_path,
        asset_roots=[projects_root],
        model_backend="deterministic",
        allow_test_backends=True,
        reference_runner_mode="deterministic",
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://metis.test"
        ) as client:
            assets = (await client.post("/api/v1/assets/scan")).json()
            project_id = project_asset_id(assets, project)
            await client.post(
                f"/api/v1/projects/{project_id}/open",
                json={"mode": "grok_bootstrap_local"},
            )
            yield client, project_id, project


@pytest.mark.asyncio
async def test_protections_round_trip_and_resolve(tmp_path: Path) -> None:
    async for client, project_id, project in _client(tmp_path):
        empty = (await client.get(f"/api/v1/projects/{project_id}/protections")).json()
        assert empty["protected_files"] == []

        saved = await client.put(
            f"/api/v1/projects/{project_id}/protections",
            json={"protected_files": ["extractor.py", "assets/*.xlsx"]},
        )
        assert saved.status_code == 200
        body = saved.json()
        assert body["protected_files"] == ["assets/*.xlsx", "extractor.py"]
        # The preview is what makes a pattern safe to set: a person sees what
        # it currently matches before they rely on it.
        assert body["resolved"] == ["assets/DHL_Template.xlsx", "extractor.py"]
        assert body["unmatched"] == []

        again = (await client.get(f"/api/v1/projects/{project_id}/protections")).json()
        assert again["protected_files"] == ["assets/*.xlsx", "extractor.py"]
        assert (project / ".metis" / "project-settings.json").is_file(), (
            "settings must be durable on disk"
        )


@pytest.mark.asyncio
async def test_a_pattern_that_matches_nothing_is_shown_as_unmatched(
    tmp_path: Path,
) -> None:
    async for client, project_id, _project in _client(tmp_path):
        body = (
            await client.put(
                f"/api/v1/projects/{project_id}/protections",
                json={"protected_files": ["billing_*.py"]},
            )
        ).json()

        # Saved, but visibly resolving to nothing: a person can correct it
        # before a run relies on it.
        assert body["protected_files"] == ["billing_*.py"]
        assert body["resolved"] == []
        assert body["unmatched"] == ["billing_*.py"]


@pytest.mark.asyncio
async def test_a_traversal_or_absolute_path_is_refused(tmp_path: Path) -> None:
    async for client, project_id, _project in _client(tmp_path):
        for candidate in ("../outside.py", "/etc/passwd"):
            response = await client.put(
                f"/api/v1/projects/{project_id}/protections",
                json={"protected_files": [candidate]},
            )
            assert response.status_code == 409, candidate
            assert "inside the project" in response.json()["detail"]


@pytest.mark.asyncio
async def test_a_corrupt_settings_file_protects_nothing_rather_than_failing(
    tmp_path: Path,
) -> None:
    async for client, project_id, project in _client(tmp_path):
        (project / ".metis" / "project-settings.json").write_text(
            "{not json", encoding="utf-8"
        )

        body = (await client.get(f"/api/v1/projects/{project_id}/protections")).json()

        # Opening a project must not fail because a settings file is damaged.
        # It protects nothing, and the resolved contract is emitted visibly
        # before coding either way.
        assert body["protected_files"] == []


@pytest.mark.asyncio
async def test_persistent_and_task_protections_union_at_admission(
    tmp_path: Path,
) -> None:
    async for client, project_id, project in _client(tmp_path):
        await client.put(
            f"/api/v1/projects/{project_id}/protections",
            json={"protected_files": ["config.py"]},
        )
        stored = json.loads(
            (project / ".metis" / "project-settings.json").read_text(encoding="utf-8")
        )["protected_files"]

        contract, resolution = build_direct_contract(
            prompt="port it. do not touch extractor.py or the DHL template",
            project=project,
            tree=TREE,
            project_protected=stored,
        )

        assert resolution.needs_user is False
        assert contract.task_protected == (
            "assets/DHL_Template.xlsx",
            "extractor.py",
        )
        assert contract.project_protected == ("config.py",)
        assert set(contract.protected_files) == {
            "assets/DHL_Template.xlsx",
            "extractor.py",
            "config.py",
        }
        # Hashes are frozen for the union, at admission, not when the setting
        # was saved.
        assert set(contract.protected_hashes) == set(contract.protected_files)


def test_no_project_tool_can_write_these_settings() -> None:
    """The negative that makes the setting a boundary rather than a suggestion."""

    names = {str(tool.get("name") or "") for tool in unrestricted_project_tools()}

    assert names, "the tool catalogue should not be empty"
    # No tool names protections, settings, or anything that could reach them.
    for name in names:
        assert "protect" not in name, name
        assert "setting" not in name, name
    # The only writer is the HTTP route, which a model never calls.
    assert "write_project_settings" not in names
