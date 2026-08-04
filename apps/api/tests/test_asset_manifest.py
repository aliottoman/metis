"""P1.4: an approved build becomes a launchable asset, written by the host.

Launch requires a reviewed .metis/asset.json; models are prohibited from
writing .metis. These tests pin the resolution: after an approved apply the
framework derives the manifest from what actually reached disk, the asset
library accepts it as a configured launch recipe, an existing manifest is
never overwritten, and the projected environment reaches only assets whose
files actually use the capability.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from waqil_api.asset_library import AssetManager
from waqil_api.config import Settings
from waqil_api.model_provider import DeterministicModelProvider
from waqil_api.project_env import asset_environment, capabilities_of_tree
from waqil_api.project_workspace import ProjectWorkspaceService

FASTAPI_MAIN = (
    "from fastapi import FastAPI\n"
    "from appkit.oci_responses import OciResponses\n\n"
    "app = FastAPI()\n"
)


async def _service(tmp_path: Path, *, with_requirements: bool = True):
    projects_root = tmp_path / "Projects"
    project = projects_root / "ledger"
    (project / "app").mkdir(parents=True)
    (project / "app" / "main.py").write_text(FASTAPI_MAIN, encoding="utf-8")
    (project / "appkit").mkdir()
    (project / "appkit" / "__init__.py").write_text(
        'SCAFFOLD_VERSION = "0.1.0"\n', encoding="utf-8"
    )
    if with_requirements:
        (project / "requirements.txt").write_text(
            "fastapi\nuvicorn\nopenai\nhttpx\noci-genai-auth\n", encoding="utf-8"
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
    return service, assets, discovered[0].id, project, settings


@pytest.mark.asyncio
async def test_manifest_is_written_and_the_library_accepts_it(tmp_path: Path) -> None:
    service, assets, asset_id, project, _ = await _service(tmp_path)
    written = await service.ensure_asset_manifest(asset_id)
    assert written == ".metis/asset.json"

    manifest = json.loads((project / ".metis" / "asset.json").read_text())
    assert manifest["launch"]["command"][0] == "{uv}"
    assert "app.main:app" in manifest["launch"]["command"]
    assert manifest["env"] == [
        "OCI_CONFIG_FILE",
        "OCI_PROFILE",
        "OCI_RESPONSES_BASE_URL",
        "OCI_RESPONSES_MODEL_ID",
        "OCI_RESPONSES_PROJECT_ID",
    ]
    assert manifest["metis"]["scaffold_version"] == "0.1.0"

    # The library, rescanned, sees a configured (not yet approved) launch.
    (view,) = await assets.scan()
    assert view.launch_configured is True
    assert view.launch_approved is False
    assert view.launch_command[0] == "{uv}"


@pytest.mark.asyncio
async def test_manifest_never_overwrites_and_needs_an_entrypoint(tmp_path: Path) -> None:
    service, _, asset_id, project, _ = await _service(tmp_path)
    handwritten = (project / ".metis")
    handwritten.mkdir()
    (handwritten / "asset.json").write_text('{"name": "mine"}', encoding="utf-8")
    assert await service.ensure_asset_manifest(asset_id) == ""
    assert json.loads((project / ".metis" / "asset.json").read_text()) == {"name": "mine"}


@pytest.mark.asyncio
async def test_projects_without_fastapi_stay_manual(tmp_path: Path) -> None:
    service, _, asset_id, project, _ = await _service(tmp_path)
    (project / "app" / "main.py").write_text("print('cli tool')\n", encoding="utf-8")
    assert await service.ensure_asset_manifest(asset_id) == ""
    assert not (project / ".metis" / "asset.json").exists()


@pytest.mark.asyncio
async def test_python_fallback_when_no_requirements_exist(tmp_path: Path) -> None:
    service, _, asset_id, project, _ = await _service(tmp_path, with_requirements=False)
    assert await service.ensure_asset_manifest(asset_id) == ".metis/asset.json"
    manifest = json.loads((project / ".metis" / "asset.json").read_text())
    assert manifest["launch"]["command"][:3] == ["{python}", "-m", "uvicorn"]


@pytest.mark.asyncio
async def test_projection_follows_the_capabilities_on_disk(tmp_path: Path) -> None:
    _, _, _, project, settings = await _service(tmp_path)
    assert capabilities_of_tree(project) == frozenset({"oci_responses"})

    live = Settings(
        _env_file=None,
        allow_oci_responses=True,
        oci_responses_project_id="ocid1.aiproject.oc1.test",
    )
    projected = asset_environment(live, project)
    assert projected["OCI_RESPONSES_PROJECT_ID"] == "ocid1.aiproject.oc1.test"

    # Strip the capability use and the projection follows.
    (project / "app" / "main.py").write_text(
        "from fastapi import FastAPI\napp = FastAPI()\n", encoding="utf-8"
    )
    assert asset_environment(live, project) == {}
