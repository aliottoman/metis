from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from waqil_api.asset_library import AssetManager
from waqil_api.config import Settings
from waqil_api.contracts import ProjectToolCallV1
from waqil_api.model_provider import DeterministicModelProvider
from waqil_api.main import create_app
from waqil_api.project_workspace import ProjectWorkspaceError, ProjectWorkspaceService


@pytest.mark.asyncio
async def test_project_workspace_bootstraps_tools_and_evolving_notes(tmp_path: Path) -> None:
    projects_root = tmp_path / "Projects"
    project = projects_root / "demo"
    source = project / "src"
    source.mkdir(parents=True)
    (project / "README.md").write_text("# Demo\nA small typed service.\n", encoding="utf-8")
    (project / "package.json").write_text(
        json.dumps({"name": "demo", "scripts": {"test": "node --test"}}),
        encoding="utf-8",
    )
    (source / "main.ts").write_text(
        "export function greet(name: string) {\n  return `Hello ${name}`;\n}\n",
        encoding="utf-8",
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

    opened = await service.open(discovered[0].id)
    assert opened.initialized is True
    assert opened.file_count == 3
    assert (project / ".metis" / "project-context.json").is_file()
    notes = project / ".metis" / "METIS.md"
    assert "Metis project context" in notes.read_text(encoding="utf-8")

    listing = await service.execute(
        opened.id, ProjectToolCallV1(name="list_files", arguments={"path": "src"})
    )
    assert listing["files"] == ["src/main.ts"]
    search = await service.execute(
        opened.id, ProjectToolCallV1(name="search_code", arguments={"query": "greet"})
    )
    assert search["matches"][0]["path"] == "src/main.ts"
    read = await service.execute(
        opened.id,
        ProjectToolCallV1(
            name="read_file", arguments={"path": "src/main.ts", "start_line": 1}
        ),
    )
    assert "Hello ${name}" in read["content"]

    patch_call = ProjectToolCallV1(
        name="apply_patch",
        arguments={
            "path": "src/main.ts",
            "original": "Hello ${name}",
            "replacement": "Welcome ${name}",
        },
    )
    preview = await service.preview(opened.id, patch_call)
    assert preview["path"] == "src/main.ts"
    result = await service.execute(opened.id, patch_call)
    assert result["changed"] is True
    assert "Welcome" in (source / "main.ts").read_text(encoding="utf-8")

    await service.record_learnings(
        opened.id,
        "run_test",
        ["The entrypoint is src/main.ts.", "API_KEY=do-not-store"],
    )
    note_text = notes.read_text(encoding="utf-8")
    assert "The entrypoint is src/main.ts." in note_text
    assert "do-not-store" not in note_text


@pytest.mark.asyncio
async def test_project_tools_fail_closed_on_internal_and_secret_paths(tmp_path: Path) -> None:
    projects_root = tmp_path / "Projects"
    project = projects_root / "demo"
    project.mkdir(parents=True)
    (project / "README.md").write_text("# Demo\n", encoding="utf-8")
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        repo_root=tmp_path,
        asset_roots=[projects_root],
        model_backend="deterministic",
        allow_test_backends=True,
    )
    assets = AssetManager(settings.asset_roots, catalog_path=settings.asset_catalog_path)
    project_id = (await assets.scan())[0].id
    service = ProjectWorkspaceService(settings, assets, DeterministicModelProvider())
    await service.open(project_id)

    for relative in ("../outside.txt", ".env", ".metis/METIS.md", ".git/config"):
        with pytest.raises(ProjectWorkspaceError):
            await service.execute(
                project_id,
                ProjectToolCallV1(name="create_file", arguments={"path": relative, "content": "x"}),
            )


def test_project_chat_pins_mode_and_uses_project_agent_path(tmp_path: Path) -> None:
    projects_root = tmp_path / "Projects"
    project = projects_root / "demo"
    project.mkdir(parents=True)
    (project / "README.md").write_text("# Demo\n", encoding="utf-8")
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        repo_root=tmp_path,
        asset_roots=[projects_root],
        model_backend="deterministic",
        reference_runner_mode="deterministic",
        allow_test_backends=True,
    )
    with TestClient(create_app(settings)) as client:
        project_id = client.post("/api/v1/assets/scan").json()[0]["id"]
        opened = client.post(
            f"/api/v1/projects/{project_id}/open",
            json={"mode": "grok_bootstrap_local"},
        )
        assert opened.status_code == 200
        conversation_id = client.post("/api/v1/conversations", json={}).json()["id"]
        accepted = client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json={
                "content": "Explain the entrypoint.",
                "project_id": project_id,
                "project_mode": "grok_bootstrap_local",
            },
        )
        assert accepted.status_code == 202
        run_id = accepted.json()["run_id"]
        for _ in range(100):
            run = client.get(f"/api/v1/runs/{run_id}").json()
            if run["status"] in {"completed", "failed"}:
                break
            time.sleep(0.01)
        assert run["status"] == "completed"
        session = client.get(
            f"/api/v1/conversations/{conversation_id}/project"
        ).json()
        assert session == {
            "conversation_id": conversation_id,
            "project_id": project_id,
            "mode": "grok_bootstrap_local",
            "updated_at": session["updated_at"],
        }
        messages = client.get(
            f"/api/v1/conversations/{conversation_id}/messages"
        ).json()
        assert messages[-1]["role"] == "assistant"
        assert "deterministic project response" in messages[-1]["content"]


def test_project_write_waits_for_exact_approval_then_resumes(tmp_path: Path) -> None:
    projects_root = tmp_path / "Projects"
    project = projects_root / "demo"
    project.mkdir(parents=True)
    (project / "README.md").write_text("# Demo\n", encoding="utf-8")
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        repo_root=tmp_path,
        asset_roots=[projects_root],
        model_backend="deterministic",
        reference_runner_mode="deterministic",
        allow_test_backends=True,
    )
    with TestClient(create_app(settings)) as client:
        project_id = client.post("/api/v1/assets/scan").json()[0]["id"]
        assert client.post(
            f"/api/v1/projects/{project_id}/open",
            json={"mode": "grok_bootstrap_local"},
        ).status_code == 200
        conversation_id = client.post("/api/v1/conversations", json={}).json()["id"]
        run_id = client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json={
                "content": "[project-create-test]",
                "project_id": project_id,
                "project_mode": "grok_bootstrap_local",
            },
        ).json()["run_id"]
        for _ in range(100):
            run = client.get(f"/api/v1/runs/{run_id}").json()
            if run["status"] == "awaiting_approval":
                break
            time.sleep(0.01)
        assert run["status"] == "awaiting_approval"
        assert not (project / "generated.txt").exists()
        recoverable = client.get("/api/v1/runs?status=awaiting_approval").json()
        approval = next(item["approval"] for item in recoverable if item["run"]["id"] == run_id)
        assert approval["kind"] == "project_write"
        decided = client.post(
            f"/api/v1/runs/{run_id}/decisions",
            json={"approval_id": approval["id"], "decision": "approve"},
        )
        assert decided.status_code == 200
        for _ in range(100):
            run = client.get(f"/api/v1/runs/{run_id}").json()
            if run["status"] in {"completed", "failed"}:
                break
            time.sleep(0.01)
        assert run["status"] == "completed"
        assert (project / "generated.txt").read_text(encoding="utf-8").startswith("created")
        notes = (project / ".metis" / "METIS.md").read_text(encoding="utf-8")
        assert "Approved `create_file` on `generated.txt`" in notes
