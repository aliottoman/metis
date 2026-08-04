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

    for relative in (
        "../outside.txt", ".env", ".env.local", ".env.production",
        "server.key", ".metis/METIS.md", ".git/config",
    ):
        with pytest.raises(ProjectWorkspaceError):
            await service.execute(
                project_id,
                ProjectToolCallV1(name="create_file", arguments={"path": relative, "content": "x"}),
            )

    # …but an example env file names variables instead of holding them, and
    # refusing it was a silent, total failure: ten measured builds across three
    # models were each asked for `.env.example`, planned it, tried to write it,
    # and were refused — which read as the models forgetting a file.
    for relative in (".env.example", ".env.sample", ".env.template"):
        result = await service.execute(
            project_id,
            ProjectToolCallV1(
                name="create_file",
                arguments={"path": relative, "content": "OCI_COMPARTMENT_ID=\n"},
            ),
        )
        assert result["path"] == relative


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
        # A single write now rides the same staged-build gate as a large one:
        # the card lists the whole (here one-file) changeset.
        assert approval["kind"] == "project_apply_build"
        assert "generated.txt" in approval["summary"]
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
        assert "Approved a staged build touching 1 file(s)" in notes


async def _service_for(tmp_path: Path) -> tuple[ProjectWorkspaceService, str]:
    projects_root = tmp_path / "Projects"
    project = projects_root / "demo"
    (project / "app").mkdir(parents=True)
    (project / "README.md").write_text("# Demo\n", encoding="utf-8")
    (project / "app" / "main.py").write_text("print('hi')\n", encoding="utf-8")
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
    return service, discovered[0].id


@pytest.mark.asyncio
@pytest.mark.parametrize("spelling", ["", ".", "./", "/", "*"])
async def test_root_spellings_all_list_the_whole_project(
    tmp_path: Path, spelling: str
) -> None:
    """An unrecognized root spelling used to return an empty project.

    That is what an agent reads as "there is nothing here", so it lists again
    instead of building — the loop behind a spent step budget.
    """
    service, asset_id = await _service_for(tmp_path)

    listed, _ = await service.execute_staged(
        asset_id, ProjectToolCallV1(name="list_files", arguments={"path": spelling}), {}
    )

    assert "README.md" in listed["files"]
    assert "app/main.py" in listed["files"]


@pytest.mark.asyncio
async def test_a_real_prefix_still_filters(tmp_path: Path) -> None:
    service, asset_id = await _service_for(tmp_path)

    listed, _ = await service.execute_staged(
        asset_id, ProjectToolCallV1(name="list_files", arguments={"path": "./app"}), {}
    )

    assert listed["files"] == ["app/main.py"]


@pytest.mark.asyncio
async def test_an_absolute_path_is_refused_with_the_corrected_form(
    tmp_path: Path,
) -> None:
    service, asset_id = await _service_for(tmp_path)
    call = ProjectToolCallV1(
        name="create_file",
        arguments={
            "path": "/workspace/agent-showcase/app/__init__.py",
            "content": "# app package\n",
        },
    )

    with pytest.raises(ProjectWorkspaceError) as error:
        await service.execute_staged(asset_id, call, {})

    # Without the corrected form the model just resends the same path.
    assert "app/__init__.py" in str(error.value)
    assert "relative to the project root" in str(error.value)


@pytest.mark.asyncio
async def test_a_leading_dot_slash_write_path_is_accepted(tmp_path: Path) -> None:
    service, asset_id = await _service_for(tmp_path)
    call = ProjectToolCallV1(
        name="create_file", arguments={"path": "./app/new.py", "content": "x = 1\n"}
    )

    _, staged = await service.execute_staged(asset_id, call, {})

    assert staged is not None and "app/new.py" in staged


# ── The staged-syntax gate: catch broken files before they reach the disk ───


def _staged(files: dict[str, str]) -> dict[str, dict[str, str]]:
    """Wrap raw file text in the minimal staged-entry shape the checker reads."""
    return {path: {"content": content} for path, content in files.items()}


@pytest.mark.asyncio
async def test_a_refused_create_names_the_next_file_the_build_owes(tmp_path: Path) -> None:
    """"Write a different path" left the model guessing, and it guessed the same
    path again — 43 create_file calls for 11 files in one live build. The host
    has the manifest, so the refusal says which file is actually outstanding."""
    projects_root = tmp_path / "Projects"
    (projects_root / "demo").mkdir(parents=True)
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        repo_root=tmp_path,
        asset_roots=[projects_root],
        model_backend="deterministic",
        allow_test_backends=True,
    )
    assets = AssetManager(settings.asset_roots, catalog_path=settings.asset_catalog_path)
    asset_id = (await assets.scan())[0].id
    service = ProjectWorkspaceService(settings, assets, DeterministicModelProvider())
    staged = _staged({"app/main.py": "X = 1\n"})
    call = ProjectToolCallV1(
        name="create_file", arguments={"path": "app/main.py", "content": "Y = 2\n"}
    )

    with pytest.raises(ProjectWorkspaceError) as named:
        await service.execute_staged(asset_id, call, staged, ["app/config.py", "README.md"])
    with pytest.raises(ProjectWorkspaceError) as unnamed:
        await service.execute_staged(asset_id, call, staged)

    assert "write the next file you planned: app/config.py" in str(named.value)
    assert named.value.wrong_target is True
    assert named.value.argument_shape is False
    # Without a manifest there is nothing to name, and it must not invent one.
    assert "write a different path" in str(unnamed.value)


def test_staged_syntax_gate_flags_a_python_indentation_error() -> None:
    """The exact class of defect a real qwen build shipped: a method whose
    decorator and def disagree on indentation, so the file will not import."""
    from waqil_api.project_workspace import staged_syntax_errors

    broken = (
        "class A:\n"
        "     @property\n"
        "    def name(self):\n"
        "        return 1\n"
    )
    errors = staged_syntax_errors(_staged({"app/agents/extractor.py": broken}))

    assert [item["path"] for item in errors] == ["app/agents/extractor.py"]
    assert "IndentationError" in errors[0]["error"]


def test_staged_syntax_gate_passes_valid_python_and_json() -> None:
    from waqil_api.project_workspace import staged_syntax_errors

    clean = _staged(
        {
            "app/main.py": "def f() -> int:\n    return 1\n",
            "data/config.json": '{"a": 1, "b": [2, 3]}\n',
        }
    )
    assert staged_syntax_errors(clean) == []


def test_staged_syntax_gate_flags_malformed_json() -> None:
    from waqil_api.project_workspace import staged_syntax_errors

    errors = staged_syntax_errors(_staged({"config.json": '{"a": 1,,}'}))
    assert len(errors) == 1
    assert "JSONDecodeError" in errors[0]["error"]


def test_staged_syntax_gate_skips_languages_it_cannot_safely_parse() -> None:
    """A clean result means "nothing checkable is broken", not "all correct":
    TS/JS/CSS/HTML have no safe stdlib parser, so they are passed through."""
    from waqil_api.project_workspace import staged_syntax_errors

    unparseable_but_skipped = _staged(
        {
            "app/x.ts": "const y: number = ;",   # invalid TS, but not our job to judge
            "app/static/style.css": '"""not css""" * { color: red }',
            "app/static/index.html": "<div><span></div>",
        }
    )
    assert staged_syntax_errors(unparseable_but_skipped) == []
