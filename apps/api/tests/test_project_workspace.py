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


def test_project_ask_user_pauses_the_build_and_resumes_with_the_answer(
    tmp_path: Path,
) -> None:
    """The loop's talk channel: a scoped turn pauses on the model's question,
    the user answers on the same endpoint the planner path uses, and the SAME
    turn continues with the answer as the ask_user call's result. Before this
    channel existed, a model that wanted to clarify could only fabricate a
    completion — the original live failure this guards against."""
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
                "content": "[project-ask-test]",
                "project_id": project_id,
                "project_mode": "grok_bootstrap_local",
            },
        ).json()["run_id"]
        for _ in range(200):
            run = client.get(f"/api/v1/runs/{run_id}").json()
            if run["status"] in {"awaiting_input", "completed", "failed"}:
                break
            time.sleep(0.01)
        assert run["status"] == "awaiting_input", run

        # The question is recoverable exactly like a planner-path pause.
        recoverable = client.get("/api/v1/runs?status=awaiting_input").json()
        pending = next(
            item["elicitation"] for item in recoverable if item["run"]["id"] == run_id
        )
        assert pending["question"] == "Which option would you like?"
        assert pending["options"] == ["Option A", "Option B"]

        answered = client.post(
            f"/api/v1/runs/{run_id}/answers", json={"option": "Option A"}
        )
        assert answered.status_code == 200, answered.text
        for _ in range(200):
            run = client.get(f"/api/v1/runs/{run_id}").json()
            if run["status"] in {"completed", "failed"}:
                break
            time.sleep(0.01)
        assert run["status"] == "completed", run

        messages = client.get(
            f"/api/v1/conversations/{conversation_id}/messages"
        ).json()
        final = messages[-1]
        assert final["role"] == "assistant"
        # The answer flowed back into the loop and out through respond…
        assert "They answered: Option A" in final["content"]
        # …and a talk turn never wears the empty-build footer.
        assert "No file changes were staged" not in final["content"]


def test_project_respond_answers_without_the_stage_footer(tmp_path: Path) -> None:
    """respond is an answer, not a completion claim: no staged files, no
    footer, no premature-finish pushback."""
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
                "content": "[project-respond-test]",
                "project_id": project_id,
                "project_mode": "grok_bootstrap_local",
            },
        ).json()["run_id"]
        for _ in range(200):
            run = client.get(f"/api/v1/runs/{run_id}").json()
            if run["status"] in {"completed", "failed"}:
                break
            time.sleep(0.01)
        assert run["status"] == "completed", run
        messages = client.get(
            f"/api/v1/conversations/{conversation_id}/messages"
        ).json()
        final = messages[-1]
        assert final["role"] == "assistant"
        assert "This project uses FastAPI" in final["content"]
        assert "No file changes were staged" not in final["content"]


def test_the_ollama_lane_can_write_a_project_map_when_no_cloud_key_exists() -> None:
    """Opening a project must not require an OCI or Cohere key.

    It used to: the router bootstrapped on Grok or Cohere and nothing else, and
    the workspace refused outright when neither was configured. With the Grok
    lane switched off and a spent Cohere trial quota — the real state of this
    install — every new project became unopenable with a bare 500, even though
    the Ollama lane the user actually runs on can answer the same structured
    request. It is also the better fallback for a call that ships the whole
    repository: a pinned local model keeps that snapshot on-device.
    """
    import asyncio

    from waqil_api.model_provider import RoutedModelProvider

    seen: dict[str, object] = {}

    class Local:
        async def bootstrap_project(self, snapshot, *, model_aliases=None):
            seen["called"] = "local"
            seen["aliases"] = model_aliases
            return "map-from-ollama"

    class Cloud:
        def __init__(self, available: bool) -> None:
            self.available = available

        async def bootstrap_project(self, snapshot):
            seen["called"] = "cloud"
            return "map-from-cloud"

    # Neither cloud key configured — the Ollama lane writes the map, and the
    # user's pinned model is what it is told to use.
    router = RoutedModelProvider(Local(), Cloud(False), Cloud(False))
    result = asyncio.run(
        router.bootstrap_project({"project": {}}, model_aliases={"planner": "glm-5.2:cloud"})
    )
    assert result == "map-from-ollama"
    assert seen["called"] == "local"
    assert seen["aliases"] == {"planner": "glm-5.2:cloud"}

    # A configured cloud key still keeps first refusal.
    seen.clear()
    router = RoutedModelProvider(Local(), Cloud(True), Cloud(False))
    assert asyncio.run(router.bootstrap_project({"project": {}})) == "map-from-cloud"
    assert seen["called"] == "cloud"


def test_a_dead_cloud_key_falls_through_to_the_ollama_lane() -> None:
    """`available` means "configured", not "working".

    Cohere on this install has a key and a spent trial quota, so an
    availability check routed every project map to a provider that answers
    429 — and opening any new project failed with a bare 500. A map is a
    single idempotent call, so a failure must fall through to the next
    provider rather than end the request.
    """
    import asyncio

    from waqil_api.model_provider import ModelProviderError, RoutedModelProvider

    class Local:
        async def bootstrap_project(self, snapshot, *, model_aliases=None):
            return "map-from-ollama"

    class DeadCloud:
        available = True

        async def bootstrap_project(self, snapshot):
            raise ModelProviderError('Cohere returned HTTP 429: {"message":"Trial key"}')

    class OffCloud:
        available = False

        async def bootstrap_project(self, snapshot):  # pragma: no cover
            raise AssertionError("an unconfigured provider must not be tried")

    router = RoutedModelProvider(Local(), OffCloud(), DeadCloud())
    assert asyncio.run(router.bootstrap_project({"project": {}})) == "map-from-ollama"

    # When nothing can answer, the CLOUD cause is what surfaces — it is the
    # actionable one, where the local error is just the last thing to fail.
    class DeadLocal:
        async def bootstrap_project(self, snapshot, *, model_aliases=None):
            raise ModelProviderError("ollama is not running")

    router = RoutedModelProvider(DeadLocal(), OffCloud(), DeadCloud())
    try:
        asyncio.run(router.bootstrap_project({"project": {}}))
        raise AssertionError("expected the map to fail")
    except ModelProviderError as exc:
        assert "429" in str(exc)


@pytest.mark.asyncio
async def test_the_plan_is_written_to_the_project_and_survives_revision(tmp_path) -> None:
    """Plan-as-file: the build plan lands in .metis as a checklist the next
    context window can read, replaces itself on revision instead of stacking,
    and marks what actually landed."""
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

    await service.record_plan(
        project_id,
        {"files": ["app/main.py", "app/static/index.html"], "intent": "build", "scope": "narrow"},
    )
    notes = (project / ".metis" / "METIS.md").read_text(encoding="utf-8")
    assert "- [ ] `app/main.py`" in notes
    assert "- [ ] `app/static/index.html`" in notes
    plan = json.loads((project / ".metis" / "plan.json").read_text(encoding="utf-8"))
    assert plan["files"] == ["app/main.py", "app/static/index.html"]

    # A revision REPLACES the section — one plan in the file, ever.
    await service.record_plan(
        project_id,
        {"files": ["app.py"], "intent": "edit", "scope": "narrow",
         "reason": "the project is Streamlit; there is no app/ package"},
    )
    notes = (project / ".metis" / "METIS.md").read_text(encoding="utf-8")
    assert notes.count("Current build plan") == 1
    assert "- [ ] `app.py`" in notes
    assert "app/static/index.html" not in notes.split("<!-- metis-plan:start -->")[1]
    assert "the project is Streamlit" in notes

    # Applied files come back checked.
    await service.record_plan(
        project_id, {"files": ["app.py"], "intent": "edit", "scope": "narrow"},
        done=["app.py"],
    )
    notes = (project / ".metis" / "METIS.md").read_text(encoding="utf-8")
    assert "- [x] `app.py`" in notes


@pytest.mark.asyncio
async def test_repo_map_ranks_the_real_tree_and_follows_the_request(tmp_path: Path) -> None:
    """The map a build step is handed, built from a real project on disk."""
    projects_root = tmp_path / "Projects"
    project = projects_root / "shop"
    (project / "app").mkdir(parents=True)
    (project / "README.md").write_text("# Shop\n", encoding="utf-8")
    (project / "app" / "models.py").write_text(
        "class Order:\n    pass\n\n\nclass Customer:\n    pass\n", encoding="utf-8"
    )
    (project / "app" / "billing.py").write_text(
        "from app.models import Order\n\n\ndef charge(order: Order) -> int:\n    return 1\n",
        encoding="utf-8",
    )
    (project / "app" / "shipping.py").write_text(
        "from app.models import Order\n\n\ndef dispatch(order: Order) -> None:\n    return None\n",
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

    text = await service.repo_map(opened.id, request="charge the order", max_chars=4_000)
    # Definitions, with the line numbers that let the model verify them.
    assert "app/models.py" in text
    assert "class Order" in text
    assert "def charge" in text
    # models.py is what both other modules import, so it leads on structure;
    # billing.py is what the request named, so it outranks its sibling.
    assert text.index("app/billing.py") < text.index("app/shipping.py")
    # A file that declares nothing is still named — one line beats a read.
    assert "README.md" in text

    # Extraction is cached per file by (mtime, size); a changed file is re-read
    # and a deleted one stops contributing symbols that no longer exist.
    (project / "app" / "shipping.py").unlink()
    again = await service.repo_map(opened.id, request="charge the order", max_chars=4_000)
    assert "app/shipping.py" not in again
    assert "def charge" in again

    # A budget of zero sends no map at all — the before-side of the measurement.
    assert await service.repo_map(opened.id, request="charge", max_chars=0) == ""
