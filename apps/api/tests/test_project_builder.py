"""Staged project builds: the act→observe→decide loop and its single gate.

The loop may run for dozens of steps, but the doctrine holds at exactly one
place: nothing reaches the real tree except through the one batch approval,
and what it writes is byte-for-byte what the card showed.
"""
from __future__ import annotations

import ast
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from waqil_api.asset_library import AssetManager
from waqil_api.config import Settings
from waqil_api.control_plane import (
    _annotate_summary,
    _blocking_reason,
    _distinct_findings,
    _reference_notes,
    _sandbox_verdict,
)
from waqil_api.contracts import (
    ProjectAgentStepWireV1,
    ProjectBuildStepWireV1,
    ProjectToolCallV1,
    grammar_risks,
    grammar_schema,
    value_constraints,
)
from waqil_api.main import create_app
from waqil_api.model_provider import LOCAL_DECODE_SCHEMAS, DeterministicModelProvider
from waqil_api.project_workspace import ProjectWorkspaceError, ProjectWorkspaceService


async def _noop_emit(*args: Any, **kwargs: Any) -> None:
    """Stand in for the event bus where a test only cares about the return value."""


async def _empty_context(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """A project with nothing in it, for tests about the loop rather than the tree."""
    return {"manifest": {"file_tree": []}, "metis_md": ""}


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        repo_root=tmp_path,
        asset_roots=[tmp_path / "Projects"],
        model_backend="deterministic",
        reference_runner_mode="deterministic",
        allow_test_backends=True,
        **overrides,
    )


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "Projects" / "demo"
    (project / "src").mkdir(parents=True)
    (project / "README.md").write_text("# Demo\n", encoding="utf-8")
    (project / "src" / "main.ts").write_text(
        "export function greet(name: string) {\n  return `Hello ${name}`;\n}\n",
        encoding="utf-8",
    )
    return project


async def _service(settings: Settings) -> tuple[ProjectWorkspaceService, str]:
    assets = AssetManager(
        settings.asset_roots,
        approval_path=settings.asset_approval_path,
        catalog_path=settings.asset_catalog_path,
    )
    discovered = await assets.scan()
    service = ProjectWorkspaceService(settings, assets, DeterministicModelProvider())
    opened = await service.open(discovered[0].id)
    return service, opened.id


def _drive(client: TestClient, run_id: str, until: set[str]) -> dict[str, Any]:
    for _ in range(200):
        run = client.get(f"/api/v1/runs/{run_id}").json()
        if run["status"] in until:
            return run
        time.sleep(0.01)
    raise AssertionError(f"run never reached {until}: {run['status']}")


def _tool_results(client: TestClient, run_id: str, tool: str) -> list[dict[str, Any]]:
    """Every project.tool_result payload for one tool, in order.

    The events endpoint replays SSE frames rather than a JSON array, so the
    payloads come out of the ``data:`` lines.
    """
    stream = client.get(f"/api/v1/runs/{run_id}/events?after=0").text
    payloads = []
    for line in stream.splitlines():
        if not line.startswith("data: "):
            continue
        event = json.loads(line[len("data: ") :])
        if event.get("type") == "project.tool_result":
            payload = event.get("payload", {})
            if payload.get("tool") == tool:
                payloads.append(payload)
    return payloads


# ── The read/edit contract ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_block_copied_from_a_read_can_be_patched_back(tmp_path: Path) -> None:
    """The property the whole loop rests on, and the one nothing used to check.

    apply_patch needs `original` to appear in the file exactly once, and the
    only way the model can see the file is read_file. While reads were
    line-numbered those two could not both hold: the model copied the prefix it
    was shown and matched nothing, 4 times out of 4 against a real model. So a
    block taken verbatim out of a read must patch cleanly, and the test takes it
    the way a model would — straight out of `content`."""
    _project(tmp_path)
    service, asset_id = await _service(_settings(tmp_path))

    _, staged = await service.execute_staged(
        asset_id,
        ProjectToolCallV1(
            name="create_file",
            arguments={
                "path": "src/app.ts",
                "content": "export function greet(name: string) {\n  return name;\n}\n",
            },
        ),
        {},
    )
    read, _ = await service.execute_staged(
        asset_id, ProjectToolCallV1(name="read_file", arguments={"path": "src/app.ts"}), staged
    )

    assert read["truncated"] is False
    assert read["start_line"] == 1 and read["end_line"] == 3
    # No line-number prefix anywhere in the text the model is handed.
    assert not any(line[:8].strip().isdigit() for line in read["content"].splitlines())

    block = read["content"].splitlines()[1]  # exactly what a model would copy
    _, patched = await service.execute_staged(
        asset_id,
        ProjectToolCallV1(
            name="apply_patch",
            arguments={
                "path": "src/app.ts",
                "original": block,
                "replacement": "  return `hi ${name}`;",
            },
        ),
        staged,
    )
    assert patched is not None
    assert "return `hi ${name}`;" in patched["src/app.ts"]["content"]


@pytest.mark.asyncio
async def test_a_staged_file_is_found_however_the_model_spells_the_path(
    tmp_path: Path,
) -> None:
    """"./app/x.py" and "app/x.py" are the same file. The overlay is keyed by the
    canonical form the write produced, so a lookup on the raw argument used to
    miss the model's own staged work and report the file as unavailable."""
    _project(tmp_path)
    service, asset_id = await _service(_settings(tmp_path))

    _, staged = await service.execute_staged(
        asset_id,
        ProjectToolCallV1(
            name="create_file", arguments={"path": "src/new.ts", "content": "const x = 1;\n"}
        ),
        {},
    )
    read, _ = await service.execute_staged(
        asset_id,
        ProjectToolCallV1(name="read_file", arguments={"path": "./src/new.ts"}),
        staged,
    )
    assert read["staged"] is True
    assert "const x = 1;" in read["content"]


@pytest.mark.asyncio
async def test_a_patch_that_matches_nothing_is_shown_the_real_text(tmp_path: Path) -> None:
    """A zero match is usually a model patching a file it never read, inventing
    the block it expects to find — a live build burned four steps that way. The
    host is holding the real text, so it sends the opening of it back instead of
    asking for a read that costs another whole step."""
    project = _project(tmp_path)
    (project / "notes.md").write_text("# Notes\n\nThe real first line.\n", encoding="utf-8")
    service, asset_id = await _service(_settings(tmp_path))

    with pytest.raises(ProjectWorkspaceError) as error:
        await service.execute_staged(
            asset_id,
            ProjectToolCallV1(
                name="apply_patch",
                arguments={
                    "path": "notes.md",
                    "original": "# A heading the model imagined",
                    "replacement": "# Notes",
                },
            ),
            {},
        )
    message = str(error.value)
    assert "matched 0 times" in message
    assert "The real first line." in message  # the actual bytes, not a description
    assert error.value.argument_shape is True
    # The whole refusal has to survive the 1000-character clip the loop applies.
    assert len(message) < 1_000


@pytest.mark.asyncio
async def test_creating_a_staged_path_again_points_at_apply_patch(tmp_path: Path) -> None:
    """A refusal that names no alternative gets repeated. One model spent eight
    consecutive steps re-creating a file it had already staged."""
    _project(tmp_path)
    service, asset_id = await _service(_settings(tmp_path))

    _, staged = await service.execute_staged(
        asset_id,
        ProjectToolCallV1(
            name="create_file", arguments={"path": "src/new.ts", "content": "const x = 1;\n"}
        ),
        {},
    )
    with pytest.raises(ProjectWorkspaceError) as error:
        await service.execute_staged(
            asset_id,
            ProjectToolCallV1(
                name="create_file", arguments={"path": "src/new.ts", "content": "const y = 2;\n"}
            ),
            staged,
        )
    assert "apply_patch" in str(error.value)
    # Wanting a different path is a decision the model has to make, not an
    # argument it got wrong: narrowing the grammar to create_file would trap it.
    assert error.value.argument_shape is False


@pytest.mark.asyncio
async def test_a_missing_required_argument_is_flagged_as_fixable(tmp_path: Path) -> None:
    """Missing keys are the one refusal resending the same tool can fix, so they
    are the only ones that earn a narrowed grammar on the next step."""
    _project(tmp_path)
    service, asset_id = await _service(_settings(tmp_path))

    with pytest.raises(ProjectWorkspaceError) as empty:
        await service.execute_staged(
            asset_id,
            ProjectToolCallV1(name="create_file", arguments={"path": "src/a.ts"}),
            {},
        )
    assert empty.value.argument_shape is True
    assert "content" in str(empty.value)

    with pytest.raises(ProjectWorkspaceError) as diff:
        await service.execute_staged(
            asset_id,
            ProjectToolCallV1(
                name="apply_patch", arguments={"path": "src/app.ts", "patch": "@@ -1 +1 @@"}
            ),
            {},
        )
    assert diff.value.argument_shape is True
    assert "not a diff" in str(diff.value)


# ── The overlay ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_overlay_reads_show_staged_work_and_disk_stays_clean(tmp_path: Path) -> None:
    project = _project(tmp_path)
    service, asset_id = await _service(_settings(tmp_path))

    _, staged = await service.execute_staged(
        asset_id,
        ProjectToolCallV1(
            name="create_file",
            arguments={"path": "src/new.ts", "content": "export const x = 1;\n"},
        ),
        {},
    )
    assert staged is not None and staged["src/new.ts"]["origin"] == "create"
    assert not (project / "src" / "new.ts").exists()

    # The model observes its own staged file exactly as it would a real one.
    read, _ = await service.execute_staged(
        asset_id, ProjectToolCallV1(name="read_file", arguments={"path": "src/new.ts"}), staged
    )
    assert read["staged"] is True and "export const x = 1;" in read["content"]

    listing, _ = await service.execute_staged(
        asset_id, ProjectToolCallV1(name="list_files", arguments={"path": "src"}), staged
    )
    assert listing["files"] == ["src/main.ts", "src/new.ts"]

    found, _ = await service.execute_staged(
        asset_id,
        ProjectToolCallV1(name="search_code", arguments={"query": "const x"}),
        staged,
    )
    assert [item["path"] for item in found["matches"]] == ["src/new.ts"]


@pytest.mark.asyncio
async def test_overlay_patch_chains_and_shadows_the_disk_copy(tmp_path: Path) -> None:
    _project(tmp_path)
    service, asset_id = await _service(_settings(tmp_path))

    _, staged = await service.execute_staged(
        asset_id,
        ProjectToolCallV1(
            name="apply_patch",
            arguments={
                "path": "src/main.ts",
                "original": "Hello ${name}",
                "replacement": "Welcome ${name}",
            },
        ),
        {},
    )
    assert staged is not None
    base_sha = staged["src/main.ts"]["base_sha256"]
    assert base_sha  # pinned to the disk text the patch was computed against

    # A second patch works on the staged text and keeps the original base pin.
    _, staged = await service.execute_staged(
        asset_id,
        ProjectToolCallV1(
            name="apply_patch",
            arguments={
                "path": "src/main.ts",
                "original": "Welcome ${name}",
                "replacement": "Salaam ${name}",
            },
        ),
        staged,
    )
    assert staged is not None
    assert "Salaam" in staged["src/main.ts"]["content"]
    assert staged["src/main.ts"]["base_sha256"] == base_sha

    # Search must see only the staged text, never the shadowed disk copy.
    found, _ = await service.execute_staged(
        asset_id, ProjectToolCallV1(name="search_code", arguments={"query": "Hello"}), staged
    )
    assert found["matches"] == []


@pytest.mark.asyncio
async def test_overlay_enforces_jail_and_budgets(tmp_path: Path) -> None:
    _project(tmp_path)
    settings = _settings(tmp_path, project_staged_max_files=2)
    service, asset_id = await _service(settings)

    for path in ("../outside.txt", ".env.local", ".metis/notes.md"):
        with pytest.raises(ProjectWorkspaceError):
            await service.execute_staged(
                asset_id,
                ProjectToolCallV1(
                    name="create_file", arguments={"path": path, "content": "x\n"}
                ),
                {},
            )

    with pytest.raises(ProjectWorkspaceError, match="refuses to overwrite"):
        await service.execute_staged(
            asset_id,
            ProjectToolCallV1(
                name="create_file", arguments={"path": "README.md", "content": "x\n"}
            ),
            {},
        )

    staged: dict[str, Any] = {}
    for index in range(2):
        _, staged = await service.execute_staged(
            asset_id,
            ProjectToolCallV1(
                name="create_file",
                arguments={"path": f"src/f{index}.txt", "content": "x\n"},
            ),
            staged,
        )
    with pytest.raises(ProjectWorkspaceError, match="capped at 2 files"):
        await service.execute_staged(
            asset_id,
            ProjectToolCallV1(
                name="create_file", arguments={"path": "src/f2.txt", "content": "x\n"}
            ),
            staged,
        )


@pytest.mark.asyncio
async def test_materialize_skips_files_that_drifted_after_staging(tmp_path: Path) -> None:
    """Approval covered the staged bytes, not whatever landed on disk since."""
    project = _project(tmp_path)
    service, asset_id = await _service(_settings(tmp_path))

    _, staged = await service.execute_staged(
        asset_id,
        ProjectToolCallV1(
            name="apply_patch",
            arguments={
                "path": "src/main.ts",
                "original": "Hello ${name}",
                "replacement": "Welcome ${name}",
            },
        ),
        {},
    )
    assert staged is not None
    _, staged = await service.execute_staged(
        asset_id,
        ProjectToolCallV1(
            name="create_file", arguments={"path": "src/new.ts", "content": "ok\n"}
        ),
        staged,
    )
    assert staged is not None

    # Both targets drift before approval: the patched file is edited on disk,
    # and a file appears exactly where the created one would land.
    (project / "src" / "main.ts").write_text("// rewritten by hand\n", encoding="utf-8")
    (project / "src" / "new.ts").write_text("already here\n", encoding="utf-8")

    report = await service.materialize_staged(asset_id, staged)
    assert report["applied"] == []
    assert {item["path"] for item in report["skipped"]} == {"src/main.ts", "src/new.ts"}
    assert (project / "src" / "main.ts").read_text(encoding="utf-8") == "// rewritten by hand\n"
    assert (project / "src" / "new.ts").read_text(encoding="utf-8") == "already here\n"


# ── The loop through the full control plane ────────────────────────────────


def test_staged_build_loops_observes_and_applies_once(tmp_path: Path) -> None:
    project = _project(tmp_path)
    with TestClient(create_app(_settings(tmp_path))) as client:
        project_id = client.post("/api/v1/assets/scan").json()[0]["id"]
        assert client.post(
            f"/api/v1/projects/{project_id}/open",
            json={"mode": "grok_bootstrap_local"},
        ).status_code == 200
        conversation_id = client.post("/api/v1/conversations", json={}).json()["id"]
        run_id = client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json={
                "content": "[project-build-test]",
                "project_id": project_id,
                "project_mode": "grok_bootstrap_local",
            },
        ).json()["run_id"]

        run = _drive(client, run_id, {"awaiting_approval", "completed", "failed"})
        assert run["status"] == "awaiting_approval"
        # Five loop steps ran without a single interrupt; disk is untouched.
        assert not (project / "src" / "build").exists()

        recoverable = client.get("/api/v1/runs?status=awaiting_approval").json()
        approval = next(
            item["approval"] for item in recoverable if item["run"]["id"] == run_id
        )
        assert approval["kind"] == "project_apply_build"
        assert "2 file(s)" in approval["summary"]
        assert "create `src/build/alpha.txt`" in approval["summary"]
        assert "create `src/build/nested/beta.txt`" in approval["summary"]

        assert client.post(
            f"/api/v1/runs/{run_id}/decisions",
            json={"approval_id": approval["id"], "decision": "approve"},
        ).status_code == 200
        run = _drive(client, run_id, {"completed", "failed"})
        assert run["status"] == "completed"

        # The applied file carries the refinement made after the read-back —
        # proof the loop acted on what it observed, not on what it first wrote.
        assert (project / "src" / "build" / "alpha.txt").read_text(
            encoding="utf-8"
        ) == "alpha final\n"
        assert (project / "src" / "build" / "nested" / "beta.txt").read_text(
            encoding="utf-8"
        ) == "beta content\n"

        messages = client.get(
            f"/api/v1/conversations/{conversation_id}/messages"
        ).json()
        assert "Applied 2 file(s)" in messages[-1]["content"]
        notes = (project / ".metis" / "METIS.md").read_text(encoding="utf-8")
        assert "Approved a staged build touching 2 file(s)" in notes


def test_rejected_build_writes_nothing(tmp_path: Path) -> None:
    project = _project(tmp_path)
    with TestClient(create_app(_settings(tmp_path))) as client:
        project_id = client.post("/api/v1/assets/scan").json()[0]["id"]
        assert client.post(
            f"/api/v1/projects/{project_id}/open",
            json={"mode": "grok_bootstrap_local"},
        ).status_code == 200
        conversation_id = client.post("/api/v1/conversations", json={}).json()["id"]
        run_id = client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json={
                "content": "[project-build-test]",
                "project_id": project_id,
                "project_mode": "grok_bootstrap_local",
            },
        ).json()["run_id"]
        run = _drive(client, run_id, {"awaiting_approval", "completed", "failed"})
        assert run["status"] == "awaiting_approval"
        recoverable = client.get("/api/v1/runs?status=awaiting_approval").json()
        approval = next(
            item["approval"] for item in recoverable if item["run"]["id"] == run_id
        )
        assert client.post(
            f"/api/v1/runs/{run_id}/decisions",
            json={"approval_id": approval["id"], "decision": "reject"},
        ).status_code == 200
        run = _drive(client, run_id, {"completed", "failed"})
        assert run["status"] == "completed"
        assert not (project / "src" / "build").exists()
        messages = client.get(
            f"/api/v1/conversations/{conversation_id}/messages"
        ).json()
        assert "nothing was written" in messages[-1]["content"]


# ── The manifest gate: a build is finished when its files exist ─────────────


def test_finishing_short_of_the_planned_files_is_declined_until_they_exist(
    tmp_path: Path,
) -> None:
    """The end-to-end shape of the failure the gate was built for.

    The scripted model stages one of its two planned files and reports the whole
    job done — which the old "did you stage anything" guard accepted, because
    something was staged. It writes the second file only after the host declines
    the finish, so both files reaching disk is proof the decline is what caused
    the second one to be written."""
    project = _project(tmp_path)
    with TestClient(create_app(_settings(tmp_path))) as client:
        project_id = client.post("/api/v1/assets/scan").json()[0]["id"]
        assert client.post(
            f"/api/v1/projects/{project_id}/open",
            json={"mode": "grok_bootstrap_local"},
        ).status_code == 200
        conversation_id = client.post("/api/v1/conversations", json={}).json()["id"]
        run_id = client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json={
                "content": "[project-manifest-test] Build out alpha.txt and beta.txt from scratch.",
                "project_id": project_id,
                "project_mode": "grok_bootstrap_local",
            },
        ).json()["run_id"]

        run = _drive(client, run_id, {"awaiting_approval", "completed", "failed"})
        assert run["status"] == "awaiting_approval"
        approval = next(
            item["approval"]
            for item in client.get("/api/v1/runs?status=awaiting_approval").json()
            if item["run"]["id"] == run_id
        )
        # The card offers both planned files, not the one the model stopped at.
        assert "alpha.txt" in approval["summary"] and "beta.txt" in approval["summary"]
        assert client.post(
            f"/api/v1/runs/{run_id}/decisions",
            json={"approval_id": approval["id"], "decision": "approve"},
        ).status_code == 200
        run = _drive(client, run_id, {"completed", "failed"})
        assert run["status"] == "completed"

        assert (project / "alpha.txt").read_text(encoding="utf-8") == "alpha\n"
        assert (project / "beta.txt").read_text(encoding="utf-8") == "beta\n"


# ── Reference material reaches the model that writes the code ──────────────


def test_a_missing_env_var_in_the_sandbox_does_not_block_approval() -> None:
    """The container runs the project without the project's environment, so a
    config module that correctly raises on a missing required setting looks
    broken there. That is the shape the reference recommends, and blocking on it
    meant correct fail-fast code could never be applied."""
    verification = {
        "errors": [
            {
                "path": "app/config.py",
                "error": "import app.main failed: ValueError: OCI_COMPARTMENT_ID must be set",
                "rung": "runtime",
            }
        ]
    }

    assert _blocking_reason(verification) is None


def test_an_invented_symbol_still_blocks_approval() -> None:
    """No environment makes a name exist that is not in the module. With the OCI
    packages baked into the verify image, the container is the only rung that can
    catch this — a real build invented `load_client_config`."""
    verification = {
        "errors": [
            {
                "path": "app/oci_client.py",
                "error": (
                    "import app.oci_client failed: ImportError: cannot import name "
                    "'load_client_config' from 'oci_genai_auth'"
                ),
                "rung": "runtime",
            }
        ]
    }

    reason = _blocking_reason(verification)

    assert reason and "load_client_config" in reason


def test_a_syntax_error_always_blocks_approval() -> None:
    """Static rungs read the staged text as written; nothing excuses them."""
    verification = {
        "errors": [
            {"path": "app/a.py", "error": "SyntaxError: bad", "rung": "syntax"},
            {"path": "app/b.py", "error": "unresolved import", "rung": "wiring"},
        ]
    }

    reason = _blocking_reason(verification)

    assert reason and "2 problem(s)" in reason


def test_the_same_defect_from_two_rungs_is_reported_once() -> None:
    """A live Grok build turned one undeclared import into "3 problem(s)" on the
    card, because the parser-level check and the container that failed to import
    it both reported it. The count is the first thing read, so it must mean
    distinct defects."""
    same = {"path": "app/oci_client.py", "error": "imports ocigenaiauth (line 5)"}
    other = {"path": "app/main.py", "error": "unresolved import"}

    distinct = _distinct_findings([same, dict(same), other, dict(same)])

    assert distinct == [same, other]


def test_reference_notes_are_read_from_disk_whole(tmp_path: Path) -> None:
    """The reference is sent, not searched. A document that fits the budget goes
    in whole — half an API reference is how a model gets a confident wrong
    answer instead of no answer."""
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "oci-responses-api.md").write_text("A" * 400, encoding="utf-8")
    (reference / "README.md").write_text("index, not facts", encoding="utf-8")

    notes = _reference_notes("build an oci responses demo", reference, max_characters=5_000)

    assert [note["source"] for note in notes] == ["reference/oci-responses-api.md"]
    assert notes[0]["text"] == "A" * 400


def test_reference_notes_rank_by_how_often_the_request_uses_the_title(
    tmp_path: Path,
) -> None:
    """Both titles contain "app"/"api", so mere presence tied them and the tie
    broke alphabetically — which lost the OCI reference on an OCI build.
    Counting occurrences ranks the subject the request actually dwells on."""
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "oci-responses-api.md").write_text("O" * 3_000, encoding="utf-8")
    (reference / "fastapi-static-app.md").write_text("F" * 3_000, encoding="utf-8")

    prompt = "Build an OCI app: OCI Responses API, OCI model, OCI profile. FastAPI app."
    notes = _reference_notes(prompt, reference, max_characters=3_000)

    assert [note["source"] for note in notes] == ["reference/oci-responses-api.md"]


def test_the_top_reference_is_truncated_rather_than_dropped(tmp_path: Path) -> None:
    """A budget under one document's size used to send nothing at all, which is
    how every local build ran with no reference. The best document is cut down,
    and it says that it was cut."""
    reference = tmp_path / "reference"
    reference.mkdir()
    body = "# Title\n\nopening that matters\n\n" + ("## Later\n\n" + "x" * 4_000)
    (reference / "oci-responses-api.md").write_text(body, encoding="utf-8")

    notes = _reference_notes("oci oci oci", reference, max_characters=2_500)

    assert len(notes) == 1
    assert "opening that matters" in notes[0]["text"]
    assert "truncated" in notes[0]["text"]
    assert len(notes[0]["text"]) <= 2_500


def test_reference_notes_are_absent_when_the_budget_is_zero(tmp_path: Path) -> None:
    """Disabling the reference is a setting, not a code change."""
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "a.md").write_text("facts", encoding="utf-8")

    assert _reference_notes("anything", reference, max_characters=0) == []
    assert _reference_notes("anything", tmp_path / "missing", max_characters=100) == []


def test_a_build_turn_receives_the_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reference must reach the request the build model actually sees, on
    every step. Retrieval was tried first and never surfaced it once; this is the
    assertion that the deterministic path does."""
    _project(tmp_path)
    seen: list[dict[str, Any]] = []
    original = DeterministicModelProvider.project_step

    async def capture(self: Any, request: dict[str, Any], **kwargs: Any) -> Any:
        seen.append(request)
        return await original(self, request, **kwargs)

    monkeypatch.setattr(DeterministicModelProvider, "project_step", capture)

    with TestClient(create_app(_settings(tmp_path))) as client:
        project_id = client.post("/api/v1/assets/scan").json()[0]["id"]
        assert client.post(
            f"/api/v1/projects/{project_id}/open",
            json={"mode": "grok_bootstrap_local"},
        ).status_code == 200
        conversation_id = client.post("/api/v1/conversations", json={}).json()["id"]
        run_id = client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json={
                "content": "[project-manifest-test] Build out alpha.txt and beta.txt from scratch.",
                "project_id": project_id,
                "project_mode": "grok_bootstrap_local",
            },
        ).json()["run_id"]
        _drive(client, run_id, {"awaiting_approval", "completed", "failed"})

    # Every step carries the key, so the model is never left to guess at an
    # external API on some steps and not others.
    assert seen
    assert all("reference_notes" in request for request in seen)
    assert all(isinstance(request["reference_notes"], list) for request in seen)


# ── The staged-syntax gate: broken code never reaches the disk unflagged ────


def test_a_write_that_does_not_parse_is_refused_at_stage_time(tmp_path: Path) -> None:
    """The parse check runs on the write, not on the finished changeset. A file
    that will not parse is refused before it enters the overlay, the model sends
    the corrected file on its next step, and only that version is ever staged."""
    project = _project(tmp_path)
    with TestClient(create_app(_settings(tmp_path))) as client:
        project_id = client.post("/api/v1/assets/scan").json()[0]["id"]
        assert client.post(
            f"/api/v1/projects/{project_id}/open",
            json={"mode": "grok_bootstrap_local"},
        ).status_code == 200
        conversation_id = client.post("/api/v1/conversations", json={}).json()["id"]
        run_id = client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json={
                "content": "[project-syntax-gate-test]",
                "project_id": project_id,
                "project_mode": "grok_bootstrap_local",
            },
        ).json()["run_id"]

        run = _drive(client, run_id, {"awaiting_approval", "completed", "failed"})
        assert run["status"] == "awaiting_approval"
        recoverable = client.get("/api/v1/runs?status=awaiting_approval").json()
        approval = next(
            item["approval"] for item in recoverable if item["run"]["id"] == run_id
        )
        # The changeset offered is clean, so the card is approvable.
        assert approval["blocked_reason"] is None
        assert "would stop this project working" not in approval["summary"]
        assert client.post(
            f"/api/v1/runs/{run_id}/decisions",
            json={"approval_id": approval["id"], "decision": "approve"},
        ).status_code == 200
        run = _drive(client, run_id, {"completed", "failed"})
        assert run["status"] == "completed"

        # The applied file is the repaired, parseable version.
        applied = (project / "app" / "broken.py").read_text(encoding="utf-8")
        assert applied == "def f():\n    return 1\n"
        ast.parse(applied)  # would raise if the broken version had been applied

        # The refusal happened on the write: the overlay never grew to hold the
        # broken text, so no staged-file count reflects it.
        writes = _tool_results(client, run_id, "create_file")
        assert [write["ok"] for write in writes] == [False, True]
        assert writes[0]["staged"] is False


def test_a_staged_import_that_resolves_nowhere_is_sent_back_and_fixed(
    tmp_path: Path,
) -> None:
    """The rung above parsing. Every staged file is valid Python on its own, so
    the syntax gate is clean, but the entrypoint imports a module the build never
    writes — the project would fail on its first import. The cross-file gate
    catches it, hands it back, and only the repaired file is offered."""
    project = _project(tmp_path)
    with TestClient(create_app(_settings(tmp_path))) as client:
        project_id = client.post("/api/v1/assets/scan").json()[0]["id"]
        assert client.post(
            f"/api/v1/projects/{project_id}/open",
            json={"mode": "grok_bootstrap_local"},
        ).status_code == 200
        conversation_id = client.post("/api/v1/conversations", json={}).json()["id"]
        run_id = client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json={
                "content": "[project-wiring-gate-test]",
                "project_id": project_id,
                "project_mode": "grok_bootstrap_local",
            },
        ).json()["run_id"]

        run = _drive(client, run_id, {"awaiting_approval", "completed", "failed"})
        assert run["status"] == "awaiting_approval"
        recoverable = client.get("/api/v1/runs?status=awaiting_approval").json()
        approval = next(
            item["approval"] for item in recoverable if item["run"]["id"] == run_id
        )
        # The gate ran and came back clean, so the card carries no warning.
        assert "would stop this project working" not in approval["summary"]
        assert client.post(
            f"/api/v1/runs/{run_id}/decisions",
            json={"approval_id": approval["id"], "decision": "approve"},
        ).status_code == 200
        run = _drive(client, run_id, {"completed", "failed"})
        assert run["status"] == "completed"

        applied = (project / "app" / "main.py").read_text(encoding="utf-8")
        assert "app.missing" not in applied
        assert "def helper():" in applied


def test_a_model_that_only_writes_unparseable_code_stages_nothing(
    tmp_path: Path,
) -> None:
    """A model that never sends anything that parses has every write refused, so
    the turn ends with an empty overlay. There is no changeset, no approval card
    and nothing on disk — code that cannot parse has no route to the user."""
    project = _project(tmp_path)
    with TestClient(create_app(_settings(tmp_path))) as client:
        project_id = client.post("/api/v1/assets/scan").json()[0]["id"]
        assert client.post(
            f"/api/v1/projects/{project_id}/open",
            json={"mode": "grok_bootstrap_local"},
        ).status_code == 200
        conversation_id = client.post("/api/v1/conversations", json={}).json()["id"]
        run_id = client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json={
                "content": "[project-syntax-unfixable-test]",
                "project_id": project_id,
                "project_mode": "grok_bootstrap_local",
            },
        ).json()["run_id"]

        run = _drive(client, run_id, {"awaiting_approval", "completed", "failed"})
        # It terminates on the step budget rather than looping, and says so
        # instead of reporting a build it did not produce.
        assert run["status"] == "completed"
        assert not (project / "app" / "broken.py").exists()
        assert client.get("/api/v1/runs?status=awaiting_approval").json() == []

        writes = _tool_results(client, run_id, "create_file")
        assert writes and not any(write["ok"] for write in writes)
        assert not any(write["staged"] for write in writes)


def test_a_changeset_with_a_hard_error_cannot_be_approved(tmp_path: Path) -> None:
    """Every file parses, so the stage-time gate passes them, but the entrypoint
    imports a module the build never writes and the model never repairs it. The
    card carries the error, Approve is refused by the API, and Reject still
    works — a build the host proved cannot run has no route to disk."""
    project = _project(tmp_path)
    with TestClient(create_app(_settings(tmp_path))) as client:
        project_id = client.post("/api/v1/assets/scan").json()[0]["id"]
        assert client.post(
            f"/api/v1/projects/{project_id}/open",
            json={"mode": "grok_bootstrap_local"},
        ).status_code == 200
        conversation_id = client.post("/api/v1/conversations", json={}).json()["id"]
        run_id = client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json={
                "content": "[project-wiring-unfixable-test]",
                "project_id": project_id,
                "project_mode": "grok_bootstrap_local",
            },
        ).json()["run_id"]

        run = _drive(client, run_id, {"awaiting_approval", "completed", "failed"})
        # The work is still offered for review, not silently discarded.
        assert run["status"] == "awaiting_approval"
        approval = next(
            item["approval"]
            for item in client.get("/api/v1/runs?status=awaiting_approval").json()
            if item["run"]["id"] == run_id
        )
        assert approval["kind"] == "project_apply_build"
        assert "would stop this project working" in approval["summary"]
        assert approval["blocked_reason"]
        assert "app/main.py" in approval["blocked_reason"]

        # Approving is refused by the API, not merely hidden in the UI.
        refused = client.post(
            f"/api/v1/runs/{run_id}/decisions",
            json={"approval_id": approval["id"], "decision": "approve"},
        )
        assert refused.status_code == 409
        assert "app/main.py" in refused.json()["detail"]
        assert not (project / "app" / "main.py").exists()

        # Rejecting the same blocked approval stays available.
        assert client.post(
            f"/api/v1/runs/{run_id}/decisions",
            json={"approval_id": approval["id"], "decision": "reject"},
        ).status_code == 200
        run = _drive(client, run_id, {"completed", "failed"})
        assert not (project / "app" / "main.py").exists()


# ── Surviving a model that misses the contract ─────────────────────────────


@pytest.mark.parametrize(
    "payload",
    [
        # The exact shape north-mini returned: the tool call, no envelope.
        {"name": "list_files", "arguments": {"path": "."}},
        {"tool": "list_files", "args": {"path": "."}},
        {"status": "tool", "name": "list_files", "arguments": {"path": "."}},
        {"tool_call": {"name": "list_files", "arguments": {"path": "."}}},
    ],
)
def test_collapsed_step_envelopes_are_coerced(payload: dict[str, Any]) -> None:
    from waqil_api.contracts import ProjectAgentStepV1

    step = ProjectAgentStepV1.model_validate(payload)
    assert step.status == "tool"
    assert step.tool_call is not None and step.tool_call.name == "list_files"


def test_a_bare_response_is_read_as_completion() -> None:
    from waqil_api.contracts import ProjectAgentStepV1

    step = ProjectAgentStepV1.model_validate({"response": "all done"})
    assert step.status == "complete" and step.response == "all done"


def test_coercion_widens_shape_not_authority() -> None:
    """A tool the host does not offer is still refused, however it is shaped."""
    from waqil_api.contracts import ProjectAgentStepV1

    with pytest.raises(Exception):
        ProjectAgentStepV1.model_validate({"name": "run_shell", "arguments": {}})


@pytest.mark.asyncio
async def test_a_backend_refusal_ends_the_turn_at_once_and_records_the_real_cause() -> None:
    """A permanent backend failure is not a model mistake and must not be
    retried as one. Feeding it back three times produced six doomed calls and a
    message blaming the model, while the only description of the actual fault
    lived in a checkpoint blob. Now the turn ends on the first one, the cause is
    a run event, and the staged work is still offered."""
    from waqil_api.control_plane import ControlPlane
    from waqil_api.model_provider import PermanentModelError

    emitted: list[tuple[str, dict[str, Any]]] = []

    class RecordingEvents:
        async def emit(self, run_id, conversation_id, event_type, payload):
            emitted.append((event_type, payload))

    plane = SimpleNamespace(events=RecordingEvents())
    state = {"run_id": "run_x", "conversation_id": "conv_x"}
    staged = {"src/a.ts": {"content": "x", "origin": "create", "base_sha256": "", "bytes": 1}}

    result = await ControlPlane._blocked_project_step(
        plane,
        state,
        PermanentModelError("failed to parse grammar", reason="grammar_compile"),
        3,
        staged,
    )

    # The turn ends here: response_text with no pending call is what the router
    # reads as finished, and the staged file still reaches its approval card.
    assert result["project_pending_call"] == {}
    assert "verify-schemas" in result["response_text"]
    assert "1 file change(s) staged" in result["response_text"]
    # It never suggests the model misbehaved, because the model never replied.
    assert "could not read" not in result["response_text"]
    assert "project_malformed_streak" not in result
    event_type, payload = emitted[0]
    assert event_type == "project.step_blocked"
    assert payload["reason"] == "grammar_compile"
    assert "failed to parse grammar" in payload["detail"]
    assert isinstance(json.dumps(payload), str)  # the write happens inside a transaction


@pytest.mark.asyncio
async def test_unreadable_steps_become_evidence_then_end_the_turn(tmp_path: Path) -> None:
    """One bad step is recoverable; a run of them ends the turn with the work."""
    from waqil_api.control_plane import _MAX_MALFORMED_PROJECT_STEPS, ControlPlane

    state: dict[str, Any] = {"project_trace": [], "project_malformed_streak": 0}
    staged = {"src/a.ts": {"content": "x", "origin": "create", "base_sha256": "", "bytes": 1}}

    for attempt in range(1, _MAX_MALFORMED_PROJECT_STEPS):
        result = ControlPlane._malformed_project_step(
            None, state, ValueError("bad json"), attempt, staged
        )
        assert result["project_malformed_streak"] == attempt
        # No answer and no call: the router sends this straight back for another try.
        assert result["project_pending_call"] == {}
        assert "response_text" not in result
        assert "not a readable step" in result["project_trace"][-1]["result"]["error"]
        state = {**state, **result}

    final = ControlPlane._malformed_project_step(
        None, state, ValueError("bad json"), 9, staged
    )
    assert "stopped this turn" in final["response_text"]
    # The staged work is offered rather than discarded.
    assert "1 file change(s) staged" in final["response_text"]


def test_step_budget_still_offers_the_staged_work(tmp_path: Path) -> None:
    """Running out of steps proposes what was built, instead of dropping it."""
    project = _project(tmp_path)
    settings = _settings(tmp_path, project_agent_max_steps=4)
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
                "content": "[project-build-test]",
                "project_id": project_id,
                "project_mode": "grok_bootstrap_local",
            },
        ).json()["run_id"]
        run = _drive(client, run_id, {"awaiting_approval", "completed", "failed"})
        assert run["status"] == "awaiting_approval"
        recoverable = client.get("/api/v1/runs?status=awaiting_approval").json()
        approval = next(
            item["approval"] for item in recoverable if item["run"]["id"] == run_id
        )
        assert approval["kind"] == "project_apply_build"
        assert client.post(
            f"/api/v1/runs/{run_id}/decisions",
            json={"approval_id": approval["id"], "decision": "approve"},
        ).status_code == 200
        run = _drive(client, run_id, {"completed", "failed"})
        assert run["status"] == "completed"
        messages = client.get(
            f"/api/v1/conversations/{conversation_id}/messages"
        ).json()
        assert "step limit" in messages[-1]["content"]
        # The four steps staged both files and the refinement patch before the
        # budget hit, so approval applies the finished refinement.
        assert (project / "src" / "build" / "alpha.txt").read_text(
            encoding="utf-8"
        ) == "alpha final\n"


def test_a_repeated_read_returns_a_correction_instead_of_the_same_bytes() -> None:
    """The loop that spent a whole build: list_files '.' fifteen times over."""
    from waqil_api.control_plane import _repeated_project_call

    call = ProjectToolCallV1(name="list_files", arguments={"path": ""})
    state = {
        "project_trace": [
            {"tool": "read_file", "arguments": {"path": "a.py"}, "result": {"ok": True}},
            {"tool": "list_files", "arguments": {"path": ""}, "result": {"ok": True}},
        ]
    }

    repeat = _repeated_project_call(state, call)

    assert repeat is not None
    assert repeat["ok"] is False
    assert "trace entry 2" in repeat["error"]


def test_a_first_time_read_is_not_treated_as_a_repeat() -> None:
    from waqil_api.control_plane import _repeated_project_call

    state = {
        "project_trace": [
            {"tool": "list_files", "arguments": {"path": "app"}, "result": {"ok": True}},
        ]
    }

    assert (
        _repeated_project_call(
            state, ProjectToolCallV1(name="list_files", arguments={"path": "tests"})
        )
        is None
    )


def test_a_repeated_write_or_check_still_runs() -> None:
    """Only reads are suppressed: a write is refused on its own merits and a
    check may legitimately re-run once something changed."""
    from waqil_api.control_plane import _repeated_project_call

    for name, arguments in (
        ("create_file", {"path": "a.py", "content": "x"}),
        ("run_check", {"name": "tests"}),
    ):
        state = {
            "project_trace": [
                {"tool": name, "arguments": arguments, "result": {"ok": True}}
            ]
        }
        assert (
            _repeated_project_call(state, ProjectToolCallV1(name=name, arguments=arguments))
            is None
        )


def test_a_previously_failed_read_may_be_retried() -> None:
    from waqil_api.control_plane import _repeated_project_call

    state = {
        "project_trace": [
            {
                "tool": "read_file",
                "arguments": {"path": "app/main.py"},
                "result": {"ok": False, "error": "unavailable"},
            }
        ]
    }

    assert (
        _repeated_project_call(
            state, ProjectToolCallV1(name="read_file", arguments={"path": "app/main.py"})
        )
        is None
    )


def test_a_completion_that_staged_nothing_says_so(tmp_path: Path) -> None:
    """A real run once "completed" a 15-file build with zero writes and a
    fabricated summary. The host line is what makes that visible."""
    _project(tmp_path)
    with TestClient(create_app(_settings(tmp_path))) as client:
        project_id = client.post("/api/v1/assets/scan").json()[0]["id"]
        assert client.post(
            f"/api/v1/projects/{project_id}/open",
            json={"mode": "grok_bootstrap_local"},
        ).status_code == 200
        conversation_id = client.post("/api/v1/conversations", json={}).json()["id"]
        run_id = client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json={
                "content": "Build the whole service now.",
                "project_id": project_id,
                "project_mode": "grok_bootstrap_local",
            },
        ).json()["run_id"]

        run = _drive(client, run_id, {"completed", "failed"})
        assert run["status"] == "completed"

        messages = client.get(
            f"/api/v1/conversations/{conversation_id}/messages"
        ).json()
        assert "No file changes were staged in this turn" in messages[-1]["content"]


def test_a_completion_with_staged_work_carries_no_disclaimer(tmp_path: Path) -> None:
    """The note is for the empty case only; a staged build shows its file list
    on the approval card instead."""
    _project(tmp_path)
    with TestClient(create_app(_settings(tmp_path))) as client:
        project_id = client.post("/api/v1/assets/scan").json()[0]["id"]
        assert client.post(
            f"/api/v1/projects/{project_id}/open",
            json={"mode": "grok_bootstrap_local"},
        ).status_code == 200
        conversation_id = client.post("/api/v1/conversations", json={}).json()["id"]
        run_id = client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json={
                "content": "[project-build-test]",
                "project_id": project_id,
                "project_mode": "grok_bootstrap_local",
            },
        ).json()["run_id"]

        run = _drive(client, run_id, {"awaiting_approval", "completed", "failed"})
        assert run["status"] == "awaiting_approval"

        recoverable = client.get("/api/v1/runs?status=awaiting_approval").json()
        approval = next(
            item["approval"] for item in recoverable if item["run"]["id"] == run_id
        )
        assert client.post(
            f"/api/v1/runs/{run_id}/decisions",
            json={"approval_id": approval["id"], "decision": "approve"},
        ).status_code == 200
        run = _drive(client, run_id, {"completed", "failed"})
        assert run["status"] == "completed"

        messages = client.get(
            f"/api/v1/conversations/{conversation_id}/messages"
        ).json()
        assert "No file changes were staged" not in messages[-1]["content"]


# ── Declining a build that "finished" without writing ──────────────────────


def test_a_fabricated_build_completion_is_declined_then_the_files_get_written(
    tmp_path: Path,
) -> None:
    """The reported bug: a build request that "completed" with a file list and
    zero writes. The host now declines the empty finish and re-prompts, so the
    model actually stages the file, and approval writes it to disk."""
    project = _project(tmp_path)
    with TestClient(create_app(_settings(tmp_path))) as client:
        project_id = client.post("/api/v1/assets/scan").json()[0]["id"]
        assert client.post(
            f"/api/v1/projects/{project_id}/open",
            json={"mode": "grok_bootstrap_local"},
        ).status_code == 200
        conversation_id = client.post("/api/v1/conversations", json={}).json()["id"]
        run_id = client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json={
                # A build instruction (source path + "from scratch") whose model
                # script fabricates a completion before it is made to write.
                "content": (
                    "[project-empty-finish-test] Build out app/main.py from "
                    "scratch and create requirements.txt."
                ),
                "project_id": project_id,
                "project_mode": "grok_bootstrap_local",
            },
        ).json()["run_id"]

        run = _drive(client, run_id, {"awaiting_approval", "completed", "failed"})
        assert run["status"] == "awaiting_approval"
        recoverable = client.get("/api/v1/runs?status=awaiting_approval").json()
        approval = next(
            item["approval"] for item in recoverable if item["run"]["id"] == run_id
        )
        assert approval["kind"] == "project_apply_build"
        assert client.post(
            f"/api/v1/runs/{run_id}/decisions",
            json={"approval_id": approval["id"], "decision": "approve"},
        ).status_code == 200
        run = _drive(client, run_id, {"completed", "failed"})
        assert run["status"] == "completed"

        # The file the model only wrote after being declined is on disk.
        assert (project / "app" / "main.py").read_text(encoding="utf-8") == "print('hi')\n"


def test_a_build_that_never_writes_is_declined_a_bounded_number_of_times(
    tmp_path: Path,
) -> None:
    """A model that keeps fabricating completions is not looped forever: after a
    bounded number of nudges the empty finish stands, with the honest footer."""
    from waqil_api.control_plane import _MAX_EMPTY_PROJECT_FINISHES

    _project(tmp_path)
    with TestClient(create_app(_settings(tmp_path))) as client:
        project_id = client.post("/api/v1/assets/scan").json()[0]["id"]
        assert client.post(
            f"/api/v1/projects/{project_id}/open",
            json={"mode": "grok_bootstrap_local"},
        ).status_code == 200
        conversation_id = client.post("/api/v1/conversations", json={}).json()["id"]
        run_id = client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json={
                # A build instruction with no script: the deterministic model
                # completes empty every time, so the guard's ceiling decides it.
                "content": "Build out app/main.py from scratch and create requirements.txt.",
                "project_id": project_id,
                "project_mode": "grok_bootstrap_local",
            },
        ).json()["run_id"]

        run = _drive(client, run_id, {"completed", "failed"})
        assert run["status"] == "completed"
        messages = client.get(
            f"/api/v1/conversations/{conversation_id}/messages"
        ).json()
        assert "No file changes were staged in this turn" in messages[-1]["content"]
        # Two declines then the completion that stands: the loop is bounded.
        events = client.get(f"/api/v1/runs/{run_id}/events?after=0").text
        assert events.count('"project.agent_step"') == _MAX_EMPTY_PROJECT_FINISHES + 1


def test_a_bare_empty_completion_is_challenged_regardless_of_phrasing(
    tmp_path: Path,
) -> None:
    """The claim-based contract: finishing is a work claim, so an empty one is
    challenged — bounded, then allowed to stand with the footer — no matter how
    the user phrased the request. The old guard keyed on build verbs in the
    prompt, which read "revamp this asset…" as conversation and let a
    fabricated finish stand; phrasing no longer plays any part. An actual
    answer belongs on the respond tool, which completes in one step with no
    footer (see test_project_respond_answers_without_the_stage_footer)."""
    from waqil_api.control_plane import _MAX_EMPTY_PROJECT_FINISHES

    _project(tmp_path)
    with TestClient(create_app(_settings(tmp_path))) as client:
        project_id = client.post("/api/v1/assets/scan").json()[0]["id"]
        assert client.post(
            f"/api/v1/projects/{project_id}/open",
            json={"mode": "grok_bootstrap_local"},
        ).status_code == 200
        conversation_id = client.post("/api/v1/conversations", json={}).json()["id"]
        run_id = client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json={
                "content": "What does src/main.ts do?",
                "project_id": project_id,
                "project_mode": "grok_bootstrap_local",
            },
        ).json()["run_id"]

        run = _drive(client, run_id, {"completed", "failed"})
        assert run["status"] == "completed"
        events = client.get(f"/api/v1/runs/{run_id}/events?after=0").text
        # Two bounded declines, then the completion stands — same ceiling as
        # the build-phrased case above, because the phrasing is irrelevant now.
        assert events.count('"project.agent_step"') == _MAX_EMPTY_PROJECT_FINISHES + 1
        messages = client.get(
            f"/api/v1/conversations/{conversation_id}/messages"
        ).json()
        assert "No file changes were staged in this turn" in messages[-1]["content"]


def test_premature_finish_records_evidence_and_loops_back() -> None:
    """The decline is shaped so the router sends the step straight back with the
    record of what was skipped, and the streak advances toward its ceiling."""
    from waqil_api.control_plane import ControlPlane

    state: dict[str, Any] = {"project_trace": [], "project_empty_finish_streak": 1}

    result = ControlPlane._premature_finish(None, state, 4, 1)

    assert result["project_empty_finish_streak"] == 2
    assert result["project_iterations"] == 5
    # No answer and no pending call: the router routes this back to project_step.
    assert result["project_pending_call"] == {}
    assert "response_text" not in result
    evidence = result["project_trace"][-1]
    assert evidence["result"]["ok"] is False
    assert "zero files staged" in evidence["result"]["error"]


@pytest.mark.parametrize(
    "payload",
    [
        # The exact reply that ended a real run three times over.
        {"status": "tool", "tool_call": {"name": "read_file", "parameters": {"path": "a.py"}}},
        {"status": "tool", "tool_call": {"name": "list_files", "params": {"path": ""}}},
        {"status": "tool", "tool_call": {"name": "create_file", "inputs": {"path": "x", "content": "y"}}},
        {"name": "read_file", "parameters": {"path": "a.py"}},
    ],
)
def test_argument_key_synonyms_are_coerced(payload: dict[str, Any]) -> None:
    from waqil_api.contracts import ProjectAgentStepV1

    step = ProjectAgentStepV1.model_validate(payload)
    assert step.status == "tool"
    assert step.tool_call is not None
    assert step.tool_call.arguments  # the synonym's contents came through


def test_an_unrecognized_extra_key_is_still_refused() -> None:
    """The coercion renames known synonyms; it must not become a catch-all."""
    from waqil_api.contracts import ProjectToolCallV1

    with pytest.raises(Exception):
        ProjectToolCallV1.model_validate(
            {"name": "read_file", "arguments": {}, "shell": "rm -rf /"}
        )


def test_both_argument_keys_at_once_stays_a_refusal() -> None:
    """When "arguments" and a synonym disagree, the intent is ambiguous —
    that is a case to reject as evidence, not to guess about."""
    from waqil_api.contracts import ProjectToolCallV1

    with pytest.raises(Exception):
        ProjectToolCallV1.model_validate(
            {"name": "read_file", "arguments": {"path": "real.py"}, "parameters": {"path": "decoy"}}
        )


def test_the_wire_schema_is_flat_so_the_grammar_does_not_collapse() -> None:
    """The nested tool_call union produced an anyOf-over-$ref grammar that the
    MLX backend collapsed to empty output, ending real build turns. The wire
    form must stay free of both so the constrained decode succeeds."""
    import json as _json

    from waqil_api.contracts import ProjectAgentStepWireV1

    schema = ProjectAgentStepWireV1.model_json_schema()
    props = _json.dumps(schema["properties"])
    assert "$ref" not in props
    assert "anyOf" not in props
    assert "$defs" not in schema
    # The tool name is a closed enum the grammar itself enforces, pinned to
    # the canonical roster rather than restated — a restated copy is exactly
    # how inspect_api went advertised-but-unusable for months.
    from waqil_api.contracts import PROJECT_TOOL_REQUIRED_ARGUMENTS

    assert set(schema["properties"]["tool"]["enum"]) == {
        "", *PROJECT_TOOL_REQUIRED_ARGUMENTS,
    }


def test_wire_tool_step_converts_to_the_nested_step() -> None:
    from waqil_api.contracts import ProjectAgentStepWireV1

    step = ProjectAgentStepWireV1.model_validate(
        {"status": "tool", "tool": "read_file", "arguments": {"path": "a.py"}}
    ).to_step()

    assert step.status == "tool"
    assert step.tool_call is not None
    assert step.tool_call.name == "read_file"
    assert step.tool_call.arguments == {"path": "a.py"}


def test_wire_completion_converts_and_drops_the_tool() -> None:
    from waqil_api.contracts import ProjectAgentStepWireV1

    step = ProjectAgentStepWireV1.model_validate(
        {"status": "complete", "response": "built it", "learnings": ["x"]}
    ).to_step()

    assert step.status == "complete"
    assert step.response == "built it"
    assert step.learnings == ["x"]
    assert step.tool_call is None


def test_wire_rejects_a_tool_step_with_no_tool() -> None:
    from waqil_api.contracts import ProjectAgentStepWireV1

    with pytest.raises(ValueError):
        ProjectAgentStepWireV1.model_validate({"status": "tool", "tool": ""}).to_step()


def test_wire_rejects_an_invented_tool_name() -> None:
    from waqil_api.contracts import ProjectAgentStepWireV1

    with pytest.raises(Exception):
        ProjectAgentStepWireV1.model_validate({"status": "tool", "tool": "run_shell"})


# ── The build-turn grammar: an empty completion is unexpressible ────────────


def test_the_build_wire_schema_is_flat_and_forbids_completion() -> None:
    """The root fix: on a build turn the grammar itself cannot emit an empty
    completion, and it stays flat so the MLX grammar-collapse protection holds."""
    import json as _json

    from waqil_api.contracts import ProjectBuildStepWireV1

    # The projection is what actually reaches the model, so it is what the
    # grammar guarantees have to be asserted against.
    schema = grammar_schema(ProjectBuildStepWireV1)
    props = _json.dumps(schema["properties"])
    assert "$ref" not in props
    assert "anyOf" not in props
    assert "$defs" not in schema
    # status can only ever be "tool" — "complete" is not in the grammar. It is a
    # one-element enum, NOT a const: Ollama's llama.cpp grammar compiler rejects
    # const with HTTP 400 "failed to parse grammar" (it hard-failed build turns
    # on Qwen3-Coder), so grammar_schema rewrites every const to an enum.
    status_prop = schema["properties"]["status"]
    assert status_prop.get("enum") == ["tool"]
    assert "const" not in status_prop
    # The empty tool member is gone, so the model must name a real tool.
    from waqil_api.contracts import PROJECT_TOOL_REQUIRED_ARGUMENTS

    tool_enum = set(schema["properties"]["tool"]["enum"])
    assert "" not in tool_enum
    # respond is excluded on purpose: a prose exit from a build turn is the
    # escape this grammar exists to close. ask_user stays — a mid-build model
    # can genuinely be blocked on the user.
    assert tool_enum == set(PROJECT_TOOL_REQUIRED_ARGUMENTS) - {"respond"}
    # tool is required: there is no default to fall back on.
    assert "tool" in schema.get("required", [])


def test_no_schema_sent_to_a_local_model_carries_a_grammar_hostile_keyword() -> None:
    """The standing invariant behind three separate outages.

    Every keyword that only bounds a value has to be gone before a schema
    becomes a decoding grammar. `const` made llama.cpp reject the request
    outright, and a `maxLength` of 2000 or more still does — that one alone took
    out five of these eight schemas with HTTP 400 "failed to parse grammar",
    which the loop could only report as the model replying unintelligibly. This
    asserts the projection, not the contracts: the contracts keep their bounds
    and Pydantic keeps enforcing them after the reply arrives."""

    def has_const(node: Any) -> bool:
        if isinstance(node, dict):
            if "const" in node:
                return True
            return any(has_const(v) for k, v in node.items() if k != "description")
        if isinstance(node, list):
            return any(has_const(v) for v in node)
        return False

    for cls in LOCAL_DECODE_SCHEMAS:
        projected = grammar_schema(cls)
        assert not has_const(projected), f"{cls.__name__} emits const"
        assert not value_constraints(projected), (
            f"{cls.__name__} still carries {value_constraints(projected)}"
        )


def test_every_local_structured_call_uses_a_registered_schema() -> None:
    """The registry is only a guarantee if it is complete.

    A new `self._structured(SomeContract, ...)` in the Ollama provider that is
    missing from LOCAL_DECODE_SCHEMAS would skip both the projection assertion
    above and the backend preflight — which is exactly how a schema no grammar
    can compile reaches a user."""
    import ast
    import inspect

    from waqil_api import model_provider

    source = inspect.getsource(model_provider)
    provider = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "OllamaModelProvider"
    )
    called: set[str] = set()
    for node in ast.walk(provider):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        function = node.func
        if (
            isinstance(function, ast.Attribute)
            and function.attr in {"_structured", "_structured_unchecked", "_decode_structured"}
            and isinstance(node.args[0], ast.Name)
        ):
            called.add(node.args[0].id)

    # project_step picks between the two wire schemas through a local variable.
    registered = {cls.__name__ for cls in LOCAL_DECODE_SCHEMAS} | {"schema"}
    assert called, "no structured call sites found — the AST walk is wrong, not the code"
    assert called <= registered, f"unregistered local decode schemas: {sorted(called - registered)}"


def test_the_project_wire_schemas_stay_flat_enough_to_compile() -> None:
    """$ref/anyOf is what collapsed MLX output to empty. The project step's own
    schemas must never need a backend to resolve a reference or pick a branch;
    the richer planning schemas are allowed them and are known to compile."""
    for cls in (ProjectAgentStepWireV1, ProjectBuildStepWireV1):
        assert not grammar_risks(grammar_schema(cls)), cls.__name__


def test_the_projection_drops_bounds_and_rewrites_const() -> None:
    from waqil_api.contracts import _project_for_grammar

    node = {
        "a": {"const": "x"},
        "b": [{"const": 1}],
        "c": {"type": "string", "maxLength": 40000, "minLength": 1},
        "d": {"type": "array", "items": {"type": "string"}, "maxItems": 16},
    }
    assert _project_for_grammar(node) == {
        "a": {"enum": ["x"]},
        "b": [{"enum": [1]}],
        "c": {"type": "string"},
        "d": {"type": "array", "items": {"type": "string"}},
    }
    # The source object is untouched: the contract keeps every bound it declared.
    assert node["c"]["maxLength"] == 40000


def test_the_build_wire_schema_cannot_be_a_completion() -> None:
    from waqil_api.contracts import ProjectBuildStepWireV1

    with pytest.raises(Exception):
        ProjectBuildStepWireV1.model_validate(
            {"status": "complete", "response": "I created 16 files."}
        )


def test_the_build_wire_step_converts_to_a_tool_step() -> None:
    from waqil_api.contracts import ProjectBuildStepWireV1

    step = ProjectBuildStepWireV1.model_validate(
        {"status": "tool", "tool": "create_file",
         "arguments": {"path": "app/main.py", "content": "x\n"}}
    ).to_step()

    assert step.status == "tool"
    assert step.tool_call is not None
    assert step.tool_call.name == "create_file"
    assert step.tool_call.arguments == {"path": "app/main.py", "content": "x\n"}


def test_build_turn_stays_on_while_the_build_is_demonstrably_unfinished() -> None:
    """build_turn drives the narrowed grammar, and there is deliberately only
    one such flag. It means "this build is not done yet", which is true when
    nothing is staged and equally true when a planned file is still missing —
    staging one of eighteen files used to reopen finishing and let the model
    declare a whole build complete."""
    import types

    from waqil_api.control_plane import ControlPlane

    stub = types.SimpleNamespace(
        settings=types.SimpleNamespace(
            project_agent_max_steps=48,
            # The reference is read from disk on every build step; these
            # unit stubs point it at a directory that does not exist, so
            # the request carries an empty list rather than real files.
            project_reference_enabled=True,
            project_reference_dir=Path("/nonexistent-reference"),
            project_reference_max_chars=14_000,
            project_reference_max_chars_local=6_000,
        )
    )
    build = {"prompt": "Build out app/main.py from scratch and create requirements.txt."}
    question = {"prompt": "What does app/main.py do?"}
    staged = {"app/main.py": {"content": "x", "origin": "create", "base_sha256": "", "bytes": 1}}

    def request(state, staged_changes, planned=None) -> dict[str, Any]:
        return ControlPlane._project_step_request(
            stub, state, {}, [], staged_changes, 0, planned
        )

    assert request(build, {})["build_turn"] is True
    # No manifest: unchanged from before — one staged file reopens finishing.
    assert request(build, staged)["build_turn"] is False
    # Not a build instruction: never forced to call a tool.
    assert request(question, {})["build_turn"] is False

    # With a manifest the gate holds until every planned file exists.
    plan = ["app/main.py", "requirements.txt"]
    partial = request(build, staged, plan)
    assert partial["build_turn"] is True
    assert partial["files_still_to_write"] == ["requirements.txt"]

    done = request(build, {**staged, "requirements.txt": {"content": "fastapi\n"}}, plan)
    assert done["build_turn"] is False
    assert done["files_still_to_write"] == []
    # A plain question is never gated, manifest or not.
    assert request(question, {}, plan)["build_turn"] is False


def test_a_finish_with_planned_files_unwritten_is_declined_with_their_names() -> None:
    """The generalized guard. "Did you stage anything" passes a build that wrote
    five of eighteen files; "did you stage what you said you would" does not,
    and the model is told exactly which ones are missing rather than being left
    to work it out from a staged-file count."""
    from waqil_api.control_plane import ControlPlane

    state: dict[str, Any] = {"project_trace": []}
    result = ControlPlane._premature_finish(
        None, state, 4, 0, ["app/config.py", "README.md"]
    )

    # No response and no pending call: the router sends this back for another step.
    assert result["project_pending_call"] == {}
    assert "response_text" not in result
    assert result["project_empty_finish_streak"] == 1
    error = result["project_trace"][-1]["result"]["error"]
    assert "app/config.py" in error and "README.md" in error
    assert "still not staged" in error


def _execute_plane(outcome: Exception | None, **state_overrides: Any):
    """A control plane with just enough around _project_execute to run one tool."""
    from waqil_api.control_plane import ControlPlane

    class Workspace:
        async def execute_staged(self, project_id, call, staged, next_paths=()):
            if outcome is not None:
                raise outcome
            return {"ok": True}, dict(staged)

    plane = object.__new__(ControlPlane)
    plane.projects = Workspace()
    plane.events = SimpleNamespace(emit=_noop_emit)
    plane.settings = SimpleNamespace(project_verify_max_runs=2)
    plane._guard = _noop_emit
    plane._stage = _noop_emit
    state = {
        "run_id": "run_x",
        "conversation_id": "conv_x",
        "model_aliases": {"_project_id": "asset_x"},
        "project_trace": [],
        "project_pending_call": {
            "name": "create_file",
            "arguments": {"path": "a.py", "content": "x\n"},
        },
        **state_overrides,
    }
    return ControlPlane, plane, state


@pytest.mark.asyncio
async def test_only_an_argument_shaped_refusal_narrows_the_next_step() -> None:
    """The wire between the two halves of P3. A refusal the model can fix by
    resending sets project_retry_tool, which narrows the next grammar to that
    tool's required keys. A semantic one must not: pinning the model to a call
    whose target is simply unavailable trapped a live build in an eight-step
    retry loop, which is the failure this distinction exists to avoid."""
    from waqil_api.project_workspace import ProjectWorkspaceError

    ControlPlane, plane, state = _execute_plane(
        ProjectWorkspaceError("needs content", argument_shape=True)
    )
    fixable = await ControlPlane._project_execute(plane, state)
    assert fixable["project_retry_tool"] == "create_file"

    ControlPlane, plane, state = _execute_plane(
        ProjectWorkspaceError("refuses to overwrite an existing file")
    )
    semantic = await ControlPlane._project_execute(plane, state)
    assert semantic["project_retry_tool"] == ""

    # And a call that works clears any narrowing the previous step set.
    ControlPlane, plane, state = _execute_plane(None, project_retry_tool="create_file")
    assert (await ControlPlane._project_execute(plane, state))["project_retry_tool"] == ""


@pytest.mark.asyncio
async def test_a_repeated_read_returns_the_answer_and_counts_toward_the_breaker() -> None:
    """The livelock a real edit turn hit: 44 of 48 steps re-reading one file.

    Two faults compounded. The refusal said the earlier result was "still
    above", but each refusal is itself a trace entry, so a run of them pushed
    the successful read out of the bounded window — the model was sent to look
    at something it could no longer see. And the repeat guard returned before
    the no-progress breaker was consulted, so a repeated read could never trip
    it and the only ceiling was the step budget. The turn ended with nothing
    staged."""
    from waqil_api.control_plane import _MAX_TARGET_REFUSALS

    read = {"name": "read_file", "arguments": {"path": "app/jobs.py"}}
    ControlPlane, plane, state = _execute_plane(None, project_pending_call=read)
    state["project_trace"] = [
        {
            "tool": "read_file",
            "arguments": {"path": "app/jobs.py"},
            "result": {"ok": True, "output": {"content": "class Job:\n    pass\n"}},
        }
    ]

    for attempt in range(1, _MAX_TARGET_REFUSALS + 1):
        result = await ControlPlane._project_execute(plane, state)
        entry = result["project_trace"][-1]["result"]
        # The bytes travel with the refusal instead of a pointer to them.
        assert entry["output"] == {"content": "class Job:\n    pass\n"}
        assert "repeated below" in entry["error"]
        # And it counts, so the breaker can eventually stop the loop.
        assert result["project_blocked_targets"]["read_file:app/jobs.py"] == attempt
        state = {**state, **result, "project_pending_call": read}

    closed = await ControlPlane._project_execute(plane, state)
    assert "closed for this turn" in closed["project_trace"][-1]["result"]["error"]


@pytest.mark.asyncio
async def test_a_stalled_turn_gets_its_honest_exit_back() -> None:
    """The manifest gate withholds `complete`, which is right while the model is
    making progress and a trap when it cannot make any. An edit turn whose
    planned file already exists on disk is only satisfiable by a patch; a model
    that cannot produce one has no legal move — it cannot write, and it cannot
    finish. Once the overlay has gone unchanged for long enough the gate has to
    release, so the turn ends with an account of what it could not do."""
    import types

    from waqil_api.control_plane import _MAX_STALL_STEPS, ControlPlane

    stub = types.SimpleNamespace(
        settings=types.SimpleNamespace(
            project_agent_max_steps=48,
            # The reference is read from disk on every build step; these
            # unit stubs point it at a directory that does not exist, so
            # the request carries an empty list rather than real files.
            project_reference_enabled=True,
            project_reference_dir=Path("/nonexistent-reference"),
            project_reference_max_chars=14_000,
            project_reference_max_chars_local=6_000,
        )
    )
    # Verbatim from the turn that livelocked, so the test covers the real case.
    build = {
        "prompt": (
            "Add a per-job timeline: record each status transition with a "
            "timestamp in app/jobs.py, return it from GET /api/jobs/{id}."
        )
    }
    plan = ["app/jobs.py"]

    def request(stall: int) -> dict[str, Any]:
        return ControlPlane._project_step_request(
            stub, {**build, "project_stall_steps": stall}, {}, [], {}, 0, plan
        )

    # Still moving: the gate holds and finishing stays out of the grammar.
    working = request(_MAX_STALL_STEPS - 1)
    assert working["build_turn"] is True
    assert working["files_still_to_write"] == ["app/jobs.py"]

    # Stuck: the gate releases so the model can say so.
    stalled = request(_MAX_STALL_STEPS)
    assert stalled["build_turn"] is False
    assert stalled["files_still_to_write"] == []


@pytest.mark.asyncio
async def test_progress_is_measured_in_staged_bytes_not_steps_taken() -> None:
    """The stall counter has to track the overlay, not the clock: a step that
    stages something is progress however long it took, and a step that changes
    nothing is not progress however busy it looked."""
    from waqil_api.project_workspace import ProjectWorkspaceError

    ControlPlane, plane, state = _execute_plane(ProjectWorkspaceError("already exists"))
    refused = await ControlPlane._project_execute(plane, state)
    assert refused["project_stall_steps"] == 1

    ControlPlane, plane, state = _execute_plane(None, project_stall_steps=5)
    plane.projects = SimpleNamespace(execute_staged=_stages_a_file)
    progressed = await ControlPlane._project_execute(plane, state)
    assert progressed["project_stall_steps"] == 0


@pytest.mark.asyncio
async def test_a_refused_write_target_carries_the_files_still_owed() -> None:
    """The refusal is semantic, so it must not narrow to the same tool — but it
    is the one semantic refusal with a mechanical answer, and the host knows the
    manifest. The next step is pinned to the owed files plus the refused path."""
    ControlPlane, plane, state = _execute_plane(
        ProjectWorkspaceError("refuses to overwrite", wrong_target=True),
        project_planned_files=["a.py", "b.py", "c.py"],
        project_staged={"a.py": {"content": "x\n"}},
    )
    refused = await ControlPlane._project_execute(plane, state)

    assert refused["project_retry_tool"] == ""
    assert refused["project_write_pin"] == ["b.py", "c.py", "a.py"]

    # With the manifest complete there is nothing owed, so nothing is pinned.
    ControlPlane, plane, state = _execute_plane(
        ProjectWorkspaceError("refuses to overwrite", wrong_target=True),
        project_planned_files=["a.py"],
        project_staged={"a.py": {"content": "x\n"}},
    )
    assert (await ControlPlane._project_execute(plane, state))["project_write_pin"] == []


@pytest.mark.asyncio
async def test_a_successful_call_clears_the_write_pin() -> None:
    """A narrowing that outlives the refusal that justified it would restrict
    the model's write targets for the rest of the turn."""
    ControlPlane, plane, state = _execute_plane(
        None,
        project_planned_files=["a.py", "b.py"],
        project_write_pin=["b.py", "a.py"],
    )

    assert (await ControlPlane._project_execute(plane, state))["project_write_pin"] == []


async def _stages_a_file(project_id, call, staged, next_paths=()):
    """A workspace call that actually changes the overlay."""
    return {"ok": True}, {**staged, "a.py": {"content": "x\n"}}


@pytest.mark.asyncio
async def test_a_target_that_keeps_failing_is_closed_for_the_turn() -> None:
    """P4's no-progress breaker. Before it, the step budget was the only thing
    that stopped a model repeating a call it could not get past — one spent
    eight consecutive steps re-creating a file it had already staged, because
    the refusal never suggested anything else."""
    from waqil_api.control_plane import _MAX_TARGET_REFUSALS
    from waqil_api.project_workspace import ProjectWorkspaceError

    ControlPlane, plane, state = _execute_plane(ProjectWorkspaceError("already exists"))
    for attempt in range(1, _MAX_TARGET_REFUSALS + 1):
        result = await ControlPlane._project_execute(plane, state)
        assert result["project_blocked_targets"]["create_file:a.py"] == attempt
        state = {**state, **result, "project_pending_call": state["project_pending_call"]}

    # Past the limit the workspace is not consulted at all, and the model is
    # told to do something else rather than handed the same refusal again.
    plane.projects = SimpleNamespace(
        execute_staged=_unreachable_tool  # would raise if the breaker did not fire
    )
    closed = await ControlPlane._project_execute(plane, state)
    error = closed["project_trace"][-1]["result"]["error"]
    assert "closed for this turn" in error
    assert "Do something different" in error


async def _unreachable_tool(*args: Any, **kwargs: Any):
    """A workspace call that must not happen once a target is closed."""
    raise AssertionError("the breaker should have short-circuited this call")


def test_the_step_request_carries_the_tool_catalog_the_local_model_needs() -> None:
    """The local model is given no function schemas — the system prompt names
    the tools and never their arguments, which is why it kept sending
    apply_patch a "patch" key. Grok has had this all along."""
    import types

    from waqil_api.control_plane import ControlPlane

    stub = types.SimpleNamespace(
        settings=types.SimpleNamespace(
            project_agent_max_steps=48,
            # The reference is read from disk on every build step; these
            # unit stubs point it at a directory that does not exist, so
            # the request carries an empty list rather than real files.
            project_reference_enabled=True,
            project_reference_dir=Path("/nonexistent-reference"),
            project_reference_max_chars=14_000,
            project_reference_max_chars_local=6_000,
        )
    )
    request = ControlPlane._project_step_request(
        stub, {"prompt": "Build app/main.py from scratch."}, {}, [], {}, 0
    )
    catalog = {tool["name"]: tool for tool in request["available_tools"]}
    assert catalog["apply_patch"]["required_arguments"] == [
        "path",
        "original",
        "replacement",
    ]
    assert catalog["create_file"]["required_arguments"] == ["path", "content"]
    assert "not a diff" in catalog["apply_patch"]["note"]
    assert "cannot be empty" in catalog["create_file"]["note"]


def test_a_planned_path_is_normalized_the_way_the_overlay_keys_it() -> None:
    """A planned path is compared by equality against the overlay's keys, so any
    normalization that disagrees with the workspace leaves the gate waiting for
    a file that was already written. Dotfiles are the trap: lstrip("./") takes a
    character set, not a prefix, and would turn ".gitignore" into "gitignore"."""
    from waqil_api.contracts import ProjectBuildPlanV1

    plan = ProjectBuildPlanV1(
        files=[
            "./app/main.py",
            ".gitignore",
            ".github/workflows/ci.yml",
            "/etc/passwd",
            "../escape.py",
            "a.py",
            "a.py",
            "  b.py  ",
            "",
        ]
    )
    assert plan.files == [
        "app/main.py",
        ".gitignore",
        ".github/workflows/ci.yml",
        "a.py",
        "b.py",
    ]
    # Removing the lstrip also puts the two guards below it back in reach: while
    # it was stripping the leading "/" itself, startswith("/") could never fire.
    assert "/etc/passwd" not in plan.files
    assert not any(".." in path for path in plan.files)


@pytest.mark.asyncio
async def test_a_planned_dotfile_matches_the_key_the_overlay_actually_stores(
    tmp_path: Path,
) -> None:
    """The invariant that spans the two modules, which asserting either side
    alone would miss: the gate compares a planned path to an overlay key by
    equality, so the manifest's normalization and the workspace's have to agree
    exactly. They disagreed on dotfiles, and PROJECT_PLAN_SYSTEM asks the planner
    for configuration files — so .gitignore is the common case, not an exotic
    one, and the gate would have waited forever for a file already written."""
    from waqil_api.contracts import ProjectBuildPlanV1

    _project(tmp_path)
    service, asset_id = await _service(_settings(tmp_path))
    planned = ProjectBuildPlanV1(
        files=["./.gitignore", ".github/workflows/ci.yml"]
    ).files

    staged: dict[str, Any] = {}
    for path in planned:
        _, staged = await service.execute_staged(
            asset_id,
            ProjectToolCallV1(
                name="create_file", arguments={"path": path, "content": "x\n"}
            ),
            staged,
        )

    assert sorted(staged) == sorted(planned)
    assert [path for path in planned if path not in staged] == []


@pytest.mark.asyncio
async def test_the_manifest_is_taken_after_looking_around_and_costs_no_step(
    tmp_path: Path,
) -> None:
    """The plan is a property of the turn, not one of the model's steps.

    It runs inside the existing node rather than as a graph node of its own, so
    the checkpoint schema and the recursion budget are untouched, and it must
    not consume a project_iteration — the step budget is tight enough that
    charging one pushes real work past it.

    It is also taken AFTER the loop has read a little, not before: a plan
    written from the request alone is a plan the project can contradict, and
    an owed manifest narrows create_file to exactly what it named."""
    from waqil_api.control_plane import ControlPlane, _PLAN_AFTER_STEPS

    calls = {"count": 0}

    class PlanningModel:
        async def project_plan_files(self, request, *, model_aliases=None):
            calls["count"] += 1
            return ["alpha.txt", "beta.txt"]

    plane = SimpleNamespace(
        model=PlanningModel(),
        settings=SimpleNamespace(project_staged_max_files=48),
        events=SimpleNamespace(emit=_noop_emit),
    )
    state = {
        "prompt": "Build out a new service from scratch and create every file it needs.",
        "run_id": "run_x",
        "conversation_id": "conv_x",
    }

    # Not while the loop is still looking around.
    for early in range(_PLAN_AFTER_STEPS):
        assert (await ControlPlane._project_manifest(plane, state, {}, early, {}))[
            "files"
        ] is None
    assert calls["count"] == 0

    plan = await ControlPlane._project_manifest(
        plane, state, {}, _PLAN_AFTER_STEPS, {}
    )
    assert plan["files"] == ["alpha.txt", "beta.txt"]
    assert plan["scenarios"] == []
    assert plan["taken"] is True
    assert calls["count"] == 1

    # Never re-asked: once per turn, then carried in state.
    carried = await ControlPlane._project_manifest(
        plane, {**state, "project_planned_files": plan["files"]}, {}, 9, {}
    )
    assert carried["files"] == plan["files"]
    assert carried["taken"] is False
    # Nor after a plan that legitimately named nothing — `taken` is what stops
    # a question being re-planned on every step that follows it.
    assert (
        await ControlPlane._project_manifest(
            plane, {**state, "project_plan_taken": True}, {}, 9, {}
        )
    )["taken"] is False
    assert calls["count"] == 1

    # A turn that staged a file before the threshold is planned at once: the
    # gate cannot be established late once work is already in the overlay.
    early_stage = await ControlPlane._project_manifest(plane, state, {}, 0, {"a": {}})
    assert early_stage["files"] == ["alpha.txt", "beta.txt"]
    assert calls["count"] == 2

    # The prefilter still keeps plan calls off turns that are not about code.
    assert (
        await ControlPlane._project_manifest(
            plane, {**state, "prompt": "Thanks, that all looks good."}, {}, 9, {}
        )
    )["files"] is None
    assert calls["count"] == 2


@pytest.mark.asyncio
async def test_the_plan_declares_the_turns_intent_and_the_regexes_step_back() -> None:
    """Classification moves to the model that has the repository in front of it.

    The regex is a keyword match on the user's wording, and it is wrong in both
    directions: a question that happens to contain "add" and a source path
    reads as a build, and a whole-application rewire reads as no build at all
    (it never says "create" and names no file). A provider that declares intent
    now decides; one that does not keeps the old predicate exactly."""
    from waqil_api.control_plane import _writes_files

    # Over-fires: a question carrying a create verb and a path.
    question = "Where would I add a new route in app/main.py?"
    assert _writes_files({"prompt": question}) is True
    assert _writes_files({"prompt": question, "project_build_intent": "question"}) is False

    # Under-fires: the live Logivity turn, which writes across the whole UI.
    rewire = "Rewire this app's entire UI onto the Metis design language in appkit."
    assert _writes_files({"prompt": rewire}) is False
    assert _writes_files({"prompt": rewire, "project_build_intent": "edit"}) is True
    # "build" and "edit" both write; only "question" turns the gate off.
    assert _writes_files({"prompt": rewire, "project_build_intent": "build"}) is True


def test_an_infra_failure_is_not_reported_as_an_unreadable_model_reply() -> None:
    """A 429/quota/5xx/timeout is the backend failing to answer, not the model
    replying unintelligibly. Live proof: an exhausted Cohere trial key (HTTP
    429) drove the loop to say "I could not read 3 replies from the model" —
    blaming the model for the backend being out of quota."""
    from waqil_api.model_provider import (
        ModelProviderError,
        PermanentModelError,
        classify_backend_unavailable,
    )

    quota = ModelProviderError(
        'Cohere returned HTTP 429: {"message":"You are using a Trial key, '
        'which is limited to 1000 API calls / month"}'
    )
    assert classify_backend_unavailable(quota) == "rate_limited"
    assert classify_backend_unavailable(ModelProviderError("Cohere kept failing after 3 attempts")) == "backend_error"
    assert classify_backend_unavailable(ModelProviderError("Grok call timed out after 120 seconds")) == "backend_timeout"

    # Genuinely malformed model replies stay malformed — the model DID answer,
    # unreadably, and the loop must feed that back as evidence, not end the turn.
    assert classify_backend_unavailable(ModelProviderError("hosted model returned prose instead of a project tool call")) is None
    assert classify_backend_unavailable(ModelProviderError("Grok returned invalid project tool arguments")) is None

    # A permanent pre-model refusal is classify_model_error's job; this must
    # not shadow it (both would end the turn, but with different guidance).
    assert classify_backend_unavailable(PermanentModelError("failed to parse grammar", reason="grammar_compile")) is None


@pytest.mark.asyncio
async def test_a_backend_outage_ends_the_turn_honestly_not_as_malformed() -> None:
    """The loop half of Finding D: a provider that raises a 429 must end the
    turn with the honest 'backend unavailable' message and NOT advance the
    malformed-reply streak."""
    from waqil_api.control_plane import ControlPlane
    from waqil_api.model_provider import ModelProviderError

    class QuotaModel:
        async def project_plan_files(self, request, *, model_aliases=None):
            return SimpleNamespace(files=[], scenarios=[], intent="edit", scope="narrow")

        async def project_step(self, request, *, model_aliases=None):
            raise ModelProviderError('Cohere returned HTTP 429: {"message":"Trial key"}')

    plane = object.__new__(ControlPlane)
    plane.model = QuotaModel()
    plane.events = SimpleNamespace(emit=_noop_emit)
    plane.projects = SimpleNamespace(context=_empty_context, stage_scaffold=None)
    plane.settings = SimpleNamespace(
        project_agent_max_steps=48, project_staged_max_files=48,
        project_spec_rewrite=False, project_spec_rewrite_max_chars=1800,
        project_reference_enabled=False, project_reference_dir=Path("/none"),
        project_reference_max_chars=0, project_reference_max_chars_local=0,
    )
    plane._guard = _noop_emit
    plane._stage = _noop_emit

    result = await ControlPlane._project_step(plane, {
        "prompt": "Add a /health route to app/main.py.",
        "run_id": "run_x", "conversation_id": "conv_x",
        "model_aliases": {"_project_id": "asset_x"},
        "project_iterations": 0,
    })

    assert "rate-limited or out of quota" in result["response_text"]
    # Not counted as an unreadable reply.
    assert result.get("project_malformed_streak", 0) == 0
    assert "could not read" not in result["response_text"]


@pytest.mark.asyncio
async def test_a_run_of_reads_without_a_write_ends_the_turn() -> None:
    """Finding B: reads succeed, so neither the refusal streak nor the stall
    gate stops a model that just keeps reading. The exploration ceiling does —
    one live turn read 40+ files in a row and hit the step budget with nothing
    staged."""
    from waqil_api.control_plane import ControlPlane, _explore_budget

    plane = object.__new__(ControlPlane)
    plane.projects = SimpleNamespace(context=_empty_context)
    plane.events = SimpleNamespace(emit=_noop_emit)
    plane.settings = SimpleNamespace(
        project_agent_max_steps=48, project_verify_bonus_steps=0,
    )
    plane._guard = _noop_emit
    plane._stage = _noop_emit
    base = {
        "prompt": "Rework the dashboard.",
        "run_id": "r", "conversation_id": "c",
        "model_aliases": {"_project_id": "asset_x"},
        "project_iterations": 20,
    }

    # Nothing staged: the honest dead-end message, not an offer.
    dry = await ControlPlane._project_step(
        plane, {**base, "project_consecutive_reads": _explore_budget(base)}
    )
    assert "without writing" in dry["response_text"]
    assert dry["project_consecutive_reads"] == 0

    # With staged work: the run of reads still ends the turn, but offers it.
    staged = {**base, "project_staged": {"app/main.py": {"bytes": 10}}}
    offered = await ControlPlane._project_step(
        plane, {**staged, "project_consecutive_reads": _explore_budget(staged)}
    )
    assert "ready to review" in offered["response_text"]


def test_the_exploration_budget_scales_with_what_was_planned() -> None:
    """A flat ceiling is wrong at both ends.

    A live GLM-5.2 turn planning a five-file whole-app conversion read sixteen
    things — the project plus the vendored appkit, roughly once each — and a
    flat sixteen cut it off exactly when exploration was done and writing was
    next. A one-file edit that has read ten times is already lost. So the
    allowance follows the plan, and stays capped."""
    from waqil_api.control_plane import _explore_budget, _explore_nudge_at

    unplanned = _explore_budget({"project_planned_files": []})
    one_file = _explore_budget({"project_planned_files": ["a.py"]})
    whole_app = _explore_budget(
        {"project_planned_files": ["a.py", "b.py", "c.py", "d.py", "e.py"]}
    )
    assert unplanned < one_file < whole_app
    # The GLM conversion (5 planned files, 16 reads) now survives to write.
    assert whole_app > 16
    # And nothing reads forever: the live kimi pathology (3 planned, 40+ reads)
    # is still ended well before the step budget.
    assert _explore_budget({"project_planned_files": ["a", "b", "c"]}) < 40
    assert _explore_budget({"project_planned_files": ["x"] * 50}) <= 28
    # The nudge always lands before the end, with room to act on it.
    for plan in ([], ["a"], ["a"] * 5, ["a"] * 50):
        state = {"project_planned_files": plan}
        assert _explore_nudge_at(state) < _explore_budget(state)


@pytest.mark.asyncio
async def test_a_web_reskin_seeds_appkit_on_step_one_without_the_plan() -> None:
    """The Logivity lesson, twice over: appkit is what a reskin needs, and the
    plan call that would carry the intent is unreliable on the hosted lane — so
    the scaffold must be seeded from the REQUEST, up front, with no model call.
    And seeding it must not count as model progress, or plan-after-exploration
    would collapse to step one just because appkit was staged."""
    from waqil_api.control_plane import ControlPlane, _model_has_written
    from waqil_api.contracts import ProjectAgentStepV1, ProjectToolCallV1

    seeded_calls = {"n": 0}

    async def _stage_scaffold(project_id, staged, capabilities):
        seeded_calls["n"] += 1
        nxt = dict(staged)
        nxt["appkit/static/theme.css"] = {"content": ":root{}", "origin": "create", "bytes": 6}
        nxt["appkit/web.py"] = {"content": "x=1", "origin": "create", "bytes": 3}
        return nxt, ["appkit/static/theme.css", "appkit/web.py"]

    plans = {"n": 0}

    class Model:
        async def project_plan_files(self, request, *, model_aliases=None):
            plans["n"] += 1
            # The hosted lane's unreliability, reproduced: the plan call raises,
            # so nothing downstream can depend on it having run.
            raise RuntimeError("hosted structured decode failed")

        async def project_step(self, request, *, model_aliases=None):
            # Assert the scaffold reached the model on the very first step.
            assert request["scaffold"], "appkit note must be present on step 1"
            return ProjectAgentStepV1(
                status="tool", tool_call=ProjectToolCallV1(name="list_files", arguments={}),
            )

    plane = object.__new__(ControlPlane)
    plane.model = Model()
    plane.events = SimpleNamespace(emit=_noop_emit)
    plane.projects = SimpleNamespace(context=_empty_context, stage_scaffold=_stage_scaffold)
    plane.settings = SimpleNamespace(
        project_agent_max_steps=48, project_staged_max_files=48,
        project_spec_rewrite=False, project_spec_rewrite_max_chars=1800,
        project_reference_enabled=False, project_reference_dir=Path("/none"),
        project_reference_max_chars=0, project_reference_max_chars_local=0,
    )
    plane._guard = _noop_emit
    plane._stage = _noop_emit

    result = await ControlPlane._project_step(plane, {
        "prompt": "Rewire this app's entire UI onto the Metis design language in appkit, "
                  "mount it with mount_appkit_static(app), link /appkit/theme.css first.",
        "run_id": "run_x", "conversation_id": "conv_x",
        "model_aliases": {"_project_id": "asset_x"},
        "project_iterations": 0,
    })

    # appkit was seeded from the request, with no working plan behind it.
    staged = result["project_staged"]
    assert any(p.startswith("appkit/") for p in staged)
    assert seeded_calls["n"] == 1
    # And the seed is not model progress: _model_has_written sees past it, so
    # the plan-after-exploration gate would still hold.
    assert _model_has_written(staged) is False


@pytest.mark.asyncio
async def test_a_no_op_revise_plan_does_not_burn_a_revision() -> None:
    """Finding C: a revision that restates the current plan (one live run did it
    twice) is accepted but not counted, and the model is pointed back at the
    work rather than re-announced to."""
    from waqil_api.control_plane import ControlPlane

    plane = object.__new__(ControlPlane)
    plane.events = SimpleNamespace(emit=_noop_emit)
    plane.settings = SimpleNamespace(project_staged_max_files=48)
    state = {
        "prompt": "x", "run_id": "r", "conversation_id": "c",
        "project_planned_files": ["app/main.py", "app/static/index.html"],
        "project_plan_revisions": 0, "project_trace": [],
    }
    call = ProjectToolCallV1(
        name="revise_plan",
        arguments={"files": ["app/main.py", "app/static/index.html"], "reason": "same"},
    )
    result = await ControlPlane._revise_project_plan(plane, state, call, 5, {})
    assert result.get("project_plan_revisions", 0) == 0  # not counted
    assert "unchanged" in result["project_trace"][-1]["result"]["output"]["note"]


@pytest.mark.asyncio
async def test_planning_the_build_does_not_spend_one_of_the_models_steps() -> None:
    """The manifest is a property of the turn, not one of the model's steps.

    A step spent here would come straight out of the build: the budget is tight
    enough that one test pins project_agent_max_steps at 4 and depends on four
    real tool calls fitting. So a first step that takes a manifest must advance
    project_iterations exactly once, the same as any other step."""
    from waqil_api.contracts import ProjectAgentStepV1, ProjectToolCallV1
    from waqil_api.control_plane import ControlPlane

    class PlanningModel:
        async def project_plan_files(self, request, *, model_aliases=None):
            return ["alpha.txt"]

        async def project_step(self, request, *, model_aliases=None):
            return ProjectAgentStepV1(
                status="tool",
                tool_call=ProjectToolCallV1(
                    name="create_file",
                    arguments={"path": "alpha.txt", "content": "a\n"},
                ),
            )

    plane = object.__new__(ControlPlane)
    plane.model = PlanningModel()
    plane.events = SimpleNamespace(emit=_noop_emit)
    plane.projects = SimpleNamespace(context=_empty_context)
    plane.settings = SimpleNamespace(
        project_agent_max_steps=48,
        project_staged_max_files=48,
        # The spec-rewrite stage stands down here; these tests are about the
        # manifest and the guards, not the rewrite (test_spec_rewrite owns that).
        project_spec_rewrite=False,
        project_spec_rewrite_max_chars=1800,
        # Reference lookup is part of every build step; point it at nothing so
        # these manifest tests stay about the manifest.
        project_reference_enabled=True,
        project_reference_dir=Path("/nonexistent-reference"),
        project_reference_max_chars=14_000,
        project_reference_max_chars_local=6_000,
    )
    plane._guard = _noop_emit
    plane._stage = _noop_emit

    from waqil_api.control_plane import _PLAN_AFTER_STEPS

    result = await ControlPlane._project_step(
        plane,
        {
            "prompt": "Build out alpha.txt from scratch.",
            "run_id": "run_x",
            "conversation_id": "conv_x",
            "model_aliases": {"_project_id": "asset_x"},
            "project_iterations": _PLAN_AFTER_STEPS,
        },
    )

    assert result["project_planned_files"] == ["alpha.txt"]
    # The step, and only the step: planning rides inside it and charges nothing.
    assert result["project_iterations"] == _PLAN_AFTER_STEPS + 1
    assert result["project_pending_call"]["name"] == "create_file"


@pytest.mark.asyncio
async def test_the_manifest_survives_an_unreadable_reply_on_the_step_it_lands() -> None:
    """The manifest is taken once, so it has to be carried out of every outcome
    of the step that takes it — including the ones that return early. Losing it
    to a single malformed reply would drop the gate for the rest of the turn,
    and the next step, no longer the one that asks, would never ask again."""
    from waqil_api.control_plane import ControlPlane, _PLAN_AFTER_STEPS
    from waqil_api.model_provider import ModelProviderError

    class UnreadableModel:
        async def project_plan_files(self, request, *, model_aliases=None):
            return ["alpha.txt", "beta.txt"]

        async def project_step(self, request, *, model_aliases=None):
            raise ModelProviderError("model returned invalid ProjectBuildStepWireV1")

    plane = object.__new__(ControlPlane)
    plane.model = UnreadableModel()
    plane.events = SimpleNamespace(emit=_noop_emit)
    plane.projects = SimpleNamespace(context=_empty_context)
    plane.settings = SimpleNamespace(
        project_agent_max_steps=48,
        project_staged_max_files=48,
        # The spec-rewrite stage stands down here; these tests are about the
        # manifest and the guards, not the rewrite (test_spec_rewrite owns that).
        project_spec_rewrite=False,
        project_spec_rewrite_max_chars=1800,
        # Reference lookup is part of every build step; point it at nothing so
        # these manifest tests stay about the manifest.
        project_reference_enabled=True,
        project_reference_dir=Path("/nonexistent-reference"),
        project_reference_max_chars=14_000,
        project_reference_max_chars_local=6_000,
    )
    plane._guard = _noop_emit
    plane._stage = _noop_emit
    state = {
        "prompt": "Build out alpha.txt and beta.txt from scratch.",
        "run_id": "run_x",
        "conversation_id": "conv_x",
        "model_aliases": {"_project_id": "asset_x"},
        "project_iterations": _PLAN_AFTER_STEPS,
    }

    result = await ControlPlane._project_step(plane, state)

    assert result["project_planned_files"] == ["alpha.txt", "beta.txt"]
    assert result["project_plan_taken"] is True
    assert result["project_malformed_streak"] == 1


@pytest.mark.asyncio
async def test_a_manifest_the_model_cannot_produce_leaves_the_loop_as_it_was() -> None:
    """The gate only sharpens an existing guard, so losing it must cost nothing
    else. A build with no manifest behaves exactly as it did before."""
    from waqil_api.control_plane import ControlPlane, _PLAN_AFTER_STEPS

    class FailingModel:
        async def project_plan_files(self, request, *, model_aliases=None):
            raise RuntimeError("the planner call failed")

    plane = SimpleNamespace(
        model=FailingModel(),
        settings=SimpleNamespace(project_staged_max_files=48),
        events=SimpleNamespace(emit=_noop_emit),
    )
    plan = await ControlPlane._project_manifest(
        plane,
        {
            "prompt": "Build out the whole app from scratch.",
            "run_id": "r",
            "conversation_id": "c",
        },
        {},
        _PLAN_AFTER_STEPS,
        {},
    )
    # Infra failure degrades; [] is reserved for "asked, got nothing".
    assert plan["files"] is None
    # And it is not recorded as taken, so a later step may still get a plan.
    assert plan["taken"] is False


@pytest.mark.asyncio
async def test_a_plan_taken_this_step_reaches_this_steps_request() -> None:
    """The declaration must not be a step late.

    The step that takes the plan is the one that has to act on it: a turn the
    model has just called a question would otherwise still be sent the build
    grammar, in which answering the user is not expressible at all."""
    from waqil_api.control_plane import ControlPlane, _PLAN_AFTER_STEPS

    seen: list[dict[str, Any]] = []

    class QuestionModel:
        async def project_plan_files(self, request, *, model_aliases=None):
            return SimpleNamespace(files=[], scenarios=[], intent="question", scope="narrow")

        async def project_step(self, request, *, model_aliases=None):
            seen.append(request)
            from waqil_api.contracts import ProjectAgentStepV1, ProjectToolCallV1

            return ProjectAgentStepV1(
                status="tool",
                tool_call=ProjectToolCallV1(
                    name="respond", arguments={"message": "It is a Streamlit app."}
                ),
            )

    plane = object.__new__(ControlPlane)
    plane.model = QuestionModel()
    plane.events = SimpleNamespace(emit=_noop_emit)
    plane.projects = SimpleNamespace(context=_empty_context)
    plane.settings = SimpleNamespace(
        project_agent_max_steps=48,
        project_staged_max_files=48,
        project_spec_rewrite=False,
        project_spec_rewrite_max_chars=1800,
        project_reference_enabled=False,
        project_reference_dir=Path("/nonexistent-reference"),
        project_reference_max_chars=0,
        project_reference_max_chars_local=0,
    )
    plane._guard = _noop_emit
    plane._stage = _noop_emit

    result = await ControlPlane._project_step(
        plane,
        {
            # The regex reads this as a build: it carries "add" and a source path.
            "prompt": "Where would I add a new route in app/main.py?",
            "run_id": "run_x",
            "conversation_id": "conv_x",
            "model_aliases": {"_project_id": "asset_x"},
            "project_iterations": _PLAN_AFTER_STEPS,
        },
    )

    assert seen[0]["build_turn"] is False
    assert result["project_build_intent"] == "question"
    assert result["project_plan_taken"] is True
    assert result["response_text"] == "It is a Streamlit app."


@pytest.mark.asyncio
async def test_batched_reads_all_run_inside_the_one_step_that_carried_them() -> None:
    """The loop half of read batching: every read in the reply is executed and
    lands in the trace, without a model round-trip between them — and the batch
    is cleared afterwards so it cannot be replayed against a later step."""
    from waqil_api.control_plane import ControlPlane

    seen: list[str] = []

    class Workspace:
        async def execute_staged(self, project_id, call, staged, next_paths=()):
            seen.append(f"{call.name}:{call.arguments.get('path', '')}")
            return {"ok": True, "lines": []}, None

    plane = object.__new__(ControlPlane)
    plane.projects = Workspace()
    plane.events = SimpleNamespace(emit=_noop_emit)
    plane.settings = SimpleNamespace(project_verify_max_runs=2)
    plane._guard = _noop_emit
    plane._stage = _noop_emit
    state = {
        "run_id": "run_x",
        "conversation_id": "conv_x",
        "model_aliases": {"_project_id": "asset_x"},
        "project_trace": [],
        "project_pending_call": {"name": "list_files", "arguments": {}},
        "project_pending_reads": [
            {"name": "read_file", "arguments": {"path": "app/main.py"}},
            {"name": "read_file", "arguments": {"path": "app/config.py"}},
            # The same read twice: the repeat guard sees the trace as it grows,
            # so the second is answered from the first rather than re-run.
            {"name": "read_file", "arguments": {"path": "app/main.py"}},
        ],
    }

    result = await ControlPlane._project_execute(plane, state)

    assert seen == ["list_files:", "read_file:app/main.py", "read_file:app/config.py"]
    assert [entry["tool"] for entry in result["project_trace"]] == [
        "list_files",
        "read_file",
        "read_file",
        "read_file",
    ]
    assert result["project_trace"][-1]["result"]["ok"] is False
    assert result["project_pending_reads"] == []


@pytest.mark.asyncio
async def test_a_refused_step_still_clears_its_batch() -> None:
    """_project_evidence is on every path out of a step, including the ones
    that refuse the primary call and return early. A batch left behind on one
    of those would be run against the next step's unrelated pending call."""
    from waqil_api.project_workspace import ProjectWorkspaceError

    ControlPlane, plane, state = _execute_plane(
        ProjectWorkspaceError("refuses to overwrite an existing file"),
        project_pending_reads=[{"name": "read_file", "arguments": {"path": "a.py"}}],
    )
    assert (await ControlPlane._project_execute(plane, state))[
        "project_pending_reads"
    ] == []


@pytest.mark.asyncio
async def test_revise_plan_replaces_a_manifest_the_project_contradicts() -> None:
    """The way out of a plan the evidence disproved.

    This is the live Logivity deadlock in miniature: a manifest naming
    `app/static/index.html` in a project that is a Streamlit script, which
    narrowed create_file's path to an enum of two files that could not exist
    there. The model had explored, knew the plan was wrong, and had no legal
    move it believed in — so it generated calls its own backend refused, three
    in a row, and the turn ended having written nothing."""
    from waqil_api.control_plane import ControlPlane, _MAX_PLAN_REVISIONS

    events: list[tuple[str, dict[str, Any]]] = []

    async def _record(run_id, conversation_id, kind, payload):
        events.append((kind, payload))

    plane = object.__new__(ControlPlane)
    plane.events = SimpleNamespace(emit=_record)
    plane.settings = SimpleNamespace(project_staged_max_files=48)
    state = {
        "prompt": "Rewire this app's entire UI onto the Metis design language.",
        "run_id": "run_x",
        "conversation_id": "conv_x",
        "project_planned_files": ["app/static/index.html", "app/static/style.css"],
        "project_planned_scenarios": [{"name": "homepage loads"}],
        "project_trace": [],
    }
    call = ProjectToolCallV1(
        name="revise_plan",
        arguments={
            "files": ["app.py"],
            "reason": "This is a Streamlit app: it has no app/static and serves no HTML.",
        },
    )

    result = await ControlPlane._revise_project_plan(plane, state, call, 5, {})

    assert result["project_planned_files"] == ["app.py"]
    assert result["project_plan_revisions"] == 1
    # A corrected plan is progress, so it must not advance the stall counter
    # that releases the manifest gate.
    assert result["project_stall_steps"] == 0
    # The old plan's acceptance scenarios described files that are gone.
    assert result["project_planned_scenarios"] == []
    # It costs the step it took, and it never becomes a pending workspace call.
    assert result["project_iterations"] == 6
    assert result["project_pending_call"] == {}
    assert result["project_trace"][-1]["result"]["ok"] is True
    assert [kind for kind, _ in events] == ["project.plan_revised"]
    assert events[0][1]["previous_files"] == [
        "app/static/index.html",
        "app/static/style.css",
    ]

    # Bounded: re-planning instead of writing looks identical for two turns.
    spent = {**state, "project_plan_revisions": _MAX_PLAN_REVISIONS}
    refused = await ControlPlane._revise_project_plan(plane, spent, call, 5, {})
    assert refused["project_trace"][-1]["result"]["ok"] is False
    assert "limit for one turn" in refused["project_trace"][-1]["result"]["error"]
    assert "project_planned_files" not in refused


@pytest.mark.asyncio
async def test_revise_plan_never_reaches_the_workspace(tmp_path: Path) -> None:
    """It edits the turn's plan in graph state and touches no file, so the
    workspace must refuse it by name — never as an unsupported tool, which
    would mean it had fallen off the roster."""
    projects_root = tmp_path / "Projects"
    (projects_root / "demo").mkdir(parents=True)
    (projects_root / "demo" / "main.py").write_text("VALUE = 1\n", encoding="utf-8")
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

    with pytest.raises(ProjectWorkspaceError) as refused:
        await service.execute_staged(
            discovered[0].id,
            ProjectToolCallV1(
                name="revise_plan", arguments={"files": [], "reason": "wrong stack"}
            ),
            {},
        )
    assert "unsupported project tool" not in str(refused.value)


def test_sandbox_verdict_is_honest_when_a_masked_import_left_the_app_unrun() -> None:
    """A build whose entrypoint could not import — because a declared package is
    absent from the verify image, so the failure was downgraded to a warning —
    must not be described as having "imported and served its routes". This is the
    exact qwen `PyPDF2` case measured live: app.main never imported at all."""
    checks = [
        {"name": "import app.main", "kind": "import", "ok": False, "missing_module": "PyPDF2"},
        {"name": "import app.config", "kind": "import", "ok": True},
        {"name": "application object", "kind": "application", "ok": True,
         "detail": "no ASGI application found; import checks only"},
    ]
    verdict = _sandbox_verdict(checks)
    assert verdict.startswith("⚠")
    assert "could not import" in verdict
    assert "app.main" in verdict
    assert "served its routes" not in verdict


def test_sandbox_verdict_confirms_only_when_routes_were_actually_served() -> None:
    """The strong green verdict is reserved for a run that actually exercised the
    project's own routes — the grok case, whose app booted and served `/`."""
    served = [
        {"name": "import app.main", "kind": "import", "ok": True},
        {"name": "application object", "kind": "application", "ok": True,
         "detail": "FastAPI declaring 3 route(s), 1 written by this project"},
        {"name": "GET /", "kind": "request", "ok": True, "detail": "HTTP 200"},
    ]
    verdict = _sandbox_verdict(served)
    assert verdict.startswith("✅")
    assert "served its routes" in verdict

    # An app that loaded but exposed no routes of its own, and a changeset with no
    # app object at all, are both honest ✅s that do NOT claim routes were served.
    loaded_no_routes = [
        {"name": "import app.main", "kind": "import", "ok": True},
        {"name": "application object", "kind": "application", "ok": True,
         "detail": "FastAPI declaring 2 route(s), 0 written by this project"},
    ]
    assert _sandbox_verdict(loaded_no_routes).startswith("✅")
    assert "served its routes" not in _sandbox_verdict(loaded_no_routes)

    no_app = [{"name": "import app.util", "kind": "import", "ok": True}]
    assert _sandbox_verdict(no_app).startswith("✅")
    assert "no runnable application" in _sandbox_verdict(no_app)


def test_annotate_summary_leads_with_the_masked_import_warning() -> None:
    """End to end: the card for the masked-import case leads with the ⚠ verdict,
    not a green check, even though there are zero blocking errors."""
    verification = {
        "errors": [],
        "warnings": [{"path": "app/main.py", "error": "PyPDF2 is declared but not installed in the offline verify image"}],
        "notes": [],
        "checks": [
            {"name": "import app.main", "kind": "import", "ok": False, "missing_module": "PyPDF2"},
            {"name": "application object", "kind": "application", "ok": True,
             "detail": "no ASGI application found; import checks only"},
        ],
    }
    card = _annotate_summary("staged files", verification)
    assert card.startswith("⚠ Could not confirm this project runs")
    assert "PyPDF2 is declared but not installed" in card


def test_annotate_summary_does_not_call_env_explained_failures_defects() -> None:
    """A config that fail-fasts on a missing env var is unimportable in the
    sandbox but not broken. The card must not headline it as N problems that
    "would stop this working", and must collapse one cause seen from many
    importers into a single line. Measured on a real qwen build that showed
    "7 problem(s)" for one correct `raise`."""
    def failed_import(module: str) -> dict[str, str]:
        return {
            "path": "app/config.py",
            "rung": "runtime",
            "error": (
                f"import {module} failed: ValueError: OCI_COMPARTMENT_ID "
                "environment variable is required. (line 8)"
            ),
        }

    verification = {
        "errors": [
            failed_import("app.main"),
            failed_import("app.config"),
            failed_import("app.orchestrator"),
            failed_import("app.agents.planner"),
            failed_import("app.agents.extractor"),
        ],
        "warnings": [],
        "notes": [],
        "checks": [
            {"name": "import app.main", "kind": "import", "ok": False},
            {"name": "application object", "kind": "application", "ok": True,
             "detail": "no ASGI application found; import checks only"},
        ],
    }
    card = _annotate_summary("staged files", verification)
    assert "would stop this project working" not in card
    assert "limit of the check" in card
    assert "5 modules could not import" in card  # one collapsed line, not five
    # Correct fail-fast code stays approvable — a ValueError is not provably broken.
    assert _blocking_reason(verification) is None


def test_annotate_summary_still_flags_a_genuinely_blocking_error() -> None:
    verification = {
        "errors": [{
            "path": "app/main.py",
            "rung": "wiring",
            "error": "mounts StaticFiles at directory 'app/static', but no file in the project creates it",
        }],
        "warnings": [],
        "notes": [],
        "checks": [],
    }
    card = _annotate_summary("staged files", verification)
    assert card.startswith("⚠️")
    assert "would stop this project working" in card
    assert _blocking_reason(verification) is not None


def test_an_acceptance_failure_is_not_excused_as_an_environment_limit() -> None:
    """Measured on a live Command A+ build. A scenario that ran and answered
    HTTP 422 was filed under "could not be exercised in the sandbox … a limit of
    the check, not a proven defect", and rendered as "2 modules could not import
    — GET /convert returned HTTP 422". It was exercised; it answered; it
    answered wrongly. That is the one rung that can say the app serves routes
    without doing what was asked, and excusing it defeats the rung."""
    verification = {
        "errors": [
            {
                "path": "app/main.py",
                "rung": "runtime",
                "kind": "acceptance",
                "error": (
                    "acceptance: Celsius to Fahrenheit conversion failed: "
                    "GET /convert returned HTTP 422, expected 2xx"
                ),
            },
            {
                "path": "app/config.py",
                "rung": "runtime",
                "kind": "import",
                "error": (
                    "import app.config failed: ValueError: OCI_COMPARTMENT_ID "
                    "environment variable is required. (line 8)"
                ),
            },
        ],
        "warnings": [],
        "notes": [],
        "checks": [{"name": "import app.main", "kind": "import", "ok": True}],
    }
    card = _annotate_summary("staged files", verification)
    assert "ran and answered, but not the way the plan said" in card
    assert "GET /convert returned HTTP 422" in card
    # The import failure keeps its own honest heading, and neither borrows the
    # other's wording.
    assert "Could not be exercised in the sandbox" in card
    assert "modules could not import — GET /convert" not in card
    # Neither is dressed up as a blocking defect: both are still advisory.
    assert "would stop this project working" not in card


@pytest.mark.asyncio
async def test_a_drifting_turn_is_narrowed_to_one_file_before_it_is_ended() -> None:
    """Auto-chunking, triggered by behaviour rather than by a file count.

    The live evidence: the same four-file conversion GLM-5.2 read 22 times and
    never wrote, split into one-file requests, produced every file with clean
    verification. So a turn that stops converging is narrowed to a single owed
    file instead of being ended — and only ended if it will not write even that.

    Deliberately NOT keyed on how many files the plan names: a threshold like
    "chunk at 3+" would slow a model that takes them in one pass
    (deepseek-v4-pro wrote six) and still miss one that stalls on two.
    """
    from waqil_api.control_plane import (
        ControlPlane, _FOCUSED_EXPLORE_STEPS, _explore_budget,
    )

    plane = object.__new__(ControlPlane)
    plane.projects = SimpleNamespace(context=_empty_context)
    plane.events = SimpleNamespace(emit=_noop_emit)
    plane.settings = SimpleNamespace(project_agent_max_steps=48, project_verify_bonus_steps=0)
    plane._guard = _noop_emit
    plane._stage = _noop_emit

    drifting = {
        "prompt": "Convert this to FastAPI.",
        "run_id": "r", "conversation_id": "c",
        "model_aliases": {"_project_id": "asset_x"},
        "project_iterations": 20,
        "project_planned_files": ["app/main.py", "app/static/index.html", "app/static/app.js"],
        "project_staged": {},
    }
    drifting["project_consecutive_reads"] = _explore_budget(drifting)

    narrowed = await ControlPlane._project_step(plane, drifting)
    # Narrowed to the first owed file, not ended.
    assert narrowed["project_focus_path"] == "app/main.py"
    assert narrowed["project_focus_rounds"] == 1
    assert narrowed["project_consecutive_reads"] == 0
    assert "response_text" not in narrowed

    # The step request now offers ONLY that file — which is what narrows
    # create_file's enum on the tool-calling lanes — and says so in words.
    plane.settings = SimpleNamespace(
        project_agent_max_steps=48, project_reference_enabled=False,
        project_reference_dir=Path("/none"), project_reference_max_chars=0,
        project_reference_max_chars_local=0,
    )
    request = ControlPlane._project_step_request(
        plane, {**drifting, **narrowed}, {}, [], {}, 21,
        planned=drifting["project_planned_files"],
    )
    assert request["files_still_to_write"] == ["app/main.py"]
    assert "Write ONLY this one file" in request["attention"]
    assert "app/main.py" in request["attention"]

    # Still refusing to write the one named file — now the turn ends.
    stuck = {**drifting, **narrowed, "project_consecutive_reads": _FOCUSED_EXPLORE_STEPS}
    plane.settings = SimpleNamespace(project_agent_max_steps=48, project_verify_bonus_steps=0)
    ended = await ControlPlane._project_step(plane, stuck)
    assert "without writing" in ended["response_text"]


@pytest.mark.asyncio
async def test_a_model_that_keeps_writing_is_never_narrowed() -> None:
    """The ladder must be invisible to a model that is doing the work.

    deepseek-v4-pro wrote six files in one turn; nothing here should have
    slowed it down. Every write resets the read run, so the narrowing branch
    is never reached."""
    from waqil_api.contracts import ProjectAgentStepV1, ProjectToolCallV1
    from waqil_api.control_plane import ControlPlane, _explore_budget

    class Writer:
        async def project_plan_files(self, request, *, model_aliases=None):
            return SimpleNamespace(files=[], scenarios=[], intent="build", scope="narrow")

        async def project_step(self, request, *, model_aliases=None):
            # The whole point: a productive model is never handed the narrowed
            # instruction, because it never stopped converging.
            assert "Write ONLY this one file" not in str(request.get("attention", ""))
            return ProjectAgentStepV1(
                status="tool",
                tool_call=ProjectToolCallV1(
                    name="create_file", arguments={"path": "c.py", "content": "C = 1\n"}
                ),
            )

    plane = object.__new__(ControlPlane)
    plane.model = Writer()
    plane.projects = SimpleNamespace(context=_empty_context)
    plane.events = SimpleNamespace(emit=_noop_emit)
    plane.settings = SimpleNamespace(
        project_agent_max_steps=48, project_verify_bonus_steps=0,
        project_staged_max_files=48, project_spec_rewrite=False,
        project_spec_rewrite_max_chars=1800, project_reference_enabled=False,
        project_reference_dir=Path("/none"), project_reference_max_chars=0,
        project_reference_max_chars_local=0,
    )
    plane._guard = _noop_emit
    plane._stage = _noop_emit

    productive = {
        "prompt": "Convert this to FastAPI.",
        "run_id": "r", "conversation_id": "c",
        "model_aliases": {"_project_id": "asset_x"},
        "project_iterations": 20,
        "project_planned_files": ["a.py", "b.py", "c.py", "d.py"],
        "project_staged": {"a.py": {"bytes": 1}, "b.py": {"bytes": 1}},
        # A write just landed, so the read run is zero.
        "project_consecutive_reads": 0,
    }
    assert _explore_budget(productive) > 0
    result = await ControlPlane._project_step(plane, productive)
    # Not narrowed, not ended — it simply proceeds to the next write.
    assert "project_focus_path" not in result
    assert "response_text" not in result
    assert result["project_pending_call"]["name"] == "create_file"


def test_reads_are_weighted_by_whether_they_learned_anything() -> None:
    """Read COUNT is the wrong primitive; productivity is the right one.

    Measured on two real 48-step turns that look identical from the outside:
    the productive one (GLM-5.2, three files written) executed every call it
    made, while the pathological one (kimi, zero files) had 36 of its 48 steps
    refused as repeats — the model asking for what was already in its trace.

    So a big model reading widely through a big codebase is left alone, and a
    doom loop trips in a handful of steps. This is the same conclusion the
    published harness work reaches: iteration counts alone are too coarse, and
    repeated identical calls are the signal worth acting on.
    """
    from waqil_api.control_plane import (
        ControlPlane, _UNPRODUCTIVE_READ_WEIGHT, _explore_budget,
    )

    call = ProjectToolCallV1(name="read_file", arguments={"path": "a.py"})
    fresh = ControlPlane._project_evidence(
        {"project_consecutive_reads": 4}, call, {"ok": True}, 0
    )
    assert fresh["project_consecutive_reads"] == 5  # a fresh read costs 1

    looped = ControlPlane._project_evidence(
        {"project_consecutive_reads": 4}, call, {"ok": False}, 0, unproductive=True
    )
    assert looped["project_consecutive_reads"] == 4 + _UNPRODUCTIVE_READ_WEIGHT

    # A write still clears the run entirely.
    wrote = ControlPlane._project_evidence(
        {"project_consecutive_reads": 20},
        ProjectToolCallV1(name="create_file", arguments={"path": "a.py", "content": "x"}),
        {"ok": True}, 0,
    )
    assert wrote["project_consecutive_reads"] == 0

    # The real numbers: GLM's longest fresh run survives, kimi's loop does not.
    budget = _explore_budget({"project_planned_files": ["a", "b", "c", "d"]})
    assert 18 < budget, "GLM's 18 consecutive fresh reads must not be cut off"
    assert 6 * _UNPRODUCTIVE_READ_WEIGHT > budget, "kimi's loop must trip quickly"


@pytest.mark.asyncio
async def test_a_dead_coder_lane_falls_down_the_ladder_mid_turn() -> None:
    """The per-role ladder, walked live: a lane that fails to ANSWER advances
    to the next rung and the same step retries there — visibly, and without
    ending the turn, which is what a single dead lane used to do. A malformed
    REPLY must never advance the ladder: that is the model's own mistake, and
    the evidence loop is how it corrects itself."""
    import json as _json

    from waqil_api.contracts import ProjectAgentStepV1, ProjectToolCallV1
    from waqil_api.control_plane import ControlPlane
    from waqil_api.model_provider import ModelProviderError

    calls: list[str] = []
    events: list[tuple[str, dict[str, Any]]] = []

    async def _emit(run_id, conversation_id, kind, payload):
        events.append((kind, payload))

    class LadderModel:
        async def project_plan_files(self, request, *, model_aliases=None):
            raise RuntimeError("plan unavailable")

        async def project_step(self, request, *, model_aliases=None):
            calls.append(f"{model_aliases['_provider']}:{model_aliases.get('coder')}")
            if model_aliases["_provider"] == "local":
                raise ModelProviderError("Ollama returned HTTP 429: too many requests")
            return ProjectAgentStepV1(
                status="tool",
                tool_call=ProjectToolCallV1(name="list_files", arguments={}),
            )

    plane = object.__new__(ControlPlane)
    plane.model = LadderModel()
    plane.events = SimpleNamespace(emit=_emit)
    plane.projects = SimpleNamespace(context=_empty_context)
    plane.settings = SimpleNamespace(
        project_agent_max_steps=48, project_staged_max_files=48,
        project_spec_rewrite=False, project_spec_rewrite_max_chars=1800,
        project_reference_enabled=False, project_reference_dir=Path("/none"),
        project_reference_max_chars=0, project_reference_max_chars_local=0,
    )
    plane._guard = _noop_emit
    plane._stage = _noop_emit

    chain = _json.dumps([
        {"provider": "local", "model": "glm-5.2:cloud"},
        {"provider": "cohere", "model": None},
    ])
    result = await ControlPlane._project_step(plane, {
        "prompt": "Add a /health route.",
        "run_id": "r", "conversation_id": "c",
        "model_aliases": {
            "_project_id": "asset_x", "_provider": "local",
            "coder": "glm-5.2:cloud", "_chain_coder": chain,
        },
        "project_iterations": 0,
    })

    # Both rungs were tried in ONE step, and the step succeeded on the second.
    assert calls == ["local:glm-5.2:cloud", "cohere:glm-5.2:cloud"]
    assert result["project_pending_call"]["name"] == "list_files"
    # The turn remembers the working rung, so later steps skip the dead one.
    assert result["project_chain_index"] == 1
    fallback = [p for k, p in events if k == "run.model_fallback"]
    assert len(fallback) == 1
    assert fallback[0]["from"] == "glm-5.2:cloud"
    assert fallback[0]["to"] == "Cohere Command A+"
    assert fallback[0]["reason"] == "rate_limited"

    # A malformed reply does NOT advance the ladder.
    calls.clear()
    events.clear()

    class MalformedModel(LadderModel):
        async def project_step(self, request, *, model_aliases=None):
            calls.append(model_aliases["_provider"])
            raise ModelProviderError("hosted model returned invalid project tool arguments")

    plane.model = MalformedModel()
    result = await ControlPlane._project_step(plane, {
        "prompt": "Add a /health route.",
        "run_id": "r", "conversation_id": "c",
        "model_aliases": {
            "_project_id": "asset_x", "_provider": "local",
            "coder": "glm-5.2:cloud", "_chain_coder": chain,
        },
        "project_iterations": 0,
    })
    assert calls == ["local"]  # one attempt, no ladder walk
    assert result["project_malformed_streak"] == 1
    assert not [k for k, _ in events if k == "run.model_fallback"]


@pytest.mark.asyncio
async def test_an_exhausted_ladder_ends_the_turn_naming_the_cause() -> None:
    """When every rung is dead the turn ends exactly as a single dead lane
    used to — honestly, with the backend named — rather than looping."""
    import json as _json

    from waqil_api.contracts import ProjectAgentStepV1  # noqa: F401
    from waqil_api.control_plane import ControlPlane
    from waqil_api.model_provider import ModelProviderError

    class DeadEverywhere:
        async def project_plan_files(self, request, *, model_aliases=None):
            raise RuntimeError("plan unavailable")

        async def project_step(self, request, *, model_aliases=None):
            raise ModelProviderError("returned HTTP 429: rate limit")

    plane = object.__new__(ControlPlane)
    plane.model = DeadEverywhere()
    plane.events = SimpleNamespace(emit=_noop_emit)
    plane.projects = SimpleNamespace(context=_empty_context)
    plane.settings = SimpleNamespace(
        project_agent_max_steps=48, project_staged_max_files=48,
        project_spec_rewrite=False, project_spec_rewrite_max_chars=1800,
        project_reference_enabled=False, project_reference_dir=Path("/none"),
        project_reference_max_chars=0, project_reference_max_chars_local=0,
    )
    plane._guard = _noop_emit
    plane._stage = _noop_emit

    result = await ControlPlane._project_step(plane, {
        "prompt": "Add a /health route.",
        "run_id": "r", "conversation_id": "c",
        "model_aliases": {
            "_project_id": "asset_x", "_provider": "local", "coder": "m1",
            "_fallbacks_coder": _json.dumps([{"provider": "local", "model": "m2"}]),
        },
        "project_iterations": 0,
    })
    assert "rate-limited or out of quota" in result["response_text"]
    assert result.get("project_malformed_streak", 0) == 0


@pytest.mark.asyncio
async def test_a_narrowed_turn_has_its_reads_closed_structurally() -> None:
    """The permission half of explore→act: a turn narrowed to one file gets a
    small confirming allowance, then reads are REFUSED by the executor — the
    refusal is the evidence — rather than merely advised against."""
    from waqil_api.control_plane import ControlPlane, _FOCUSED_READ_ALLOWANCE

    executed: list[str] = []

    class Workspace:
        async def execute_staged(self, project_id, call, staged, next_paths=()):
            executed.append(call.name)
            return {"ok": True, "lines": []}, None

    plane = object.__new__(ControlPlane)
    plane.projects = Workspace()
    plane.events = SimpleNamespace(emit=_noop_emit)
    plane.settings = SimpleNamespace(project_verify_max_runs=2)
    plane._guard = _noop_emit
    plane._stage = _noop_emit
    base = {
        "run_id": "r", "conversation_id": "c",
        "model_aliases": {"_project_id": "asset_x"},
        "project_trace": [], "project_focus_path": "app/main.py",
        "project_pending_call": {"name": "read_file", "arguments": {"path": "x.py"}},
    }

    # Within the allowance the read executes.
    fine = await ControlPlane._project_execute(
        plane, {**base, "project_consecutive_reads": _FOCUSED_READ_ALLOWANCE - 1}
    )
    assert executed == ["read_file"]
    assert fine["project_trace"][-1]["result"]["ok"] is True

    # Past it the executor refuses without executing, naming the owed file.
    refused = await ControlPlane._project_execute(
        plane, {**base, "project_consecutive_reads": _FOCUSED_READ_ALLOWANCE}
    )
    assert executed == ["read_file"]  # nothing new ran
    error = refused["project_trace"][-1]["result"]["error"]
    assert "narrowed to writing app/main.py" in error
    # A write is never gated.
    write = await ControlPlane._project_execute(plane, {
        **base,
        "project_consecutive_reads": 20,
        "project_pending_call": {"name": "create_file",
                                  "arguments": {"path": "app/main.py", "content": "x"}},
    })
    assert executed == ["read_file", "create_file"]
    assert write["project_trace"][-1]["result"]["ok"] is True


@pytest.mark.asyncio
async def test_the_explore_act_arc_is_emitted_as_phase_events() -> None:
    """The arc the research pattern names, made visible: exploring on the
    first step of a build turn, building when a plan with files lands."""
    from waqil_api.contracts import ProjectAgentStepV1, ProjectToolCallV1
    from waqil_api.control_plane import ControlPlane, _PLAN_AFTER_STEPS

    events: list[tuple[str, dict[str, Any]]] = []

    async def _emit(run_id, conversation_id, kind, payload):
        events.append((kind, payload))

    class Model:
        async def project_plan_files(self, request, *, model_aliases=None):
            return SimpleNamespace(
                files=["app/main.py"], scenarios=[], intent="build", scope="narrow"
            )

        async def project_step(self, request, *, model_aliases=None):
            return ProjectAgentStepV1(
                status="tool", tool_call=ProjectToolCallV1(name="list_files", arguments={}),
            )

    plane = object.__new__(ControlPlane)
    plane.model = Model()
    plane.events = SimpleNamespace(emit=_emit)
    plane.projects = SimpleNamespace(context=_empty_context, stage_scaffold=None)
    plane.settings = SimpleNamespace(
        project_agent_max_steps=48, project_staged_max_files=48,
        project_spec_rewrite=False, project_spec_rewrite_max_chars=1800,
        project_reference_enabled=False, project_reference_dir=Path("/none"),
        project_reference_max_chars=0, project_reference_max_chars_local=0,
    )
    plane._guard = _noop_emit
    plane._stage = _noop_emit
    base = {
        "prompt": "Add a /health route to app/main.py.",
        "run_id": "r", "conversation_id": "c",
        "model_aliases": {"_project_id": "asset_x"},
    }

    opened = await ControlPlane._project_step(plane, {**base, "project_iterations": 0})
    assert opened["project_phase"] == "exploring"
    assert [p["phase"] for k, p in events if k == "project.phase"] == ["exploring"]

    events.clear()
    planned = await ControlPlane._project_step(plane, {
        **base,
        "project_iterations": _PLAN_AFTER_STEPS,
        "project_phase": "exploring",
    })
    assert planned["project_phase"] == "building"
    assert [p["phase"] for k, p in events if k == "project.phase"] == ["building"]
