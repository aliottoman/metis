"""Behavioural tests for reviewed verification checks.

These deliberately execute real child processes: the whole value of the feature
is that a check's exit code and output are genuine evidence, so a suite that
faked the subprocess would prove nothing about the part that matters.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from waqil_api.asset_library import AssetManager
from waqil_api.config import Settings
from waqil_api.contracts import ProjectToolCallV1
from waqil_api.model_provider import DeterministicModelProvider
from waqil_api.project_verification import (
    ProjectVerificationService,
    explain_command,
    read_recipe,
)
from waqil_api.project_workspace import (
    ProjectWorkspaceError,
    ProjectWorkspaceService,
    VerificationNotApprovedError,
)


def _write_recipe(project: Path, checks: list[dict]) -> None:
    metis = project / ".metis"
    metis.mkdir(parents=True, exist_ok=True)
    (metis / "verify.json").write_text(
        json.dumps({"schema_version": "1", "checks": checks}), encoding="utf-8"
    )


async def _service(tmp_path: Path, **overrides) -> tuple[ProjectWorkspaceService, str, Path]:
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
        **overrides,
    )
    assets = AssetManager(
        settings.asset_roots,
        approval_path=settings.asset_approval_path,
        catalog_path=settings.asset_catalog_path,
    )
    discovered = await assets.scan()
    verification = ProjectVerificationService(
        settings, approval_path=settings.project_verify_approval_path
    )
    service = ProjectWorkspaceService(
        settings, assets, DeterministicModelProvider(), verification=verification
    )
    return service, discovered[0].id, project


@pytest.mark.asyncio
async def test_declared_check_runs_and_reports_real_exit_code(tmp_path: Path) -> None:
    service, project_id, project = await _service(tmp_path)
    _write_recipe(
        project,
        [
            {
                "name": "test",
                "command": ["{python}", "-c", "print('all good')"],
                "description": "The demo suite.",
            },
            {
                "name": "lint",
                "command": ["{python}", "-c", "import sys; print('bad'); sys.exit(3)"],
            },
        ],
    )
    await service.approve_verification(project_id)

    passing = await service.execute(
        project_id, ProjectToolCallV1(name="run_check", arguments={"name": "test"})
    )
    assert passing["ok"] is True
    assert passing["exit_code"] == 0
    assert "all good" in passing["output"]

    # A failing check is evidence, not an error: the loop needs to read it.
    failing = await service.execute(
        project_id, ProjectToolCallV1(name="run_check", arguments={"name": "lint"})
    )
    assert failing["ok"] is False
    assert failing["exit_code"] == 3
    assert "bad" in failing["output"]


@pytest.mark.asyncio
async def test_unapproved_recipe_refuses_to_run(tmp_path: Path) -> None:
    service, project_id, project = await _service(tmp_path)
    _write_recipe(project, [{"name": "test", "command": ["{python}", "-c", "pass"]}])

    with pytest.raises(VerificationNotApprovedError):
        await service.execute(
            project_id, ProjectToolCallV1(name="run_check", arguments={"name": "test"})
        )


@pytest.mark.asyncio
async def test_editing_the_recipe_revokes_approval(tmp_path: Path) -> None:
    service, project_id, project = await _service(tmp_path)
    _write_recipe(project, [{"name": "test", "command": ["{python}", "-c", "pass"]}])
    approved = await service.approve_verification(project_id)
    assert approved.approved is True

    # Same check name, different command: the fingerprint must not survive it,
    # or a reviewed recipe would be a one-time key to any later command.
    _write_recipe(
        project, [{"name": "test", "command": ["{python}", "-c", "print('changed')"]}]
    )
    view = await service.verification_view(project_id)
    assert view.approved is False
    with pytest.raises(VerificationNotApprovedError):
        await service.execute(
            project_id, ProjectToolCallV1(name="run_check", arguments={"name": "test"})
        )


@pytest.mark.asyncio
async def test_agent_cannot_supply_or_extend_a_command(tmp_path: Path) -> None:
    service, project_id, project = await _service(tmp_path)
    marker = project / "pwned.txt"
    _write_recipe(project, [{"name": "test", "command": ["{python}", "-c", "pass"]}])
    await service.approve_verification(project_id)

    # Every shape of "run this instead" the model could try is a name lookup
    # miss, because argv never comes from the model in the first place.
    for arguments in (
        {"name": "test", "command": ["{python}", "-c", f"open({str(marker)!r},'w')"]},
        {"name": "test; touch pwned.txt"},
        {"name": "../../etc/passwd"},
        {"name": "build"},
    ):
        if arguments.get("command"):
            # An extra key is ignored outright; the declared command still runs.
            result = await service.execute(
                project_id, ProjectToolCallV1(name="run_check", arguments=arguments)
            )
            assert result["ok"] is True
        else:
            with pytest.raises(ProjectWorkspaceError, match="unknown check"):
                await service.execute(
                    project_id,
                    ProjectToolCallV1(name="run_check", arguments=arguments),
                )
    assert not marker.exists()


@pytest.mark.asyncio
async def test_a_hanging_check_is_stopped_and_reported(tmp_path: Path) -> None:
    service, project_id, project = await _service(tmp_path, project_verify_timeout_seconds=5)
    _write_recipe(
        project,
        [
            {
                "name": "hang",
                "command": ["{python}", "-c", "import time; time.sleep(120)"],
                "timeout_seconds": 5,
            }
        ],
    )
    await service.approve_verification(project_id)

    result = await service.execute(
        project_id, ProjectToolCallV1(name="run_check", arguments={"name": "hang"})
    )
    assert result["timed_out"] is True
    assert result["ok"] is False
    assert "Timed out" in result["output"]


@pytest.mark.asyncio
async def test_noisy_output_is_bounded_but_keeps_both_ends(tmp_path: Path) -> None:
    service, project_id, project = await _service(
        tmp_path, project_verify_output_chars=2_000
    )
    _write_recipe(
        project,
        [
            {
                "name": "noisy",
                "command": [
                    "{python}",
                    "-c",
                    "print('FIRST'); print('x' * 400_000); print('LAST')",
                ],
            }
        ],
    )
    await service.approve_verification(project_id)

    result = await service.execute(
        project_id, ProjectToolCallV1(name="run_check", arguments={"name": "noisy"})
    )
    assert result["truncated"] is True
    assert len(result["output"]) < 20_000
    # The invocation is at the top and the failure is at the bottom; a window
    # that kept only one end would routinely hide the reason a check failed.
    assert "FIRST" in result["output"]
    assert "LAST" in result["output"]


@pytest.mark.asyncio
async def test_a_missing_program_reports_why_instead_of_crashing(tmp_path: Path) -> None:
    service, project_id, project = await _service(tmp_path)
    _write_recipe(
        project, [{"name": "test", "command": ["metis-definitely-not-installed"]}]
    )
    await service.approve_verification(project_id)

    result = await service.execute(
        project_id, ProjectToolCallV1(name="run_check", arguments={"name": "test"})
    )
    assert result["ok"] is False
    assert "could not be started" in result["output"]


@pytest.mark.asyncio
async def test_a_broken_recipe_is_reported_not_silently_ignored(tmp_path: Path) -> None:
    service, project_id, project = await _service(tmp_path)
    metis = project / ".metis"
    metis.mkdir(parents=True, exist_ok=True)
    (metis / "verify.json").write_text("{not json", encoding="utf-8")

    view = await service.verification_view(project_id)
    assert view.configured is False
    assert view.error is not None and "valid JSON" in view.error
    # Believing verification is configured when it is not is the failure mode
    # this guards: the explanation has to say so out loud.
    assert "could not be used" in view.explanation


@pytest.mark.asyncio
async def test_shell_string_commands_are_rejected(tmp_path: Path) -> None:
    project = tmp_path / "demo"
    project.mkdir(parents=True)
    _write_recipe(project, [{"name": "test", "command": "make test && rm -rf /"}])
    settings = Settings(_env_file=None, data_dir=tmp_path / "data", repo_root=tmp_path)

    recipe = read_recipe(project, settings)
    assert recipe.checks == ()
    assert recipe.error is not None and "argv array" in recipe.error


@pytest.mark.asyncio
async def test_approval_view_explains_the_recipe_in_plain_english(tmp_path: Path) -> None:
    service, project_id, project = await _service(tmp_path)
    _write_recipe(
        project,
        [
            {"name": "test", "command": ["make", "test"]},
            {"name": "types", "command": ["npx", "tsc", "--noEmit"]},
        ],
    )

    view = await service.verification_view(project_id)
    assert view.configured is True and view.approved is False
    assert [check.name for check in view.checks] == ["test", "types"]
    # The card has to stand on its own for someone who did not write the argv.
    assert "Makefile" in view.explanation
    assert "without writing any output files" in view.explanation
    assert "your own macOS user account" in view.boundary


def test_explanations_cover_the_common_ecosystems() -> None:
    assert "Makefile" in explain_command(("make", "test"))
    assert "pnpm" in explain_command(("pnpm", "build"))
    assert "uv-managed" in explain_command(("uv", "run", "pytest"))
    assert "Rust" in explain_command(("cargo", "test"))
    assert "Go test suite" in explain_command(("go", "test", "./..."))
    # An unrecognized command must still say something true rather than nothing.
    assert "unfamiliar-tool" in explain_command(("unfamiliar-tool", "--strict"))
    assert "not pinned by this approval" in explain_command(("./scripts/check.sh",))


def test_check_run_waits_for_recipe_approval_then_runs(tmp_path: Path) -> None:
    """The inline gate: the first check in a project raises the one-time
    approval, and approving it both trusts the recipe and runs what was asked."""
    from fastapi.testclient import TestClient

    from waqil_api.main import create_app

    projects_root = tmp_path / "Projects"
    project = projects_root / "demo"
    project.mkdir(parents=True)
    (project / "README.md").write_text("# Demo\n", encoding="utf-8")
    _write_recipe(
        project,
        [
            {
                "name": "test",
                "command": ["{python}", "-c", "print('suite ok')"],
                "description": "The demo suite.",
            }
        ],
    )
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

        view = client.get(f"/api/v1/projects/{project_id}/verification").json()
        assert view["configured"] is True and view["approved"] is False
        assert "Makefile" not in view["explanation"]
        assert "your own macOS user account" in view["boundary"]

        conversation_id = client.post("/api/v1/conversations", json={}).json()["id"]
        run_id = client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json={
                "content": "[project-check-test]",
                "project_id": project_id,
                "project_mode": "grok_bootstrap_local",
            },
        ).json()["run_id"]

        run = {}
        for _ in range(200):
            run = client.get(f"/api/v1/runs/{run_id}").json()
            if run["status"] == "awaiting_approval":
                break
            time.sleep(0.01)
        assert run["status"] == "awaiting_approval"

        recoverable = client.get("/api/v1/runs?status=awaiting_approval").json()
        approval = next(
            item["approval"] for item in recoverable if item["run"]["id"] == run_id
        )
        assert approval["kind"] == "project_verify"
        # The card must explain the command, not merely display it.
        assert "written inline in the recipe" in approval["summary"]
        assert "Metis never lets the model invent a command" in approval["summary"]

        client.post(
            f"/api/v1/runs/{run_id}/decisions",
            json={"approval_id": approval["id"], "decision": "approve"},
        )
        for _ in range(400):
            run = client.get(f"/api/v1/runs/{run_id}").json()
            if run["status"] in {"completed", "failed"}:
                break
            time.sleep(0.01)
        assert run["status"] == "completed"
        assert "passed" in run["result"]["response"]

        # Approval persists, so the next turn does not ask again.
        after = client.get(f"/api/v1/projects/{project_id}/verification").json()
        assert after["approved"] is True

        # The events endpoint is an SSE stream, so assert on the replayed text.
        events = client.get(f"/api/v1/runs/{run_id}/events?after=0").text
        assert "event: project.check_result" in events
        assert '"name":"test"' in events
        assert '"ok":true' in events
        assert "event: project.verification_decided" in events


def test_revoking_approval_makes_the_next_check_ask_again(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from waqil_api.main import create_app

    projects_root = tmp_path / "Projects"
    project = projects_root / "demo"
    project.mkdir(parents=True)
    (project / "README.md").write_text("# Demo\n", encoding="utf-8")
    _write_recipe(project, [{"name": "test", "command": ["{python}", "-c", "pass"]}])
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        repo_root=tmp_path,
        asset_roots=[projects_root],
        model_backend="deterministic",
        allow_test_backends=True,
    )
    with TestClient(create_app(settings)) as client:
        project_id = client.post("/api/v1/assets/scan").json()[0]["id"]
        approved = client.post(
            f"/api/v1/projects/{project_id}/verification/approve"
        ).json()
        assert approved["approved"] is True

        revoked = client.post(
            f"/api/v1/projects/{project_id}/verification/revoke"
        ).json()
        assert revoked["approved"] is False
        assert revoked["configured"] is True
