"""Hermetic fixtures for token-conscious existing-project capability canaries."""

from __future__ import annotations

import ast
import json
import runpy
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import waqil_api.project_capability_scenarios as scenarios_module
from waqil_api.config import Settings
from waqil_api.project_capability_eval import (
    TimelineEvent,
    release_gate_passes,
    score_evaluation,
    summarize_coding_usage,
    summarize_run,
)
from waqil_api.project_capability_scenarios import (
    FASTAPI_REPAIR_SCENARIO,
    REPAIR_SEED_FILES,
    SCENARIOS,
    UI_REVAMP_SCENARIO,
    qualify_scenario,
    run_scenario_acceptance,
)
from waqil_api.project_sandbox import SandboxOutcome


def _script() -> tuple[Path, Path]:
    repo = Path(__file__).resolve().parents[3]
    return repo, repo / "scripts/project_capability_eval.py"


def _preview(*arguments: str) -> dict[str, object]:
    repo, script = _script()
    result = subprocess.run(
        [sys.executable, str(script), *arguments],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def _clear_app_modules() -> None:
    for name in list(sys.modules):
        if name == "app" or name.startswith("app."):
            sys.modules.pop(name, None)


def test_scenario_registry_has_parseable_exact_scopes_and_realistic_seeds(
    tmp_path: Path,
) -> None:
    assert tuple(SCENARIOS) == ("meridian", "ui-revamp", "fastapi-repair")
    for scenario in (UI_REVAMP_SCENARIO, FASTAPI_REPAIR_SCENARIO):
        ast.parse(scenario.acceptance_source)
        assert len(scenario.required_files) == len(set(scenario.required_files))
        assert not set(scenario.required_files) & set(scenario.protected_files)
        project = tmp_path / scenario.slug
        scenario.initialize(project)
        seeded = {
            str(path.relative_to(project))
            for path in project.rglob("*")
            if path.is_file()
        }
        assert set(scenario.seed_files) == seeded
        assert scenario.protected_files
        assert all((project / path).is_file() for path in scenario.protected_files)
        assert all(path in scenario.request for path in scenario.required_files)


@pytest.mark.parametrize("key", ["ui-revamp", "fastapi-repair"])
def test_scenario_dry_preview_is_exact_and_never_live(key: str) -> None:
    scenario = SCENARIOS[key]
    preview = _preview("--scenario", key)

    assert preview["live"] is False
    assert preview["scenario_key"] == key
    assert preview["scenario"] == scenario.name
    assert preview["request"] == scenario.request
    assert preview["required_files"] == list(scenario.required_files)
    assert preview["protected_files"] == list(scenario.protected_files)
    assert preview["seed_files"] == sorted(scenario.seed_files)
    assert preview["qualification"]["mode"] == "binary"
    assert preview["qualification"]["pass_score"] == 100
    assert preview["qualification"]["fail_score"] == 0


def test_local_fallback_ladders_are_explicit_deduped_and_token_bounded() -> None:
    preview = _preview(
        "--scenario",
        "ui-revamp",
        "--provider",
        "local",
        "--orchestrator-model",
        "glm-5.2:cloud",
        "--coder-model",
        "deepseek-v4-pro:cloud",
        "--planner-fallback-model",
        "glm-5.2:cloud",
        "--planner-fallback-model",
        "qwen3.7:cloud",
        "--coder-fallback-model",
        "kimi-k3:cloud",
        "--coder-fallback-model",
        "kimi-k3:cloud",
        "--coder-fallback-model",
        "kimi-k2.7-code:cloud",
        "--max-tokens-per-coding-turn",
        "8192",
    )

    assert [item["model"] for item in preview["role_chains"]["planner"]] == [
        "glm-5.2:cloud",
        "qwen3.7:cloud",
    ]
    assert [item["model"] for item in preview["role_chains"]["coder"]] == [
        "deepseek-v4-pro:cloud",
        "kimi-k3:cloud",
        "kimi-k2.7-code:cloud",
    ]
    assert preview["max_tokens_per_coding_turn"] == 8192

    repo, script = _script()
    rejected = subprocess.run(
        [
            sys.executable,
            str(script),
            "--provider",
            "cline",
            "--coder-fallback-model",
            "unapproved-cloud-model",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode == 2
    assert "only for the local Ollama route" in rejected.stderr

    unbounded = subprocess.run(
        [sys.executable, str(script), "--max-tokens-per-coding-turn", "32769"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert unbounded.returncode == 2
    assert "between 4096 and 32768" in unbounded.stderr


def test_clinecore_iteration_cap_is_independent_reported_and_evaluation_local(
    tmp_path: Path,
) -> None:
    repo, script = _script()
    env_file = tmp_path / "evaluation.env"
    env_file.write_text(
        "WAQIL_CLINE_SIDECAR_MAX_ITERATIONS=13\n",
        encoding="utf-8",
    )
    preview = _preview(
        "--coding-engine",
        "clinecore",
        "--env-file",
        str(env_file),
        "--max-steps",
        "17",
        "--max-coding-iterations",
        "9",
    )
    assert preview["max_steps_per_turn"] == 17
    assert preview["max_coding_iterations"] == 9

    script_module = runpy.run_path(str(script))
    arguments = script_module["_parser"]().parse_args(
        [
            "--coding-engine",
            "clinecore",
            "--env-file",
            str(env_file),
            "--max-steps",
            "17",
            "--max-coding-iterations",
            "9",
        ]
    )
    evaluation_settings = script_module["_settings"](
        arguments,
        repo_root=repo,
        data_dir=tmp_path / "data",
        project_parent=tmp_path / "Projects",
    )
    assert evaluation_settings.project_agent_max_steps == 17
    assert evaluation_settings.cline_sidecar_max_iterations == 9
    assert Settings(_env_file=env_file).cline_sidecar_max_iterations == 13

    inherited = _preview(
        "--coding-engine",
        "clinecore",
        "--env-file",
        str(env_file),
    )
    assert inherited["max_coding_iterations"] == 13

    for invalid in ("0", "201"):
        rejected = subprocess.run(
            [
                sys.executable,
                str(script),
                "--coding-engine",
                "clinecore",
                "--max-coding-iterations",
                invalid,
            ],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        assert rejected.returncode == 2
        assert "between 1 and 200" in rejected.stderr

    # clinecore is the default engine, so omitting --coding-engine no longer
    # exercises the mismatch; the rejection requires explicitly selecting the
    # historical-baseline legacy engine instead.
    wrong_engine = subprocess.run(
        [
            sys.executable,
            str(script),
            "--coding-engine",
            "legacy",
            "--max-coding-iterations",
            "9",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert wrong_engine.returncode == 2
    assert "only with --coding-engine clinecore" in wrong_engine.stderr


@pytest.mark.asyncio
async def test_evaluator_snapshots_exact_pending_checkpoint_overlay() -> None:
    _, script = _script()
    script_module = runpy.run_path(str(script))
    raw = {
        "app/static/index.html": {
            "content": "pending bytes\n",
            "origin": "model",
            "base_sha256": "abc",
        }
    }

    class ControlPlane:
        @staticmethod
        def _config(conversation_id: str, run_id: str) -> dict[str, object]:
            assert (conversation_id, run_id) == ("conversation_1", "run_1")
            return {"configurable": {"thread_id": "conversation_1:run_1"}}

    class Checkpoint:
        checkpoint = {"channel_values": {"project_staged": raw}}

    class Checkpointer:
        async def aget_tuple(self, config: dict[str, object]) -> Checkpoint:
            assert config["configurable"] == {"thread_id": "conversation_1:run_1"}
            return Checkpoint()

    class Runtime:
        control_plane = ControlPlane()
        checkpointer = Checkpointer()

    snapshot = await script_module["_read_pending_project_overlay"](
        Runtime(), "conversation_1", "run_1"
    )
    snapshot["app/static/index.html"]["content"] = "snapshot mutation"

    assert raw["app/static/index.html"]["content"] == "pending bytes\n"
    assert snapshot["app/static/index.html"]["base_sha256"] == "abc"


def test_binary_qualification_cannot_average_away_one_failed_gate() -> None:
    scenario = UI_REVAMP_SCENARIO
    attempt = {
        "plan": {"files": list(scenario.required_files)},
        "writes": {"unique_successful_paths": list(scenario.required_files)},
        "verification": {
            "attempts": 1,
            "after_last_successful_write": True,
            "blocking": 0,
        },
        "approval": {"offered": True, "blocked": False},
    }
    acceptance = {"available": True, "passed": True}

    passed = qualify_scenario(scenario, [attempt], acceptance, approved=True)
    failed = qualify_scenario(
        scenario,
        [{**attempt, "writes": {"unique_successful_paths": []}}],
        acceptance,
        approved=True,
    )

    assert passed == {
        "mode": "binary",
        "passed": True,
        "score": 100,
        "gates": {
            "exact_requested_scope": True,
            "required_files_written": True,
            "clean_post_write_verification": True,
            "repair_continuity": True,
            "approved_disposable_changeset": True,
            "required_acceptance_probe": True,
        },
        "failed_gates": [],
    }
    assert failed["score"] == 0
    assert failed["failed_gates"] == ["required_files_written"]
    hidden_scope = qualify_scenario(
        scenario,
        [
            {
                **attempt,
                "writes": {
                    **attempt["writes"],
                    "host_scaffold_paths": ["appkit/web.py"],
                },
            }
        ],
        acceptance,
        approved=True,
    )
    assert hidden_scope["failed_gates"] == ["exact_requested_scope"]
    report = {
        "score": {"total": 100},
        "approved": True,
        "acceptance": {"available": True, "passed": True},
        "attempts": [{"writes": {"path_evidence": "complete"}}],
        "qualification": failed,
    }
    assert release_gate_passes(report, 70) is False
    report["qualification"] = passed
    assert release_gate_passes(report, 70) is True


@pytest.mark.parametrize(
    "repair_paths",
    [[], ["outside-host-plan.py"]],
    ids=["no-op", "out-of-plan"],
)
def test_binary_qualification_does_not_credit_noncausal_repair_continuations(
    repair_paths: list[str],
) -> None:
    scenario = UI_REVAMP_SCENARIO
    initial = {
        "plan": {"files": list(scenario.required_files)},
        "writes": {
            "successful": len(scenario.required_files),
            "unique_successful_paths": list(scenario.required_files),
        },
        "model_steps": 1,
        "verification": {
            "attempts": 1,
            "after_last_successful_write": True,
            "blocking": 1,
        },
        "approval": {"offered": True, "blocked": True},
    }
    repair = {
        "repair_continuation": {
            "required": True,
            "verified": True,
            "from_run": "run_1",
            "files": list(scenario.required_files),
        },
        "writes": {
            "successful": len(repair_paths),
            "unique_successful_paths": repair_paths,
        },
        "model_steps": 1,
        "verification": {
            "attempts": 1,
            "after_last_successful_write": True,
            "blocking": 0,
        },
        "approval": {"offered": True, "blocked": False},
    }

    qualification = qualify_scenario(
        scenario,
        [initial, repair],
        {"available": True, "passed": True},
        approved=True,
    )

    assert qualification["passed"] is False
    assert qualification["failed_gates"] == ["repair_continuity"]


def test_cline_round_changed_paths_are_exact_write_and_repair_evidence() -> None:
    required = UI_REVAMP_SCENARIO.required_files
    events = [
        TimelineEvent(
            type="project.build_planned",
            payload={
                "intent": "build",
                "scope": "whole_app",
                "files": list(required),
                "scenarios": ["preserve API", "responsive UI"],
            },
        ),
        TimelineEvent(
            type="project.coding_round",
            payload={
                "session_id": "coding_1",
                "sidecar_session_id": "sidecar_1",
                "model": "deepseek-v4-pro:cloud",
                "state": "idle",
                "changed_paths": list(required),
                "usage": {"inputTokens": 100, "outputTokens": 200, "requests": 1},
            },
        ),
        TimelineEvent(
            type="project.staged_verified",
            payload={"errors": 0, "warnings": 0, "ran": 3},
        ),
    ]

    summary = summarize_run(
        run={"id": "run_1", "status": "awaiting_approval"},
        events=events,
        duration_seconds=1,
        approval={"id": "approval_1", "blocked_reason": ""},
        required_files=required,
    )

    assert summary["writes"]["unique_successful_paths"] == sorted(required)
    assert summary["writes"]["planned_successful"] == len(required)
    assert summary["writes"]["path_evidence"] == "complete"
    assert summary["model_steps"] == 1
    assert summary["verification"]["after_last_successful_write"] is True


def test_direct_contract_drives_scoring_approval_and_binary_qualification() -> None:
    scenario = UI_REVAMP_SCENARIO
    required = scenario.required_files
    summary = summarize_run(
        run={"id": "run_direct", "status": "awaiting_approval"},
        events=[
            TimelineEvent(
                type="project.direct_contract",
                payload={
                    "contract_version": 1,
                    "mode": "direct_contract",
                    "writable_roots": ["."],
                    "protected_files": list(scenario.protected_files),
                    "unresolved": [],
                    "approval_required": True,
                    "check_budget": 8,
                },
            ),
            TimelineEvent(
                type="project.coding_round",
                payload={
                    "session_id": "coding_direct",
                    "sidecar_session_id": "sidecar_direct",
                    "model": "cline-pass/kimi-k3",
                    "state": "idle",
                    "changed_paths": list(required),
                },
            ),
            TimelineEvent(
                type="project.staged_verified",
                payload={"errors": 0, "warnings": 0, "ran": 7},
            ),
        ],
        duration_seconds=1,
        approval={"id": "approval_1", "blocked_reason": ""},
        required_files=required,
    )
    summary["preapproval_overlay"] = {
        "valid": True,
        "scope_mode": "direct_contract",
        "contract_observed": True,
        "exact_requested_scope": True,
    }
    acceptance = {
        "available": True,
        "passed": True,
        "checks_total": 1,
        "checks_passed": 1,
    }

    qualification = qualify_scenario(scenario, [summary], acceptance, approved=True)
    score = score_evaluation([summary], acceptance, max_steps=48)

    assert summary["plan"]["present"] is False
    assert summary["scope_contract"]["mode"] == "direct_contract"
    assert summary["scope_contract"]["required_file_coverage"] == 1.0
    assert qualification["passed"] is True
    assert qualification["gates"]["exact_requested_scope"] is True
    assert score["categories"]["planning"] == 20.0
    assert score["signals"]["write_completion"] == 1.0


def test_seeded_restart_usage_counts_inherited_tokens_once_and_child_request() -> None:
    parent_usage = {
        "inputTokens": 89_997,
        "outputTokens": 4_611,
        "totalTokens": 94_608,
        "requests": 1,
    }
    rounds = [
        {
            "session_id": "coding_1",
            "sidecar_session_id": "deepseek_parent",
            "model": "deepseek-v4-pro:cloud",
            "usage": parent_usage,
        },
        {
            "session_id": "coding_1",
            "sidecar_session_id": "kimi_child",
            # Historical events lack an explicit parent; ordered identity
            # change inside one host session is the durable restart lineage.
            "model": "kimi-k3:cloud",
            "usage": parent_usage,
        },
    ]

    usage, evidence, lineage = summarize_coding_usage(rounds)

    assert usage == {
        "inputTokens": 89_997,
        "outputTokens": 4_611,
        "totalTokens": 94_608,
        # Requests are child-local sidecar metadata, so the failed Kimi call is
        # still a real second request even though it consumed no new tokens.
        "requests": 2,
    }
    assert evidence == "inferred_sidecar_ancestry_deltas"
    assert lineage[1]["parent_sidecar_session_id"] == "deepseek_parent"
    assert lineage[1]["semantics"] == "inherited_cumulative_delta"
    assert lineage[1]["counted"] == {
        "inputTokens": 0,
        "outputTokens": 0,
        "totalTokens": 0,
        "requests": 1,
    }

    report = summarize_run(
        run={"id": "run_1", "status": "failed"},
        events=[
            TimelineEvent(type="project.coding_round", payload=item) for item in rounds
        ],
        duration_seconds=1,
    )
    assert report["coding_engine"]["usage"] == usage
    assert report["coding_engine"]["usage_evidence"] == evidence
    assert report["coding_engine"]["usage_lineage"] == lineage


def test_seeded_restart_usage_counts_genuine_child_delta_and_local_reset() -> None:
    cumulative_rounds = [
        {
            "session_id": "coding_1",
            "sidecar_session_id": "parent",
            "usage": {
                "inputTokens": 100,
                "outputTokens": 20,
                "totalTokens": 120,
                "requests": 1,
            },
        },
        {
            "session_id": "coding_1",
            "sidecar_session_id": "child",
            "parent_sidecar_session_id": "parent",
            "usage": {
                "inputTokens": 135,
                "outputTokens": 29,
                "totalTokens": 164,
                "requests": 2,
            },
        },
    ]
    reset_rounds = [
        cumulative_rounds[0],
        {
            "session_id": "coding_1",
            "sidecar_session_id": "fake_child",
            "parent_sidecar_session_id": "parent",
            "usage": {
                "inputTokens": 11,
                "outputTokens": 3,
                "totalTokens": 14,
                "requests": 1,
            },
        },
    ]

    cumulative, evidence, lineage = summarize_coding_usage(cumulative_rounds)
    reset, _, reset_lineage = summarize_coding_usage(reset_rounds)

    assert cumulative == {
        "inputTokens": 135,
        "outputTokens": 29,
        "totalTokens": 164,
        "requests": 3,
    }
    assert evidence == "sidecar_ancestry_deltas"
    assert lineage[1]["counted"]["inputTokens"] == 35
    assert lineage[1]["counted"]["outputTokens"] == 9
    assert reset == {
        "inputTokens": 111,
        "outputTokens": 23,
        "totalTokens": 134,
        "requests": 2,
    }
    assert reset_lineage[1]["semantics"] == "session_local_or_reset"


def test_ui_revamp_probe_passes_a_trusted_backend_preserving_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = UI_REVAMP_SCENARIO
    scenario.initialize(tmp_path)
    (tmp_path / "app/static/index.html").write_text(
        textwrap.dedent(
            """\
            <!doctype html><html><head><meta name="viewport" content="width=device-width">
            <title>Atlas Operations Console</title><link rel="stylesheet" href="/static/styles.css"></head>
            <body><nav aria-label="Primary">Dashboard</nav><main><h1>Atlas Operations Console</h1>
            <label>Search <input id="search"></label><section aria-label="Status">Status cards</section>
            <table><caption>Projects</caption></table></main><script src="/static/app.js"></script></body></html>
            """
        ),
        encoding="utf-8",
    )
    (tmp_path / "app/static/styles.css").write_text(
        ":root { --ink: #123; } :focus-visible { outline: 2px solid; }\n"
        "@media (max-width: 700px) { nav { display: none; } }\n"
        "@media (prefers-reduced-motion: reduce) { * { transition: none; } }\n",
        encoding="utf-8",
    )
    (tmp_path / "app/static/app.js").write_text(
        "let loading=true; fetch('/api/health'); fetch('/api/projects').catch(error => "
        "console.error(error)); function save(id){ return fetch(`/api/projects/${id}/status`, "
        "{method:'PATCH'}); }\n",
        encoding="utf-8",
    )
    (tmp_path / "tests/test_ui_contract.py").write_text(
        "def test_aria(): assert 'aria'\n"
        "def test_viewport(): assert 'viewport'\n"
        "def test_script(): assert '/api/projects' and 'app.js'\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        "Run with uvicorn. Test with pytest. API contract: /api/projects.\n",
        encoding="utf-8",
    )
    _clear_app_modules()
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        exec(compile(scenario.acceptance_source, scenario.probe_module, "exec"), {})
    finally:
        _clear_app_modules()


def test_fastapi_repair_probe_passes_a_trusted_continuity_fix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = FASTAPI_REPAIR_SCENARIO
    scenario.initialize(tmp_path)
    (tmp_path / "app/db.py").write_text(
        textwrap.dedent(
            """\
            import os
            import sqlite3
            from pathlib import Path

            def database_path() -> Path:
                return Path(os.getenv("INCIDENT_DB_PATH", "./incidents.sqlite3"))

            def connect() -> sqlite3.Connection:
                connection = sqlite3.connect(database_path())
                connection.row_factory = sqlite3.Row
                return connection

            def initialize_schema() -> None:
                with connect() as connection:
                    connection.execute(
                        "CREATE TABLE IF NOT EXISTS incidents ("
                        "id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, "
                        "severity TEXT NOT NULL, status TEXT NOT NULL)"
                    )
            """
        ),
        encoding="utf-8",
    )
    (tmp_path / "app/repository.py").write_text(
        textwrap.dedent(
            """\
            from .db import connect

            def get_incident(incident_id: int):
                with connect() as connection:
                    row = connection.execute(
                        "SELECT id, title, severity, status FROM incidents WHERE id = ?",
                        (incident_id,),
                    ).fetchone()
                return dict(row) if row else None

            def create_incident(title: str, severity: str):
                with connect() as connection:
                    cursor = connection.execute(
                        "INSERT INTO incidents (title, severity, status) VALUES (?, ?, ?)",
                        (title, severity, "open"),
                    )
                    incident_id = int(cursor.lastrowid)
                return get_incident(incident_id)

            def list_incidents():
                with connect() as connection:
                    rows = connection.execute(
                        "SELECT id, title, severity, status FROM incidents ORDER BY id"
                    ).fetchall()
                return [dict(row) for row in rows]

            def update_incident(incident_id: int, status: str):
                with connect() as connection:
                    cursor = connection.execute(
                        "UPDATE incidents SET status = ? WHERE id = ?", (status, incident_id)
                    )
                    if cursor.rowcount == 0:
                        return None
                return get_incident(incident_id)
            """
        ),
        encoding="utf-8",
    )
    (tmp_path / "app/routes.py").write_text(
        textwrap.dedent(
            """\
            from fastapi import APIRouter, HTTPException, status
            from .models import IncidentCreate, IncidentUpdate
            from .repository import create_incident, get_incident, list_incidents, update_incident

            router = APIRouter()

            @router.get("/health")
            def health(): return {"status": "ok"}

            @router.post("/incidents", status_code=status.HTTP_201_CREATED)
            def create(body: IncidentCreate): return create_incident(body.title, body.severity)

            @router.get("/incidents")
            def listing(): return list_incidents()

            @router.get("/incidents/{incident_id}")
            def get(incident_id: int):
                value = get_incident(incident_id)
                if value is None: raise HTTPException(status_code=404, detail="incident not found")
                return value

            @router.patch("/incidents/{incident_id}")
            def update(incident_id: int, body: IncidentUpdate):
                value = update_incident(incident_id, body.status)
                if value is None: raise HTTPException(status_code=404, detail="incident not found")
                return value
            """
        ),
        encoding="utf-8",
    )
    (tmp_path / "tests/test_incidents.py").write_text(
        "def test_create_201(): assert 201\n"
        "def test_missing_404(): assert 404\n"
        "def test_invalid_422(): assert 422\n"
        "def test_resolved_after_create_app(): assert 'resolved' and 'create_app'\n",
        encoding="utf-8",
    )
    assert "priority TEXT" in REPAIR_SEED_FILES["app/db.py"]
    _clear_app_modules()
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        exec(compile(scenario.acceptance_source, scenario.probe_module, "exec"), {})
    finally:
        _clear_app_modules()


@pytest.mark.asyncio
async def test_generic_sandbox_runner_requires_the_named_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = UI_REVAMP_SCENARIO
    scenario.initialize(tmp_path)
    observed: dict[str, object] = {}

    class Sandbox:
        def __init__(self, _settings: object) -> None:
            pass

        async def verify(self, **kwargs: object) -> SandboxOutcome:
            observed.update(kwargs)
            return SandboxOutcome(
                available=True,
                checks=[
                    {
                        "name": f"import {scenario.probe_module}",
                        "kind": "import",
                        "ok": True,
                    }
                ],
            )

        async def release_machine(self, *, reason: str) -> bool:
            assert reason == "capability evaluation complete"
            return True

    monkeypatch.setattr(scenarios_module, "ProjectSandboxService", Sandbox)
    result = await run_scenario_acceptance(scenario, tmp_path, object())  # type: ignore[arg-type]

    assert result["passed"] is True
    staged = observed["staged"]
    assert isinstance(staged, dict)
    assert (
        staged[f"{scenario.probe_module}.py"]["content"] == scenario.acceptance_source
    )


@pytest.mark.asyncio
async def test_pending_overlay_acceptance_materializes_exact_bytes_without_source_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = UI_REVAMP_SCENARIO
    scenario.initialize(tmp_path)
    source_before = (tmp_path / "app/static/index.html").read_bytes()
    pending_html = "<!doctype html><title>Pending Atlas console</title>\n"
    observed: dict[str, object] = {}

    class Sandbox:
        def __init__(self, _settings: object) -> None:
            pass

        async def verify(self, **kwargs: object) -> SandboxOutcome:
            observed.update(kwargs)
            return SandboxOutcome(
                available=True,
                checks=[
                    {
                        "name": f"import {scenario.probe_module}",
                        "kind": "import",
                        "ok": True,
                    }
                ],
            )

        async def release_machine(self, *, reason: str) -> bool:
            assert reason == "capability evaluation complete"
            return False

    monkeypatch.setattr(scenarios_module, "ProjectSandboxService", Sandbox)
    result = await run_scenario_acceptance(
        scenario,
        tmp_path,
        object(),  # type: ignore[arg-type]
        staged={
            "app/static/index.html": {
                "content": pending_html,
                "origin": "model",
            }
        },
    )

    verifier_overlay = observed["staged"]
    assert isinstance(verifier_overlay, dict)
    assert verifier_overlay["app/static/index.html"]["content"] == pending_html
    assert (
        verifier_overlay[f"{scenario.probe_module}.py"]["content"]
        == scenario.acceptance_source
    )
    assert (tmp_path / "app/static/index.html").read_bytes() == source_before
    assert result["passed"] is True
