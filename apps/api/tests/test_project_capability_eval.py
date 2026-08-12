"""Hermetic tests for the permanent realistic project-build scorecard."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import types
from copy import deepcopy
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse

import waqil_api.project_capability_eval as capability_eval
from waqil_api.model_preference import CLINEPASS_MODELS
from waqil_api.project_capability_eval import (
    combined_observed_tokens,
    summarize_planner_usage,
    MERIDIAN_ACCEPTANCE_SOURCE,
    REQUIRED_FILES,
    TimelineEvent,
    acceptance_repair_prompt,
    acceptance_staged_overlay,
    bounded_acceptance_findings,
    initialize_meridian_project,
    pending_overlay_digest,
    project_file_hashes,
    project_asset_id,
    read_timeline,
    release_gate_passes,
    run_meridian_acceptance,
    score_evaluation,
    summarize_repair_continuation,
    summarize_run,
    validate_pending_overlay,
)
from waqil_api.project_sandbox import SandboxOutcome


def _event(event_type: str, **payload: object) -> TimelineEvent:
    return TimelineEvent(type=event_type, payload=dict(payload))


def test_cline_eval_rejects_credit_billed_models_and_deduplicates_backups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = Path(__file__).resolve().parents[3]
    script = repo / "scripts" / "project_capability_eval.py"
    empty_env = tmp_path / "empty.env"
    empty_env.write_text("", encoding="utf-8")
    monkeypatch.delenv("WAQIL_CLINE_ORCHESTRATOR_MODEL", raising=False)
    monkeypatch.delenv("WAQIL_CLINE_CODER_MODEL", raising=False)

    default_preview = subprocess.run(
        [sys.executable, str(script), "--env-file", str(empty_env)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    default_chains = json.loads(default_preview.stdout)["role_chains"]
    assert [entry["model"] for entry in default_chains["planner"]] == [
        "cline-pass/qwen3.7-plus",
        "cline-pass/glm-5.2",
    ]

    paid = subprocess.run(
        [
            sys.executable,
            str(script),
            "--orchestrator-model",
            "anthropic/claude-opus-4.6",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert paid.returncode == 2
    assert "only verified ClinePass models" in paid.stderr

    preview = subprocess.run(
        [
            sys.executable,
            str(script),
            "--orchestrator-model",
            "cline-pass/qwen3.7-plus",
            "--coder-model",
            "cline-pass/kimi-k3",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    chains = json.loads(preview.stdout)["role_chains"]
    for chain in chains.values():
        identities = [(entry["provider"], entry["model"]) for entry in chain]
        assert len(identities) == len(set(identities))
        assert all(entry["model"] in CLINEPASS_MODELS for entry in chain)


def test_cline_eval_honours_an_environment_selected_planner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = Path(__file__).resolve().parents[3]
    script = repo / "scripts" / "project_capability_eval.py"
    empty_env = tmp_path / "empty.env"
    empty_env.write_text("", encoding="utf-8")
    monkeypatch.setenv("WAQIL_CLINE_ORCHESTRATOR_MODEL", "cline-pass/qwen3.7-max")

    preview = subprocess.run(
        [sys.executable, str(script), "--env-file", str(empty_env)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(preview.stdout)
    assert report["orchestrator_model"] == "cline-pass/qwen3.7-max"
    assert [entry["model"] for entry in report["role_chains"]["planner"]] == [
        "cline-pass/qwen3.7-max",
        "cline-pass/glm-5.2",
    ]


def test_cli_threshold_is_not_satisfied_by_a_non_live_preview() -> None:
    repo = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "project_capability_eval.py"),
            "--fail-below",
            "0",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1


def test_live_evaluation_seed_is_visible_to_project_discovery(tmp_path: Path) -> None:
    project = tmp_path / "Projects" / "meridian-evidence-desk"
    initialize_meridian_project(project)

    assert (project / "README.md").is_file()
    assert any(not path.name.startswith(".") for path in project.iterdir())
    expected_id = (
        "asset_"
        + hashlib.sha256(str(project.resolve()).encode("utf-8")).hexdigest()[:20]
    )
    assert (
        project_asset_id(
            [{"id": expected_id, "name": "Meridian Evidence Desk"}],
            project,
        )
        == expected_id
    )


def test_summary_counts_plan_work_refusals_and_blockers() -> None:
    events = [
        _event(
            "project.build_planned",
            files=list(REQUIRED_FILES),
            scenarios=["health", "reject", "workflow", "question", "audit"],
            intent="build",
            scope="whole_app",
        ),
        _event("project.agent_step", step=1, tool="read_file"),
        _event("project.agent_step", step=2, tool="create_file"),
        _event("project.tool_result", tool="read_file", ok=True, staged=False),
        _event(
            "project.tool_result",
            tool="create_file",
            path="app/main.py",
            ok=True,
            staged=True,
        ),
        _event(
            "project.tool_result",
            tool="apply_patch",
            path="app/main.py",
            ok=False,
            staged=False,
        ),
        _event("project.direction", path="app/main.py"),
        _event("project.direction_failed", error="planner unavailable"),
        _event(
            "run.model_exhausted",
            role="planner",
            operation="project_direction",
            reason="backend_timeout",
        ),
        _event(
            "project.staged_verified",
            errors=3,
            warnings=2,
            ran=6,
            notes=[],
            findings=[
                {
                    "path": "app/repository.py",
                    "error": "return type is wrong",
                    "severity": "error",
                }
            ],
        ),
    ]
    summary = summarize_run(
        run={"id": "run_1", "status": "awaiting_approval", "last_error": None},
        events=events,
        duration_seconds=12.3456,
        approval={"blocked_reason": "three defects"},
    )

    assert summary["plan"]["required_file_coverage"] == 1.0
    assert summary["plan"]["missing_required_files"] == []
    assert summary["model_steps"] == 2
    assert summary["writes"] == {
        "attempted": 2,
        "successful": 1,
        "refused": 1,
        "unique_successful": 1,
        "unique_successful_paths": ["app/main.py"],
        "planned_successful": 1,
        "missing_planned_paths": sorted(set(REQUIRED_FILES) - {"app/main.py"}),
        "host_scaffold_paths": [],
        "path_evidence": "complete",
    }
    assert summary["reads"] == 1
    assert summary["directions"] == 1
    assert summary["direction_failures"] == 1
    assert summary["model_exhaustions"][0]["reason"] == "backend_timeout"
    assert summary["verification"]["blocking"] == 3
    assert summary["verification"]["after_last_successful_write"] is True
    assert summary["verification"]["findings"][0]["path"] == "app/repository.py"
    assert summary["approval"]["blocked"] is True


def test_summary_fails_closed_on_terminal_cline_state_but_keeps_recorded_status() -> (
    None
):
    aborted = summarize_run(
        run={"id": "run_1", "status": "completed", "last_error": None},
        events=[
            _event(
                "project.coding_round",
                session_id="coding_1",
                sidecar_session_id="sidecar_1",
                state="aborted",
                changed_paths=[],
            )
        ],
        duration_seconds=1.0,
    )
    completed = summarize_run(
        run={"id": "run_2", "status": "completed", "last_error": None},
        events=[
            _event(
                "project.coding_round",
                session_id="coding_2",
                sidecar_session_id="sidecar_2",
                state="completed",
                changed_paths=[],
            )
        ],
        duration_seconds=1.0,
    )

    assert aborted["status"] == "failed"
    assert aborted["recorded_status"] == "completed"
    assert aborted["coding_engine"]["terminal_failure"] is True
    assert aborted["coding_engine"]["terminal_failure_states"] == ["aborted"]
    assert "aborted" in aborted["last_error"]
    assert completed["status"] == "completed"
    assert completed["recorded_status"] == "completed"
    assert completed["coding_engine"]["terminal_failure"] is False
    assert completed["last_error"] == ""


def test_summary_recovers_only_a_verified_exact_controlled_iteration_stop() -> None:
    summary = summarize_run(
        run={
            "id": "run_1",
            "status": "awaiting_approval",
            "last_error": None,
        },
        events=[
            _event(
                "project.coding_round",
                session_id="coding_1",
                sidecar_session_id="deepseek_parent",
                state="failed",
                engine_state="failed",
                finish_reason="error",
                controlled_stop=True,
                controlled_stop_reason="max_iterations",
                changed_paths=["app/main.py"],
            ),
            _event("project.staged_verified", errors=1, warnings=0, ran=3),
            _event(
                "project.coding_round",
                session_id="coding_1",
                sidecar_session_id="kimi_child",
                parent_sidecar_session_id="deepseek_parent",
                state="idle",
                engine_state="completed",
                finish_reason="completed",
                changed_paths=["app/main.py"],
            ),
            _event("project.staged_verified", errors=0, warnings=0, ran=3),
        ],
        duration_seconds=1.0,
    )

    # Keep the raw SDK truth for diagnosis, while the exact host-verified
    # budget boundary no longer overrides the durable approval state.
    assert summary["coding_engine"]["states"] == ["failed", "idle"]
    assert summary["status"] == "awaiting_approval"
    assert summary["recorded_status"] == "awaiting_approval"
    assert summary["coding_engine"]["terminal_failure"] is False
    assert summary["coding_engine"]["terminal_failure_states"] == []
    assert summary["coding_engine"]["controlled_stop_recovered_terminal_states"] == [
        "failed"
    ]
    assert summary["coding_engine"]["fallback_recovered_terminal_states"] == []

    report = {
        "score": {"total": 90.0},
        "approved": True,
        "acceptance": {"available": True, "passed": True},
        "attempts": [summary],
    }
    assert release_gate_passes(report, 80.0) is True


@pytest.mark.parametrize(
    ("round_overrides", "include_verification"),
    [
        ({"controlled_stop": False}, True),
        ({"controlled_stop_reason": ""}, True),
        ({"changed_paths": []}, True),
        ({"state": "aborted", "engine_state": "aborted"}, True),
        ({}, False),
    ],
)
def test_summary_does_not_recover_incomplete_or_arbitrary_terminal_rounds(
    round_overrides: dict[str, object], include_verification: bool
) -> None:
    payload: dict[str, object] = {
        "session_id": "coding_1",
        "sidecar_session_id": "sidecar_1",
        "state": "failed",
        "engine_state": "failed",
        "finish_reason": "error",
        "controlled_stop": True,
        "controlled_stop_reason": "max_iterations",
        "changed_paths": ["app/main.py"],
    }
    payload.update(round_overrides)
    events = [_event("project.coding_round", **payload)]
    if include_verification:
        events.append(_event("project.staged_verified", errors=0, warnings=0, ran=3))

    summary = summarize_run(
        run={"id": "run_1", "status": "completed", "last_error": None},
        events=events,
        duration_seconds=1.0,
    )

    assert summary["status"] == "failed"
    assert summary["coding_engine"]["terminal_failure"] is True
    assert summary["coding_engine"]["controlled_stop_recovered_terminal_states"] == []


def test_summary_accepts_only_an_explicitly_fallback_recovered_terminal_round() -> None:
    summary = summarize_run(
        run={"id": "run_1", "status": "completed", "last_error": None},
        events=[
            _event(
                "project.coding_round",
                session_id="coding_k3",
                sidecar_session_id="sidecar_k3",
                state="failed",
                changed_paths=[],
            ),
            _event(
                "run.model_fallback",
                role="coder",
                reason="rate_limited",
                terminal_state="failed",
                session_id="coding_k3",
                to="kimi-k2.7-code:cloud",
            ),
            _event(
                "project.coding_round",
                session_id="coding_k2_7",
                sidecar_session_id="sidecar_k2_7",
                state="completed",
                changed_paths=["app/main.py"],
            ),
        ],
        duration_seconds=1.0,
    )

    assert summary["status"] == "completed"
    assert summary["coding_engine"]["terminal_failure"] is False
    assert summary["coding_engine"]["terminal_failure_states"] == []
    assert summary["coding_engine"]["fallback_recovered_terminal_states"] == ["failed"]
    assert summary["coding_engine"]["controlled_stop_recovered_terminal_states"] == []


def test_summary_uses_the_merged_revision_as_the_current_plan() -> None:
    merged = [*REQUIRED_FILES, "app/static/style.css"]
    summary = summarize_run(
        run={"id": "run_1", "status": "awaiting_approval", "last_error": None},
        events=[
            _event(
                "project.build_planned",
                files=list(REQUIRED_FILES),
                scenarios=["health", "workflow"],
                intent="build",
                scope="whole_app",
            ),
            _event(
                "project.plan_revised",
                files=merged,
                proposed_files=["app/static/index.html", "app/static/style.css"],
                retained_omissions=list(REQUIRED_FILES[:-1]),
            ),
        ],
        duration_seconds=1.0,
    )

    assert summary["plan"]["files"] == merged
    assert summary["plan"]["scenarios"] == ["health", "workflow"]
    assert summary["plan"]["revisions"] == 1
    assert summary["plan"]["required_file_coverage"] == 1.0


def test_score_counts_unique_planned_paths_not_repeat_patches() -> None:
    planned = ["one.py", "two.py", "three.py", "four.py"]
    attempt = {
        "plan": {
            "present": True,
            "intent": "build",
            "scope": "whole_app",
            "files": planned,
            "scenarios": ["one", "two", "three", "four", "five"],
            "required_file_coverage": 1.0,
        },
        "writes": {
            "attempted": 4,
            "successful": 4,
            "refused": 0,
            "planned_successful": 1,
            "path_evidence": "complete",
        },
        "model_steps": 4,
        "verification": {"attempts": 0, "blocking": 0},
        "approval": {"offered": False, "blocked": False},
    }

    score = score_evaluation([attempt], {"available": False}, max_steps=48)

    assert score["signals"]["write_completion"] == 0.25
    assert score["signals"]["write_completion_evidence"] == "unique_planned_paths"


def test_score_never_uses_raw_write_count_when_paths_are_missing() -> None:
    attempt = {
        "plan": {
            "present": True,
            "intent": "build",
            "scope": "whole_app",
            "files": ["one.py", "two.py", "three.py", "four.py"],
            "scenarios": ["one", "two", "three", "four", "five"],
            "required_file_coverage": 1.0,
        },
        "writes": {
            "attempted": 16,
            "successful": 16,
            "refused": 0,
            "planned_successful": 1,
            "path_evidence": "partial",
        },
        "model_steps": 16,
        "verification": {"attempts": 0, "blocking": 0},
        "approval": {"offered": False, "blocked": False},
    }

    partial = score_evaluation([attempt], {"available": False}, max_steps=48)
    unavailable_attempt = deepcopy(attempt)
    unavailable_attempt["writes"] = {
        **attempt["writes"],
        "planned_successful": 0,
        "path_evidence": "unavailable",
    }
    unavailable = score_evaluation(
        [unavailable_attempt], {"available": False}, max_steps=48
    )

    assert partial["signals"]["write_completion"] == 0.25
    assert partial["signals"]["write_completion_evidence"] == (
        "partial_unique_planned_paths"
    )
    assert unavailable["signals"]["write_completion"] == 0.0
    assert unavailable["signals"]["write_completion_evidence"] == "unavailable"


def test_score_measures_follow_up_convergence_and_real_acceptance() -> None:
    first = {
        "plan": {
            "present": True,
            "intent": "build",
            "scope": "whole_app",
            "files": list(REQUIRED_FILES),
            "scenarios": ["one", "two", "three", "four", "five"],
            "required_file_coverage": 1.0,
        },
        "writes": {
            "attempted": 20,
            "successful": 16,
            "refused": 4,
            "unique_successful_paths": list(REQUIRED_FILES),
            "planned_successful": 16,
            "path_evidence": "complete",
        },
        "model_steps": 36,
        "verification": {"attempts": 2, "blocking": 5},
        "approval": {"offered": True, "blocked": True},
    }
    repaired = deepcopy(first)
    repaired["verification"] = {
        "attempts": 1,
        "after_last_successful_write": True,
        "blocking": 0,
    }
    repaired["approval"] = {"offered": True, "blocked": False}
    repaired["repair_continuation"] = {
        "required": True,
        "verified": True,
        "from_run": "run_1",
        "files": ["app/main.py"],
    }
    acceptance = {
        "available": True,
        "passed": True,
        "checks_total": 12,
        "checks_passed": 12,
    }

    score = score_evaluation([first, repaired], acceptance, max_steps=48)

    assert score["categories"]["planning"] == 20.0
    assert score["categories"]["verification"] == 20.0
    assert score["categories"]["repair_convergence"] == 15.0
    assert score["categories"]["acceptance"] == 20.0
    assert score["signals"]["initial_blocking"] == 5
    assert score["signals"]["final_blocking"] == 0
    assert score["signals"]["blocking_reduction"] == 1.0
    assert score["signals"]["repair_continuation_verified"] is True
    assert score["signals"]["repair_execution_verified"] is True
    assert score["signals"]["post_repair_verification_verified"] is True
    assert score["signals"]["repair_chain_verified"] is True
    assert score["total"] >= 85


def test_noop_continuation_cannot_erase_prior_blockers() -> None:
    first = {
        "plan": {},
        "writes": {"attempted": 3, "successful": 2, "refused": 1},
        "model_steps": 4,
        "verification": {
            "attempts": 2,
            "after_last_successful_write": True,
            "blocking": 1,
        },
        "approval": {"offered": True, "blocked": True},
    }
    quota_exhausted_continuation = {
        "plan": {},
        "writes": {"attempted": 0, "successful": 0, "refused": 0},
        "model_steps": 0,
        # This is the exact empty-summary shape that previously looked clean.
        "verification": {
            "attempts": 0,
            "after_last_successful_write": False,
            "blocking": 0,
        },
        "approval": {"offered": True, "blocked": True},
        "repair_continuation": {
            "required": True,
            "verified": True,
            "from_run": "run_1",
            "files": ["app/main.py"],
        },
    }

    score = score_evaluation(
        [first, quota_exhausted_continuation],
        {"available": False},
        max_steps=48,
    )

    assert score["categories"]["repair_convergence"] == 0.0
    assert score["signals"]["initial_blocking"] == 1
    assert score["signals"]["final_blocking"] == 1
    assert score["signals"]["blocking_reduction"] == 0.0
    assert score["signals"]["repair_continuation_verified"] is True
    assert score["signals"]["repair_execution_verified"] is False
    assert score["signals"]["post_repair_verification_verified"] is False
    assert score["signals"]["repair_chain_verified"] is False
    assert score["signals"]["verified_repair_attempts"] == 0

    out_of_plan_continuation = {
        **quota_exhausted_continuation,
        "model_steps": 1,
        "writes": {
            "attempted": 1,
            "successful": 1,
            "refused": 0,
            "unique_successful_paths": ["outside-host-plan.py"],
        },
        "verification": {
            "attempts": 1,
            "after_last_successful_write": True,
            "blocking": 0,
        },
        "approval": {"offered": True, "blocked": False},
    }
    first_with_plan = {
        **first,
        "plan": {"files": ["app/main.py"]},
    }

    out_of_plan_score = score_evaluation(
        [first_with_plan, out_of_plan_continuation],
        {"available": False},
        max_steps=48,
    )

    assert out_of_plan_score["signals"]["repair_execution_verified"] is False
    assert out_of_plan_score["signals"]["repair_chain_verified"] is False
    assert out_of_plan_score["signals"]["verified_repair_attempts"] == 0
    assert out_of_plan_score["signals"]["final_blocking"] == 1


def test_repair_requires_verification_after_its_final_successful_write() -> None:
    summary = summarize_run(
        run={"id": "run_2", "status": "awaiting_approval", "last_error": None},
        events=[
            _event("project.agent_step", step=1, tool="apply_patch"),
            _event("project.staged_verified", errors=0, warnings=0, ran=4),
            _event(
                "project.tool_result",
                tool="apply_patch",
                path="app/main.py",
                ok=True,
                staged=True,
            ),
        ],
        duration_seconds=1.0,
        approval={"blocked_reason": ""},
    )

    assert summary["verification"]["attempts"] == 1
    assert summary["verification"]["blocking"] == 0
    assert summary["verification"]["after_last_successful_write"] is False
    summary["repair_continuation"] = {
        "required": True,
        "verified": True,
        "from_run": "run_1",
        "files": ["app/main.py"],
    }
    first = {
        "plan": {},
        "writes": {"attempted": 1, "successful": 1, "refused": 0},
        "model_steps": 1,
        "verification": {"attempts": 1, "blocking": 2},
        "approval": {"offered": True, "blocked": True},
    }

    score = score_evaluation([first, summary], {"available": False}, max_steps=48)

    assert score["categories"]["repair_convergence"] == 0.0
    assert score["signals"]["final_blocking"] == 2
    assert score["signals"]["post_repair_verification_verified"] is False
    assert score["signals"]["verified_repair_attempts"] == 0


def test_repair_continuation_requires_the_exact_prior_nonempty_overlay() -> None:
    carried = summarize_repair_continuation(
        [
            _event(
                "project.staged_resumed",
                from_run="run_1",
                files=["app/main.py", "app/config.py"],
            )
        ],
        expected_from_run="run_1",
    )
    wrong_run = summarize_repair_continuation(
        [_event("project.staged_resumed", from_run="run_old", files=["app.py"])],
        expected_from_run="run_1",
    )
    missing = summarize_repair_continuation([], expected_from_run="run_1")

    assert carried["verified"] is True
    assert carried["files"] == ["app/config.py", "app/main.py"]
    assert wrong_run["verified"] is False
    assert "different run" in wrong_run["reason"]
    assert missing["verified"] is False
    assert "no project.staged_resumed" in missing["reason"]


def test_an_unverified_follow_up_gets_no_repair_or_clean_rebuild_credit() -> None:
    first = {
        "plan": {},
        "writes": {"attempted": 1, "successful": 1, "refused": 0},
        "model_steps": 1,
        "verification": {"attempts": 1, "blocking": 2},
        "approval": {"offered": True, "blocked": True},
    }
    fresh_rebuild = {
        **deepcopy(first),
        "verification": {"attempts": 1, "blocking": 0},
        "approval": {"offered": True, "blocked": False},
        "repair_continuation": {
            "required": True,
            "verified": False,
            "reason": "the follow-up emitted no project.staged_resumed event",
        },
    }
    acceptance = {
        "available": True,
        "passed": True,
        "checks_total": 5,
        "checks_passed": 5,
    }

    score = score_evaluation([first, fresh_rebuild], acceptance, max_steps=48)

    assert score["categories"]["repair_convergence"] == 0.0
    assert score["categories"]["acceptance"] == 0.0
    assert score["signals"]["final_blocking"] == 2
    assert score["signals"]["repair_continuation_verified"] is False
    assert score["signals"]["verified_repair_attempts"] == 0


def test_failed_required_probe_gets_no_acceptance_credit() -> None:
    attempt = {
        "plan": {},
        "writes": {},
        "model_steps": 0,
        "verification": {},
        "approval": {"blocked": True},
    }
    acceptance = {
        "available": True,
        "passed": False,
        "checks_total": 10,
        "checks_passed": 9,
    }

    score = score_evaluation([attempt], acceptance, max_steps=48)

    assert score["signals"]["acceptance_ratio"] == 0.0
    assert score["categories"]["acceptance"] == 0.0
    assert score["signals"]["required_acceptance_passed"] is False
    assert score["verdict"] == "failed required acceptance"


def test_release_gate_requires_approval_acceptance_and_complete_path_evidence() -> None:
    report = {
        "score": {"total": 90.0},
        "approved": True,
        "acceptance": {"available": True, "passed": True},
        "attempts": [{"writes": {"path_evidence": "complete"}}],
    }

    assert release_gate_passes(report, 80.0) is True
    recovered_terminal = {
        **report,
        "attempts": [
            {
                "writes": {"path_evidence": "complete"},
                "coding_engine": {
                    "states": ["failed", "completed"],
                    "terminal_failure": False,
                },
            }
        ],
    }
    assert release_gate_passes(recovered_terminal, 80.0) is True
    for broken in (
        {**report, "approved": False},
        {**report, "acceptance": {"available": True, "passed": False}},
        {**report, "acceptance": {"available": False, "passed": False}},
        {
            **report,
            "attempts": [{"writes": {"path_evidence": "partial"}}],
        },
        {
            **report,
            "attempts": [
                {
                    "writes": {"path_evidence": "complete"},
                    "coding_engine": {"states": ["aborted"]},
                }
            ],
        },
        {**report, "score": {"total": 79.99}},
    ):
        assert release_gate_passes(broken, 80.0) is False


def test_a_backend_failure_gets_no_phantom_success_credit() -> None:
    attempt = {
        "plan": {},
        "writes": {"attempted": 0, "successful": 0, "refused": 0},
        "model_steps": 0,
        "verification": {"attempts": 0, "blocking": 0},
        "approval": {"offered": False, "blocked": False},
    }

    score = score_evaluation([attempt], {"available": False}, max_steps=48)

    assert score["total"] == 0.0
    assert score["categories"]["execution"] == 0.0
    assert score["categories"]["verification"] == 0.0
    assert score["categories"]["repair_convergence"] == 0.0


def test_timeline_reader_filters_one_run_and_decodes_payload(tmp_path: Path) -> None:
    database = tmp_path / "waqil.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE run_events (run_id TEXT, type TEXT, payload_json TEXT)"
        )
        connection.executemany(
            "INSERT INTO run_events VALUES (?, ?, ?)",
            [
                (
                    "run_1",
                    "project.build_planned",
                    json.dumps({"files": ["app/main.py"]}),
                ),
                ("run_2", "project.tool_result", json.dumps({"tool": "read_file"})),
                ("run_1", "project.staged_verified", json.dumps({"errors": 0})),
            ],
        )

    events = read_timeline(database, "run_1")

    assert [item.type for item in events] == [
        "project.build_planned",
        "project.staged_verified",
    ]
    assert events[0].payload["files"] == ["app/main.py"]


def test_pending_overlay_guard_requires_exact_scope_and_protected_integrity(
    tmp_path: Path,
) -> None:
    required = ("app/main.py", "README.md")
    protected = ("app/api.py",)
    (tmp_path / "app").mkdir()
    (tmp_path / "app/api.py").write_text("protected\n", encoding="utf-8")
    baseline = project_file_hashes(tmp_path, protected)
    staged = {
        "app/main.py": {"content": "new app\n"},
        "README.md": {"content": "new docs\n"},
    }

    clean = validate_pending_overlay(
        staged,
        required_files=required,
        protected_files=protected,
        planned_files=required,
        baseline_protected_hashes=baseline,
        current_protected_hashes=project_file_hashes(tmp_path, protected),
    )

    assert clean["valid"] is True
    assert clean["overlay_paths"] == ["README.md", "app/main.py"]
    assert clean["overlay_digest"] == pending_overlay_digest(staged)

    unsafe = validate_pending_overlay(
        {
            "app/main.py": {"content": "new app\n"},
            "app/api.py": {"content": "changed protected file\n"},
            "outside.py": {"content": "scope escape\n"},
        },
        required_files=required,
        protected_files=protected,
        planned_files=("app/main.py", "outside.py"),
        baseline_protected_hashes=baseline,
        current_protected_hashes={"app/api.py": "different"},
    )

    assert unsafe["valid"] is False
    assert unsafe["missing_required_files"] == ["README.md"]
    assert unsafe["out_of_scope_paths"] == ["app/api.py", "outside.py"]
    assert unsafe["protected_paths_touched"] == ["app/api.py"]
    assert unsafe["exact_planned_scope"] is False
    assert unsafe["source_protected_hashes_intact"] is False


def test_acceptance_overlay_and_repair_prompt_are_exact_bounded_and_non_mutating() -> (
    None
):
    staged = {"app/main.py": {"content": "original pending bytes\n", "origin": "model"}}
    merged = acceptance_staged_overlay(
        staged,
        probe_path="metis_eval_acceptance.py",
        probe_source="assert True\n",
    )
    acceptance = {
        "available": True,
        "passed": False,
        "required_probe": {
            "name": "import metis_eval_acceptance",
            "ok": False,
            "error_type": "AssertionError",
            "detail": "UI tests omit 'viewport'",
            "where": "metis_eval_acceptance.py line 24",
        },
        "blocking_findings": [
            {
                "path": "tests/test_ui_contract.py",
                "error": "acceptance: responsive coverage failed",
            }
        ],
    }

    prompt = acceptance_repair_prompt("Continue from the overlay.", acceptance)
    findings = bounded_acceptance_findings(acceptance)

    assert staged == {
        "app/main.py": {"content": "original pending bytes\n", "origin": "model"}
    }
    assert merged["app/main.py"]["content"] == "original pending bytes\n"
    assert merged["metis_eval_acceptance.py"]["origin"] == "host-evaluation"
    assert findings == [
        "import metis_eval_acceptance: AssertionError: UI tests omit 'viewport': "
        "metis_eval_acceptance.py line 24",
        "tests/test_ui_contract.py: acceptance: responsive coverage failed",
    ]
    assert all(f"- {finding}" in prompt for finding in findings)
    assert "Make a real in-scope edit" in prompt

    with pytest.raises(ValueError, match="collides"):
        acceptance_staged_overlay(
            {"metis_eval_acceptance.py": {"content": "model collision"}},
            probe_path="metis_eval_acceptance.py",
            probe_source="assert True\n",
        )


def test_rejected_settled_coding_usage_is_counted_but_never_becomes_write_credit() -> (
    None
):
    rejected = {
        "operation_id": "operation_1",
        "session_id": "coding_1",
        "sidecar_session_id": "sidecar_1",
        "model": "deepseek-v4-pro:cloud",
        "state": "completed",
        "reason": "safe mirror import refused the settled overlay",
        "changed_paths": ["must-not-count.py"],
        "usage": {
            "inputTokens": 120,
            "outputTokens": 30,
            "totalTokens": 150,
            "requests": 1,
        },
    }
    summary = summarize_run(
        run={"id": "run_1", "status": "failed", "last_error": rejected["reason"]},
        events=[
            _event("project.coding_rejected", **rejected),
            # Defensive event replay must not double-charge the settled call.
            _event("project.coding_rejected", **rejected),
        ],
        duration_seconds=1.0,
        approval=None,
        required_files=("must-not-count.py",),
    )

    assert summary["coding_engine"]["rounds"] == 0
    assert summary["coding_engine"]["rejections"] == 1
    assert summary["coding_engine"]["usage"] == {
        "inputTokens": 120,
        "outputTokens": 30,
        "totalTokens": 150,
        "requests": 1,
    }
    assert summary["coding_engine"]["states"] == ["completed"]
    assert summary["writes"]["successful"] == 0
    assert summary["writes"]["unique_successful_paths"] == []
    assert summary["model_steps"] == 0


def test_acceptance_probe_is_parseable_and_covers_the_chained_workflow() -> None:
    ast.parse(MERIDIAN_ACCEPTANCE_SOURCE)
    for contract in (
        '"/api/documents"',
        '/review",',
        '/approve")',
        '/questions",',
        "SELECT status FROM documents",
        "SELECT event_type FROM audit_events",
        'client.get("/static/app.js")',
        'root / "README.md"',
        'root / "tests" / "test_workflows.py"',
        'required_dependencies.add("pydantic")',
        "sys.modules.pop(module_name, None)",
    ):
        assert contract in MERIDIAN_ACCEPTANCE_SOURCE


def test_acceptance_probe_reports_all_readme_guidance_in_one_finding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    package = types.ModuleType("app")
    package.__path__ = []  # type: ignore[attr-defined]
    main = types.ModuleType("app.main")
    main.create_app = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "app", package)
    monkeypatch.setitem(sys.modules, "app.main", main)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "README.md").write_text(
        "Install the project and start the service.", encoding="utf-8"
    )

    with pytest.raises(AssertionError) as captured:
        exec(compile(MERIDIAN_ACCEPTANCE_SOURCE, "<meridian-acceptance>", "exec"), {})

    message = str(captured.value)
    assert "uvicorn start command" in message
    assert "OCI image/PDF setup" in message
    assert "credentialless/no-OCI TXT workflow" in message


@pytest.mark.asyncio
async def test_acceptance_report_does_not_count_generic_sandbox_checks_as_workflows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    generic = [
        {"name": f"import app.module_{index}", "kind": "import", "ok": True}
        for index in range(19)
    ]
    required = {
        "name": "import metis_eval_acceptance",
        "kind": "import",
        "ok": False,
        "detail": "the UI does not load its interaction script",
    }
    outcome = SandboxOutcome(
        available=True,
        checks=[*generic, required],
        findings=[
            {
                "path": "metis_eval_acceptance.py",
                "error": required["detail"],
                "severity": "error",
            }
        ],
    )

    class Sandbox:
        def __init__(self, _settings: object) -> None:
            pass

        async def verify(self, **_kwargs: object) -> SandboxOutcome:
            return outcome

        async def release_machine(self, *, reason: str) -> bool:
            assert reason == "capability evaluation complete"
            return False

    monkeypatch.setattr(capability_eval, "ProjectSandboxService", Sandbox)

    acceptance = await run_meridian_acceptance(tmp_path, object())  # type: ignore[arg-type]

    assert acceptance["checks_total"] == 1
    assert acceptance["checks_passed"] == 0
    assert acceptance["sandbox_checks_total"] == 20
    assert acceptance["sandbox_checks_passed"] == 19
    assert acceptance["passed"] is False
    assert acceptance["reason"] == required["detail"]


def test_acceptance_probe_passes_a_reference_implementation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Execute the probe against trusted fixture code, never model-authored code."""

    def create_app() -> FastAPI:
        application = FastAPI()
        database = Path(os.environ["MERIDIAN_DB_PATH"])

        def connect() -> sqlite3.Connection:
            connection = sqlite3.connect(database)
            connection.row_factory = sqlite3.Row
            connection.execute(
                "CREATE TABLE IF NOT EXISTS documents "
                "(id TEXT PRIMARY KEY, status TEXT, fields_json TEXT)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS audit_events "
                "(document_id TEXT, event_type TEXT)"
            )
            connection.commit()
            return connection

        @application.get("/", response_class=HTMLResponse)
        def page() -> str:
            return (
                '<meta name="viewport"><h1>Meridian Evidence Desk</h1>'
                '<p>Upload Review Audit</p><script src="/static/app.js"></script>'
            )

        @application.get("/static/app.js", response_class=PlainTextResponse)
        def script() -> str:
            return "fetch('/api/health')"

        @application.get("/api/health")
        def health() -> dict[str, str]:
            return {"status": "ok"}

        @application.post("/api/documents")
        async def upload(file: UploadFile) -> dict[str, object]:
            if file.filename and file.filename.endswith(".exe"):
                raise HTTPException(status_code=415, detail="unsupported")
            fields = {
                "invoice_number": "INV-1042",
                "vendor": "Nimbus Systems",
                "currency": "AED",
                "total": "431.75",
            }
            with connect() as connection:
                connection.execute(
                    "INSERT INTO documents VALUES (?, ?, ?)",
                    ("doc_1", "needs_review", json.dumps(fields)),
                )
                connection.execute(
                    "INSERT INTO audit_events VALUES (?, ?)", ("doc_1", "upload")
                )
                connection.commit()
            return {"document_id": "doc_1", "status": "needs_review", "fields": fields}

        @application.get("/api/documents/{document_id}")
        def document(document_id: str) -> dict[str, object]:
            with connect() as connection:
                row = connection.execute(
                    "SELECT status, fields_json FROM documents WHERE id = ?",
                    (document_id,),
                ).fetchone()
            assert row is not None
            return {
                "document_id": document_id,
                "status": row[0],
                "fields": json.loads(row[1]),
            }

        @application.post("/api/documents/{document_id}/review")
        def review(document_id: str, body: dict[str, object]) -> dict[str, str]:
            with connect() as connection:
                row = connection.execute(
                    "SELECT fields_json FROM documents WHERE id = ?", (document_id,)
                ).fetchone()
                fields = json.loads(row[0])
                fields.update(body["fields"])
                connection.execute(
                    "UPDATE documents SET status = ?, fields_json = ? WHERE id = ?",
                    ("reviewed", json.dumps(fields), document_id),
                )
                connection.execute(
                    "INSERT INTO audit_events VALUES (?, ?)", (document_id, "review")
                )
                connection.commit()
            return {"status": "reviewed"}

        @application.post("/api/documents/{document_id}/approve")
        def approve(document_id: str) -> dict[str, str]:
            with connect() as connection:
                connection.execute(
                    "UPDATE documents SET status = ? WHERE id = ?",
                    ("approved", document_id),
                )
                connection.execute(
                    "INSERT INTO audit_events VALUES (?, ?)", (document_id, "approve")
                )
                connection.commit()
            return {"status": "approved"}

        @application.post("/api/documents/{document_id}/questions")
        def question(document_id: str, body: dict[str, str]) -> dict[str, object]:
            return {
                "answer": "INV-1042 has an approved total of 431.75",
                "citations": [
                    {"source": f"document:{document_id}", "quote": "Total: 431.75"}
                ],
            }

        return application

    package = types.ModuleType("app")
    package.__path__ = []  # type: ignore[attr-defined]
    main = types.ModuleType("app.main")
    main.create_app = create_app  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "app", package)
    monkeypatch.setitem(sys.modules, "app.main", main)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "README.md").write_text(
        "Run with uvicorn. Configure OCI for image/PDF extraction. "
        "No-OCI local TXT workflow: no network or OCI credentials are needed.",
        encoding="utf-8",
    )
    (tmp_path / "requirements.txt").write_text(
        "fastapi\npython-multipart\nuvicorn\n",
        encoding="utf-8",
    )
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_workflows.py").write_text(
        "def test_health():\n    assert '/api/health'\n\n"
        "def test_upload_rejects_executables():\n"
        "    assert '/api/documents' and 415\n\n"
        "def test_questions_have_citations():\n    assert 'citations'\n",
        encoding="utf-8",
    )

    exec(compile(MERIDIAN_ACCEPTANCE_SOURCE, "<meridian-acceptance>", "exec"), {})


# ── Combined spend accounting ──────────────────────────────────────────────
# Planner calls never ran through the sidecar, so a run's "total tokens" meant
# "coder tokens" and a ceiling could be passed without a planner being counted.


def test_planner_usage_is_summarized_apart_from_coder_usage() -> None:
    summary = summarize_planner_usage(
        [
            _event(
                "run.planner_usage",
                role="planner",
                model="glm-5.2:cloud",
                fallback=False,
                corrected=False,
                latency_ms=1200,
                usage={
                    "prompt_tokens": 8000,
                    "completion_tokens": 900,
                    "total_tokens": 8900,
                },
            ),
            _event(
                "run.planner_usage",
                role="planner",
                model="glm-5.2:cloud",
                fallback=False,
                corrected=True,
                latency_ms=800,
                usage={
                    "prompt_tokens": 9000,
                    "completion_tokens": 700,
                    "total_tokens": 9700,
                },
            ),
            _event("project.coding_round", usage={"totalTokens": 999}),
        ]
    )

    assert summary["calls"] == 2
    assert summary["corrections"] == 1
    assert summary["models"] == ["glm-5.2:cloud"]
    assert summary["latency_ms"] == 2000
    assert summary["usage"]["totalTokens"] == 18_600
    assert summary["complete"] is True
    assert summary["calls_without_usage"] == 0


def test_a_planner_that_reports_no_usage_is_named_not_counted_as_free() -> None:
    summary = summarize_planner_usage(
        [
            _event("run.planner_usage", model="local-planner", usage={}),
            _event(
                "run.planner_usage",
                model="local-planner",
                usage={"total_tokens": 500},
            ),
        ]
    )

    assert summary["calls"] == 2
    assert summary["calls_without_usage"] == 1
    assert summary["complete"] is False
    assert summary["usage"]["totalTokens"] == 500

    combined = combined_observed_tokens({"totalTokens": 10_000}, summary)
    assert combined["coder_total_tokens"] == 10_000
    assert combined["planner_total_tokens"] == 500
    assert combined["total_tokens"] == 10_500
    # The honest part: a ceiling compared against this total must know the
    # figure is a floor, not the whole bill.
    assert combined["partially_unknown"] is True
    assert combined["planner_calls_without_usage"] == 1


def test_a_fully_observed_run_is_not_flagged_as_partially_unknown() -> None:
    summary = summarize_planner_usage(
        [
            _event(
                "run.planner_usage", model="glm-5.2:cloud", usage={"total_tokens": 400}
            )
        ]
    )

    combined = combined_observed_tokens({"totalTokens": 600}, summary)

    assert combined["total_tokens"] == 1_000
    assert combined["partially_unknown"] is False


def test_a_run_with_no_planner_call_reports_zero_without_claiming_completeness() -> (
    None
):
    summary = summarize_planner_usage([_event("project.coding_round", usage={})])

    assert summary["calls"] == 0
    assert summary["complete"] is False
    assert combined_observed_tokens({"totalTokens": 5}, summary)["total_tokens"] == 5


def test_planner_quality_records_that_normalization_was_required() -> None:
    """A plan the host canonicalized is not a plan the planner got right."""

    summary = summarize_planner_usage(
        [
            _event(
                "run.planner_usage", model="glm-5.2:cloud", usage={"total_tokens": 10}
            ),
            _event(
                "project.plan_normalized",
                codes=["support_slice_merged"],
                moved=[{"paths": ["tests/test_ui_contract.py", "README.md"]}],
            ),
        ]
    )

    assert summary["normalizations"] == 1
    assert summary["normalization_codes"] == ["support_slice_merged"]
    assert summary["normalized_paths"] == ["README.md", "tests/test_ui_contract.py"]
    assert summary["plan_accepted_as_written"] is False


def test_a_plan_taken_as_written_is_scored_as_such() -> None:
    summary = summarize_planner_usage(
        [_event("run.planner_usage", model="glm-5.2:cloud", usage={"total_tokens": 10})]
    )

    assert summary["normalizations"] == 0
    assert summary["plan_accepted_as_written"] is True
