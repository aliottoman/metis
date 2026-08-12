"""Hermetic checks for coding-engine attribution in the capability evaluator."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from waqil_api.project_capability_eval import TimelineEvent, summarize_run


def test_preview_keeps_clinecore_as_the_rollout_default() -> None:
    repo = Path(__file__).resolve().parents[3]
    preview = subprocess.run(
        [sys.executable, str(repo / "scripts" / "project_capability_eval.py")],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )

    assert json.loads(preview.stdout)["coding_engine"] == "clinecore"


def test_preview_still_accepts_legacy_as_an_explicit_historical_baseline() -> None:
    """legacy is retired as a default/production route, never as an option:

    it must remain explicitly selectable for a historical/test-baseline
    comparison run.
    """
    repo = Path(__file__).resolve().parents[3]
    preview = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "project_capability_eval.py"),
            "--coding-engine",
            "legacy",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )

    assert json.loads(preview.stdout)["coding_engine"] == "legacy"


def test_preview_selects_clinecore_without_calling_a_model(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[3]
    empty_env = tmp_path / "empty.env"
    empty_env.write_text("", encoding="utf-8")
    preview = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "project_capability_eval.py"),
            "--env-file",
            str(empty_env),
            "--coding-engine",
            "clinecore",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )

    report = json.loads(preview.stdout)
    assert report["live"] is False
    assert report["coding_engine"] == "clinecore"


def test_preview_rejects_a_provider_the_clinecore_adapter_cannot_run() -> None:
    repo = Path(__file__).resolve().parents[3]
    preview = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "project_capability_eval.py"),
            "--coding-engine",
            "clinecore",
            "--provider",
            "oci",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )

    assert preview.returncode == 2
    assert "supports only ClinePass and Ollama" in preview.stderr


def test_summary_attributes_cline_rounds_models_and_usage() -> None:
    summary = summarize_run(
        run={"id": "run_1", "status": "awaiting_approval"},
        events=[
            TimelineEvent(
                type="project.coding_started",
                payload={
                    "engine": "clinecore",
                    "session_id": "coding_1",
                    "model": "cline-pass/deepseek-v4-pro",
                },
            ),
            TimelineEvent(
                type="project.coding_round",
                payload={
                    "session_id": "coding_1",
                    "sidecar_session_id": "sidecar_1",
                    "model": "cline-pass/deepseek-v4-pro",
                    "state": "idle",
                    "usage": {
                        "inputTokens": 20,
                        "outputTokens": 7,
                        "totalTokens": 27,
                        "requests": 1,
                    },
                },
            ),
            TimelineEvent(
                type="project.coding_round",
                payload={
                    "session_id": "coding_1",
                    "sidecar_session_id": "sidecar_1",
                    "model": "cline-pass/deepseek-v4-pro",
                    "state": "idle",
                    "usage": {
                        "inputTokens": 25,
                        "outputTokens": 9,
                        "totalTokens": 34,
                        "requests": 2,
                    },
                },
            ),
            TimelineEvent(
                type="project.coding_round",
                payload={
                    "session_id": "coding_1",
                    "sidecar_session_id": "sidecar_2",
                    "model": "cline-pass/kimi-k3",
                    "state": "idle",
                    "usage": {
                        "inputTokens": 11,
                        "outputTokens": 4,
                        "totalTokens": 15,
                        "requests": 1,
                    },
                },
            ),
        ],
        duration_seconds=2.0,
    )

    engine = summary["coding_engine"]
    assert {
        key: engine[key]
        for key in (
            "observed",
            "sessions",
            "rounds",
            "models",
            "states",
            "terminal_failure",
            "terminal_failure_states",
            "fallback_recovered_terminal_states",
            "controlled_stop_recovered_terminal_states",
        )
    } == {
        "observed": "clinecore",
        "sessions": ["coding_1"],
        "rounds": 3,
        "models": ["cline-pass/deepseek-v4-pro", "cline-pass/kimi-k3"],
        "states": ["idle", "idle", "idle"],
        "terminal_failure": False,
        "terminal_failure_states": [],
        "fallback_recovered_terminal_states": [],
        "controlled_stop_recovered_terminal_states": [],
    }
    assert engine["usage"] == {
        "inputTokens": 36,
        "outputTokens": 13,
        "totalTokens": 49,
        "requests": 3,
    }
    assert engine["usage_evidence"] == "inferred_sidecar_ancestry_deltas"
    assert engine["requests_semantics"] == "sidecar_turns_not_provider_iterations"
    assert engine["usage_lineage"] == [
        {
            "sidecar_session_id": "sidecar_1",
            "parent_sidecar_session_id": "",
            "host_session_id": "coding_1",
            "model": "cline-pass/deepseek-v4-pro",
            "semantics": "session_root",
            "raw": {
                "inputTokens": 25,
                "outputTokens": 9,
                "totalTokens": 34,
                "requests": 2,
            },
            "counted": {
                "inputTokens": 25,
                "outputTokens": 9,
                "totalTokens": 34,
                "requests": 2,
            },
        },
        {
            "sidecar_session_id": "sidecar_2",
            "parent_sidecar_session_id": "sidecar_1",
            "host_session_id": "coding_1",
            "model": "cline-pass/kimi-k3",
            "semantics": "session_local_or_reset",
            "raw": {
                "inputTokens": 11,
                "outputTokens": 4,
                "totalTokens": 15,
                "requests": 1,
            },
            "counted": {
                "inputTokens": 11,
                "outputTokens": 4,
                "totalTokens": 15,
                "requests": 1,
            },
        },
    ]
