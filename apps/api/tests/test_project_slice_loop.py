"""Intermediate verification stays on one dependency slice until it is clean."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from waqil_api.control_plane import ControlPlane


PLAN = [
    "app/__init__.py",
    "app/config.py",
    "app/db.py",
    "app/models.py",
    "app/repository.py",
    "app/extraction.py",
    "app/services.py",
]


async def _noop(*args: object, **kwargs: object) -> None:
    return None


class _Events:
    def __init__(self) -> None:
        self.items: list[tuple[str, dict[str, Any]]] = []

    async def emit(
        self,
        run_id: str,
        conversation_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        self.items.append((event_type, payload))


def _plane(
    verification: dict[str, Any],
    *,
    expected_plan: list[str] | None = None,
    expected_scenarios: list[dict[str, Any]] | None = None,
) -> tuple[ControlPlane, _Events]:
    plane = object.__new__(ControlPlane)
    events = _Events()
    plane.events = events
    plane.projects = SimpleNamespace()
    plane.settings = SimpleNamespace(project_agent_max_steps=48)
    plane._guard = _noop

    async def verify(project_id: str, staged: dict, **kwargs: Any) -> dict[str, Any]:
        assert project_id == "asset_x"
        assert kwargs["planned"] == (expected_plan or PLAN[:6])
        assert kwargs["scenarios"] == (expected_scenarios or [])
        return verification

    plane._verify_staged_changeset = verify
    return plane, events


def _state(**extra: Any) -> dict[str, Any]:
    staged = {
        path: {
            "content": f"# {path}\n",
            "origin": "create",
            "base_sha256": "",
            "bytes": len(path) + 3,
        }
        for path in PLAN[:6]
    }
    return {
        "run_id": "run_x",
        "conversation_id": "conv_x",
        "model_aliases": {"_project_id": "asset_x"},
        "project_iterations": 6,
        "project_staged": staged,
        "project_planned_files": PLAN,
        "project_planned_scenarios": [{"name": "finished app"}],
        "project_direction": {},
        "project_focus_path": "",
        "project_syntax_retries": 0,
        "project_verified_prefix": 0,
        "project_slice_verifications": 0,
        "project_repair_context": {},
        "project_trace": [],
        **extra,
    }


def _verification(*errors: dict[str, str]) -> dict[str, Any]:
    return {
        "errors": list(errors),
        "warnings": [],
        "notes": [],
        "checks": ["imports"],
    }


def _coder_chain(*models: str) -> str:
    return json.dumps(
        [{"provider": "cline", "model": f"cline-pass/{model}"} for model in models]
    )


@pytest.mark.asyncio
async def test_a_clean_dependency_slice_advances_without_calling_the_coder() -> None:
    plane, events = _plane(_verification())

    result = await ControlPlane._project_step(plane, _state())

    assert result["project_verified_prefix"] == 6
    assert result["project_slice_verifications"] == 1
    assert result["project_repair_context"] == {}
    assert result["project_syntax_retries"] == 0
    assert [kind for kind, _ in events.items] == [
        "project.staged_verified",
        "project.slice_checked",
    ]
    assert events.items[-1][1]["through"] == "app/extraction.py"


@pytest.mark.asyncio
async def test_a_failed_slice_pins_its_first_finding_before_the_next_file() -> None:
    finding = {
        "path": "app/repository.py",
        "error": "save_document returns a row where Document is required",
        "severity": "error",
        "kind": "typecheck",
    }
    plane, _ = _plane(_verification(finding))

    result = await ControlPlane._project_step(plane, _state())

    assert result["project_direction"]["path"] == "app/repository.py"
    assert result["project_write_pin"] == ["app/repository.py"]
    assert result["project_repair_context"]["files"] == PLAN[:6]
    assert result["project_repair_context"]["findings"] == [finding]
    assert "save_document" in result["project_direction"]["instruction"]
    assert "app/services.py" not in result["project_write_pin"]


@pytest.mark.asyncio
async def test_verifier_can_append_an_existing_dependency_to_the_repair_plan() -> None:
    """The manifest gate must permit the exact repair the host just ordered."""
    target = "app/legacy_db.py"
    finding = {
        "path": target,
        "error": "runtime acceptance failed because the schema was not initialized",
        "severity": "error",
        "kind": "acceptance",
    }
    plane, events = _plane(_verification(finding))
    state = _state(
        project_context={"manifest": {"file_tree": [*PLAN, target]}},
    )

    result = await ControlPlane._project_step(plane, state)

    assert result["project_planned_files"] == [*PLAN, target]
    assert result["project_direction"]["path"] == target
    assert result["project_write_pin"] == [target]
    assert result["project_repair_context"]["files"] == [*PLAN[:6], target]
    revision = [
        payload for kind, payload in events.items if kind == "project.plan_revised"
    ]
    assert revision[-1]["source"] == "verifier"
    assert revision[-1]["files"] == [*PLAN, target]


@pytest.mark.asyncio
async def test_clean_appended_repair_does_not_skip_an_unchecked_prefix_file() -> None:
    target = "app/legacy_db.py"
    checked = [*PLAN[:6], target]
    plane, _ = _plane(_verification(), expected_plan=checked)
    state = _state(
        project_planned_files=[*PLAN, target],
        project_verified_prefix=0,
        project_slice_verifications=1,
        project_repair_context={
            "files": checked,
            "findings": [
                {
                    "path": target,
                    "error": "schema was not initialized",
                    "severity": "error",
                }
            ],
        },
    )
    state["project_staged"][target] = {
        "content": "# repaired dependency\n",
        "origin": "patch",
        "base_sha256": "abc",
        "bytes": 22,
    }

    result = await ControlPlane._project_step(plane, state)

    # app/services.py is PLAN[6] and was neither staged nor checked. The
    # appended tail repair must not make its numeric prefix position look clean.
    assert result["project_verified_prefix"] == 6
    assert result["project_repair_context"] == {}


@pytest.mark.asyncio
async def test_a_successful_repair_rechecks_the_same_slice_before_advancing() -> None:
    plane, events = _plane(_verification())
    context = {
        "files": PLAN[:6],
        "findings": [
            {
                "path": "app/repository.py",
                "error": "bad return type",
                "severity": "error",
            }
        ],
    }
    state = _state(
        project_syntax_retries=1,
        project_slice_verifications=1,
        project_repair_context=context,
    )
    state["project_staged"]["app/repository.py"]["content"] = "# repaired\n"

    result = await ControlPlane._project_step(plane, state)

    assert result["project_verified_prefix"] == 6
    assert result["project_slice_verifications"] == 1
    assert result["project_repair_context"] == {}
    assert result["project_syntax_retries"] == 0
    assert events.items[-1][1]["repair"] is True


@pytest.mark.asyncio
async def test_two_successful_but_irrelevant_repairs_switch_coder_rungs() -> None:
    finding = {
        "path": "app/repository.py",
        "error": "save_document returns a row where Document is required",
        "severity": "error",
        "kind": "typecheck",
    }
    plane, events = _plane(_verification(finding))
    state = _state(
        model_aliases={
            "_project_id": "asset_x",
            "_provider": "cline",
            "_chain_coder": _coder_chain("deepseek-v4-pro", "kimi-k3"),
        },
        project_chain_index=0,
        project_syntax_retries=1,
        project_slice_verifications=1,
        project_repair_context={"files": PLAN[:6], "findings": [finding]},
    )
    # The first patch changed bytes, but not the defect the verifier names.
    state["project_staged"]["app/repository.py"]["content"] = "# cosmetic one\n"
    first = await ControlPlane._project_step(plane, state)

    assert first["project_repair_context"]["unchanged_verifications"] == 1
    assert first.get("project_chain_index", 0) == 0

    # A second accepted patch is equally irrelevant. Simulate execute_staged's
    # release of the repaired target so the same dependency slice re-verifies.
    staged = {path: dict(value) for path, value in state["project_staged"].items()}
    staged["app/repository.py"]["content"] = "# cosmetic two\n"
    second_state = {
        **state,
        **first,
        "project_staged": staged,
        "project_direction": {},
        "project_focus_path": "",
        "project_write_pin": [],
    }
    second = await ControlPlane._project_step(plane, second_state)

    assert second["project_chain_index"] == 1
    assert second["project_repair_context"]["files"] == PLAN[:6]
    assert second["project_repair_context"]["findings"] == [finding]
    assert second["project_repair_context"]["unchanged_verifications"] == 0
    assert second["project_direction"]["path"] == "app/repository.py"
    fallback = [
        payload for kind, payload in events.items if kind == "run.model_fallback"
    ]
    assert fallback[-1]["from"] == "Cline (cline-pass/deepseek-v4-pro)"
    assert fallback[-1]["to"] == "Cline (cline-pass/kimi-k3)"
    assert fallback[-1]["reason"] == "unchanged_verifier_findings"


@pytest.mark.asyncio
async def test_a_smaller_finding_set_is_progress_not_a_model_failure() -> None:
    surviving = {
        "path": "app/repository.py",
        "error": "bad return type",
        "severity": "error",
        "kind": "typecheck",
    }
    removed = {
        "path": "app/extraction.py",
        "error": "missing extract_text",
        "severity": "error",
        "kind": "wiring",
    }
    plane, events = _plane(_verification(surviving))
    result = await ControlPlane._project_step(
        plane,
        _state(
            model_aliases={
                "_project_id": "asset_x",
                "_provider": "cline",
                "_chain_coder": _coder_chain("deepseek-v4-pro", "kimi-k3"),
            },
            project_chain_index=0,
            project_syntax_retries=2,
            project_slice_verifications=1,
            project_repair_context={
                "files": PLAN[:6],
                "findings": [surviving, removed],
                "unchanged_verifications": 1,
            },
        ),
    )

    assert result["project_repair_context"]["findings"] == [surviving]
    assert result["project_repair_context"]["unchanged_verifications"] == 0
    assert result.get("project_chain_index", 0) == 0
    assert not [kind for kind, _ in events.items if kind == "run.model_fallback"]


@pytest.mark.asyncio
async def test_unchanged_findings_stop_honestly_when_no_fallback_remains() -> None:
    finding = {
        "path": "app/repository.py",
        "error": "bad return type",
        "severity": "error",
        "kind": "typecheck",
    }
    plane, events = _plane(_verification(finding))
    result = await ControlPlane._project_step(
        plane,
        _state(
            model_aliases={
                "_project_id": "asset_x",
                "_provider": "cline",
                "_chain_coder": _coder_chain("deepseek-v4-pro"),
            },
            project_chain_index=0,
            project_syntax_retries=2,
            project_slice_verifications=1,
            project_repair_context={
                "files": PLAN[:6],
                "findings": [finding],
                "unchanged_verifications": 1,
            },
        ),
    )

    assert "same 1 blocking finding" in result["response_text"]
    assert "no configured coder backup remains" in result["response_text"]
    assert result["project_repair_context"]["findings"] == [finding]
    assert result["project_repair_context"]["unchanged_verifications"] == 2
    assert result["project_direction"] == {}
    assert [kind for kind, _ in events.items][-1] == "project.repair_stalled"


@pytest.mark.asyncio
async def test_full_changeset_fallback_keeps_the_same_repair_queue() -> None:
    finding = {
        "path": "app/services.py",
        "error": "TypeError: question answer has no supporting citation",
        "severity": "error",
        "kind": "runtime",
        "rung": "sandbox",
    }
    scenarios = [{"name": "finished app"}]
    plane, _ = _plane(
        _verification(finding),
        expected_plan=PLAN,
        expected_scenarios=scenarios,
    )
    state = _state(
        model_aliases={
            "_project_id": "asset_x",
            "_provider": "cline",
            "_chain_coder": _coder_chain("deepseek-v4-pro", "kimi-k3"),
        },
        project_chain_index=0,
        project_syntax_retries=2,
        project_slice_verifications=1,
        project_repair_context={
            "files": PLAN,
            "findings": [finding],
            "unchanged_verifications": 1,
        },
    )
    state["project_staged"][PLAN[-1]] = {
        "content": "# app/services.py\n",
        "origin": "create",
        "base_sha256": "",
        "bytes": 18,
    }

    result = await ControlPlane._project_step(plane, state)

    assert result["project_chain_index"] == 1
    assert result["project_repair_context"]["files"] == PLAN
    assert result["project_repair_context"]["findings"] == [finding]
    assert result["project_direction"]["path"] == "app/services.py"


def test_the_coder_request_keeps_the_exact_causal_repair_queue() -> None:
    plane = object.__new__(ControlPlane)
    plane.settings = SimpleNamespace(
        project_agent_max_steps=48,
        project_reference_enabled=False,
        project_reference_dir=None,
        project_reference_max_chars=0,
        project_reference_max_chars_local=0,
    )
    context = {
        "files": PLAN[:6],
        "findings": [
            {
                "path": "app/repository.py",
                "error": "bad return type",
                "severity": "error",
            }
        ],
    }
    state = _state(
        prompt="Build it",
        project_plan_taken=True,
        project_plan_revisions=0,
        project_plan_revision_calls=0,
        project_stall_steps=0,
        project_repair_context=context,
        project_direction={
            "path": "app/repository.py",
            "instruction": "Fix the bad return type",
            "reuse": [],
            "read": [],
        },
    )

    request = ControlPlane._project_step_request(
        plane,
        state,
        {"manifest": {"file_tree": []}},
        [],
        state["project_staged"],
        7,
        PLAN,
    )

    assert request["verification_repair"] == context
    assert request["files_still_to_write"] == ["app/repository.py"]
    assert request["reads_closed"] is True
