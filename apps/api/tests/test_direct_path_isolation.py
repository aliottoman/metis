"""The direct path does not quietly keep the machinery it replaced.

Retiring a design is easy to claim and easy to get wrong: a flag flips, the
new path runs, and the old planner still executes underneath because one call
site was missed. Every function the direct path is supposed to have left
behind is replaced here with one that fails the test if it is called at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from waqil_api.control_plane import ControlPlane


RETIRED = (
    ("waqil_api.control_plane", "validate_effective_plan", "topology validator"),
    ("waqil_api.control_plane", "normalize_build_plan", "plan normalizer"),
    ("waqil_api.control_plane", "synthesize_single_slice", "slice synthesizer"),
    ("waqil_api.control_plane", "next_vertical_slice", "slice frontier selector"),
    ("waqil_api.control_plane", "initial_coding_prompt", "legacy prompt builder"),
    ("waqil_api.control_plane", "repair_coding_prompt", "legacy repair prompt"),
)


def _direct_state(**overrides: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "run_id": "run_1",
        "conversation_id": "conv_1",
        "prompt": "port it to fastapi",
        "model_aliases": {
            "_project_id": "asset_1",
            "_coding_engine": "clinecore",
            "_build_path": "cline_direct",
        },
        "project_contract": {
            "writable_roots": ["."],
            "protected_files": ["extractor.py"],
            "protected_hashes": {"extractor.py": "0" * 64},
            "acceptance": [],
            "authorized_scope": "Any file in this project.",
        },
    }
    state.update(overrides)
    return state


@pytest.fixture
def forbid_legacy(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Make every retired entry point fail loudly if it is reached."""

    called: list[str] = []

    for module, name, label in RETIRED:

        def _refuse(*_args: Any, __label: str = label, **_kwargs: Any) -> Any:
            called.append(__label)
            raise AssertionError(f"the direct path called the {__label}")

        monkeypatch.setattr(f"{module}.{name}", _refuse, raising=True)
    return called


@pytest.mark.asyncio
async def test_the_manifest_planner_is_never_reached_on_the_direct_path(
    forbid_legacy: list[str],
) -> None:
    """`_project_manifest` returns from the contract without asking a model."""

    class _Planner:
        def __init__(self) -> None:
            self.calls = 0

        async def __call__(self, *args: Any, **kwargs: Any) -> Any:
            self.calls += 1
            raise AssertionError("the direct path called the manifest planner")

    planner = _Planner()
    plane = type(
        "_Plane",
        (),
        {
            "_project_planner_call": planner,
            "events": type("_E", (), {"emit": staticmethod(_noop)})(),
        },
    )()

    plan = await ControlPlane._project_manifest(
        plane, _direct_state(), prompt_context={}, iterations=0, staged={}
    )

    assert planner.calls == 0
    assert forbid_legacy == []
    # No manifest, no slices, and no synthesized topology.
    assert plan["files"] is None
    assert plan["slices"] == []
    assert plan["slices_synthesized"] is False
    assert plan["scope"] == "direct"
    assert plan["taken"] is True


async def _noop(*_args: Any, **_kwargs: Any) -> None:
    return None


def test_the_direct_predicate_is_frozen_in_the_run_not_read_from_settings() -> None:
    # A rollback mid-run must not split a run across two designs.
    assert ControlPlane._uses_cline_direct(_direct_state()) is True
    assert (
        ControlPlane._uses_cline_direct(
            {"model_aliases": {"_build_path": "planner_slices"}}
        )
        is False
    )
    # Absent means the frozen legacy route, never the new one by default.
    assert ControlPlane._uses_cline_direct({"model_aliases": {}}) is False
    assert ControlPlane._uses_cline_direct({}) is False


def test_the_unified_prompt_never_tells_the_model_to_read_a_file_it_must_create() -> (
    None
):
    """The contradiction that cost a live session three failed edits."""

    from waqil_api.project_coding_engine import direct_coding_prompt

    prompt = direct_coding_prompt(
        task="port it",
        existing_files=["app.py", "config.py"],
        new_files=["README.md", "app/main.py"],
        protected_files=["extractor.py"],
        checks=["imports", "pytest"],
        findings=[{"path": "README.md", "error": "was planned and never written"}],
        attempt=2,
    )

    existing_block = prompt.split("FILES THAT ALREADY EXIST")[1].split(
        "FILES YOU MUST CREATE"
    )[0]
    create_block = prompt.split("FILES YOU MUST CREATE")[1].split("PROTECTED FILES")[0]

    for path in ("README.md", "app/main.py"):
        assert path in create_block, path
        assert path not in existing_block, path
    for path in ("app.py", "config.py"):
        assert path in existing_block, path
        assert path not in create_block, path

    # The instruction appears exactly once, and its opposite never applies to
    # a file in the create list.
    assert "do NOT call read_files on any path in this list first" in prompt
    assert "Read the exact current files before editing" not in prompt
    # A continuation carries findings in the SAME prompt, not a second one
    # appended underneath with its own contradictory preamble.
    assert prompt.count("FILES YOU MUST CREATE") == 1
    assert "VERIFIER FINDINGS FROM ROUND 1" in prompt


def test_a_path_named_both_ways_is_treated_as_existing() -> None:
    # Reading a file that exists is never the harmful mistake; being told to
    # create one that already exists is.
    from waqil_api.project_coding_engine import direct_coding_prompt

    prompt = direct_coding_prompt(
        task="t", existing_files=["app.py"], new_files=["app.py", "new.py"]
    )

    create_block = prompt.split("FILES YOU MUST CREATE")[1].split("PROTECTED FILES")[0]
    assert "app.py" not in create_block
    assert "new.py" in create_block


def test_the_direct_prompt_states_the_check_vocabulary_and_forbids_commands() -> None:
    from waqil_api.coding_contracts import HOST_CHECKS
    from waqil_api.project_coding_engine import direct_coding_prompt

    prompt = direct_coding_prompt(task="t", existing_files=[], checks=list(HOST_CHECKS))

    for check in HOST_CHECKS:
        assert check in prompt, check
    assert "You cannot pass a command, arguments, a path or a shell string" in prompt
    # It also has to say who judges the result, or a model will claim it did.
    # Compared without line wrapping: the prompt is hard-wrapped for reading.
    flat = " ".join(prompt.split())
    assert "Never claim a check passed" in flat


def test_a_falsely_clean_completion_is_still_judged_by_metis(tmp_path: Path) -> None:
    """The session's own verdict is not an input to approval.

    A model that reports success while leaving a blocking defect must still be
    stopped. Nothing in the direct path reads a model's completion claim: the
    approval gate reads the host verification, and only that.
    """

    from waqil_api.project_coding_engine import direct_coding_prompt

    prompt = direct_coding_prompt(task="t", existing_files=["app.py"])

    flat = " ".join(prompt.split())
    assert "Metis owns the boundary" in flat
    assert "computes the byte diff independently" in flat
    assert "never claim the work is approved" in flat
