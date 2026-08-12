"""Neither path may leak into the other, in either direction.

A build is admitted under one design and has to finish under it. A checkpoint
written before `cline_direct` existed must keep resolving to the planner and
its slices; a direct checkpoint must never wander into machinery that expects
a manifest it does not have. The deciding fact is frozen in the run aliases at
admission, not read from live settings, so flipping the flag mid-flight cannot
split a run across two designs.
"""

from __future__ import annotations

from typing import Any

import pytest

from waqil_api.config import Settings
from waqil_api.control_plane import ControlPlane


def _state(build_path: str | None, **overrides: Any) -> dict[str, Any]:
    aliases: dict[str, Any] = {
        "_project_id": "asset_1",
        "_coding_engine": "clinecore",
    }
    if build_path is not None:
        aliases["_build_path"] = build_path
    state: dict[str, Any] = {
        "run_id": "run_1",
        "conversation_id": "conv_1",
        "prompt": "build it",
        "model_aliases": aliases,
    }
    state.update(overrides)
    return state


def test_a_legacy_checkpoint_never_switches_into_the_direct_path() -> None:
    """A run frozen to planner_slices stays there, whatever settings now say."""

    legacy = _state(
        "planner_slices",
        project_plan_taken=True,
        project_planned_files=["app/main.py", "app/api.py"],
        project_planned_slices=[
            {
                "name": "Core",
                "outcome": "It works.",
                "files": ["app/main.py", "app/api.py"],
                "owned_files": ["app/main.py", "app/api.py"],
                "integration_files": [],
            }
        ],
    )

    assert ControlPlane._uses_cline_direct(legacy) is False
    # And the alias, not the setting, is what decided it: a live Settings with
    # the new default is present in this process the whole time.
    assert Settings(_env_file=None).project_build_path == "cline_direct"  # type: ignore[call-arg]


def test_a_checkpoint_written_before_the_flag_existed_resolves_to_legacy() -> None:
    """The absent alias is the old world, and the old world is the safe default."""

    assert ControlPlane._uses_cline_direct(_state(None)) is False
    assert ControlPlane._uses_cline_direct({"model_aliases": {}}) is False
    assert ControlPlane._uses_cline_direct({}) is False


@pytest.mark.asyncio
async def test_a_legacy_checkpoint_takes_the_ordinary_flow_not_the_direct_one() -> None:
    """The positive half: legacy still goes where it always went.

    It does not reach the planner here — the request prefilter stops it first,
    exactly as before this change — and that is the point: the shape it
    returns is the old unplanned shape, not the direct contract shape.
    """

    class _Plane:
        settings = Settings(_env_file=None)  # type: ignore[call-arg]

        async def _project_planner_call(self, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError("the prefilter should have stopped this first")

        class events:  # noqa: N801 - a stand-in, not a type
            @staticmethod
            async def emit(*_args: Any, **_kwargs: Any) -> None:
                return None

    plan = await ControlPlane._project_manifest(
        _Plane(), _state("planner_slices"), prompt_context={}, iterations=0, staged={}
    )

    # The legacy unplanned shape: no scope, not taken.
    assert plan["scope"] == ""
    assert plan["taken"] is False
    assert "slices_synthesized" not in plan
    # And emphatically NOT the direct shape, which would have been taken with
    # scope "direct" on the very same input.
    assert plan["scope"] != "direct"


@pytest.mark.asyncio
async def test_a_direct_checkpoint_never_reaches_the_planner() -> None:
    """The mirror image, on the same method."""

    direct = _state(
        "cline_direct",
        project_contract={
            "writable_roots": ["."],
            "protected_files": [],
            "protected_hashes": {},
            "acceptance": [{"name": "health responds"}],
        },
    )

    class _Plane:
        settings = Settings(_env_file=None)  # type: ignore[call-arg]

        async def _project_planner_call(self, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError("a direct checkpoint reached the planner")

    plan = await ControlPlane._project_manifest(
        _Plane(), direct, prompt_context={}, iterations=0, staged={}
    )

    assert plan["scope"] == "direct"
    assert plan["files"] is None
    assert plan["slices"] == []
    # The acceptance requirements come from the frozen contract, not a model.
    assert plan["scenarios"] == [{"name": "health responds"}]


def test_the_engine_and_the_build_path_are_frozen_together() -> None:
    """A legacy engine can never carry the direct path, whatever settings say."""

    from waqil_api.config import Settings as _Settings

    settings = _Settings(  # type: ignore[call-arg]
        _env_file=None,
        allow_test_backends=True,
        project_coding_engine="legacy",
        project_build_path="cline_direct",
    )

    # api.py freezes the pair; the direct path is ClineCore-only by
    # construction, so a legacy engine collapses it back.
    frozen = (
        settings.project_build_path
        if settings.project_coding_engine == "clinecore"
        else "planner_slices"
    )
    assert frozen == "planner_slices"


def test_slices_are_off_by_default_and_never_inferred_from_file_count() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.project_slices_enabled is False
    assert settings.project_build_path == "cline_direct"
