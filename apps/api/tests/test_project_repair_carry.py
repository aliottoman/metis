"""A follow-up after a failed acceptance probe routes, or stops honestly.

Before this, carrying an undecided changeset into a repair turn reset the
verified frontier to zero. Every already-proven slice was replanned and
recoded to fix one file -- a live two-slice canary spent a second complete
build correcting one seed string. These pin the routing decision made at
carry time, including the case where nothing may be routed at all.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from waqil_api.config import Settings
from waqil_api.control_plane import ControlPlane


PLANNED = [
    "app/db.py",
    "app/main.py",
    "tests/test_workflows.py",
    "app/static/app.js",
    "app/static/index.html",
]

SLICES = [
    {
        "name": "SQLite-backed persistence",
        "outcome": "GET /api/projects reads from SQLite.",
        "files": PLANNED[:3],
        "owned_files": PLANNED[:3],
        "integration_files": [],
        "scenario_names": ["GET lists persisted projects"],
    },
    {
        "name": "Status workflow and UI",
        "outcome": "PATCH persists a status change.",
        "files": ["app/static/app.js", "app/static/index.html"] + PLANNED[1:3],
        "owned_files": ["app/static/app.js", "app/static/index.html"],
        "integration_files": PLANNED[1:3],
        "scenario_names": ["PATCH changes status"],
    },
]

STAGED = {path: {"content": f"# {path}\n", "bytes": len(path) + 3} for path in PLANNED}


class Events:
    def __init__(self) -> None:
        self.items: list[tuple[str, dict[str, Any]]] = []

    async def emit(
        self,
        run_id: str,
        conversation_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        del run_id, conversation_id
        self.items.append((event_type, payload))


def _plane(tmp_path: Path, *, findings: list[dict[str, Any]]) -> ControlPlane:
    plane = object.__new__(ControlPlane)
    plane.settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        repo_root=tmp_path,
        asset_roots=[tmp_path / "Projects"],
        model_backend="deterministic",
        allow_test_backends=True,
    )
    plane.events = Events()
    values = {
        "project_staged": dict(STAGED),
        "project_planned_files": list(PLANNED),
        "project_planned_slices": [dict(item) for item in SLICES],
        "project_planned_scenarios": [],
        "project_verified_prefix": len(PLANNED),
        "project_build_intent": "edit",
        "project_build_scope": "narrow",
        "project_repair_context": {"findings": findings, "files": []},
        "project_context": {"manifest": {"file_tree": list(PLANNED)}},
    }

    class Database:
        async def latest_awaiting_project_approval(
            self, conversation_id: str
        ) -> tuple[str, str]:
            del conversation_id
            return "run_prior", "asset_x"

        async def get_pending_approval(self, run_id: str) -> Any:
            del run_id
            return SimpleNamespace(
                kind="project_apply_build",
                blocked_reason="acceptance probe failed",
                summary="the seeded project is missing from the listing",
            )

    plane.database = Database()
    plane.checkpointer = SimpleNamespace(
        aget_tuple=_checkpoint(values),
    )
    plane._config = lambda conversation_id, run_id: {  # type: ignore[method-assign]
        "conversation_id": conversation_id,
        "run_id": run_id,
    }
    return plane


def _checkpoint(values: dict[str, Any]):
    async def aget_tuple(config: Any) -> Any:
        del config
        return SimpleNamespace(checkpoint={"channel_values": values})

    return aget_tuple


def _state() -> dict[str, Any]:
    return {
        "run_id": "run_next",
        "conversation_id": "conv_x",
        "prompt": "Fix the seeded project name.",
        "model_aliases": {"_project_id": "asset_x", "_coding_engine": "clinecore"},
    }


def _finding(path: str) -> dict[str, Any]:
    return {
        "path": path,
        "error": "acceptance: seeded project missing from listing",
        "severity": "error",
        "kind": "acceptance",
    }


@pytest.mark.asyncio
async def test_an_owned_file_defect_keeps_the_frontier_and_routes_one_slice(
    tmp_path: Path,
) -> None:
    plane = _plane(tmp_path, findings=[_finding("app/db.py")])

    carried = await ControlPlane._carry_pending_overlay(plane, _state())  # type: ignore[arg-type]

    # The frontier survives: those slices really were verified, and rolling it
    # to zero is what made one bad seed string cost a whole second build.
    assert carried["project_verified_prefix"] == len(PLANNED)
    repair = carried["project_repair_slice"]
    assert repair["target"] == "app/db.py"
    assert repair["attribution"] == "owned_file"
    assert repair["files"] == PLANNED[:3]
    assert repair["scenario_names"] == [
        "GET lists persisted projects",
        "PATCH changes status",
    ]
    # Every file the repair may not write is hashed first.
    assert set(carried["project_repair_hashes"]) == {
        "app/static/app.js",
        "app/static/index.html",
    }
    assert (
        carried["project_repair_hashes"]["app/static/app.js"]
        == hashlib.sha256(
            STAGED["app/static/app.js"]["content"].encode("utf-8")
        ).hexdigest()
    )
    # A routed repair always opens a fresh session over the carried overlay.
    assert carried["project_coding_session_id"] == ""
    [routed] = [
        payload
        for kind, payload in plane.events.items
        if kind == "project.repair_routed"
    ]
    assert routed["slice"] == "SQLite-backed persistence"
    assert routed["authorized_paths"] == PLANNED[:3]


@pytest.mark.asyncio
async def test_a_shared_file_defect_routes_to_the_latest_integrating_slice(
    tmp_path: Path,
) -> None:
    plane = _plane(tmp_path, findings=[_finding("app/main.py")])

    carried = await ControlPlane._carry_pending_overlay(plane, _state())  # type: ignore[arg-type]

    repair = carried["project_repair_slice"]
    assert repair["attribution"] == "integration_file"
    assert repair["name"] == "Status workflow and UI"
    assert "app/db.py" not in repair["files"]
    assert set(carried["project_repair_hashes"]) == {"app/db.py"}


@pytest.mark.asyncio
async def test_a_pathless_acceptance_failure_stops_without_arming_a_repair(
    tmp_path: Path,
) -> None:
    # The exact live shape: the host acceptance probe asserted, and its own
    # module is the only path on the finding. Nothing app-owned is named.
    plane = _plane(
        tmp_path,
        findings=[
            {"error": "AssertionError: seeded project missing", "severity": "error"}
        ],
    )

    carried = await ControlPlane._carry_pending_overlay(plane, _state())  # type: ignore[arg-type]

    # No repair scope is armed, so the coding round has nothing to start and
    # no model is pointed at a guessed file.
    assert not carried.get("project_repair_slice")
    assert not carried.get("project_repair_hashes")
    assert carried["project_verified_prefix"] == len(PLANNED)
    [unattributed] = [
        payload
        for kind, payload in plane.events.items
        if kind == "project.repair_unattributed"
    ]
    assert "no exact staged project file" in unattributed["reason"]
    assert not any(kind == "project.repair_routed" for kind, _ in plane.events.items)


@pytest.mark.asyncio
async def test_a_host_probe_path_outside_the_overlay_arms_no_repair(
    tmp_path: Path,
) -> None:
    plane = _plane(tmp_path, findings=[_finding("metis_eval_probe.py")])

    carried = await ControlPlane._carry_pending_overlay(plane, _state())  # type: ignore[arg-type]

    assert not carried.get("project_repair_slice")
    assert any(kind == "project.repair_unattributed" for kind, _ in plane.events.items)


@pytest.mark.asyncio
async def test_an_incomplete_build_still_re_establishes_its_own_frontier(
    tmp_path: Path,
) -> None:
    # Only a build that verified every slice may keep its frontier. A partial
    # build's remaining slices were never proven, so it starts over at zero.
    plane = _plane(tmp_path, findings=[_finding("app/db.py")])
    values = (await plane.checkpointer.aget_tuple(None)).checkpoint["channel_values"]
    values["project_verified_prefix"] = 3

    carried = await ControlPlane._carry_pending_overlay(plane, _state())  # type: ignore[arg-type]

    assert carried["project_verified_prefix"] == 0
    assert not carried.get("project_repair_slice")


@pytest.mark.asyncio
async def test_a_follow_up_carrying_no_findings_is_new_work_not_a_repair(
    tmp_path: Path,
) -> None:
    # A verified build the user follows up on with a fresh request is not an
    # acceptance repair. Preserving its frontier would make the next turn
    # answer "everything is already verified" and do none of the new work.
    plane = _plane(tmp_path, findings=[])

    carried = await ControlPlane._carry_pending_overlay(plane, _state())  # type: ignore[arg-type]

    assert carried["project_verified_prefix"] == 0
    assert not carried.get("project_repair_slice")
    assert not any(
        kind in {"project.repair_routed", "project.repair_unattributed"}
        for kind, _ in plane.events.items
    )
