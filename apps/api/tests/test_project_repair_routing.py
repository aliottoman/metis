"""One acceptance defect routes to one slice, never to a whole rebuild."""

from __future__ import annotations

from typing import Any

from waqil_api.project_repair_routing import (
    changed_unaffected_paths,
    cumulative_scenarios,
    route_acceptance_finding,
    slice_ownership,
    staged_file_hashes,
)


# The exact topology the live two-slice canary produced: slice 1 owns
# persistence, slice 2 owns the UI and explicitly reopens the entrypoint and
# the workflow test that slice 1 created.
SLICES: list[dict[str, Any]] = [
    {
        "name": "SQLite-backed persistence",
        "outcome": "GET /api/projects reads from SQLite.",
        "files": ["app/db.py", "app/main.py", "tests/test_workflows.py"],
        "owned_files": ["app/db.py", "app/main.py", "tests/test_workflows.py"],
        "integration_files": [],
        "scenario_names": ["GET lists persisted projects"],
    },
    {
        "name": "Status workflow and UI",
        "outcome": "PATCH persists a status change and the UI acts on it.",
        "files": [
            "app/static/app.js",
            "app/static/index.html",
            "app/main.py",
            "tests/test_workflows.py",
        ],
        "owned_files": ["app/static/app.js", "app/static/index.html"],
        "integration_files": ["app/main.py", "tests/test_workflows.py"],
        "scenario_names": ["PATCH changes status", "GET reflects patched status"],
    },
]


def _staged(**overrides: str) -> dict[str, dict[str, Any]]:
    content = {
        "app/db.py": "SEED = 'harbor migration'\n",
        "app/main.py": "APP = True\n",
        "tests/test_workflows.py": "def test_x(): ...\n",
        "app/static/app.js": "fetch('/api/projects');\n",
        "app/static/index.html": "<html></html>\n",
        **overrides,
    }
    return {path: {"content": value} for path, value in content.items()}


def _finding(path: str, error: str = "seed value is wrong") -> dict[str, Any]:
    return {"path": path, "error": error, "severity": "error", "kind": "acceptance"}


def test_ownership_index_separates_exclusive_work_from_integration_points() -> None:
    owner, latest_integrator = slice_ownership(SLICES)

    assert owner == {
        "app/db.py": 0,
        "app/main.py": 0,
        "tests/test_workflows.py": 0,
        "app/static/app.js": 1,
        "app/static/index.html": 1,
    }
    assert latest_integrator == {"app/main.py": 1, "tests/test_workflows.py": 1}


def test_an_owned_file_defect_routes_to_its_owning_slice_alone() -> None:
    route = route_acceptance_finding(
        findings=[_finding("app/db.py")],
        slices=SLICES,
        staged=_staged(),
    )

    assert route.routed is True
    assert route.slice_index == 0
    assert route.attribution == "owned_file"
    assert route.slice_name == "SQLite-backed persistence"
    # Exactly slice 1's own declared scope -- never slice 2's owned UI files.
    assert route.authorized_paths == (
        "app/db.py",
        "app/main.py",
        "tests/test_workflows.py",
    )
    assert "app/static/app.js" not in route.authorized_paths
    assert "app/static/index.html" not in route.authorized_paths


def test_a_shared_integration_file_routes_to_the_latest_integrating_slice() -> None:
    # app/main.py is OWNED by slice 1 but explicitly reopened by slice 2, so
    # slice 2's work is what is actually in those bytes now.
    route = route_acceptance_finding(
        findings=[_finding("app/main.py", "PATCH returned 404")],
        slices=SLICES,
        staged=_staged(),
    )

    assert route.slice_index == 1
    assert route.attribution == "integration_file"
    assert route.slice_name == "Status workflow and UI"
    assert set(route.authorized_paths) == set(SLICES[1]["files"])


def test_every_earlier_and_downstream_scenario_is_replayed_after_a_repair() -> None:
    route = route_acceptance_finding(
        findings=[_finding("app/db.py")],
        slices=SLICES,
        staged=_staged(),
    )

    # A slice-1 repair can regress slice 1's own claim AND everything slice 2
    # built on top of it, so the whole proven set is replayed -- not just the
    # repaired slice's own scenario.
    assert route.rerun_scenarios == (
        "GET lists persisted projects",
        "PATCH changes status",
        "GET reflects patched status",
    )
    assert cumulative_scenarios(SLICES, through_index=0) == [
        "GET lists persisted projects"
    ]


def test_a_pathless_acceptance_finding_stops_instead_of_guessing_a_slice() -> None:
    route = route_acceptance_finding(
        findings=[
            {"error": "AssertionError: seeded project missing", "severity": "error"}
        ],
        slices=SLICES,
        staged=_staged(),
    )

    assert route.routed is False
    assert route.slice_index == -1
    assert route.authorized_paths == ()
    assert "no exact staged project file" in route.reason


def test_a_host_probe_path_outside_the_overlay_is_not_attributed_to_a_slice() -> None:
    # The exact live shape: the acceptance probe's own module failed, and its
    # path is the probe file, not app-owned code.
    route = route_acceptance_finding(
        findings=[_finding("metis_eval_projects_dashboard.py", "AssertionError")],
        slices=SLICES,
        staged=_staged(),
    )

    assert route.routed is False
    assert "no exact staged project file" in route.reason


def test_a_staged_file_no_slice_claims_is_not_repaired_speculatively() -> None:
    route = route_acceptance_finding(
        findings=[_finding("app/orphan.py")],
        slices=SLICES,
        staged={**_staged(), "app/orphan.py": {"content": "X = 1\n"}},
    )

    assert route.routed is False
    assert "not owned or integrated by any planned slice" in route.reason


def test_defects_outside_the_routed_scope_wait_for_their_own_repair() -> None:
    route = route_acceptance_finding(
        findings=[_finding("app/db.py"), _finding("app/static/app.js")],
        slices=SLICES,
        staged=_staged(),
    )

    assert route.slice_index == 0
    assert "app/static/app.js" not in route.authorized_paths
    assert "deferred to their own routed repairs: app/static/app.js" in route.reason
    # Only the in-scope defect rides into this repair's finding set.
    assert [str(item["path"]) for item in route.findings] == ["app/db.py"]


def test_unaffected_hashes_prove_a_repair_touched_only_its_authorized_file() -> None:
    staged = _staged()
    before = staged_file_hashes(staged, exclude=["app/db.py"])

    assert "app/db.py" not in before
    assert set(before) == {
        "app/main.py",
        "tests/test_workflows.py",
        "app/static/app.js",
        "app/static/index.html",
    }

    repaired = {**staged, "app/db.py": {"content": "SEED = 'Harbor Migration'\n"}}
    assert changed_unaffected_paths(before, repaired) == []

    # The failure this guard exists for: a repair that also rewrote a file it
    # was never authorized to touch.
    overreached = {**repaired, "app/static/app.js": {"content": "// rewritten\n"}}
    assert changed_unaffected_paths(before, overreached) == ["app/static/app.js"]

    # A deletion is a change too, not an absence of evidence.
    deleted = {
        path: value for path, value in repaired.items() if path != "app/static/app.js"
    }
    assert changed_unaffected_paths(before, deleted) == ["app/static/app.js"]


def test_an_empty_slice_plan_never_routes_a_repair() -> None:
    route = route_acceptance_finding(
        findings=[_finding("app/db.py")],
        slices=[],
        staged=_staged(),
    )

    assert route.routed is False
    assert "no vertical slice plan" in route.reason


def test_a_legacy_slice_without_an_ownership_split_still_routes() -> None:
    legacy = [
        {
            "name": "Everything",
            "outcome": "One chunk.",
            "files": ["app/db.py", "app/main.py"],
            "scenario_names": ["it works"],
        }
    ]

    route = route_acceptance_finding(
        findings=[_finding("app/db.py")],
        slices=legacy,
        staged=_staged(),
    )

    assert route.routed is True
    assert route.slice_index == 0
    assert route.authorized_paths == ("app/db.py", "app/main.py")
