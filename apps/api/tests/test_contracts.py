from __future__ import annotations

import pytest
from pydantic import ValidationError

from waqil_api.config import Settings
from waqil_api.contracts import (
    MAX_PLAN_SCENARIOS,
    AcceptanceScenarioV1,
    ProjectVerticalSliceV1,
    ArchitectureSpecV1,
    MessageCreateV1,
    ProjectBuildPlanV1,
)


def _scenario(name: str) -> dict[str, str]:
    return {"name": name, "method": "GET", "path": "/x", "expect_status": "2xx"}


def test_a_slice_naming_more_scenarios_than_the_host_keeps_still_parses() -> None:
    """The exact GLM canary failure: a planner-declared slice named seven
    scenario_names on a schema that then capped them at five. Rejecting the
    whole ProjectBuildPlanV1 for that discarded a valid, complete 16-file
    plan -- recovery must keep it.

    Seven names are now KEPT, not trimmed: the per-slice five-item cap was a
    separate, smaller policy than the plan's own eight-scenario limit, and it
    silently discarded planner commitments -- which is what refused a correct
    six-scenario Atlas plan a live run later produced."""
    names = [f"scenario_{index}" for index in range(7)]
    files = [f"app/f{index}.py" for index in range(16)]
    plan = ProjectBuildPlanV1(
        intent="build",
        scope="narrow",
        files=files,
        scenarios=[_scenario(name) for name in names],
        slices=[
            {
                "name": "Slice 1",
                "outcome": "x",
                "files": files[:6],
                "scenario_names": names[:2],
            },
            {
                "name": "Slice 2",
                "outcome": "x",
                "files": files[6:12],
                "scenario_names": names[:2],
            },
            # The exact overflow shape observed live: 7 names on one slice.
            {
                "name": "Slice 3",
                "outcome": "x",
                "files": files[12:],
                "scenario_names": names,
            },
        ],
    )
    # No file was ever at risk -- the six-file-per-slice bound stayed a hard
    # reject the whole time; only the harmless scenario_names overflow needed
    # to become recoverable.
    assert plan.files == files
    assert [s.files for s in plan.slices] == [files[:6], files[6:12], files[12:]]
    # Deduplicated and filtered to declared scenarios -- and every declared
    # name survives, because a slice may legitimately be responsible for all
    # of the plan's scenarios.
    assert plan.slices[2].scenario_names == names
    assert all(len(s.scenario_names) <= MAX_PLAN_SCENARIOS for s in plan.slices)


def test_a_slice_naming_an_undeclared_scenario_drops_only_that_name() -> None:
    plan = ProjectBuildPlanV1(
        intent="build",
        scope="narrow",
        files=["app/a.py", "app/b.py"],
        scenarios=[_scenario("real_one")],
        slices=[
            {
                "name": "Slice 1",
                "outcome": "x",
                "files": ["app/a.py", "app/b.py"],
                "scenario_names": ["real_one", "real_one", "made_up", "also_made_up"],
            }
        ],
    )
    assert plan.slices[0].scenario_names == ["real_one"]


def test_a_valid_multi_slice_plan_is_unaffected_by_the_recovery_path() -> None:
    """The ordinary, well-formed case must parse exactly as before: recovery
    for harmless excess must never change behavior when there is no excess."""
    files = [f"app/f{index}.py" for index in range(12)]
    plan = ProjectBuildPlanV1(
        intent="build",
        scope="narrow",
        files=files,
        scenarios=[_scenario("health"), _scenario("create")],
        slices=[
            {
                "name": "Slice 1",
                "outcome": "the health endpoint responds",
                "files": files[:6],
                "scenario_names": ["health"],
            },
            {
                "name": "Slice 2",
                "outcome": "records can be created",
                "files": files[6:],
                "scenario_names": ["create"],
            },
        ],
    )
    assert plan.files == files
    assert len(plan.slices) == 2
    assert plan.slices[0].files == files[:6]
    assert plan.slices[0].scenario_names == ["health"]
    assert plan.slices[1].files == files[6:]
    assert plan.slices[1].scenario_names == ["create"]


def test_contracts_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        MessageCreateV1(content="hello", unexpected=True)


def test_message_knowledge_scope_is_explicit_and_bounded() -> None:
    assert MessageCreateV1(content="hello").knowledge_scope == "auto"
    assert (
        MessageCreateV1(content="hello", knowledge_scope="notion").knowledge_scope
        == "notion"
    )
    with pytest.raises(ValidationError):
        MessageCreateV1(content="hello", knowledge_scope="internet")


def test_architecture_contract_matches_portable_tool() -> None:
    spec = ArchitectureSpecV1.model_validate(
        {
            "title": "Service",
            "provider": "generic",
            "direction": "LR",
            "components": [{"id": "api", "label": "API", "kind": "service"}],
            "edges": [],
            "boundaries": [],
            "assumptions": [],
            "unresolved_ambiguities": [],
        }
    )
    assert spec.components[0].id == "api"


def test_non_loopback_ollama_is_rejected(tmp_path) -> None:
    with pytest.raises(ValidationError):
        Settings(data_dir=tmp_path, ollama_base_url="https://example.com")


def test_architecture_references_and_boundaries_are_strict() -> None:
    with pytest.raises(ValidationError):
        ArchitectureSpecV1.model_validate(
            {
                "title": "Invalid",
                "components": [{"id": "api", "label": "API", "kind": "service"}],
                "edges": [{"source": "api", "target": "missing"}],
            }
        )
    with pytest.raises(ValidationError):
        ArchitectureSpecV1.model_validate(
            {
                "title": "Overlap",
                "components": [{"id": "api", "label": "API", "kind": "service"}],
                "edges": [],
                "boundaries": [
                    {"id": "one", "label": "One", "component_ids": ["api"]},
                    {"id": "two", "label": "Two", "component_ids": ["api"]},
                ],
            }
        )


def test_an_exact_json_assertion_is_optional_bounded_and_closed_enum() -> None:
    """The strict alternative to containment, and its guard rails."""

    # Absent by default: every scenario written before this field behaves
    # exactly as it did.
    plain = AcceptanceScenarioV1(name="listing responds", path="/api/projects")
    assert plain.expect_json_exact is None
    assert plain.json_match == "exact"

    exact = AcceptanceScenarioV1(
        name="listing is exactly the seeded records",
        path="/api/projects",
        expect_json_exact=[
            {"id": "proj-1", "name": "Harbor Migration"},
            {"id": "proj-2", "name": "Invoice Intelligence"},
        ],
        json_match="unordered_array",
    )
    assert exact.expect_json_exact[0]["name"] == "Harbor Migration"
    assert exact.json_match == "unordered_array"

    # Set semantics are a closed choice, never inferred from the data.
    with pytest.raises(ValidationError):
        AcceptanceScenarioV1(
            name="guessy", path="/api/projects", json_match="set_of_whatever"
        )

    # Bounded: this rides the local grammar, the checkpoint, and the sandbox
    # request, so an unbounded structure is refused rather than truncated.
    with pytest.raises(ValidationError):
        AcceptanceScenarioV1(
            name="enormous",
            path="/api/projects",
            expect_json_exact=[{"id": f"proj-{index}" * 40} for index in range(200)],
        )


# ── Ownership is authored; `files` is derived ──────────────────────────────
# A live planner returned slices whose combined `files` disagreed with its own
# owned/integration split, and the plan was rejected for the disagreement.
# Restating the union was redundant work the host can do exactly.


def _slice(**fields: object) -> dict[str, object]:
    base: dict[str, object] = {"name": "Slice", "outcome": "It works."}
    base.update(fields)
    return base


def test_a_slice_may_author_ownership_alone_and_files_is_derived() -> None:
    plan = ProjectBuildPlanV1(
        intent="edit",
        scope="narrow",
        files=["app/a.py", "app/b.py", "app/c.py"],
        scenarios=[_scenario("one")],
        slices=[
            _slice(owned_files=["app/a.py", "app/b.py"], scenario_names=["one"]),
            _slice(owned_files=["app/c.py"], integration_files=["app/a.py"]),
        ],
    )

    assert plan.slices[0].files == ["app/a.py", "app/b.py"]
    # Derived canonically: owned first, then the integration points.
    assert plan.slices[1].files == ["app/c.py", "app/a.py"]
    assert plan.slices[1].owned_files == ["app/c.py"]
    assert plan.slices[1].integration_files == ["app/a.py"]


def test_a_redundant_disagreeing_files_list_is_ignored_not_reconciled() -> None:
    """The exact live failure shape, now harmless."""

    plan = ProjectBuildPlanV1(
        intent="edit",
        scope="narrow",
        files=["app/a.py", "app/b.py"],
        scenarios=[],
        slices=[
            _slice(owned_files=["app/a.py"]),
            # `files` disagrees with the ownership split; ownership wins.
            _slice(
                files=["app/b.py", "app/a.py", "app/invented.py"],
                owned_files=["app/b.py"],
                integration_files=["app/a.py"],
            ),
        ],
    )

    assert plan.slices[1].files == ["app/b.py", "app/a.py"]
    assert "app/invented.py" not in plan.slices[1].files


def test_a_files_only_plan_keeps_its_existing_behaviour(*_: object) -> None:
    plan = ProjectBuildPlanV1(
        intent="edit",
        scope="narrow",
        files=["app/a.py", "app/b.py"],
        scenarios=[],
        slices=[_slice(files=["app/a.py", "app/b.py"])],
    )

    assert plan.slices[0].files == ["app/a.py", "app/b.py"]
    assert plan.slices[0].owned_files == ["app/a.py", "app/b.py"]
    assert plan.slices[0].integration_files == []


def test_a_file_owned_and_integrated_by_one_slice_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ProjectVerticalSliceV1(
            name="Slice",
            outcome="x",
            owned_files=["app/a.py"],
            integration_files=["app/a.py"],
        )


def test_an_ownership_partition_that_loses_a_planned_file_is_cleared() -> None:
    plan = ProjectBuildPlanV1(
        intent="edit",
        scope="narrow",
        files=["app/a.py", "app/b.py", "app/c.py"],
        scenarios=[],
        slices=[_slice(owned_files=["app/a.py", "app/b.py"])],
    )

    # Cleared to the host's safe deterministic partition rather than executed.
    assert plan.slices == []


def test_an_ownership_partition_that_duplicates_or_invents_is_cleared() -> None:
    duplicated = ProjectBuildPlanV1(
        intent="edit",
        scope="narrow",
        files=["app/a.py", "app/b.py"],
        scenarios=[],
        slices=[
            _slice(owned_files=["app/a.py"]),
            _slice(owned_files=["app/a.py", "app/b.py"]),
        ],
    )
    assert duplicated.slices == []

    invented = ProjectBuildPlanV1(
        intent="edit",
        scope="narrow",
        files=["app/a.py"],
        scenarios=[],
        slices=[_slice(owned_files=["app/a.py", "app/never_planned.py"])],
    )
    assert invented.slices == []


def test_an_ownership_partition_that_reorders_the_manifest_is_cleared() -> None:
    plan = ProjectBuildPlanV1(
        intent="edit",
        scope="narrow",
        files=["app/a.py", "app/b.py"],
        scenarios=[],
        slices=[_slice(owned_files=["app/b.py"]), _slice(owned_files=["app/a.py"])],
    )

    assert plan.slices == []


def test_an_integration_path_must_still_name_a_strictly_earlier_owned_file() -> None:
    forward = ProjectBuildPlanV1(
        intent="edit",
        scope="narrow",
        files=["app/a.py", "app/b.py"],
        scenarios=[],
        slices=[
            # Integrates a file the LATER slice owns.
            _slice(owned_files=["app/a.py"], integration_files=["app/b.py"]),
            _slice(owned_files=["app/b.py"]),
        ],
    )
    assert forward.slices == []

    own_slice = ProjectBuildPlanV1(
        intent="edit",
        scope="narrow",
        files=["app/a.py", "app/b.py"],
        scenarios=[],
        slices=[
            _slice(owned_files=["app/a.py"]),
            _slice(owned_files=["app/b.py"], integration_files=["app/a.py"]),
        ],
    )
    assert [item.owned_files for item in own_slice.slices] == [
        ["app/a.py"],
        ["app/b.py"],
    ]
