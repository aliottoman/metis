"""Slice topology is judged before a model is asked to build it.

The plan this exists for came out of a live qualification run: a five-file
narrow edit split into a UI shell, its JavaScript, a tests-only slice and a
docs-only slice. Nothing rejected it, and a coder spent 270,522 tokens on the
first horizontal slice before an external gate noticed.
"""

from __future__ import annotations

from typing import Any

from waqil_api.contracts import MAX_SLICE_FILES
from waqil_api.project_plan_validation import (
    support_kind,
    validate_build_plan,
)


ATLAS_FILES = [
    "app/static/index.html",
    "app/static/styles.css",
    "app/static/app.js",
    "tests/test_ui_contract.py",
    "README.md",
]

# Copied from the rejected run's durable checkpoint, names included.
ATLAS_FOUR_SLICE_PLAN: list[dict[str, Any]] = [
    {
        "name": "Semantic HTML shell and design tokens",
        "outcome": "The index page becomes a semantic Atlas Operations Console.",
        "files": ["app/static/index.html", "app/static/styles.css"],
        "owned_files": ["app/static/index.html", "app/static/styles.css"],
        "integration_files": [],
        "scenario_names": ["ui_page_has_console_landmarks"],
    },
    {
        "name": "Client application logic",
        "outcome": "app.js fetches the API and renders the table.",
        "files": ["app/static/app.js"],
        "owned_files": ["app/static/app.js"],
        "integration_files": [],
        "scenario_names": ["projects_exact_records"],
    },
    {
        "name": "UI contract tests",
        "outcome": "A pytest suite verifies the page and its assets.",
        "files": ["tests/test_ui_contract.py"],
        "owned_files": ["tests/test_ui_contract.py"],
        "integration_files": [],
        "scenario_names": [],
    },
    {
        "name": "Documentation update",
        "outcome": "README.md documents the console and the API contract.",
        "files": ["README.md"],
        "owned_files": ["README.md"],
        "integration_files": [],
        "scenario_names": [],
    },
]

ATLAS_TWO_SLICE_PLAN: list[dict[str, Any]] = [
    {
        "name": "Console shell renders live project data",
        "outcome": "The page loads, styles itself and lists the seeded projects.",
        "files": [
            "app/static/index.html",
            "app/static/styles.css",
            "app/static/app.js",
        ],
        "owned_files": [
            "app/static/index.html",
            "app/static/styles.css",
            "app/static/app.js",
        ],
        "integration_files": [],
        "scenario_names": ["projects_exact_records"],
    },
    {
        "name": "Status workflow, its contract tests and docs",
        "outcome": "A row's status action PATCHes the API and the page updates.",
        "files": [
            "tests/test_ui_contract.py",
            "README.md",
            "app/static/app.js",
        ],
        "owned_files": ["tests/test_ui_contract.py", "README.md"],
        "integration_files": ["app/static/app.js"],
        "scenario_names": ["patch_status_returns_updated_project"],
    },
]

# A shape normalization must refuse rather than approximate: the tests come
# BEFORE any runtime slice, so there is no earlier outcome for them to belong
# to and merging forward would invent a dependency the planner never declared.
ATLAS_SUPPORT_FIRST_PLAN: list[dict[str, Any]] = [
    {
        "name": "UI contract tests",
        "outcome": "A pytest suite verifies the page and its assets.",
        "files": ["tests/test_ui_contract.py"],
        "owned_files": ["tests/test_ui_contract.py"],
        "integration_files": [],
        "scenario_names": [],
    },
    {
        "name": "Console shell and logic",
        "outcome": "The page renders the seeded projects.",
        "files": [
            "app/static/index.html",
            "app/static/styles.css",
            "app/static/app.js",
            "README.md",
        ],
        "owned_files": [
            "app/static/index.html",
            "app/static/styles.css",
            "app/static/app.js",
            "README.md",
        ],
        "integration_files": [],
        "scenario_names": ["projects_exact_records"],
    },
]

ATLAS_SUPPORT_FIRST_FILES = [
    "tests/test_ui_contract.py",
    "app/static/index.html",
    "app/static/styles.css",
    "app/static/app.js",
    "README.md",
]


SCENARIOS = [
    {"name": "ui_page_has_console_landmarks"},
    {"name": "projects_exact_records"},
    {"name": "patch_status_returns_updated_project"},
]


def test_support_files_are_recognized_by_shape_not_by_guesswork() -> None:
    assert support_kind("tests/test_ui_contract.py") == "test"
    assert support_kind("app/widget_test.py") == "test"
    assert support_kind("web/button.spec.ts") == "test"
    assert support_kind("README.md") == "documentation"
    assert support_kind("docs/architecture.rst") == "documentation"
    assert support_kind("requirements.txt") == "dependency"
    assert support_kind("package.json") == "dependency"
    # Runtime files, including ones that merely live near tests.
    assert support_kind("app/static/app.js") == ""
    assert support_kind("app/main.py") == ""
    assert support_kind("app/contest.py") == ""


def test_the_exact_rejected_atlas_plan_is_rejected_for_its_support_slices() -> None:
    result = validate_build_plan(
        ATLAS_FILES, ATLAS_FOUR_SLICE_PLAN, scenarios=SCENARIOS
    )

    assert result.ok is False
    assert result.codes.count("support_only_slice") == 2
    joined = " ".join(result.findings)
    assert "tests/test_ui_contract.py" in joined
    assert "README.md" in joined
    assert "delivers nothing runnable" in joined
    assert "attach them to the slice whose feature they validate" in joined
    # The correction is written to be handed straight to a planner.
    correction = result.correction_text()
    assert "rejected before any code was written" in correction
    assert "integration_files" in correction


def test_a_valid_two_slice_atlas_plan_with_an_integration_file_passes() -> None:
    result = validate_build_plan(ATLAS_FILES, ATLAS_TWO_SLICE_PLAN, scenarios=SCENARIOS)

    assert result.ok is True, result.findings
    assert result.correction_text() == ""


def test_a_narrow_edit_is_judged_by_the_same_rule_as_a_whole_app_build() -> None:
    # The gap the live failure exposed: the only structural rule in the
    # contract applied to whole_app builds, so a narrow edit could be sliced
    # horizontally with nothing objecting.
    files = ["app/service.py", "tests/test_service.py"]
    horizontal = [
        {
            "name": "Service",
            "outcome": "The service does the thing.",
            "files": ["app/service.py"],
            "owned_files": ["app/service.py"],
            "integration_files": [],
        },
        {
            "name": "Tests",
            "outcome": "Tests cover the service.",
            "files": ["tests/test_service.py"],
            "owned_files": ["tests/test_service.py"],
            "integration_files": [],
        },
    ]

    result = validate_build_plan(files, horizontal)

    assert result.ok is False
    assert "support_only_slice" in result.codes


def test_a_genuinely_documentation_only_request_is_still_plannable() -> None:
    """The rule must not make a README edit unplannable.

    A support-only slice is a defect because the work belonged beside a
    feature. When the plan has no feature at all, there is nothing to attach
    it to and the request really is a docs edit.
    """

    result = validate_build_plan(
        ["README.md", "docs/setup.md"],
        [
            {
                "name": "Documentation refresh",
                "outcome": "The README and setup guide describe the current app.",
                "files": ["README.md", "docs/setup.md"],
                "owned_files": ["README.md", "docs/setup.md"],
                "integration_files": [],
            }
        ],
    )

    assert result.ok is True, result.findings


def test_valid_greenfield_vertical_slices_pass() -> None:
    files = [
        "app/db.py",
        "app/main.py",
        "tests/test_workflows.py",
        "app/static/app.js",
        "app/static/index.html",
    ]
    plan = [
        {
            "name": "Seeded project listing",
            "outcome": "GET /api/projects returns the seeded records.",
            "files": ["app/db.py", "app/main.py", "tests/test_workflows.py"],
            "owned_files": ["app/db.py", "app/main.py", "tests/test_workflows.py"],
            "integration_files": [],
            "scenario_names": ["listing"],
        },
        {
            "name": "Dashboard UI",
            "outcome": "The page renders the listing from the API.",
            "files": ["app/static/app.js", "app/static/index.html", "app/main.py"],
            "owned_files": ["app/static/app.js", "app/static/index.html"],
            "integration_files": ["app/main.py"],
            "scenario_names": ["dashboard"],
        },
    ]

    result = validate_build_plan(
        files, plan, scenarios=[{"name": "listing"}, {"name": "dashboard"}]
    )

    assert result.ok is True, result.findings


def test_an_undeclared_re_edit_of_an_earlier_file_is_rejected() -> None:
    files = ["app/main.py", "app/extra.py"]
    plan = [
        {
            "name": "Core",
            "outcome": "The app serves.",
            "files": ["app/main.py"],
            "owned_files": ["app/main.py"],
            "integration_files": [],
        },
        {
            "name": "Extra",
            "outcome": "The extra endpoint works.",
            # Re-opens app/main.py without declaring it.
            "files": ["app/extra.py", "app/main.py"],
            "owned_files": ["app/extra.py"],
            "integration_files": [],
        },
    ]

    result = validate_build_plan(files, plan)

    assert result.ok is False
    assert "undeclared_reedit" in result.codes


def test_ownership_dependency_order_and_scenario_names_are_all_checked() -> None:
    files = ["app/a.py", "app/b.py"]

    twice = validate_build_plan(
        files,
        [
            {"name": "one", "files": ["app/a.py"], "owned_files": ["app/a.py"]},
            {
                "name": "two",
                "files": ["app/a.py", "app/b.py"],
                "owned_files": ["app/a.py", "app/b.py"],
            },
        ],
    )
    assert "owned_more_than_once" in twice.codes

    unowned = validate_build_plan(
        files,
        [{"name": "one", "files": ["app/a.py"], "owned_files": ["app/a.py"]}],
    )
    assert "unowned_file" in unowned.codes

    out_of_order = validate_build_plan(
        files,
        [
            {"name": "one", "files": ["app/b.py"], "owned_files": ["app/b.py"]},
            {"name": "two", "files": ["app/a.py"], "owned_files": ["app/a.py"]},
        ],
    )
    assert "dependency_order" in out_of_order.codes

    unknown = validate_build_plan(
        files,
        [
            {
                "name": "one",
                "files": files,
                "owned_files": files,
                "scenario_names": ["never declared"],
            }
        ],
        scenarios=[{"name": "declared"}],
    )
    assert "unknown_scenario" in unknown.codes


def test_a_plan_without_declared_slices_is_left_to_the_host_chunker() -> None:
    # Absent slices are not a topology defect: vertical_build_slices owns that
    # case and its mechanical chunks never produce a support-only slice.
    assert validate_build_plan(ATLAS_FILES, []).ok is True
    assert validate_build_plan([], ATLAS_FOUR_SLICE_PLAN).ok is True


def test_no_general_rule_fixes_the_number_of_slices() -> None:
    """Two slices is an Atlas qualification requirement, not a Metis rule."""

    files = [f"app/feature_{index}.py" for index in range(4)]
    plan = [
        {
            "name": f"Feature {index}",
            "outcome": f"Feature {index} works end to end.",
            "files": [path],
            "owned_files": [path],
            "integration_files": [],
        }
        for index, path in enumerate(files)
    ]

    assert validate_build_plan(files, plan).ok is True


# ── The gate, in the control plane ─────────────────────────────────────────
# Validation being correct is not the same as it running before inference.
# These drive _project_manifest with a fake planner and a coder that raises
# if it is ever touched.

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from waqil_api.config import Settings  # noqa: E402
from waqil_api.contracts import (  # noqa: E402
    AcceptanceScenarioV1,
    ProjectBuildPlanV1,
    ProjectVerticalSliceV1,
)
from waqil_api.control_plane import ControlPlane  # noqa: E402


class ExplodingCoder:
    """Any coder call at all is the defect these tests exist to catch."""

    def __init__(self) -> None:
        self.calls = 0

    async def start(self, **kwargs: Any) -> Any:
        self.calls += 1
        raise AssertionError("a coder session was opened behind a rejected plan")

    async def continue_session(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        raise AssertionError("a coder session was continued behind a rejected plan")


class Planner:
    """Returns a scripted plan per call, and counts the calls."""

    def __init__(self, plans: list[ProjectBuildPlanV1]) -> None:
        self.plans = list(plans)
        self.requests: list[dict[str, Any]] = []
        self.last_usage = {
            "prompt_tokens": 900,
            "completion_tokens": 100,
            "total_tokens": 1000,
        }

    async def project_plan_files(
        self, request: dict[str, Any], *, model_aliases: dict[str, str] | None = None
    ) -> ProjectBuildPlanV1:
        del model_aliases
        self.requests.append(request)
        return self.plans.pop(0)


class Events:
    def __init__(self) -> None:
        self.items: list[tuple[str, dict[str, Any]]] = []

    async def emit(
        self, run_id: str, conversation_id: str, kind: str, payload: dict
    ) -> None:
        del run_id, conversation_id
        self.items.append((kind, payload))


def _plan(
    slices: list[dict[str, Any]], files: list[str] | None = None
) -> ProjectBuildPlanV1:
    return ProjectBuildPlanV1(
        files=files or ATLAS_FILES,
        intent="edit",
        scope="narrow",
        scenarios=[
            AcceptanceScenarioV1(name=item["name"], path="/api/projects")
            for item in SCENARIOS
        ],
        slices=[ProjectVerticalSliceV1.model_validate(item) for item in slices],
    )


def _plane(
    tmp_path: Path, planner: Planner, coder: Any, **overrides: Any
) -> ControlPlane:
    plane = object.__new__(ControlPlane)
    plane.settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        repo_root=tmp_path,
        asset_roots=[tmp_path / "Projects"],
        model_backend="deterministic",
        allow_test_backends=True,
        project_coding_engine="clinecore",
        **overrides,
    )
    plane.model = planner
    plane.project_coding = coder
    plane.events = Events()
    return plane


def _state() -> dict[str, Any]:
    return {
        "run_id": "run_x",
        "conversation_id": "conv_x",
        "prompt": "Revamp the Atlas Projects web UI without changing its API.",
        "model_aliases": {
            "_project_id": "asset_x",
            "_coding_engine": "clinecore",
            "_provider": "local",
            "_chain_planner": json.dumps(
                [{"provider": "local", "model": "glm-5.2:cloud"}]
            ),
        },
        "project_iterations": 8,
        "project_staged": {},
        "project_trace": [],
        "project_planner_chain_index": 0,
    }


@pytest.mark.asyncio
async def test_a_rejected_plan_opens_no_coding_session_and_spends_no_coder_tokens(
    tmp_path: Path,
) -> None:
    """Fail-fast mode: the four-slice Atlas plan stops before inference."""

    planner = Planner([_plan(ATLAS_SUPPORT_FIRST_PLAN, ATLAS_SUPPORT_FIRST_FILES)])
    coder = ExplodingCoder()
    plane = _plane(tmp_path, planner, coder, project_plan_corrections=0)

    plan = await ControlPlane._project_manifest(
        plane, _state(), prompt_context={}, iterations=8, staged={}
    )

    assert coder.calls == 0
    assert plan["taken"] is False
    assert plan["plan_rejected"] is True
    assert "no earlier slice delivers an outcome" in plan["plan_error"]
    kinds = [kind for kind, _ in plane.events.items]
    assert "project.plan_rejected" in kinds
    # Never announced as a usable plan.
    assert "project.build_planned" not in kinds
    rejected = next(p for k, p in plane.events.items if k == "project.plan_rejected")
    assert "support_slice_before_runtime" in rejected["codes"]
    assert rejected["corrections_left"] == 0
    # Exactly one planner call: fail-fast must not hide the planner defect
    # behind a correction.
    assert len(planner.requests) == 1
    assert "plan_correction" not in planner.requests[0]


@pytest.mark.asyncio
async def test_a_rejected_plan_stops_the_turn_honestly_before_any_coding(
    tmp_path: Path,
) -> None:
    planner = Planner([_plan(ATLAS_SUPPORT_FIRST_PLAN, ATLAS_SUPPORT_FIRST_FILES)])
    coder = ExplodingCoder()
    plane = _plane(tmp_path, planner, coder, project_plan_corrections=0)
    state = _state()

    plan = await ControlPlane._project_manifest(
        plane, state, prompt_context={}, iterations=8, staged={}
    )
    response = await ControlPlane._project_plan_failure_response(plane, state, plan)

    assert coder.calls == 0
    assert response is not None
    assert response["project_pending_call"] == {}
    failure = next(p for k, p in plane.events.items if k == "project.plan_failed")
    assert failure["reason"] == "invalid_slice_plan"


@pytest.mark.asyncio
async def test_one_bounded_correction_is_offered_then_the_turn_stops(
    tmp_path: Path,
) -> None:
    """Production default: one corrective replan, carrying exact findings."""

    planner = Planner(
        [
            _plan(ATLAS_SUPPORT_FIRST_PLAN, ATLAS_SUPPORT_FIRST_FILES),
            _plan(ATLAS_SUPPORT_FIRST_PLAN, ATLAS_SUPPORT_FIRST_FILES),
        ]
    )
    coder = ExplodingCoder()
    plane = _plane(tmp_path, planner, coder, project_plan_corrections=1)

    plan = await ControlPlane._project_manifest(
        plane, _state(), prompt_context={}, iterations=8, staged={}
    )

    assert coder.calls == 0
    assert plan["plan_rejected"] is True
    # Two planner calls, the second carrying the findings verbatim.
    assert len(planner.requests) == 2
    correction = planner.requests[1]["plan_correction"]
    assert "rejected before any code was written" in correction
    assert "support files" in correction
    rejections = [p for k, p in plane.events.items if k == "project.plan_rejected"]
    assert [item["attempt"] for item in rejections] == [1, 2]
    assert [item["corrections_left"] for item in rejections] == [1, 0]


@pytest.mark.asyncio
async def test_a_corrected_plan_is_accepted_and_the_build_proceeds(
    tmp_path: Path,
) -> None:
    planner = Planner(
        [
            _plan(ATLAS_SUPPORT_FIRST_PLAN, ATLAS_SUPPORT_FIRST_FILES),
            _plan(ATLAS_TWO_SLICE_PLAN),
        ]
    )
    plane = _plane(tmp_path, planner, ExplodingCoder(), project_plan_corrections=1)

    plan = await ControlPlane._project_manifest(
        plane, _state(), prompt_context={}, iterations=8, staged={}
    )

    assert plan["taken"] is True
    assert plan.get("plan_rejected") is None
    assert plan["files"] == ATLAS_FILES
    assert [item["name"] for item in plan["slices"]] == [
        "Console shell renders live project data",
        "Status workflow, its contract tests and docs",
    ]
    kinds = [kind for kind, _ in plane.events.items]
    assert kinds.count("project.plan_rejected") == 1
    assert "project.build_planned" in kinds


@pytest.mark.asyncio
async def test_a_valid_plan_is_never_delayed_by_the_gate(tmp_path: Path) -> None:
    planner = Planner([_plan(ATLAS_TWO_SLICE_PLAN)])
    plane = _plane(tmp_path, planner, ExplodingCoder())

    plan = await ControlPlane._project_manifest(
        plane, _state(), prompt_context={}, iterations=8, staged={}
    )

    assert plan["taken"] is True
    assert len(planner.requests) == 1
    assert not any(k == "project.plan_rejected" for k, _ in plane.events.items)


@pytest.mark.asyncio
async def test_planner_usage_is_recorded_separately_from_coder_usage(
    tmp_path: Path,
) -> None:
    planner = Planner([_plan(ATLAS_TWO_SLICE_PLAN)])
    plane = _plane(tmp_path, planner, ExplodingCoder())
    state = _state()

    await ControlPlane._project_manifest(
        plane, state, prompt_context={}, iterations=8, staged={}
    )

    [usage] = [p for k, p in plane.events.items if k == "run.planner_attempt"]
    assert usage["role"] == "planner"
    assert usage["model"] == "glm-5.2:cloud"
    assert usage["operation"] == "project_plan_files"
    assert usage["ok"] is True
    assert usage["usage"]["total_tokens"] == 1000
    assert usage["usage_available"] is True
    assert usage["fallback"] is False
    assert usage["latency_ms"] >= 0
    # Counted into the turn's own planner total, so a run-level ceiling can
    # include planner spend instead of meaning "coder only".
    assert state["project_planner_tokens"] == 1000


@pytest.mark.asyncio
async def test_a_provider_without_usage_reports_zero_rather_than_pretending(
    tmp_path: Path,
) -> None:
    planner = Planner([_plan(ATLAS_TWO_SLICE_PLAN)])
    planner.last_usage = {}
    plane = _plane(tmp_path, planner, ExplodingCoder())
    state = _state()

    await ControlPlane._project_manifest(
        plane, state, prompt_context={}, iterations=8, staged={}
    )

    [usage] = [p for k, p in plane.events.items if k == "run.planner_attempt"]
    assert usage["usage"] == {}
    assert usage["usage_available"] is False
    assert state["project_planner_tokens"] == 0


# ── The effective plan, not just the declared one ──────────────────────────
# A planner that declares no slices used to skip topology review entirely:
# the host then partitions the manifest into six-file chunks, and a manifest
# whose tail is a README puts that README in a chunk of its own.


def test_an_unsliced_manifest_becomes_one_slice_instead_of_a_docs_only_chunk() -> None:
    from waqil_api.project_plan_validation import validate_effective_plan
    from waqil_api.project_slices import synthesize_single_slice

    manifest = [f"app/module_{index}.py" for index in range(6)] + ["README.md"]

    # The old mechanical partition really did strand the README in a chunk of
    # its own -- a support-only slice Metis wrote, not a model. Synthesis
    # replaces it with one slice that owns the manifest exactly.
    synthesized = synthesize_single_slice(manifest, [])
    assert synthesized is not None
    assert synthesized["owned_files"] == manifest
    assert synthesized["integration_files"] == []

    assert validate_build_plan(manifest, []).ok is True
    assert validate_effective_plan(manifest, []).ok is True


def test_synthesis_carries_every_declared_scenario_onto_the_slice() -> None:
    from waqil_api.project_slices import synthesize_single_slice

    manifest = ["app/main.py", "app/api.py", "README.md"]
    declared = [
        {"name": "health responds"},
        {"name": "upload extracts the invoice"},
        {"name": "the workbook downloads"},
    ]

    synthesized = synthesize_single_slice(manifest, declared)

    assert synthesized is not None
    assert synthesized["scenario_names"] == [
        "health responds",
        "upload extracts the invoice",
        "the workbook downloads",
    ]


def test_a_manifest_larger_than_the_bound_is_never_synthesized() -> None:
    from waqil_api.project_slices import synthesize_single_slice

    # Nine files: where to draw the boundary is a design decision, and the
    # host declines to make one up.
    manifest = [f"app/module_{index}.py" for index in range(MAX_SLICE_FILES + 1)]

    assert synthesize_single_slice(manifest, []) is None


def test_a_valid_deterministic_partition_proceeds_untouched() -> None:
    from waqil_api.project_plan_validation import validate_effective_plan

    # Every chunk carries runtime files, so the automatic partition is fine.
    manifest = [f"app/module_{index}.py" for index in range(8)]

    assert validate_effective_plan(manifest, []).ok is True
    # And a small manifest that fits one chunk, support files included.
    assert (
        validate_effective_plan(
            ["app/main.py", "tests/test_main.py", "README.md"], []
        ).ok
        is True
    )


def test_declared_slices_are_still_judged_before_the_partition() -> None:
    from waqil_api.project_plan_validation import validate_effective_plan

    # The Atlas plan is an exact partition, so the effective plan IS the
    # declared plan -- and the findings must name the planner's slices rather
    # than blaming an automatic partition it never used.
    result = validate_effective_plan(
        ATLAS_FILES, ATLAS_FOUR_SLICE_PLAN, scenarios=SCENARIOS
    )

    assert result.ok is False
    assert "automatic six-file partition" not in " ".join(result.findings)
    assert "UI contract tests" in " ".join(result.findings)


@pytest.mark.asyncio
async def test_an_unsliced_logivity_sized_manifest_is_synthesized_not_partitioned(
    tmp_path: Path,
) -> None:
    """The exact shape that produced a README-and-requirements support slice."""

    manifest = [f"app/module_{index}.py" for index in range(6)] + ["README.md"]
    plan = ProjectBuildPlanV1(
        files=manifest, intent="build", scope="narrow", scenarios=[], slices=[]
    )
    planner = Planner([plan])
    coder = ExplodingCoder()
    plane = _plane(tmp_path, planner, coder, project_plan_corrections=0)

    result = await ControlPlane._project_manifest(
        plane, _state(), prompt_context={}, iterations=8, staged={}
    )

    assert coder.calls == 0
    assert result["taken"] is True
    assert result["slices_synthesized"] is True
    assert [item["owned_files"] for item in result["slices"]] == [manifest]
    kinds = [kind for kind, _ in plane.events.items]
    assert "project.plan_synthesized" in kinds
    assert "project.plan_rejected" not in kinds
    evidence = next(p for k, p in plane.events.items if k == "project.plan_synthesized")
    assert evidence["reason"] == "planner_declared_no_slices"
    assert evidence["slice"]["owned_files"] == manifest
    assert evidence["slice"]["integration_files"] == []
    planned = next(p for k, p in plane.events.items if k == "project.build_planned")
    assert planned["slices_synthesized"] is True


@pytest.mark.asyncio
async def test_an_unsliced_manifest_past_the_bound_is_rejected_not_partitioned(
    tmp_path: Path,
) -> None:
    # Nine files and no slices, with the corrective attempt already spent:
    # the honest answer is a refusal, not a mechanical split.
    manifest = [f"app/module_{index}.py" for index in range(MAX_SLICE_FILES + 1)]
    plan = ProjectBuildPlanV1(
        files=manifest, intent="build", scope="narrow", scenarios=[], slices=[]
    )
    plane = _plane(
        tmp_path, Planner([plan]), ExplodingCoder(), project_plan_corrections=0
    )

    result = await ControlPlane._project_manifest(
        plane, _state(), prompt_context={}, iterations=8, staged={}
    )

    assert result["plan_rejected"] is True
    assert result["taken"] is False
    assert "declared no slices" in result["plan_error"]
    assert "inventing an architecture" in result["plan_error"]
    kinds = [kind for kind, _ in plane.events.items]
    assert "project.plan_synthesized" not in kinds
    assert "project.build_planned" not in kinds


@pytest.mark.asyncio
async def test_an_unsliced_manifest_past_the_bound_asks_the_planner_once(
    tmp_path: Path,
) -> None:
    # With a correction available, the planner is asked to make the split
    # itself -- and a second reply that declares slices is accepted as its
    # own work, not recorded as synthesized.
    manifest = [f"app/module_{index}.py" for index in range(MAX_SLICE_FILES + 1)]
    unsliced = ProjectBuildPlanV1(
        files=manifest, intent="build", scope="narrow", scenarios=[], slices=[]
    )
    sliced = ProjectBuildPlanV1(
        files=manifest,
        intent="build",
        scope="narrow",
        scenarios=[],
        slices=[
            {
                "name": "Core",
                "outcome": "The first modules work end to end.",
                "files": manifest[:5],
                "owned_files": manifest[:5],
                "integration_files": [],
            },
            {
                "name": "Rest",
                "outcome": "The remaining modules work end to end.",
                "files": manifest[5:],
                "owned_files": manifest[5:],
                "integration_files": [],
            },
        ],
    )
    planner = Planner([unsliced, sliced])
    plane = _plane(tmp_path, planner, ExplodingCoder(), project_plan_corrections=1)

    result = await ControlPlane._project_manifest(
        plane, _state(), prompt_context={}, iterations=8, staged={}
    )

    assert result["taken"] is True
    assert result["slices_synthesized"] is False
    assert [item["name"] for item in result["slices"]] == ["Core", "Rest"]
    assert "project.plan_synthesized" not in [k for k, _ in plane.events.items]


@pytest.mark.asyncio
async def test_empty_planner_slices_with_a_valid_partition_proceed(
    tmp_path: Path,
) -> None:
    manifest = [f"app/module_{index}.py" for index in range(8)]
    plan = ProjectBuildPlanV1(
        files=manifest, intent="build", scope="narrow", scenarios=[], slices=[]
    )
    plane = _plane(tmp_path, Planner([plan]), ExplodingCoder())

    result = await ControlPlane._project_manifest(
        plane, _state(), prompt_context={}, iterations=8, staged={}
    )

    assert result["taken"] is True
    assert result["files"] == manifest
    assert not any(k == "project.plan_rejected" for k, _ in plane.events.items)


# ── Deterministic support-slice normalization ──────────────────────────────
# Two live planner attempts produced the same predictable shape: correct
# runtime slices, then the tests and README in a slice of their own. That is
# arithmetic the host can settle exactly, so it does — visibly, and only when
# it is safe.

from waqil_api.project_plan_validation import (  # noqa: E402
    normalize_build_plan,
    validate_effective_plan,
)


# The exact three-slice plan the remediation canary produced.
ATLAS_THREE_SLICE_PLAN: list[dict[str, Any]] = [
    {
        "name": "Console HTML shell and design tokens",
        "outcome": "The page is a semantic Atlas Operations Console.",
        "files": ["app/static/index.html", "app/static/styles.css"],
        "owned_files": ["app/static/index.html", "app/static/styles.css"],
        "integration_files": [],
        "scenario_names": ["ui_page_has_console_landmarks"],
    },
    {
        "name": "Client application logic and API integration",
        "outcome": "app.js fetches the API and renders the table.",
        "files": ["app/static/app.js"],
        "owned_files": ["app/static/app.js"],
        "integration_files": [],
        "scenario_names": ["projects_exact_records"],
    },
    {
        "name": "UI contract tests and README documentation",
        "outcome": "Tests and docs cover the console.",
        "files": ["tests/test_ui_contract.py", "README.md"],
        "owned_files": ["tests/test_ui_contract.py", "README.md"],
        "integration_files": [],
        "scenario_names": [],
    },
]


def test_the_captured_three_slice_plan_normalizes_into_two_valid_slices() -> None:
    result = normalize_build_plan(ATLAS_FILES, ATLAS_THREE_SLICE_PLAN)

    assert result.ok and result.applied
    assert result.codes == ("support_slice_merged",)
    assert [item["name"] for item in result.slices] == [
        "Console HTML shell and design tokens",
        "Client application logic and API integration",
    ]
    assert result.slices[1]["owned_files"] == [
        "app/static/app.js",
        "tests/test_ui_contract.py",
        "README.md",
    ]
    assert result.moved[0]["paths"] == ["tests/test_ui_contract.py", "README.md"]
    assert (
        result.moved[0]["into_slice"] == "Client application logic and API integration"
    )
    assert sorted(result.moved[0]["kinds"]) == ["documentation", "test"]
    # And the normalized plan is what the gate then accepts.
    assert validate_effective_plan(ATLAS_FILES, list(result.slices)).ok is True


def test_the_earlier_separate_tests_and_docs_slices_also_normalize() -> None:
    result = normalize_build_plan(ATLAS_FILES, ATLAS_FOUR_SLICE_PLAN)

    assert result.ok and result.applied
    assert len(result.slices) == 2
    # Both support slices fold into the nearest preceding outcome.
    assert [item["paths"] for item in result.moved] == [
        ["tests/test_ui_contract.py"],
        ["README.md"],
    ]
    assert validate_effective_plan(ATLAS_FILES, list(result.slices)).ok is True


def test_every_planned_path_survives_normalization_exactly_once() -> None:
    for plan in (ATLAS_THREE_SLICE_PLAN, ATLAS_FOUR_SLICE_PLAN):
        result = normalize_build_plan(ATLAS_FILES, plan)
        owned = [path for item in result.slices for path in item["owned_files"]]
        # Exactly once, in the planner's own dependency order, nothing added.
        assert owned == ATLAS_FILES
        assert sorted(owned) == sorted(ATLAS_FILES)


def test_normalization_preserves_scenarios_and_integration_declarations() -> None:
    plan = [
        {
            "name": "Listing",
            "outcome": "The listing works.",
            "files": ["app/db.py", "app/main.py"],
            "owned_files": ["app/db.py", "app/main.py"],
            "integration_files": [],
            "scenario_names": ["listing"],
        },
        {
            "name": "Status action",
            "outcome": "The status action works.",
            "files": ["app/status.py", "app/main.py"],
            "owned_files": ["app/status.py"],
            "integration_files": ["app/main.py"],
            "scenario_names": ["status"],
        },
        {
            "name": "Tests",
            "outcome": "Tests cover both.",
            "files": ["tests/test_all.py"],
            "owned_files": ["tests/test_all.py"],
            "integration_files": [],
            "scenario_names": ["regression"],
        },
    ]
    files = ["app/db.py", "app/main.py", "app/status.py", "tests/test_all.py"]

    result = normalize_build_plan(files, plan)

    assert result.ok and result.applied
    merged = result.slices[1]
    # The integration declaration survives the merge, and the file it names is
    # NOT swallowed into ownership.
    assert merged["integration_files"] == ["app/main.py"]
    assert merged["owned_files"] == ["app/status.py", "tests/test_all.py"]
    # Both slices' scenarios are kept; none is dropped to make room.
    assert merged["scenario_names"] == ["status", "regression"]


def test_normalization_is_pure_and_needs_no_model_or_coder(monkeypatch) -> None:
    """No inference of any kind: this is arithmetic over the planner's own plan."""

    import waqil_api.project_plan_validation as module

    def explode(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("normalization must not call out to anything")

    # Nothing in the module may reach a network, a model, or a subprocess.
    monkeypatch.setattr(module, "vertical_build_slices", explode, raising=False)
    result = normalize_build_plan(ATLAS_FILES, ATLAS_THREE_SLICE_PLAN)

    assert result.applied is True
    # The input plan is never mutated in place; callers keep their original.
    assert ATLAS_THREE_SLICE_PLAN[2]["owned_files"] == [
        "tests/test_ui_contract.py",
        "README.md",
    ]
    assert len(ATLAS_THREE_SLICE_PLAN) == 3


def test_a_support_slice_before_any_outcome_is_refused_not_merged_forward() -> None:
    result = normalize_build_plan(ATLAS_SUPPORT_FIRST_FILES, ATLAS_SUPPORT_FIRST_PLAN)

    assert result.ok is False
    assert result.applied is False
    assert result.codes == ("support_slice_before_runtime",)
    assert "no earlier slice delivers an outcome" in result.rejected


def test_a_merge_that_would_overflow_the_slice_limit_is_refused() -> None:
    files = [f"app/module_{index}.py" for index in range(8)] + ["README.md"]
    plan = [
        {
            "name": "Everything",
            "outcome": "It all works.",
            "files": files[:8],
            "owned_files": files[:8],
            "integration_files": [],
        },
        {
            "name": "Docs",
            "outcome": "Docs explain it.",
            "files": ["README.md"],
            "owned_files": ["README.md"],
            "integration_files": [],
        },
    ]

    result = normalize_build_plan(files, plan)

    assert result.ok is False
    assert result.codes == ("merge_exceeds_slice_limit",)
    assert f"past the {MAX_SLICE_FILES}-file slice limit" in result.rejected


def test_a_merge_that_now_fits_the_wider_slice_bound_is_applied() -> None:
    # The same shape one file smaller: six modules plus a docs slice merges
    # cleanly into a seven-file slice, which the sidecar's write scope can
    # now carry.
    files = [f"app/module_{index}.py" for index in range(6)] + ["README.md"]
    plan = [
        {
            "name": "Everything",
            "outcome": "It all works.",
            "files": files[:6],
            "owned_files": files[:6],
            "integration_files": [],
        },
        {
            "name": "Docs",
            "outcome": "Docs explain it.",
            "files": ["README.md"],
            "owned_files": ["README.md"],
            "integration_files": [],
        },
    ]

    result = normalize_build_plan(files, plan)

    assert result.ok is True
    assert result.applied is True
    assert [item["owned_files"] for item in result.slices] == [files]


def test_a_merge_past_the_plan_scenario_limit_is_still_refused() -> None:
    files = ["app/a.py", "tests/test_a.py"]
    plan = [
        {
            "name": "Feature",
            "outcome": "It works.",
            "files": ["app/a.py"],
            "owned_files": ["app/a.py"],
            "integration_files": [],
            "scenario_names": [
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
            ],
        },
        {
            "name": "Tests",
            "outcome": "Tests cover it.",
            "files": ["tests/test_a.py"],
            "owned_files": ["tests/test_a.py"],
            "integration_files": [],
            "scenario_names": ["nine"],
        },
    ]

    result = normalize_build_plan(files, plan)

    # Nine distinct names is past the PLAN's own eight-scenario limit, so the
    # merge is refused rather than quietly dropping one. Six was refused
    # before this fix and must not be any more -- see the Atlas test above.
    assert result.ok is False
    assert result.codes == ("merge_exceeds_scenario_limit",)
    assert "discard a planner commitment" in result.rejected


def test_normalization_cannot_introduce_a_protected_or_unplanned_path() -> None:
    protected = {
        "app/main.py",
        "app/api.py",
        "app/store.py",
        "tests/test_api_contract.py",
    }

    for plan in (ATLAS_THREE_SLICE_PLAN, ATLAS_FOUR_SLICE_PLAN):
        result = normalize_build_plan(ATLAS_FILES, plan)
        paths = {
            path
            for item in result.slices
            for path in item["owned_files"] + item["integration_files"]
        }
        # Only ever a subset of what the planner already committed to, so a
        # protected file cannot appear by construction.
        assert paths <= set(ATLAS_FILES)
        assert not paths & protected


def test_an_already_valid_plan_is_returned_untouched() -> None:
    result = normalize_build_plan(ATLAS_FILES, ATLAS_TWO_SLICE_PLAN)

    assert result.applied is False
    assert result.codes == ()
    assert result.moved == ()
    assert [dict(item) for item in result.slices] == ATLAS_TWO_SLICE_PLAN


def test_a_documentation_only_request_is_never_normalized() -> None:
    files = ["README.md", "docs/setup.md"]
    plan = [
        {
            "name": "Docs",
            "outcome": "The docs are current.",
            "files": files,
            "owned_files": files,
            "integration_files": [],
        }
    ]

    result = normalize_build_plan(files, plan)

    assert result.ok is True
    assert result.applied is False


@pytest.mark.asyncio
async def test_the_gate_normalizes_and_records_it_without_a_coder_call(
    tmp_path: Path,
) -> None:
    planner = Planner([_plan(ATLAS_THREE_SLICE_PLAN)])
    coder = ExplodingCoder()
    plane = _plane(tmp_path, planner, coder, project_plan_corrections=0)

    plan = await ControlPlane._project_manifest(
        plane, _state(), prompt_context={}, iterations=8, staged={}
    )

    assert coder.calls == 0
    # One planner call: normalization replaces a corrective replan, it does
    # not add one.
    assert len(planner.requests) == 1
    assert plan["taken"] is True
    assert [item["name"] for item in plan["slices"]] == [
        "Console HTML shell and design tokens",
        "Client application logic and API integration",
    ]

    kinds = [kind for kind, _ in plane.events.items]
    assert "project.plan_normalized" in kinds
    assert "project.plan_rejected" not in kinds
    assert "project.build_planned" in kinds

    evidence = next(p for k, p in plane.events.items if k == "project.plan_normalized")
    # The original topology is preserved for auditing, beside the new one.
    assert [item["name"] for item in evidence["original"]] == [
        "Console HTML shell and design tokens",
        "Client application logic and API integration",
        "UI contract tests and README documentation",
    ]
    assert len(evidence["normalized"]) == 2
    assert evidence["codes"] == ["support_slice_merged"]
    assert evidence["moved"][0]["paths"] == [
        "tests/test_ui_contract.py",
        "README.md",
    ]


@pytest.mark.asyncio
async def test_a_valid_plan_emits_no_normalization_evidence(tmp_path: Path) -> None:
    planner = Planner([_plan(ATLAS_TWO_SLICE_PLAN)])
    plane = _plane(tmp_path, planner, ExplodingCoder())

    plan = await ControlPlane._project_manifest(
        plane, _state(), prompt_context={}, iterations=8, staged={}
    )

    assert plan["taken"] is True
    assert not any(k == "project.plan_normalized" for k, _ in plane.events.items)


def test_the_normalized_owner_is_what_repair_routing_and_verification_target() -> None:
    """Downstream consumers see the merged slice, not the planner's original.

    A defect in the moved test file must route to the slice that now owns it,
    and that slice's cumulative scenarios must include the ones it absorbed.
    """

    from waqil_api.project_repair_routing import route_acceptance_finding

    result = normalize_build_plan(ATLAS_FILES, ATLAS_THREE_SLICE_PLAN)
    slices = [dict(item) for item in result.slices]
    staged = {path: {"content": f"# {path}\n"} for path in ATLAS_FILES}

    route = route_acceptance_finding(
        findings=[
            {
                "path": "tests/test_ui_contract.py",
                "error": "assertion failed",
                "severity": "error",
                "kind": "test",
            }
        ],
        slices=slices,
        staged=staged,
    )

    assert route.routed is True
    assert route.slice_name == "Client application logic and API integration"
    assert route.attribution == "owned_file"
    # The repair may write the merged slice's whole scope, README included.
    assert set(route.authorized_paths) == {
        "app/static/app.js",
        "tests/test_ui_contract.py",
        "README.md",
    }
    # Cumulative verification still replays everything proven so far.
    assert route.rerun_scenarios == (
        "ui_page_has_console_landmarks",
        "projects_exact_records",
    )


# ── Trailing-only normalization ────────────────────────────────────────────
# Canonicalization is confined to support work that TRAILS the last outcome,
# where its owner is unambiguous. Anywhere else, and for any dependency file,
# the plan is refused rather than guessed at.


def test_a_support_slice_between_two_runtime_slices_is_refused_as_ambiguous() -> None:
    files = ["app/a.py", "tests/test_a.py", "app/b.py"]
    plan = [
        {
            "name": "Feature A",
            "outcome": "A works.",
            "files": ["app/a.py"],
            "owned_files": ["app/a.py"],
            "integration_files": [],
        },
        {
            "name": "Tests",
            "outcome": "Tests cover something.",
            "files": ["tests/test_a.py"],
            "owned_files": ["tests/test_a.py"],
            "integration_files": [],
        },
        {
            "name": "Feature B",
            "outcome": "B works.",
            "files": ["app/b.py"],
            "owned_files": ["app/b.py"],
            "integration_files": [],
        },
    ]

    result = normalize_build_plan(files, plan)

    assert result.ok is False
    assert result.applied is False
    assert result.codes == ("support_slice_between_runtime",)
    assert "ambiguous" in result.rejected
    # It could belong to either neighbour; the host refuses to choose.
    assert "place it in that slice yourself" in result.rejected


def test_a_trailing_dependency_slice_is_refused_never_moved() -> None:
    files = ["app/a.py", "requirements.txt"]
    plan = [
        {
            "name": "Feature",
            "outcome": "It works.",
            "files": ["app/a.py"],
            "owned_files": ["app/a.py"],
            "integration_files": [],
        },
        {
            "name": "Dependencies",
            "outcome": "Dependencies are declared.",
            "files": ["requirements.txt"],
            "owned_files": ["requirements.txt"],
            "integration_files": [],
        },
    ]

    result = normalize_build_plan(files, plan)

    assert result.ok is False
    assert result.codes == ("dependency_slice_not_movable",)
    assert "never moved automatically" in result.rejected


def test_a_trailing_slice_mixing_docs_with_a_dependency_is_refused() -> None:
    """Dependency contamination refuses the whole merge, not just that file."""

    files = ["app/a.py", "README.md", "requirements.txt"]
    plan = [
        {
            "name": "Feature",
            "outcome": "It works.",
            "files": ["app/a.py"],
            "owned_files": ["app/a.py"],
            "integration_files": [],
        },
        {
            "name": "Docs and deps",
            "outcome": "Docs and dependencies.",
            "files": ["README.md", "requirements.txt"],
            "owned_files": ["README.md", "requirements.txt"],
            "integration_files": [],
        },
    ]

    result = normalize_build_plan(files, plan)

    assert result.ok is False
    assert result.codes == ("dependency_slice_not_movable",)


def test_a_dependency_only_request_keeps_its_exemption() -> None:
    files = ["requirements.txt", "pyproject.toml"]
    plan = [
        {
            "name": "Dependencies",
            "outcome": "Pin the dependency set.",
            "files": files,
            "owned_files": files,
            "integration_files": [],
        }
    ]

    result = normalize_build_plan(files, plan)

    assert result.ok is True
    assert result.applied is False


def test_multiple_trailing_support_slices_merge_in_manifest_order() -> None:
    files = [
        "app/a.py",
        "app/b.py",
        "tests/test_a.py",
        "docs/guide.md",
        "README.md",
    ]
    plan = [
        {
            "name": "Feature A",
            "outcome": "A works.",
            "files": ["app/a.py"],
            "owned_files": ["app/a.py"],
            "integration_files": [],
            "scenario_names": ["a"],
        },
        {
            "name": "Feature B",
            "outcome": "B works.",
            "files": ["app/b.py"],
            "owned_files": ["app/b.py"],
            "integration_files": [],
            "scenario_names": ["b"],
        },
        {
            "name": "Tests",
            "outcome": "Tests cover B.",
            "files": ["tests/test_a.py"],
            "owned_files": ["tests/test_a.py"],
            "integration_files": [],
            "scenario_names": ["regression"],
        },
        {
            "name": "Guide",
            "outcome": "The guide explains B.",
            "files": ["docs/guide.md"],
            "owned_files": ["docs/guide.md"],
            "integration_files": [],
            "scenario_names": [],
        },
        {
            "name": "Readme",
            "outcome": "The README explains B.",
            "files": ["README.md"],
            "owned_files": ["README.md"],
            "integration_files": [],
            "scenario_names": [],
        },
    ]

    result = normalize_build_plan(files, plan)

    assert result.ok and result.applied
    assert [item["name"] for item in result.slices] == ["Feature A", "Feature B"]
    # All three trailing slices fold into the LAST outcome, in manifest order.
    assert result.slices[1]["owned_files"] == [
        "app/b.py",
        "tests/test_a.py",
        "docs/guide.md",
        "README.md",
    ]
    assert [item["paths"] for item in result.moved] == [
        ["tests/test_a.py"],
        ["docs/guide.md"],
        ["README.md"],
    ]
    # Feature A is untouched -- trailing work never reaches back past the last
    # outcome.
    assert result.slices[0]["owned_files"] == ["app/a.py"]
    assert result.slices[0]["scenario_names"] == ["a"]


def test_trailing_normalization_changes_no_path_scenario_or_authority() -> None:
    """The invariants, checked together on the captured Atlas plan."""

    protected = {
        "app/main.py",
        "app/api.py",
        "app/store.py",
        "tests/test_api_contract.py",
    }
    before_paths = sorted(
        path for item in ATLAS_THREE_SLICE_PLAN for path in item["files"]
    )
    before_scenarios = sorted(
        name for item in ATLAS_THREE_SLICE_PLAN for name in item["scenario_names"]
    )
    before_integration = sorted(
        path for item in ATLAS_THREE_SLICE_PLAN for path in item["integration_files"]
    )

    result = normalize_build_plan(ATLAS_FILES, ATLAS_THREE_SLICE_PLAN)

    after_paths = sorted(path for item in result.slices for path in item["files"])
    after_scenarios = sorted(
        name for item in result.slices for name in item["scenario_names"]
    )
    after_integration = sorted(
        path for item in result.slices for path in item["integration_files"]
    )

    assert after_paths == before_paths
    assert after_scenarios == before_scenarios
    assert after_integration == before_integration
    # Ownership is a permutation of the manifest, exactly once each.
    owned = [path for item in result.slices for path in item["owned_files"]]
    assert owned == ATLAS_FILES
    # No protected path can be reached, because none was ever in the plan.
    assert not set(after_paths) & protected


def test_repair_routing_after_trailing_normalization_targets_the_merged_owner() -> None:
    from waqil_api.project_repair_routing import route_acceptance_finding

    result = normalize_build_plan(ATLAS_FILES, ATLAS_FOUR_SLICE_PLAN)
    slices = [dict(item) for item in result.slices]
    staged = {path: {"content": f"# {path}\n"} for path in ATLAS_FILES}

    for target, expected in (
        ("README.md", "Client application logic"),
        ("tests/test_ui_contract.py", "Client application logic"),
        ("app/static/index.html", "Semantic HTML shell and design tokens"),
    ):
        route = route_acceptance_finding(
            findings=[
                {
                    "path": target,
                    "error": "failed",
                    "severity": "error",
                    "kind": "acceptance",
                }
            ],
            slices=slices,
            staged=staged,
        )
        assert route.routed is True
        assert route.slice_name == expected, target
        assert target in route.authorized_paths


# ── One shared scenario bound ──────────────────────────────────────────────
# A separate, smaller five-item per-slice cap silently dropped planner
# commitments and refused a correct six-scenario Atlas plan. There is now one
# number: the plan's own limit.


def test_there_is_exactly_one_scenario_bound_and_it_is_the_plan_limit() -> None:
    from waqil_api.contracts import MAX_PLAN_SCENARIOS
    from waqil_api.project_plan_validation import MAX_NORMALIZED_SCENARIOS
    from waqil_api.project_slices import MAX_PLAN_SCENARIOS as slices_bound

    assert MAX_PLAN_SCENARIOS == 8
    assert MAX_NORMALIZED_SCENARIOS == MAX_PLAN_SCENARIOS
    assert slices_bound == MAX_PLAN_SCENARIOS


def test_six_and_eight_declared_scenarios_survive_parsing_and_materialization() -> None:
    from waqil_api.contracts import ProjectBuildPlanV1
    from waqil_api.project_slices import next_vertical_slice, vertical_build_slices

    for count in (6, 8):
        names = [f"scenario_{index}" for index in range(count)]
        plan = ProjectBuildPlanV1(
            files=["app/a.py", "app/b.py"],
            intent="edit",
            scope="narrow",
            scenarios=[
                AcceptanceScenarioV1(name=name, path="/api/x") for name in names
            ],
            slices=[
                ProjectVerticalSliceV1(
                    name="One outcome",
                    outcome="It works.",
                    files=["app/a.py", "app/b.py"],
                    owned_files=["app/a.py", "app/b.py"],
                    integration_files=[],
                    scenario_names=names,
                )
            ],
        )
        # Parsing keeps every declared name.
        assert plan.slices[0].scenario_names == names, count

        declared = [item.model_dump(mode="json") for item in plan.slices]
        materialized = vertical_build_slices(plan.files, declared)
        assert materialized[0]["scenario_names"] == names, count
        active = next_vertical_slice(plan.files, declared, verified_count=0)
        assert active is not None and active["scenario_names"] == names, count


def test_unknown_scenario_names_are_removed_without_losing_valid_ones() -> None:
    from waqil_api.contracts import ProjectBuildPlanV1

    plan = ProjectBuildPlanV1(
        files=["app/a.py"],
        intent="edit",
        scope="narrow",
        scenarios=[
            AcceptanceScenarioV1(name="declared_one", path="/x"),
            AcceptanceScenarioV1(name="declared_two", path="/y"),
        ],
        slices=[
            ProjectVerticalSliceV1(
                name="One",
                outcome="It works.",
                files=["app/a.py"],
                owned_files=["app/a.py"],
                integration_files=[],
                # A duplicate and an undeclared name, beside two valid ones.
                scenario_names=[
                    "declared_one",
                    "never_declared",
                    "declared_two",
                    "declared_one",
                ],
            )
        ],
    )

    assert plan.slices[0].scenario_names == ["declared_one", "declared_two"]


def test_the_schema_bound_stays_fail_closed_above_eight() -> None:
    from pydantic import ValidationError

    from waqil_api.contracts import ProjectBuildPlanV1

    with pytest.raises(ValidationError):
        ProjectBuildPlanV1(
            files=["app/a.py"],
            intent="edit",
            scope="narrow",
            scenarios=[
                AcceptanceScenarioV1(name=f"s{index}", path="/x") for index in range(9)
            ],
        )


def test_the_six_scenario_atlas_merge_now_succeeds() -> None:
    """The exact live rejection: six scenarios across two merged slices."""

    plan = [
        {
            "name": "Console shell and styling",
            "outcome": "The page loads and is styled.",
            "files": ["app/static/index.html", "app/static/styles.css"],
            "owned_files": ["app/static/index.html", "app/static/styles.css"],
            "integration_files": [],
            "scenario_names": ["landmarks", "styles"],
        },
        {
            "name": "Console data and interaction layer",
            "outcome": "The table renders and the status action works.",
            "files": ["app/static/app.js"],
            "owned_files": ["app/static/app.js"],
            "integration_files": [],
            "scenario_names": ["listing", "patch", "patch_404", "patch_422"],
        },
        {
            "name": "UI contract tests and documentation",
            "outcome": "Tests and docs cover the console.",
            "files": ["tests/test_ui_contract.py", "README.md"],
            "owned_files": ["tests/test_ui_contract.py", "README.md"],
            "integration_files": [],
            "scenario_names": ["listing", "patch"],
        },
    ]

    result = normalize_build_plan(ATLAS_FILES, plan)

    assert result.ok and result.applied, result.rejected
    merged = result.slices[1]
    # Every scenario both slices named, deduplicated, none dropped.
    assert merged["scenario_names"] == ["listing", "patch", "patch_404", "patch_422"]
    assert merged["owned_files"] == [
        "app/static/app.js",
        "tests/test_ui_contract.py",
        "README.md",
    ]


def test_a_merged_slice_may_carry_all_eight_plan_scenarios() -> None:
    names = [f"s{index}" for index in range(8)]
    plan = [
        {
            "name": "Feature",
            "outcome": "It works.",
            "files": ["app/a.py"],
            "owned_files": ["app/a.py"],
            "integration_files": [],
            "scenario_names": names[:5],
        },
        {
            "name": "Tests",
            "outcome": "Tests cover it.",
            "files": ["tests/test_a.py"],
            "owned_files": ["tests/test_a.py"],
            "integration_files": [],
            "scenario_names": names[5:],
        },
    ]

    result = normalize_build_plan(["app/a.py", "tests/test_a.py"], plan)

    assert result.ok and result.applied, result.rejected
    assert result.slices[0]["scenario_names"] == names


def test_repair_routing_replays_every_scenario_of_a_six_scenario_plan() -> None:
    from waqil_api.project_repair_routing import route_acceptance_finding

    plan = [
        {
            "name": "Shell",
            "outcome": "The page loads.",
            "files": ["app/static/index.html", "app/static/styles.css"],
            "owned_files": ["app/static/index.html", "app/static/styles.css"],
            "integration_files": [],
            "scenario_names": ["landmarks", "styles"],
        },
        {
            "name": "Data layer",
            "outcome": "The table renders.",
            "files": ["app/static/app.js", "tests/test_ui_contract.py", "README.md"],
            "owned_files": [
                "app/static/app.js",
                "tests/test_ui_contract.py",
                "README.md",
            ],
            "integration_files": [],
            "scenario_names": ["listing", "patch", "patch_404", "patch_422"],
        },
    ]
    staged = {path: {"content": f"# {path}\n"} for path in ATLAS_FILES}

    route = route_acceptance_finding(
        findings=[
            {
                "path": "README.md",
                "error": "failed",
                "severity": "error",
                "kind": "acceptance",
            }
        ],
        slices=plan,
        staged=staged,
    )

    # All six, none trimmed to five.
    assert route.rerun_scenarios == (
        "landmarks",
        "styles",
        "listing",
        "patch",
        "patch_404",
        "patch_422",
    )
    assert len(route.rerun_scenarios) == 6
