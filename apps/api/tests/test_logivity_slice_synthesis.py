"""The Logivity revamp, at the point where it used to fall apart.

Logivity is a Streamlit invoice extractor: an `app.py` that owns the UI, plus
`extractor.py`, `excel_writer.py`, `surcharge_mapping.py` and a DHL `.xlsx`
template that carry the OCI vision call and the Excel contract. Porting it to
FastAPI plus a static frontend touches the first and must not touch the rest.

Both captured planner replies for that request named a manifest and declared
no slices. The host's answer was a mechanical six-file partition, which put
`README.md` and `requirements.txt` alone in a second chunk -- a support-only
slice, with an outcome sentence Metis wrote and an integration boundary no
model proposed. The topology gate then rejected it, correctly, for a defect
the host had introduced.

The fix is to invent less: one slice owning the whole manifest.
"""

from __future__ import annotations

from typing import Any

from waqil_api.contracts import MAX_SLICE_FILES, ProjectBuildPlanV1
from waqil_api.project_plan_validation import (
    support_kind,
    validate_build_plan,
    validate_effective_plan,
)
from waqil_api.project_slices import (
    next_vertical_slice,
    synthesize_single_slice,
    vertical_build_slices,
)


# ── The two captured manifests ─────────────────────────────────────────────

# Reply A: seven files. The chunker split this 6 + 1.
LOGIVITY_SEVEN = [
    "app/main.py",
    "app/routes.py",
    "app/static/index.html",
    "app/static/app.js",
    "app/static/styles.css",
    "requirements.txt",
    "README.md",
]

# Reply B: eight files, adding a contract test. The chunker split this 6 + 2,
# and that trailing pair was tests-and-docs -- support only.
LOGIVITY_EIGHT = [
    "app/main.py",
    "app/routes.py",
    "app/static/index.html",
    "app/static/app.js",
    "app/static/styles.css",
    "tests/test_extraction_contract.py",
    "requirements.txt",
    "README.md",
]

# The files the port must not touch: the OCI vision call, the Excel writer,
# the surcharge mapping and the DHL template.
LOGIVITY_PROTECTED = [
    "extractor.py",
    "excel_writer.py",
    "surcharge_mapping.py",
    "DHL_Template.xlsx",
    "config.py",
]

SCENARIOS: list[dict[str, Any]] = [
    {"name": "the page serves its own stylesheet and script"},
    {"name": "uploading the sample invoice returns the extracted fields"},
    {"name": "the filled workbook downloads"},
    {"name": "extraction output is unchanged from the streamlit build"},
]


def _synthesized(manifest: list[str]) -> dict[str, Any]:
    result = synthesize_single_slice(manifest, SCENARIOS)
    assert result is not None, f"{len(manifest)} files should synthesize"
    return result


# ── Both manifests become one valid slice ──────────────────────────────────


def test_the_seven_file_manifest_becomes_one_valid_slice() -> None:
    slice_ = _synthesized(LOGIVITY_SEVEN)

    assert slice_["owned_files"] == LOGIVITY_SEVEN
    assert slice_["files"] == LOGIVITY_SEVEN
    assert slice_["integration_files"] == []
    assert validate_build_plan(LOGIVITY_SEVEN, [slice_], scenarios=SCENARIOS).ok


def test_the_eight_file_manifest_becomes_one_valid_slice() -> None:
    slice_ = _synthesized(LOGIVITY_EIGHT)

    assert len(slice_["owned_files"]) == MAX_SLICE_FILES
    assert slice_["owned_files"] == LOGIVITY_EIGHT
    assert validate_build_plan(LOGIVITY_EIGHT, [slice_], scenarios=SCENARIOS).ok


def test_both_manifests_pass_the_effective_gate_with_no_declared_slices() -> None:
    # The end-to-end statement: a plan with no slices no longer has to be
    # rejected to be safe.
    assert validate_effective_plan(LOGIVITY_SEVEN, [], scenarios=SCENARIOS).ok
    assert validate_effective_plan(LOGIVITY_EIGHT, [], scenarios=SCENARIOS).ok


# ── The defect that used to be introduced by the host ──────────────────────


def test_readme_and_requirements_no_longer_form_a_support_only_slice() -> None:
    # The exact old failure. Both trailing files are support files, so any
    # partition that isolates them produces a slice delivering nothing
    # runnable.
    assert support_kind("README.md") == "documentation"
    assert support_kind("requirements.txt") == "dependency"

    old_partition = vertical_build_slices(LOGIVITY_SEVEN, [])
    assert old_partition[-1]["owned_files"] == ["README.md"]
    stranded = validate_build_plan(LOGIVITY_SEVEN, old_partition, scenarios=SCENARIOS)
    assert stranded.ok is False
    assert "support_only_slice" in stranded.codes

    # Synthesis produces no such slice, and nothing in the effective plan is
    # support-only.
    slice_ = _synthesized(LOGIVITY_SEVEN)
    scope = slice_["owned_files"] + slice_["integration_files"]
    assert not all(support_kind(path) for path in scope)


def test_the_eight_file_manifest_no_longer_strands_tests_and_docs() -> None:
    old_partition = vertical_build_slices(LOGIVITY_EIGHT, [])
    assert old_partition[-1]["owned_files"] == [
        "requirements.txt",
        "README.md",
    ]
    assert all(support_kind(path) for path in old_partition[-1]["owned_files"])

    slice_ = _synthesized(LOGIVITY_EIGHT)
    assert slice_["owned_files"] == LOGIVITY_EIGHT


# ── Order, ownership and scope ─────────────────────────────────────────────


def test_manifest_order_and_exact_ownership_are_preserved() -> None:
    for manifest in (LOGIVITY_SEVEN, LOGIVITY_EIGHT):
        slice_ = _synthesized(manifest)
        # Exact, ordered, no duplicates, nothing added and nothing dropped.
        assert slice_["owned_files"] == manifest
        assert len(set(slice_["owned_files"])) == len(manifest)
        # The synthesized slice is the whole partition, so the frontier walk
        # selects it at zero and finds nothing after it.
        assert next_vertical_slice(manifest, [slice_], verified_count=0) == slice_
        assert (
            next_vertical_slice(manifest, [slice_], verified_count=len(manifest))
            is None
        )


def test_every_declared_scenario_is_carried_onto_the_slice() -> None:
    slice_ = _synthesized(LOGIVITY_EIGHT)

    assert slice_["scenario_names"] == [item["name"] for item in SCENARIOS]


def test_the_protected_extractor_and_template_never_enter_scope() -> None:
    """The whole point of the request: the OCI call and Excel output stand."""

    for manifest in (LOGIVITY_SEVEN, LOGIVITY_EIGHT):
        slice_ = _synthesized(manifest)
        scope = set(slice_["owned_files"]) | set(slice_["integration_files"])
        assert scope.isdisjoint(LOGIVITY_PROTECTED)
        # Synthesis owns the manifest and only the manifest -- it can never
        # widen scope to a file the plan did not name.
        assert scope == set(manifest)


def test_synthesis_cannot_widen_scope_even_if_scenarios_mention_other_files() -> None:
    # Scenario text is not a scope input. A scenario naming extractor.py must
    # not pull it into the slice.
    scenarios = SCENARIOS + [{"name": "extractor.py is byte-identical"}]

    slice_ = synthesize_single_slice(LOGIVITY_SEVEN, scenarios)

    assert slice_ is not None
    assert slice_["owned_files"] == LOGIVITY_SEVEN
    assert "extractor.py" not in slice_["owned_files"]


# ── Beyond the bound, and plans that already declare slices ────────────────


def test_a_ninth_file_is_never_mechanically_partitioned() -> None:
    oversized = LOGIVITY_EIGHT + ["app/static/print.css"]

    assert synthesize_single_slice(oversized, SCENARIOS) is None
    # And the effective gate does not invent a finding about a partition
    # nothing will build; the control plane corrects or refuses instead.
    assert validate_effective_plan(oversized, [], scenarios=SCENARIOS).ok


def test_a_plan_that_declares_its_own_slices_is_left_alone() -> None:
    declared = [
        {
            "name": "FastAPI shell serving the page",
            "outcome": "The app serves index.html with its assets.",
            "files": LOGIVITY_SEVEN[:5],
            "owned_files": LOGIVITY_SEVEN[:5],
            "integration_files": [],
            "scenario_names": [SCENARIOS[0]["name"]],
        },
        {
            "name": "Packaging and docs for the port",
            "outcome": "The port installs and is documented.",
            "files": LOGIVITY_SEVEN[5:] + ["app/main.py"],
            "owned_files": LOGIVITY_SEVEN[5:],
            "integration_files": ["app/main.py"],
            "scenario_names": [SCENARIOS[2]["name"]],
        },
    ]

    effective = vertical_build_slices(LOGIVITY_SEVEN, declared)

    assert [item["name"] for item in effective] == [
        "FastAPI shell serving the page",
        "Packaging and docs for the port",
    ]
    assert effective[1]["integration_files"] == ["app/main.py"]


def test_the_contract_accepts_an_eight_file_slice_end_to_end() -> None:
    # The bound has to agree all the way down to the schema, or a synthesized
    # slice would be built and then refused at persistence.
    plan = ProjectBuildPlanV1(
        files=LOGIVITY_EIGHT,
        intent="edit",
        scope="narrow",
        scenarios=[],
        slices=[
            {
                "name": "Complete implementation",
                "outcome": "Deliver every planned file as one increment.",
                "files": LOGIVITY_EIGHT,
                "owned_files": LOGIVITY_EIGHT,
                "integration_files": [],
            }
        ],
    )

    assert plan.slices[0].owned_files == LOGIVITY_EIGHT
