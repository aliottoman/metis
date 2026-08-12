"""A plan must cover every outcome the evaluator's probe asserts.

The live Atlas canary declared six acceptance scenarios against a probe that
asserts eight, and nothing noticed until after the coder had spent 292,436
tokens. Counting scenarios would not have caught it either: six is not
obviously wrong, and the two that were missing were the two static-asset
checks. Coverage has to be judged by identity.
"""

from __future__ import annotations

from typing import Any

from waqil_api.project_capability_scenarios import (
    SCENARIOS,
    UI_REQUIRED_SCENARIOS,
    missing_required_scenarios,
)


# Copied verbatim from the retained canary checkpoint.
ATLAS_DECLARED_SIX: list[dict[str, Any]] = [
    {"name": "console page loads with landmarks", "method": "GET", "path": "/"},
    {"name": "health contract exact", "method": "GET", "path": "/api/health"},
    {"name": "projects contract exact", "method": "GET", "path": "/api/projects"},
    {
        "name": "patch status healthy",
        "method": "PATCH",
        "path": "/api/projects/atlas-2/status",
    },
    {
        "name": "patch status 422 invalid",
        "method": "PATCH",
        "path": "/api/projects/atlas-1/status",
    },
    {
        "name": "patch status 404 unknown",
        "method": "PATCH",
        "path": "/api/projects/atlas-999/status",
    },
]


def test_the_atlas_probe_declares_eight_required_outcomes() -> None:
    assert len(UI_REQUIRED_SCENARIOS) == 8
    assert SCENARIOS["ui-revamp"].required_scenarios == UI_REQUIRED_SCENARIOS


def test_the_exact_six_of_eight_canary_plan_is_short_two_outcomes() -> None:
    missing = missing_required_scenarios(SCENARIOS["ui-revamp"], ATLAS_DECLARED_SIX)

    # Six declared, and exactly the two static-asset outcomes uncovered.
    assert len(ATLAS_DECLARED_SIX) == 6
    assert missing == [
        "the stylesheet is served",
        "the interaction script is served",
    ]


def test_coverage_is_identity_not_count() -> None:
    """Eight scenarios that duplicate one outcome still fail."""

    padded = ATLAS_DECLARED_SIX + [
        {"name": "health again", "method": "GET", "path": "/api/health"},
        {"name": "health once more", "method": "GET", "path": "/api/health"},
    ]

    assert len(padded) == 8
    assert missing_required_scenarios(SCENARIOS["ui-revamp"], padded) == [
        "the stylesheet is served",
        "the interaction script is served",
    ]


def test_a_complete_plan_passes_however_the_planner_worded_it() -> None:
    """The planner's own phrasing is never required to match ours."""

    complete = ATLAS_DECLARED_SIX + [
        {
            "name": "styles.css is reachable",
            "method": "GET",
            "path": "/static/styles.css",
        },
        {"name": "app.js is reachable", "method": "GET", "path": "/static/app.js"},
    ]

    assert missing_required_scenarios(SCENARIOS["ui-revamp"], complete) == []


def test_a_scenario_naming_the_asset_in_prose_also_counts() -> None:
    complete = ATLAS_DECLARED_SIX + [
        {"name": "the stylesheet loads", "method": "GET", "path": "/"},
        {"name": "the interaction script loads", "method": "GET", "path": "/"},
    ]

    assert missing_required_scenarios(SCENARIOS["ui-revamp"], complete) == []


def test_an_empty_plan_is_missing_everything() -> None:
    missing = missing_required_scenarios(SCENARIOS["ui-revamp"], [])

    assert missing == list(UI_REQUIRED_SCENARIOS)


def test_a_scenario_set_without_required_outcomes_never_blocks() -> None:
    """Meridian and the repair scenario declare none, and are unaffected."""

    for key in ("meridian", "fastapi-repair"):
        assert SCENARIOS[key].required_scenarios == ()
        assert missing_required_scenarios(SCENARIOS[key], []) == []
