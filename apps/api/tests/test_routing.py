"""P9 — generalized routing (declarative tools + the tool_definition route).

The host still disposes: it honors a planner-proposed slug only to select an
already-registered tool, recomputing runnable/buildable state and risk from the
trusted catalog, and it opens the definition route on an explicit "toolify this"
or a planner-proposed new tool. The architecture path stays byte-identical
(covered in test_tool_registry.py); here we pin the new behaviors.
"""
from __future__ import annotations

import pytest

from waqil_api.contracts import PlanEnvelopeV1, PlanningRequestV1, RiskLevel
from waqil_api.model_provider import (
    RoutingCatalog,
    ToolRoute,
    default_routing_catalog,
    is_explicit_toolify_request,
    normalize_plan_semantics,
    validate_plan_semantics,
)

ARCH = default_routing_catalog().architecture_tool


def _request(prompt: str, *, attach: bool = False) -> PlanningRequestV1:
    return PlanningRequestV1(
        run_id="r",
        conversation_id="c",
        prompt=prompt,
        attachment_ids=["a1"] if attach else [],
    )


def _catalog(*tools: ToolRoute, factory=True, definition=True) -> RoutingCatalog:
    slugs = {ARCH.slug} | {tool.slug for tool in tools}
    return RoutingCatalog(
        architecture_tool=ARCH,
        known_slugs=frozenset(slugs),
        tools=tools,
        factory_enabled=factory,
        definition_enabled=definition,
    )


def _declarative(slug: str, *, runnable=False, buildable=False, disabled=False) -> ToolRoute:
    return ToolRoute(
        slug=slug,
        existing_risk=RiskLevel.R2,
        factory_risk=RiskLevel.R3,
        input_pipeline="attachment_text",
        runnable=runnable,
        buildable=buildable,
        disabled=disabled,
    )


def _proposed(route: str, slug: str | None) -> PlanEnvelopeV1:
    return PlanEnvelopeV1(summary="x", route=route, tool_slug=slug, risk_level=RiskLevel.R0)


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("turn this into a tool", True),
        ("please toolify this readme summarizer", True),
        ("build a reusable tool for summaries", True),
        ("save this workflow as a tool", True),
        ("make this into a tool", True),
        ("what is a tool?", False),
        ("summarize this readme", False),
        ("use the build tool in the garage", False),
    ],
)
def test_explicit_toolify_detection(prompt: str, expected: bool) -> None:
    assert is_explicit_toolify_request(prompt) is expected


def test_runnable_declarative_routes_existing_tool() -> None:
    catalog = _catalog(_declarative("readme-summary", runnable=True))
    plan = normalize_plan_semantics(
        _proposed("existing_tool", "readme-summary"), _request("do it", attach=True), catalog
    )
    assert plan.route == "existing_tool"
    assert plan.tool_slug == "readme-summary"
    assert plan.risk_level == RiskLevel.R2
    validate_plan_semantics(plan, _request("do it", attach=True), catalog)


def test_buildable_declarative_routes_tool_factory() -> None:
    catalog = _catalog(_declarative("readme-summary", buildable=True))
    plan = normalize_plan_semantics(
        _proposed("existing_tool", "readme-summary"), _request("do it", attach=True), catalog
    )
    assert plan.route == "tool_factory"
    assert plan.risk_level == RiskLevel.R3
    validate_plan_semantics(plan, _request("do it", attach=True), catalog)


def test_input_not_ready_falls_back_to_direct() -> None:
    catalog = _catalog(_declarative("readme-summary", runnable=True))
    # No attachment → the attachment_text pipeline can't be satisfied → direct.
    plan = normalize_plan_semantics(
        _proposed("existing_tool", "readme-summary"), _request("do it", attach=False), catalog
    )
    assert plan.route == "direct"


def test_disabled_tool_never_routes() -> None:
    catalog = _catalog(_declarative("readme-summary", runnable=True, disabled=True))
    plan = normalize_plan_semantics(
        _proposed("existing_tool", "readme-summary"), _request("do it", attach=True), catalog
    )
    assert plan.route == "direct"


def test_explicit_toolify_opens_definition_route() -> None:
    catalog = _catalog()
    plan = normalize_plan_semantics(_proposed("direct", None), _request("turn this into a tool"), catalog)
    assert plan.route == "tool_definition"
    assert plan.tool_slug is None
    assert plan.risk_level == RiskLevel.R3
    validate_plan_semantics(plan, _request("turn this into a tool"), catalog)


def test_planner_proposed_definition_is_honored() -> None:
    catalog = _catalog()
    plan = normalize_plan_semantics(
        _proposed("tool_definition", None), _request("draft me a widget summarizer"), catalog
    )
    assert plan.route == "tool_definition"


def test_factory_kill_switch_blocks_build_and_definition() -> None:
    catalog = _catalog(_declarative("readme-summary", buildable=True), factory=False)
    # Buildable tool → cannot build while paused.
    plan = normalize_plan_semantics(
        _proposed("existing_tool", "readme-summary"), _request("do it", attach=True), catalog
    )
    assert plan.route == "direct"
    # And no new definitions may be drafted.
    plan = normalize_plan_semantics(_proposed("direct", None), _request("turn this into a tool"), catalog)
    assert plan.route == "direct"


def test_definition_kill_switch_blocks_only_drafting() -> None:
    catalog = _catalog(_declarative("readme-summary", runnable=True), definition=False)
    # Drafting is off…
    plan = normalize_plan_semantics(_proposed("direct", None), _request("turn this into a tool"), catalog)
    assert plan.route == "direct"
    # …but running an already-active tool still works.
    plan = normalize_plan_semantics(
        _proposed("existing_tool", "readme-summary"), _request("do it", attach=True), catalog
    )
    assert plan.route == "existing_tool"


def test_pending_tool_is_built_not_redrafted() -> None:
    # The reported bug: after Gate-1 approval the tool is buildable; a build/toolify
    # follow-up must BUILD it, not draft a brand-new (and possibly unrelated) tool.
    catalog = _catalog(_declarative("meeting-tracker", buildable=True))
    for prompt in ("build it", "build the tool now", "create this into a tool", "make it"):
        plan = normalize_plan_semantics(_proposed("direct", None), _request(prompt), catalog)
        assert plan.route == "tool_factory", prompt
        assert plan.tool_slug == "meeting-tracker", prompt
    # A fresh toolify when NOTHING is pending still drafts a new tool.
    plan = normalize_plan_semantics(_proposed("direct", None), _request("turn this into a tool"), _catalog())
    assert plan.route == "tool_definition"
    # An unrelated request while a tool is pending does NOT trigger a build.
    plan = normalize_plan_semantics(_proposed("direct", None), _request("what is the capital of France?"), catalog)
    assert plan.route == "direct"


def test_validate_rejects_risk_mismatch_for_declarative() -> None:
    catalog = _catalog(_declarative("readme-summary", runnable=True))
    bad = PlanEnvelopeV1(
        summary="x", route="existing_tool", tool_slug="readme-summary", risk_level=RiskLevel.R0
    )
    with pytest.raises(ValueError, match="risk does not match"):
        validate_plan_semantics(bad, _request("do it", attach=True), catalog)


def test_validate_tool_definition_requires_r3_and_no_slug() -> None:
    catalog = _catalog()
    with pytest.raises(ValueError, match="carries no slug"):
        validate_plan_semantics(
            PlanEnvelopeV1(summary="x", route="tool_definition", tool_slug="x", risk_level=RiskLevel.R3),
            _request("q"),
            catalog,
        )
