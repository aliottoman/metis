"""Tool Factory v2 — P7 registry foundation.

Proves the registry seeds the reference-architecture tool as entry #1, that
routing driven by the registry catalog is byte-identical to the pre-registry v1
behavior, and that the framework-of-control invariants hold (unknown profiles
fail closed, unregistered slugs never route).
"""
from __future__ import annotations

import tempfile
import types
from pathlib import Path

import pytest

from waqil_api import capability_profiles
from waqil_api.contracts import (
    CapabilityProfileV1,
    PlanEnvelopeV1,
    PlanningRequestV1,
    RiskLevel,
    ToolDefinitionV1,
)
from waqil_api.database import Database
from waqil_api.model_provider import (
    RoutingCatalog,
    ToolRoute,
    default_routing_catalog,
    normalize_plan_semantics,
    validate_plan_semantics,
)
from waqil_api.tool_registry import (
    REFERENCE_ARCHITECTURE_SLUG,
    ToolRegistry,
    definition_hash,
    finalize,
)


async def _registry(model_authoring: bool = False) -> tuple[Database, ToolRegistry]:
    db = Database(Path(tempfile.mkdtemp()) / "w.db")
    await db.open()
    settings = types.SimpleNamespace(tool_model_authoring=model_authoring)
    registry = ToolRegistry(db, settings)
    await registry.seed_builtins()
    return db, registry


def _arch_request(active: bool = False) -> PlanningRequestV1:
    active_tools = (
        [
            {
                "slug": REFERENCE_ARCHITECTURE_SLUG,
                "active_version_id": "ver_x",
                "version": "0.3.0",
                "content_hash": "abc",
            }
        ]
        if active
        else []
    )
    return PlanningRequestV1(
        run_id="run_1",
        conversation_id="conv_1",
        prompt="Create a reference architecture diagram from the attached README",
        active_tools=active_tools,
    )


def _neutral_plan() -> PlanEnvelopeV1:
    return PlanEnvelopeV1(summary="x", route="direct", risk_level=RiskLevel.R0)


# ── Seeding ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_seeds_reference_architecture_as_entry_one() -> None:
    db, registry = await _registry()
    catalog = await registry.catalog()
    assert [entry["slug"] for entry in catalog] == [REFERENCE_ARCHITECTURE_SLUG]
    definition = await registry.get(REFERENCE_ARCHITECTURE_SLUG)
    assert definition is not None
    # Default install: v1 behavior — routing facts and exact-canonical profile,
    # no runtime model access (so real Podman runs are unaffected).
    assert definition.version == "1"
    assert definition.route_facts.existing_risk == RiskLevel.R2
    assert definition.route_facts.factory_risk == RiskLevel.R3
    assert definition.route_facts.input_pipeline == "architecture_spec"
    assert definition.capability_profile.code_allowlist == "diagrams-render-v1"
    assert definition.capability_profile.model_access.enabled is False
    assert definition_hash(definition) == definition.content_hash
    await db.close()


@pytest.mark.asyncio
async def test_model_authoring_flag_activates_v2_definition() -> None:
    db, registry = await _registry(model_authoring=True)
    definition = await registry.get(REFERENCE_ARCHITECTURE_SLUG)
    assert definition.version == "2"
    access = definition.capability_profile.model_access
    assert access.enabled is True
    assert access.roles == ["coder"]
    assert access.max_calls_per_run == 1
    assert "author_diagram_code" in access.prompt_templates
    assert definition.capability_profile.runtime_allowlists == {
        "diagram_code": "diagrams-draw-v2"
    }
    # Routing is unchanged whether or not authoring is enabled.
    assert definition.route_facts.existing_risk == RiskLevel.R2
    await db.close()


@pytest.mark.asyncio
async def test_seed_is_idempotent() -> None:
    db, registry = await _registry()
    await registry.seed_builtins()
    await registry.seed_builtins()
    assert len(await registry.list_active()) == 1
    await db.close()


@pytest.mark.asyncio
async def test_catalog_never_exposes_capability_profile() -> None:
    db, registry = await _registry()
    entry = (await registry.catalog())[0]
    # The planner-facing catalog is identity + intent only.
    assert set(entry) == {"slug", "name", "description", "intent_examples"}
    assert "capability_profile" not in entry
    assert "route_facts" not in entry
    await db.close()


# ── Behavior-identical routing ───────────────────────────────────────────────


def _catalog_from_definition(definition: ToolDefinitionV1) -> RoutingCatalog:
    return RoutingCatalog(
        architecture_tool=ToolRoute(
            slug=definition.slug,
            existing_risk=definition.route_facts.existing_risk,
            factory_risk=definition.route_facts.factory_risk,
            input_pipeline=definition.route_facts.input_pipeline,
        ),
        known_slugs=frozenset({definition.slug}),
    )


@pytest.mark.asyncio
async def test_registry_catalog_routes_identically_to_v1_defaults() -> None:
    db, registry = await _registry()
    definition = await registry.get(REFERENCE_ARCHITECTURE_SLUG)
    registry_catalog = _catalog_from_definition(definition)
    v1_catalog = default_routing_catalog()

    for active in (False, True):
        request = _arch_request(active=active)
        via_registry = normalize_plan_semantics(_neutral_plan(), request, registry_catalog)
        via_v1 = normalize_plan_semantics(_neutral_plan(), request, v1_catalog)
        # Registry-driven routing must equal the hardcoded v1 routing exactly.
        assert via_registry.model_dump() == via_v1.model_dump()
        # And it must be the expected route.
        assert via_registry.route == ("existing_tool" if active else "tool_factory")
        assert via_registry.tool_slug == REFERENCE_ARCHITECTURE_SLUG
        assert via_registry.risk_level == (RiskLevel.R2 if active else RiskLevel.R3)
        validate_plan_semantics(via_registry, request, registry_catalog)
    await db.close()


def test_non_architecture_request_routes_direct() -> None:
    request = PlanningRequestV1(
        run_id="r", conversation_id="c", prompt="What is the capital of France?"
    )
    plan = normalize_plan_semantics(_neutral_plan(), request, default_routing_catalog())
    assert plan.route == "direct"
    assert plan.tool_slug is None
    assert plan.risk_level == RiskLevel.R0
    validate_plan_semantics(plan, request, default_routing_catalog())


# ── Framework-of-control invariants ──────────────────────────────────────────


def test_validate_rejects_unregistered_slug() -> None:
    request = _arch_request(active=True)
    rogue = PlanEnvelopeV1(
        summary="x",
        route="existing_tool",
        tool_slug="pdf-magic-tool",  # not in the catalog
        risk_level=RiskLevel.R2,
    )
    with pytest.raises(ValueError, match="only registered tool capabilities"):
        validate_plan_semantics(request=request, plan=rogue, catalog=default_routing_catalog())


@pytest.mark.asyncio
async def test_unknown_capability_profile_fails_closed() -> None:
    db = Database(Path(tempfile.mkdtemp()) / "w.db")
    await db.open()
    registry = ToolRegistry(db)
    bad = finalize(
        ToolDefinitionV1(
            slug="bad-tool",
            version="1",
            name="Bad",
            description="references a profile that does not exist",
            capability_profile=CapabilityProfileV1(code_allowlist="does-not-exist"),
        )
    )
    with pytest.raises(ValueError, match="unknown capability profile"):
        registry._require_profiles(bad)
    await db.close()


@pytest.mark.asyncio
async def test_builtin_content_change_at_same_version_upgrades_in_place() -> None:
    # Reproduces the deploy migration where a builtin's *content* changes (e.g. a
    # new `archetype` field) but its version does not: the seed must upgrade the
    # existing (slug, version) row in place, not INSERT and hit UNIQUE(slug,version).
    db = Database(Path(tempfile.mkdtemp()) / "w.db")
    await db.open()
    original = finalize(
        ToolDefinitionV1(
            slug=REFERENCE_ARCHITECTURE_SLUG,
            version="1",
            name="Reference Architecture Generator",
            description="old",
            capability_profile=CapabilityProfileV1(code_allowlist="diagrams-render-v1"),
        )
    )
    await db.upsert_tool_definition(original, activate=True)
    upgraded = finalize(original.model_copy(update={"archetype": "architecture", "content_hash": ""}))
    assert upgraded.content_hash != original.content_hash
    await db.upsert_tool_definition(upgraded, activate=True)  # must not raise
    active = await db.get_active_tool_definition(REFERENCE_ARCHITECTURE_SLUG)
    assert active is not None and active.content_hash == upgraded.content_hash
    assert active.archetype == "architecture"
    rows = [r for r in await db.list_tool_definitions() if r.slug == REFERENCE_ARCHITECTURE_SLUG]
    assert len(rows) == 1  # upgraded in place, not duplicated
    # Re-seeding the same content again is idempotent.
    await db.upsert_tool_definition(upgraded, activate=True)
    assert len([r for r in await db.list_tool_definitions() if r.slug == REFERENCE_ARCHITECTURE_SLUG]) == 1
    await db.close()


def test_all_builtin_profiles_are_registered() -> None:
    from waqil_api.tool_registry import builtin_definitions

    for definition in builtin_definitions():
        profile = definition.capability_profile
        for name in [profile.code_allowlist, *profile.runtime_allowlists.values()]:
            assert capability_profiles.exists(name), f"missing profile: {name}"
