"""Tool registry — the source of truth for what tools exist and what they may do.

A tool is a declarative ``ToolDefinitionV1`` record. The registry seeds the
host's built-in definitions (the reference-architecture tool is entry #1),
validates that every definition references only *registered* capability
profiles (fail closed), and exposes two very different views:

- ``catalog()`` — name / description / intent examples only. This is all the
  planner LLM ever sees; it cannot read contracts, profiles, or permissions.
- ``get(slug)`` — the full definition, read host-side to drive routing, risk,
  and execution. Executable truth never reaches the model.

Definitions are content-hashed and immutable; changing one is a new version.
"""
from __future__ import annotations

import hashlib
import json

from . import capability_profiles
from .contracts import (
    CapabilityProfileV1,
    ModelAccessV1,
    ProposalStatus,
    RiskLevel,
    ToolDefinitionRecordV1,
    ToolDefinitionV1,
    ToolRouteFactsV1,
)
from .database import Database
from .model_provider import is_explicit_toolify_request

REFERENCE_ARCHITECTURE_SLUG = "reference-architecture-generator"


def definition_hash(definition: ToolDefinitionV1) -> str:
    """Stable content hash over a definition's semantic fields (excludes the
    hash itself and the created_at timestamp)."""
    payload = definition.model_copy(update={"content_hash": ""}).model_dump(
        mode="json", exclude={"created_at"}
    )
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def finalize(definition: ToolDefinitionV1) -> ToolDefinitionV1:
    """Stamp the content hash onto a definition."""
    return definition.model_copy(update={"content_hash": definition_hash(definition)})


# The pinned system prompt the reference-architecture tool uses to have the
# model author diagram code at runtime. Frozen in the definition; the tool fills
# only the (spec, formats, reference) parameters, never this text.
_AUTHOR_DIAGRAM_CODE_PROMPT = (
    "You are Metis's diagram code author. Given an architecture specification, "
    "write ONE Python program using the `diagrams` library that renders it.\n"
    "Hard rules — code outside these will be rejected and discarded:\n"
    "- Import only: `from pathlib import Path`, "
    "`from diagrams import Cluster, Diagram, Edge`, "
    "`from diagrams.generic.blank import Blank`.\n"
    "- Set OUTPUT_STEM = str(Path(__file__).resolve().parent / \"architecture\").\n"
    "- One `with Diagram(...)` using filename=OUTPUT_STEM, the given outformat, "
    "show=False, and the given direction. Improve layout with graph_attr, "
    "node_attr, and edge_attr (e.g. splines='ortho', sensible nodesep/ranksep).\n"
    "- Represent EVERY component as exactly one Blank(label) node and EVERY edge "
    "with the >> operator (optionally through Edge(label=...)). Put each boundary "
    "in one Cluster.\n"
    "- No other imports, calls, names, control flow, comprehensions, or I/O.\n"
    "Return only the program text (a ```python block is fine)."
)


def _reference_architecture_definition(model_authoring: bool) -> ToolDefinitionV1:
    """The built-in reference-architecture tool.

    Routing is identical either way (R2 existing / R3 factory, architecture-spec
    pipeline). When ``model_authoring`` is off (default) this is v1: the model
    copies the exact canonical source (``diagrams-render-v1``), no runtime model
    access — real Podman runs are unaffected. When on, it is v2: the model
    *authors* the diagram code via one brokered ``coder`` call using the pinned
    template, validated against the richer ``diagrams-draw-v2`` allowlist, with
    the deterministic canonical source as the safe fallback. v2 requires the
    sandbox image rebuilt with the matching in-container validator."""
    if model_authoring:
        capability_profile = CapabilityProfileV1(
            code_allowlist="diagrams-render-v1",
            runtime_allowlists={"diagram_code": "diagrams-draw-v2"},
            model_access=ModelAccessV1(
                enabled=True,
                roles=["coder"],
                max_calls_per_run=1,
                max_tokens_per_call=6_144,
                prompt_templates={"author_diagram_code": _AUTHOR_DIAGRAM_CODE_PROMPT},
            ),
        )
        version = "2"
    else:
        capability_profile = CapabilityProfileV1(
            code_allowlist="diagrams-render-v1",
            model_access=ModelAccessV1(),  # disabled — v1 behavior
        )
        version = "1"
    return finalize(
        ToolDefinitionV1(
            slug=REFERENCE_ARCHITECTURE_SLUG,
            version=version,
            name="Reference Architecture Generator",
            archetype="architecture",
            description=(
                "Extract a reference architecture from an attached project README "
                "and render it as a diagram."
            ),
            intent_examples=[
                "Create a reference architecture from this README",
                "Draw an architecture diagram of the attached project",
                "Map the components and topology of this repository",
            ],
            route_facts=ToolRouteFactsV1(
                existing_risk=RiskLevel.R2,
                factory_risk=RiskLevel.R3,
                input_pipeline="architecture_spec",
            ),
            capability_profile=capability_profile,
            status="defined",
        )
    )


def builtin_definitions(model_authoring: bool = False) -> list[ToolDefinitionV1]:
    """Host-shipped definitions the registry always ensures are present."""
    return [_reference_architecture_definition(model_authoring)]


class ToolRegistry:
    """Reads/seeds tool definitions. Holds only the database handle and settings."""

    def __init__(self, database: Database, settings: object | None = None) -> None:
        self._db = database
        self._settings = settings

    def _model_authoring(self) -> bool:
        return bool(getattr(self._settings, "tool_model_authoring", False))

    def trusted_auto_activation_eligible(
        self, definition: ToolDefinitionV1
    ) -> bool:
        """Whether an evaluated definition fits the narrow trusted fast path."""
        if not bool(
            getattr(self._settings, "tool_trusted_auto_activation", True)
        ):
            return False
        profile = definition.capability_profile
        if profile.network != "none" or profile.filesystem != "run-io":
            return False
        if definition.route_facts.existing_risk not in {RiskLevel.R0, RiskLevel.R1, RiskLevel.R2}:
            return False
        try:
            self._require_profiles(definition)
        except ValueError:
            return False
        return True

    async def reconcile_trusted_evaluated_builds(self) -> list[str]:
        """Activate pre-upgrade evaluated builds that now fit the fast path.

        This makes the new lifecycle apply to an already-built tool instead of
        leaving it stranded behind the retired second approval gate.
        """
        activated: list[str] = []
        if not bool(getattr(self._settings, "tool_trusted_auto_activation", True)):
            return activated
        for build in await self._db.list_tool_definition_builds(status="evaluated"):
            if not build.eval_report or not build.eval_report.passed:
                continue
            definition = await self._db.get_tool_definition_by_id(build.definition_id)
            if definition is None or not self.trusted_auto_activation_eligible(definition):
                continue
            action_id = f"trusted-auto-activation:{build.id}:{build.content_hash[:16]}"
            await self._db.decide_tool_definition_build(
                build.id,
                "active",
                "Auto-activated after passing the trusted local tool boundary.",
                action_id,
            )
            activated.append(build.id)
        return activated

    async def reconcile_trusted_definition_proposals(self) -> list[str]:
        """Promote old safe proposals created by an explicit build request.

        Earlier Metis versions stopped after drafting even when the user had
        already pressed the explicit build action. Reconciliation preserves the
        immutable audit proposal but makes those definitions buildable.
        """
        promoted: list[str] = []
        if not bool(getattr(self._settings, "tool_trusted_auto_activation", True)):
            return promoted
        proposals = await self._db.list_tool_definition_proposals(status="pending")
        for proposal in proposals:
            source = (
                await self._db.get_run_execution_record(proposal.source_run_id)
                if proposal.source_run_id
                else None
            )
            prompt = str((source or {}).get("prompt", ""))
            if not is_explicit_toolify_request(prompt):
                continue
            definition = await self._db.get_tool_definition_by_id(
                proposal.definition_id
            )
            if definition is None or not self.trusted_auto_activation_eligible(definition):
                continue
            action_id = (
                f"trusted-auto-definition:{proposal.id}:"
                f"{definition.content_hash[:16]}"
            )
            await self._db.decide_tool_definition_proposal(
                proposal.id,
                ProposalStatus.APPROVED.value,
                "Explicit user build request; trusted local capability profile.",
                action_id,
            )
            promoted.append(proposal.slug)
        return promoted

    def _require_profiles(self, definition: ToolDefinitionV1) -> None:
        profile = definition.capability_profile
        referenced = [profile.code_allowlist, *profile.runtime_allowlists.values()]
        for name in referenced:
            if not capability_profiles.exists(name):
                raise ValueError(
                    f"tool '{definition.slug}' references unknown capability "
                    f"profile: {name}"
                )

    async def seed_builtins(self) -> None:
        """Ensure each built-in definition's *latest* host version is the active
        one. Reserved builtin slugs are host-owned, so an older builtin version is
        upgraded on deploy; the previous rows remain as immutable history. Runs at
        startup and is idempotent (a matching active hash is a no-op).

        Once users can author definitions, reserved builtin slugs stay
        forbidden to user creation, preserving this "builtins track host" rule
        without ever clobbering a user-authored tool."""
        for definition in builtin_definitions(self._model_authoring()):
            self._require_profiles(definition)
            active = await self._db.get_active_tool_definition(definition.slug)
            if active is None or active.content_hash != definition.content_hash:
                await self._db.upsert_tool_definition(definition, activate=True)

    async def get(self, slug: str) -> ToolDefinitionV1 | None:
        """The active definition for a slug, or None. Host-side executable truth."""
        return await self._db.get_active_tool_definition(slug)

    async def list_active(self) -> list[ToolDefinitionV1]:
        return await self._db.list_active_tool_definitions()

    async def catalog(self) -> list[dict[str, object]]:
        """The planner-facing view: identity + intent only, never capabilities."""
        return [
            {
                "slug": definition.slug,
                "name": definition.name,
                "description": definition.description,
                "intent_examples": definition.intent_examples,
            }
            for definition in await self.list_active()
        ]

    async def records(self) -> list[ToolDefinitionRecordV1]:
        """The registry-browser view: each live definition with its host-derived
        lifecycle state (runnable/buildable/disabled/pending build). Unlike the
        planner catalog, this DOES expose the full definition (capability profile,
        budgets) — it is host-side truth for the human operator, not the model."""
        definitions = await self._db.list_active_tool_definitions()
        build_index = await self._db.declarative_build_index()
        active_image = {row["slug"] for row in await self._db.list_active_tools()}
        disabled = set(getattr(self._settings, "tool_disabled_slugs", []) or [])
        records: list[ToolDefinitionRecordV1] = []
        for definition in definitions:
            is_architecture = definition.route_facts.input_pipeline == "architecture_spec"
            builds = build_index.get(definition.slug, {})
            runnable = (
                definition.slug in active_image if is_architecture else bool(builds.get("active"))
            )
            if is_architecture:
                buildable = not runnable
            else:
                buildable = await self._db.get_buildable_definition(definition.slug) is not None
            records.append(
                ToolDefinitionRecordV1(
                    definition=definition,
                    active=True,
                    runnable=runnable,
                    buildable=buildable,
                    disabled=definition.slug in disabled,
                    pending_build=bool(builds.get("evaluated")),
                )
            )
        return records
