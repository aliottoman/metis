"""P9 — the safe tool-authoring menu (the draft → definition trust boundary).

The model may propose display text; the host assigns every capability from a
reviewed archetype. These tests pin that boundary: a matching draft becomes a
bounded, content-hashed definition; a non-matching or hostile draft is refused;
budgets can never exceed the global ceiling; identical drafts dedup while
revisions diverge.
"""
from __future__ import annotations

import pytest

from waqil_api import capability_profiles, tool_authoring
from waqil_api.contracts import ToolDefinitionDraftV1
from waqil_api.tool_authoring import ToolAuthoringError
from waqil_api.tool_registry import REFERENCE_ARCHITECTURE_SLUG, definition_hash


def _draft(**overrides) -> ToolDefinitionDraftV1:
    base = dict(
        name="Readme Summary",
        description="summarize a project readme into a card",
        intent="Summarize this README",
        requested_capabilities=["summarize text"],
    )
    base.update(overrides)
    return ToolDefinitionDraftV1(**base)


def test_text_summary_draft_hardens_to_declarative_definition() -> None:
    definition = tool_authoring.harden_draft(_draft(), slug="readme-summary", max_broker_calls=4)
    assert definition.slug == "readme-summary"
    assert definition.archetype == "text-summary"
    assert definition.status == "proposed"
    # Host-assigned capability profile: declarative (no code exec), one broker call.
    profile = definition.capability_profile
    assert profile.code_allowlist == "declarative-host-v1"
    assert profile.runtime_allowlists == {}
    assert profile.network == "none"
    assert profile.model_access.enabled is True
    assert profile.model_access.roles == ["reviewer"]
    assert profile.model_access.max_calls_per_run == 1
    assert "summarize" in profile.model_access.prompt_templates
    # Content-hashed and self-consistent.
    assert definition.content_hash and definition_hash(definition) == definition.content_hash
    # The referenced code profile is a registered host profile.
    assert capability_profiles.exists(profile.code_allowlist)


def test_model_never_sets_capabilities_via_requested_capabilities() -> None:
    # A draft asking for network/secrets/shell cannot widen the profile: the host
    # ignores requested_capabilities entirely and uses the archetype's profile.
    hostile = _draft(
        requested_capabilities=[
            "summarize text",
            "network access",
            "read secrets",
            "run shell commands",
            "execute arbitrary python",
        ]
    )
    definition = tool_authoring.harden_draft(hostile, slug="readme-summary", max_broker_calls=4)
    assert definition.capability_profile.network == "none"
    assert definition.capability_profile.runtime_allowlists == {}
    assert definition.capability_profile.model_access.max_calls_per_run == 1


def test_unmatched_request_falls_back_to_code_authoring() -> None:
    # No specific archetype matches → the general code-authoring archetype (the
    # model writes AST-gated run() code). Capability stays bounded: no network,
    # authored-code profile, model access via the injected bridge only.
    draft = ToolDefinitionDraftV1(name="Ticket Classifier", description="classify support tickets by topic")
    definition = tool_authoring.harden_draft(draft, slug="ticket-classifier", max_broker_calls=4)
    assert definition.archetype == "code-authoring"
    assert definition.capability_profile.code_allowlist == "pure-python-authored-v1"
    assert definition.capability_profile.network == "none"
    assert definition.author_system_prompt  # the pinned authoring prompt is carried
    assert definition.capability_profile.model_access.enabled is True


def test_reserved_builtin_slug_is_refused() -> None:
    with pytest.raises(ToolAuthoringError, match="reserved"):
        tool_authoring.harden_draft(_draft(), slug=REFERENCE_ARCHITECTURE_SLUG, max_broker_calls=4)


def test_budget_over_global_ceiling_fails_closed() -> None:
    # The text-summary archetype wants 1 broker call; a ceiling of 0 must refuse it.
    with pytest.raises(ToolAuthoringError, match="global ceiling"):
        tool_authoring.harden_draft(_draft(), slug="readme-summary", max_broker_calls=0)


def test_identical_drafts_are_deduped_by_derived_version() -> None:
    a = tool_authoring.harden_draft(_draft(), slug="readme-summary", max_broker_calls=4)
    b = tool_authoring.harden_draft(_draft(), slug="readme-summary", max_broker_calls=4)
    assert a.version == b.version
    assert a.content_hash == b.content_hash


def test_revised_draft_gets_a_new_version() -> None:
    a = tool_authoring.harden_draft(_draft(), slug="readme-summary", max_broker_calls=4)
    b = tool_authoring.harden_draft(
        _draft(description="summarize a readme, but also list the license"),
        slug="readme-summary",
        max_broker_calls=4,
    )
    assert a.version != b.version
    assert a.content_hash != b.content_hash


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Readme Summary", "readme-summary"),
        ("  Project  Summary!!  Card ", "project-summary-card"),
        ("###", "tool"),  # degenerate → fallback
        ("A" * 80, "a" * 48),  # bounded length
    ],
)
def test_slugify(name: str, expected: str) -> None:
    assert tool_authoring.slugify(name) == expected


def test_draft_coerces_nonstring_fields_from_the_model() -> None:
    # Local models sometimes return a JSON schema object where a string sketch was
    # asked for. The draft must coerce (not crash), and still harden.
    draft = ToolDefinitionDraftV1.model_validate(
        {
            "name": "meeting_notes_action_tracker",
            "description": "Extract structured action items from meeting notes.",
            "intent": "Turn notes into a tracker.",
            "requested_capabilities": ["parse text", {"x": 1}],
            "input_sketch": {"type": "object", "properties": {"notes": {"type": "string"}}},
            "output_sketch": {"type": "object", "properties": {"items": {"type": "array"}}},
        }
    )
    assert isinstance(draft.input_sketch, str) and isinstance(draft.output_sketch, str)
    assert draft.requested_capabilities == ["parse text", '{"x": 1}']
    definition = tool_authoring.harden_draft(
        draft, slug=tool_authoring.slugify(draft.name), max_broker_calls=4
    )
    assert definition.archetype == "code-authoring"
    assert definition.slug == "meeting-notes-action-tracker"


def test_archetype_menu_is_registered_and_named() -> None:
    assert "text-summary" in tool_authoring.archetype_names()
    archetype = tool_authoring.get_archetype("text-summary")
    assert archetype is not None
    # Every archetype's referenced code profile must be a registered host profile.
    assert capability_profiles.exists(archetype.capability_profile.code_allowlist)
    # It ships hermetic eval fixtures (host-owned scripted broker replies).
    assert archetype.eval_fixtures
