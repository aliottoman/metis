"""Project selection is an explicit routing choice, even for research turns."""

from __future__ import annotations

import pytest

from chat_evidence_cases import PROJECT_CASES
from waqil_api.control_plane import ControlPlane
from waqil_api.evidence_routing import EvidencePlanV1


@pytest.mark.parametrize("case", PROJECT_CASES, ids=lambda case: case.id)
def test_selected_project_controls_project_route(case) -> None:
    plane = type("_Plane", (), {"projects": object()})()
    aliases = {"_project_id": "asset_1"} if case.selected_project else {}
    needs_public_docs = case.id == "selected_project_recent_doc_edit"
    state = {
        "prompt": case.prompt,
        "model_aliases": aliases,
        "evidence_plan": EvidencePlanV1(
            sources=["private", "web"] if needs_public_docs else ["private"],
            action=case.expected_action,
        ).model_dump(mode="json"),
        "knowledge_snippets": [{"provider": "web"}] if needs_public_docs else [],
    }
    route = ControlPlane._route_after_retrieve(plane, state)  # type: ignore[arg-type]
    assert route == case.expected_path.replace("chat", "plan")


def test_project_edit_waits_when_required_public_docs_are_unavailable() -> None:
    plane = type("_Plane", (), {"projects": object()})()
    state = {
        "prompt": "Check current Cline SDK docs, then update our integration.",
        "model_aliases": {"_project_id": "asset_1"},
        "evidence_plan": EvidencePlanV1(
            sources=["private", "web"], action="plan"
        ).model_dump(mode="json"),
        "knowledge_snippets": [],
    }
    assert ControlPlane._route_after_retrieve(plane, state) == "web_unavailable"  # type: ignore[arg-type]


def test_explicit_notion_scope_overrides_selected_project() -> None:
    plane = type("_Plane", (), {"projects": object()})()
    state = {
        "prompt": "What do my notes say about the release?",
        "model_aliases": {"_project_id": "asset_1", "_knowledge_scope": "notion"},
    }
    assert ControlPlane._route_after_retrieve(plane, state) == "plan"  # type: ignore[arg-type]
