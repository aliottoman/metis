from __future__ import annotations

import pytest
from pydantic import ValidationError

from waqil_api.config import Settings
from waqil_api.contracts import ArchitectureSpecV1, MessageCreateV1


def test_contracts_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        MessageCreateV1(content="hello", unexpected=True)


def test_message_knowledge_scope_is_explicit_and_bounded() -> None:
    assert MessageCreateV1(content="hello").knowledge_scope == "auto"
    assert MessageCreateV1(content="hello", knowledge_scope="notion").knowledge_scope == "notion"
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
