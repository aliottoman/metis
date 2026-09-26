"""The faster chat seat must not change planning or a user's model choice."""

from __future__ import annotations

import copy
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from waqil_api.contracts import ModelResultV1, PlanEnvelopeV1, RoleChainEntryV1
from waqil_api.control_plane import ControlPlane
from waqil_api.evidence_routing import EvidencePlanV1, plan_evidence
from waqil_api.model_preference import ModelPreferenceStore
from waqil_api.model_provider import ClineModelProvider, ModelProviderError
from waqil_api.policy import PolicyEngine


FLASH = "cline-pass/mimo-v2.6-flash"
QWEN = "cline-pass/qwen3.7-plus"


class _Events:
    def __init__(self) -> None:
        self.items: list[tuple[str, dict[str, Any]]] = []

    async def emit(self, _run_id, _conversation_id, kind, payload) -> None:
        self.items.append((kind, payload))

    def payloads(self, kind: str) -> list[dict[str, Any]]:
        return [payload for event, payload in self.items if event == kind]


class _Model:
    def __init__(self, settings, *, flash_failure: str = "") -> None:
        self.provider = ClineModelProvider(settings)
        self.flash_failure = flash_failure
        self.calls: list[tuple[str, str]] = []

    def _selected(self, role: str, aliases: dict[str, str]) -> str:
        return self.provider._model_for(role, aliases)

    async def _structured(self, schema, **kwargs):
        assert schema is EvidencePlanV1
        self.calls.append(
            ("evidence", self._selected(kwargs["role"], kwargs["model_aliases"]))
        )
        return EvidencePlanV1(sources=[], action="answer")

    async def project_spec(self, _request, *, model_aliases=None):
        selected = self._selected("planner", model_aliases)
        self.calls.append(("project", selected))
        return {"selected": selected}

    async def generate(
        self, request, on_token=None, *, model_aliases=None, on_reasoning=None
    ):
        del on_reasoning
        selected = self._selected(request.role, model_aliases)
        self.calls.append(("answer", selected))
        if selected == FLASH and self.flash_failure:
            if self.flash_failure == "after_delta" and on_token is not None:
                await on_token("partial answer")
            raise ModelProviderError("synthetic Flash failure")
        content = f"Answered by {selected}."
        if on_token is not None:
            await on_token(content)
        return ModelResultV1(model=selected, content=content)


def _plane(settings, model: _Model) -> ControlPlane:
    plane = object.__new__(ControlPlane)
    plane.settings = settings
    plane.model = model
    plane.events = _Events()
    plane.database = type(
        "_Database", (), {"is_cancel_requested": AsyncMock(return_value=False)}
    )()
    plane.policy = PolicyEngine()
    return plane


def _state(aliases: dict[str, str]) -> dict[str, Any]:
    return {
        "run_id": "synthetic_run",
        "conversation_id": "synthetic_conversation",
        "prompt": "Explain Python lists.",
        "model_aliases": aliases,
        "plan": PlanEnvelopeV1(summary="Answer", route="direct").model_dump(
            mode="json"
        ),
        "evidence_plan": EvidencePlanV1(sources=[], action="answer").model_dump(
            mode="json"
        ),
    }


def _cline_store(settings) -> ModelPreferenceStore:
    settings.cline_api_key = "synthetic-subscription-key"
    return ModelPreferenceStore(settings)


@pytest.mark.asyncio
async def test_split_chat_uses_flash_while_evidence_and_project_planning_use_qwen(
    settings,
) -> None:
    store = _cline_store(settings)
    store.save("split", None, provider="cline")
    aliases = store.resolve_aliases()
    original = copy.deepcopy(aliases)
    model = _Model(settings)
    plane = _plane(settings, model)

    evidence, method = await plan_evidence(
        model,
        prompt="Explain Python lists.",
        recent_messages=[],
        conversation_summary="",
        has_attachment=False,
        has_project=False,
        has_customer=False,
        model_aliases=aliases,
    )
    state = _state(aliases)
    state["evidence_plan"] = evidence.model_dump(mode="json")
    project = await plane._project_planner_call(state, "project_spec", {})
    answer = await plane._synthesize(state)

    assert method == "semantic"
    assert project == {"selected": QWEN}
    assert model.calls == [
        ("evidence", QWEN),
        ("project", QWEN),
        ("answer", FLASH),
    ]
    assert answer["response_text"] == f"Answered by {FLASH}."
    assert aliases == original  # The answer override is confined to its call.


@pytest.mark.asyncio
@pytest.mark.parametrize("choice", ["pinned", "planner_chain"])
async def test_user_model_choices_are_not_replaced_by_chat_default(
    settings, choice: str
) -> None:
    store = _cline_store(settings)
    if choice == "pinned":
        selected = "cline-pass/mimo-v2.5"
        store.save("pinned", selected, provider="cline")
    else:
        selected = "cline-pass/qwen3.7-max"
        store.save(
            "split",
            None,
            provider="cline",
            role_chains={
                "planner": [RoleChainEntryV1(provider="cline", model=selected)]
            },
        )
    aliases = store.resolve_aliases()
    assert "_cline_chat_model" not in aliases
    assert json.loads(aliases["_chain_planner"])[0]["model"] == selected
    model = _Model(settings)
    plane = _plane(settings, model)
    state = _state(aliases)

    await plane._project_planner_call(state, "project_spec", {})
    await plane._synthesize(state)

    assert model.calls[0] == ("project", selected)
    assert model.calls[1][0] == "answer"
    assert model.calls[1][1] != FLASH
    if choice == "pinned":
        assert model.calls[1][1] == selected


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["before_delta", "after_delta"])
async def test_flash_falls_back_to_qwen_only_before_answer_text(
    settings, failure: str
) -> None:
    store = _cline_store(settings)
    store.save("split", None, provider="cline")
    aliases = store.resolve_aliases()
    model = _Model(settings, flash_failure=failure)
    plane = _plane(settings, model)
    state = _state(aliases)

    if failure == "after_delta":
        with pytest.raises(ModelProviderError, match="synthetic Flash failure"):
            await plane._synthesize(state)
        assert model.calls == [("answer", FLASH)]
        assert plane.events.payloads("message.delta") == [{"delta": "partial answer"}]
        assert plane.events.payloads("model.response") == []
    else:
        answer = await plane._synthesize(state)
        assert model.calls == [("answer", FLASH), ("answer", QWEN)]
        assert answer["response_text"] == f"Answered by {QWEN}."
        assert plane.events.payloads("message.delta") == [
            {"delta": f"Answered by {QWEN}."}
        ]
        assert plane.events.payloads("model.response")[0]["fallback"] is True
    assert "_cline_model" not in aliases
