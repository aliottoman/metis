"""Model Broker — the one channel through which a tool may reach the model.

Tools never hold the model provider. When a tool's capability profile grants
``model_access``, the host hands it a ``ModelBroker`` scoped to a single run.
The broker enforces the framework of control:

- **Pinned prompts.** The system prompt is a template frozen in the approved
  tool definition. A tool supplies *parameters* to fill into the user turn; it
  can never author or mutate the system prompt, so it cannot repurpose the model.
- **Budgets.** ``max_calls_per_run`` is hard-enforced; exhaustion raises
  ``BrokerBudgetExceeded`` and the caller falls back to its deterministic path.
- **Allowed roles only.** A tool may only invoke the model roles its profile
  lists (e.g. ``coder``).
- **Audit.** Every call emits a durable ``run.broker_call`` event (template,
  role, prompt/params hashes, response size) — visible and replayable.

The broker returns the model's text as *data*. It never executes that text; a
tool that wants to run model-authored code must first pass it through a named
capability profile (see ``capability_profiles``), with the sandbox as the outer
wall. This host-side broker is the single enforcement point; an in-sandbox
stdio transport (a tool calling the model mid-execution) plugs into the same
object rather than reimplementing any of these controls.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .contracts import ModelAccessV1, ModelRequestV1, ModelResultV1


class ScriptedModel:
    """A network-free model that replays canned replies in call order.

    Used to keep the factory's build-time evaluation hermetic: a declarative
    tool's eval cases run through a real ``ModelBroker`` whose model is this
    scripted stand-in, so the pass/fail gate is deterministic and never touches a
    live model. When the script is exhausted it returns an empty reply (which the
    tool's deterministic fallback then handles), so evals can also probe the
    fallback path."""

    name = "scripted"

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self._index = 0

    async def generate(
        self, request: ModelRequestV1, *, model_aliases: dict[str, Any] | None = None
    ) -> ModelResultV1:
        if self._index < len(self._replies):
            content = self._replies[self._index]
            self._index += 1
        else:
            content = ""
        return ModelResultV1(model="scripted", content=content, fallback=True)


class BrokerError(RuntimeError):
    """A brokered model call violated the tool's declared model access."""


class BrokerBudgetExceeded(BrokerError):
    """The tool's per-run model-call budget is exhausted."""


def _sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class ModelBroker:
    """A per-run, budget-enforced, audited handle to the model for one tool."""

    def __init__(
        self,
        *,
        model: Any,
        access: ModelAccessV1,
        events: Any,
        run_id: str,
        conversation_id: str,
        tool_slug: str,
        model_aliases: dict[str, str] | None = None,
    ) -> None:
        self._model = model
        self._access = access
        self._events = events
        self._run_id = run_id
        self._conversation_id = conversation_id
        self._tool_slug = tool_slug
        self._model_aliases = model_aliases or {}
        self._calls_made = 0

    @property
    def enabled(self) -> bool:
        return self._access.enabled

    @property
    def calls_made(self) -> int:
        return self._calls_made

    @property
    def calls_remaining(self) -> int:
        return max(0, self._access.max_calls_per_run - self._calls_made)

    def _require(self, role: str, template_id: str) -> None:
        if not self._access.enabled:
            raise BrokerError(f"tool '{self._tool_slug}' has no model access")
        if role not in self._access.roles:
            raise BrokerError(
                f"role '{role}' is not permitted for tool '{self._tool_slug}'"
            )
        if template_id not in self._access.prompt_templates:
            raise BrokerError(f"unknown prompt template: {template_id}")
        if self._calls_made >= self._access.max_calls_per_run:
            raise BrokerBudgetExceeded(
                f"broker call budget exhausted "
                f"({self._access.max_calls_per_run}) for tool '{self._tool_slug}'"
            )

    async def call(
        self, *, template_id: str, role: str, params: dict[str, Any]
    ) -> str:
        """Run one brokered model call and return its text as data.

        Raises ``BrokerError`` (bad role/template) or ``BrokerBudgetExceeded``
        *before* spending the model call, so callers can fall back cleanly."""
        self._require(role, template_id)
        system_prompt = self._access.prompt_templates[template_id]
        user_prompt = json.dumps(params, ensure_ascii=False, sort_keys=True)
        # Count the call before invoking so a mid-flight failure still consumes
        # budget (a runaway tool cannot retry unbounded).
        self._calls_made += 1
        result = await self._model.generate(
            ModelRequestV1(
                role=role, system_prompt=system_prompt, user_prompt=user_prompt
            ),
            model_aliases=self._model_aliases,
        )
        content = result.content or ""
        await self._events.emit(
            self._run_id,
            self._conversation_id,
            "run.broker_call",
            {
                "tool": self._tool_slug,
                "template": template_id,
                "role": role,
                "prompt_sha": _sha16(system_prompt),
                "params_sha": _sha16(user_prompt),
                "response_chars": len(content),
                "call_index": self._calls_made,
                "budget": self._access.max_calls_per_run,
                "max_tokens_per_call": self._access.max_tokens_per_call,
                "model": result.model,
            },
        )
        return content
