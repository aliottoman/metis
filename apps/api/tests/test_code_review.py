"""P11 — the optional Grok review of authored tool code.

Exercises the real ControlPlane._author_and_review / _review_authored_code with a
FAKE reviewer (no OCI). Pins the invariants: the host AST-gate validates whatever
code is used, so an improvement is accepted only if it ALSO passes the gate (the
reviewer can never widen capabilities); an unsafe verdict blocks the build; and
any reviewer error/unavailability is fail-soft (the AST-gated original is used).
"""
from __future__ import annotations

import types

import pytest

from waqil_api import tool_authoring
from waqil_api.contracts import ToolDefinitionDraftV1
from waqil_api.control_plane import AuthoredReviewRejected, ControlPlane

ORIGINAL = "def run(inputs, model):\n    return {'v': 1}\n"
IMPROVED_VALID = "def run(inputs, model):\n    return {'v': 2, 'better': True}\n"
IMPROVED_UNSAFE = "import os\n\n\ndef run(inputs, model):\n    return {'v': os.getcwd()}\n"


class _Events:
    def __init__(self) -> None:
        self.types: list[str] = []

    async def emit(self, run_id, conversation_id, event_type, payload=None, checkpoint_id=None):
        self.types.append(event_type)


class _FakeAuthor:
    async def author_tool_code(self, definition, *, model_aliases=None):
        return ORIGINAL


class _FakeReviewer:
    def __init__(self, result, *, available=True, boom=False) -> None:
        self._result = result
        self._available = available
        self._boom = boom

    def tool_review_available(self) -> bool:
        return self._available

    def grok_review(self, code, task):
        if self._boom:
            raise RuntimeError("oci unavailable")
        return self._result


def _definition():
    draft = ToolDefinitionDraftV1(name="Sentiment Tool", description="classify sentiment of text")
    return tool_authoring.harden_draft(draft, slug="sentiment-tool", max_broker_calls=4)


def _cp(reviewer) -> types.SimpleNamespace:
    cp = types.SimpleNamespace(
        model=_FakeAuthor(),
        reviewer=reviewer,
        events=_Events(),
        settings=types.SimpleNamespace(oci_grok_model="xai.grok-4.3"),
    )
    cp._stage = ControlPlane._stage.__get__(cp)
    cp._review_authored_code = ControlPlane._review_authored_code.__get__(cp)
    return cp


_STATE = {"run_id": "r", "conversation_id": "c", "model_aliases": {}}


@pytest.mark.asyncio
async def test_valid_improvement_is_applied() -> None:
    cp = _cp(_FakeReviewer({"safe": True, "reasons": ["tightened"], "improved_code": IMPROVED_VALID}))
    code, review = await ControlPlane._author_and_review(cp, _STATE, _definition())
    assert code.strip() == IMPROVED_VALID.strip()
    assert review["applied"] is True and review["reviewed"] is True
    assert "tool.code_reviewed" in cp.events.types


@pytest.mark.asyncio
async def test_improvement_that_fails_the_gate_is_rejected() -> None:
    # Grok's "improvement" adds `import os` → fails the AST profile → discarded,
    # the original AST-gated code is kept. The reviewer cannot widen capabilities.
    cp = _cp(_FakeReviewer({"safe": True, "reasons": [], "improved_code": IMPROVED_UNSAFE}))
    code, review = await ControlPlane._author_and_review(cp, _STATE, _definition())
    assert code.strip() == ORIGINAL.strip()
    assert review["applied"] is False


@pytest.mark.asyncio
async def test_unsafe_verdict_blocks_the_build() -> None:
    cp = _cp(_FakeReviewer({"safe": False, "reasons": ["exfiltrates data"], "improved_code": ""}))
    with pytest.raises(AuthoredReviewRejected, match="unsafe"):
        await ControlPlane._author_and_review(cp, _STATE, _definition())


@pytest.mark.asyncio
async def test_reviewer_error_is_fail_soft() -> None:
    cp = _cp(_FakeReviewer({}, boom=True))
    code, review = await ControlPlane._author_and_review(cp, _STATE, _definition())
    assert code.strip() == ORIGINAL.strip()
    assert review["reviewed"] is False  # skipped; AST-gate remains the control


@pytest.mark.asyncio
async def test_reviewer_unavailable_skips_review() -> None:
    cp = _cp(_FakeReviewer({"safe": True}, available=False))
    code, review = await ControlPlane._author_and_review(cp, _STATE, _definition())
    assert code.strip() == ORIGINAL.strip()
    assert review["reviewed"] is False


@pytest.mark.asyncio
async def test_no_reviewer_wired_skips_review() -> None:
    cp = _cp(None)
    code, review = await ControlPlane._author_and_review(cp, _STATE, _definition())
    assert code.strip() == ORIGINAL.strip()
    assert review["reviewed"] is False
