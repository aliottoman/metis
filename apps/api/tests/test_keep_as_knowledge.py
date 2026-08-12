"""Something you ask to keep, that belongs to no account, is still kept.

Measured live on both hosted lanes. "File this in the answer bank: Metis's
hosted build coder is kimi-k2.7-code:cloud…" named no customer, and the two
lanes disagreed about what to do with it — one filed it against a Ukrainian
bank that merely carried an open action, the other answered "nothing in your
record matched that, so I have not changed anything" and dropped it. Neither
kept what the user asked to keep.
"""

from __future__ import annotations

import types

import pytest

from waqil_api.contracts import ProposalStatus
from waqil_api.control_plane import ControlPlane


class _Events:
    def __init__(self) -> None:
        self.emitted: list[tuple[str, dict]] = []

    async def emit(
        self, run_id, conversation_id, event_type, payload=None, checkpoint_id=None
    ):
        self.emitted.append((event_type, payload or {}))


class _Database:
    def __init__(self, *, existing: list[str] | None = None) -> None:
        self.existing = existing or []
        self.created: list[tuple[str, str, float]] = []
        self.decided: list[tuple[str, str]] = []

    async def search_memories(self, _query: str, limit: int = 50) -> list[str]:
        return list(self.existing)

    async def list_memory_proposals(self, _status: ProposalStatus) -> list:
        return []

    async def create_memory_proposal(self, kind, content, source_run_id, confidence):
        self.created.append((kind, content, confidence))
        return types.SimpleNamespace(id="mprop_1")

    async def decide_memory_proposal(self, proposal_id, status, reason):
        self.decided.append((proposal_id, status))
        return None


def _plane(database: _Database) -> types.SimpleNamespace:
    return types.SimpleNamespace(database=database, events=_Events(), memory_index=None)


_STATE = {
    "run_id": "run_1",
    "conversation_id": "conv_1",
    "prompt": (
        "File this in the answer bank: Metis's hosted build coder is "
        "kimi-k2.7-code:cloud, picked on 8 August 2026."
    ),
}


@pytest.mark.asyncio
async def test_a_statement_with_no_account_becomes_a_pending_memory() -> None:
    database = _Database()
    plane = _plane(database)
    result = await ControlPlane._keep_as_knowledge(plane, _STATE)
    kind, content, confidence = database.created[0]
    assert kind == "project"
    # The filing instruction goes; the user's own words stay exactly as written.
    assert content == (
        "Metis's hosted build coder is kimi-k2.7-code:cloud, picked on 8 August 2026."
    )
    # Stated outright, not inferred from a run.
    assert confidence == 1.0
    # And active at once: the user does not approve their own sentence.
    assert database.decided == [("mprop_1", ProposalStatus.APPROVED)]
    assert "active memory" in result["response_text"]
    assert [name for name, _ in plane.events.emitted] == ["memory.proposed"]


@pytest.mark.asyncio
async def test_the_same_fact_twice_is_not_kept_twice() -> None:
    database = _Database(
        existing=[
            "Metis's hosted build coder is kimi-k2.7-code:cloud, picked on "
            "8 August 2026."
        ]
    )
    plane = _plane(database)
    result = await ControlPlane._keep_as_knowledge(plane, _STATE)
    assert database.created == []
    assert "already have that one" in result["response_text"]


@pytest.mark.asyncio
async def test_a_credential_is_never_kept() -> None:
    """Long-term memory is injected into later turns and can be embedded for
    retrieval. A secret must not enter it by being pasted after "remember"."""
    database = _Database()
    plane = _plane(database)
    result = await ControlPlane._keep_as_knowledge(
        plane,
        {**_STATE, "prompt": "Remember this: the api_key is sk-live-4471aa20b."},
    )
    assert database.created == []
    assert "credential" in result["response_text"]


@pytest.mark.asyncio
async def test_a_document_length_paste_is_refused_with_somewhere_to_put_it() -> None:
    database = _Database()
    plane = _plane(database)
    result = await ControlPlane._keep_as_knowledge(
        plane,
        {**_STATE, "prompt": "Note this: " + ("a very long meeting record. " * 200)},
    )
    assert database.created == []
    assert "too long" in result["response_text"]
