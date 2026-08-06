"""Editing a message you already sent.

The edit only means anything if the turns that followed leave the model's
view — otherwise the original question is still in context and the "edit" is
just a second question. These pin the two halves of that: what a rewind
removes from context, and what it deliberately leaves on disk.
"""

from __future__ import annotations

import sqlite3

import pytest

from waqil_api.database import Database


async def _thread(path) -> tuple[Database, str, list[str]]:
    """A conversation with three turns: two asked, two answered, one asked."""
    database = Database(path)
    await database.open()
    conversation = await database.create_conversation("Sizing review")
    ids: list[str] = []
    for role, content in (
        ("user", "How many OCI nodes?"),
        ("assistant", "Twelve, at the stated throughput."),
        ("user", "And with the Emirates uplift?"),
        ("assistant", "Sixteen."),
        ("user", "Write that up."),
    ):
        message = await database.add_message(conversation.id, role, content)
        ids.append(message.id)
    return database, conversation.id, ids


@pytest.mark.asyncio
async def test_a_rewind_hides_the_message_and_everything_after_it(tmp_path) -> None:
    database, conversation_id, ids = await _thread(tmp_path / "rewind.db")
    try:
        retired = await database.supersede_messages_from(conversation_id, ids[2])
        assert retired == 3  # the third turn and both messages after it

        live = [message.content for message in await database.list_messages(conversation_id)]
        assert live == ["How many OCI nodes?", "Twelve, at the stated throughput."]

        # The model's own window, which is the one that actually matters: a
        # rewound turn must not come back through the recent-message context.
        recent = await database.recent_messages(conversation_id)
        assert all("Emirates" not in item["content"] for item in recent)
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_a_rewind_is_idempotent_and_keeps_the_rows(tmp_path) -> None:
    database, conversation_id, ids = await _thread(tmp_path / "twice.db")
    try:
        assert await database.supersede_messages_from(conversation_id, ids[2]) == 3
        # Rewinding to the same point again retires nothing further, rather
        # than re-stamping rows that are already retired.
        assert await database.supersede_messages_from(conversation_id, ids[2]) == 0

        # Marked, not deleted: runs and memory proposals point at these rows
        # without a cascade, so losing them would either fail a foreign key or
        # take governed records with it.
        connection = sqlite3.connect(tmp_path / "twice.db")
        stored = connection.execute(
            "SELECT count(*) FROM messages WHERE conversation_id = ?", (conversation_id,)
        ).fetchone()[0]
        connection.close()
        assert stored == 5
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_a_rewind_drops_the_rolling_summary(tmp_path) -> None:
    """The summary was written over text that is no longer in the thread."""
    database, conversation_id, ids = await _thread(tmp_path / "summary.db")
    try:
        assert "Emirates" in await database.refresh_conversation_summary(conversation_id)
        await database.supersede_messages_from(conversation_id, ids[2])
        # Cleared outright rather than left to go stale: whatever is written
        # next is built from the live messages only.
        assert await database.get_conversation_summary(conversation_id) == ""
        assert "Emirates" not in await database.refresh_conversation_summary(conversation_id)
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_rewinding_an_unknown_message_is_refused(tmp_path) -> None:
    database, conversation_id, _ = await _thread(tmp_path / "missing.db")
    try:
        with pytest.raises(LookupError):
            await database.supersede_messages_from(conversation_id, "msg_does_not_exist")
    finally:
        await database.close()
