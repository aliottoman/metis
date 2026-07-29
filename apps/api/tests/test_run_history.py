"""Behavioural tests for run history and durable-memory harvesting."""
from __future__ import annotations

import pytest

from waqil_api.config import Settings
from waqil_api.corpus import CorpusService
from waqil_api.database import Database
from waqil_api.run_history import RESERVED_LABEL, RunHistoryService, changes_from_trace

from test_corpus import FakeRetrieval


async def _history(tmp_path, available: bool = True, **settings_kwargs):
    database = Database(tmp_path / "waqil.db")
    await database.open()
    settings = Settings(
        _env_file=None, data_dir=tmp_path / "data", repo_root=tmp_path, **settings_kwargs
    )
    corpus = CorpusService(settings, database, FakeRetrieval(available=available))
    return RunHistoryService(settings, database, corpus), database, corpus


@pytest.mark.asyncio
async def test_a_completed_run_becomes_a_findable_document(tmp_path) -> None:
    history, _, _ = await _history(tmp_path)

    path = await history.record(
        run_id="run_abc123",
        conversation_id="conv_1",
        prompt="Why does the importer drop the last row?",
        response="The reader used an exclusive bound; it now reads to len(rows).",
        changes=[{"tool": "apply_patch", "path": "src/importer.py"}],
        artifacts=[{"name": "importer-trace.txt"}],
        project_name="Data tools",
    )
    assert path is not None and path.is_file()
    text = path.read_text(encoding="utf-8")

    # The heading carries the request, because Markdown chunking uses headings
    # as the breadcrumb for every passage cut out of this document.
    assert text.startswith("# Why does the importer drop the last row?")
    assert "exclusive bound" in text
    assert "- Edited `src/importer.py`" in text
    assert "importer-trace.txt" in text
    assert "Data tools" in text


@pytest.mark.asyncio
async def test_only_approved_and_applied_changes_are_recorded(tmp_path) -> None:
    trace = [
        {"tool": "read_file", "result": {"ok": True, "output": {"path": "a.py"}}},
        {
            "tool": "apply_patch",
            "result": {"ok": True, "approved": True, "output": {"path": "applied.py"}},
        },
        {
            "tool": "create_file",
            "result": {"ok": False, "approved": False, "error": "declined"},
            "arguments": {"path": "rejected.py"},
        },
        {
            "tool": "apply_patch",
            "result": {"ok": False, "approved": True, "error": "context drifted"},
            "arguments": {"path": "failed.py"},
        },
    ]

    changes = changes_from_trace(trace)

    # A record that listed declined or failed edits would describe work that
    # never happened, which is worse than recording nothing.
    assert changes == [{"tool": "apply_patch", "path": "applied.py"}]


@pytest.mark.asyncio
async def test_the_document_never_contains_file_contents(tmp_path) -> None:
    history, _, _ = await _history(tmp_path)

    path = await history.record(
        run_id="run_secretive",
        conversation_id="conv_1",
        prompt="Rotate the client credentials helper.",
        response="Updated the helper to read from the environment.",
        changes=[{"tool": "apply_patch", "path": "src/auth.py"}],
    )
    assert path is not None
    text = path.read_text(encoding="utf-8")

    # Consent to index one's own notes must not become consent to index the
    # contents of every file an agent happened to touch.
    assert "src/auth.py" in text
    assert "original" not in text and "replacement" not in text


@pytest.mark.asyncio
async def test_history_is_registered_unconsented_and_indexes_nothing(tmp_path) -> None:
    history, _, corpus = await _history(tmp_path)
    await history.record(
        run_id="run_1",
        conversation_id="conv_1",
        prompt="A question worth remembering.",
        response="An answer worth finding later.",
    )

    source = await history.ensure_source()
    assert source is not None and source.label == RESERVED_LABEL
    # Visible in Knowledge with its switch off: enabling history stays a choice.
    assert source.consent is False

    await history.index()
    refreshed = await corpus.get_source(source.id)
    assert refreshed is not None and refreshed.chunk_count == 0


@pytest.mark.asyncio
async def test_consented_history_is_indexed_and_retrievable(tmp_path) -> None:
    history, _, corpus = await _history(tmp_path)
    await history.record(
        run_id="run_flaky",
        conversation_id="conv_1",
        prompt="Why did the nightly import job start failing?",
        response="A timezone change moved the cutoff; the window is now computed in UTC.",
    )
    source = await history.ensure_source()
    assert source is not None
    await corpus.set_consent(source.id, True, "indexed for recall")

    await history.index()

    refreshed = await corpus.get_source(source.id)
    assert refreshed is not None and refreshed.chunk_count > 0
    hits = await corpus.retrieve("nightly import job timezone cutoff")
    assert hits and "UTC" in hits[0].text


@pytest.mark.asyncio
async def test_registering_the_source_twice_is_idempotent(tmp_path) -> None:
    history, _, corpus = await _history(tmp_path)

    first = await history.ensure_source()
    second = await history.ensure_source()

    assert first is not None and second is not None
    assert first.id == second.id
    assert len(await corpus.list_sources()) == 1


@pytest.mark.asyncio
async def test_an_unfinished_run_is_not_recorded(tmp_path) -> None:
    history, _, _ = await _history(tmp_path)

    # A run with no answer is not history; writing it would put an empty
    # document into retrieval that can only ever dilute a later search.
    assert await history.record(
        run_id="run_x", conversation_id="c", prompt="ask", response="  "
    ) is None
    assert await history.record(
        run_id="../escape", conversation_id="c", prompt="ask", response="answer"
    ) is None


@pytest.mark.asyncio
async def test_history_can_be_switched_off_entirely(tmp_path) -> None:
    history, _, _ = await _history(tmp_path, run_history_enabled=False)

    assert await history.record(
        run_id="run_1", conversation_id="c", prompt="ask", response="answer"
    ) is None
    assert not history.root.exists()
