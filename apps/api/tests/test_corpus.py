from __future__ import annotations

import hashlib
import re

import pytest

from waqil_api.config import Settings
from waqil_api.corpus import CorpusService
from waqil_api.database import Database

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class FakeRetrieval:
    """Deterministic, offline stand-in for CohereRetrieval.

    Embeddings are a stable bag-of-words hash so cosine recall is meaningful,
    and rerank scores by query/document token overlap — enough to assert that
    the right file and symbol come back without touching OCI.
    """

    def __init__(self, available: bool = True, dim: int = 32) -> None:
        self._available = available
        self._dim = dim
        self.embed_calls = 0

    def available(self) -> bool:
        return self._available

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        for token in _tokens(text):
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self._dim
            vector[bucket] += 1.0
        return vector

    def embed(self, texts, input_type: str = "search_document") -> list[list[float]]:
        self.embed_calls += 1
        return [self._vector(text) for text in texts]

    def rerank(self, query, documents, top_n=None):
        query_tokens = set(_tokens(query))
        scored = sorted(
            (
                (index, len(query_tokens & set(_tokens(document))))
                for index, document in enumerate(documents)
            ),
            key=lambda pair: -pair[1],
        )
        return [(index, float(score)) for index, score in scored[: (top_n or len(documents))]]

    def extract_entities(self, text: str):
        """Deterministic stand-in for Command A entity extraction: emits known
        entities when their token appears, plus a Metis->uses->Cohere relation."""
        from waqil_api.entity_graph import Entity, EntityExtraction, Relation

        entities, relations = [], []
        lowered = text.lower()
        if "metis" in lowered:
            entities.append(Entity("Metis", "project"))
        if "cohere" in lowered:
            entities.append(Entity("Cohere", "organization"))
        if len(entities) == 2:
            relations.append(Relation("Metis", "uses", "Cohere"))
        return EntityExtraction(entities=entities, relations=relations)


async def _corpus(tmp_path, available: bool = True, **settings_kwargs):
    database = Database(tmp_path / "waqil.db")
    await database.open()
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        allow_test_backends=True,
        **settings_kwargs,
    )
    return database, CorpusService(settings, database, FakeRetrieval(available=available))


# A tiny two-function module: charge_card() calls _settle(). The query below
# matches charge_card by tokens but shares none with _settle, so _settle can be
# reached only through the call graph — the exact case hybrid retrieval targets.
_SERVICE = (
    "def charge_card(amount):\n"
    "    '''charge the customer credit card via the payment gateway'''\n"
    "    return _settle(amount)\n"
    "\n"
    "def _settle(amount):\n"
    "    '''zzz internal ledger reconciliation zzz'''\n"
    "    return amount\n"
)


async def _indexed_service(tmp_path, **settings_kwargs):
    database, corpus = await _corpus(tmp_path, **settings_kwargs)
    root = tmp_path / "proj"
    root.mkdir()
    (root / "service.py").write_text(_SERVICE)
    source = await corpus.register_source(str(root), "proj", "code")
    await corpus.set_consent(source.id, True, "ok")
    await corpus.index_source(source.id)
    return database, corpus, source


@pytest.mark.asyncio
async def test_consent_gate_blocks_indexing(tmp_path) -> None:
    database, corpus = await _corpus(tmp_path)
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")
    source = await corpus.register_source(str(root), "proj", "code")
    with pytest.raises(PermissionError):
        await corpus.index_source(source.id)
    await database.close()


@pytest.mark.asyncio
async def test_register_source_tolerates_pasted_quotes_and_whitespace(tmp_path) -> None:
    database, corpus = await _corpus(tmp_path)
    # Each pasted form normalizes to the clean resolved path. A fresh directory
    # per form keeps them from colliding on the "already registered" guard —
    # which itself proves they all resolve to the same canonical string.
    forms = {
        "single_quoted": lambda p: f"'{p}'",
        "double_quoted": lambda p: f'"{p}"',
        "padded": lambda p: f"  {p}  ",
        "quoted_and_padded": lambda p: f"  '{p}'  ",
    }
    for name, wrap in forms.items():
        root = tmp_path / name
        root.mkdir()
        source = await corpus.register_source(wrap(str(root)), None, "code")
        assert source.root_path == str(root.resolve())
    await database.close()


@pytest.mark.asyncio
async def test_register_source_still_rejects_a_genuinely_missing_path(tmp_path) -> None:
    database, corpus = await _corpus(tmp_path)
    with pytest.raises(FileNotFoundError):
        await corpus.register_source(str(tmp_path / "nope"), None, "code")
    await database.close()


@pytest.mark.asyncio
async def test_index_then_retrieve_returns_cited_symbol(tmp_path) -> None:
    database, corpus = await _corpus(tmp_path)
    root = tmp_path / "proj"
    root.mkdir()
    (root / "auth.py").write_text(
        "def login(user):\n"
        "    '''authenticate the user and issue a session token'''\n"
        "    return True\n"
    )
    (root / "notes.md").write_text("# Billing\nStripe handles billing and invoices.\n")
    source = await corpus.register_source(str(root), "proj", "code")
    source = await corpus.set_consent(source.id, True, "ok")
    result = await corpus.index_source(source.id)
    assert result.status == "indexed"
    assert result.files_indexed == 2
    assert result.chunks >= 2

    snippets = await corpus.retrieve("how does login authenticate the user")
    assert snippets, "expected a retrieval hit"
    assert snippets[0].rel_path == "auth.py"
    assert snippets[0].symbol == "login"
    assert snippets[0].source_label == "proj"


@pytest.mark.asyncio
async def test_retrieve_degrades_to_cosine_when_rerank_is_unavailable(tmp_path) -> None:
    # Reproduces the Frankfurt case: embed works, but the rerank model is not
    # served in the region (a 404). Retrieval must still return similarity-ranked
    # results instead of failing the whole query.
    database, corpus = await _corpus(tmp_path)
    root = tmp_path / "proj"
    root.mkdir()
    (root / "auth.py").write_text(
        "def login(user):\n"
        "    '''authenticate the user and issue a session token'''\n"
        "    return True\n"
    )
    (root / "notes.md").write_text("# Billing\nStripe handles billing and invoices.\n")
    source = await corpus.register_source(str(root), "proj", "code")
    await corpus.set_consent(source.id, True, "ok")
    await corpus.index_source(source.id)

    def _rerank_404(*_args, **_kwargs):
        raise RuntimeError("Entity with key cohere.rerank-v3.5 not found")

    corpus._retrieval.rerank = _rerank_404  # simulate the regional 404

    snippets = await corpus.retrieve("how does login authenticate the user")
    assert snippets, "retrieval must degrade gracefully, not fail, without rerank"
    assert snippets[0].rel_path == "auth.py"
    # Scores now come from cosine similarity, so they stay within a sane [0, 1].
    assert all(0.0 <= snippet.score <= 1.0001 for snippet in snippets)
    await database.close()


@pytest.mark.asyncio
async def test_incremental_reindex_only_touches_changed_files(tmp_path) -> None:
    database, corpus = await _corpus(tmp_path)
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("def a():\n    return 1\n")
    (root / "b.py").write_text("def b():\n    return 2\n")
    source = await corpus.register_source(str(root), "proj", "code")
    await corpus.set_consent(source.id, True, "ok")
    await corpus.index_source(source.id)

    fake: FakeRetrieval = corpus._retrieval  # type: ignore[assignment]
    calls_before = fake.embed_calls
    (root / "b.py").write_text("def b():\n    return 22\n")  # change one file
    result = await corpus.index_source(source.id)
    assert result.files_indexed == 1
    assert result.files_skipped == 1
    # Exactly one changed file was re-embedded this pass.
    assert fake.embed_calls == calls_before + 1

    (root / "a.py").unlink()  # removed file is dropped
    result = await corpus.index_source(source.id)
    assert result.files_removed == 1
    await database.close()


@pytest.mark.asyncio
async def test_revoking_consent_purges_local_vectors(tmp_path) -> None:
    database, corpus = await _corpus(tmp_path)
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("def a():\n    return 1\n")
    source = await corpus.register_source(str(root), "proj", "code")
    await corpus.set_consent(source.id, True, "ok")
    await corpus.index_source(source.id)
    assert await corpus.retrieve("return") != []

    revoked = await corpus.set_consent(source.id, False, "no more")
    assert revoked.status == "revoked"
    assert revoked.chunk_count == 0
    assert await corpus.retrieve("return") == []
    await database.close()


@pytest.mark.asyncio
async def test_retrieve_is_empty_when_unavailable(tmp_path) -> None:
    database, corpus = await _corpus(tmp_path, available=False)
    assert corpus.available() is False
    assert await corpus.retrieve("anything") == []
    await database.close()


@pytest.mark.asyncio
async def test_retrieve_can_be_restricted_to_notion_provider(tmp_path) -> None:
    database, corpus = await _corpus(tmp_path)
    local_root = tmp_path / "local"
    notion_root = tmp_path / "notion"
    local_root.mkdir()
    notion_root.mkdir()
    (local_root / "decision.md").write_text("The launch colour is red.\n")
    (notion_root / "decision.md").write_text("The launch colour is violet.\n")
    local = await corpus.register_source(str(local_root), "Local notes", "notes")
    notion = await corpus.register_source(
        str(notion_root), "Notion", "notes", provider="notion"
    )
    await corpus.set_consent(local.id, True, "ok")
    await corpus.set_consent(notion.id, True, "ok")
    await corpus.index_source(local.id)
    await corpus.index_source(notion.id)

    snippets = await corpus.retrieve("What is the launch colour?", provider="notion")
    assert snippets
    assert {snippet.provider for snippet in snippets} == {"notion"}
    assert all(snippet.source_label == "Notion" for snippet in snippets)
    assert any("violet" in snippet.text for snippet in snippets)
    await database.close()


# ── P4: code graph + hybrid retrieve ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_indexing_builds_the_code_graph(tmp_path) -> None:
    database, corpus, _ = await _indexed_service(tmp_path)
    stats = await corpus.graph_stats()
    assert stats.node_count > 0
    assert stats.nodes_by_kind.get("function", 0) >= 2
    assert stats.edges_by_kind.get("calls", 0) >= 1
    await database.close()


@pytest.mark.asyncio
async def test_graph_lookup_resolves_callers_and_callees(tmp_path) -> None:
    database, corpus, _ = await _indexed_service(tmp_path)

    charge = await corpus.graph_lookup("charge_card")
    assert any(d.qualname.endswith("charge_card") for d in charge.definitions)
    assert any(callee.dst_name == "_settle" for callee in charge.callees)

    settle = await corpus.graph_lookup("_settle")
    assert any(caller.caller.endswith("charge_card") for caller in settle.callers)
    await database.close()


@pytest.mark.asyncio
async def test_hybrid_expansion_pulls_in_call_graph_neighbour(tmp_path) -> None:
    database, corpus, _ = await _indexed_service(tmp_path)
    # A recall hit on charge_card alone; _settle shares no query tokens and would
    # never be recalled or reranked on its own.
    seed = [{"id": "seed-not-a-real-chunk", "symbol": "charge_card"}]
    expanded = await corpus._graph_expand(seed)
    symbols = {item.get("symbol") for item in expanded}
    assert "_settle" in symbols, "callee chunk should be pulled in by graph expansion"
    await database.close()


@pytest.mark.asyncio
async def test_graph_expansion_flag_disables_neighbour_pull(tmp_path) -> None:
    database, corpus, _ = await _indexed_service(tmp_path, corpus_graph_expand=False)
    seed = [{"id": "seed", "symbol": "charge_card"}]
    expanded = await corpus._graph_expand(seed)
    assert expanded == seed  # no expansion when the flag is off
    await database.close()


@pytest.mark.asyncio
async def test_revoking_consent_purges_the_code_graph(tmp_path) -> None:
    database, corpus, source = await _indexed_service(tmp_path)
    assert (await corpus.graph_stats()).node_count > 0
    await corpus.set_consent(source.id, False, "revoked")
    stats = await corpus.graph_stats()
    assert stats.node_count == 0
    assert stats.edge_count == 0
    await database.close()


@pytest.mark.asyncio
async def test_graph_updates_incrementally_when_a_call_is_removed(tmp_path) -> None:
    database, corpus, source = await _indexed_service(tmp_path)
    assert any(
        c.dst_name == "_settle" for c in (await corpus.graph_lookup("charge_card")).callees
    )
    # Rewrite the file so charge_card no longer calls _settle.
    (tmp_path / "proj" / "service.py").write_text(
        "def charge_card(amount):\n    return amount\n"
    )
    await corpus.index_source(source.id)
    callees = (await corpus.graph_lookup("charge_card")).callees
    assert not any(c.dst_name == "_settle" for c in callees)
    await database.close()


# ── P6b: entity graph (Graph-RAG Stage 2, opt-in) ────────────────────────────


async def _indexed_notes(tmp_path, **settings_kwargs):
    database, corpus = await _corpus(tmp_path, **settings_kwargs)
    root = tmp_path / "kb"
    root.mkdir()
    (root / "stack.md").write_text("# Stack\nMetis uses Cohere for retrieval.\n")
    (root / "impl.py").write_text("# Metis and Cohere\ndef go():\n    return 1\n")
    source = await corpus.register_source(str(root), "kb", "mixed")
    await corpus.set_consent(source.id, True, "ok")
    await corpus.index_source(source.id)
    return database, corpus, source


@pytest.mark.asyncio
async def test_entity_graph_is_extracted_from_prose_when_enabled(tmp_path) -> None:
    database, corpus, _ = await _indexed_notes(tmp_path, corpus_entity_graph=True)
    stats = await corpus.entity_stats()
    assert stats.node_count >= 2
    assert stats.nodes_by_kind.get("project", 0) >= 1

    lookup = await corpus.entity_lookup("Cohere")
    assert "organization" in lookup.kinds
    # Metis --uses--> Cohere, so Cohere has an inbound relation from Metis.
    assert any(
        rel.src_name == "Metis" and rel.relation == "uses" for rel in lookup.relations_in
    )
    await database.close()


@pytest.mark.asyncio
async def test_entity_graph_is_off_by_default(tmp_path) -> None:
    database, corpus, _ = await _indexed_notes(tmp_path)  # flag defaults False
    assert (await corpus.entity_stats()).node_count == 0
    await database.close()


@pytest.mark.asyncio
async def test_entities_are_not_extracted_from_code_files(tmp_path) -> None:
    # impl.py mentions Metis and Cohere in a comment, but code files are covered
    # by the deterministic code graph, so no entity extraction runs on them.
    database, corpus, _ = await _indexed_notes(tmp_path, corpus_entity_graph=True)
    lookup = await corpus.entity_lookup("Metis")
    assert all(rel.rel_path == "stack.md" for rel in lookup.relations_out)
    await database.close()


@pytest.mark.asyncio
async def test_revoking_consent_purges_the_entity_graph(tmp_path) -> None:
    database, corpus, source = await _indexed_notes(tmp_path, corpus_entity_graph=True)
    assert (await corpus.entity_stats()).node_count > 0
    await corpus.set_consent(source.id, False, "revoked")
    assert (await corpus.entity_stats()).node_count == 0
    await database.close()
