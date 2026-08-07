"""The answer bank: hybrid retrieval and the rules that keep it a memory.

Run history was already indexed and never served this purpose, so the test
that matters is not "can it retrieve" — it is that an atom is a reviewed unit
with citations, that lexical and semantic recall are fused rather than chosen
between, and that nothing enters unreviewed.
"""
from __future__ import annotations

import json

from waqil_api.answer_bank import AnswerBank, _cosine, _embed_text, _pack, _unpack
from waqil_api.config import Settings


class _FakeDatabase:
    def __init__(self, atoms: list[dict], lexical: list[tuple[str, float]]) -> None:
        self.atoms = {atom["id"]: atom for atom in atoms}
        self.lexical = lexical
        self.vectors: list[dict] = []

    async def search_answer_atoms_lexical(self, query: str, limit: int = 20):
        return self.lexical

    async def answer_atom_vectors(self):
        return self.vectors

    async def answer_atoms_by_id(self, ids: list[str]):
        return [self.atoms[key] for key in ids if key in self.atoms]

    async def atoms_missing_vectors(self):
        return [atom for atom in self.atoms.values() if atom["id"] not in
                {row["atom_id"] for row in self.vectors}]

    async def store_answer_atom_vector(self, atom_id, model, vector):
        self.vectors.append({"atom_id": atom_id, "vector": vector})


def _atom(atom_id: str, question: str, answer: str, citations=None) -> dict:
    return {
        "id": atom_id,
        "question": question,
        "paraphrases_json": json.dumps(["how long must I commit for?"]),
        "answer": answer,
        "citations_json": json.dumps(citations or []),
        "entities_json": json.dumps(["DAC"]),
        "status": "active",
    }


async def test_lexical_recall_alone_answers_when_there_is_no_cloud() -> None:
    """The half that handles exact identifiers must work with no embeddings at
    all — 744 and H100_X2 are precisely where dense retrieval is weakest."""
    atoms = [_atom("a1", "What is the DAC minimum commitment?", "744 unit-hours.")]
    bank = AnswerBank(Settings(), _FakeDatabase(atoms, [("a1", 1.0)]), retrieval=None)
    found = await bank.retrieve("744 unit-hours minimum", top_k=3)
    assert len(found) == 1
    assert found[0].provider == "answer"
    assert "744" in found[0].text
    assert found[0].score == 1.0


async def test_fusion_prefers_an_atom_both_rankers_found() -> None:
    """Reciprocal rank fusion, not score addition: a BM25 rank and a cosine
    similarity are not on a common scale, so only rank position is comparable.

    The property is agreement — an atom both rankers surfaced beats one only
    the lexical side found, even when the lexical side ranked it lower."""

    class _Retrieval:
        def available(self) -> bool:
            return True

        def embed(self, texts, input_type="search_document"):
            return [[1.0, 0.0] for _ in texts]

        def rerank(self, query, documents, top_n=None):
            # Identity, so the assertion measures fusion rather than reranking.
            return [(index, 1.0) for index in range(len(documents))]

    atoms = [
        _atom("lexonly", "found by keywords alone", "answer one"),
        _atom("both", "found by keywords and by meaning", "answer two"),
    ]
    database = _FakeDatabase(atoms, [("lexonly", 1.0), ("both", 2.0)])
    # Only "both" has been embedded, so only it can appear in dense recall.
    database.vectors = [{"atom_id": "both", "vector": _pack([1.0, 0.0])}]

    bank = AnswerBank(Settings(), database, retrieval=_Retrieval())
    found = await bank.retrieve("query", top_k=2)
    assert [item.rel_path for item in found][0] == "both"


async def test_a_banked_answer_carries_the_evidence_that_made_it_true() -> None:
    atoms = [
        _atom(
            "a1",
            "Why dedicated over on-demand?",
            "Isolation and predictable throughput.",
            citations=["Notion — DAC competition", "OCI docs — decision table"],
        )
    ]
    bank = AnswerBank(Settings(), _FakeDatabase(atoms, [("a1", 1.0)]), retrieval=None)
    found = await bank.retrieve("dedicated vs on demand", top_k=1)
    assert "Originally grounded in" in found[0].text
    assert "Notion — DAC competition" in found[0].text


async def test_a_disabled_bank_retrieves_nothing() -> None:
    bank = AnswerBank(
        Settings(answer_bank_enabled=False),
        _FakeDatabase([_atom("a1", "q", "a")], [("a1", 1.0)]),
        retrieval=None,
    )
    assert await bank.retrieve("anything") == []


async def test_retrieval_never_raises_when_the_store_is_broken() -> None:
    class _Broken(_FakeDatabase):
        async def search_answer_atoms_lexical(self, query, limit=20):
            raise RuntimeError("index is cold")

    bank = AnswerBank(Settings(), _Broken([], []), retrieval=None)
    assert await bank.retrieve("anything") == []


def test_paraphrases_are_embedded_with_the_question() -> None:
    """The next customer rarely uses the canonical wording, so the paraphrase
    is often the closest thing to how they will actually ask."""
    text = _embed_text(_atom("a1", "What is the minimum?", "744 unit-hours."))
    assert "What is the minimum?" in text
    assert "how long must I commit for?" in text
    assert "744 unit-hours." in text


def test_vectors_round_trip_and_cosine_is_sane() -> None:
    vector = [0.5, -0.25, 0.75]
    assert _unpack(_pack(vector)) == [0.5, -0.25, 0.75]
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert _cosine([], [1.0]) == 0.0
