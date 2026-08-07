"""The answer bank — what you have already worked out, kept as knowledge.

Run history was indexed long before this and never functioned as an answer
bank, which is the whole reason this module exists. A chunk is a slice of
transcript that happened to sit near an answer; retrieving it returns "a
conversation where DAC came up". An **atom** is the answer itself: the
canonical question, the wording that was actually defensible, and the
citations that made it so. The gap was never retrieval quality — it was that
there was no unit of knowledge to retrieve.

Three things make it a memory rather than a pile:

* **Proposal-first.** Nothing enters the bank unreviewed, and extraction is
  deliberately stingy. A bank that proposes five atoms per run becomes two
  hundred pending reviews in a week, and then it becomes nothing at all.
* **Hybrid retrieval.** These answers turn on tokens dense vectors blur — 744,
  H100_X2, A100_80G_X2, version strings — so lexical and semantic rankings are
  fused rather than chosen between.
* **It is a source, not a feature.** Atoms are returned as ordinary
  ``KnowledgeSnippetV1`` records, so a banked answer is citable in chat, in a
  generated deck, and inside a customer answer, and the claim gate checks its
  figures like any other evidence.
"""
from __future__ import annotations

import asyncio
import json
import math
import struct
from typing import Any

from .config import Settings
from .contracts import KnowledgeSnippetV1
from .database import Database

# Reciprocal rank fusion. The constant damps the head of each list so one
# ranker cannot dominate the other; 60 is the value the original RRF work
# settled on and it behaves well when the two lists disagree, which for
# lexical-vs-semantic is most of the time.
_RRF_K = 60


def _pack(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def _unpack(blob: bytes) -> list[float]:
    return list(struct.unpack(f"{len(blob) // 4}f", blob))


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _embed_text(atom: dict[str, Any]) -> str:
    """What gets embedded: the question, its paraphrases, and the answer.

    Paraphrases are included deliberately — the same question arrives worded
    differently by every customer, and the paraphrase is often closer to how
    the next one will ask it than the canonical form is."""
    try:
        paraphrases = json.loads(atom.get("paraphrases_json") or "[]")
    except (ValueError, TypeError):
        paraphrases = []
    parts = [str(atom.get("question", ""))]
    parts.extend(str(item) for item in paraphrases)
    parts.append(str(atom.get("answer", "")))
    return "\n".join(part for part in parts if part)


class AnswerBank:
    """Stores, embeds, and retrieves answer atoms."""

    def __init__(
        self,
        settings: Settings,
        database: Database,
        retrieval: Any | None = None,
    ) -> None:
        self.settings = settings
        self.database = database
        # Optional cloud retrieval. Without it the bank still works on the
        # lexical half alone, which for exact identifiers is the stronger half.
        self.retrieval = retrieval

    def enabled(self) -> bool:
        return self.settings.answer_bank_enabled

    async def propose(self, atom: dict[str, Any]) -> dict[str, Any]:
        return await self.database.create_answer_atom(atom)

    async def decide(
        self, atom_id: str, status: str, supersedes: list[str] | None = None
    ) -> dict[str, Any] | None:
        """Keep, reject, or retire an atom — and retire what it replaces.

        Supersession is resolved here, at review time, rather than at
        retrieval. That is the whole point: the bank must never hold two
        active answers that disagree, because by the time retrieval has to
        choose between them there is no one left who knows which is current.
        """
        decided = await self.database.decide_answer_atom(atom_id, status)
        if decided is None:
            return None
        if status == "active":
            for replaced in supersedes or []:
                if replaced == atom_id:
                    continue
                await self.database.decide_answer_atom(
                    replaced, "superseded", superseded_by=atom_id
                )
            # Newly active and unembedded is unreachable by meaning; syncing
            # here keeps the dense half current without a scheduled pass.
            await self.sync()
        return decided

    async def sync(self) -> int:
        """Embed active atoms that have no vector yet. Returns how many."""
        if self.retrieval is None or not getattr(self.retrieval, "available", lambda: False)():
            return 0
        pending = await self.database.atoms_missing_vectors()
        if not pending:
            return 0
        texts = [_embed_text(atom) for atom in pending]
        try:
            vectors = await asyncio.to_thread(
                self.retrieval.embed, texts, "search_document"
            )
        except Exception:  # noqa: BLE001 - the lexical half still answers
            return 0
        model = getattr(self.settings, "oci_embed_model", "")
        for atom, vector in zip(pending, vectors):
            await self.database.store_answer_atom_vector(
                str(atom["id"]), model, _pack(list(vector))
            )
        return len(pending)

    async def retrieve(self, query: str, *, top_k: int = 3) -> list[KnowledgeSnippetV1]:
        """Hybrid recall, fused, then reranked. Never raises.

        Lexical and semantic rankings are merged by reciprocal rank fusion
        rather than by score, because a BM25 rank and a cosine similarity are
        not on a common scale and normalising them invents a comparison that
        does not exist. Rank position is the only thing both agree on.
        """
        if not self.enabled():
            return []
        try:
            lexical, dense, by_entity = await asyncio.gather(
                self.database.search_answer_atoms_lexical(query, limit=20),
                self._dense_recall(query, limit=20),
                self._entity_recall(query, limit=20),
            )
        except Exception:  # noqa: BLE001 - retrieval never fails a turn
            return []

        # Three complementary signals, fused by rank. Entity overlap is the
        # one that finds an atom sharing no distinctive keyword and sitting
        # elsewhere in vector space — the case where a customer names the
        # product but phrases everything else differently.
        fused: dict[str, float] = {}
        for ranking in (lexical, dense, by_entity):
            for rank, (atom_id, _) in enumerate(ranking):
                fused[atom_id] = fused.get(atom_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
        if not fused:
            return []

        ordered = sorted(fused.items(), key=lambda item: -item[1])[: max(top_k * 4, 8)]
        atoms = {
            str(row["id"]): row
            for row in await self.database.answer_atoms_by_id(
                [atom_id for atom_id, _ in ordered]
            )
        }
        candidates = [atoms[atom_id] for atom_id, _ in ordered if atom_id in atoms]
        if not candidates:
            return []
        candidates = await self._rerank(query, candidates, top_k)
        return [self._snippet(atom) for atom in candidates[:top_k]]

    async def _entity_recall(self, query: str, limit: int) -> list[tuple[str, float]]:
        """Atoms whose entities the question names.

        Entities are matched against the question's own words rather than
        extracted by a model: a recall arm that costs a model call per turn
        would be paid on every question to help on a few."""
        known = await self.database.answer_entity_counts()
        if not known:
            return []
        lowered = f" {query.lower()} "
        named = [
            str(row["entity"])
            for row in known
            if str(row["entity"]) and f" {row['entity']} " in lowered
        ]
        if not named:
            return []
        rows = await self.database.answer_atoms_by_entities(named, limit=limit)
        return [(str(row["id"]), float(row.get("overlap") or 1)) for row in rows]

    async def conflicts(self, atom: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
        """Active atoms this one might replace.

        Candidate-finding only, and deliberately cheap: shared entities plus
        an active status. Whether one actually supersedes another is a
        judgement the reviewer makes, on a card that shows both."""
        try:
            entities = json.loads(atom.get("entities_json") or "[]")
        except (ValueError, TypeError):
            entities = []
        if not entities:
            return []
        return await self.database.answer_atoms_by_entities(
            [str(item) for item in entities],
            limit=limit,
            exclude=str(atom.get("id", "")),
        )

    async def _dense_recall(self, query: str, limit: int) -> list[tuple[str, float]]:
        if self.retrieval is None or not getattr(self.retrieval, "available", lambda: False)():
            return []
        rows = await self.database.answer_atom_vectors()
        if not rows:
            return []
        try:
            query_vectors = await asyncio.to_thread(
                self.retrieval.embed, [query], "search_query"
            )
        except Exception:  # noqa: BLE001 - fall back to the lexical half
            return []
        probe = list(query_vectors[0])
        scored = [
            (str(row["atom_id"]), _cosine(probe, _unpack(row["vector"])))
            for row in rows
        ]
        scored.sort(key=lambda item: -item[1])
        return scored[:limit]

    async def _rerank(
        self, query: str, candidates: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        """Cross-encoder rerank over the fused shortlist.

        Fusion decides who is worth reading; the reranker decides who actually
        answers the question. Absent the cloud reranker the fused order stands,
        which is a weaker ordering rather than a broken one."""
        if self.retrieval is None or not getattr(self.retrieval, "available", lambda: False)():
            return candidates
        documents = [_embed_text(atom) for atom in candidates]
        try:
            ranked = await asyncio.to_thread(
                self.retrieval.rerank, query, documents, min(top_k, len(documents))
            )
        except Exception:  # noqa: BLE001 - fused order is a fine second best
            return candidates
        return [candidates[index] for index, _ in ranked if 0 <= index < len(candidates)]

    @staticmethod
    def _snippet(atom: dict[str, Any]) -> KnowledgeSnippetV1:
        try:
            citations = json.loads(atom.get("citations_json") or "[]")
        except (ValueError, TypeError):
            citations = []
        trail = f"\n\nOriginally grounded in: {'; '.join(str(c) for c in citations[:4])}" if citations else ""
        return KnowledgeSnippetV1(
            source_label="Your answer",
            provider="answer",
            rel_path=str(atom.get("id", "")),
            symbol=str(atom.get("question", ""))[:120],
            text=f"Q: {atom.get('question', '')}\nA: {atom.get('answer', '')}{trail}",
            # Reviewed and kept by the user, like a customer record — not a
            # fuzzy hit whose relevance still has to be argued.
            score=1.0,
        )
