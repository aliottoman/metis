"""Tier-1 personal knowledge RAG — index the user's own material, retrieve it
just-in-time, and cite it.

Pipeline: walk a consented source → structure-aware chunk (`chunking.py`) →
Cohere embed-v4 (on-demand) → store normalized float32 vectors LOCALLY (SQLite).
At query time: embed the question → cosine recall the top ``corpus_recall_k`` →
Cohere rerank → keep the top ``corpus_top_k`` → return cited snippets.

Two invariants matter:
- **Consent is the egress boundary.** `index_source` refuses to embed a source
  that has not been granted consent, and refuses entirely when cloud retrieval
  is unavailable — so nothing leaves the Mac by default.
- **Incremental by content hash.** Only changed/new files are re-embedded and
  removed files are dropped, so "index all my code" is cheap to keep fresh.

`numpy` is imported lazily so this module imports in a base install without the
optional `cloud` extra; retrieval simply reports itself unavailable there.
"""
from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from . import code_graph
from .chunking import chunk_text, lang_for
from .config import Settings
from .contracts import (
    CodeGraphLookupV1,
    CodeGraphStatsV1,
    CorpusReindexResultV1,
    CorpusSourceV1,
    EntityGraphLookupV1,
    EntityGraphStatsV1,
    KnowledgeSnippetV1,
)
from .database import Database
from .embeddings import CohereRetrieval, CohereUnavailable

# Prose file languages that get cloud entity extraction (Stage 2). Code files
# are covered by the deterministic code graph instead.
_PROSE_LANGS = {"markdown", "rst", "text"}

_SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".wakil", ".data",
    ".idea", ".vscode", "dist", "build", ".next", ".mypy_cache", ".pytest_cache",
    ".uv-cache", ".pnpm-store", ".ruff_cache", ".turbo", "target", ".gradle",
}

# Folded into every file's content hash. Bump it whenever chunking changes, so
# an incremental reindex re-embeds files whose bytes are identical but whose
# chunks would now be built differently — otherwise a chunker improvement never
# reaches an already-indexed corpus.
_INDEX_VERSION = "2"


@dataclass(frozen=True)
class _ScanReport:
    """Why the files under a source root were or were not indexed. Turns a scan
    that quietly matched nothing into a diagnosable failure."""

    total_files: int = 0
    skipped_dir: int = 0
    skipped_extension: int = 0
    skipped_size: int = 0


def max_bytes_mb(value: int) -> str:
    """Render a byte cap for a user-facing message."""
    return f"{value / 1_048_576:.1f} MB"


def _clean_path_input(raw: str) -> str:
    """Normalize a pasted directory path.

    Dragging a folder into a text field or copying a path from quoted text
    often brings surrounding whitespace and matching quotes along (``'/a/b'``).
    A real directory never starts or ends with a quote, so we strip one matching
    pair plus surrounding whitespace, turning an easy paste mistake into the
    path the user meant.
    """
    cleaned = (raw or "").strip()
    for quote in ("'", '"'):
        if len(cleaned) >= 2 and cleaned[0] == quote and cleaned[-1] == quote:
            cleaned = cleaned[1:-1].strip()
            break
    return cleaned


class CorpusService:
    """Owns corpus indexing + retrieval over the local vector store."""

    def __init__(
        self, settings: Settings, database: Database, retrieval: CohereRetrieval
    ) -> None:
        self._settings = settings
        self._db = database
        self._retrieval = retrieval

    def available(self) -> bool:
        """True when cloud retrieval is opted-in, configured, and importable."""
        return self._retrieval.available()

    # ── Source management ────────────────────────────────────────────────────

    async def register_source(
        self,
        root_path: str,
        label: str | None,
        kind: str,
        *,
        provider: str = "local",
    ) -> CorpusSourceV1:
        cleaned = _clean_path_input(root_path)
        path = Path(cleaned).expanduser()
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"path is not an existing directory: {cleaned}")
        resolved = str(path.resolve())
        return await self._db.create_corpus_source(
            resolved,
            (label or path.name or resolved).strip(),
            kind,
            provider=provider,
        )

    async def list_sources(self) -> list[CorpusSourceV1]:
        return await self._db.list_corpus_sources()

    async def get_source(self, source_id: str) -> CorpusSourceV1 | None:
        return await self._db.get_corpus_source(source_id)

    async def set_consent(
        self, source_id: str, consent: bool, reason: str | None
    ) -> CorpusSourceV1:
        return await self._db.set_corpus_consent(source_id, consent, reason)

    async def delete_source(self, source_id: str) -> bool:
        return await self._db.delete_corpus_source(source_id)

    # ── Indexing ─────────────────────────────────────────────────────────────

    async def index_source(self, source_id: str) -> CorpusReindexResultV1:
        """Incrementally (re)index a consented source. Safe to run as a
        background task: every failure path records an 'error' status rather
        than raising out of the task."""
        source = await self._db.get_corpus_source(source_id)
        if source is None:
            raise KeyError("corpus source not found")
        if not source.consent:
            raise PermissionError(
                "source has not been granted cloud-embedding consent"
            )
        if not self._retrieval.available():
            raise CohereUnavailable("cloud embeddings are not available")

        await self._db.begin_corpus_indexing(source_id)
        try:
            existing = await self._db.get_corpus_file_index(source_id)
            root = Path(source.root_path)
            scanned, report = await asyncio.to_thread(self._scan, root)
            # A source full of files that matched nothing is a misconfiguration,
            # not a successful empty index. Reporting it as 'indexed' is how a
            # silently-skipped corpus reaches retrieval as "no results found".
            if not scanned and report.total_files:
                raise ValueError(
                    f"found {report.total_files} file(s) under {root} but none "
                    f"were indexable: {report.skipped_dir} inside skipped "
                    f"directories, {report.skipped_extension} of unsupported "
                    f"file types, {report.skipped_size} over the "
                    f"{max_bytes_mb(self._settings.corpus_max_file_bytes)} size "
                    "cap. Nothing was embedded."
                )
            seen: set[str] = set()
            indexed = skipped = 0
            for rel_path, content_hash, lang, path in scanned:
                seen.add(rel_path)
                if existing.get(rel_path) == content_hash:
                    skipped += 1
                    continue
                text = await asyncio.to_thread(self._read_text, path)
                if text is None:
                    continue
                chunks = chunk_text(
                    text,
                    lang,
                    max_chars=self._settings.corpus_chunk_chars,
                    overlap=self._settings.corpus_chunk_overlap,
                )
                rows: list[dict] = []
                if chunks:
                    vectors = await asyncio.to_thread(
                        self._retrieval.embed, [c.text for c in chunks], "search_document"
                    )
                    rows = [
                        self._chunk_row(chunk, vector)
                        for chunk, vector in zip(chunks, vectors)
                    ]
                graph_nodes, graph_edges = self._extract_graph(lang, text, rel_path)
                entity_nodes, entity_edges = await self._extract_entities(lang, text)
                await self._db.upsert_corpus_file(
                    source_id, rel_path, content_hash, lang, rows,
                    graph_nodes=graph_nodes, graph_edges=graph_edges,
                    entity_nodes=entity_nodes, entity_edges=entity_edges,
                )
                indexed += 1
            removed = await self._db.remove_corpus_files(
                source_id, [path for path in existing if path not in seen]
            )
            source = await self._db.finish_corpus_indexing(
                source_id, "indexed", embed_model=self._settings.oci_embed_model
            )
            return CorpusReindexResultV1(
                source_id=source_id,
                status=source.status,
                files_indexed=indexed,
                files_skipped=skipped,
                files_removed=removed,
                chunks=source.chunk_count,
                message=(
                    f"indexed {indexed} file(s), skipped {skipped} unchanged, "
                    f"removed {removed}; {source.chunk_count} chunks total"
                ),
            )
        except Exception as error:  # noqa: BLE001 - record and re-raise for the caller
            await self._db.finish_corpus_indexing(
                source_id, "error", last_error=str(error)[:500]
            )
            raise

    # ── Retrieval ────────────────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        limit: int | None = None,
        *,
        on_stage: Callable[[str, str], Awaitable[None]] | None = None,
        provider: str | None = None,
    ) -> list[KnowledgeSnippetV1]:
        """Embed → cosine recall → rerank → cited top-k. Returns [] (never raises)
        when retrieval is unavailable or the corpus is empty, so a turn degrades
        to the local path instead of failing.

        `on_stage(stage, label)` — if given — is awaited before the two slow
        cloud round-trips (embed, rerank) so the UI can narrate them live. It is
        advisory: a raising callback would propagate, so callers pass a
        never-raising emitter."""
        query = (query or "").strip()
        if not query or not self._retrieval.available():
            return []
        if provider not in {None, "local", "notion"}:
            raise ValueError("unknown corpus provider")
        rows = await self._db.corpus_search_vectors(provider)
        if not rows:
            return []
        top_k = limit or self._settings.corpus_top_k
        recall_k = max(self._settings.corpus_recall_k, top_k)
        if on_stage is not None:
            await on_stage("embedding", "Embedding your question…")
        query_vectors = await asyncio.to_thread(
            self._retrieval.embed, [query], "search_query"
        )
        recall = await asyncio.to_thread(
            self._cosine_recall, query_vectors[0], rows, recall_k
        )
        if not recall:
            return []
        recall_ids = [chunk_id for chunk_id, _ in recall]
        recall_scores = {chunk_id: score for chunk_id, score in recall}
        candidates = await self._db.corpus_chunks_by_ids(recall_ids)
        order = {chunk_id: position for position, chunk_id in enumerate(recall_ids)}
        candidates.sort(key=lambda item: order.get(item["id"], len(order)))
        candidates = await self._graph_expand(candidates, provider=provider)
        if on_stage is not None:
            await on_stage("reranking", "Reranking the best matches…")
        try:
            ranked = await asyncio.to_thread(
                self._retrieval.rerank,
                query,
                [item["text"] for item in candidates],
                top_k,
            )
        except Exception:  # noqa: BLE001 - degrade, don't fail the whole retrieval
            # Rerank may be unavailable in the configured region; fall back to cosine order.
            ranked = sorted(
                (
                    (index, recall_scores.get(item["id"], 0.0))
                    for index, item in enumerate(candidates)
                ),
                key=lambda pair: pair[1],
                reverse=True,
            )[:top_k]
        ranked_items = [(candidates[index], score) for index, score in ranked]
        snippets = [self._snippet(item, score) for item, score in ranked_items]
        return snippets + await self._page_neighbours(ranked_items)

    async def _page_neighbours(
        self, ranked_items: list[tuple[dict, float]]
    ) -> list[KnowledgeSnippetV1]:
        """Complete the top-ranked documents with their remaining passages.

        Vector search returns the passages most similar to the question, which is
        the wrong shape for "summarize everything about X": when one document is
        about X end to end, its later sections lose top-k slots to unrelated
        documents that merely mention X. Once a document has clearly won, pull
        the rest of it in document order so an answer can cover the whole page
        rather than the fragments that happened to match.

        Neighbours inherit their anchor's score — they are context for an
        already-relevant document, so inheriting keeps them above the caller's
        relevance floor without raising the top score the grounding review
        reads."""
        settings = self._settings
        pages = settings.corpus_page_expand_pages
        per_query = settings.corpus_page_expand_k
        if not (settings.corpus_page_expand and ranked_items) or not (pages and per_query):
            return []
        anchors: dict[tuple[str, str], float] = {}
        for item, score in ranked_items:
            key = (str(item.get("source_id", "")), str(item.get("rel_path", "")))
            if not all(key) or key in anchors:
                continue
            anchors[key] = float(score)
            if len(anchors) >= pages:
                break
        if not anchors:
            return []
        rows = await self._db.corpus_chunks_by_paths(
            list(anchors), [item["id"] for item, _ in ranked_items], per_query
        )
        return [
            self._snippet(row, anchors.get((row["source_id"], row["rel_path"]), 0.0))
            for row in rows
        ]

    @staticmethod
    def _snippet(item: dict, score: float) -> KnowledgeSnippetV1:
        return KnowledgeSnippetV1(
            source_label=item["source_label"],
            provider=item.get("source_provider", "local"),
            rel_path=item["rel_path"],
            symbol=item.get("symbol"),
            start_line=item.get("start_line"),
            text=item["text"],
            score=score,
        )

    # ── Code graph (Graph-RAG Stage 1) ───────────────────────────────────────

    async def graph_stats(self) -> CodeGraphStatsV1:
        """Node/edge counts over the deterministic code graph. Purely local — it
        never touches the cloud, so it works even with embeddings disabled."""
        return CodeGraphStatsV1.model_validate(await self._db.code_graph_stats())

    async def graph_lookup(self, name: str) -> CodeGraphLookupV1:
        """Resolve a symbol name to its definitions, callers, and callees."""
        return CodeGraphLookupV1.model_validate(
            await self._db.code_graph_lookup(name)
        )

    # ── Entity graph (Graph-RAG Stage 2) ─────────────────────────────────────

    async def entity_stats(self) -> EntityGraphStatsV1:
        """Entity node/edge counts. Local read; empty unless entity extraction
        has been enabled and a prose source indexed."""
        return EntityGraphStatsV1.model_validate(
            await self._db.entity_graph_stats()
        )

    async def entity_lookup(self, name: str) -> EntityGraphLookupV1:
        """Resolve an entity name to its kinds and relationships (both directions)."""
        return EntityGraphLookupV1.model_validate(
            await self._db.entity_graph_lookup(name)
        )

    async def _graph_expand(
        self, candidates: list[dict], *, provider: str | None = None
    ) -> list[dict]:
        """Add call-graph neighbours of the top recalled chunks to the candidate
        pool, so the reranker can promote structurally related code (callers /
        callees) that pure vector recall missed. The reranker still decides what
        survives, so an off-topic neighbour simply loses — expansion widens recall
        without polluting the final top-k."""
        if provider == "notion" or not (
            self._settings.corpus_graph_expand
            and self._settings.corpus_graph_expand_k
        ):
            return candidates
        seeds = [
            candidate["symbol"]
            for candidate in candidates[: self._settings.corpus_graph_expand_seeds]
            if candidate.get("symbol")
        ]
        if not seeds:
            return candidates
        neighbour_names = await self._db.code_graph_neighbor_names(seeds)
        if not neighbour_names:
            return candidates
        existing = [candidate["id"] for candidate in candidates]
        extra = await self._db.corpus_chunks_by_symbols(
            list(neighbour_names), existing, self._settings.corpus_graph_expand_k
        )
        return candidates + extra

    # ── Internals ────────────────────────────────────────────────────────────

    def _extract_graph(
        self, lang: str, text: str, rel_path: str
    ) -> tuple[list[dict], list[dict]]:
        """Deterministic AST graph for a Python file, as DB-ready row dicts.
        Fail-soft: any parse issue yields no graph rather than aborting the run."""
        if not (self._settings.corpus_graph_enabled and lang == "python"):
            return [], []
        try:
            nodes, edges = code_graph.extract(text, rel_path)
        except Exception:  # noqa: BLE001 - a bad file never fails the index run
            return [], []
        node_rows = [
            {
                "kind": node.kind,
                "name": node.name,
                "qualname": node.qualname,
                "start_line": node.start_line,
                "end_line": node.end_line,
            }
            for node in nodes
        ]
        edge_rows = [
            {
                "kind": edge.kind,
                "src": edge.src,
                "dst_name": edge.dst_name,
                "dst_raw": edge.dst_raw,
                "line": edge.line,
            }
            for edge in edges
        ]
        return node_rows, edge_rows

    async def _extract_entities(
        self, lang: str, text: str
    ) -> tuple[list[dict], list[dict]]:
        """Cloud entity extraction (Stage 2) for a prose file, as DB-ready rows.

        Opt-in and prose-only; the cloud call runs off the event loop and is
        fail-soft, so a bad reply or a hiccup yields no entities rather than
        failing the index run."""
        if not (self._settings.corpus_entity_graph and lang in _PROSE_LANGS):
            return [], []
        try:
            extraction = await asyncio.to_thread(
                self._retrieval.extract_entities,
                text[: self._settings.corpus_entity_max_chars],
            )
        except Exception:  # noqa: BLE001 - a bad extraction never fails the run
            return [], []
        entity_nodes = [
            {"name": entity.name, "kind": entity.kind}
            for entity in extraction.entities
        ]
        entity_edges = [
            {"src_name": relation.source, "relation": relation.relation,
             "dst_name": relation.target}
            for relation in extraction.relations
        ]
        return entity_nodes, entity_edges

    def _scan(
        self, root: Path
    ) -> tuple[list[tuple[str, str, str, Path]], _ScanReport]:
        """Walk `root`, returning (rel_path, content_hash, lang, path) for each
        indexable text file under the size cap, plus a report of what was
        rejected and why.

        The junk-directory test runs against the path *relative to the source
        root*: a directory named in `_SKIP_DIRS` only counts when it sits inside
        the source. Testing the absolute path would let an ancestor the user
        never chose (the Notion mirror lives under `.data/`) disqualify every
        file in the source."""
        results: list[tuple[str, str, str, Path]] = []
        max_bytes = self._settings.corpus_max_file_bytes
        total = skipped_dir = skipped_extension = skipped_size = 0
        for path in root.rglob("*"):
            if path.is_dir():
                continue
            try:
                relative = path.relative_to(root)
            except ValueError:  # a symlink escaping the root
                continue
            total += 1
            if any(part in _SKIP_DIRS for part in relative.parts):
                skipped_dir += 1
                continue
            lang = lang_for(path.suffix)
            if lang is None:
                skipped_extension += 1
                continue
            try:
                if path.stat().st_size > max_bytes:
                    skipped_size += 1
                    continue
                data = path.read_bytes()
            except OSError:
                continue
            content_hash = hashlib.sha256(
                f"{_INDEX_VERSION}:".encode() + data
            ).hexdigest()
            results.append((str(relative), content_hash, lang, path))
        return results, _ScanReport(
            total_files=total,
            skipped_dir=skipped_dir,
            skipped_extension=skipped_extension,
            skipped_size=skipped_size,
        )

    @staticmethod
    def _read_text(path: Path) -> str | None:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    @staticmethod
    def _chunk_row(chunk, vector: list[float]) -> dict:
        import numpy as np

        array = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(array))
        if norm > 0:
            array = array / norm
        return {
            "symbol": chunk.symbol,
            "start_line": chunk.start_line,
            "text": chunk.text,
            "embedding": array.astype(np.float32).tobytes(),
            "dim": int(array.shape[0]),
        }

    @staticmethod
    def _cosine_recall(
        query_vector: list[float], rows, k: int
    ) -> list[tuple[str, float]]:
        """Top-k (chunk_id, cosine_similarity) pairs, most-similar first. The
        score is carried so retrieval can still rank by similarity if the rerank
        stage is unavailable."""
        import numpy as np

        query = np.asarray(query_vector, dtype=np.float32)
        norm = float(np.linalg.norm(query))
        if norm > 0:
            query = query / norm
        dim = int(query.shape[0])
        ids: list[str] = []
        vectors: list = []
        for row in rows:
            if int(row["dim"]) != dim:  # skip vectors from a different embed model
                continue
            ids.append(row["id"])
            vectors.append(np.frombuffer(row["embedding"], dtype=np.float32))
        if not vectors:
            return []
        matrix = np.vstack(vectors)  # stored vectors are already unit-normalized
        similarities = matrix @ query
        top = np.argsort(-similarities)[:k]
        return [(ids[int(i)], float(similarities[int(i)])) for i in top]
