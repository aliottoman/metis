"""Completed runs, written back into the corpus as retrievable documents.

Metis already remembers conversations, but only as rows it can replay — not as
knowledge it can find. A question like "how did we fix the flaky import last
month" has no path to an answer, because past work was never indexed.

Each completed run becomes one Markdown document under the data directory and
flows through the ordinary corpus pipeline: same chunking, same consent, same
embeddings, same citations. That reuse is the point — run history is not a new
retrieval system, it is a new source for the one that already exists.

What a document deliberately does *not* contain is the content of any diff. The
record answers "what did we do and which files changed", not "what did line 40
say" — a file's text may hold credentials that the user never intended to send
anywhere, and consent to index their own notes should not quietly become consent
to index their secrets.
"""
from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import Settings
from .contracts import CorpusSourceV1
from .database import Database


RESERVED_LABEL = "Run history"
_RUN_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")
_SECRETISH = re.compile(
    r"(?i)(password|passwd|secret|private[_ -]?key|api[_ -]?key|access[_ -]?token)"
)


def _clean(value: Any, limit: int) -> str:
    return " ".join(str(value or "").replace("\x00", "").split())[:limit]


class RunHistoryService:
    """Materializes finished runs and keeps their corpus source current."""

    def __init__(
        self, settings: Settings, database: Database, corpus: Any | None = None
    ) -> None:
        self._settings = settings
        self._db = database
        self._corpus = corpus

    @property
    def root(self) -> Path:
        return self._settings.data_dir / "corpus" / "runs"

    async def record(
        self,
        *,
        run_id: str,
        conversation_id: str,
        prompt: str,
        response: str,
        changes: list[dict[str, Any]] | None = None,
        artifacts: list[dict[str, Any]] | None = None,
        project_name: str = "",
    ) -> Path | None:
        """Write one run's document. Returns the path, or None when skipped."""
        if not self._settings.run_history_enabled:
            return None
        if not _RUN_ID.fullmatch(run_id or ""):
            return None
        prompt = (prompt or "").strip()
        response = (response or "").strip()
        if not prompt or not response:
            return None

        stamp = datetime.now(UTC)
        directory = self.root / stamp.strftime("%Y-%m")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{run_id}.md"
        path.write_text(
            self._document(
                run_id=run_id,
                conversation_id=conversation_id,
                stamp=stamp,
                prompt=prompt,
                response=response,
                changes=changes or [],
                artifacts=artifacts or [],
                project_name=project_name,
            ),
            encoding="utf-8",
        )
        return path

    def _document(
        self,
        *,
        run_id: str,
        conversation_id: str,
        stamp: datetime,
        prompt: str,
        response: str,
        changes: list[dict[str, Any]],
        artifacts: list[dict[str, Any]],
        project_name: str,
    ) -> str:
        limit = self._settings.run_history_max_chars
        title = _clean(prompt, 110) or "Untitled request"
        # The heading carries the request itself, because Markdown chunking uses
        # headings as the breadcrumb: a passage from the middle of this document
        # should still say which question it came from.
        lines = [
            f"# {title}",
            "",
            f"- Date: {stamp.date().isoformat()}",
            f"- Run: `{run_id}`",
            f"- Conversation: `{conversation_id}`",
        ]
        if project_name:
            lines.append(f"- Project: {project_name}")
        lines += ["", "## Request", "", prompt[:limit], "", "## Outcome", "", response[:limit]]

        if changes:
            lines += ["", "## Files changed", ""]
            for change in changes[:40]:
                tool = _clean(change.get("tool"), 40)
                target = _clean(change.get("path"), 300)
                if not target:
                    continue
                verb = "Created" if tool == "create_file" else "Edited"
                lines.append(f"- {verb} `{target}`")
        if artifacts:
            lines += ["", "## Artifacts", ""]
            for artifact in artifacts[:20]:
                name = _clean(artifact.get("name") or artifact.get("filename"), 200)
                if name:
                    lines.append(f"- `{name}`")
        return "\n".join(lines).rstrip() + "\n"

    async def ensure_source(self) -> CorpusSourceV1 | None:
        """Register the reserved run-history source, once, without consent.

        Registering it un-consented is what makes the feature discoverable: the
        source appears in Knowledge with its consent switch off, so enabling
        history is a visible choice rather than a hidden setting.
        """
        if self._corpus is None or not self._settings.run_history_enabled:
            return None
        root = self.root
        root.mkdir(parents=True, exist_ok=True)
        resolved = str(root.resolve())
        for source in await self._corpus.list_sources():
            if source.root_path == resolved:
                return source
        try:
            return await self._corpus.register_source(resolved, RESERVED_LABEL, "notes")
        except ValueError:  # registered concurrently
            for source in await self._corpus.list_sources():
                if source.root_path == resolved:
                    return source
            return None

    async def index(self) -> None:
        """Fold new run documents into the index, if history is consented.

        Indexing is incremental by content hash, so a run that adds one document
        embeds one document. Without consent this is a no-op: the documents stay
        on disk, and nothing about them leaves the machine.
        """
        source = await self.ensure_source()
        if source is None or not source.consent:
            return
        if self._corpus is None or not self._corpus.available():
            return
        await self._corpus.index_source(source.id)


def changes_from_trace(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The mutations that were actually approved and applied in a run.

    Proposals the user rejected, and edits that errored, are not history — a
    record that listed them would describe work that never happened.
    """
    changes: list[dict[str, Any]] = []
    for entry in trace or []:
        tool = entry.get("tool")
        if tool not in {"apply_patch", "create_file"}:
            continue
        result = entry.get("result") or {}
        if not result.get("ok") or not result.get("approved"):
            continue
        output = result.get("output") or {}
        path = output.get("path") or (entry.get("arguments") or {}).get("path")
        if path and not _SECRETISH.search(str(path)):
            changes.append({"tool": tool, "path": path})
    return changes
