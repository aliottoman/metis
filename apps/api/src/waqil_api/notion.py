"""Read-only, manually refreshed Notion mirror for the personal corpus.

The connector deliberately keeps Notion access separate from retrieval:

* Notion is read only. Sync uses search/retrieve/query endpoints and never
  writes back to the workspace.
* A sync materializes Markdown beneath ``.data/corpus/notion``. The existing
  consented corpus pipeline owns chunking, embeddings, reranking, and citations.
* The secret remains local (mode 0600) and is never included in an API response.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .config import Settings
from .contracts import (
    NotionConnectionUpdateV1,
    NotionConnectionV1,
    NotionSyncResultV1,
)
from .corpus import CorpusService
from .database import Database


_NOTION_API = "https://api.notion.com/v1"
_PAGE_ID = re.compile(
    r"(?<![0-9a-fA-F])([0-9a-fA-F]{32}|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})(?![0-9a-fA-F])"
)
_SLUG = re.compile(r"[^a-z0-9]+")
_REQUEST_INTERVAL_SECONDS = 0.42
_MAX_REQUEST_ATTEMPTS = 8
_MAX_RATE_LIMIT_WAIT_SECONDS = 120.0


class NotionError(RuntimeError):
    """A clean, user-facing Notion connection or sync failure."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code


@dataclass(frozen=True)
class NotionDocument:
    page_id: str
    title: str
    url: str
    last_edited_time: str
    markdown: str


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _normalize_page_id(value: str) -> str:
    match = _PAGE_ID.search((value or "").strip())
    if match is None:
        raise ValueError(f"Notion page ID or URL is invalid: {value!r}")
    compact = match.group(1).replace("-", "").lower()
    return (
        f"{compact[:8]}-{compact[8:12]}-{compact[12:16]}-"
        f"{compact[16:20]}-{compact[20:]}"
    )


def _plain_text(items: Any) -> str:
    if not isinstance(items, list):
        return ""
    return "".join(
        str(item.get("plain_text", ""))
        for item in items
        if isinstance(item, dict)
    ).strip()


def _page_title(page: dict[str, Any]) -> str:
    properties = page.get("properties")
    if isinstance(properties, dict):
        for value in properties.values():
            if not isinstance(value, dict):
                continue
            if value.get("type") == "title":
                title = _plain_text(value.get("title"))
                if title:
                    return title
    title = _plain_text(page.get("title"))
    return title or "Untitled"


class NotionApiClient:
    """Tiny stdlib client pinned to the configured Notion API version."""

    def __init__(self, token: str, version: str) -> None:
        self._token = token
        self._version = version
        self._last_request_started_at: float | None = None

    def _wait_for_request_slot(self) -> None:
        """Stay below Notion's average request budget during large syncs."""
        now = time.monotonic()
        if self._last_request_started_at is not None:
            remaining = (
                _REQUEST_INTERVAL_SECONDS
                - (now - self._last_request_started_at)
            )
            if remaining > 0:
                time.sleep(remaining)
        self._last_request_started_at = time.monotonic()

    @staticmethod
    def _rate_limit_wait(error: HTTPError, attempt: int) -> float:
        """Honor Retry-After and progressively cool down repeated 429s."""
        try:
            retry_after = float(error.headers.get("Retry-After", "1"))
        except (TypeError, ValueError):
            retry_after = 1.0
        exponential_wait = 0.5 * (2**attempt)
        return min(
            max(retry_after, exponential_wait, 0.5) + 0.1,
            _MAX_RATE_LIMIT_WAIT_SECONDS,
        )

    def _request(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        request = Request(
            f"{_NOTION_API}{path}",
            data=payload,
            method=method,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Notion-Version": self._version,
                "Content-Type": "application/json",
                "User-Agent": "Metis/0.1 Notion RAG",
            },
        )
        for attempt in range(_MAX_REQUEST_ATTEMPTS):
            self._wait_for_request_slot()
            try:
                with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed host
                    decoded = json.loads(response.read().decode("utf-8"))
                    return decoded if isinstance(decoded, dict) else {}
            except HTTPError as error:
                if error.code == 429 and attempt < _MAX_REQUEST_ATTEMPTS - 1:
                    time.sleep(self._rate_limit_wait(error, attempt))
                    continue
                try:
                    error_payload = json.loads(error.read().decode("utf-8"))
                    detail = error_payload.get("message")
                    error_code = error_payload.get("code")
                except Exception:  # noqa: BLE001 - error body is untrusted
                    detail = None
                    error_code = None
                if error.code == 429:
                    raise NotionError(
                        "Notion is still rate-limiting this sync after several "
                        "automatic retries. Wait a minute, then sync again.",
                        status_code=error.code,
                        error_code=str(error_code or "") or None,
                    ) from error
                raise NotionError(
                    f"Notion returned {error.code}: {detail or error.reason}",
                    status_code=error.code,
                    error_code=str(error_code or "") or None,
                ) from error
            except (URLError, TimeoutError, OSError) as error:
                if attempt < _MAX_REQUEST_ATTEMPTS - 1:
                    time.sleep(0.5 * (2**attempt))
                    continue
                raise NotionError(f"Could not reach Notion: {error}") from error
        raise NotionError("Notion request failed")

    def search_pages(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            body: dict[str, Any] = {
                "page_size": 100,
                "filter": {"property": "object", "value": "page"},
            }
            if cursor:
                body["start_cursor"] = cursor
            page = self._request("POST", "/search", body)
            results.extend(
                item for item in page.get("results", []) if isinstance(item, dict)
            )
            cursor = page.get("next_cursor") if page.get("has_more") else None
            if not cursor:
                return results

    def retrieve_page(self, page_id: str) -> dict[str, Any]:
        return self._request("GET", f"/pages/{quote(page_id)}")

    def retrieve_markdown(self, block_id: str) -> dict[str, Any]:
        return self._request("GET", f"/pages/{quote(block_id)}/markdown")

    def block_children(self, block_id: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            path = f"/blocks/{quote(block_id)}/children?page_size=100"
            if cursor:
                path += f"&start_cursor={quote(cursor)}"
            page = self._request("GET", path)
            results.extend(
                item for item in page.get("results", []) if isinstance(item, dict)
            )
            cursor = page.get("next_cursor") if page.get("has_more") else None
            if not cursor:
                return results

    def retrieve_database(self, database_id: str) -> dict[str, Any]:
        return self._request("GET", f"/databases/{quote(database_id)}")

    def query_data_source(self, data_source_id: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            body: dict[str, Any] = {"page_size": 100}
            if cursor:
                body["start_cursor"] = cursor
            page = self._request(
                "POST", f"/data_sources/{quote(data_source_id)}/query", body
            )
            results.extend(
                item for item in page.get("results", []) if isinstance(item, dict)
            )
            cursor = page.get("next_cursor") if page.get("has_more") else None
            if not cursor:
                return results


class NotionService:
    def __init__(
        self,
        settings: Settings,
        database: Database,
        corpus: CorpusService,
        *,
        client_factory: type[NotionApiClient] = NotionApiClient,
    ) -> None:
        self._settings = settings
        self._db = database
        self._corpus = corpus
        self._client_factory = client_factory

    def _load(self) -> dict[str, Any]:
        try:
            raw = json.loads(self._settings.notion_config_path.read_text("utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError) as error:
            raise NotionError(f"Notion settings could not be read: {error}") from error
        return raw if isinstance(raw, dict) else {}

    def _save(self, config: dict[str, Any]) -> None:
        path = self._settings.notion_config_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)

    def _token(self, config: dict[str, Any]) -> str:
        return str(config.get("access_token") or self._settings.notion_token).strip()

    async def _ensure_source(self, label: str):
        source = await self._db.get_corpus_source_by_provider("notion")
        if source is None:
            source = await self._corpus.register_source(
                str(self._settings.notion_mirror_dir),
                label,
                "notes",
                provider="notion",
            )
        elif source.label != label:
            source = await self._db.update_corpus_source_label(source.id, label)
        return source

    async def status(self) -> NotionConnectionV1:
        config = self._load()
        source = await self._db.get_corpus_source_by_provider("notion")
        token_configured = bool(self._token(config))
        return NotionConnectionV1(
            configured=token_configured,
            token_configured=token_configured,
            root_page_ids=list(config.get("root_page_ids") or []),
            label=str(config.get("label") or "Notion"),
            source=source,
            last_synced_at=config.get("last_synced_at"),
            page_count=int(config.get("page_count") or 0),
            last_error=config.get("last_error"),
        )

    async def configure(
        self, update: NotionConnectionUpdateV1
    ) -> NotionConnectionV1:
        config = self._load()
        roots: list[str] = []
        for raw in update.root_page_ids:
            normalized = _normalize_page_id(raw)
            if normalized not in roots:
                roots.append(normalized)
        if update.access_token is not None:
            config["access_token"] = update.access_token.strip()
        config.update(
            root_page_ids=roots,
            label=update.label.strip(),
            last_error=None,
        )
        if not self._token(config):
            raise ValueError("a Notion internal integration token is required")
        self._save(config)
        await self._ensure_source(config["label"])
        return await self.status()

    async def sync(self) -> NotionSyncResultV1:
        config = self._load()
        token = self._token(config)
        if not token:
            raise NotionError("Connect Notion before syncing")
        label = str(config.get("label") or "Notion")
        source = await self._ensure_source(label)
        roots = list(config.get("root_page_ids") or [])
        try:
            documents, skipped_inaccessible = await asyncio.to_thread(
                self._fetch_documents, token, roots
            )
            written, removed = await asyncio.to_thread(
                self._materialize, documents
            )
            config.update(
                last_synced_at=_now(),
                page_count=len(documents),
                last_error=None,
            )
            self._save(config)
        except Exception as error:  # noqa: BLE001 - persist a useful sync status
            config["last_error"] = str(error)[:500]
            self._save(config)
            if isinstance(error, NotionError):
                raise
            raise NotionError(str(error)) from error

        index_result = None
        message = f"Synced {len(documents)} Notion page(s) into the local mirror."
        if skipped_inaccessible:
            message += (
                f" Skipped {skipped_inaccessible} inaccessible linked item(s); "
                'share them with the integration "Metis" to include them.'
            )
        if source.consent and self._corpus.available():
            try:
                index_result = await self._corpus.index_source(source.id)
            except Exception as error:  # noqa: BLE001 - mirror succeeded; report index cleanly
                config["last_error"] = (
                    f"Pages were mirrored, but RAG indexing failed: {error}"
                )[:500]
                self._save(config)
                raise NotionError(config["last_error"]) from error
            source = await self._corpus.get_source(source.id) or source
            message += " The consented RAG index is up to date."
        elif not source.consent:
            message += " Enable RAG indexing to embed and retrieve these pages."
        else:
            message += " Cloud embeddings are unavailable, so the mirror was not indexed."
        return NotionSyncResultV1(
            pages_fetched=len(documents),
            pages_written=written,
            pages_removed=removed,
            source=source,
            index_result=index_result,
            message=message,
        )

    def _fetch_documents(
        self, token: str, root_page_ids: list[str]
    ) -> tuple[list[NotionDocument], int]:
        client = self._client_factory(token, self._settings.notion_api_version)
        pages: dict[str, dict[str, Any]] = {}
        pending: list[str] = []
        explicit_roots = set(root_page_ids)
        skipped_inaccessible_ids: set[str] = set()
        if root_page_ids:
            pending.extend(root_page_ids)
        else:
            for page in client.search_pages():
                page_id = str(page.get("id") or "")
                if page_id and not page.get("in_trash"):
                    pages[page_id] = page
                    pending.append(page_id)

        documents: list[NotionDocument] = []
        visited: set[str] = set()
        while pending:
            page_id = pending.pop(0)
            if page_id in visited:
                continue
            if len(visited) >= self._settings.notion_sync_max_pages:
                raise NotionError(
                    f"Notion sync exceeded the {self._settings.notion_sync_max_pages}-page safety limit"
                )
            visited.add(page_id)
            page = pages.get(page_id)
            if page is None:
                try:
                    page = client.retrieve_page(page_id)
                except NotionError as error:
                    if error.status_code == 404 and page_id not in explicit_roots:
                        skipped_inaccessible_ids.add(page_id)
                        continue
                    raise
            if page.get("in_trash"):
                continue
            try:
                markdown = client.retrieve_markdown(page_id)
            except NotionError as error:
                if error.status_code == 404 and page_id not in explicit_roots:
                    skipped_inaccessible_ids.add(page_id)
                    continue
                raise
            content = str(markdown.get("markdown") or "").strip()
            for unknown_id in markdown.get("unknown_block_ids") or []:
                unknown_id = str(unknown_id)
                try:
                    extra = client.retrieve_markdown(unknown_id)
                except NotionError as error:
                    # Notion returns object_not_found for unknown blocks caused
                    # by permissions. The parent page remains valid and its
                    # markdown already contains an <unknown> placeholder.
                    if error.status_code == 404:
                        skipped_inaccessible_ids.add(unknown_id)
                        continue
                    raise
                extra_content = str(extra.get("markdown") or "").strip()
                if extra_content:
                    content += f"\n\n{extra_content}"
            documents.append(
                NotionDocument(
                    page_id=page_id,
                    title=_page_title(page),
                    url=str(page.get("url") or ""),
                    last_edited_time=str(page.get("last_edited_time") or ""),
                    markdown=content,
                )
            )
            try:
                child_pages, child_databases = self._discover_children(client, page_id)
            except NotionError as error:
                if error.status_code == 404:
                    skipped_inaccessible_ids.add(page_id)
                    child_pages, child_databases = set(), set()
                else:
                    raise
            for child in child_pages:
                if child not in visited:
                    pending.append(child)
            for database_id in child_databases:
                try:
                    database = client.retrieve_database(database_id)
                    for data_source in database.get("data_sources") or []:
                        data_source_id = str(data_source.get("id") or "")
                        if not data_source_id:
                            continue
                        for row in client.query_data_source(data_source_id):
                            child_id = str(row.get("id") or "")
                            if child_id:
                                pages[child_id] = row
                                pending.append(child_id)
                except NotionError:
                    # A database may be visible as a block while its source was
                    # not shared. Keep the readable pages instead of aborting all.
                    continue
        return documents, len(skipped_inaccessible_ids)

    @staticmethod
    def _discover_children(
        client: NotionApiClient, page_id: str
    ) -> tuple[set[str], set[str]]:
        pages: set[str] = set()
        databases: set[str] = set()
        pending_blocks = [page_id]
        visited_blocks: set[str] = set()
        while pending_blocks:
            block_id = pending_blocks.pop()
            if block_id in visited_blocks:
                continue
            visited_blocks.add(block_id)
            for block in client.block_children(block_id):
                kind = block.get("type")
                child_id = str(block.get("id") or "")
                if kind == "child_page" and child_id:
                    pages.add(child_id)
                elif kind == "child_database" and child_id:
                    databases.add(child_id)
                elif block.get("has_children") and child_id:
                    pending_blocks.append(child_id)
        return pages, databases

    def _materialize(self, documents: list[NotionDocument]) -> tuple[int, int]:
        root = self._settings.notion_mirror_dir
        root.mkdir(parents=True, exist_ok=True)
        expected: set[str] = set()
        written = 0
        for document in documents:
            slug = _SLUG.sub("-", document.title.lower()).strip("-")[:70] or "untitled"
            compact_id = document.page_id.replace("-", "")
            filename = f"{slug}--{compact_id}.md"
            expected.add(filename)
            text = (
                f"# {document.title}\n\n"
                f"Notion page ID: `{document.page_id}`  \n"
                f"Notion URL: {document.url or 'Unavailable'}  \n"
                f"Last edited: {document.last_edited_time or 'Unknown'}\n\n"
                f"{document.markdown}\n"
            )
            path = root / filename
            try:
                unchanged = path.read_text("utf-8") == text
            except FileNotFoundError:
                unchanged = False
            if not unchanged:
                path.write_text(text, encoding="utf-8")
                written += 1
        removed = 0
        for path in root.glob("*.md"):
            if path.name not in expected:
                path.unlink()
                removed += 1
        return written, removed
