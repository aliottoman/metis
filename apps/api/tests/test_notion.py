from __future__ import annotations

import io
import json
import stat
from urllib.error import HTTPError

import pytest

from waqil_api.config import Settings
from waqil_api.contracts import NotionConnectionUpdateV1
from waqil_api.database import Database
from waqil_api.notion import NotionApiClient, NotionService


PAGE_ID = "11111111-2222-3333-4444-555555555555"


class _CorpusStub:
    def __init__(self, database: Database) -> None:
        self.database = database

    async def register_source(self, root_path, label, kind, *, provider="local"):
        return await self.database.create_corpus_source(
            root_path, label, kind, provider=provider
        )

    async def get_source(self, source_id):
        return await self.database.get_corpus_source(source_id)

    def available(self) -> bool:
        return False

    async def index_source(self, source_id):  # pragma: no cover - consent stays off
        raise AssertionError("unconsented Notion mirror must not be indexed")


class _FakeNotionClient:
    def __init__(self, token: str, version: str) -> None:
        assert token == "secret_test_notion_token"
        assert version == "2026-03-11"

    def search_pages(self):
        return [
            {
                "object": "page",
                "id": PAGE_ID,
                "url": "https://www.notion.so/Test-11111111222233334444555555555555",
                "last_edited_time": "2026-07-22T08:00:00.000Z",
                "in_trash": False,
                "properties": {
                    "title": {
                        "type": "title",
                        "title": [{"plain_text": "Launch notes"}],
                    }
                },
            }
        ]

    def retrieve_page(self, page_id):
        raise AssertionError(f"search metadata should cover {page_id}")

    def retrieve_markdown(self, page_id):
        assert page_id == PAGE_ID
        return {
            "object": "page_markdown",
            "markdown": "The launch colour is violet.",
            "truncated": False,
            "unknown_block_ids": [],
        }

    def block_children(self, page_id):
        assert page_id == PAGE_ID
        return []


class _NotionClientWithInaccessibleChild(_FakeNotionClient):
    UNKNOWN_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

    def retrieve_markdown(self, page_id):
        if page_id == self.UNKNOWN_ID:
            from waqil_api.notion import NotionError

            raise NotionError(
                f"Notion returned 404: Could not find page with ID: {page_id}.",
                status_code=404,
                error_code="object_not_found",
            )
        assert page_id == PAGE_ID
        return {
            "object": "page_markdown",
            "markdown": (
                "The readable parent content remains available.\n\n"
                f'<unknown url="https://notion.so/{self.UNKNOWN_ID}"/>'
            ),
            "truncated": True,
            "unknown_block_ids": [self.UNKNOWN_ID],
        }


class _NotionClientWithMissingExplicitRoot(_FakeNotionClient):
    def retrieve_page(self, page_id):
        from waqil_api.notion import NotionError

        raise NotionError(
            f"Notion returned 404: Could not find page with ID: {page_id}.",
            status_code=404,
            error_code="object_not_found",
        )


class _JsonResponse:
    def __init__(self, value: dict) -> None:
        self._value = value

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(self._value).encode("utf-8")


def test_notion_client_recovers_from_sustained_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    sleeps: list[float] = []

    def fake_urlopen(request, timeout):
        nonlocal attempts
        attempts += 1
        if attempts <= 5:
            raise HTTPError(
                request.full_url,
                429,
                "Too Many Requests",
                {"Retry-After": "1"},
                io.BytesIO(
                    b'{"message":"You have been rate limited. Please try again."}'
                ),
            )
        return _JsonResponse({"results": [], "has_more": False})

    monkeypatch.setattr("waqil_api.notion.urlopen", fake_urlopen)
    monkeypatch.setattr("waqil_api.notion.time.sleep", sleeps.append)

    client = NotionApiClient("secret", "2026-03-11")
    assert client.search_pages() == []
    assert attempts == 6
    assert len(sleeps) >= 5
    assert max(sleeps) >= 8.0


def test_notion_client_paces_successive_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    timestamps = iter([0.0, 0.0, 0.1, 0.1])

    monkeypatch.setattr(
        "waqil_api.notion.urlopen",
        lambda request, timeout: _JsonResponse({}),
    )
    monkeypatch.setattr("waqil_api.notion.time.sleep", sleeps.append)
    monkeypatch.setattr(
        "waqil_api.notion.time.monotonic",
        lambda: next(timestamps),
    )

    client = NotionApiClient("secret", "2026-03-11")
    client.retrieve_page(PAGE_ID)
    client.retrieve_markdown(PAGE_ID)

    assert sleeps == [pytest.approx(0.32)]


@pytest.mark.asyncio
async def test_manual_notion_sync_mirrors_markdown_without_exposing_token(tmp_path) -> None:
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        allow_test_backends=True,
    )
    settings.prepare_directories()
    database = Database(settings.database_path)
    await database.open()
    service = NotionService(
        settings,
        database,
        _CorpusStub(database),  # type: ignore[arg-type]
        client_factory=_FakeNotionClient,  # type: ignore[arg-type]
    )

    connection = await service.configure(
        NotionConnectionUpdateV1(
            access_token="secret_test_notion_token",
            root_page_ids=[],
            label="Work Notion",
        )
    )
    assert connection.configured is True
    assert connection.source is not None
    assert connection.source.provider == "notion"
    assert "secret" not in connection.model_dump_json()
    mode = stat.S_IMODE(settings.notion_config_path.stat().st_mode)
    assert mode == 0o600

    synced = await service.sync()
    assert synced.pages_fetched == 1
    assert synced.pages_written == 1
    assert synced.index_result is None
    files = list(settings.notion_mirror_dir.glob("*.md"))
    assert len(files) == 1
    assert "# Launch notes" in files[0].read_text("utf-8")
    assert "violet" in files[0].read_text("utf-8")

    unchanged = await service.sync()
    assert unchanged.pages_written == 0
    status = await service.status()
    assert status.page_count == 1
    assert status.last_synced_at is not None
    await database.close()


@pytest.mark.asyncio
async def test_sync_keeps_parent_page_when_unknown_child_is_not_shared(tmp_path) -> None:
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        allow_test_backends=True,
    )
    settings.prepare_directories()
    database = Database(settings.database_path)
    await database.open()
    service = NotionService(
        settings,
        database,
        _CorpusStub(database),  # type: ignore[arg-type]
        client_factory=_NotionClientWithInaccessibleChild,  # type: ignore[arg-type]
    )
    await service.configure(
        NotionConnectionUpdateV1(
            access_token="secret_test_notion_token",
            root_page_ids=[],
            label="Work Notion",
        )
    )

    synced = await service.sync()

    assert synced.pages_fetched == 1
    assert "Skipped 1 inaccessible linked item" in synced.message
    mirrored = next(settings.notion_mirror_dir.glob("*.md")).read_text("utf-8")
    assert "readable parent content remains available" in mirrored
    status = await service.status()
    assert status.last_error is None
    await database.close()


@pytest.mark.asyncio
async def test_sync_still_fails_when_explicit_root_page_is_not_shared(tmp_path) -> None:
    from waqil_api.notion import NotionError

    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        allow_test_backends=True,
    )
    settings.prepare_directories()
    database = Database(settings.database_path)
    await database.open()
    service = NotionService(
        settings,
        database,
        _CorpusStub(database),  # type: ignore[arg-type]
        client_factory=_NotionClientWithMissingExplicitRoot,  # type: ignore[arg-type]
    )
    await service.configure(
        NotionConnectionUpdateV1(
            access_token="secret_test_notion_token",
            root_page_ids=[PAGE_ID],
            label="Work Notion",
        )
    )

    with pytest.raises(NotionError, match="404"):
        await service.sync()

    status = await service.status()
    assert status.last_error is not None
    assert "404" in status.last_error
    await database.close()
