"""A supplied public URL is the requested source, even with a model query."""

from __future__ import annotations

from waqil_api.config import Settings
from waqil_api.web_research import WebResearch


async def test_explicit_url_does_not_launch_model_supplemental_search(monkeypatch) -> None:
    async def public_dns(url: str) -> bool:
        return True

    async def search_queries(self, client, queries):
        raise AssertionError("search should not run for a URL-only request")

    async def read_page(self, client, url):
        assert url == "https://example.com/release"
        return url, "Official release", "The released feature is described here."

    monkeypatch.setattr("waqil_api.web_research._public_dns", public_dns)
    monkeypatch.setattr(WebResearch, "_search_queries", search_queries)
    monkeypatch.setattr(WebResearch, "_read_page", read_page)
    snippets = await WebResearch(Settings()).retrieve(
        "Summarize https://example.com/release for me.",
        queries=["unrelated model-supplied search"],
    )
    assert len(snippets) == 1
    assert snippets[0].source_url == "https://example.com/release"
