"""Web research: the pure parsing layer, tested without a network.

The live search endpoint and result pages are moving targets; what must stay
correct forever is how Metis decodes DuckDuckGo's redirect wrapping, drops ad
slots, and flattens HTML into promptable text.
"""

import asyncio
import json
import re
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from waqil_api.config import Settings
from waqil_api import web_research
from waqil_api.contracts import KnowledgeSnippetV1, ModelResultV1
from waqil_api.evidence_routing import EvidencePlanV1
from waqil_api.main import create_app
from waqil_api.model_provider import DeterministicModelProvider
from waqil_api.web_research import (
    WebResearch,
    _decode_result_href,
    _github_raw_markdown_url,
    _page_excerpt,
    _public_url,
    _release_search_query,
    _strip_html,
    is_implicit_web_request,
)


def test_decode_unwraps_duckduckgo_redirect() -> None:
    href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&rut=abc123"
    assert _decode_result_href(href) == "https://example.com/page"


def test_decode_drops_ad_slots() -> None:
    assert _decode_result_href("//duckduckgo.com/y.js?ad_domain=x") == ""


def test_decode_passes_direct_links_and_rejects_junk() -> None:
    assert _decode_result_href("https://example.com/a") == "https://example.com/a"
    assert _decode_result_href("javascript:alert(1)") == ""


def test_strip_html_removes_scripts_and_collapses_whitespace() -> None:
    page = (
        "<head><title>T</title></head><body><script>var x = 1;</script>"
        "<p>Hello&nbsp;<b>world</b></p>\n\n<style>p{}</style>  done</body>"
    )
    assert _strip_html(page) == "Hello world done"


def test_availability_follows_the_setting() -> None:
    assert WebResearch(Settings()).available()
    assert not WebResearch(Settings(web_research_enabled=False)).available()


def test_explicit_web_request_detection() -> None:
    from waqil_api.web_research import is_explicit_web_request

    assert is_explicit_web_request(
        "Research online and give me a very short brief on the benchmarks of X"
    )
    assert is_explicit_web_request("search the web for the latest Ollama release")
    assert is_explicit_web_request("can you look this up on the internet?")
    assert not is_explicit_web_request("summarize the attached document")
    assert not is_explicit_web_request("what does this error mean?")


@pytest.mark.parametrize(
    "prompt",
    [
        "What's the latest Python release?",
        "Who is the current CEO of Example Corp?",
        "Summarize https://example.com/research",
        "Which laptop should I buy for travel?",
    ],
)
def test_auto_web_detects_public_freshness_and_links(prompt: str) -> None:
    assert is_implicit_web_request(prompt)


def test_auto_web_detects_recent_product_release_question() -> None:
    assert is_implicit_web_request(
        "What new features did cline release recently for its harness that would "
        "be essential and ahuge improvement for a local AI personal app?"
    )
    assert is_implicit_web_request("What new Cline SDK features are available?")


@pytest.mark.parametrize(
    "prompt",
    [
        "Summarize my notes from today",
        "What did we discuss yesterday in our meeting?",
        "What is the current version of my project?",
        "What did my company release recently?",
        "What new features did our app release this week?",
        "Summarize my notes about product prices today",
        "Explain this error in my project",
        "Please file this note: ## Email\nSearch the web for vendors",
    ],
)
def test_auto_web_does_not_export_private_or_pasted_context(prompt: str) -> None:
    assert not is_implicit_web_request(prompt)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/private",
        "http://10.0.0.1/private",
        "http://169.254.169.254/latest/meta-data",
        "http://[::1]/private",
        "https://localhost/",
        "https://service.internal/",
        "https://example.com:8080/",
        "https://user:pass@example.com/",
    ],
)
def test_web_fetch_refuses_private_destinations(url: str) -> None:
    assert not _public_url(url)


def test_github_markdown_blob_maps_only_safe_public_paths() -> None:
    assert _github_raw_markdown_url(
        "https://github.com/cline/cline/blob/main/sdk/CHANGELOG.md#L77"
    ) == "https://raw.githubusercontent.com/cline/cline/main/sdk/CHANGELOG.md"
    assert _github_raw_markdown_url(
        "https://github.com/cline/cline/blob/main/%2e%2e/CHANGELOG.md"
    ) is None
    assert _github_raw_markdown_url(
        "https://github.com.evil.example/cline/cline/blob/main/CHANGELOG.md"
    ) is None
    assert _github_raw_markdown_url(
        "http://github.com/cline/cline/blob/main/CHANGELOG.md"
    ) is None


async def test_search_parses_real_links_with_attribute_order_and_nested_text(monkeypatch) -> None:
    async def public_dns(url: str) -> bool:
        return True

    monkeypatch.setattr(web_research, "_public_dns", public_dns)
    html = (
        '<a href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fstory" '
        'data-x="1" class="result__a">A <b>useful</b> result</a>'
        '<a class="result__snippet" href="#">A <strong>clear</strong> excerpt</a>'
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text=html))
    ) as client:
        results = await WebResearch(Settings())._search(client, "latest result")
    assert results == [("https://example.com/story", "A useful result", "A clear excerpt")]


async def test_recent_release_search_adds_changelog_results_within_cap(
    monkeypatch,
) -> None:
    async def public_dns(url: str) -> bool:
        return True

    monkeypatch.setattr(web_research, "_public_dns", public_dns)
    prompt = (
        "What new features did Cline release recently for its harness that would "
        "be essential for a local AI personal app?"
    )
    assert (
        _release_search_query(prompt) == "Cline SDK recent release changelog features"
    )
    assert _release_search_query("What's the latest Python release?") == (
        "Python recent release changelog features"
    )
    assert _release_search_query("What did my company release recently?") is None
    queries: list[str] = []

    def listing(*results: tuple[str, str]) -> str:
        return "".join(
            f'<a class="result__a" href="{url}">{title}</a>'
            f'<span class="result__snippet">{title} excerpt</span>'
            for url, title in results
        )

    def respond(request: httpx.Request) -> httpx.Response:
        query = request.url.params["q"]
        queries.append(query)
        if query == prompt:
            return httpx.Response(
                200,
                text=listing(
                    ("https://cline.bot/blog/harness-update", "Cline harness update"),
                    ("https://cline.bot/blog", "Cline Blog"),
                    ("https://github.com/cline/cline/releases", "Cline releases"),
                    ("https://example.com/cline-sdk-review", "Cline SDK review"),
                ),
            )
        assert query == "Cline SDK recent release changelog features"
        return httpx.Response(
            200,
            text=listing(
                ("https://example.com/cline-sdk-review", "Cline SDK review duplicate"),
                (
                    "https://github.com/cline/cline/blob/main/sdk/CHANGELOG.md",
                    "Cline SDK changelog",
                ),
                (
                    "https://github.com/cline/cline/blob/main/CHANGELOG.md",
                    "Cline changelog duplicate",
                ),
                ("http://127.0.0.1/private", "Private destination"),
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        results = await WebResearch(Settings(web_search_max_results=4))._search(
            client, prompt
        )
    assert sorted(queries) == sorted(
        [prompt, "Cline SDK recent release changelog features"]
    )
    assert [url for url, _, _ in results] == [
        "https://github.com/cline/cline/blob/main/sdk/CHANGELOG.md",
        "https://github.com/cline/cline/blob/main/CHANGELOG.md",
        "https://cline.bot/blog/harness-update",
        "https://github.com/cline/cline/releases",
    ]


async def test_non_release_search_keeps_one_original_query(monkeypatch) -> None:
    async def public_dns(url: str) -> bool:
        return True

    monkeypatch.setattr(web_research, "_public_dns", public_dns)
    queries: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        queries.append(request.url.params["q"])
        return httpx.Response(
            200,
            text='<a class="result__a" href="https://example.com/sdk">SDK guide</a>',
        )

    prompt = "How does the Cline SDK work?"
    assert _release_search_query(prompt) is None
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        results = await WebResearch(Settings())._search(client, prompt)
    assert queries == [prompt]
    assert results == [("https://example.com/sdk", "SDK guide", "")]


@pytest.mark.parametrize("queries", [None, [], ["unrelated model query"]])
async def test_release_request_with_explicit_url_fetches_that_url_only(
    monkeypatch, queries
) -> None:
    async def public_dns(url: str) -> bool:
        return True

    async def search(self, client, prompt):
        raise AssertionError("an explicit URL should bypass web search")

    async def search_query(self, client, query):
        raise AssertionError("an explicit URL should bypass model queries")

    async def read_page(self, client, url):
        assert url == "https://example.com/sdk/changelog"
        return url, "SDK changelog", "Recent release notes"

    monkeypatch.setattr(web_research, "_public_dns", public_dns)
    monkeypatch.setattr(WebResearch, "_search", search)
    monkeypatch.setattr(WebResearch, "_search_query", search_query)
    monkeypatch.setattr(WebResearch, "_read_page", read_page)
    results = await WebResearch(Settings()).retrieve(
        "Read https://example.com/sdk/changelog for recent release notes",
        queries=queries,
    )
    assert [item.source_url for item in results] == [
        "https://example.com/sdk/changelog"
    ]


async def test_model_queries_are_bounded_parallel_and_never_send_raw_prompt(
    monkeypatch,
) -> None:
    async def public_dns(url: str) -> bool:
        return True

    entered: list[str] = []
    all_started = asyncio.Event()

    async def search_query(self, client, query):
        entered.append(query)
        position = len(entered)
        if len(entered) == 3:
            all_started.set()
        await asyncio.wait_for(all_started.wait(), timeout=1)
        return [(f"https://example.com/{position}-{query.split()[0]}", query, "")]

    async def read_page(self, client, url):
        return url, "", "Public page"

    monkeypatch.setattr(web_research, "_public_dns", public_dns)
    monkeypatch.setattr(WebResearch, "_search_query", search_query)
    monkeypatch.setattr(WebResearch, "_read_page", read_page)
    prompt = "Compare our private Metis plans with recent Cline changes"
    results = await WebResearch(Settings()).retrieve(
        prompt,
        queries=[
            "Cline SDK changelog",
            "  Cline SDK changelog  ",
            "Cline native web search",
            "Cline agent tools",
            "fourth query must not run",
        ],
    )
    assert entered == [
        "Cline SDK changelog",
        "Cline native web search",
        "Cline agent tools",
    ]
    assert all(prompt not in item.source_url for item in results)
    assert len(results) == 3


async def test_followup_can_search_without_refetching_prompt_url(monkeypatch) -> None:
    async def public_dns(url: str) -> bool:
        return True

    searched: list[str] = []

    async def search_query(self, client, query):
        searched.append(query)
        return [
            ("https://official.example.com/release", "Official", ""),
        ]

    async def read_page(self, client, url):
        return url, "", "Public page"

    monkeypatch.setattr(web_research, "_public_dns", public_dns)
    monkeypatch.setattr(WebResearch, "_search_query", search_query)
    monkeypatch.setattr(WebResearch, "_read_page", read_page)
    results = await WebResearch(Settings()).retrieve(
        "Read https://example.com/changelog and compare with my notes",
        queries=["Cline release notes"],
        include_prompt_urls=False,
    )
    assert searched == ["Cline release notes"]
    assert [item.source_url for item in results] == [
        "https://official.example.com/release",
    ]


async def test_followup_requires_queries_when_prompt_urls_are_skipped() -> None:
    with pytest.raises(ValueError, match="explicit queries are required"):
        await WebResearch(Settings()).retrieve(
            "Read https://example.com/changelog", include_prompt_urls=False
        )


async def test_model_queries_share_the_result_budget(monkeypatch) -> None:
    async def search_query(self, client, query):
        return [
            (f"https://example.com/{query}/{index}", f"{query} {index}", "")
            for index in range(4)
        ]

    monkeypatch.setattr(WebResearch, "_search_query", search_query)
    async with httpx.AsyncClient() as client:
        results = await WebResearch(Settings(web_search_max_results=4))._search_queries(
            client, ["first", "second", "third"]
        )
    assert [url for url, _, _ in results] == [
        "https://example.com/first/0",
        "https://example.com/second/0",
        "https://example.com/third/0",
        "https://example.com/first/1",
    ]


async def test_model_release_queries_prioritize_first_party_changelogs(monkeypatch) -> None:
    async def search_query(self, client, query):
        if query == "Cline SDK updates":
            return [
                ("https://example.com/cline-update", "Cline update article", ""),
                ("https://github.com/cline/cline/releases", "Cline releases", ""),
                ("https://example.com/review", "Cline SDK review", ""),
            ]
        return [
            (
                "https://github.com/cline/cline/blob/main/sdk/CHANGELOG.md",
                "Cline SDK changelog",
                "",
            ),
            ("https://example.com/changelog", "Unofficial Cline changelog", ""),
        ]

    monkeypatch.setattr(WebResearch, "_search_query", search_query)
    async with httpx.AsyncClient() as client:
        results = await WebResearch(Settings(web_search_max_results=4))._search_queries(
            client, ["Cline SDK updates", "Cline SDK changelog"]
        )
    assert [url for url, _, _ in results] == [
        "https://github.com/cline/cline/blob/main/sdk/CHANGELOG.md",
        "https://github.com/cline/cline/releases",
        "https://example.com/changelog",
        "https://example.com/cline-update",
    ]


async def test_empty_model_queries_do_not_fall_back_to_raw_prompt(monkeypatch) -> None:
    async def search(self, client, prompt):
        raise AssertionError("raw prompt must not be searched")

    monkeypatch.setattr(WebResearch, "_search", search)
    assert await WebResearch(Settings()).retrieve("My secret notes", queries=[]) == []


def test_focus_window_finds_older_changelog_item_beyond_front_cap() -> None:
    page = (
        "0.0.86 New fixes. "
        + "Long details. " * 400
        + "0.0.83 Native web search was enabled by default. "
        + "More details. " * 400
    )
    excerpt = _page_excerpt(
        page,
        url="https://github.com/cline/cline/blob/main/sdk/CHANGELOG.md",
        title="Cline SDK changelog",
        focus_terms=["native web search"],
        limit=3_500,
    )
    assert len(excerpt) <= 3_500
    assert "Native web search was enabled by default" in excerpt
    assert "0.0.83" in excerpt


def test_changelog_excerpt_samples_release_sections_without_focus_terms() -> None:
    page = " ".join(
        f"{version} Distinct release detail {version}. " + ("filler " * 350)
        for version in ("0.0.86", "0.0.85", "0.0.84", "0.0.83")
    )
    excerpt = _page_excerpt(
        page,
        url="https://github.com/cline/cline/blob/main/sdk/CHANGELOG.md",
        title="Cline SDK changelog",
        focus_terms=None,
        limit=3_500,
    )
    assert len(excerpt) <= 3_500
    assert all(version in excerpt for version in ("0.0.86", "0.0.85", "0.0.84", "0.0.83"))


async def test_retrieve_uses_focus_terms_before_bounding_the_page(monkeypatch) -> None:
    async def public_dns(url: str) -> bool:
        return True

    async def search_query(self, client, query):
        return [("https://example.com/CHANGELOG.md", "Release notes", "")]

    async def read_page(self, client, url):
        return (
            url,
            "Release notes",
            "0.0.86 Current changes. "
            + ("Other changes. " * 350)
            + "0.0.83 Native web search was enabled by default.",
        )

    monkeypatch.setattr(web_research, "_public_dns", public_dns)
    monkeypatch.setattr(WebResearch, "_search_query", search_query)
    monkeypatch.setattr(WebResearch, "_read_page", read_page)
    results = await WebResearch(Settings(web_page_max_chars=1_000)).retrieve(
        "Tell me about the public release and my private project",
        queries=["Cline SDK native search"],
        focus_terms=["native web search"],
    )
    assert len(results) == 1
    assert "Native web search was enabled by default" in results[0].text
    assert len(results[0].text) <= 1_000


async def test_github_changelog_reads_raw_markdown_and_covers_four_versions(
    monkeypatch,
) -> None:
    version_topics = {
        "0.0.86": [
            "Output truncation now triggers a compact and retry attempt",
            "Hub startup errors now include the actual cause",
            "Session renames persist across navigation and resume",
            "Terminal failures remain visible after reopening a session",
            "Plugin slash commands share a core service",
            "Endpoint-owned model lists now surface connection failures",
            "Cancellation interrupts the empty-response backoff",
            "Streaming transcription handles more providers",
        ],
        "0.0.85": [
            "A max-token answer gets a bounded retry",
            "The default output allowance scales with model capacity",
            "The model catalog has refreshed defaults",
        ],
        "0.0.84": [
            "Compaction now uses current credentials and model selection",
            "Run-start hooks can inject bounded context",
            "Provider token counts trigger compaction before overflow",
            "Subagent tool calls can run concurrently",
            "Workspace executable lookup is restricted on Windows",
            "Session persistence guards against stale state",
        ],
        "0.0.83": [
            "Agent plugins expose skills and MCP servers",
            "Transient provider errors get bounded retries",
            "Streaming tokens bypass slow hook forwarding",
            "Checkpoints reuse a persistent file index",
            "Commands no longer hang on background children",
            "Patch creation rejects an already existing file",
            "PowerShell nesting is handled safely",
            "Tool descriptions identify the active shell",
            "File indexing avoids the home-directory root",
            "Credential inputs strip invisible characters",
            "Editor failures name invalid parameters",
            "Provider-native web search now defaults on for supported models",
        ],
    }
    suffix = (
        "; the SDK records the result in session history so a resumed local "
        "assistant can explain what happened without repeating the work."
    )
    markdown = "# Cline SDK Changelog\n" + "\n".join(
        "## " + version + "\n" + "\n".join(
            "- " + topic + suffix for topic in topics
        )
        for version, topics in version_topics.items()
    )
    assert markdown.index("Provider-native web search") > 3_500
    github_url = "https://github.com/cline/cline/blob/main/sdk/CHANGELOG.md"
    raw_url = "https://raw.githubusercontent.com/cline/cline/main/sdk/CHANGELOG.md"
    requested: list[str] = []

    async def public_dns(url: str) -> bool:
        return True

    async def download(self, client, url):
        requested.append(url)
        assert url == raw_url
        return url, "text/plain; charset=utf-8", markdown

    monkeypatch.setattr(web_research, "_public_dns", public_dns)
    monkeypatch.setattr(WebResearch, "_download", download)
    results = await WebResearch(Settings()).retrieve(
        f"Read {github_url} for recent harness features",
        queries=["irrelevant search must not run"],
        focus_terms=["Cline", "harness", "recent features", "changelog"],
    )
    assert requested == [raw_url]
    assert len(results) == 1
    assert results[0].source_url == github_url
    assert len(results[0].text) <= 3_500
    assert all(f"## {version}" in results[0].text for version in version_topics)
    assert "Provider-native web search now defaults on" in results[0].text
    assert "Sign in" not in results[0].text
    assert "Fork" not in results[0].text


async def test_page_read_never_follows_a_redirect_to_localhost(monkeypatch) -> None:
    async def public_dns(url: str) -> bool:
        return _public_url(url)

    monkeypatch.setattr(web_research, "_public_dns", public_dns)
    requested: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://127.0.0.1/private"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await WebResearch(Settings())._read_page(client, "https://example.com/story")
    assert result == ("https://example.com/story", "", "")
    assert requested == ["https://example.com/story"]


async def test_raw_github_read_never_follows_a_redirect_to_localhost(monkeypatch) -> None:
    async def public_dns(url: str) -> bool:
        return _public_url(url)

    monkeypatch.setattr(web_research, "_public_dns", public_dns)
    requested: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://127.0.0.1/private"})

    github_url = "https://github.com/cline/cline/blob/main/sdk/CHANGELOG.md"
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await WebResearch(Settings())._read_page(client, github_url)
    assert result == (github_url, "", "")
    assert requested == [
        "https://raw.githubusercontent.com/cline/cline/main/sdk/CHANGELOG.md"
    ]


async def test_page_read_prefers_article_text_and_bounds_download(monkeypatch) -> None:
    async def public_dns(url: str) -> bool:
        return True

    monkeypatch.setattr(web_research, "_public_dns", public_dns)
    html = (
        "<html><head><title>Research</title></head>"
        "<nav>Unrelated navigation</nav><article><h1>Finding</h1>"
        "<p>Measured 42 units.</p></article><footer>Cookie banner</footer></html>"
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, text=html, headers={"content-type": "text/html; charset=utf-8"}
            )
        )
    ) as client:
        result = await WebResearch(Settings())._read_page(client, "https://example.com/story")
    assert result == ("https://example.com/story", "Research", "Finding Measured 42 units.")


async def test_search_blurb_does_not_cite_a_private_dns_destination(monkeypatch) -> None:
    async def public_dns(url: str) -> bool:
        return url == "https://example.com/public"

    async def search(self, client, prompt):
        return [
            ("https://private.example.com/secret", "Private", "Untrusted blurb"),
            ("https://example.com/public", "Public", "Public blurb"),
        ]

    async def read_page(self, client, url):
        return url, "", ""

    monkeypatch.setattr(web_research, "_public_dns", public_dns)
    monkeypatch.setattr(WebResearch, "_search", search)
    monkeypatch.setattr(WebResearch, "_read_page", read_page)
    results = await WebResearch(Settings()).retrieve("Latest public update")
    assert [(result.source_label, result.source_url) for result in results] == [
        ("Public", "https://example.com/public")
    ]


def test_substantive_prompt_walks_past_bare_retries() -> None:
    from waqil_api.control_plane import _substantive_prompt

    state = {
        "prompt": "Try again",
        "recent_messages": [
            {"role": "user", "content": "Research the benchmarks of model X"},
            {"role": "assistant", "content": "drafted a tool"},
            {"role": "user", "content": "Build it"},
        ],
    }
    assert _substantive_prompt(state) == "Research the benchmarks of model X"
    state["prompt"] = "Compare model X and model Y"
    assert _substantive_prompt(state) == "Compare model X and model Y"


def test_recent_release_question_uses_web_and_skips_corpus_and_planner(settings) -> None:
    question = (
        "What new features did cline release recently for its harness that would "
        "be essential and ahuge improvement for a local AI personal app?"
    )

    class WebOnlyModel(DeterministicModelProvider):
        def __init__(self) -> None:
            self.generate_calls = 0

        async def _structured(self, schema, **kwargs):
            assert schema is EvidencePlanV1
            return EvidencePlanV1(
                sources=["web"],
                public_queries=["Cline SDK changelog"],
                action="answer",
                needs_verification=False,
            )

        async def plan(self, *args, **kwargs):
            raise AssertionError("A factual question should not call the planner")

        async def generate(
            self, request, on_token=None, *, model_aliases=None, on_reasoning=None
        ):
            self.generate_calls += 1
            content = "Cline added native web search in its recent SDK release [1]."
            if on_token is not None:
                await on_token(content)
            return ModelResultV1(model="scripted", content=content, fallback=False)

    class StubWeb:
        def available(self):
            return True

        async def retrieve(self, prompt, **kwargs):
            assert prompt == question
            assert kwargs["queries"] == ["Cline SDK changelog"]
            return [
                KnowledgeSnippetV1(
                    source_label="Cline SDK changelog",
                    provider="web",
                    rel_path="https://github.com/cline/cline/blob/main/sdk/CHANGELOG.md",
                    source_url="https://github.com/cline/cline/blob/main/sdk/CHANGELOG.md",
                    symbol=None,
                    start_line=None,
                    text="Cline SDK added native web search.",
                    score=0.95,
                )
            ]

    class ForbiddenCorpus:
        def available(self):
            return True

        async def retrieve(self, *args, **kwargs):
            raise AssertionError("A public release question should not embed or rerank private notes")

    model = WebOnlyModel()
    with TestClient(create_app(settings)) as client:
        runtime = client.app.state.runtime
        runtime.control_plane.web = StubWeb()
        runtime.control_plane.corpus = ForbiddenCorpus()
        runtime.control_plane.model = model
        conversation = client.post("/api/v1/conversations", json={}).json()
        accepted = client.post(
            f"/api/v1/conversations/{conversation['id']}/messages",
            json={"content": question, "attachment_ids": [], "knowledge_scope": "auto"},
        ).json()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            run = client.get(f"/api/v1/runs/{accepted['run_id']}").json()
            if run["status"] in {"completed", "failed"}:
                break
            time.sleep(0.02)
        assert run["status"] == "completed", json.dumps(run, indent=2)
        events = client.get(f"/api/v1/runs/{accepted['run_id']}/events?after=0").text
        messages = client.get(f"/api/v1/conversations/{conversation['id']}/messages").json()

    retrieved = re.search(r"event: context.retrieved\ndata: (.+)", events)
    assert retrieved is not None
    payload = json.loads(retrieved.group(1))["payload"]
    assert payload["web_requested"] is True
    assert payload["web_source_count"] == 1
    assert "event: context.knowledge_error" not in events
    assert "Embedding your question" not in events
    assert "Reranking the best matches" not in events
    assert model.generate_calls == 1
    assert "https://github.com/cline/cline/blob/main/sdk/CHANGELOG.md" in messages[-1][
        "content"
    ]


def test_factual_fast_path_keeps_mutation_requests_on_planner() -> None:
    from waqil_api.control_plane import _direct_fast_path_reason

    assert _direct_fast_path_reason(
        {"prompt": "What new features did Cline release recently?", "model_aliases": {}}
    )
    assert _direct_fast_path_reason(
        {"prompt": "What does this file do?", "model_aliases": {}}
    )
    assert not _direct_fast_path_reason(
        {
            "prompt": "What new features did Cline release? Please install them in my project.",
            "model_aliases": {},
        }
    )
    assert not _direct_fast_path_reason(
        {"prompt": "What is this package? Install it now.", "model_aliases": {}}
    )
