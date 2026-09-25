"""Web research: the pure parsing layer, tested without a network.

The live search endpoint and result pages are moving targets; what must stay
correct forever is how Metis decodes DuckDuckGo's redirect wrapping, drops ad
slots, and flattens HTML into promptable text.
"""

import httpx
import pytest

from waqil_api.config import Settings
from waqil_api import web_research
from waqil_api.web_research import (
    WebResearch,
    _decode_result_href,
    _public_url,
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


@pytest.mark.parametrize(
    "prompt",
    [
        "Summarize my notes from today",
        "What did we discuss yesterday in our meeting?",
        "What is the current version of my project?",
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
