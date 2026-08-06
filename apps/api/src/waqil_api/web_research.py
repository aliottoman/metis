"""Web research — the retrieve node's window onto the live internet.

Every other evidence lane (corpus, Notion mirror, customer records) reads
things the user already owns. This lane reaches out, so two properties keep
it honest:

* It runs only when the user explicitly chooses the Web scope for a message.
  Auto never silently ships a prompt to a search engine — the same consent
  posture cloud embedding takes.
* Results enter the answer as ordinary knowledge snippets with a ``web``
  provider, so citations, the grounding review, and the Sources list treat a
  web page exactly like any other evidence — and the reader gets a clickable
  URL instead of an unverifiable claim.

Search is DuckDuckGo's keyless HTML endpoint; page text comes from a
deliberately boring tag-stripper. Both choices trade fidelity for zero new
dependencies and zero API keys — good enough to ground an answer, and every
failure degrades to fewer snippets rather than a dead turn.
"""
from __future__ import annotations

import asyncio
import html as html_module
import re
from urllib.parse import parse_qs, quote_plus, urlparse

import httpx

from .config import Settings
from .contracts import KnowledgeSnippetV1

_SEARCH_ENDPOINT = "https://html.duckduckgo.com/html/?q={query}"

# A browser-shaped User-Agent: the HTML endpoint answers plain clients with an
# empty shell or a 403, and there is no keyless API alternative to fall back to.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

_RESULT_LINK = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL
)
_RESULT_SNIPPET = re.compile(
    r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', re.DOTALL
)
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.DOTALL | re.IGNORECASE)
_DROP_BLOCKS = re.compile(
    r"<(script|style|noscript|svg|head|nav|footer|form|iframe)\b.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
_TAG = re.compile(r"<[^>]+>")
_PROMPT_URL = re.compile(r"https?://[^\s<>\"')\]]+")


def _strip_html(fragment: str) -> str:
    """Collapse an HTML fragment to readable text. Boring on purpose: a real
    article extractor would be better prose but another dependency, and the
    grounding review only needs the facts to be present, not pretty."""
    text = _DROP_BLOCKS.sub(" ", fragment)
    text = _TAG.sub(" ", text)
    text = html_module.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _decode_result_href(href: str) -> str:
    """DuckDuckGo wraps every result behind ``//duckduckgo.com/l/?uddg=<url>``;
    the real destination is the ``uddg`` parameter. Ad slots route through
    ``y.js`` and carry no ``uddg`` — returning "" drops them."""
    if href.startswith("//"):
        href = f"https:{href}"
    parsed = urlparse(html_module.unescape(href))
    if parsed.netloc.endswith("duckduckgo.com"):
        if "y.js" in parsed.path:
            return ""
        uddg = parse_qs(parsed.query).get("uddg", [""])[0]
        return uddg
    return href if parsed.scheme in ("http", "https") else ""


class WebResearch:
    """Search the web and read pages, returning citable knowledge snippets."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def available(self) -> bool:
        return self.settings.web_research_enabled

    async def retrieve(self, prompt: str) -> list[KnowledgeSnippetV1]:
        """Evidence for one prompt: pages behind any URLs the user pasted, or
        the top search hits when they pasted none. A pasted URL is already the
        user saying "this page" — searching around it would bury it."""
        urls = _PROMPT_URL.findall(prompt)[: self.settings.web_search_max_results]
        async with httpx.AsyncClient(
            headers=_HEADERS,
            follow_redirects=True,
            timeout=self.settings.web_fetch_timeout_seconds,
        ) as client:
            if urls:
                found = [(url, "", "") for url in urls]
            else:
                found = await self._search(client, prompt)
            pages = await asyncio.gather(
                *(self._read_page(client, url) for url, _, _ in found),
                return_exceptions=True,
            )
        snippets: list[KnowledgeSnippetV1] = []
        for rank, ((url, title, blurb), page) in enumerate(zip(found, pages)):
            page_title, body = page if isinstance(page, tuple) else ("", "")
            # A page that would not load still has its search blurb — a thin
            # snippet beats silently shrinking the evidence set.
            text = body or blurb
            if not text:
                continue
            label = title or page_title or urlparse(url).netloc
            snippets.append(
                KnowledgeSnippetV1(
                    source_label=label[:200],
                    provider="web",
                    rel_path=url,
                    symbol=None,
                    start_line=None,
                    text=text[: self.settings.web_page_max_chars],
                    # Search rank restated as a monotone score, so anything
                    # downstream that sorts or displays by score stays sane.
                    score=round(0.95 - rank * 0.05, 2),
                )
            )
        return snippets

    async def _search(
        self, client: httpx.AsyncClient, prompt: str
    ) -> list[tuple[str, str, str]]:
        """Top (url, title, blurb) results for the prompt's first 300 chars —
        long prompts are conversations, not queries, and the engine only reads
        the head anyway."""
        query = quote_plus(" ".join(prompt.split())[:300])
        response = await client.get(_SEARCH_ENDPOINT.format(query=query))
        response.raise_for_status()
        links = _RESULT_LINK.findall(response.text)
        blurbs = [_strip_html(match) for match in _RESULT_SNIPPET.findall(response.text)]
        results: list[tuple[str, str, str]] = []
        seen: set[str] = set()
        for index, (href, title_html) in enumerate(links):
            url = _decode_result_href(href)
            if not url or url in seen:
                continue
            seen.add(url)
            blurb = blurbs[index] if index < len(blurbs) else ""
            results.append((url, _strip_html(title_html), blurb))
            if len(results) >= self.settings.web_search_max_results:
                break
        return results

    async def _read_page(
        self, client: httpx.AsyncClient, url: str
    ) -> tuple[str, str]:
        """(title, text) of one page. Non-HTML answers (PDFs, images) return
        empty text rather than binary soup in the prompt."""
        response = await client.get(url)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "html" not in content_type and "text" not in content_type:
            return "", ""
        document = response.text[:800_000]
        title_match = _TITLE.search(document)
        title = _strip_html(title_match.group(1)) if title_match else ""
        return title, _strip_html(document)
