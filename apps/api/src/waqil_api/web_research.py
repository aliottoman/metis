"""Bounded web retrieval for chat answers.

The web is an evidence source, not an authority over Metis. Search results and
page text are passed to the answer graph as citable, untrusted snippets. Auto
uses this lane for clear public/freshness requests; the Web scope always uses it.
"""

from __future__ import annotations

import asyncio
import html as html_module
import ipaddress
import re
import socket
from html.parser import HTMLParser
from urllib.parse import parse_qs, quote_plus, urljoin, urlsplit

import httpx

from .config import Settings
from .contracts import KnowledgeSnippetV1
from .prompt_scope import user_instruction

_SEARCH_ENDPOINT = "https://html.duckduckgo.com/html/?q={query}"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
_MAX_DOWNLOAD_BYTES = 900_000
_MAX_REDIRECTS = 3
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.DOTALL | re.IGNORECASE)
_MAIN = re.compile(r"<(?:main|article)\b[^>]*>(.*?)</(?:main|article)>", re.DOTALL | re.IGNORECASE)
_DROP_BLOCKS = re.compile(
    r"<(script|style|noscript|svg|head|nav|footer|form|iframe)\b.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
_TAG = re.compile(r"<[^>]+>")
_PROMPT_URL = re.compile(r"https?://[^\s<>\"')\]]+", re.IGNORECASE)
_PRIVATE_CONTEXT = re.compile(
    r"\b(?:my|our|this)\s+(?:notes?|meetings?|conversations?|projects?|"
    r"workspace|customers?|accounts?|tasks?|calendar|documents?|files?|"
    r"company|business|organization|team|apps?)\b",
    re.IGNORECASE,
)
_FRESHNESS = re.compile(
    r"\b(?:latest|newest|recent(?:ly)?|breaking|up[ -]to[ -]date|today|tonight|"
    r"yesterday|tomorrow|this week|this month|this year|as of now|"
    r"new(?:ly)? (?:[\w-]+ ){0,3}(?:features?|versions?|releases?|updates?)|"
    r"(?:released|announced|updated) (?:recently|lately)|changelog)\b",
    re.IGNORECASE,
)
_PUBLIC_LOOKUP = re.compile(
    r"\b(?:what(?:'s| is)|who(?:'s| is)|when is|how much is|which)\b"
    r"[^?\n]{0,100}\b(?:current|now|price|weather|forecast|ceo|president|"
    r"prime minister|exchange rate|release date)\b",
    re.IGNORECASE,
)
_SHOPPING_OR_TRAVEL = re.compile(
    r"\b(?:recommend|best|compare|which)\b[^?\n]{0,100}"
    r"\b(?:buy|purchase|book|visit|travel|hotel|restaurant|laptop|phone|"
    r"product|vendor|service)\b",
    re.IGNORECASE,
)
_EXPLICIT_WEB = (
    re.compile(
        r"\b(?:search|research|look\s+(?:it|this|that|them|these)?\s*up|check|find|browse)\b"
        r"[^.?!\n]{0,50}\b(?:online|on\s+the\s+(?:web|internet)|the\s+web|the\s+internet)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:search|scan|check)\s+the\s+(?:web|internet)\b", re.IGNORECASE),
    re.compile(r"\b(?:web|internet)\s+search\b", re.IGNORECASE),
    re.compile(
        r"\b(?:google|research\s+online|search\s+online|look\s+online)\b", re.IGNORECASE
    ),
)
_RELEASE_SUBJECT = (
    re.compile(
        r"\b(?:did|has)\s+([a-z][\w.+-]*(?:\s+[a-z][\w.+-]*){0,2})"
        r"\s+(?:release|ship|announce)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:latest|newest|recent)\s+"
        r"([a-z][\w.+-]*(?:\s+[a-z][\w.+-]*){0,2})\s+"
        r"(?:release|version|update|changelog)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b([a-z][\w.+-]*(?:\s+[a-z][\w.+-]*){0,2})\s+"
        r"(?:latest|newest|recent)\s+(?:release|version|update|changelog)\b",
        re.IGNORECASE,
    ),
)
_RELEASE_WORD = re.compile(
    r"\b(?:releases?|released|versions?|updates?|changelog)\b", re.IGNORECASE
)
_RELEASE_DOCUMENT = re.compile(r"\b(?:changelog|release[-_/ ]?notes?)\b", re.IGNORECASE)
_RELEASE_LOW_SIGNAL = re.compile(r"\b(?:reviews?|social)\b", re.IGNORECASE)
_SOCIAL_HOSTS = {"x.com", "twitter.com", "reddit.com", "facebook.com", "linkedin.com"}
_RELEASE_GENERIC_SUBJECT = {"it", "they", "we", "you", "this", "that", "a", "the"}


def is_explicit_web_request(prompt: str) -> bool:
    """A direct request in the user's instruction, excluding pasted material."""
    instruction = user_instruction(prompt)
    return any(pattern.search(instruction) for pattern in _EXPLICIT_WEB)


def is_implicit_web_request(prompt: str) -> bool:
    """Clear cases where a useful answer needs live public information.

    Keep this deterministic and conservative. An ordinary question about the
    user's private data must stay on local retrieval even when it says "today".
    """
    instruction = user_instruction(prompt)
    if _PROMPT_URL.search(instruction):
        return True
    if _PRIVATE_CONTEXT.search(instruction):
        return False
    return bool(
        _FRESHNESS.search(instruction)
        or _PUBLIC_LOOKUP.search(instruction)
        or _SHOPPING_OR_TRAVEL.search(instruction)
    )


def _release_search_query(prompt: str) -> str | None:
    """Add one focused query for clearly named, recent public releases."""
    if _PROMPT_URL.search(prompt) or _PRIVATE_CONTEXT.search(prompt):
        return None
    if not _FRESHNESS.search(prompt) or not _RELEASE_WORD.search(prompt):
        return None
    for pattern in _RELEASE_SUBJECT:
        match = pattern.search(prompt)
        if match is None:
            continue
        subject = " ".join(match.group(1).split())
        if subject.casefold() in _RELEASE_GENERIC_SUBJECT or subject.split()[0].casefold() in {
            "my",
            "our",
            "your",
            "this",
            "that",
            "the",
        }:
            continue
        qualifier = (
            "SDK "
            if re.search(r"\b(?:sdk|harness)\b", prompt, re.IGNORECASE)
            and not re.search(r"\bsdk\b", subject, re.IGNORECASE)
            else ""
        )
        return f"{subject} {qualifier}recent release changelog features"
    return None


def _strip_html(fragment: str) -> str:
    text = _DROP_BLOCKS.sub(" ", fragment)
    text = _TAG.sub(" ", text)
    text = html_module.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _decode_result_href(href: str) -> str:
    if href.startswith("//"):
        href = f"https:{href}"
    parsed = urlsplit(html_module.unescape(href))
    if parsed.hostname and (
        parsed.hostname == "duckduckgo.com"
        or parsed.hostname.endswith(".duckduckgo.com")
    ):
        if "y.js" in parsed.path:
            return ""
        return parse_qs(parsed.query).get("uddg", [""])[0]
    return href if parsed.scheme in ("http", "https") else ""


def _public_url(url: str) -> bool:
    """Reject local, credentialed, malformed and non-web destinations."""
    if len(url) > 2_048 or re.search(r"[\s\\<>]", url):
        return False
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https") or not host or parsed.username:
        return False
    if port not in (None, 80, 443):
        return False
    if host.casefold() in {"localhost", "localhost.localdomain"} or host.casefold().endswith(
        (".localhost", ".local", ".internal", ".test", ".invalid")
    ):
        return False
    try:
        return ipaddress.ip_address(host).is_global
    except ValueError:
        return "." in host and not host.startswith(".") and not host.endswith(".")


async def _public_dns(url: str) -> bool:
    """Fail closed if a public-looking hostname resolves onto a private address."""
    if not _public_url(url):
        return False
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass
    try:
        resolved = await asyncio.wait_for(
            asyncio.to_thread(socket.getaddrinfo, host, parsed.port or 443, type=socket.SOCK_STREAM),
            timeout=3.0,
        )
    except (OSError, TimeoutError):
        return False
    return bool(resolved) and all(
        ipaddress.ip_address(item[4][0]).is_global for item in resolved
    )


class _SearchParser(HTMLParser):
    """Read results by class, independent of quote style and attribute order."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self.blurbs: list[str] = []
        self._kind = ""
        self._href = ""
        self._depth = 0
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._kind:
            if tag not in {"area", "br", "embed", "hr", "img", "input", "link", "meta", "source", "wbr"}:
                self._depth += 1
            return
        values = dict(attrs)
        classes = set((values.get("class") or "").split())
        if tag == "a" and "result__a" in classes:
            self._kind = "link"
            self._href = values.get("href") or ""
        elif "result__snippet" in classes:
            self._kind = "blurb"
        if self._kind:
            self._depth = 1
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._kind:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if not self._kind:
            return
        self._depth -= 1
        if self._depth:
            return
        value = " ".join(" ".join(self._text).split())
        if self._kind == "link":
            self.links.append((self._href, value))
        else:
            self.blurbs.append(value)
        self._kind = ""
        self._href = ""
        self._text = []


class WebResearch:
    """Search public pages and return bounded, linked evidence snippets."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def available(self) -> bool:
        return self.settings.web_research_enabled

    async def retrieve(self, prompt: str) -> list[KnowledgeSnippetV1]:
        instruction = user_instruction(prompt)
        urls = [url.rstrip(".,;:") for url in _PROMPT_URL.findall(instruction)]
        urls = urls[: self.settings.web_search_max_results]
        async with httpx.AsyncClient(
            headers=_HEADERS,
            follow_redirects=False,
            trust_env=False,
            timeout=self.settings.web_fetch_timeout_seconds,
        ) as client:
            found = [(url, "", "") for url in urls] if urls else await self._search(client, instruction)
            # Do not keep even a search-result blurb when the destination
            # resolves to a private host; failed page reads may use that blurb.
            allowed = await asyncio.gather(*(_public_dns(url) for url, _, _ in found))
            found = [item for item, public in zip(found, allowed) if public]
            pages = await asyncio.gather(
                *(self._read_page(client, url) for url, _, _ in found),
                return_exceptions=True,
            )
        snippets: list[KnowledgeSnippetV1] = []
        for rank, ((url, title, blurb), page) in enumerate(zip(found, pages)):
            final_url, page_title, body = page if isinstance(page, tuple) else (url, "", "")
            text = body or blurb
            if not text:
                continue
            label = title or page_title or urlsplit(final_url).netloc
            snippets.append(
                KnowledgeSnippetV1(
                    source_label=label[:200],
                    provider="web",
                    rel_path=final_url,
                    source_url=final_url,
                    symbol=None,
                    start_line=None,
                    text=text[: self.settings.web_page_max_chars],
                    score=round(0.95 - rank * 0.05, 2),
                )
            )
        return snippets

    async def _download(
        self, client: httpx.AsyncClient, url: str
    ) -> tuple[str, str, str]:
        """Return final URL, content type, and bounded text; validate every hop."""
        for _ in range(_MAX_REDIRECTS + 1):
            if not await _public_dns(url):
                raise ValueError("URL is not a public web address")
            async with client.stream("GET", url) as response:
                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get("location", "")
                    if not location:
                        raise ValueError("redirect has no destination")
                    url = urljoin(url, location)
                    continue
                response.raise_for_status()
                chunks: list[bytes] = []
                total = 0
                async for chunk in response.aiter_bytes():
                    remaining = _MAX_DOWNLOAD_BYTES - total
                    if remaining <= 0:
                        break
                    chunks.append(chunk[:remaining])
                    total += min(len(chunk), remaining)
                    if total >= _MAX_DOWNLOAD_BYTES:
                        break
                encoding = response.encoding or "utf-8"
                return url, response.headers.get("content-type", ""), b"".join(chunks).decode(
                    encoding, errors="replace"
                )
        raise ValueError("too many redirects")

    async def _search(
        self, client: httpx.AsyncClient, prompt: str
    ) -> list[tuple[str, str, str]]:
        supplemental = _release_search_query(prompt)
        if (
            supplemental is None
            or supplemental.casefold() == " ".join(prompt.split()).casefold()
        ):
            return await self._search_query(client, prompt)

        async def focused_search() -> list[tuple[str, str, str]]:
            try:
                return await self._search_query(client, supplemental)
            except (httpx.HTTPError, OSError, TimeoutError, ValueError):
                return []

        primary, focused = await asyncio.gather(
            self._search_query(client, prompt), focused_search()
        )
        wants_sdk = bool(re.search(r"\b(?:sdk|harness)\b", prompt, re.IGNORECASE))

        def label(item: tuple[str, str, str]) -> str:
            return f"{urlsplit(item[0]).path} {item[1]}"

        def low_signal(item: tuple[str, str, str]) -> bool:
            parsed = urlsplit(item[0])
            host = (parsed.hostname or "").casefold()
            index_page = parsed.path.casefold().rstrip("/") in {"/blog", "/news"}
            return index_page or bool(_RELEASE_LOW_SIGNAL.search(label(item))) or any(
                host == social or host.endswith(f".{social}") for social in _SOCIAL_HOSTS
            )

        # A focused query can surface a release document below generic links.
        # Put citable changelogs/notes first, especially SDK documentation when
        # the user asks about an SDK or harness. Preserve broad result order for
        # the remaining sources while leaving reviews/social links until last.
        documents = [
            (item, source, index)
            for source, results in ((0, focused), (1, primary))
            for index, item in enumerate(results)
            if _RELEASE_DOCUMENT.search(label(item)) and not low_signal(item)
        ]
        documents.sort(
            key=lambda entry: (
                0
                if wants_sdk
                and re.search(r"\bsdk\b", label(entry[0]), re.IGNORECASE)
                else 1,
                entry[1],
                entry[2],
            )
        )
        remainder = [
            item
            for results in (primary, focused)
            for item in results
            if not (_RELEASE_DOCUMENT.search(label(item)) and not low_signal(item))
        ]
        ordered = (
            [item for item, _, _ in documents]
            + [item for item in remainder if not low_signal(item)]
            + [item for item in remainder if low_signal(item)]
        )
        merged: list[tuple[str, str, str]] = []
        seen: set[str] = set()
        for item in ordered:
            if item[0] in seen:
                continue
            seen.add(item[0])
            merged.append(item)
            if len(merged) >= self.settings.web_search_max_results:
                break
        return merged

    async def _search_query(
        self, client: httpx.AsyncClient, prompt: str
    ) -> list[tuple[str, str, str]]:
        query = quote_plus(" ".join(prompt.split())[:300])
        if not query:
            return []
        _, _, document = await self._download(client, _SEARCH_ENDPOINT.format(query=query))
        parser = _SearchParser()
        parser.feed(document)
        results: list[tuple[str, str, str]] = []
        seen: set[str] = set()
        for index, (href, title) in enumerate(parser.links):
            url = _decode_result_href(href)
            if not _public_url(url) or url in seen:
                continue
            seen.add(url)
            blurb = parser.blurbs[index] if index < len(parser.blurbs) else ""
            results.append((url, title, blurb))
            if len(results) >= self.settings.web_search_max_results:
                break
        return results

    async def _read_page(self, client: httpx.AsyncClient, url: str) -> tuple[str, str, str]:
        try:
            final_url, content_type, document = await self._download(client, url)
        except (httpx.HTTPError, OSError, TimeoutError, ValueError):
            return url, "", ""
        if "html" not in content_type and "text" not in content_type:
            return final_url, "", ""
        title_match = _TITLE.search(document)
        title = _strip_html(title_match.group(1)) if title_match else ""
        main_match = _MAIN.search(document)
        body = _strip_html(main_match.group(1) if main_match else document)
        return final_url, title, body
