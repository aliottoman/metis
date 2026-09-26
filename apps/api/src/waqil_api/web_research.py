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
_BRAVE_CONTEXT_ENDPOINT = "https://api.search.brave.com/res/v1/llm/context"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
_MAX_DOWNLOAD_BYTES = 900_000
_MAX_REDIRECTS = 3
_MAX_SEARCH_QUERIES = 3
_MAX_FOCUS_TERMS = 8
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
_RELEASES_PATH = re.compile(r"(?:^|/)releases?(?:/|$)", re.IGNORECASE)
_MODEL_RELEASE_INTENT = re.compile(
    r"\b(?:changelog|latest|newest|recent|releases?|shipped|since|versions?|updates?)\b",
    re.IGNORECASE,
)
_GENERIC_QUERY_WORDS = {
    "about", "changes", "docs", "features", "github", "harness", "latest",
    "native", "new", "notes", "official", "recent", "release", "releases", "search",
    "sdk", "shipped", "tool", "tools", "update", "updates", "version", "web",
}
_RELEASE_LOW_SIGNAL = re.compile(r"\b(?:reviews?|social)\b", re.IGNORECASE)
_SOCIAL_HOSTS = {"x.com", "twitter.com", "reddit.com", "facebook.com", "linkedin.com"}
_RELEASE_GENERIC_SUBJECT = {"it", "they", "we", "you", "this", "that", "a", "the"}
_VERSION = re.compile(r"(?<![\w.])v?\d{1,4}\.\d{1,3}(?:\.\d{1,3})?(?![\w.])", re.IGNORECASE)
_MARKDOWN_VERSION_HEADING = re.compile(
    r"^##\s+\[?(v?\d+(?:\.\d+){1,3})\]?[^\n]*$", re.MULTILINE | re.IGNORECASE
)
_CAPABILITY_WORDS = re.compile(
    r"\b(?:agent|compaction|connector|context|mcp|model|plugin|provider|"
    r"retry|search|session|stream|tool)s?\b",
    re.IGNORECASE,
)
_NEW_CAPABILITY = re.compile(
    r"\b(?:added|adds|enabled by default|introduces|new|now (?:allows|enabled|supports)|support for)\b",
    re.IGNORECASE,
)
_LOW_SIGNAL_FOCUS = {"change", "changes", "feature", "features", "latest", "new", "recent", "release", "releases", "update", "updates"}


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


def _github_raw_markdown_url(url: str) -> str | None:
    """Map a public GitHub Markdown blob to its raw text, without URL tricks."""
    if not _public_url(url):
        return None
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != "github.com":
        return None
    parts = parsed.path.strip("/").split("/")
    if len(parts) < 5 or parts[2] != "blob" or not parts[-1].casefold().endswith(".md"):
        return None
    if any(
        part in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", part)
        for part in parts
    ):
        return None
    return "https://raw.githubusercontent.com/" + "/".join(
        [parts[0], parts[1], parts[3], *parts[4:]]
    )


def _compact_markdown_changelog(
    text: str, *, focus_terms: list[str] | None, limit: int
) -> str | None:
    """Give several recent versions room, selecting substantive release bullets."""
    all_headings = list(_MARKDOWN_VERSION_HEADING.finditer(text))
    if not all_headings:
        return None
    count = min(len(all_headings), max(2, min(4, limit // 650)))
    headings = all_headings[:count]
    title_match = re.search(r"^#\s+([^\n]+)", text, re.MULTILINE)
    title = title_match.group(0)[:120] if title_match else ""
    title_words = set(re.findall(r"[a-z0-9]+", title.casefold()))
    section_budget = max(120, (limit - len(title) - 2 * count) // count)
    useful_focus = [
        " ".join(term.split()).casefold()
        for term in focus_terms or []
        if isinstance(term, str) and len(term.strip()) >= 4
        and not all(
            word in title_words
            for word in re.findall(r"[a-z0-9]+", term.casefold())
        )
        and term.casefold() not in {"new features", "recent features", "changelog"}
    ][:_MAX_FOCUS_TERMS]
    sections: list[str] = []
    for index, heading in enumerate(headings):
        next_start = (
            all_headings[index + 1].start()
            if index + 1 < len(all_headings)
            else len(text)
        )
        body = text[heading.end() : next_start]
        bullets: list[tuple[int, str, float]] = []
        for bullet_index, line in enumerate(body.splitlines()):
            match = re.match(r"^\s*[-*]\s+(.+)", line)
            if not match:
                continue
            content = " ".join(match.group(1).split())
            lowered = content.casefold()
            score = min(4, len(_CAPABILITY_WORDS.findall(content))) * 3
            score += sum(12 for term in useful_focus if term in lowered)
            score += 6 if _NEW_CAPABILITY.search(content) else 0
            score += 4 if not bullets else 0
            score -= 5 if "refreshed the model catalog" in lowered else 0
            bullets.append((bullet_index, content, score))

        heading_text = heading.group(0).strip()
        remaining = section_budget - len(heading_text) - 1
        chosen: list[tuple[int, str]] = []
        max_bullet = min(260, max(110, remaining // 2))
        for bullet_index, content, _ in sorted(
            bullets, key=lambda item: (-item[2], item[0])
        ):
            clipped = content[:max_bullet].rsplit(" ", 1)[0] if len(content) > max_bullet else content
            line = f"- {clipped}"
            if len(line) + 1 > remaining:
                continue
            chosen.append((bullet_index, line))
            remaining -= len(line) + 1
            if len(chosen) >= 4:
                break
        if not chosen and remaining > 10:
            fallback = " ".join(body.split())[:remaining]
            if fallback:
                chosen.append((0, fallback))
        section = "\n".join(
            [heading_text, *(line for _, line in sorted(chosen))]
        )
        sections.append(section)
    result = "\n\n".join(([title] if title else []) + sections)
    return result[:limit]


def _page_excerpt(
    text: str,
    *,
    url: str,
    title: str,
    focus_terms: list[str] | None,
    limit: int,
) -> str:
    """Keep relevant passages from a long page, including older release notes.

    A changelog's first few thousand characters often describe only its newest
    release. Extract short, separated windows so an older item found by a
    focused query can still be cited within the same bounded prompt budget.
    """
    if len(text) <= limit:
        return text

    if _RELEASE_DOCUMENT.search(f"{urlsplit(url).path} {title}") and "\n## " in text:
        compact = _compact_markdown_changelog(
            text, focus_terms=focus_terms, limit=limit
        )
        if compact:
            return compact

    folded = text.casefold()
    terms: list[str] = []
    for raw in focus_terms or []:
        if not isinstance(raw, str):
            continue
        term = " ".join(raw.split())[:80].casefold()
        if len(term) < 4 or term in _LOW_SIGNAL_FOCUS or term in terms:
            continue
        terms.append(term)
        if len(terms) >= _MAX_FOCUS_TERMS:
            break

    changelog = bool(_RELEASE_DOCUMENT.search(f"{urlsplit(url).path} {title}"))
    window_size = max(320, min(780, (limit - 320) // 4))
    candidates: list[tuple[float, int, int]] = []
    for term in terms:
        positions: list[int] = []
        cursor = 0
        while len(positions) < 40:
            position = folded.find(term, cursor)
            if position < 0:
                break
            positions.append(position)
            cursor = position + len(term)
        rarity = 60 / (1 + len(positions))
        for position in positions:
            start = max(0, position - min(210, window_size // 3))
            end = min(len(text), start + window_size)
            coverage = sum(other in folded[start:end] for other in terms)
            candidates.append((rarity + coverage * 8, start, end))

    if changelog:
        # Include several distinct release sections even when the model did
        # not provide a focus term. Version markers are positional hints, not
        # a claim that a particular release exists.
        for index, match in enumerate(_VERSION.finditer(text)):
            if index >= 12:
                break
            start = match.start()
            candidates.append((20 - index, start, min(len(text), start + window_size)))

    if not candidates:
        return text[:limit]

    selected: list[tuple[int, int]] = []
    budget = limit - min(300, limit // 6)
    for _, start, end in sorted(candidates, key=lambda item: (-item[0], item[1])):
        if any(start < chosen_end and end > chosen_start for chosen_start, chosen_end in selected):
            continue
        if end - start > budget:
            continue
        selected.append((start, end))
        budget -= end - start + 5
        if len(selected) >= 4 or budget < 320:
            break

    if not selected:
        return text[:limit]
    intro = text[: min(300, limit // 6)].rstrip()
    passages = [intro] if all(start >= len(intro) for start, _ in selected) else []
    passages.extend(text[start:end].strip() for start, end in sorted(selected))
    return " … ".join(passage for passage in passages if passage)[:limit]


def _model_result_priority(
    item: tuple[str, str, str], *, query_subjects: set[str], wants_sdk: bool
) -> tuple[int, int]:
    """Prefer first-party release records over articles for release research."""
    url, title, _ = item
    parsed = urlsplit(url)
    host = (parsed.hostname or "").casefold()
    path = parsed.path.casefold()
    label = f"{path} {title}"
    low_signal = (
        path.rstrip("/") in {"/blog", "/news"}
        or bool(_RELEASE_LOW_SIGNAL.search(label))
        or any(host == social or host.endswith(f".{social}") for social in _SOCIAL_HOSTS)
    )
    if low_signal:
        return (3, 0)
    release_document = bool(_RELEASE_DOCUMENT.search(label) or _RELEASES_PATH.search(path))
    if not release_document:
        return (2, 0)
    host_parts = host.split(".")
    # Match the registrable-looking domain label, not an arbitrary brand
    # subdomain such as "cline.evil.example".
    domain_word = host_parts[-2] if len(host_parts) >= 2 else ""
    path_parts = [part for part in path.strip("/").split("/") if part]
    repository_owner = path_parts[0] if host == "github.com" and path_parts else ""
    first_party = bool(query_subjects & {domain_word, repository_owner})
    sdk_match = wants_sdk and "sdk" in label
    return (0 if first_party else 1, 0 if sdk_match else 1)


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

    async def retrieve(
        self,
        prompt: str,
        *,
        queries: list[str] | None = None,
        focus_terms: list[str] | None = None,
        include_prompt_urls: bool = True,
    ) -> list[KnowledgeSnippetV1]:
        """Return public evidence for a turn.

        ``queries=None`` keeps the legacy user-prompt search for an explicit
        Web request. Supplying queries, including an empty list, opts out of
        that fallback. User URLs take precedence over search on the first pass;
        a later coverage check may search separately with
        ``include_prompt_urls=False`` and explicit queries.
        """
        if not include_prompt_urls and queries is None:
            raise ValueError("explicit queries are required when skipping prompt URLs")
        instruction = user_instruction(prompt)
        urls = (
            [url.rstrip(".,;:") for url in _PROMPT_URL.findall(instruction)]
            if include_prompt_urls
            else []
        )
        urls = urls[: self.settings.web_search_max_results]
        async with httpx.AsyncClient(
            headers=_HEADERS,
            follow_redirects=False,
            trust_env=False,
            timeout=self.settings.web_fetch_timeout_seconds,
        ) as client:
            # Brave returns extracted passages with their source URLs. Keep
            # those passages as evidence directly: fetching every result again
            # would add latency and can replace a useful excerpt with page chrome.
            # Direct user URLs still go through the host's validated page reader.
            if not urls and self.settings.brave_search_api_key.strip():
                if queries is None:
                    focused = _release_search_query(instruction)
                    brave_queries = (
                        [focused, instruction] if focused and focused != instruction
                        else [instruction]
                    )
                else:
                    brave_queries = queries
                return await self._brave_search_queries(
                    client, brave_queries, focus_terms=focus_terms
                )
            found = [(url, "", "") for url in urls]
            if not urls:
                if queries is not None:
                    # An explicit empty list means no search. Never fall back
                    # to sending the user's full prompt to the public engine.
                    found.extend(await self._search_queries(client, queries))
                else:
                    found = await self._search(client, instruction)
            unique: list[tuple[str, str, str]] = []
            seen: set[str] = set()
            for item in found:
                if item[0] in seen:
                    continue
                seen.add(item[0])
                unique.append(item)
                if len(unique) >= self.settings.web_search_max_results:
                    break
            found = unique
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
            text = _page_excerpt(
                text,
                url=final_url,
                title=label,
                focus_terms=focus_terms,
                limit=self.settings.web_page_max_chars,
            )
            snippets.append(
                KnowledgeSnippetV1(
                    source_label=label[:200],
                    provider="web",
                    rel_path=final_url,
                    source_url=final_url,
                    symbol=None,
                    start_line=None,
                    text=text,
                    score=round(0.95 - rank * 0.05, 2),
                )
            )
        return snippets

    async def _brave_context_query(
        self, client: httpx.AsyncClient, query: str
    ) -> list[tuple[str, str, str]]:
        """Read linked source passages from Brave's LLM Context API."""
        maximum_urls = self.settings.web_search_max_results
        response = await client.post(
            _BRAVE_CONTEXT_ENDPOINT,
            json={
                "q": query,
                "count": max(10, maximum_urls * 3),
                "maximum_number_of_urls": maximum_urls,
                "maximum_number_of_tokens": max(2048, maximum_urls * 1024),
                "maximum_number_of_tokens_per_url": 1024,
                "maximum_number_of_snippets": maximum_urls * 4,
            },
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": self.settings.brave_search_api_key.strip(),
            },
            timeout=self.settings.brave_search_timeout_seconds,
        )
        if response.is_redirect:
            raise ValueError("Brave Search unexpectedly redirected the request")
        response.raise_for_status()
        if len(response.content) > _MAX_DOWNLOAD_BYTES:
            raise ValueError("Brave Search returned an oversized response")
        try:
            payload = response.json()
        except ValueError as exc:
            raise ValueError("Brave Search returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("Brave Search returned an invalid response")
        grounding = payload.get("grounding")
        if grounding is None:
            return []
        if not isinstance(grounding, dict):
            raise ValueError("Brave Search returned invalid grounding")
        generic = grounding.get("generic") or []
        if not isinstance(generic, list):
            raise ValueError("Brave Search returned invalid grounding")
        results: list[tuple[str, str, str]] = []
        for item in generic[: maximum_urls * 3]:
            if not isinstance(item, dict):
                continue
            url = item.get("url")
            if not isinstance(url, str) or not _public_url(url):
                continue
            title = item.get("title")
            snippets = item.get("snippets")
            if not isinstance(snippets, list):
                continue
            passages = [
                text.strip()[: self.settings.web_page_max_chars]
                for text in snippets[:12]
                if isinstance(text, str) and text.strip()
            ]
            if not passages:
                continue
            results.append((url, str(title or "")[:200], "\n\n".join(passages)))
        return results

    async def _brave_search_queries(
        self,
        client: httpx.AsyncClient,
        queries: list[str],
        *,
        focus_terms: list[str] | None,
    ) -> list[KnowledgeSnippetV1]:
        """Search bounded queries and keep only public, cited passages."""
        normalized: list[str] = []
        seen_queries: set[str] = set()
        for raw in queries:
            if not isinstance(raw, str):
                continue
            query = " ".join(raw.split())[:300]
            if query and query.casefold() not in seen_queries:
                normalized.append(query)
                seen_queries.add(query.casefold())
            if len(normalized) >= _MAX_SEARCH_QUERIES:
                break
        if not normalized:
            return []
        # All queries are already public-only in Auto. A configured Brave
        # failure is surfaced to the graph instead of silently switching engines.
        batches = await asyncio.gather(
            *(self._brave_context_query(client, query) for query in normalized)
        )
        merged: dict[str, tuple[str, list[str]]] = {}
        for rank in range(max((len(batch) for batch in batches), default=0)):
            for batch in batches:
                if rank >= len(batch):
                    continue
                url, title, passage = batch[rank]
                if url in merged:
                    old_title, passages = merged[url]
                    if passage not in passages:
                        passages.append(passage)
                    merged[url] = (old_title or title, passages)
                else:
                    merged[url] = (title, [passage])
        candidates = [
            (url, title, "\n\n".join(passages))
            for url, (title, passages) in merged.items()
        ]
        query_text = " ".join(normalized)
        if _MODEL_RELEASE_INTENT.search(query_text) or _VERSION.search(query_text):
            query_subjects = {
                token for token in re.findall(r"[a-z0-9]+", query_text.casefold())
                if len(token) >= 3 and token not in _GENERIC_QUERY_WORDS
            }
            wants_sdk = bool(re.search(r"\bsdk\b", query_text, re.IGNORECASE))
            candidates.sort(
                key=lambda item: _model_result_priority(
                    item, query_subjects=query_subjects, wants_sdk=wants_sdk
                )
            )
        allowed = await asyncio.gather(
            *(_public_dns(url) for url, _, _ in candidates)
        )
        snippets: list[KnowledgeSnippetV1] = []
        for (url, title, passage), public in zip(candidates, allowed):
            if not public:
                continue
            label = title or urlsplit(url).netloc
            text = _page_excerpt(
                passage,
                url=url,
                title=label,
                focus_terms=focus_terms,
                limit=self.settings.web_page_max_chars,
            )
            if not text.strip():
                continue
            snippets.append(
                KnowledgeSnippetV1(
                    source_label=label[:200],
                    provider="web",
                    rel_path=url,
                    source_url=url,
                    symbol=None,
                    start_line=None,
                    text=text,
                    score=round(0.95 - len(snippets) * 0.05, 2),
                )
            )
            if len(snippets) >= self.settings.web_search_max_results:
                break
        return snippets

    async def _search_queries(
        self, client: httpx.AsyncClient, queries: list[str]
    ) -> list[tuple[str, str, str]]:
        """Run a few independent, model-written public queries concurrently."""
        normalized: list[str] = []
        seen_queries: set[str] = set()
        for raw in queries:
            if not isinstance(raw, str):
                continue
            query = " ".join(raw.split())[:300]
            if query and query.casefold() not in seen_queries:
                normalized.append(query)
                seen_queries.add(query.casefold())
            if len(normalized) >= _MAX_SEARCH_QUERIES:
                break
        if not normalized:
            return []
        batches = await asyncio.gather(
            *(self._search_query(client, query) for query in normalized),
            return_exceptions=True,
        )
        successful: list[list[tuple[str, str, str]]] = []
        for batch in batches:
            if isinstance(batch, BaseException):
                if isinstance(batch, asyncio.CancelledError):
                    raise batch
                continue
            successful.append(batch)
        results: list[tuple[str, str, str]] = []
        seen: set[str] = set()
        for rank in range(max((len(batch) for batch in successful), default=0)):
            for batch in successful:
                if rank >= len(batch):
                    continue
                item = batch[rank]
                if item[0] in seen:
                    continue
                seen.add(item[0])
                results.append(item)
        query_text = " ".join(normalized)
        if _MODEL_RELEASE_INTENT.search(query_text) or _VERSION.search(query_text):
            query_subjects = {
                token
                for token in re.findall(r"[a-z0-9]+", query_text.casefold())
                if len(token) >= 3 and token not in _GENERIC_QUERY_WORDS
            }
            wants_sdk = bool(re.search(r"\bsdk\b", query_text, re.IGNORECASE))
            results.sort(
                key=lambda item: _model_result_priority(
                    item, query_subjects=query_subjects, wants_sdk=wants_sdk
                )
            )
        return results[: self.settings.web_search_max_results]

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
        raw_markdown_url = _github_raw_markdown_url(url)
        try:
            final_url, content_type, document = await self._download(
                client, raw_markdown_url or url
            )
        except (httpx.HTTPError, OSError, TimeoutError, ValueError):
            return url, "", ""
        if raw_markdown_url:
            # The raw host is validated by _download at every redirect. Keep
            # the normal GitHub page as the user-facing citation URL.
            if "html" in content_type:
                return url, "", ""
            title_match = re.search(r"^#\s+([^\n]+)", document, re.MULTILINE)
            title = title_match.group(1).strip() if title_match else ""
            return url, title, document.strip()
        if "html" not in content_type and "text" not in content_type:
            return final_url, "", ""
        title_match = _TITLE.search(document)
        title = _strip_html(title_match.group(1)) if title_match else ""
        main_match = _MAIN.search(document)
        body = _strip_html(main_match.group(1) if main_match else document)
        return final_url, title, body
