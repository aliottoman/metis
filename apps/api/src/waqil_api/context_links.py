"""Bounded, local context connections. Never indexes or reads raw source bodies.

Only reviewed account evidence and passages already admitted by corpus consent
reach this module. Conversation history can clarify a follow-up's account, but
it does not become evidence or grant permission for any record-writing action.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote, urlencode, urlsplit

from .queue_update import candidate_accounts, resolve_account

_FOLLOW_UP = re.compile(
    r"\b(?:it|its|they|their|them|that|those|this|these)\b"
    r"|^(?:and\b|also\b|what about\b|how about\b)"
    r"|^what(?:'s| is| should (?:i|we) do) next\b",
    re.IGNORECASE,
)
_READ_QUESTION = re.compile(
    r"^(?:and\s+|also\s+)?(?:what|who|when|where|why|how|which|is|are|does|do|can you (?:summari[sz]e|explain|show|compare))\b",
    re.IGNORECASE,
)
_WRITE = re.compile(
    r"\b(?:delete|remove|save|record|update|change|cancel|approve|send|email|file|mark|create)\b",
    re.IGNORECASE,
)
_MULTI_ACCOUNT = re.compile(r"\b(?:compare|between|versus|vs|and)\b", re.IGNORECASE)


def _resolve_read_account(
    prompt: str, accounts: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, bool]:
    # The write resolver ranks a full name above an acronym. A comparison can
    # contain both, so unequal scores still must not carry one account forward.
    if _MULTI_ACCOUNT.search(prompt) and len(candidate_accounts(prompt, accounts)) > 1:
        return None, True
    resolved, tied = resolve_account(prompt, accounts)
    return resolved, bool(tied)


def context_account(
    prompt: str, recent_messages: list[dict[str, Any]], accounts: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, bool]:
    """Resolve one explicit name, or a short read-only follow-up to that name.

    A new topic, ambiguous account mention, or an assistant-invented name cannot
    carry account scope forward. At most three user turns are considered.
    """
    direct, tied = _resolve_read_account(prompt, accounts)
    if direct is not None or tied:
        return direct, False
    if (
        len(prompt) > 300
        or not _FOLLOW_UP.search(prompt)
        or not _READ_QUESTION.search(prompt)
        or _WRITE.search(prompt)
    ):
        return None, False
    questions = [
        str(item.get("content", ""))
        for item in recent_messages
        if item.get("role") == "user"
    ]
    for previous in reversed(questions[-3:]):
        resolved, ambiguous = _resolve_read_account(previous, accounts)
        if ambiguous:
            return None, False
        if resolved is not None:
            return resolved, True
        if (
            not _FOLLOW_UP.search(previous)
            or not _READ_QUESTION.search(previous)
            or _WRITE.search(previous)
        ):
            break
    return None, False


def customer_record_url(account_id: str, tab: str, record_id: str = "") -> str:
    values = {"account": account_id, "tab": tab}
    # These record kinds have explicit deep-link handling after the account
    # loads. A fragment alone can arrive before the requested record exists.
    key = {"facts": "fact", "actions": "action"}.get(tab)
    if key and record_id.startswith(f"{key}-"):
        values[key] = record_id[len(key) + 1 :]
    url = "/customers?" + urlencode(values)
    return url + (f"#{quote(record_id, safe='')}" if record_id else "")


def source_record_url(account_id: str, source_id: str) -> str:
    return "/customers?" + urlencode(
        {"account": account_id, "tab": "sources", "source": source_id}
    )


def meeting_source_url(meeting_id: str, turn_id: str | None = None) -> str:
    values = {"meeting": meeting_id}
    if turn_id:
        values["turn"] = turn_id
    return "/meetings?" + urlencode(values)


def safe_source_url(raw: Any) -> str | None:
    """Keep generated provenance links inside the app or on the public web."""
    if not isinstance(raw, str) or not raw or re.search(r"[\s\\<>\(\)]", raw):
        return None
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return None
    if raw.startswith("/") and not raw.startswith("//") and not parsed.netloc:
        return raw
    if parsed.scheme in {"http", "https"} and parsed.netloc and not parsed.username:
        return raw
    return None


async def link_run_history(database: Any, snippets: list[dict[str, Any]]) -> None:
    """Link already-retrieved history to its real run; never fetch other text."""
    cache: dict[str, Any] = {}
    for snippet in snippets:
        if (
            snippet.get("provider") != "local"
            or snippet.get("source_label") != "Run history"
        ):
            continue
        filename = str(snippet.get("rel_path", "")).rsplit("/", 1)[-1]
        match = re.fullmatch(r"(run_[A-Za-z0-9_-]{1,80})\.md", filename)
        if not match:
            continue
        run_id = match.group(1)
        if run_id not in cache:
            if len(cache) >= 8:
                continue
            try:
                cache[run_id] = await database.get_run(run_id)
            except Exception:  # metadata enrichment must not fail retrieval
                cache[run_id] = None
        run = cache[run_id]
        if run is None:
            continue
        snippet["source_url"] = "/?" + urlencode(
            {"conversation": run.conversation_id, "run": run.id}
        )
        # A project name comes only from the passage already admitted by the
        # source's consent, not from another project or arbitrary files.
        project = re.search(
            r"^- Project: (.{1,120})$", str(snippet.get("text", "")), re.MULTILINE
        )
        if project:
            snippet["source_label"] = f"Project conversation · {project.group(1)}"
        else:
            snippet["source_label"] = "Past conversation"
