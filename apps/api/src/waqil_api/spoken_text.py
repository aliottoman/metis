"""Text a voice can say.

What reads well on a screen is not what sounds like a person. Markdown
syntax, tables, URLs and bracketed citation markers are punctuation a reader
sees through and a listener hears as gibberish — "asterisk asterisk", "open
bracket one close bracket", a URL spelled letter by letter — so speech gets
its own rendition rather than the written answer with the markup left in.

Deterministic on purpose, and that is the point rather than a limitation.
This is the floor under everything spoken: the fallback when a model omits or
mangles the spoken half of its reply, and the whole rendition for the morning
brief, which has to be sayable on a morning when no model is reachable at all.
"""

from __future__ import annotations

import re
from datetime import datetime

from .contracts import AttentionItemV1, MorningBriefV1

# Two to four sentences is a spoken answer; past that a listener has stopped
# holding the beginning. The character ceiling is the backstop for one very
# long sentence, which no sentence count can catch.
SPOKEN_MAX_CHARS = 700
SPOKEN_MAX_SENTENCES = 4

_FENCED_CODE = re.compile(r"```.*?(?:```|\Z)", re.DOTALL)
_INDENTED_CODE = re.compile(r"^(?: {4}|\t).*$", re.MULTILINE)
_TABLE_DIVIDER = re.compile(r"^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$")
_MARKDOWN_LINK = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")
_BARE_URL = re.compile(r"<?\b(?:https?://|www\.)\S+>?")
# `[1]`, `[^2]`, `[doc-3]`, `[S1]` — a citation marker is a reference the eye
# skips and the ear cannot. Deliberately narrow: it must not eat ordinary
# bracketed prose like "[the second one]".
_CITATION = re.compile(r"\[\^?[A-Za-z]{0,4}[-\s]?\d{1,3}(?:\s*,\s*\d{1,3})*\]")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_BULLET = re.compile(r"^\s*(?:[-*+]|\d{1,3}[.)])\s+", re.MULTILINE)
_BLOCKQUOTE = re.compile(r"^\s*>+\s?", re.MULTILINE)
_EMPHASIS = re.compile(r"(\*{1,3}|_{1,3}|~~)(?=\S)(.+?)(?<=\S)\1", re.DOTALL)
_INLINE_CODE = re.compile(r"`+([^`]*)`+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE = re.compile(r"\s+")


def to_speech(
    text: str,
    *,
    max_chars: int = SPOKEN_MAX_CHARS,
    max_sentences: int = SPOKEN_MAX_SENTENCES,
) -> str:
    """One written answer, rendered as something a voice can read aloud.

    Lossy by design. A table becomes the fact that a table exists, a code
    block disappears, a link keeps its label and loses its address. The
    written answer is still on screen in full — this is the version for the
    ear, and an ear cannot scroll back.
    """
    if not text or not text.strip():
        return ""
    working = _FENCED_CODE.sub(" ", text)
    working = _describe_tables(working)
    working = _INDENTED_CODE.sub(" ", working)
    working = _INLINE_CODE.sub(r"\1", working)
    # Links before citations and bare URLs: the label is worth keeping and the
    # address never is, and a stripped address would leave a naked bracket.
    working = _MARKDOWN_LINK.sub(lambda match: match.group(1) or " ", working)
    working = _CITATION.sub(" ", working)
    working = _BARE_URL.sub(" ", working)
    working = _BLOCKQUOTE.sub("", working)
    working = _HEADING.sub("", working)
    working = _EMPHASIS.sub(r"\2", working)
    working = _flatten_lists(working)
    working = _WHITESPACE.sub(" ", working).strip()
    working = re.sub(r"\s+([.,;:!?])", r"\1", working)
    working = re.sub(r"([.,;:!?])\1+", r"\1", working)
    return _bounded(working, max_chars=max_chars, max_sentences=max_sentences)


def _describe_tables(text: str) -> str:
    """A table becomes its dimensions, which is all a listener can use.

    Read aloud, a table is a stream of pipes and dashes. Named instead — "a
    table of four rows and three columns" — it tells the listener the answer
    exists on screen and stops the voice from spelling it out.
    """
    lines = text.splitlines()
    output: list[str] = []
    index = 0
    while index < len(lines):
        block: list[str] = []
        while index < len(lines) and "|" in lines[index]:
            block.append(lines[index])
            index += 1
        if len(block) >= 2 and any(_TABLE_DIVIDER.match(line) for line in block):
            columns = max(
                len([cell for cell in line.strip().strip("|").split("|")])
                for line in block
            )
            rows = len([line for line in block if not _TABLE_DIVIDER.match(line)]) - 1
            output.append(
                f"There is a table of {_count(max(rows, 0), 'row')} "
                f"and {_count(columns, 'column')} on screen."
            )
            continue
        output.extend(block)
        if index < len(lines):
            output.append(lines[index])
            index += 1
    return "\n".join(output)


def _flatten_lists(text: str) -> str:
    """List items become sentences, so the voice never says "dash"."""
    lines: list[str] = []
    for raw in text.splitlines():
        line = _BULLET.sub("", raw).strip()
        if not line:
            continue
        # A bullet is a sentence's worth of meaning without a sentence's
        # punctuation; give it one so the flattened run does not become a
        # single breathless clause.
        if line[-1] not in ".!?:;,":
            line += "."
        lines.append(line)
    return " ".join(lines)


def _bounded(text: str, *, max_chars: int, max_sentences: int) -> str:
    """The hard ceiling: whole sentences where possible, a clean cut where not."""
    sentences = [item for item in _SENTENCE_SPLIT.split(text) if item.strip()]
    kept: list[str] = []
    length = 0
    for sentence in sentences[:max_sentences]:
        addition = len(sentence) + (1 if kept else 0)
        if kept and length + addition > max_chars:
            break
        kept.append(sentence)
        length += addition
    spoken = " ".join(kept).strip()
    if len(spoken) <= max_chars:
        return spoken
    # One sentence longer than the whole budget. Cut on a word boundary rather
    # than mid-word, and end it so the voice does not trail off mid-breath.
    clipped = spoken[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:")
    return f"{clipped}." if clipped and clipped[-1] not in ".!?" else clipped


def _count(value: int, noun: str) -> str:
    return f"{value} {noun}" if value == 1 else f"{value} {noun}s"


def brief_to_speech(brief: MorningBriefV1) -> str:
    """The morning brief, as one spoken paragraph.

    Composed here rather than by a model, for the reason the brief itself
    splits host-counted facts from model-written prose: the counts are the
    part that must never drift, and this is only ever rearranging them into
    sentences. It also means the Listen button still works on a morning when
    no model is reachable — the facts are the brief, and they are all local.
    """
    parts: list[str] = []

    if brief.waiting_total == 0:
        parts.append("Nothing is waiting for a decision.")
    else:
        parts.append(f"{_count(brief.waiting_total, 'thing')} waiting.")

    narrative = to_speech(brief.narrative, max_sentences=2)
    if narrative:
        parts.append(narrative)

    if brief.focus:
        named = _spoken_item(brief.focus[0])
        if named:
            parts.append(f"At the top of the queue: {named}.")

    if brief.changed:
        spoken_changes = [
            cleaned
            for line in brief.changed[:3]
            if (cleaned := to_speech(line, max_sentences=1).rstrip("."))
        ]
        if spoken_changes:
            parts.append(f"{_window_phrase(brief)}, {_series(spoken_changes)}.")

    recommendation = to_speech(brief.recommendation, max_sentences=1)
    if recommendation:
        parts.append(recommendation)

    # Deliberately not passed through the sentence ceiling: this is a briefing
    # the owner asked for by pressing a button, not a conversational turn.
    # Every sentence in it was bounded on the way in.
    return " ".join(part for part in parts if part).strip()


def _spoken_item(item: AttentionItemV1) -> str:
    """One queue item as a clause: "commitment, send the sizing, which is overdue"."""
    title = to_speech(item.title, max_sentences=1).rstrip(".")
    if not title:
        return ""
    label = item.kind_label.strip().lower()
    named = f"{label}, {title}" if label else title
    return f"{named}, which is overdue" if item.overdue else named


def _window_phrase(brief: MorningBriefV1) -> str:
    """ "In the last day", from the window the brief was actually composed over."""
    if not isinstance(brief.since, datetime) or not isinstance(
        brief.generated_at, datetime
    ):
        return "Recently"
    hours = round((brief.generated_at - brief.since).total_seconds() / 3600)
    if hours <= 0:
        return "Recently"
    if hours == 24:
        return "In the last day"
    if hours % 24 == 0:
        return f"In the last {_count(hours // 24, 'day')}"
    return f"In the last {_count(hours, 'hour')}"


def _series(items: list[str]) -> str:
    """ "a, b and c" — the spoken form of a list."""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} and {items[-1]}"
