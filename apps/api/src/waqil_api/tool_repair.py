"""Deterministic repair of a model's tool call, before it counts as malformed.

Every lane already pays for the same small mistakes. A model wraps its JSON in a
code fence, names the file argument ``filepath``, sends a line number as the
string ``"12"``, or quotes a path with backticks — and the host refuses the call,
spends a step on the refusal, and on a bad run spends the turn. None of those are
ambiguous, and none of them are the model being wrong about the *work*.

So they are repaired here, in one place both decode paths pass through, under
three rules that keep this a shape fix rather than a licence:

* **Never invent authority.** A tool name outside the roster is still refused; a
  required argument that is simply absent is still missing. Repair renames and
  re-types what the model sent — it never supplies a value the model did not.
* **Never guess between two readings.** ``patch`` could mean ``original`` or
  ``replacement`` and there is no way to tell, so it is left alone and the
  existing refusal-and-pin path handles it. A repair that is right most of the
  time is worse than a refusal that is right every time, because the refusal is
  visible and a wrong repair is not.
* **Say what was repaired.** Every fix returns a note. The loop emits them, so
  "which repairs actually fire" is a measured question — one that never fires
  should be deleted, and one that fires constantly is telling us about a prompt
  bug rather than a model bug.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .contracts import (
    PROJECT_TOOL_ARGUMENT_PROPERTIES,
    PROJECT_TOOL_OPTIONAL_ARGUMENTS,
    PROJECT_TOOL_REQUIRED_ARGUMENTS,
)

# Spellings models reach for, mapped to the key the roster actually defines.
# Only unambiguous ones: each maps to exactly one argument, for every tool that
# takes it. Deliberately NOT here: `patch` (could be original or replacement),
# `code` (could be content or replacement), `line` (start or end).
_ALIASES: dict[str, str] = {
    "file": "path",
    "filename": "path",
    "file_name": "path",
    "filepath": "path",
    "file_path": "path",
    "target": "path",
    "contents": "content",
    "text": "content",
    "body": "content",
    "data": "content",
    "source": "content",
    "old": "original",
    "old_text": "original",
    "old_string": "original",
    "search": "original",
    "find": "original",
    "new": "replacement",
    "new_text": "replacement",
    "new_string": "replacement",
    "replace": "replacement",
    "replace_with": "replacement",
    "pattern": "query",
    "q": "query",
    "search_query": "query",
    "term": "query",
    "check": "name",
    "check_name": "name",
    "start": "start_line",
    "from_line": "start_line",
    "end": "end_line",
    "to_line": "end_line",
    "max": "limit",
    "max_results": "limit",
    "count": "limit",
    "prompt": "question",
    "choices": "options",
    "answer": "message",
    "response": "message",
    "paths": "files",
    "file_list": "files",
    "why": "reason",
    "explanation": "reason",
}

_FENCE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*\n(.*?)\n?\s*```\s*$", re.DOTALL)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")
_PY_LITERAL = re.compile(r"(?<![\w\"])(True|False|None)(?![\w\"])")


def _normalise_key(key: str) -> str:
    """``File Path`` / ``file-path`` / ``FILEPATH`` all compare equal."""
    return re.sub(r"[^a-z0-9]", "", key.lower())


# The table above is written in the spelling a person would read; lookups happen
# in normalised form, so it is folded once here rather than at every call site.
# (Written the other way round it silently matched nothing, which is a repair
# layer's worst failure mode: no error, no fix, no trace.)
_ALIASES_NORMAL: dict[str, str] = {
    _normalise_key(spelling): target for spelling, target in _ALIASES.items()
}


def _known_keys(tool: str) -> list[str]:
    return list(
        dict.fromkeys(
            PROJECT_TOOL_REQUIRED_ARGUMENTS.get(tool, [])
            + PROJECT_TOOL_OPTIONAL_ARGUMENTS.get(tool, [])
        )
    )


def repair_json_text(raw: str) -> tuple[str, list[str]]:
    """Make a model's near-JSON parseable, or hand it back unchanged.

    Only transformations whose intent is unambiguous. The result is *tried*, not
    trusted: the caller parses it, and a repair that does not parse leaves the
    original error exactly where it was.
    """
    notes: list[str] = []
    text = raw.strip()
    if not text:
        return raw, notes

    fenced = _FENCE.match(text)
    if fenced is not None:
        text = fenced.group(1).strip()
        notes.append("stripped a code fence")

    # Prose on either side of a single JSON object — "Here is the call: {...}".
    if not text.startswith(("{", "[")):
        opened = min(
            (index for index in (text.find("{"), text.find("[")) if index >= 0),
            default=-1,
        )
        closed = max(text.rfind("}"), text.rfind("]"))
        if opened >= 0 and closed > opened:
            text = text[opened : closed + 1]
            notes.append("dropped prose around the JSON")

    if _looks_parseable(text):
        return (text, notes) if notes else (raw, notes)

    repaired = _TRAILING_COMMA.sub(r"\1", text)
    if repaired != text:
        text = repaired
        notes.append("removed a trailing comma")

    repaired = _PY_LITERAL.sub(lambda m: {"True": "true", "False": "false", "None": "null"}[m.group(1)], text)
    if repaired != text:
        text = repaired
        notes.append("converted Python literals")

    if _looks_parseable(text):
        return text, notes
    # Nothing else is safe to attempt blind: rewriting quotes inside a payload
    # that legitimately contains code would corrupt the very thing being staged.
    return (text, notes) if notes else (raw, [])


def _looks_parseable(text: str) -> bool:
    try:
        json.loads(text)
    except (ValueError, TypeError):
        return False
    return True


def repair_arguments(tool: str, arguments: Any) -> tuple[dict[str, Any], list[str]]:
    """The arguments this tool can actually be called with, plus what was fixed.

    Unknown keys are kept, not dropped: the workspace's own refusal names them,
    and silently discarding an argument the model believed in produces a call
    that succeeds while doing the wrong thing — much worse than one that fails.
    """
    notes: list[str] = []
    if not isinstance(arguments, dict):
        return {}, notes
    known = _known_keys(tool)
    if not known:
        return dict(arguments), notes
    by_normal = {_normalise_key(key): key for key in known}

    repaired: dict[str, Any] = {}
    for key, value in arguments.items():
        target = key
        if key not in known:
            normal = _normalise_key(key)
            if normal in by_normal:
                target = by_normal[normal]
                notes.append(f"{key} -> {target}")
            elif _ALIASES_NORMAL.get(normal) in known:
                target = _ALIASES_NORMAL[normal]
                notes.append(f"{key} -> {target}")
        # A repair must never silently overwrite a value the model already put
        # under the correct name.
        if target in repaired and target != key:
            continue
        repaired[target] = value

    for key, value in list(repaired.items()):
        fixed, note = _repair_value(key, value)
        if note:
            repaired[key] = fixed
            notes.append(note)
    return repaired, notes


def _repair_value(key: str, value: Any) -> tuple[Any, str | None]:
    """One argument, coerced to the type its schema declares."""
    declared = PROJECT_TOOL_ARGUMENT_PROPERTIES.get(key, {}).get("type")
    if declared == "string" and isinstance(value, str):
        if key == "path":
            cleaned = _clean_path(value)
            if cleaned != value:
                return cleaned, "tidied the path"
        return value, None
    if declared == "integer" and isinstance(value, str):
        text = value.strip()
        if text.lstrip("-").isdigit():
            return int(text), f"{key} was a string"
        return value, None
    if declared == "boolean" and isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "false"):
            return text == "true", f"{key} was a string"
        return value, None
    if declared == "array" and isinstance(value, str):
        # A single item where a list was declared, which is how models spell
        # "just this one file".
        text = value.strip()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except (ValueError, TypeError):
                return value, None
            if isinstance(parsed, list):
                return parsed, f"{key} was a JSON string"
            return value, None
        return ([text], f"{key} was a single value") if text else (value, None)
    if declared == "string" and isinstance(value, (int, float, bool)):
        return str(value), f"{key} was not a string"
    return value, None


def _clean_path(value: str) -> str:
    """Strip the decoration models put around a path, never its meaning."""
    text = value.strip().strip("`").strip().strip('"').strip("'").strip()
    text = text.replace("\\", "/")
    while "//" in text:
        text = text.replace("//", "/")
    while text.startswith("./"):
        text = text[2:]
    return text
