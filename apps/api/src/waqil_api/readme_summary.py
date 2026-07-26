"""The README to summary-card declarative tool.

This is a *host-interpreted* tool: it executes no generated or sandboxed code.
Its whole behavior is
    input text → one pinned "summarize" broker call → deterministic fallback →
    output-contract check.
So its only capability is the bounded, audited Model Broker (R2). The pinned
prompt template lives in the approved definition (immutable); this module only
fills its parameters and — crucially — never fails the run: any broker error,
budget exhaustion, or malformed reply degrades to a deterministic host summary.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .contracts import ToolDefinitionV1
from .model_broker import ModelBroker
from .tool_contracts import matches_contract

_MAX_TEXT = 8_000
_OUTPUT_KEYS = ("title", "purpose", "components", "stack", "summary")


async def run(
    definition: ToolDefinitionV1,
    tool_input: dict[str, Any],
    broker: ModelBroker,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Produce a validated summary-card output. Returns ``(output, meta)`` where
    meta records who authored it and any fallback reason. Never raises for model
    problems — it degrades to the deterministic host summary."""
    text = str(tool_input.get("text", ""))[:_MAX_TEXT]
    access = definition.capability_profile.model_access
    template_id = next(iter(access.prompt_templates), "summarize")
    role = access.roles[0] if access.roles else "reviewer"
    fallback = _deterministic_summary(text)

    if not access.enabled or broker is None or not getattr(broker, "enabled", False):
        return fallback, {"authored_by": "deterministic-fallback", "fallback_reason": "no model access"}

    try:
        raw = await broker.call(
            template_id=template_id,
            role=role,
            params={
                "text": text,
                "instructions": "Summarize the project text into the required JSON keys.",
            },
        )
    except Exception as error:  # noqa: BLE001 — any broker failure degrades safely
        return fallback, {
            "authored_by": "deterministic-fallback",
            "fallback_reason": f"broker error: {str(error)[:160]}",
        }

    parsed = _parse_reply(raw)
    if parsed is None:
        return fallback, {
            "authored_by": "deterministic-fallback",
            "fallback_reason": "unparseable model reply",
        }
    output = _coerce_output(parsed, fallback)
    ok, _ = matches_contract(output, definition.output_contract or _DEFAULT_CONTRACT)
    if not ok:
        return fallback, {
            "authored_by": "deterministic-fallback",
            "fallback_reason": "model output failed the contract",
        }
    return output, {"authored_by": "model", "fallback_reason": None}


_DEFAULT_CONTRACT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "purpose": {"type": "string"},
        "components": {"type": "array", "items": {"type": "string"}},
        "stack": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": list(_OUTPUT_KEYS),
    "additionalProperties": False,
}


def _parse_reply(raw: str) -> dict[str, Any] | None:
    """Extract a JSON object from a model reply (tolerating code fences / prose)."""
    if not raw:
        return None
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fence.group(1) if fence else None
    if candidate is None:
        start = text.find("{")
        end = text.rfind("}")
        candidate = text[start : end + 1] if start != -1 and end > start else None
    if candidate is None:
        return None
    try:
        value = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _coerce_output(candidate: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    """Force a (possibly partial) parsed reply into the output contract's shape,
    filling any missing/ill-typed field from the deterministic fallback."""
    def as_str(value: Any, default: str) -> str:
        return value.strip()[:2_000] if isinstance(value, str) and value.strip() else default

    def as_list(value: Any, default: list[str]) -> list[str]:
        if isinstance(value, list):
            items = [str(item).strip()[:120] for item in value if str(item).strip()]
            return items[:24]
        return default

    return {
        "title": as_str(candidate.get("title"), fallback["title"]),
        "purpose": as_str(candidate.get("purpose"), fallback["purpose"]),
        "components": as_list(candidate.get("components"), fallback["components"]),
        "stack": as_list(candidate.get("stack"), fallback["stack"]),
        "summary": as_str(candidate.get("summary"), fallback["summary"]),
    }


def _is_marker(line: str) -> bool:
    """The header line ingest prepends to each attachment, e.g.
    ``--- README.md (untrusted attachment) ---`` — noise for the fallback."""
    stripped = line.strip()
    return "(untrusted attachment)" in stripped or bool(
        re.fullmatch(r"-{2,}.*-{2,}", stripped)
    )


def _deterministic_summary(text: str) -> dict[str, Any]:
    """A pure-host summary of README/project text — the never-fails fallback."""
    lines = [line.rstrip() for line in text.splitlines() if not _is_marker(line)]
    title = ""
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            break
        if stripped:
            title = stripped
            break
    title = (title or "Untitled Project")[:120]

    # First non-heading, non-empty paragraph → purpose (first sentence).
    body_lines = [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#") and not line.strip().startswith(("-", "*"))
    ]
    body = " ".join(body_lines)
    sentences = re.split(r"(?<=[.!?])\s+", body)
    purpose = (sentences[0] if sentences and sentences[0] else title)[:2_000]
    summary = (" ".join(sentences[:3]) if sentences else purpose)[:2_000] or purpose

    components = _bullets_under(lines, ("component", "module", "service", "part", "feature"))
    stack = _stack_terms(lines)
    return {
        "title": title,
        "purpose": purpose,
        "components": components[:24],
        "stack": stack[:24],
        "summary": summary,
    }


def _bullets_under(lines: list[str], heading_terms: tuple[str, ...]) -> list[str]:
    collecting = False
    found: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip().lower()
            collecting = any(term in heading for term in heading_terms)
            continue
        if collecting and stripped.startswith(("-", "*")):
            item = stripped.lstrip("-* ").strip()
            if item:
                found.append(item[:120])
        elif collecting and not stripped:
            continue
        elif collecting:
            collecting = False
    if found:
        return found
    # Fallback: any top-level bullets anywhere.
    return [
        line.strip().lstrip("-* ").strip()[:120]
        for line in lines
        if line.strip().startswith(("-", "*")) and line.strip().lstrip("-* ").strip()
    ]


_STACK_TERMS = ("built with", "tech stack", "stack", "technologies", "requirements", "dependencies")


def _stack_terms(lines: list[str]) -> list[str]:
    for index, line in enumerate(lines):
        lowered = line.lower()
        if not any(term in lowered for term in _STACK_TERMS):
            continue
        # Inline ("Stack: Python, Redis") or a heading whose list is the next line.
        if ":" in line and not line.strip().startswith("#"):
            payload = line.split(":", 1)[1]
        else:
            payload = next(
                (
                    later.lstrip("-* ").strip()
                    for later in lines[index + 1 :]
                    if later.strip() and not later.strip().startswith("#")
                ),
                "",
            )
        terms = [term.strip()[:120] for term in re.split(r"[,/;]| and ", payload) if term.strip()]
        cleaned = [term for term in terms if 0 < len(term) < 40]
        if cleaned:
            return cleaned
    return []
