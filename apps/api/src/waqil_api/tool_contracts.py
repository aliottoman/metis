"""A tiny, dependency-free JSON-Schema *subset* checker for tool I/O contracts.

Declarative tools declare their input/output as small JSON Schemas. We do not
need a full validator — only enough to gate a tool's output against its declared
contract (object with required string / array-of-string / object properties).
Kept deliberately minimal and total: it returns problems rather than raising.
"""
from __future__ import annotations

from typing import Any


def matches_contract(value: Any, schema: dict[str, Any]) -> tuple[bool, list[str]]:
    """Return ``(ok, problems)`` for ``value`` against a small JSON-Schema subset.

    Supported: ``type`` in {object, array, string, number, integer, boolean},
    ``properties``, ``required``, ``items`` (for arrays), and
    ``additionalProperties: false``. Unknown keywords are ignored."""
    problems: list[str] = []
    _check(value, schema, "$", problems)
    return (not problems, problems)


def _check(value: Any, schema: dict[str, Any], path: str, problems: list[str]) -> None:
    expected = schema.get("type")
    if expected and not _type_ok(value, expected):
        problems.append(f"{path}: expected {expected}, got {type(value).__name__}")
        return
    if expected == "object":
        if not isinstance(value, dict):
            problems.append(f"{path}: expected object")
            return
        properties: dict[str, Any] = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                problems.append(f"{path}.{key}: required property missing")
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    problems.append(f"{path}.{key}: unexpected property")
        for key, subschema in properties.items():
            if key in value and isinstance(subschema, dict):
                _check(value[key], subschema, f"{path}.{key}", problems)
    elif expected == "array":
        if not isinstance(value, list):
            problems.append(f"{path}: expected array")
            return
        items = schema.get("items")
        if isinstance(items, dict):
            for index, item in enumerate(value):
                _check(item, items, f"{path}[{index}]", problems)


def _type_ok(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    return True
