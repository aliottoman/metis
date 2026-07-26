"""Named AST capability profiles — the reviewed, host-owned framework of control.

A tool definition references a code profile *by name* (e.g. ``diagrams-render-v1``);
the validator that decides whether a piece of generated code is allowed to run
lives here, in trusted host code, and can never be authored or widened by a
model. This is what lets the factory generalize to many tools without ever
generalizing the trust boundary: every tool's code — whether authored at build
time or by the model at run time — must pass a named profile from this registry.

Each profile exposes ``validate(source, context) -> evidence`` which raises
``CodeProfileError`` on any violation and otherwise returns a JSON-able evidence
dict recorded with the run.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class CodeProfileError(ValueError):
    """Generated source violates the named capability profile."""


@dataclass(frozen=True)
class CodeProfile:
    name: str
    description: str
    validate: Callable[[str, dict[str, Any]], dict[str, Any]]


_REGISTRY: dict[str, CodeProfile] = {}


def register(profile: CodeProfile) -> None:
    """Register a profile. Re-registration under the same name replaces it (so a
    richer version can supersede an older one within a process)."""
    _REGISTRY[profile.name] = profile


def get(name: str) -> CodeProfile | None:
    return _REGISTRY.get(name)


def exists(name: str) -> bool:
    return name in _REGISTRY


def names() -> list[str]:
    return sorted(_REGISTRY)


def validate(name: str, source: str, context: dict[str, Any]) -> dict[str, Any]:
    """Resolve a profile by name and run its validator. Unknown profile names
    fail closed — a definition can never reference an unregistered profile."""
    profile = _REGISTRY.get(name)
    if profile is None:
        raise CodeProfileError(f"unknown capability profile: {name}")
    return profile.validate(source, context)


# The generated program must be AST-equal to the host's canonical source.
def _validate_diagrams_render_v1(source: str, context: dict[str, Any]) -> dict[str, Any]:
    from .diagram_source import DiagramSourceError, validate_diagram_source

    spec = context.get("spec")
    if spec is None:
        raise CodeProfileError("diagrams-render-v1 requires an architecture spec in context")
    try:
        return validate_diagram_source(source, spec, context.get("output_formats"))
    except DiagramSourceError as exc:
        raise CodeProfileError(str(exc)) from exc


register(
    CodeProfile(
        name="diagrams-render-v1",
        description=(
            "Exact-canonical architecture AST (v1): the source must match the "
            "host-generated diagrams program AST-for-AST — the model contributes "
            "nothing beyond copying it."
        ),
        validate=_validate_diagrams_render_v1,
    )
)


# Allows varied, styled diagram code built only from safe DSL primitives, and
# proves it covers the whole spec. Failure falls back to the canonical source.
def _validate_diagrams_draw_v2(source: str, context: dict[str, Any]) -> dict[str, Any]:
    from .diagram_source import DiagramSourceError, validate_diagram_source_v2

    spec = context.get("spec")
    if spec is None:
        raise CodeProfileError("diagrams-draw-v2 requires an architecture spec in context")
    try:
        return validate_diagram_source_v2(source, spec, context.get("output_formats"))
    except DiagramSourceError as exc:
        raise CodeProfileError(str(exc)) from exc


register(
    CodeProfile(
        name="diagrams-draw-v2",
        description=(
            "Allowlist policy for model-authored diagrams code: only the safe "
            "Blank/Cluster/Diagram/Edge DSL plus layout attributes, and every "
            "spec component and edge must be represented. Permits variety without "
            "widening the trust boundary."
        ),
        validate=_validate_diagrams_draw_v2,
    )
)


# A declarative tool runs no authored code, so it carries no code-execution
# capability. Naming the profile keeps validate fail-closed.
def _validate_declarative_host_v1(source: str, context: dict[str, Any]) -> dict[str, Any]:
    raise CodeProfileError(
        "declarative-host-v1 tools execute no generated code; there is nothing to validate"
    )


register(
    CodeProfile(
        name="declarative-host-v1",
        description=(
            "No-code-execution profile: the tool runs entirely as a host-"
            "interpreted declarative pipeline (input pipeline + one pinned broker "
            "call + deterministic fallback). It grants no sandbox or code-exec "
            "capability; validating any source against it fails closed."
        ),
        validate=_validate_declarative_host_v1,
    )
)


# For tools whose implementation the model writes: a reviewed stdlib subset with
# every escape hatch banned, reaching the model only via the model() bridge.
def _validate_pure_python_authored_v1(source: str, context: dict[str, Any]) -> dict[str, Any]:
    from .authored_code import AuthoredCodeError, validate_authored_source

    try:
        return validate_authored_source(source, context)
    except AuthoredCodeError as exc:
        raise CodeProfileError(str(exc)) from exc


register(
    CodeProfile(
        name="pure-python-authored-v1",
        description=(
            "Model-authored `run(input, model)` tool code: general Python logic "
            "over a reviewed stdlib allowlist, with imports/eval/exec/open/dunder "
            "access and all network/filesystem/process access banned. Runtime "
            "model access only through the injected budget-enforced bridge."
        ),
        validate=_validate_pure_python_authored_v1,
    )
)
