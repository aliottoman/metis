from __future__ import annotations

import ast
from typing import Any

from .contracts import ArchitectureSpecV1


class DiagramSourceError(ValueError):
    """Generated source does not exactly implement the validated specification."""


def canonical_architecture_spec(spec: ArchitectureSpecV1) -> ArchitectureSpecV1:
    """Normalize order-only differences at the trusted model/sandbox boundary."""

    return spec.model_copy(
        update={
            "components": sorted(spec.components, key=lambda item: item.id),
            "edges": sorted(
                spec.edges,
                key=lambda item: (item.source, item.target, item.label),
            ),
            "boundaries": [
                boundary.model_copy(
                    update={"component_ids": sorted(boundary.component_ids)}
                )
                for boundary in sorted(spec.boundaries, key=lambda item: item.id)
            ],
        }
    )


def canonical_diagram_source(
    spec: ArchitectureSpecV1, output_formats: list[str] | None = None
) -> str:
    """Return the sole Python program accepted by Metis's trusted host policy.

    The sandbox independently applies the same policy. Keeping a trusted host-side
    representation prevents obviously invalid model output from reaching Podman and
    gives the model one bounded, typed repair opportunity.
    """

    formats = output_formats or ["svg", "png"]
    spec = canonical_architecture_spec(spec)
    components = spec.components
    boundaries = spec.boundaries
    edges = spec.edges
    variable_by_id = {
        component.id: f"node_{index:03d}" for index, component in enumerate(components)
    }
    component_by_id = {component.id: component for component in components}
    grouped_ids = {
        component_id
        for boundary in boundaries
        for component_id in boundary.component_ids
    }

    lines = [
        "from pathlib import Path",
        "from diagrams import Cluster, Diagram, Edge",
        "from diagrams.generic.blank import Blank",
        "",
        'OUTPUT_STEM = str(Path(__file__).resolve().parent / "architecture")',
        "",
        (
            f"with Diagram({spec.title!r}, filename=OUTPUT_STEM, "
            f"outformat={formats!r}, show=False, direction={spec.direction!r}):"
        ),
    ]
    for boundary in boundaries:
        lines.append(f"    with Cluster({boundary.label!r}):")
        for component_id in sorted(boundary.component_ids):
            component = component_by_id[component_id]
            label = f"{component.label}\n[{component.kind}]"
            lines.append(f"        {variable_by_id[component_id]} = Blank({label!r})")
    for component in components:
        if component.id not in grouped_ids:
            label = f"{component.label}\n[{component.kind}]"
            lines.append(f"    {variable_by_id[component.id]} = Blank({label!r})")
    for edge in edges:
        source = variable_by_id[edge.source]
        target = variable_by_id[edge.target]
        if edge.label:
            lines.append(f"    {source} >> Edge(label={edge.label!r}) >> {target}")
        else:
            lines.append(f"    {source} >> {target}")
    return "\n".join(lines) + "\n"


def validate_diagram_source_for(
    profile: str,
    source: str,
    spec: ArchitectureSpecV1,
    output_formats: list[str] | None = None,
) -> dict[str, Any]:
    """Validate diagram source under the named capability profile. Defaults to
    the v1 exact-canonical policy so unknown/absent profiles fail safe-strict."""
    if profile == "diagrams-draw-v2":
        return validate_diagram_source_v2(source, spec, output_formats)
    return validate_diagram_source(source, spec, output_formats)


def canonical_diagram_source_for(
    profile: str, spec: ArchitectureSpecV1, output_formats: list[str] | None = None
) -> str:
    """The deterministic source for a profile — the fallback when model-authored
    code is unavailable or invalid."""
    if profile == "diagrams-draw-v2":
        return canonical_diagram_source_v2(spec, output_formats)
    return canonical_diagram_source(spec, output_formats)


# v2 keeps the same safe primitives but improves layout, and is validated by an
# allowlist rather than an exact match so the model may author varied code.

# Layout attributes passed to graphviz. Orthogonal splines + real spacing are the
# concrete fix for the "floating labels in empty space" look.
_V2_GRAPH_ATTR = {
    "splines": "ortho",
    "nodesep": "0.7",
    "ranksep": "1.1",
    "pad": "0.6",
    "fontsize": "16",
    "fontname": "Sans-Serif",
}
_V2_NODE_ATTR = {"fontsize": "13", "fontname": "Sans-Serif"}
_V2_EDGE_ATTR = {"fontsize": "12", "fontname": "Sans-Serif"}


def canonical_diagram_source_v2(
    spec: ArchitectureSpecV1, output_formats: list[str] | None = None
) -> str:
    """The v2 deterministic source: same safe primitives as v1, but with graphviz
    layout attributes for a cleaner render. Used both as the reference the model
    is asked to improve on and as the guaranteed-safe fallback if model-authored
    code fails validation or rendering."""
    formats = output_formats or ["svg", "png"]
    spec = canonical_architecture_spec(spec)
    components = spec.components
    boundaries = spec.boundaries
    edges = spec.edges
    variable_by_id = {
        component.id: f"node_{index:03d}" for index, component in enumerate(components)
    }
    component_by_id = {component.id: component for component in components}
    grouped_ids = {
        component_id
        for boundary in boundaries
        for component_id in boundary.component_ids
    }
    lines = [
        "from pathlib import Path",
        "from diagrams import Cluster, Diagram, Edge",
        "from diagrams.generic.blank import Blank",
        "",
        'OUTPUT_STEM = str(Path(__file__).resolve().parent / "architecture")',
        "",
        (
            f"with Diagram({spec.title!r}, filename=OUTPUT_STEM, "
            f"outformat={formats!r}, show=False, direction={spec.direction!r}, "
            f"graph_attr={_V2_GRAPH_ATTR!r}, node_attr={_V2_NODE_ATTR!r}, "
            f"edge_attr={_V2_EDGE_ATTR!r}):"
        ),
    ]
    for boundary in boundaries:
        lines.append(f"    with Cluster({boundary.label!r}):")
        for component_id in sorted(boundary.component_ids):
            component = component_by_id[component_id]
            label = f"{component.label}\n[{component.kind}]"
            lines.append(f"        {variable_by_id[component_id]} = Blank({label!r})")
    for component in components:
        if component.id not in grouped_ids:
            label = f"{component.label}\n[{component.kind}]"
            lines.append(f"    {variable_by_id[component.id]} = Blank({label!r})")
    for edge in edges:
        source = variable_by_id[edge.source]
        target = variable_by_id[edge.target]
        if edge.label:
            lines.append(f"    {source} >> Edge(label={edge.label!r}) >> {target}")
        else:
            lines.append(f"    {source} >> {target}")
    return "\n".join(lines) + "\n"


# The only names, attributes, and calls v2 code may contain; all else fails closed.
_V2_ALLOWED_IMPORTS = {
    ("pathlib", "Path"),
    ("diagrams", "Cluster"),
    ("diagrams", "Diagram"),
    ("diagrams", "Edge"),
    ("diagrams.generic.blank", "Blank"),
}
_V2_ALLOWED_CALL_NAMES = {"Blank", "Cluster", "Diagram", "Edge", "Path", "str"}
_V2_ALLOWED_ATTRS = {"resolve", "parent"}  # Path(__file__).resolve().parent
_V2_ALLOWED_DIAGRAM_KWARGS = {
    "filename", "outformat", "show", "direction",
    "graph_attr", "node_attr", "edge_attr",
}
_V2_ALLOWED_EDGE_KWARGS = {"label", "color", "style", "minlen", "reverse"}


class _V2Auditor(ast.NodeVisitor):
    """Walks the AST and records every node kind, name, call, and attribute so
    validate can assert the source stays within the allowlist."""

    def __init__(self) -> None:
        self.node_labels: list[str] = []
        self.rshift_count = 0
        self.edge_call_count = 0
        self.violations: list[str] = []
        self.imports: set[tuple[str, str]] = set()

    @property
    def edge_count(self) -> int:
        # `a >> Edge() >> b` is 2 rshifts but one edge; the intermediate Edge()
        # adds a rshift, so subtracting Edge() calls recovers the true edge count.
        return self.rshift_count - self.edge_call_count

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        self.violations.append("plain 'import' is not allowed; use 'from ... import'")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        module = node.module or ""
        for alias in node.names:
            if (module, alias.name) not in _V2_ALLOWED_IMPORTS:
                self.violations.append(f"disallowed import: {module}.{alias.name}")

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        func = node.func
        name = func.id if isinstance(func, ast.Name) else (
            func.attr if isinstance(func, ast.Attribute) else "<expr>"
        )
        if isinstance(func, ast.Name):
            if func.id not in _V2_ALLOWED_CALL_NAMES:
                self.violations.append(f"disallowed call: {func.id}(...)")
            if func.id == "Blank":
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(
                    node.args[0].value, str
                ):
                    self.node_labels.append(node.args[0].value)
                else:
                    self.violations.append("Blank(...) requires a literal string label")
            if func.id == "Diagram":
                for kw in node.keywords:
                    if kw.arg not in _V2_ALLOWED_DIAGRAM_KWARGS:
                        self.violations.append(f"disallowed Diagram kwarg: {kw.arg}")
            if func.id == "Edge":
                self.edge_call_count += 1
                for kw in node.keywords:
                    if kw.arg not in _V2_ALLOWED_EDGE_KWARGS:
                        self.violations.append(f"disallowed Edge kwarg: {kw.arg}")
        elif isinstance(func, ast.Attribute):
            if func.attr not in _V2_ALLOWED_ATTRS:
                self.violations.append(f"disallowed attribute call: .{func.attr}(...)")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        # Allow only Path-chain attributes and dunder __file__ access.
        if node.attr not in _V2_ALLOWED_ATTRS and not node.attr.startswith("__"):
            self.violations.append(f"disallowed attribute: .{node.attr}")
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:  # noqa: N802
        if isinstance(node.op, ast.RShift):
            self.rshift_count += 1  # node/edge relationship
        elif isinstance(node.op, ast.Div):
            pass  # pathlib join: Path(__file__).resolve().parent / "architecture"
        else:
            self.violations.append(
                "only '>>' (edges) and '/' (path join) operators are allowed"
            )
        self.generic_visit(node)

    # Structural constructs that must never appear — control flow, functions,
    # comprehensions, and other executable machinery.
    def _ban(self, kind: str) -> None:
        self.violations.append(f"disallowed construct: {kind}")

    def visit_For(self, node: ast.For) -> None:  # noqa: N802
        self._ban("for")

    def visit_While(self, node: ast.While) -> None:  # noqa: N802
        self._ban("while")

    def visit_If(self, node: ast.If) -> None:  # noqa: N802
        self._ban("if")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._ban("def")

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
        self._ban("lambda")

    def visit_ListComp(self, node: ast.ListComp) -> None:  # noqa: N802
        self._ban("comprehension")


def validate_diagram_source_v2(
    source: str,
    spec: ArchitectureSpecV1,
    output_formats: list[str] | None = None,
) -> dict[str, Any]:
    """Validate model-authored (or fallback) diagram source against the v2
    allowlist AND prove it covers every component and edge of the spec.

    Unlike v1's exact-AST match, this permits varied, styled code — the model may
    reorder, restyle, and tune layout — as long as it uses only safe primitives
    and represents the whole architecture."""
    try:
        encoded = source.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise DiagramSourceError("source must be UTF-8 encodable") from exc
    if not encoded or len(encoded) > 100_000:
        raise DiagramSourceError("source must contain 1 to 100000 UTF-8 bytes")
    if "\x00" in source or "\r" in source:
        raise DiagramSourceError("source must use LF line endings and contain no NUL bytes")
    try:
        tree = ast.parse(source, filename="diagram.py", mode="exec")
    except SyntaxError as exc:
        raise DiagramSourceError(f"invalid Python syntax: {exc.msg}") from exc

    auditor = _V2Auditor()
    auditor.visit(tree)
    if auditor.violations:
        raise DiagramSourceError(
            "source violates the diagrams-draw-v2 allowlist: "
            + "; ".join(sorted(set(auditor.violations))[:5])
        )

    spec = canonical_architecture_spec(spec)
    expected_labels = {f"{c.label}\n[{c.kind}]" for c in spec.components}
    present_labels = set(auditor.node_labels)
    missing = expected_labels - present_labels
    if missing:
        raise DiagramSourceError(
            f"source is missing {len(missing)} required component node(s)"
        )
    if len(auditor.node_labels) != len(spec.components):
        raise DiagramSourceError(
            "source must declare exactly one node per component"
        )
    if auditor.edge_count != len(spec.edges):
        raise DiagramSourceError(
            f"source declares {auditor.edge_count} edge(s); spec has {len(spec.edges)}"
        )
    return {
        "policy": "diagrams-draw-v2-allowlist",
        "status": "passed",
        "source_bytes": len(encoded),
        "ast_nodes": sum(1 for _ in ast.walk(tree)),
        "components": len(spec.components),
        "edges": len(spec.edges),
        "boundaries": len(spec.boundaries),
    }


def validate_diagram_source(
    source: str,
    spec: ArchitectureSpecV1,
    output_formats: list[str] | None = None,
) -> dict[str, Any]:
    """Prove source has the exact safe AST and semantics expected for this spec."""

    try:
        encoded = source.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise DiagramSourceError("source must be UTF-8 encodable") from exc
    if not encoded or len(encoded) > 100_000:
        raise DiagramSourceError("source must contain 1 to 100000 UTF-8 bytes")
    if "\x00" in source or "\r" in source:
        raise DiagramSourceError("source must use LF line endings and contain no NUL bytes")
    try:
        actual = ast.parse(source, filename="diagram.py", mode="exec")
    except SyntaxError as exc:
        raise DiagramSourceError(f"invalid Python syntax: {exc.msg}") from exc
    expected_source = canonical_diagram_source(spec, output_formats)
    expected = ast.parse(expected_source, filename="diagram.py", mode="exec")
    if ast.dump(actual, include_attributes=False) != ast.dump(
        expected, include_attributes=False
    ):
        raise DiagramSourceError(
            "source must exactly match the validated architecture, output path, "
            "formats, canonical imports, and allowlisted diagram structure"
        )
    return {
        "policy": "exact-canonical-architecture-ast-v1",
        "status": "passed",
        "source_bytes": len(encoded),
        "ast_nodes": sum(1 for _ in ast.walk(actual)),
        "components": len(spec.components),
        "edges": len(spec.edges),
        "boundaries": len(spec.boundaries),
    }
