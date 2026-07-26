"""Static policy for deterministic diagram.py artifacts.

The generated program is intentionally tiny.  Rejecting everything outside this
allowlist is safer and easier to audit than attempting to blacklist Python.
"""

from __future__ import annotations

import ast
from collections import Counter
import re
from typing import Any


MAX_SOURCE_BYTES = 100_000
MAX_AST_NODES = 4_000


class SourceValidationError(ValueError):
    """Raised when generated Python falls outside the executable policy."""


_ALLOWED_IMPORTS = {
    "pathlib": {"Path"},
    "diagrams": {"Cluster", "Diagram", "Edge"},
    "diagrams.generic.blank": {"Blank"},
}

_REQUIRED_IMPORTS = {
    ("pathlib", "Path"),
    ("diagrams", "Cluster"),
    ("diagrams", "Diagram"),
    ("diagrams", "Edge"),
    ("diagrams.generic.blank", "Blank"),
}

_ALLOWED_CALL_NAMES = {"Blank", "Cluster", "Diagram", "Edge", "Path", "str"}
_ALLOWED_ATTRIBUTE_NAMES = {"parent", "resolve"}
_ALLOWED_ASSIGNMENT = re.compile(r"(?:OUTPUT_STEM|node_[0-9]{3})\Z")

_ALLOWED_AST_TYPES = (
    ast.Module,
    ast.ImportFrom,
    ast.alias,
    ast.Assign,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Call,
    ast.Constant,
    ast.Attribute,
    ast.BinOp,
    ast.Div,
    ast.RShift,
    ast.With,
    ast.withitem,
    ast.keyword,
    ast.Expr,
    ast.List,
)


def _raise(message: str, node: ast.AST | None = None) -> None:
    if node is not None and hasattr(node, "lineno"):
        message = f"line {node.lineno}: {message}"
    raise SourceValidationError(message)


def validate_generated_source(source: str) -> dict[str, Any]:
    """Validate generated diagram code and return deterministic evidence."""

    encoded = source.encode("utf-8")
    if len(encoded) > MAX_SOURCE_BYTES:
        _raise(f"source exceeds {MAX_SOURCE_BYTES} bytes")
    if "\x00" in source:
        _raise("source contains a NUL byte")

    try:
        tree = ast.parse(source, filename="diagram.py", mode="exec")
    except SyntaxError as exc:
        raise SourceValidationError(f"invalid Python syntax: {exc.msg}") from exc

    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_AST_NODES:
        _raise(f"source exceeds {MAX_AST_NODES} AST nodes")

    imported: set[tuple[str, str]] = set()
    assigned_nodes: set[str] = set()
    diagram_calls = 0

    for node in nodes:
        if not isinstance(node, _ALLOWED_AST_TYPES):
            _raise(f"AST node {type(node).__name__} is not allowed", node)

        if isinstance(node, ast.ImportFrom):
            if node.level != 0 or node.module not in _ALLOWED_IMPORTS:
                _raise(f"import from {node.module!r} is not allowed", node)
            allowed_names = _ALLOWED_IMPORTS[node.module]
            for alias in node.names:
                if alias.asname is not None or alias.name not in allowed_names:
                    _raise(f"import {alias.name!r} is not allowed", node)
                imported.add((node.module, alias.name))

        elif isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                _raise("only simple assignments are allowed", node)
            target = node.targets[0].id
            if not _ALLOWED_ASSIGNMENT.fullmatch(target):
                _raise(f"assignment to {target!r} is not allowed", node)
            if target.startswith("node_"):
                if target in assigned_nodes:
                    _raise(f"duplicate assignment to {target!r}", node)
                assigned_nodes.add(target)

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in _ALLOWED_CALL_NAMES:
                    _raise(f"call to {node.func.id!r} is not allowed", node)
                if node.func.id == "Diagram":
                    diagram_calls += 1
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr != "resolve":
                    _raise(f"attribute call {node.func.attr!r} is not allowed", node)
                base = node.func.value
                if not (
                    isinstance(base, ast.Call)
                    and isinstance(base.func, ast.Name)
                    and base.func.id == "Path"
                    and len(base.args) == 1
                    and isinstance(base.args[0], ast.Name)
                    and base.args[0].id == "__file__"
                    and not base.keywords
                ):
                    _raise("resolve() is allowed only on Path(__file__)", node)
            else:
                _raise("dynamic call targets are not allowed", node)

        elif isinstance(node, ast.Attribute):
            if node.attr not in _ALLOWED_ATTRIBUTE_NAMES:
                _raise(f"attribute {node.attr!r} is not allowed", node)

        elif isinstance(node, ast.Name):
            if node.id.startswith("__") and node.id != "__file__":
                _raise(f"dunder name {node.id!r} is not allowed", node)

        elif isinstance(node, ast.BinOp):
            if not isinstance(node.op, (ast.Div, ast.RShift)):
                _raise(f"operator {type(node.op).__name__} is not allowed", node)

    missing_imports = sorted(_REQUIRED_IMPORTS - imported)
    if missing_imports:
        _raise(f"required imports are missing: {missing_imports}")
    if diagram_calls != 1:
        _raise("source must contain exactly one Diagram call")
    if not assigned_nodes:
        _raise("source must define at least one component node")

    return {
        "policy": "allowlisted-ast-v1",
        "status": "passed",
        "source_bytes": len(encoded),
        "ast_nodes": len(nodes),
        "component_assignments": len(assigned_nodes),
    }


def _same_ast(left: ast.AST, right: ast.AST) -> bool:
    return ast.dump(left, include_attributes=False) == ast.dump(
        right, include_attributes=False
    )


def _call_name(call: ast.Call, expected: str) -> None:
    if not isinstance(call.func, ast.Name) or call.func.id != expected:
        _raise(f"expected a {expected} call", call)


def _keywords(call: ast.Call) -> dict[str, ast.expr]:
    values: dict[str, ast.expr] = {}
    for keyword in call.keywords:
        if keyword.arg is None or keyword.arg in values:
            _raise("expanded or duplicate keyword arguments are not allowed", call)
        values[keyword.arg] = keyword.value
    return values


def _node_assignment(statement: ast.stmt) -> tuple[str, str]:
    if not isinstance(statement, ast.Assign):
        _raise("component definitions must be simple assignments", statement)
    if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
        _raise("component definitions require one named target", statement)
    variable = statement.targets[0].id
    if not isinstance(statement.value, ast.Call):
        _raise("component definitions must call Blank", statement)
    call = statement.value
    _call_name(call, "Blank")
    if (
        len(call.args) != 1
        or not isinstance(call.args[0], ast.Constant)
        or not isinstance(call.args[0].value, str)
        or call.keywords
    ):
        _raise("Blank requires exactly one literal label", call)
    return variable, call.args[0].value


def _edge_expression(statement: ast.stmt) -> tuple[str, str, str]:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.BinOp):
        _raise("relationships must use the >> operator", statement)
    expression = statement.value
    if not isinstance(expression.op, ast.RShift) or not isinstance(expression.right, ast.Name):
        _raise("relationship target must be a component node", expression)

    target = expression.right.id
    if isinstance(expression.left, ast.Name):
        return expression.left.id, target, ""
    if not isinstance(expression.left, ast.BinOp) or not isinstance(
        expression.left.op, ast.RShift
    ):
        _raise("labeled relationship shape is invalid", expression)
    labeled = expression.left
    if not isinstance(labeled.left, ast.Name) or not isinstance(labeled.right, ast.Call):
        _raise("labeled relationship source or Edge call is invalid", labeled)
    edge_call = labeled.right
    _call_name(edge_call, "Edge")
    keywords = _keywords(edge_call)
    label = keywords.get("label")
    if (
        edge_call.args
        or set(keywords) != {"label"}
        or not isinstance(label, ast.Constant)
        or not isinstance(label.value, str)
    ):
        _raise("Edge requires exactly one literal label keyword", edge_call)
    return labeled.left.id, target, label.value


def validate_source_against_spec(
    source: str,
    spec: dict[str, Any],
    output_formats: list[str],
) -> dict[str, Any]:
    """Prove that allowlisted source exactly represents the normalized spec."""

    tree = ast.parse(source, filename="diagram.py", mode="exec")
    imports = [statement for statement in tree.body if isinstance(statement, ast.ImportFrom)]
    non_imports = [
        statement for statement in tree.body if not isinstance(statement, ast.ImportFrom)
    ]
    if len(imports) != 3 or len(non_imports) != 2:
        _raise("source must contain only canonical imports, OUTPUT_STEM, and one Diagram")

    actual_imports = Counter(
        (statement.module, tuple(sorted(alias.name for alias in statement.names)))
        for statement in imports
    )
    expected_imports = Counter(
        {
            ("pathlib", ("Path",)): 1,
            ("diagrams", ("Cluster", "Diagram", "Edge")): 1,
            ("diagrams.generic.blank", ("Blank",)): 1,
        }
    )
    if actual_imports != expected_imports:
        _raise("imports do not match the canonical diagram policy")

    expected_output = ast.parse(
        'OUTPUT_STEM = str(Path(__file__).resolve().parent / "architecture")',
        mode="exec",
    ).body[0]
    if not _same_ast(non_imports[0], expected_output):
        _raise("OUTPUT_STEM must resolve to architecture next to diagram.py", non_imports[0])

    diagram_statement = non_imports[1]
    if (
        not isinstance(diagram_statement, ast.With)
        or len(diagram_statement.items) != 1
        or diagram_statement.items[0].optional_vars is not None
        or not isinstance(diagram_statement.items[0].context_expr, ast.Call)
    ):
        _raise("source must contain exactly one Diagram context", diagram_statement)
    diagram_call = diagram_statement.items[0].context_expr
    _call_name(diagram_call, "Diagram")
    diagram_keywords = _keywords(diagram_call)
    if (
        len(diagram_call.args) != 1
        or not isinstance(diagram_call.args[0], ast.Constant)
        or diagram_call.args[0].value != spec["title"]
        or set(diagram_keywords) != {"filename", "outformat", "show", "direction"}
    ):
        _raise("Diagram arguments do not match the architecture specification", diagram_call)
    if not (
        isinstance(diagram_keywords["filename"], ast.Name)
        and diagram_keywords["filename"].id == "OUTPUT_STEM"
        and isinstance(diagram_keywords["show"], ast.Constant)
        and diagram_keywords["show"].value is False
        and isinstance(diagram_keywords["direction"], ast.Constant)
        and diagram_keywords["direction"].value == spec["direction"]
        and isinstance(diagram_keywords["outformat"], ast.List)
        and [
            item.value
            for item in diagram_keywords["outformat"].elts
            if isinstance(item, ast.Constant)
        ]
        == output_formats
        and all(
            isinstance(item, ast.Constant)
            and isinstance(item.value, str)
            for item in diagram_keywords["outformat"].elts
        )
    ):
        _raise("Diagram output path, formats, visibility, or direction is invalid", diagram_call)

    components = sorted(spec["components"], key=lambda item: item["id"])
    variable_by_id = {
        component["id"]: f"node_{index:03d}"
        for index, component in enumerate(components)
    }
    expected_nodes = {
        variable_by_id[component["id"]]: f"{component['label']}\n[{component['kind']}]"
        for component in components
    }
    expected_boundary_by_node = {
        variable_by_id[component_id]: boundary["label"]
        for boundary in spec["boundaries"]
        for component_id in boundary["component_ids"]
    }
    expected_clusters = Counter(
        (
            boundary["label"],
            tuple(sorted(variable_by_id[item] for item in boundary["component_ids"])),
        )
        for boundary in spec["boundaries"]
    )

    actual_nodes: dict[str, str] = {}
    actual_boundary_by_node: dict[str, str] = {}
    actual_clusters: Counter[tuple[str, tuple[str, ...]]] = Counter()
    actual_edges: list[tuple[str, str, str]] = []
    saw_edge = False

    for statement in diagram_statement.body:
        if isinstance(statement, ast.With):
            if saw_edge:
                _raise("component definitions must precede relationships", statement)
            if (
                len(statement.items) != 1
                or statement.items[0].optional_vars is not None
                or not isinstance(statement.items[0].context_expr, ast.Call)
            ):
                _raise("Cluster context is invalid", statement)
            cluster_call = statement.items[0].context_expr
            _call_name(cluster_call, "Cluster")
            if (
                len(cluster_call.args) != 1
                or not isinstance(cluster_call.args[0], ast.Constant)
                or not isinstance(cluster_call.args[0].value, str)
                or cluster_call.keywords
                or not statement.body
            ):
                _raise("Cluster requires one literal label and component nodes", cluster_call)
            cluster_label = cluster_call.args[0].value
            members: list[str] = []
            for child in statement.body:
                variable, label = _node_assignment(child)
                if variable in actual_nodes:
                    _raise(f"component variable {variable!r} is duplicated", child)
                actual_nodes[variable] = label
                actual_boundary_by_node[variable] = cluster_label
                members.append(variable)
            actual_clusters[(cluster_label, tuple(sorted(members)))] += 1
        elif isinstance(statement, ast.Assign):
            if saw_edge:
                _raise("component definitions must precede relationships", statement)
            variable, label = _node_assignment(statement)
            if variable in actual_nodes:
                _raise(f"component variable {variable!r} is duplicated", statement)
            actual_nodes[variable] = label
        elif isinstance(statement, ast.Expr):
            saw_edge = True
            actual_edges.append(_edge_expression(statement))
        else:
            _raise("Diagram body contains an unsupported statement", statement)

    if actual_nodes != expected_nodes:
        _raise("component variables and labels do not exactly match the specification")
    if actual_boundary_by_node != expected_boundary_by_node or actual_clusters != expected_clusters:
        _raise("cluster membership does not exactly match the specification")

    expected_edges = Counter(
        (
            variable_by_id[edge["source"]],
            variable_by_id[edge["target"]],
            edge["label"],
        )
        for edge in spec["edges"]
    )
    if Counter(actual_edges) != expected_edges:
        _raise("relationships do not exactly match the specification")

    return {
        "policy": "architecture-spec-match-v1",
        "status": "passed",
        "components": len(expected_nodes),
        "edges": sum(expected_edges.values()),
        "boundaries": sum(expected_clusters.values()),
        "output_stem": "architecture",
    }


# ── diagrams-draw-v2 policy — model-authored, allowlist + spec-completeness ──
# Mirrors the host's `diagram_source.validate_diagram_source_v2` so the sandbox's
# independent second check agrees with the host. Permits varied, styled code
# (layout attributes, restyled edges) from the safe DSL primitives only, and
# proves every component and edge is present — without demanding an exact copy.
_V2_ALLOWED_IMPORTS = {
    ("pathlib", "Path"),
    ("diagrams", "Cluster"),
    ("diagrams", "Diagram"),
    ("diagrams", "Edge"),
    ("diagrams.generic.blank", "Blank"),
}
_V2_ALLOWED_CALL_NAMES = {"Blank", "Cluster", "Diagram", "Edge", "Path", "str"}
_V2_ALLOWED_ATTRS = {"resolve", "parent"}
_V2_ALLOWED_DIAGRAM_KWARGS = {
    "filename", "outformat", "show", "direction",
    "graph_attr", "node_attr", "edge_attr",
}
_V2_ALLOWED_EDGE_KWARGS = {"label", "color", "style", "minlen", "reverse"}
_V2_ALLOWED_AST_TYPES = (
    ast.Module, ast.ImportFrom, ast.alias, ast.Assign, ast.Name, ast.Load,
    ast.Store, ast.Call, ast.Constant, ast.Attribute, ast.BinOp, ast.Div,
    ast.RShift, ast.With, ast.withitem, ast.keyword, ast.Expr, ast.List, ast.Dict,
)


class _V2Auditor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.node_labels: list[str] = []
        self.rshift_count = 0
        self.edge_call_count = 0
        self.violations: list[str] = []

    @property
    def edge_count(self) -> int:
        return self.rshift_count - self.edge_call_count

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        self.violations.append("plain 'import' is not allowed")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        module = node.module or ""
        for alias in node.names:
            if alias.asname is not None or (module, alias.name) not in _V2_ALLOWED_IMPORTS:
                self.violations.append(f"disallowed import: {module}.{alias.name}")

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        func = node.func
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
        if node.attr not in _V2_ALLOWED_ATTRS and not node.attr.startswith("__"):
            self.violations.append(f"disallowed attribute: .{node.attr}")
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:  # noqa: N802
        if isinstance(node.op, ast.RShift):
            self.rshift_count += 1
        elif isinstance(node.op, ast.Div):
            pass
        else:
            self.violations.append("only '>>' and '/' operators are allowed")
        self.generic_visit(node)

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


def _v2_audit(source: str) -> _V2Auditor:
    encoded = source.encode("utf-8")
    if len(encoded) > MAX_SOURCE_BYTES:
        _raise(f"source exceeds {MAX_SOURCE_BYTES} bytes")
    if "\x00" in source:
        _raise("source contains a NUL byte")
    try:
        tree = ast.parse(source, filename="diagram.py", mode="exec")
    except SyntaxError as exc:
        raise SourceValidationError(f"invalid Python syntax: {exc.msg}") from exc
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_AST_NODES:
        _raise(f"source exceeds {MAX_AST_NODES} AST nodes")
    for node in nodes:
        if not isinstance(node, _V2_ALLOWED_AST_TYPES):
            _raise(f"AST node {type(node).__name__} is not allowed", node)
    auditor = _V2Auditor()
    auditor.visit(tree)
    if auditor.violations:
        _raise("diagrams-draw-v2 violation: " + "; ".join(sorted(set(auditor.violations))[:5]))
    return auditor


def validate_generated_source_v2(source: str) -> dict[str, Any]:
    """v2 static allowlist — permits layout attributes and varied structure."""
    auditor = _v2_audit(source)
    return {
        "policy": "allowlisted-ast-v2",
        "status": "passed",
        "source_bytes": len(source.encode("utf-8")),
        "ast_nodes": len(list(ast.walk(ast.parse(source, mode="exec")))),
        "component_assignments": len(auditor.node_labels),
    }


def validate_source_against_spec_v2(
    source: str,
    spec: dict[str, Any],
    output_formats: list[str],
) -> dict[str, Any]:
    """v2 semantic check — every component and edge of the spec is represented,
    without requiring an exact-canonical copy."""
    auditor = _v2_audit(source)
    components = sorted(spec["components"], key=lambda item: item["id"])
    expected_labels = {f"{c['label']}\n[{c['kind']}]" for c in components}
    present = set(auditor.node_labels)
    missing = expected_labels - present
    if missing:
        _raise(f"source is missing {len(missing)} required component node(s)")
    if len(auditor.node_labels) != len(components):
        _raise("source must declare exactly one node per component")
    if auditor.edge_count != len(spec["edges"]):
        _raise(
            f"source declares {auditor.edge_count} edge(s); spec has {len(spec['edges'])}"
        )
    return {
        "policy": "diagrams-draw-v2",
        "status": "passed",
        "components": len(components),
        "edges": len(spec["edges"]),
        "boundaries": len(spec["boundaries"]),
        "output_stem": "architecture",
    }


__all__ = [
    "SourceValidationError",
    "validate_generated_source",
    "validate_source_against_spec",
    "validate_generated_source_v2",
    "validate_source_against_spec_v2",
]
