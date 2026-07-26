"""Deterministic code graph — the Graph-RAG beachhead (Stage 1).

Parse a Python file with the stdlib ``ast`` into graph nodes (the module, its
classes, and its functions/methods) and edges (``contains`` structure,
``imports``, and best-effort ``calls``). No model is involved: extraction is
exact, cheap, and incremental by the same file content hash the corpus already
tracks — so "index all my code" gains a call/def/import graph for free.

Call resolution is intentionally lightweight. We record the callee's *name*
(and the full dotted form as written) rather than doing type inference, because
"who calls ``retrieve``" — matching on the callee name — is the high-value,
low-cost query a coding agent actually needs. Cross-file linking is therefore by
simple name, which can be ambiguous when two symbols share a name; that is an
accepted trade-off for a zero-cost, exact-by-construction graph.

Pure and dependency-free (stdlib ``ast`` only) so it unit-tests without a model,
mirroring ``chunking.py``.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass

# AST nodes that open a new lexical scope. Calls inside one of these are
# attributed to *it*, not to the enclosing function, so each definition owns
# only its own direct calls.
_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_FUNC_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)


@dataclass(frozen=True)
class GraphNode:
    """A definition in the graph. ``qualname`` is dotted and unique within a file."""

    kind: str  # "module" | "class" | "function" | "method"
    name: str  # simple name, e.g. "retrieve"
    qualname: str  # dotted, e.g. "waqil_api.corpus.CorpusService.retrieve"
    start_line: int
    end_line: int


@dataclass(frozen=True)
class GraphEdge:
    """A relationship. ``dst_name`` is the simple target used for name-based
    matching ("who calls X"); ``dst_raw`` is the full dotted form as written."""

    kind: str  # "contains" | "imports" | "calls"
    src: str  # source qualname (the module, or the enclosing function/method)
    dst_name: str  # simple target name for matching
    dst_raw: str  # full dotted target as written, e.g. "self.db.search"
    line: int


def module_qualname(rel_path: str) -> str:
    """Derive a stable dotted module name from a path relative to the source root.

    ``pkg/mod.py`` -> ``pkg.mod``; a package ``pkg/__init__.py`` collapses to
    ``pkg``. Only stability and within-source uniqueness matter (qualnames anchor
    edges), not import-system fidelity.
    """
    cleaned = rel_path.replace("\\", "/").strip("/")
    for suffix in (".pyi", ".py"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
            break
    parts = [part for part in cleaned.split("/") if part and part != "."]
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) if parts else (cleaned or "module")


def _attr_to_str(node: ast.AST) -> str:
    """Best-effort dotted string for a call target (``a.b.c``), else the leaf."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _attr_to_str(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return _attr_to_str(node.func)
    if isinstance(node, ast.Subscript):
        return _attr_to_str(node.value)
    return ""


def _callee(call: ast.Call) -> tuple[str, str] | None:
    """(simple_name, full_dotted) for a call target, or None if not resolvable."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id, func.id
    if isinstance(func, ast.Attribute):
        raw = _attr_to_str(func)
        return func.attr, (raw or func.attr)
    return None


def _import_targets(node: ast.Import | ast.ImportFrom):
    """Yield (dst_name, dst_raw) for each imported target.

    ``import a.b.c`` -> ("a.b.c", "a.b.c"); ``from a.b import foo`` ->
    ("foo", "a.b.foo"); ``from . import foo`` -> ("foo", ".foo").
    """
    if isinstance(node, ast.Import):
        for alias in node.names:
            yield alias.name, alias.name
    else:
        module = node.module or ""
        prefix = ("." * node.level) + module
        for alias in node.names:
            raw = f"{prefix}.{alias.name}" if prefix else alias.name
            yield alias.name, raw


def _calls_in_scope(node: ast.AST):
    """Yield ``ast.Call`` nodes lexically inside ``node`` but not inside a nested
    def/class (those own their own calls)."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, _SCOPE_NODES):
            continue
        if isinstance(child, ast.Call):
            yield child
        yield from _calls_in_scope(child)


def _def_start(node: ast.AST) -> int:
    """First line of a definition, counting decorators so the span matches chunking."""
    decorators = [d.lineno for d in getattr(node, "decorator_list", [])]
    return min(decorators) if decorators else node.lineno


def extract(source: str, rel_path: str) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Extract (nodes, edges) for one Python source file.

    Returns ([], []) on a syntax error, mirroring the chunker's fail-soft
    behavior so one unparseable file never aborts an index run.
    """
    module_qn = module_qualname(rel_path)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [], []

    total_lines = source.count("\n") + 1
    module_name = module_qn.rsplit(".", 1)[-1] or module_qn
    nodes: list[GraphNode] = [
        GraphNode("module", module_name, module_qn, 1, total_lines)
    ]
    edges: list[GraphEdge] = []

    def walk(body: list[ast.stmt], parent_qn: str, parent_kind: str) -> None:
        for node in body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for dst_name, dst_raw in _import_targets(node):
                    edges.append(
                        GraphEdge("imports", module_qn, dst_name, dst_raw, node.lineno)
                    )
            elif isinstance(node, _FUNC_NODES):
                qualname = f"{parent_qn}.{node.name}"
                kind = "method" if parent_kind == "class" else "function"
                nodes.append(
                    GraphNode(
                        kind, node.name, qualname,
                        _def_start(node), getattr(node, "end_lineno", node.lineno),
                    )
                )
                edges.append(
                    GraphEdge("contains", parent_qn, node.name, qualname, node.lineno)
                )
                for call in _calls_in_scope(node):
                    resolved = _callee(call)
                    if resolved is not None:
                        edges.append(
                            GraphEdge("calls", qualname, resolved[0], resolved[1], call.lineno)
                        )
                walk(node.body, qualname, kind)
            elif isinstance(node, ast.ClassDef):
                qualname = f"{parent_qn}.{node.name}"
                nodes.append(
                    GraphNode(
                        "class", node.name, qualname,
                        _def_start(node), getattr(node, "end_lineno", node.lineno),
                    )
                )
                edges.append(
                    GraphEdge("contains", parent_qn, node.name, qualname, node.lineno)
                )
                walk(node.body, qualname, "class")

    walk(tree.body, module_qn, "module")
    return nodes, edges
