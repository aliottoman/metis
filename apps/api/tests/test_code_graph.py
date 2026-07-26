from __future__ import annotations

from waqil_api.code_graph import extract, module_qualname

SOURCE = """\
import os
from a.b import foo as bar

TABLE = 1


class Widget:
    def render(self):
        helper()
        return bar()


def helper():
    os.getpid()

    def inner():
        Widget().render()

    return inner
"""


def _nodes_by_kind(nodes):
    grouped: dict[str, set[str]] = {}
    for node in nodes:
        grouped.setdefault(node.kind, set()).add(node.qualname)
    return grouped


def _edges(edges, kind):
    return [(e.src, e.dst_name, e.dst_raw) for e in edges if e.kind == kind]


def test_module_qualname_variants() -> None:
    assert module_qualname("pkg/widget.py") == "pkg.widget"
    assert module_qualname("pkg/__init__.py") == "pkg"
    assert module_qualname("a/b/c.pyi") == "a.b.c"
    assert module_qualname("solo.py") == "solo"


def test_extract_nodes_cover_module_classes_and_functions() -> None:
    nodes, _ = extract(SOURCE, "pkg/widget.py")
    grouped = _nodes_by_kind(nodes)
    assert grouped["module"] == {"pkg.widget"}
    assert grouped["class"] == {"pkg.widget.Widget"}
    assert grouped["method"] == {"pkg.widget.Widget.render"}
    # Top-level function and the nested function both appear, correctly qualified.
    assert grouped["function"] == {"pkg.widget.helper", "pkg.widget.helper.inner"}


def test_line_spans_include_decorators_and_bodies() -> None:
    nodes, _ = extract(SOURCE, "pkg/widget.py")
    render = next(n for n in nodes if n.qualname.endswith("Widget.render"))
    assert render.start_line <= render.end_line
    module = next(n for n in nodes if n.kind == "module")
    assert module.start_line == 1
    assert module.end_line >= render.end_line


def test_imports_are_captured_with_dotted_targets() -> None:
    _, edges = extract(SOURCE, "pkg/widget.py")
    imports = _edges(edges, "imports")
    assert ("pkg.widget", "os", "os") in imports
    assert ("pkg.widget", "foo", "a.b.foo") in imports


def test_calls_are_attributed_to_the_enclosing_scope() -> None:
    _, edges = extract(SOURCE, "pkg/widget.py")
    calls = _edges(edges, "calls")
    # render() calls helper() and bar()
    assert ("pkg.widget.Widget.render", "helper", "helper") in calls
    assert ("pkg.widget.Widget.render", "bar", "bar") in calls
    # helper() calls os.getpid(); its call is NOT leaked into inner()
    assert ("pkg.widget.helper", "getpid", "os.getpid") in calls
    # the nested inner() owns its own calls, keyed by its own qualname
    assert ("pkg.widget.helper.inner", "render", "Widget.render") in calls
    assert ("pkg.widget.helper.inner", "Widget", "Widget") in calls
    # calls made inside inner() are never attributed to helper()
    assert ("pkg.widget.helper", "render", "Widget.render") not in calls


def test_contains_edges_model_structure() -> None:
    _, edges = extract(SOURCE, "pkg/widget.py")
    contains = _edges(edges, "contains")
    assert ("pkg.widget", "Widget", "pkg.widget.Widget") in contains
    assert ("pkg.widget.Widget", "render", "pkg.widget.Widget.render") in contains
    assert ("pkg.widget.helper", "inner", "pkg.widget.helper.inner") in contains


def test_syntax_error_is_fail_soft() -> None:
    nodes, edges = extract("def broken(:\n    pass\n", "bad.py")
    assert nodes == []
    assert edges == []
