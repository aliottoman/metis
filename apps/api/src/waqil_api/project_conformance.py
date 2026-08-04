"""Does the changeset do what the turn set out to do?

The rungs below this one ask whether the code is well-formed: it parses, its
imports resolve, it imports and serves its routes in a container. All three can
pass on a build that does not work. Six measured builds of the same request
produced, every single time, a frontend that POSTs a JSON body to an endpoint
declaring a bare scalar — which FastAPI reads as a query parameter, so every
request 422s — and every rung called it clean, because it is well-formed code.

So this rung compares the changeset to the turn's own intentions rather than to
the language. Every check here is deterministic and derived from the code or
from the file manifest the turn already committed to. Nothing is inferred from
model-authored prose: these findings withhold the Approve button, and a
requirement a planner phrased badly would then block a correct build — the
exact failure the sandbox rung had to be de-escalated for.
"""
from __future__ import annotations

import ast
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ERROR = "error"
WARNING = "warning"

# Parameter annotations FastAPI reads from the query string rather than the body.
_SCALAR_ANNOTATIONS = frozenset({"str", "int", "float", "bool", "UUID", "Decimal"})

# Names that mean "this parameter is not the request body": the request itself,
# an uploaded file, a background-task handle, a dependency.
_NON_BODY_ANNOTATIONS = frozenset(
    {"Request", "UploadFile", "BackgroundTasks", "Response", "WebSocket"}
)

_HTTP_METHODS = frozenset({"get", "post", "put", "patch", "delete"})

# fetch("<url>", { ... }) — the URL is all this needs; the options object is
# matched separately because its keys arrive in any order.
_FETCH_CALL = re.compile(
    r"""fetch\(\s*(?P<quote>["'`])(?P<url>[^"'`]+)(?P=quote)\s*,\s*(?P<options>\{)""",
    re.S,
)
_JSON_BODY = re.compile(r"body\s*:\s*JSON\.stringify", re.I)
_METHOD_IN_OPTIONS = re.compile(r"""method\s*:\s*["'`](?P<method>\w+)["'`]""", re.I)

# `${...}` in a template literal, and `{name}` in a FastAPI route: both stand in
# for one path segment, so both normalise to the same placeholder.
_JS_INTERPOLATION = re.compile(r"\$\{[^}]*\}")
_ROUTE_PARAMETER = re.compile(r"\{[^}]*\}")


def _finding(path: str, error: str, severity: str = ERROR) -> dict[str, str]:
    """One reportable defect, in the shape every other rung returns."""
    return {"path": path, "error": error, "severity": severity}


def _normalise_path(url: str) -> str:
    """A request path with every variable segment collapsed to one placeholder."""
    without_query = url.split("?", 1)[0].split("#", 1)[0]
    collapsed = _JS_INTERPOLATION.sub("{}", without_query)
    collapsed = _ROUTE_PARAMETER.sub("{}", collapsed)
    return collapsed.rstrip("/") or "/"


def _balanced_object(text: str, start: int) -> str:
    """The `{...}` beginning at `start`, respecting nesting."""
    depth = 0
    for index in range(start, len(text)):
        character = text[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return text[start:]


def json_body_requests(script: str) -> set[tuple[str, str]]:
    """Every (METHOD, path) the frontend calls with a JSON body."""
    requests: set[tuple[str, str]] = set()
    for match in _FETCH_CALL.finditer(script):
        options = _balanced_object(script, match.start("options"))
        if not _JSON_BODY.search(options):
            continue
        method = _METHOD_IN_OPTIONS.search(options)
        requests.add(
            (
                (method.group("method") if method else "GET").upper(),
                _normalise_path(match.group("url")),
            )
        )
    return requests


def _model_names(trees: dict[str, ast.Module]) -> set[str]:
    """Classes in the changeset that look like request bodies."""
    names: set[str] = set()
    for tree in trees.values():
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = {
                base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", "")
                for base in node.bases
            }
            if bases & {"BaseModel", "Contract"} or any(
                base.endswith("Model") for base in bases if base
            ):
                names.add(node.name)
    return names


def _accepts_a_body(function: ast.FunctionDef | ast.AsyncFunctionDef, models: set[str]) -> bool:
    """Whether this handler declares anything FastAPI would read from the body."""
    arguments = [*function.args.args, *function.args.kwonlyargs]
    for argument in arguments:
        annotation = argument.annotation
        # An explicit Body()/File()/Form() default is a body parameter whatever
        # the annotation says.
        default = _default_for(function, argument)
        if isinstance(default, ast.Call):
            factory = getattr(default.func, "id", "") or getattr(default.func, "attr", "")
            if factory in {"Body", "File", "Form"}:
                return True
        if annotation is None:
            continue
        name = _annotation_name(annotation)
        if name in _NON_BODY_ANNOTATIONS or name in _SCALAR_ANNOTATIONS:
            continue
        # A Pydantic model, a dict, or a list is read from the body.
        if name in models or name in {"dict", "list", "Dict", "List", "Any"}:
            return True
    return False


def _annotation_name(annotation: ast.expr) -> str:
    """The outermost name of an annotation, ignoring Optional/Annotated wrappers."""
    if isinstance(annotation, ast.Name):
        return annotation.id
    if isinstance(annotation, ast.Attribute):
        return annotation.attr
    if isinstance(annotation, ast.Subscript):
        return _annotation_name(annotation.value)
    if isinstance(annotation, ast.BinOp):  # `Model | None`
        return _annotation_name(annotation.left)
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        return annotation.value
    return ""


def _default_for(
    function: ast.FunctionDef | ast.AsyncFunctionDef, argument: ast.arg
) -> ast.expr | None:
    """The default expression bound to one argument, if it has one."""
    positional = function.args.args
    defaults = function.args.defaults
    if argument in positional and defaults:
        offset = len(positional) - len(defaults)
        index = positional.index(argument) - offset
        if index >= 0:
            return defaults[index]
    if argument in function.args.kwonlyargs:
        index = function.args.kwonlyargs.index(argument)
        default = function.args.kw_defaults[index]
        return default
    return None


def _routes(trees: dict[str, ast.Module], models: set[str]) -> dict[tuple[str, str], tuple[str, int, bool]]:
    """Every declared route → (file, line, whether it accepts a request body)."""
    routes: dict[tuple[str, str], tuple[str, int, bool]] = {}
    for path, tree in trees.items():
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                method = getattr(decorator.func, "attr", "")
                if method not in _HTTP_METHODS or not decorator.args:
                    continue
                url = decorator.args[0]
                if not isinstance(url, ast.Constant) or not isinstance(url.value, str):
                    continue
                key = (method.upper(), _normalise_path(url.value))
                routes[key] = (path, node.lineno, _accepts_a_body(node, models))
    return routes


def request_shape_findings(
    trees: dict[str, ast.Module], scripts: dict[str, str], models: set[str]
) -> list[dict[str, str]]:
    """Frontend calls whose request body the backend would never read.

    A handler declaring `prompt: str` reads it from the query string, so a
    frontend posting `JSON.stringify({prompt})` gets 422 on every request while
    every other rung reports the project healthy. Only flagged when the frontend
    demonstrably sends a JSON body and the matching route demonstrably declares
    no body parameter — a route the changeset does not define is the wiring
    gate's business, not this one's.
    """
    routes = _routes(trees, models)
    findings: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for script_path, script in sorted(scripts.items()):
        for method, url in sorted(json_body_requests(script)):
            route = routes.get((method, url))
            if route is None or route[2] or (method, url) in seen:
                continue
            seen.add((method, url))
            handler_path, lineno, _ = route
            findings.append(
                _finding(
                    handler_path,
                    f"{script_path} sends a JSON body to {method} {url}, but the "
                    f"handler on line {lineno} declares no body parameter — FastAPI "
                    "reads its arguments from the query string, so every one of "
                    "those requests fails with 422. Take a Pydantic model.",
                )
            )
    return findings


def missing_planned_files(
    staged: dict[str, dict[str, Any]], planned: list[str], on_disk: set[str]
) -> list[dict[str, str]]:
    """Files the turn said it would write and did not.

    The manifest is the turn's own commitment, taken before any file was
    written, so an unmet entry is not a matter of opinion. `.env.example` went
    missing in six consecutive builds while every other rung reported the
    project healthy.
    """
    findings: list[dict[str, str]] = []
    for path in planned:
        if path in staged or path in on_disk:
            continue
        findings.append(
            _finding(
                path,
                "was planned for this build and never written. Create it, or "
                "say plainly in your summary why it is not needed.",
            )
        )
    return findings


def undocumented_settings(
    trees: dict[str, ast.Module], staged: dict[str, dict[str, Any]]
) -> list[dict[str, str]]:
    """Environment variables the code reads that the example file never names.

    Only runs when the changeset actually writes an example env file: without
    one there is nothing to be inconsistent with, and demanding one is a style
    opinion rather than a defect.
    """
    example = next(
        (
            str(entry.get("content", ""))
            for path, entry in staged.items()
            if Path(path).name in {".env.example", "env.example"}
        ),
        None,
    )
    if example is None:
        return []
    documented = {
        line.split("=", 1)[0].strip().lstrip("#").strip()
        for line in example.splitlines()
        if "=" in line
    }
    findings: list[dict[str, str]] = []
    for path, tree in sorted(trees.items()):
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            target = getattr(node.func, "attr", "")
            owner = getattr(getattr(node.func, "value", None), "id", "")
            if owner != "os" or target not in {"getenv", "environ"}:
                continue
            if not node.args or not isinstance(node.args[0], ast.Constant):
                continue
            name = node.args[0].value
            if not isinstance(name, str) or name in documented:
                continue
            findings.append(
                _finding(
                    path,
                    f"reads {name} from the environment on line {node.lineno}, but "
                    ".env.example never names it, so nobody running this project "
                    "would know to set it.",
                    WARNING,
                )
            )
    return findings


def staged_conformance_errors(
    staged: dict[str, dict[str, Any]],
    *,
    planned: list[str] | None = None,
    on_disk: Iterable[str] = (),
) -> list[dict[str, str]]:
    """Every way this changeset falls short of what the turn set out to do."""
    trees: dict[str, ast.Module] = {}
    scripts: dict[str, str] = {}
    for path, entry in staged.items():
        content = str(entry.get("content", ""))
        suffix = Path(path).suffix.lower()
        if suffix in {".py", ".pyi"}:
            try:
                trees[path] = ast.parse(content, filename=path)
            except (SyntaxError, ValueError):
                # The syntax rung owns this file; nothing here can add to it.
                continue
        elif suffix in {".js", ".mjs", ".ts"}:
            scripts[path] = content

    models = _model_names(trees)
    findings = [
        *missing_planned_files(staged, list(planned or []), set(on_disk)),
        *request_shape_findings(trees, scripts, models),
        *undocumented_settings(trees, staged),
    ]
    findings.sort(key=lambda item: (item["severity"] != ERROR, item["path"]))
    return findings
