"""In-sandbox verifier: import a freshly built project and exercise its app.

Runs inside the Metis verify container — no network, no writable project, a
bounded CPU and memory allowance — and is the only place model-authored project
code is ever executed. It reads one JSON request on stdin, imports the modules
that request names, looks for an ASGI application, requests the routes the
application declares, and writes a single JSON envelope on stdout.

It never raises. Every outcome, including its own failure, comes back as data
the host can read, because a verifier that dies without explaining itself is
indistinguishable from a project that is fine.
"""
from __future__ import annotations

import base64
import contextlib
import importlib
import io
import json
import os
import re
import shutil
import signal
import sys
import traceback
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

SCHEMA_VERSION = "1"
PROJECT_DIR = Path("/tmp/project")
INPUT_DIR = Path("/input")

# Bounds. The wrapper enforces the wall clock; these keep one pathological
# module or a chatty import from consuming the whole budget or the envelope.
MAX_DETAIL_CHARS = 2_000
MAX_CAPTURED_CHARS = 4_000
MAX_ROUTES = 64
MAX_REQUESTS = 12
MAX_BODY_REQUESTS = 10
DEFAULT_IMPORT_SECONDS = 20

# A real 1×1 PNG, so a route that sniffs magic bytes or decodes its upload
# treats the fixture as a genuine image instead of failing on a fake.
PNG_FIXTURE = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


class ImportTimeout(Exception):
    """Raised by the alarm handler when one module takes too long to import."""


def _fail(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    """The envelope for a verifier that could not do its job at all."""
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        "error": {"code": code, "message": message, "details": details or {}},
        "checks": [],
        "routes": [],
    }


def _bounded(value: object, limit: int = MAX_DETAIL_CHARS) -> str:
    """A string safe to put in the envelope, however large the input was."""
    text = str(value)
    return text if len(text) <= limit else f"{text[:limit]}… (truncated)"


def _read_request() -> dict[str, Any]:
    """Read the host's request, then make stdin unreadable to project code."""
    raw = sys.stdin.buffer.read()
    sys.stdin = io.StringIO("")
    if not raw:
        return {}
    payload = json.loads(raw.decode("utf-8"))
    return payload if isinstance(payload, dict) else {}


def _copy_project() -> None:
    """Copy the read-only mount into tmpfs so imports that write still work.

    Plenty of correct code touches the filesystem at import — a SQLite file, a
    log directory. Importing straight from the read-only mount would report that
    as a failure, so the project is copied into the container's own tmpfs, which
    disappears with the container.
    """
    shutil.copytree(INPUT_DIR, PROJECT_DIR, dirs_exist_ok=True)


def _frame_in_project(error: BaseException) -> str:
    """Where inside the project the failure happened, if it happened there."""
    for frame in reversed(traceback.extract_tb(error.__traceback__)):
        try:
            relative = Path(frame.filename).relative_to(PROJECT_DIR)
        except ValueError:
            continue
        return f"{relative} line {frame.lineno}"
    return ""


def _import_check(name: str, budget: int) -> tuple[dict[str, Any], ModuleType | None]:
    """Import one module under a time limit and describe what happened."""
    check: dict[str, Any] = {"name": f"import {name}", "kind": "import", "ok": False}
    signal.alarm(budget)
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as error:
        check["detail"] = _bounded(error)
        check["missing_module"] = error.name or ""
        check["error_type"] = "ModuleNotFoundError"
        check["where"] = _frame_in_project(error)
        return check, None
    except ImportTimeout:
        check["detail"] = f"importing {name} did not finish within {budget}s"
        check["error_type"] = "ImportTimeout"
        return check, None
    except BaseException as error:  # a module may raise or exit at import time
        check["detail"] = _bounded("".join(traceback.format_exception_only(error)).strip())
        check["error_type"] = type(error).__name__
        check["where"] = _frame_in_project(error)
        return check, None
    finally:
        signal.alarm(0)
    check["ok"] = True
    check["detail"] = "imported cleanly"
    return check, module


def _find_application(module: ModuleType, attribute: str) -> Any:
    """The ASGI application a module exposes, if it exposes one."""
    candidate = getattr(module, attribute, None)
    if candidate is not None and hasattr(candidate, "routes"):
        return candidate
    for value in vars(module).values():
        if hasattr(value, "routes") and hasattr(value, "router"):
            return value
    return None


def _routes_of(application: Any) -> list[dict[str, str]]:
    """Every path the application declares, with the methods it accepts."""
    routes: list[dict[str, str]] = []
    for route in getattr(application, "routes", [])[:MAX_ROUTES]:
        path = str(getattr(route, "path", ""))
        if not path:
            continue
        methods = sorted(getattr(route, "methods", None) or [])
        routes.append({"path": path, "methods": ",".join(methods) or "MOUNT"})
    return routes


def _is_project_route(route: Any) -> bool:
    """Whether a route's handler is the project's code rather than the framework's.

    FastAPI installs /docs, /redoc and /openapi.json on every application. They
    always work, they are not what the build was asked to write, and requesting
    them would spend the budget that belongs to the project's own routes.
    """
    endpoint = getattr(route, "endpoint", None)
    filename = getattr(getattr(endpoint, "__code__", None), "co_filename", "")
    return str(filename).startswith(str(PROJECT_DIR))


def _request_checks(
    application: Any, routes: list[dict[str, str]], own_paths: set[str]
) -> list[dict[str, Any]]:
    """Call each parameterless GET the application declares and record the result."""
    try:
        from starlette.testclient import TestClient
    except Exception as error:  # no test client available; imports still stand
        return [
            {
                "name": "request routes",
                "kind": "request",
                "ok": True,
                "detail": f"skipped: {_bounded(error, 200)}",
            }
        ]
    targets = [
        route["path"]
        for route in routes
        if "GET" in route["methods"]
        and "{" not in route["path"]
        and route["path"] in own_paths
    ][:MAX_REQUESTS]
    if not targets:
        return []
    checks: list[dict[str, Any]] = []
    try:
        client = TestClient(application)
    except Exception as error:
        return [
            {
                "name": "start test client",
                "kind": "request",
                "ok": False,
                "detail": _bounded("".join(traceback.format_exception_only(error)).strip()),
                "error_type": type(error).__name__,
            }
        ]
    with client:
        for path in targets:
            check: dict[str, Any] = {"name": f"GET {path}", "kind": "request"}
            try:
                response = client.get(path)
            except Exception as error:
                check["ok"] = False
                check["error_type"] = type(error).__name__
                check["detail"] = _bounded(
                    "".join(traceback.format_exception_only(error)).strip()
                )
                check["where"] = _frame_in_project(error)
            else:
                check["ok"] = response.status_code < 500
                check["detail"] = f"HTTP {response.status_code}"
            checks.append(check)
    return checks


def _resolve_schema(schema: Any, components: dict[str, Any]) -> dict[str, Any]:
    """Follow $ref pointers into components.schemas, bounded against cycles."""
    hops = 0
    while isinstance(schema, dict) and "$ref" in schema and hops < 8:
        schema = components.get(str(schema["$ref"]).rsplit("/", 1)[-1], {})
        hops += 1
    return schema if isinstance(schema, dict) else {}


def _sample_value(schema: Any, components: dict[str, Any], depth: int = 0) -> Any:
    """The smallest instance that satisfies a schema's required shape.

    Just enough to get past request validation and into the handler — which
    is the code the check exists to run. Optional fields stay absent on
    purpose: a handler that crashes without them has a real None-handling bug.
    """
    if depth > 4:
        return "x"
    schema = _resolve_schema(schema, components)
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return enum[0]
    kind = schema.get("type")
    if kind == "object" or "properties" in schema:
        properties = schema.get("properties") or {}
        return {
            name: _sample_value(properties.get(name, {}), components, depth + 1)
            for name in schema.get("required") or []
        }
    if kind == "array":
        return []
    if kind == "integer":
        return 1
    if kind == "number":
        return 1.0
    if kind == "boolean":
        return True
    return "x"


def _body_checks(application: Any, own_paths: set[str]) -> list[dict[str, Any]]:
    """Exercise the POST bodies and parameterised GETs the project declares.

    Parameterless GETs alone let a broken upload route, a broken detail route
    and a broken approval transition all pass verification — measured on real
    builds. Each request here carries the smallest body the app's own OpenAPI
    schema describes. A validation rejection (4xx) proves the route is alive
    and validating and passes; only a 5xx or an unhandled exception fails.
    """
    try:
        from starlette.testclient import TestClient
    except Exception:
        return []
    try:
        spec = application.openapi()
    except AttributeError:
        return []  # not a FastAPI app; the parameterless GET checks still ran
    except Exception as error:
        # A FastAPI app whose schema will not build (unresolvable annotations,
        # usually). Say so rather than skipping silently — an invisible skip
        # reads as "covered" on the approval card.
        return [
            {
                "name": "openapi schema",
                "kind": "request",
                "ok": True,
                "detail": (
                    "body checks skipped: the app's OpenAPI schema would not "
                    f"build ({type(error).__name__})"
                ),
            }
        ]
    components = (spec.get("components") or {}).get("schemas") or {}
    targets: list[tuple[str, str, dict[str, Any]]] = []
    for path, operations in sorted((spec.get("paths") or {}).items()):
        if path not in own_paths or not isinstance(operations, dict):
            continue
        post = operations.get("post")
        if isinstance(post, dict):
            content = (post.get("requestBody") or {}).get("content") or {}
            multipart = content.get("multipart/form-data")
            body_json = content.get("application/json")
            if multipart:
                schema = _resolve_schema((multipart.get("schema") or {}), components)
                files: dict[str, Any] = {}
                data: dict[str, Any] = {}
                for name in schema.get("required") or []:
                    prop = _resolve_schema(
                        (schema.get("properties") or {}).get(name, {}), components
                    )
                    if prop.get("format") == "binary":
                        files[name] = ("fixture.png", PNG_FIXTURE, "image/png")
                    else:
                        data[name] = "x"
                if not files:  # every multipart route takes at least the file
                    files["file"] = ("fixture.png", PNG_FIXTURE, "image/png")
                targets.append(("POST", path, {"files": files, "data": data}))
            elif body_json is not None:
                sample = _sample_value((body_json.get("schema") or {}), components)
                targets.append(("POST", path, {"json": sample}))
            else:
                targets.append(("POST", path, {}))
        if "{" in path and isinstance(operations.get("get"), dict):
            targets.append(("GET", re.sub(r"\{[^}]+\}", "1", path), {}))
    if not targets:
        return []
    checks: list[dict[str, Any]] = []
    try:
        client = TestClient(application)
    except Exception as error:
        return [
            {
                "name": "start test client",
                "kind": "request",
                "ok": False,
                "detail": _bounded("".join(traceback.format_exception_only(error)).strip()),
                "error_type": type(error).__name__,
            }
        ]
    with client:
        for method, url, kwargs in targets[:MAX_BODY_REQUESTS]:
            check = {"name": f"{method} {url}", "kind": "request"}
            try:
                response = client.request(method, url, **kwargs)
            except Exception as error:
                check["ok"] = False
                check["error_type"] = type(error).__name__
                check["detail"] = _bounded(
                    "".join(traceback.format_exception_only(error)).strip()
                )
                check["where"] = _frame_in_project(error)
            else:
                check["ok"] = response.status_code < 500
                check["detail"] = f"HTTP {response.status_code}"
            checks.append(check)
    return checks


def verify(request: dict[str, Any]) -> dict[str, Any]:
    """Run every requested check and return the envelope describing them."""
    modules = [str(name) for name in request.get("modules", []) if str(name)]
    attribute = str(request.get("app_attribute") or "app")
    budget = int(request.get("import_timeout_seconds") or DEFAULT_IMPORT_SECONDS)

    _copy_project()
    sys.path.insert(0, str(PROJECT_DIR))
    # Run *as* the project, not merely with it importable. Ordinary correct code
    # resolves paths relative to the working directory — `StaticFiles(directory=
    # "app/static")` is the common case — and a verifier sitting in its own
    # directory reports every one of them as a missing file. That is a false
    # failure, which costs more than the check is worth.
    os.chdir(PROJECT_DIR)

    checks: list[dict[str, Any]] = []
    routes: list[dict[str, str]] = []
    captured = io.StringIO()
    # Project code that prints at import would otherwise land in the middle of
    # the envelope and make it unparseable, so its output is captured and
    # returned as evidence instead.
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        application = None
        for name in modules:
            check, module = _import_check(name, budget)
            checks.append(check)
            if module is not None and application is None:
                application = _find_application(module, attribute)
        if application is not None:
            routes = _routes_of(application)
            own_paths = {
                str(getattr(route, "path", ""))
                for route in getattr(application, "routes", [])
                if _is_project_route(route)
            }
            checks.append(
                {
                    "name": "application object",
                    "kind": "application",
                    "ok": True,
                    "detail": (
                        f"{type(application).__name__} declaring {len(routes)} route(s), "
                        f"{len(own_paths)} written by this project"
                    ),
                }
            )
            checks.extend(_request_checks(application, routes, own_paths))
            checks.extend(_body_checks(application, own_paths))
        elif modules:
            checks.append(
                {
                    "name": "application object",
                    "kind": "application",
                    "ok": True,
                    "detail": "no ASGI application found; import checks only",
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "succeeded",
        "checks": checks,
        "routes": routes,
        "captured_output": _bounded(captured.getvalue(), MAX_CAPTURED_CHARS),
    }


def _on_alarm(signum: int, frame: object) -> None:
    """Turn the import watchdog into an exception the import path can catch."""
    raise ImportTimeout()


def main() -> int:
    """Read the request, verify, and emit exactly one JSON envelope."""
    # Warnings are not findings. Leaving them on puts a framework deprecation
    # notice in front of the user on every single build.
    warnings.simplefilter("ignore")
    signal.signal(signal.SIGALRM, _on_alarm)
    try:
        request = _read_request()
    except (ValueError, UnicodeDecodeError) as error:
        print(json.dumps(_fail("INVALID_REQUEST", _bounded(error, 500))))
        return 12
    try:
        envelope = verify(request)
    except BaseException as error:  # the verifier's own failure is still data
        print(
            json.dumps(
                _fail(
                    "VERIFIER_FAILED",
                    _bounded("".join(traceback.format_exception_only(error)).strip(), 500),
                    {"type": type(error).__name__},
                )
            )
        )
        return 12
    print(json.dumps(envelope))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
