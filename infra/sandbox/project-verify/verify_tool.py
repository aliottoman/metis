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
import inspect
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import traceback
import warnings
from collections import Counter
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
MAX_SCENARIOS = 8
MAX_SCENARIO_UPLOAD_BYTES = 64 * 1024
DEFAULT_IMPORT_SECONDS = 20
SCENARIO_METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE"})

# Project tests run through a fixed verifier-owned Python invocation, never an
# argv supplied by the model or the repository.  These independent ceilings
# sit inside the container's 90-second/1-GiB/pid limits: ordinary focused suites
# get useful coverage, while a very large or stuck suite cannot monopolise the
# whole build loop.
MAX_TEST_FILES = 32
MAX_TEST_ITEMS = 200
MAX_TEST_FAILURES = 4
MAX_TEST_SECONDS = 45
MAX_TEST_RESULT_CHARS = 8_000
_MISSING_MODULE = re.compile(r"No module named ['\"]([^'\"]+)['\"]")

# A real 1×1 PNG, so a route that sniffs magic bytes or decodes its upload
# treats the fixture as a genuine image instead of failing on a fake.
PNG_FIXTURE = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)

# Representative enough to exercise a real local TXT extraction path while
# remaining deterministic, networkless and credentialless. Generic multipart
# checks use this fixture for liveness; a plan must explicitly request
# ``image_upload`` when image decoding or its external adapter is the claim.
TEXT_FIXTURE = b"""Invoice Number: INV-1042
Vendor: Nimbus Systems
Invoice Date: 2026-07-31
Currency: AED
Total: 431.75
Description: Evidence processing subscription
"""

# A harmless DOS header lookalike. It is never executed; it only proves that
# an upload boundary rejects unmistakably executable content before storage or
# downstream extraction. Keeping it deterministic makes the acceptance claim
# reproducible without credentials or network access.
EXECUTABLE_FIXTURE = b"MZ-not-an-invoice"


class ImportTimeout(Exception):
    """Raised by the alarm handler when one module takes too long to import."""


def _fail(
    code: str, message: str, details: dict[str, Any] | None = None
) -> dict[str, Any]:
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
    """Attribute a failure to app code, with framework code as a fallback.

    Tracebacks end at the deepest frame. That used to point every exception
    raised by the vendored ``appkit`` at framework-owned code the model cannot
    repair, even when an app-owned route or adapter failed to handle it. Prefer
    the deepest non-appkit project frame; retain the deepest appkit frame when
    no application frame exists so a genuine scaffold defect is not hidden.
    """
    fallback = ""
    for frame in reversed(traceback.extract_tb(error.__traceback__)):
        try:
            relative = Path(frame.filename).relative_to(PROJECT_DIR)
        except ValueError:
            continue
        where = f"{relative} line {frame.lineno}"
        if not fallback:
            fallback = where
        if not relative.parts or relative.parts[0].casefold() != "appkit":
            return where
    return fallback


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
        check["detail"] = _bounded(
            "".join(traceback.format_exception_only(error)).strip()
        )
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
    return bool(_endpoint_where(endpoint))


def _endpoint_where(endpoint: Any) -> str:
    """Return an endpoint's source location only when it belongs to the project.

    A response with the wrong status or content has no exception traceback. In
    that case the router is the authoritative link between the request and the
    app-owned handler that must repair it. Resolve symlinks before making the
    path relative so a synthetic or third-party endpoint cannot name a file
    outside the copied project.
    """
    try:
        endpoint = inspect.unwrap(endpoint)
    except (TypeError, ValueError):
        pass
    code = getattr(endpoint, "__code__", None)
    if code is None:
        call = getattr(endpoint, "__call__", None)
        code = getattr(call, "__code__", None)
    filename = str(getattr(code, "co_filename", ""))
    line = int(getattr(code, "co_firstlineno", 0) or 0)
    if not filename or line <= 0:
        return ""
    try:
        root = PROJECT_DIR.resolve()
        relative = Path(filename).resolve().relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return ""
    return f"{relative} line {line}"


def _matched_route_where(application: Any, method: str, path: str) -> str:
    """Locate the project endpoint Starlette would select for one request.

    Route ``matches`` is the same mechanism Starlette itself uses, so dynamic
    path parameters, method mismatches (405), and nested mounts resolve to the
    handler rather than the module that happened to create the application.
    """
    request_path = path.split("?", 1)[0] or "/"
    scope: dict[str, Any] = {
        "type": "http",
        "http_version": "1.1",
        "method": method.upper(),
        "scheme": "http",
        "path": request_path,
        "raw_path": request_path.encode("utf-8", errors="surrogateescape"),
        "root_path": "",
        "query_string": b"",
        "headers": [],
    }

    def locate(routes: Any, current_scope: dict[str, Any], seen: set[int]) -> str:
        partial = ""
        for route in list(routes or [])[:MAX_ROUTES]:
            identity = id(route)
            if identity in seen:
                continue
            matcher = getattr(route, "matches", None)
            if not callable(matcher):
                continue
            try:
                match, child_scope = matcher(current_scope)
            except Exception:
                continue
            match_name = str(getattr(match, "name", match)).upper()
            if match_name not in {"FULL", "PARTIAL"}:
                continue
            endpoint = (
                child_scope.get("endpoint") if isinstance(child_scope, dict) else None
            ) or getattr(route, "endpoint", None)
            where = _endpoint_where(endpoint)
            if match_name == "PARTIAL":
                partial = partial or where
                continue

            nested_routes = getattr(route, "routes", None)
            if nested_routes:
                nested_scope = dict(current_scope)
                if isinstance(child_scope, dict):
                    nested_scope.update(child_scope)
                nested = locate(nested_routes, nested_scope, {*seen, identity})
                if nested:
                    return nested
            return where
        return partial

    return locate(getattr(application, "routes", []), scope, set())


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
                "detail": _bounded(
                    "".join(traceback.format_exception_only(error)).strip()
                ),
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
                if not check["ok"]:
                    where = _matched_route_where(application, "GET", path)
                    if where:
                        check["where"] = where
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
                        files[name] = ("fixture.txt", TEXT_FIXTURE, "text/plain")
                    else:
                        data[name] = "x"
                if not files:  # every multipart route takes at least the file
                    files["file"] = ("fixture.txt", TEXT_FIXTURE, "text/plain")
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
                "detail": _bounded(
                    "".join(traceback.format_exception_only(error)).strip()
                ),
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
                if not check["ok"]:
                    where = _matched_route_where(application, method, url)
                    if where:
                        check["where"] = where
            checks.append(check)
    return checks


MAX_JSON_DIFFS = 6
MAX_JSON_VALUE_CHARS = 200


def _render(value: Any) -> str:
    """One bounded, copy-pasteable JSON literal for a diff line."""
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        text = repr(value)
    return _bounded(text, MAX_JSON_VALUE_CHARS)


def _multiset_key(value: Any) -> str:
    """A stable identity for an element compared without regard to order."""
    return _render(value)


def _json_differences(
    expected: Any, actual: Any, *, unordered: bool, where: str = "$"
) -> list[str]:
    """Every structural difference, named by JSON path, expected before actual.

    A repair model can only correct the record it got wrong if the finding
    says which record, what was expected, and what was actually there. Two
    live models responded to a bare "the listing never mentions X" by adding
    a new record beside the wrong one, which satisfied the containment check
    and left the defect in place. These lines are written to make that
    workaround obviously not the fix.
    """
    if isinstance(expected, dict) and isinstance(actual, dict):
        differences: list[str] = []
        # Key ORDER never matters; key SET does. A response carrying extra
        # keys is not the exact structure that was specified.
        for key in sorted(set(expected) | set(actual)):
            child = f"{where}.{key}"
            if key not in actual:
                differences.append(
                    f"{child}: missing, expected {_render(expected[key])}"
                )
            elif key not in expected:
                differences.append(
                    f"{child}: unexpected key with value {_render(actual[key])}"
                )
            else:
                differences.extend(
                    _json_differences(
                        expected[key], actual[key], unordered=unordered, where=child
                    )
                )
            if len(differences) >= MAX_JSON_DIFFS:
                break
        return differences[:MAX_JSON_DIFFS]
    if isinstance(expected, list) and isinstance(actual, list):
        if unordered:
            # Identical members AND multiplicity, so a duplicate or an extra
            # appended record still fails.
            expected_counts = Counter(_multiset_key(item) for item in expected)
            actual_counts = Counter(_multiset_key(item) for item in actual)
            if expected_counts == actual_counts:
                return []
            differences = []
            for key, count in sorted((expected_counts - actual_counts).items()):
                differences.append(f"{where}: missing {count}x element {key}")
            for key, count in sorted((actual_counts - expected_counts).items()):
                differences.append(f"{where}: unexpected {count}x element {key}")
            return differences[:MAX_JSON_DIFFS]
        if len(expected) != len(actual):
            differences = [
                f"{where}: expected {len(expected)} element(s), found {len(actual)}"
            ]
            for index in range(min(len(expected), len(actual))):
                differences.extend(
                    _json_differences(
                        expected[index],
                        actual[index],
                        unordered=unordered,
                        where=f"{where}[{index}]",
                    )
                )
                if len(differences) >= MAX_JSON_DIFFS:
                    break
            for index in range(len(actual), len(expected)):
                differences.append(
                    f"{where}[{index}]: missing, expected {_render(expected[index])}"
                )
            for index in range(len(expected), len(actual)):
                differences.append(
                    f"{where}[{index}]: unexpected element {_render(actual[index])}"
                )
            return differences[:MAX_JSON_DIFFS]
        differences = []
        for index, (want, got) in enumerate(zip(expected, actual)):
            differences.extend(
                _json_differences(
                    want, got, unordered=unordered, where=f"{where}[{index}]"
                )
            )
            if len(differences) >= MAX_JSON_DIFFS:
                break
        return differences[:MAX_JSON_DIFFS]
    if type(expected) is not type(actual) and not (
        isinstance(expected, (int, float))
        and isinstance(actual, (int, float))
        and not isinstance(expected, bool)
        and not isinstance(actual, bool)
    ):
        return [
            f"{where}: expected {_render(expected)} "
            f"({type(expected).__name__}), found {_render(actual)} "
            f"({type(actual).__name__})"
        ]
    if expected != actual:
        # Values compare exactly, case included: "harbor migration" is not
        # "Harbor Migration" when the request named the latter.
        return [f"{where}: expected {_render(expected)}, found {_render(actual)}"]
    return []


def _scenario_checks(application: Any, scenarios: list[Any]) -> list[dict[str, Any]]:
    """Replay the plan's acceptance scenarios against the app, exactly as written.

    The structural checks above prove the app parses, imports and serves;
    real builds have passed all of that while being non-functional demoware.
    A scenario is the specification's own claim made executable — the upload
    route accepts a deterministic TXT or real image fixture, the assessment
    response names a verdict — so a failure here is the first rung that can
    say "it runs, but it does not do what was asked". Status failures carry
    the same fields as the other request checks; a content miss is marked so
    the host can keep it advisory (the scenario, not the app, may be the wrong
    party).
    """
    if not scenarios:
        return []
    try:
        from starlette.testclient import TestClient
    except Exception:
        return []
    checks: list[dict[str, Any]] = []
    try:
        client = TestClient(application)
    except Exception as error:
        return [
            {
                "name": "start test client",
                "kind": "acceptance",
                "ok": False,
                "detail": _bounded(
                    "".join(traceback.format_exception_only(error)).strip()
                ),
                "error_type": type(error).__name__,
            }
        ]
    with client:
        for raw in scenarios[:MAX_SCENARIOS]:
            if not isinstance(raw, dict):
                continue
            name = _bounded(raw.get("name") or "scenario", 120)
            method = str(raw.get("method", "GET")).upper()
            path = str(raw.get("path") or "/")
            if not path.startswith("/"):
                path = f"/{path}"
            check: dict[str, Any] = {
                "name": f"acceptance: {name}",
                "kind": "acceptance",
            }
            if method not in SCENARIO_METHODS:
                # Raw verifier input normally crossed AcceptanceScenarioV1,
                # but the sandbox is a separate trust boundary. Unknown verbs
                # must fail closed instead of silently becoming GET.
                check["ok"] = False
                check["detail"] = f"unsupported HTTP method {method!r}"
                checks.append(check)
                continue
            body_kind = str(raw.get("body_kind") or "none")
            kwargs: dict[str, Any] = {}
            if method != "GET" and body_kind == "json":
                kwargs["json"] = (
                    raw.get("body") if isinstance(raw.get("body"), dict) else {}
                )
            elif method != "GET" and body_kind in {
                "text_upload",
                "executable_upload",
                "image_upload",
            }:
                data = raw.get("body")
                form = dict(data) if isinstance(data, dict) else {}
                if body_kind == "text_upload":
                    # Backward compatibility for plans produced before the
                    # explicit executable fixture existed. Those plans used
                    # body.file for a small literal UTF-8 upload. Previously we
                    # sent the deterministic invoice *and* a duplicate text
                    # form field named file, so the route received the invoice
                    # and a refusal scenario falsely failed with HTTP 200.
                    literal = form.pop("file", None)
                    if isinstance(literal, str):
                        content = literal.encode("utf-8")
                        if len(content) > MAX_SCENARIO_UPLOAD_BYTES:
                            check["ok"] = False
                            check["detail"] = (
                                "literal text upload exceeds verifier limit"
                            )
                            checks.append(check)
                            continue
                    else:
                        content = TEXT_FIXTURE
                    fixture = ("fixture.txt", content, "text/plain")
                elif body_kind == "executable_upload":
                    fixture = (
                        "fixture.exe",
                        EXECUTABLE_FIXTURE,
                        "application/octet-stream",
                    )
                else:
                    fixture = ("fixture.png", PNG_FIXTURE, "image/png")
                kwargs["files"] = {"file": fixture}
                if form:
                    kwargs["data"] = {key: str(value) for key, value in form.items()}
            try:
                response = client.request(method, path, **kwargs)
            except Exception as error:
                check["ok"] = False
                check["error_type"] = type(error).__name__
                check["detail"] = _bounded(
                    "".join(traceback.format_exception_only(error)).strip()
                )
                check["where"] = _frame_in_project(error)
                checks.append(check)
                continue
            status = response.status_code
            expected = str(raw.get("expect_status") or "2xx_or_4xx")
            status_ok = {
                "2xx": 200 <= status < 300,
                "4xx": 400 <= status < 500,
            }.get(expected, status < 500)
            if not status_ok:
                check["ok"] = False
                check["detail"] = (
                    f"{method} {path} returned HTTP {status}, expected {expected}"
                )
                where = _matched_route_where(application, method, path)
                if where:
                    check["where"] = where
                checks.append(check)
                continue
            try:
                text = response.text.casefold()
            except Exception:
                text = ""
            missing = [
                str(needle)
                for needle in (raw.get("expect_contains") or [])[:8]
                if str(needle).casefold() not in text
            ]
            if missing:
                check["ok"] = False
                check["content_miss"] = True
                check["detail"] = (
                    f"{method} {path} answered HTTP {status} but the response never "
                    "mentions: " + ", ".join(_bounded(item, 80) for item in missing)
                )
                where = _matched_route_where(application, method, path)
                if where:
                    check["where"] = where
                checks.append(check)
                continue
            expected_json = raw.get("expect_json_exact")
            if expected_json is not None:
                # Deliberately NOT marked content_miss: a containment miss can
                # be the scenario's fault and stays advisory, but an exact
                # structural claim that fails is the app being wrong, and the
                # host must treat it as blocking.
                check["ok"] = True
                try:
                    actual_json = response.json()
                    parsed = True
                except Exception:
                    parsed = False
                if not parsed:
                    check["ok"] = False
                    check["exact_miss"] = True
                    check["detail"] = (
                        f"{method} {path} answered HTTP {status} with a body that is "
                        "not JSON, but the scenario asserts an exact JSON structure"
                    )
                else:
                    unordered = (
                        str(raw.get("json_match") or "exact") == "unordered_array"
                    )
                    differences = _json_differences(
                        expected_json, actual_json, unordered=unordered
                    )
                    if differences:
                        check["ok"] = False
                        check["exact_miss"] = True
                        check["detail"] = _bounded(
                            f"{method} {path} answered HTTP {status} but its JSON does "
                            "not match the required structure ("
                            + ("unordered array" if unordered else "exact")
                            + "): "
                            + "; ".join(differences)
                            + f". Expected: {_render(expected_json)}. "
                            f"Actual: {_render(actual_json)}. Correct the existing "
                            "value(s); adding another record does not satisfy this."
                        )
                if not check["ok"]:
                    where = _matched_route_where(application, method, path)
                    if where:
                        check["where"] = where
                    checks.append(check)
                    continue
            check["ok"] = True
            check["detail"] = f"HTTP {status}, response matches the scenario"
            checks.append(check)
    return checks


def _is_python_test_path(path: str) -> bool:
    """Whether a safe project-relative path follows a pytest file convention."""
    candidate = Path(path)
    name = candidate.name.casefold()
    return bool(
        candidate.suffix.casefold() == ".py"
        and (name.startswith("test_") or name.endswith("_test.py"))
    )


def _safe_project_relative(value: object) -> str:
    """A normalized project path, or empty when input could escape the copy."""
    raw = str(value or "").replace("\\", "/")
    candidate = Path(raw)
    if (
        not raw
        or "\x00" in raw
        or candidate.is_absolute()
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        return ""
    try:
        resolved = (PROJECT_DIR / candidate).resolve(strict=False)
        resolved.relative_to(PROJECT_DIR.resolve())
    except (OSError, RuntimeError, ValueError):
        return ""
    return candidate.as_posix()


def _python_test_targets(changed_paths: list[object]) -> list[str]:
    """Select a deterministic, bounded pytest slice for this Python change.

    Small repositories run their whole Python suite.  Large ones prioritise a
    staged test first, then tests whose filename names a staged source module,
    then a stable lexical prefix.  Selection is derived from the materialized
    project and the host's changed-path list; neither a model nor project config
    can turn a test target into a command-line option.
    """
    changed = [path for item in changed_paths if (path := _safe_project_relative(item))]
    changed_python = [path for path in changed if Path(path).suffix.casefold() == ".py"]
    if not changed_python:
        return []
    changed_tests = {path for path in changed_python if _is_python_test_path(path)}
    source_stems = {
        Path(path).stem.casefold()
        for path in changed_python
        if not _is_python_test_path(path) and Path(path).stem != "__init__"
    }
    candidates: list[str] = []
    for candidate in PROJECT_DIR.rglob("*.py"):
        try:
            relative = candidate.relative_to(PROJECT_DIR).as_posix()
        except ValueError:
            continue
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or not _is_python_test_path(relative)
        ):
            continue
        candidates.append(relative)

    def priority(path: str) -> tuple[int, str]:
        if path in changed_tests:
            return 0, path
        stem = Path(path).stem.casefold()
        normalized = stem[5:] if stem.startswith("test_") else stem[:-5]
        if normalized in source_stems:
            return 1, path
        return 2, path

    return sorted(set(candidates), key=priority)[:MAX_TEST_FILES]


def _bounded_test_result(value: object) -> str:
    """Keep both the first failure and pytest's final summary under one bound."""
    text = str(value).replace("\x00", "")
    if len(text) <= MAX_TEST_RESULT_CHARS:
        return text
    head = (MAX_TEST_RESULT_CHARS * 3) // 4
    tail = MAX_TEST_RESULT_CHARS - head
    return f"{text[:head]}\n… pytest output truncated …\n{text[-tail:]}"


def _relative_report_path(value: object) -> tuple[str, int]:
    """Map a pytest traceback location back into the copied project."""
    raw = str(value or "")
    line = 0
    if ":" in raw:
        possible_path, _, possible_line = raw.rpartition(":")
        if possible_line.isdigit():
            raw = possible_path
            line = int(possible_line)
    candidate = Path(raw)
    try:
        if not candidate.is_absolute():
            candidate = PROJECT_DIR / candidate
        relative = candidate.resolve(strict=False).relative_to(PROJECT_DIR.resolve())
    except (OSError, RuntimeError, ValueError):
        return "", 0
    return relative.as_posix(), line


def _pytest_report_where(report: Any, changed_paths: list[str]) -> str:
    """Attribute a failure to an exact staged path the repair loop may edit."""
    changed = [path for path in changed_paths if _safe_project_relative(path)]
    changed_production = [
        path
        for path in changed
        if Path(path).suffix.casefold() == ".py"
        and not _is_python_test_path(path)
        and Path(path).parts[:1] != ("appkit",)
    ]
    reported: list[tuple[str, int]] = []
    longrepr = getattr(report, "longrepr", None)
    traceback_repr = getattr(longrepr, "reprtraceback", None)
    for entry in getattr(traceback_repr, "reprentries", []) or []:
        fileloc = getattr(entry, "reprfileloc", None)
        path, line = _relative_report_path(getattr(fileloc, "path", ""))
        if path:
            try:
                line = int(getattr(fileloc, "lineno", line) or line)
            except (TypeError, ValueError):
                pass
            reported.append((path, line))
    location = getattr(report, "location", None)
    if isinstance(location, tuple) and location:
        path, _ = _relative_report_path(location[0])
        try:
            line = int(location[1]) + 1 if len(location) > 1 else 0
        except (TypeError, ValueError):
            line = 0
        if path:
            reported.append((path, line))

    # A traceback through a staged production file is the strongest repair
    # target.  A failure observed only in the test still maps to a staged source
    # path so the model is not encouraged to weaken a regression assertion.
    for path, line in reversed(reported):
        if path in changed_production:
            return f"{path} line {line}" if line > 0 else path
    if changed_production:
        return changed_production[0]
    for path, line in reversed(reported):
        if path in changed:
            return f"{path} line {line}" if line > 0 else path
    return changed[0] if changed else ""


class _PytestRecorder:
    """Small in-process plugin used only by the isolated pytest child."""

    def __init__(self, changed_paths: list[str]) -> None:
        self.changed_paths = changed_paths
        self.selected = 0
        self.deselected = 0
        self.passed = 0
        self.skipped = 0
        self.failures: list[dict[str, Any]] = []
        self._failed_nodes: set[tuple[str, str]] = set()

    def pytest_collection_modifyitems(
        self, session: Any, config: Any, items: list[Any]
    ) -> None:
        self.selected = min(len(items), MAX_TEST_ITEMS)
        if len(items) <= MAX_TEST_ITEMS:
            return
        deselected = items[MAX_TEST_ITEMS:]
        del items[MAX_TEST_ITEMS:]
        self.deselected = len(deselected)
        config.hook.pytest_deselected(items=deselected)

    def pytest_collectreport(self, report: Any) -> None:
        if getattr(report, "failed", False):
            self._record(report, phase="collection")

    def pytest_runtest_logreport(self, report: Any) -> None:
        phase = str(getattr(report, "when", "call"))
        if getattr(report, "failed", False):
            self._record(report, phase=phase)
        elif phase == "call" and getattr(report, "passed", False):
            self.passed += 1
        elif getattr(report, "skipped", False):
            self.skipped += 1

    def _record(self, report: Any, *, phase: str) -> None:
        nodeid = str(getattr(report, "nodeid", "pytest"))
        key = (nodeid, phase)
        if key in self._failed_nodes or len(self.failures) >= MAX_TEST_FAILURES:
            return
        self._failed_nodes.add(key)
        detail = _bounded_test_result(
            getattr(report, "longreprtext", "") or report.longrepr
        )
        missing = _MISSING_MODULE.search(detail)
        if not missing and "async def functions are not natively supported" in detail:
            missing_name = "pytest_asyncio"
        else:
            missing_name = missing.group(1) if missing else ""
        self.failures.append(
            {
                "name": f"pytest {nodeid}",
                "kind": "test",
                "ok": False,
                "detail": detail,
                "where": _pytest_report_where(report, self.changed_paths),
                **({"missing_module": missing_name} if missing_name else {}),
            }
        )


def _pytest_child(request: dict[str, Any]) -> dict[str, Any]:
    """Run fixed-target pytest in a fresh interpreter and return only evidence."""
    try:
        import pytest
    except Exception as error:
        return {
            "status": "unavailable",
            "detail": f"pytest is unavailable in the verify image: {type(error).__name__}",
        }
    targets = [
        path
        for item in list(request.get("targets") or [])[:MAX_TEST_FILES]
        if (path := _safe_project_relative(item)) and _is_python_test_path(path)
    ]
    changed = [
        path
        for item in list(request.get("changed_paths") or [])
        if (path := _safe_project_relative(item))
    ]
    if not targets:
        return {"status": "unavailable", "detail": "no bounded Python tests selected"}
    recorder = _PytestRecorder(changed)
    os.environ.pop("PYTEST_ADDOPTS", None)
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.chdir(PROJECT_DIR)
    sys.path.insert(0, str(PROJECT_DIR))
    output = io.StringIO()
    arguments = [
        "-q",
        "--disable-warnings",
        "--tb=short",
        f"--maxfail={MAX_TEST_FAILURES}",
        "--override-ini=addopts=",
        "-p",
        "no:cacheprovider",
        *(str(PROJECT_DIR / target) for target in targets),
    ]
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        exit_code = int(pytest.main(arguments, plugins=[recorder]))
    return {
        "status": "completed",
        "exit_code": exit_code,
        "selected": recorder.selected,
        "deselected": recorder.deselected,
        "passed": recorder.passed,
        "skipped": recorder.skipped,
        "failures": recorder.failures,
        "output": _bounded_test_result(output.getvalue()),
    }


def _run_test_child(
    targets: list[str], changed_paths: list[str]
) -> list[dict[str, Any]]:
    """Supervise the fixed pytest child without exposing a command surface."""
    request = json.dumps(
        {"targets": targets, "changed_paths": changed_paths},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    command = [sys.executable, str(Path(__file__).resolve()), "--pytest-child"]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=PROJECT_DIR,
        env={
            "HOME": "/tmp",
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
            "PATH": os.environ.get("PATH", ""),
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        },
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(request, timeout=MAX_TEST_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            process.kill()
        process.communicate()
        repair = next(
            (
                path
                for path in changed_paths
                if Path(path).suffix.casefold() == ".py"
                and not _is_python_test_path(path)
            ),
            changed_paths[0] if changed_paths else "",
        )
        return [
            {
                "name": "pytest selected project tests",
                "kind": "test",
                "ok": False,
                "detail": f"did not finish within {MAX_TEST_SECONDS} seconds",
                "where": repair,
                "error_type": "TestTimeout",
            }
        ]
    try:
        result = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return [
            {
                "name": "project tests",
                "kind": "test_unavailable",
                "ok": True,
                "detail": (
                    "pytest child returned no structured result: "
                    + _bounded_test_result(stderr.decode("utf-8", errors="replace"))
                ),
            }
        ]
    if result.get("status") != "completed":
        return [
            {
                "name": "project tests",
                "kind": "test_unavailable",
                "ok": True,
                "detail": _bounded_test_result(
                    result.get("detail") or "pytest unavailable"
                ),
            }
        ]
    failures = list(result.get("failures") or [])[:MAX_TEST_FAILURES]
    if failures:
        return failures
    exit_code = int(result.get("exit_code") or 0)
    if exit_code in {3, 4}:
        # Pytest's own internal/usage failure is verifier evidence, not evidence
        # that application behavior regressed. Keep it visible without turning
        # an unavailable plugin or incompatible project config into model work.
        return [
            {
                "name": "project tests",
                "kind": "test_unavailable",
                "ok": True,
                "detail": _bounded_test_result(
                    result.get("output") or f"pytest could not run (status {exit_code})"
                ),
            }
        ]
    if exit_code not in {0, 5}:
        repair = next(
            (
                path
                for path in changed_paths
                if Path(path).suffix.casefold() == ".py"
                and not _is_python_test_path(path)
            ),
            changed_paths[0] if changed_paths else "",
        )
        return [
            {
                "name": "pytest selected project tests",
                "kind": "test",
                "ok": False,
                "detail": _bounded_test_result(
                    result.get("output") or f"pytest exited with status {exit_code}"
                ),
                "where": repair,
                "error_type": "PytestInterrupted",
            }
        ]
    selected = int(result.get("selected") or 0)
    passed = int(result.get("passed") or 0)
    skipped = int(result.get("skipped") or 0)
    deselected = int(result.get("deselected") or 0)
    detail = f"{passed} passed, {skipped} skipped from {selected} selected test(s)"
    if deselected:
        detail += f"; {deselected} additional test(s) left outside the bounded slice"
    if exit_code == 5:
        detail = "selected test files contained no collectable tests"
    return [
        {
            "name": "pytest selected project tests",
            "kind": "test",
            "ok": True,
            "detail": detail,
        }
    ]


def _project_test_checks(request: dict[str, Any]) -> list[dict[str, Any]]:
    """Execute relevant tests only when this overlay changes Python code."""
    changed = [
        path
        for item in list(request.get("changed_paths") or [])
        if (path := _safe_project_relative(item))
    ]
    targets = _python_test_targets(changed)
    return _run_test_child(targets, changed) if targets else []


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
            checks.extend(
                _scenario_checks(application, list(request.get("scenarios") or []))
            )
        elif modules:
            checks.append(
                {
                    "name": "application object",
                    "kind": "application",
                    "ok": True,
                    "detail": "no ASGI application found; import checks only",
                }
            )
        # Tests run last and in a fresh interpreter.  Import/route probes can
        # legitimately mutate application state; sharing that process with the
        # test suite would make order-dependent failures that do not exist for a
        # normal pytest invocation.
        checks.extend(_project_test_checks(request))
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
                    _bounded(
                        "".join(traceback.format_exception_only(error)).strip(), 500
                    ),
                    {"type": type(error).__name__},
                )
            )
        )
        return 12
    print(json.dumps(envelope))
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--pytest-child"]:
        try:
            child_request = _read_request()
            child_result = _pytest_child(child_request)
        except BaseException as error:
            child_result = {
                "status": "unavailable",
                "detail": _bounded(
                    "".join(traceback.format_exception_only(error)).strip(), 500
                ),
            }
        print(json.dumps(child_result))
        raise SystemExit(0)
    raise SystemExit(main())
