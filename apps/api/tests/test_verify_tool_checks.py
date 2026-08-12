"""P0.4: verification that exercises what the app actually does.

Three layers under test, one per section: the in-container verifier's new
POST/multipart/parameterised checks (loaded from the real tool file, run
in-process against a real FastAPI app), the host's classification of
config-at-import failures, and the wiring gate's implicit python-multipart
rule. Together they close the gap the diagnostic measured: an app whose
entire upload workflow was broken passed verification on GETs alone.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from fastapi import FastAPI, File, Form, Response, UploadFile
from pydantic import BaseModel

from waqil_api.contracts import AcceptanceScenarioV1
from waqil_api.control_plane import _blocking_reason, _blocks_approval
from waqil_api.project_sandbox import classify_envelope
from waqil_api.project_wiring import staged_wiring_errors

_REPO = Path(__file__).resolve().parents[3]


def _verify_tool():
    spec = importlib.util.spec_from_file_location(
        "verify_tool_under_test",
        _REPO / "infra" / "sandbox" / "project-verify" / "verify_tool.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Module level on purpose: this test file uses postponed annotations, so
# FastAPI resolves route annotations against module globals — a model class
# defined inside a function is unresolvable there and openapi() cannot build
# (in which case _body_checks reports the skip; see the dedicated test).
class PurchaseOrder(BaseModel):
    number: str
    total: float


class Strict(BaseModel):
    count: int


def _application() -> FastAPI:
    app = FastAPI()

    @app.post("/invoices")
    async def upload(file: UploadFile = File(...)) -> dict[str, int]:
        return {"received": len(await file.read())}

    @app.post("/purchase-orders")
    async def create(order: PurchaseOrder) -> PurchaseOrder:
        return order

    @app.get("/invoices/{invoice_id}")
    async def detail(invoice_id: str) -> dict[str, str]:
        return {"id": invoice_id}

    @app.get("/totals/{x}")
    async def broken(x: str) -> dict[str, object]:
        return {"n": None + 1}  # type: ignore[operator] - the defect under test

    return app


# --- in-container body checks ----------------------------------------------


def test_body_checks_exercise_multipart_json_and_parameterised_routes() -> None:
    tool = _verify_tool()
    app = _application()
    own = {"/invoices", "/purchase-orders", "/invoices/{invoice_id}", "/totals/{x}"}
    checks = {check["name"]: check for check in tool._body_checks(app, own)}

    multipart = checks["POST /invoices"]
    assert multipart["ok"] is True, multipart
    assert multipart["detail"] == "HTTP 200"  # the fixture reached the handler

    body = checks["POST /purchase-orders"]
    assert body["ok"] is True, body
    assert body["detail"] == "HTTP 200"  # synthesized {number, total} validated

    detail = checks["GET /invoices/1"]
    assert detail["ok"] is True

    broken = checks["GET /totals/1"]
    assert broken["ok"] is False
    assert broken["error_type"] == "TypeError"


def test_generic_multipart_check_uses_credentialless_text_fixture() -> None:
    """The generic liveness rung must not choose an OCI-backed image branch.

    Image behavior remains covered by an explicit ``image_upload`` acceptance
    scenario; this broad probe checks that the upload route is alive using the
    project's required no-credentials TXT path.
    """
    tool = _verify_tool()
    observed: dict[str, object] = {}
    app = FastAPI()

    @app.post("/documents")
    async def upload(file: UploadFile = File(...)) -> dict[str, str]:
        observed.update(
            filename=file.filename,
            content_type=file.content_type,
            content=await file.read(),
        )
        return {"status": "needs_review"}

    (check,) = tool._body_checks(app, {"/documents"})

    assert check["ok"] is True, check
    assert observed["filename"] == "fixture.txt"
    assert observed["content_type"] == "text/plain"
    assert b"INV-1042" in observed["content"]


def test_body_checks_treat_validation_rejection_as_alive() -> None:
    tool = _verify_tool()
    app = FastAPI()

    @app.post("/strict")
    async def strict(payload: Strict, header_token: str) -> Strict:  # extra query
        return payload

    checks = tool._body_checks(app, {"/strict"})
    (check,) = checks
    # The missing query parameter makes this a 422 — the route is alive and
    # validating, which is a pass; only 5xx/exception is a failure.
    assert check["ok"] is True
    assert check["detail"] == "HTTP 422"


# --- repository tests against the staged sandbox copy -----------------------


def test_python_test_selection_is_relevant_deterministic_and_bounded(
    tmp_path: Path,
) -> None:
    tool = _verify_tool()
    tool.PROJECT_DIR = tmp_path
    for relative in (
        "tests/test_unrelated.py",
        "tests/test_totals.py",
        "tests/test_changed_directly.py",
        "app/not_a_test.py",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def test_placeholder():\n    assert True\n", encoding="utf-8")
    tool.MAX_TEST_FILES = 2

    selected = tool._python_test_targets(
        ["app/totals.py", "tests/test_changed_directly.py"]
    )

    assert selected == ["tests/test_changed_directly.py", "tests/test_totals.py"]
    assert tool._python_test_targets(["README.md"]) == []
    assert tool._python_test_targets(["../outside.py"]) == []


def test_a_pytest_regression_is_blocking_repair_evidence() -> None:
    outcome = classify_envelope(
        {
            "status": "succeeded",
            "checks": [
                {
                    "name": "pytest tests/test_totals.py::test_rounding",
                    "kind": "test",
                    "ok": False,
                    "detail": "assert 12.4 == 12.5",
                    "where": "app/totals.py line 18",
                }
            ],
            "routes": [],
        },
        staged={"app/totals.py": {"content": ""}},
        project_paths=["tests/test_totals.py"],
    )

    (finding,) = outcome.findings
    assert finding["path"] == "app/totals.py"
    assert finding["kind"] == "test"
    assert finding["severity"] == "error"
    assert "staged project" in finding["error"]
    assert _blocks_approval({**finding, "rung": "runtime"})


def test_a_test_dependency_missing_from_the_image_is_advisory() -> None:
    outcome = classify_envelope(
        {
            "status": "succeeded",
            "checks": [
                {
                    "name": "pytest tests/test_frames.py",
                    "kind": "test",
                    "ok": False,
                    "detail": "ModuleNotFoundError: No module named 'polars'",
                    "missing_module": "polars",
                    "where": "app/frames.py",
                }
            ],
            "routes": [],
        },
        staged={"app/frames.py": {"content": ""}},
        project_paths=["tests/test_frames.py"],
        requirements="polars>=1.0\n",
    )

    (finding,) = outcome.findings
    assert finding["severity"] == "warning"
    assert not _blocks_approval({**finding, "rung": "runtime"})


def test_a_test_that_needs_network_or_credentials_is_advisory() -> None:
    for detail in (
        "httpx.ConnectError: [Errno 101] Network is unreachable",
        "RuntimeError: OCI_CONFIG_FILE is not set",
    ):
        outcome = classify_envelope(
            {
                "status": "succeeded",
                "checks": [
                    {
                        "name": "pytest tests/test_live_api.py::test_live_call",
                        "kind": "test",
                        "ok": False,
                        "detail": detail,
                        "where": "app/client.py",
                    }
                ],
                "routes": [],
            },
            staged={"app/client.py": {"content": ""}},
            project_paths=["tests/test_live_api.py"],
        )

        (finding,) = outcome.findings
        assert finding["kind"] == "test"
        assert finding["severity"] == "warning"
        assert not _blocks_approval({**finding, "rung": "runtime"})


def test_body_checks_skip_foreign_apps_and_announce_broken_schemas() -> None:
    tool = _verify_tool()
    app = _application()
    assert tool._body_checks(app, set()) == []
    assert tool._body_checks(object(), {"/x"}) == []

    unbuildable = FastAPI()

    class Local(BaseModel):  # function-scoped: openapi() cannot resolve it
        value: str

    @unbuildable.post("/x")
    async def handler(payload: Local) -> Local:
        return payload

    checks = tool._body_checks(unbuildable, {"/x"})
    (check,) = checks
    assert check["ok"] is True
    assert "body checks skipped" in check["detail"]


def test_response_failures_point_to_the_matched_route_source(tmp_path: Path) -> None:
    """A returned 5xx/wrong answer has no traceback, so the routing table must
    identify the app-owned handler rather than falling back to app/main.py."""
    tool = _verify_tool()
    tool.PROJECT_DIR = tmp_path
    route_file = tmp_path / "app" / "routes" / "documents.py"
    route_file.parent.mkdir(parents=True)
    source = (
        "async def unavailable():\n"
        "    return Response(status_code=503)\n"
        "\n"
        "async def detail(document_id: str):\n"
        "    return {'id': document_id, 'state': 'pending'}\n"
    )
    route_file.write_text(source, encoding="utf-8")
    namespace: dict[str, object] = {"Response": Response}
    exec(compile(source, str(route_file), "exec"), namespace)

    app = FastAPI()
    app.add_api_route(
        "/api/documents",
        namespace["unavailable"],
        methods=["GET"],  # type: ignore[arg-type]
    )
    app.add_api_route(
        "/api/documents/{document_id}",
        namespace["detail"],  # type: ignore[arg-type]
        methods=["GET"],
    )
    app.add_api_route(
        "/api/uploads",
        namespace["unavailable"],
        methods=["POST"],  # type: ignore[arg-type]
    )
    own_paths = {
        "/api/documents",
        "/api/documents/{document_id}",
        "/api/uploads",
    }

    request_failure = tool._request_checks(app, tool._routes_of(app), own_paths)[0]
    assert request_failure["ok"] is False
    assert request_failure["where"] == "app/routes/documents.py line 1"

    body_failure = {
        check["name"]: check for check in tool._body_checks(app, own_paths)
    }["POST /api/uploads"]
    assert body_failure["ok"] is False
    assert body_failure["where"] == "app/routes/documents.py line 1"

    scenarios = tool._scenario_checks(
        app,
        [
            {
                "name": "documents route is available",
                "method": "GET",
                "path": "/api/documents",
                "expect_status": "2xx",
            },
            {
                "name": "approved state is returned",
                "method": "GET",
                "path": "/api/documents/doc-1",
                "expect_contains": ["approved"],
            },
        ],
    )
    assert scenarios[0]["where"] == "app/routes/documents.py line 1"
    assert scenarios[1]["where"] == "app/routes/documents.py line 4"


def test_route_source_attribution_handles_mounts_and_rejects_outside_files(
    tmp_path: Path,
) -> None:
    tool = _verify_tool()
    tool.PROJECT_DIR = tmp_path
    route_file = tmp_path / "app" / "routes" / "documents.py"
    route_file.parent.mkdir(parents=True)
    route_file.write_text(
        "async def detail(document_id: str):\n    return {'id': document_id}\n",
        encoding="utf-8",
    )
    namespace: dict[str, object] = {}
    exec(compile(route_file.read_text(), str(route_file), "exec"), namespace)
    nested = FastAPI()
    nested.add_api_route(
        "/documents/{document_id}",
        namespace["detail"],  # type: ignore[arg-type]
        methods=["GET"],
    )
    app = FastAPI()
    app.mount("/api", nested)

    assert (
        tool._matched_route_where(app, "GET", "/api/documents/doc-1?view=full")
        == "app/routes/documents.py line 1"
    )
    assert (
        tool._matched_route_where(app, "POST", "/api/documents/doc-1")
        == "app/routes/documents.py line 1"
    )

    outside = tmp_path.parent / "foreign_endpoint.py"
    outside_namespace: dict[str, object] = {}
    exec(
        compile("async def endpoint():\n    return {}\n", str(outside), "exec"),
        outside_namespace,
    )
    foreign = FastAPI()
    foreign.add_api_route(
        "/foreign",
        outside_namespace["endpoint"],
        methods=["GET"],  # type: ignore[arg-type]
    )
    assert tool._matched_route_where(foreign, "GET", "/foreign") == ""


# --- host classification of import-time configuration failures --------------


def _envelope(detail: str, error_type: str) -> dict[str, object]:
    return {
        "status": "succeeded",
        "checks": [
            {
                "name": "import app.config",
                "kind": "import",
                "ok": False,
                "error_type": error_type,
                "detail": detail,
            }
        ],
        "routes": [],
    }


def test_config_at_import_is_a_named_error_finding() -> None:
    outcome = classify_envelope(
        _envelope("ConfigError: OCI_RESPONSES_PROJECT_ID is not set.", "ConfigError"),
        staged={"app/config.py": {"content": ""}},
        project_paths=["app/config.py"],
        requirements="fastapi\n",
    )
    (finding,) = outcome.findings
    assert "raised at import over missing configuration" in finding["error"]
    assert finding["severity"] == "error"


def test_config_at_import_blocks_but_env_free_500s_only_advise() -> None:
    assert _blocks_approval(
        {
            "rung": "runtime",
            "path": "app/config.py",
            "error": (
                "import app.config raised at import over missing configuration: "
                "ConfigError: OCI_RESPONSES_PROJECT_ID is not set"
            ),
        }
    )
    # A request-time 500 with no provable shape stays advisory — the sandbox
    # has no environment, so a route that needs one may be perfectly correct.
    assert not _blocks_approval(
        {
            "rung": "runtime",
            "path": "app/main.py",
            "error": "GET /extract failed when the project ran: HTTP 500",
        }
    )
    # ...but a TypeError is provable regardless of environment.
    assert _blocks_approval(
        {
            "rung": "runtime",
            "path": "app/main.py",
            "error": "POST /totals failed: TypeError: unsupported operand type(s)",
        }
    )


def test_ordinary_import_failures_keep_their_existing_shape() -> None:
    outcome = classify_envelope(
        _envelope("division by zero", "ZeroDivisionError"),
        staged={"app/config.py": {"content": ""}},
        project_paths=["app/config.py"],
        requirements="fastapi\n",
    )
    (finding,) = outcome.findings
    assert "raised at import over missing configuration" not in finding["error"]


# --- wiring: the implicit python-multipart requirement -----------------------

_UPLOAD_APP = (
    "from fastapi import FastAPI, File, UploadFile\n"
    "app = FastAPI()\n"
    "@app.post('/upload')\n"
    "async def upload(file: UploadFile = File(...)):\n"
    "    return {'size': len(await file.read())}\n"
)


def _staged(content: str) -> dict[str, dict[str, str]]:
    return {"app/main.py": {"content": content}}


def test_multipart_usage_without_the_package_is_an_error() -> None:
    findings = staged_wiring_errors(
        _staged(_UPLOAD_APP), requirements="fastapi\nuvicorn\n"
    )
    multipart = [f for f in findings if "python-multipart" in f["error"]]
    assert multipart and multipart[0]["severity"] == "error"


def test_multipart_rule_stays_quiet_when_declared_or_unused() -> None:
    declared = staged_wiring_errors(
        _staged(_UPLOAD_APP), requirements="fastapi\npython-multipart\n"
    )
    assert not [f for f in declared if "python-multipart" in f["error"]]
    plain = staged_wiring_errors(
        _staged("from fastapi import FastAPI\napp = FastAPI()\n"),
        requirements="fastapi\n",
    )
    assert not [f for f in plain if "python-multipart" in f["error"]]


# --- acceptance scenarios ----------------------------------------------------


def _assessing_application() -> FastAPI:
    app = FastAPI()

    @app.post("/assess")
    async def assess(file: UploadFile = File(...)) -> dict[str, str]:
        await file.read()
        return {"risk": "low", "supplier": "extracted"}

    @app.get("/records")
    async def records() -> dict[str, list[str]]:
        return {"records": []}

    @app.get("/crashes")
    async def crashes() -> dict[str, int]:
        return {"n": None + 1}  # type: ignore[operator] - the defect under test

    return app


def test_acceptance_scenarios_pass_fail_and_mark_content_misses() -> None:
    """The spec's claims, replayed: a matching response passes, a crash fails
    on status, and a response that merely fails to mention something is marked
    content_miss so the host keeps it advisory."""
    tool = _verify_tool()
    checks = {
        check["name"]: check
        for check in tool._scenario_checks(
            _assessing_application(),
            [
                {
                    "name": "upload is assessed",
                    "method": "POST",
                    "path": "/assess",
                    "body_kind": "image_upload",
                    "expect_status": "2xx",
                    "expect_contains": ["risk"],
                },
                {
                    "name": "verdict is spelled out",
                    "method": "POST",
                    "path": "/assess",
                    "body_kind": "image_upload",
                    "expect_contains": ["high-risk-verdict-string"],
                },
                {
                    "name": "crash is caught",
                    "method": "GET",
                    "path": "/crashes",
                    "expect_status": "2xx",
                },
            ],
        )
    }
    assert checks["acceptance: upload is assessed"]["ok"] is True
    miss = checks["acceptance: verdict is spelled out"]
    assert miss["ok"] is False and miss.get("content_miss") is True
    crash = checks["acceptance: crash is caught"]
    assert crash["ok"] is False and not crash.get("content_miss")


def test_text_upload_scenario_is_typed_and_replays_the_txt_fixture() -> None:
    scenario = AcceptanceScenarioV1(
        name="TXT invoice is extracted locally",
        method="POST",
        path="/documents",
        body_kind="text_upload",
        expect_status="2xx",
    )
    schema = AcceptanceScenarioV1.model_json_schema()
    assert "text_upload" in schema["properties"]["body_kind"]["enum"]

    tool = _verify_tool()
    observed: dict[str, object] = {}
    app = FastAPI()

    @app.post("/documents")
    async def upload(file: UploadFile = File(...)) -> dict[str, str]:
        observed.update(
            filename=file.filename,
            content_type=file.content_type,
            content=await file.read(),
        )
        return {"invoice_number": "INV-1042", "status": "needs_review"}

    (check,) = tool._scenario_checks(app, [scenario.model_dump(mode="json")])

    assert check["ok"] is True, check
    assert observed == {
        "filename": "fixture.txt",
        "content_type": "text/plain",
        "content": tool.TEXT_FIXTURE,
    }


def test_literal_text_upload_uses_body_file_without_a_duplicate_form_field() -> None:
    """Replay old plans faithfully instead of silently swapping in an invoice."""
    scenario = AcceptanceScenarioV1(
        name="shell script is refused",
        method="POST",
        path="/documents",
        body_kind="text_upload",
        body={"file": "#!/bin/sh\necho harmless", "tenant": "meridian"},
        expect_status="4xx",
    )
    observed: dict[str, object] = {}
    app = FastAPI()

    @app.post("/documents")
    async def upload(file: UploadFile = File(...), tenant: str = Form(...)) -> Response:
        observed.update(
            filename=file.filename,
            content_type=file.content_type,
            content=await file.read(),
            tenant=tenant,
        )
        return Response(status_code=415)

    (check,) = _verify_tool()._scenario_checks(app, [scenario.model_dump(mode="json")])

    assert check["ok"] is True, check
    assert observed == {
        "filename": "fixture.txt",
        "content_type": "text/plain",
        "content": b"#!/bin/sh\necho harmless",
        "tenant": "meridian",
    }


def test_executable_upload_scenario_is_typed_and_uses_fixed_safe_fixture() -> None:
    scenario = AcceptanceScenarioV1(
        name="executable is refused",
        method="POST",
        path="/documents",
        body_kind="executable_upload",
        expect_status="4xx",
    )
    schema = AcceptanceScenarioV1.model_json_schema()
    assert "executable_upload" in schema["properties"]["body_kind"]["enum"]
    observed: dict[str, object] = {}
    app = FastAPI()

    @app.post("/documents")
    async def upload(file: UploadFile = File(...)) -> Response:
        observed.update(
            filename=file.filename,
            content_type=file.content_type,
            content=await file.read(),
        )
        return Response(status_code=415)

    (check,) = _verify_tool()._scenario_checks(app, [scenario.model_dump(mode="json")])

    assert check["ok"] is True, check
    assert observed == {
        "filename": "fixture.exe",
        "content_type": "application/octet-stream",
        "content": _verify_tool().EXECUTABLE_FIXTURE,
    }


def test_patch_scenario_is_typed_and_replays_json_with_the_exact_method() -> None:
    """A PATCH workflow must never be coerced to POST by plan or verifier."""
    scenario = AcceptanceScenarioV1(
        name="project status changes",
        method="PATCH",
        path="/api/projects/atlas-2/status",
        body_kind="json",
        body={"status": "healthy"},
        expect_status="2xx",
    )
    schema = AcceptanceScenarioV1.model_json_schema()
    assert schema["properties"]["method"]["enum"] == [
        "GET",
        "POST",
        "PUT",
        "PATCH",
        "DELETE",
    ]

    tool = _verify_tool()
    observed: dict[str, object] = {}
    app = FastAPI()

    @app.patch("/api/projects/{project_id}/status")
    async def update(project_id: str, body: dict[str, str]) -> dict[str, str]:
        observed.update(project_id=project_id, body=body)
        return {"id": project_id, **body}

    (check,) = tool._scenario_checks(app, [scenario.model_dump(mode="json")])

    assert check["ok"] is True, check
    assert observed == {"project_id": "atlas-2", "body": {"status": "healthy"}}


def test_unknown_raw_scenario_method_fails_closed_instead_of_becoming_get() -> None:
    tool = _verify_tool()
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    (check,) = tool._scenario_checks(
        app,
        [{"name": "invalid method", "method": "TRACE", "path": "/health"}],
    )

    assert check["ok"] is False
    assert check["detail"] == "unsupported HTTP method 'TRACE'"


def test_explicit_image_scenario_still_replays_a_real_png() -> None:
    """Changing broad liveness to TXT must not weaken explicit image claims."""
    tool = _verify_tool()
    observed: dict[str, object] = {}
    app = FastAPI()

    @app.post("/images")
    async def upload(file: UploadFile = File(...)) -> dict[str, str]:
        observed.update(
            filename=file.filename,
            content_type=file.content_type,
            content=await file.read(),
        )
        return {"status": "decoded"}

    (check,) = tool._scenario_checks(
        app,
        [
            {
                "name": "image is decoded",
                "method": "POST",
                "path": "/images",
                "body_kind": "image_upload",
                "expect_status": "2xx",
            }
        ],
    )

    assert check["ok"] is True, check
    assert observed == {
        "filename": "fixture.png",
        "content_type": "image/png",
        "content": tool.PNG_FIXTURE,
    }


def test_traceback_attribution_prefers_app_code_over_appkit(tmp_path: Path) -> None:
    tool = _verify_tool()
    tool.PROJECT_DIR = tmp_path
    appkit_namespace: dict[str, object] = {}
    exec(
        compile(
            "def require_config():\n    raise RuntimeError('missing')\n",
            str(tmp_path / "appkit" / "config.py"),
            "exec",
        ),
        appkit_namespace,
    )
    app_namespace = {"require_config": appkit_namespace["require_config"]}
    exec(
        compile(
            "def extract():\n    return require_config()\n",
            str(tmp_path / "app" / "extraction.py"),
            "exec",
        ),
        app_namespace,
    )

    try:
        app_namespace["extract"]()  # type: ignore[operator]
    except RuntimeError as error:
        assert tool._frame_in_project(error) == "app/extraction.py line 2"
    else:  # pragma: no cover - the compiled function always raises
        raise AssertionError("compiled app fixture did not raise")


def test_traceback_attribution_keeps_appkit_fallback(tmp_path: Path) -> None:
    tool = _verify_tool()
    tool.PROJECT_DIR = tmp_path
    namespace: dict[str, object] = {}
    exec(
        compile(
            "def require_config():\n    raise RuntimeError('missing')\n",
            str(tmp_path / "appkit" / "config.py"),
            "exec",
        ),
        namespace,
    )

    try:
        namespace["require_config"]()  # type: ignore[operator]
    except RuntimeError as error:
        assert tool._frame_in_project(error) == "appkit/config.py line 2"
    else:  # pragma: no cover - the compiled function always raises
        raise AssertionError("compiled appkit fixture did not raise")


def test_method_mismatch_is_attributed_to_the_exact_app_route(tmp_path: Path) -> None:
    """A real 405 points at its handler, not whichever test was staged first."""
    tool = _verify_tool()
    tool.PROJECT_DIR = tmp_path
    source = tmp_path / "app" / "api.py"
    namespace: dict[str, object] = {}
    exec(
        compile(
            "async def update_status(project_id: str):\n"
            "    return {'id': project_id}\n",
            str(source),
            "exec",
        ),
        namespace,
    )
    app = FastAPI()
    app.patch("/api/projects/{project_id}/status")(
        namespace["update_status"]  # type: ignore[arg-type]
    )

    assert (
        tool._matched_route_where(app, "POST", "/api/projects/atlas-2/status")
        == "app/api.py line 1"
    )


def test_acceptance_classification_blocks_crashes_and_advises_misses() -> None:
    """Host side: a scenario crash is a provable defect (error); a content
    miss stays a warning — the scenario, not the app, may be the wrong party."""
    envelope = {
        "status": "succeeded",
        "checks": [
            {
                "name": "acceptance: crash",
                "kind": "acceptance",
                "ok": False,
                "detail": "GET /crashes returned HTTP 500, expected 2xx",
            },
            {
                "name": "acceptance: wording",
                "kind": "acceptance",
                "ok": False,
                "content_miss": True,
                "detail": "response never mentions: verdict",
            },
        ],
        "routes": [],
    }
    outcome = classify_envelope(envelope, staged={"app/main.py": {}}, project_paths=[])
    severities = {
        finding["error"][:20]: finding["severity"] for finding in outcome.findings
    }
    assert any(
        finding["severity"] == "error" and "crash" in finding["error"]
        for finding in outcome.findings
    )
    assert any(
        finding["severity"] == "warning" and "wording" in finding["error"]
        for finding in outcome.findings
    ), severities


def test_acceptance_severity_drives_the_shared_approval_gate() -> None:
    wrong_status = {
        "path": "app/main.py",
        "rung": "runtime",
        "kind": "acceptance",
        "severity": "error",
        "error": "acceptance: upload failed: returned HTTP 422, expected 2xx",
    }
    crash = {
        "path": "app/db.py",
        "rung": "runtime",
        "kind": "acceptance",
        "severity": "error",
        "error": (
            "acceptance: review failed: sqlite3.ProgrammingError: "
            "Cannot operate on a closed database"
        ),
    }
    content_miss = {
        "path": "app/main.py",
        "rung": "runtime",
        "kind": "acceptance",
        "severity": "warning",
        "error": "acceptance: verdict failed: response never mentions: verdict",
    }

    assert _blocks_approval(wrong_status)
    assert _blocks_approval(crash)
    assert not _blocks_approval(content_miss)

    reason = _blocking_reason(
        {"errors": [wrong_status, crash, content_miss], "warnings": []}
    )
    assert reason is not None
    assert reason.startswith("2 problem(s)")
    assert "HTTP 422" in reason
    assert _blocking_reason({"errors": [content_miss], "warnings": []}) is None


# ── Exact JSON acceptance assertions ───────────────────────────────────────
# Containment is case-insensitive and satisfiable by extra data. Two live
# models exploited exactly that: asked to correct a wrong seed record, they
# appended a second record beside it, which made "the listing mentions Harbor
# Migration" true while the defect stayed. These pin the strict alternative.


def _listing_application(records: list[dict[str, object]]) -> FastAPI:
    app = FastAPI()

    @app.get("/api/projects")
    async def listing() -> list[dict[str, object]]:
        return records

    @app.get("/api/summary")
    async def summary() -> dict[str, object]:
        return {"total": 2, "owner": "Maya", "healthy": True}

    @app.get("/api/plain")
    async def plain() -> Response:
        return Response(content="not json at all", media_type="text/plain")

    return app


def _exact_scenario(**overrides: object) -> dict[str, object]:
    scenario: dict[str, object] = {
        "name": "listing is exactly the seeded records",
        "method": "GET",
        "path": "/api/projects",
        "expect_status": "2xx",
        "expect_json_exact": [
            {"id": "proj-1", "name": "Harbor Migration"},
            {"id": "proj-2", "name": "Invoice Intelligence"},
        ],
    }
    scenario.update(overrides)
    return scenario


def _run_scenarios(app: FastAPI, scenarios: list[dict[str, object]]):
    tool = _verify_tool()
    return {check["name"]: check for check in tool._scenario_checks(app, scenarios)}


def test_an_appended_third_record_fails_an_exact_listing_assertion() -> None:
    """The exact live workaround: the wrong record left in place, the right
    one added beside it. Containment would call this fixed; exact must not."""
    app = _listing_application(
        [
            {"id": "proj-1", "name": "Placeholder Project"},
            {"id": "proj-2", "name": "Invoice Intelligence"},
            {"id": "proj-3", "name": "Harbor Migration"},
        ]
    )
    checks = _run_scenarios(app, [_exact_scenario()])
    check = checks["acceptance: listing is exactly the seeded records"]

    assert check["ok"] is False
    # Never advisory: the host downgrades content_miss, and an exact claim
    # must never travel through that branch.
    assert check.get("content_miss") is not True
    assert check.get("exact_miss") is True
    detail = str(check["detail"])
    assert "expected 2 element(s), found 3" in detail
    # Enough to correct the record rather than add another one.
    assert "Harbor Migration" in detail and "Placeholder Project" in detail
    assert "adding another record does not satisfy this" in detail


def test_exact_values_are_case_sensitive() -> None:
    app = _listing_application(
        [
            {"id": "proj-1", "name": "harbor migration"},
            {"id": "proj-2", "name": "Invoice Intelligence"},
        ]
    )
    checks = _run_scenarios(app, [_exact_scenario()])
    check = checks["acceptance: listing is exactly the seeded records"]

    assert check["ok"] is False
    assert check.get("exact_miss") is True
    assert '$[0].name: expected "Harbor Migration", found "harbor migration"' in str(
        check["detail"]
    )


def test_object_key_order_never_affects_an_exact_assertion() -> None:
    app = _listing_application(
        [
            {"name": "Harbor Migration", "id": "proj-1"},
            {"name": "Invoice Intelligence", "id": "proj-2"},
        ]
    )
    checks = _run_scenarios(app, [_exact_scenario()])

    assert checks["acceptance: listing is exactly the seeded records"]["ok"] is True


def test_array_order_differences_fail_under_the_default_exact_match() -> None:
    app = _listing_application(
        [
            {"id": "proj-2", "name": "Invoice Intelligence"},
            {"id": "proj-1", "name": "Harbor Migration"},
        ]
    )
    checks = _run_scenarios(app, [_exact_scenario()])
    check = checks["acceptance: listing is exactly the seeded records"]

    assert check["ok"] is False
    assert check.get("exact_miss") is True
    assert "$[0].id" in str(check["detail"])


def test_unordered_array_accepts_a_reordering_but_not_a_changed_multiset() -> None:
    reordered = _listing_application(
        [
            {"id": "proj-2", "name": "Invoice Intelligence"},
            {"id": "proj-1", "name": "Harbor Migration"},
        ]
    )
    checks = _run_scenarios(reordered, [_exact_scenario(json_match="unordered_array")])
    assert checks["acceptance: listing is exactly the seeded records"]["ok"] is True

    # Same members, wrong multiplicity: a duplicate is still a defect.
    duplicated = _listing_application(
        [
            {"id": "proj-1", "name": "Harbor Migration"},
            {"id": "proj-1", "name": "Harbor Migration"},
            {"id": "proj-2", "name": "Invoice Intelligence"},
        ]
    )
    checks = _run_scenarios(duplicated, [_exact_scenario(json_match="unordered_array")])
    check = checks["acceptance: listing is exactly the seeded records"]
    assert check["ok"] is False
    assert "unexpected 1x element" in str(check["detail"])

    # And an appended third record still fails, order-independence or not.
    appended = _listing_application(
        [
            {"id": "proj-1", "name": "Placeholder Project"},
            {"id": "proj-2", "name": "Invoice Intelligence"},
            {"id": "proj-3", "name": "Harbor Migration"},
        ]
    )
    checks = _run_scenarios(appended, [_exact_scenario(json_match="unordered_array")])
    assert checks["acceptance: listing is exactly the seeded records"]["ok"] is False


def test_exact_objects_report_missing_and_unexpected_keys_and_wrong_values() -> None:
    app = _listing_application([])
    checks = _run_scenarios(
        app,
        [
            {
                "name": "summary is exact",
                "method": "GET",
                "path": "/api/summary",
                "expect_status": "2xx",
                "expect_json_exact": {"total": 3, "owner": "Maya", "region": "eu"},
            }
        ],
    )
    detail = str(checks["acceptance: summary is exact"]["detail"])

    assert "$.total: expected 3, found 2" in detail
    assert "$.region: missing, expected" in detail
    assert "$.healthy: unexpected key with value true" in detail


def test_a_non_json_body_fails_an_exact_assertion_without_crashing() -> None:
    app = _listing_application([])
    checks = _run_scenarios(
        app,
        [
            {
                "name": "plain text is not json",
                "method": "GET",
                "path": "/api/plain",
                "expect_status": "2xx",
                "expect_json_exact": {"ok": True},
            }
        ],
    )
    check = checks["acceptance: plain text is not json"]

    assert check["ok"] is False
    assert check.get("exact_miss") is True
    assert "not JSON" in str(check["detail"])


def test_a_matching_exact_assertion_passes_and_reports_no_difference() -> None:
    app = _listing_application(
        [
            {"id": "proj-1", "name": "Harbor Migration"},
            {"id": "proj-2", "name": "Invoice Intelligence"},
        ]
    )
    checks = _run_scenarios(app, [_exact_scenario()])
    check = checks["acceptance: listing is exactly the seeded records"]

    assert check["ok"] is True
    assert check.get("exact_miss") is None


def test_expect_contains_scenarios_are_untouched_by_the_new_field() -> None:
    """Backward compatibility: a scenario with no exact assertion behaves
    exactly as it did, including staying advisory on a containment miss."""
    app = _listing_application([{"id": "proj-1", "name": "Harbor Migration"}])
    checks = _run_scenarios(
        app,
        [
            {
                "name": "listing mentions the project",
                "method": "GET",
                "path": "/api/projects",
                "expect_status": "2xx",
                "expect_contains": ["harbor migration"],
            },
            {
                "name": "listing mentions something absent",
                "method": "GET",
                "path": "/api/projects",
                "expect_status": "2xx",
                "expect_contains": ["citizen portal"],
            },
        ],
    )

    assert checks["acceptance: listing mentions the project"]["ok"] is True
    miss = checks["acceptance: listing mentions something absent"]
    assert miss["ok"] is False
    assert miss.get("content_miss") is True
    assert miss.get("exact_miss") is None


def test_an_exact_failure_is_blocking_while_a_containment_miss_stays_advisory() -> None:
    """The host-side half of the rule: severity, not just the check flag."""
    outcome = classify_envelope(
        {
            "status": "succeeded",
            "checks": [
                {
                    "name": "acceptance: listing is exact",
                    "kind": "acceptance",
                    "ok": False,
                    "exact_miss": True,
                    "detail": "$[0].name: expected X, found Y",
                    "where": "app/db.py line 6",
                },
                {
                    "name": "acceptance: page mentions the title",
                    "kind": "acceptance",
                    "ok": False,
                    "content_miss": True,
                    "detail": "the response never mentions: dashboard",
                },
            ],
        },
        staged={"app/db.py": {"content": "SEED = []\n"}},
        project_paths=["app/db.py"],
    )
    by_error = {item["error"]: item for item in outcome.findings}
    exact = next(item for key, item in by_error.items() if "expected X" in key)
    advisory = next(item for key, item in by_error.items() if "never mentions" in key)

    assert exact["severity"] == "error"
    # The verifier-resolved app-owned path rides along, so the routed repair
    # can attribute it to the slice that owns that file.
    assert exact["path"] == "app/db.py"
    assert advisory["severity"] == "warning"
