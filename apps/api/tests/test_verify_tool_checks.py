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

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from waqil_api.control_plane import _blocks_approval
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
