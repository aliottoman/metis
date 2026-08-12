"""Token-conscious, production-shaped project capability scenarios.

The original Meridian OCI evaluation remains the default and keeps its report
contract in :mod:`waqil_api.project_capability_eval`.  This module adds two
smaller existing-code evaluations whose seeds, requested write scopes, and
acceptance probes are deterministic.  Neither scenario imports generated code
on the host; live acceptance uses the same networkless project sandbox as
Meridian.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .config import Settings
from .project_capability_eval import (
    MERIDIAN_ACCEPTANCE_SOURCE,
    MERIDIAN_REQUEST,
    REPAIR_REQUEST,
    REQUIRED_FILES,
    SCENARIO_NAME,
    acceptance_staged_overlay,
    initialize_meridian_project,
    repair_attempt_evidence,
)
from .project_sandbox import ProjectSandboxService


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_seed(project: Path, files: Mapping[str, str]) -> None:
    project.mkdir(parents=True, exist_ok=True)
    for relative, content in files.items():
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


@dataclass(frozen=True, slots=True)
class CapabilityScenario:
    """One immutable evaluator contract and its disposable repository seed."""

    key: str
    name: str
    slug: str
    request: str
    repair_request: str
    required_files: tuple[str, ...]
    protected_files: tuple[str, ...]
    seed_files: Mapping[str, str]
    acceptance_source: str
    probe_module: str
    initialize: Callable[[Path], None]
    allow_host_scaffold: bool = False
    # The outcomes this evaluator's own probe asserts. A plan whose declared
    # scenarios do not cover them cannot be judged by that probe, so the run
    # is stopped before any coder inference rather than after: a live Atlas
    # canary declared six scenarios against a probe asserting eight and spent
    # 292,436 coder tokens before the gap was visible.
    required_scenarios: tuple[str, ...] = ()


# What the Atlas probe actually asserts, named as outcomes rather than as the
# planner's own wording, so coverage is judged by identity and not by string
# equality with whatever phrasing a model chose.
UI_REQUIRED_SCENARIOS: tuple[str, ...] = (
    "health endpoint returns the exact service payload",
    "listing returns the exact seeded records",
    "patch changes a project status",
    "patch to an unknown project is refused",
    "patch with an invalid status is refused",
    "the console page serves",
    "the stylesheet is served",
    "the interaction script is served",
)

# Substrings that identify each required outcome in a planner-authored
# scenario. A scenario matches an outcome when its name or its request line
# carries every term in one of the groups -- identity, not phrasing.
UI_REQUIRED_SCENARIO_SIGNATURES: dict[str, tuple[tuple[str, ...], ...]] = {
    "health endpoint returns the exact service payload": (("health",),),
    "listing returns the exact seeded records": (("/api/projects", "get"),),
    "patch changes a project status": (("patch", "status"),),
    "patch to an unknown project is refused": (
        ("patch", "404"),
        ("patch", "unknown"),
        ("patch", "missing"),
        ("patch", "not found"),
    ),
    "patch with an invalid status is refused": (
        ("patch", "422"),
        ("patch", "invalid"),
        ("patch", "validation"),
    ),
    "the console page serves": (("/", "page"), ("console",), ("index",)),
    "the stylesheet is served": (("styles.css",), ("stylesheet",)),
    "the interaction script is served": (("app.js",), ("script",)),
}


def missing_required_scenarios(
    scenario: CapabilityScenario, declared: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Required outcomes no declared scenario covers, by identity.

    Counting is not coverage: six scenarios can miss two required outcomes and
    duplicate two others. Each required outcome is matched against every
    declared scenario's name, method and path together, so a planner that
    words things its own way still passes while one that simply omits an
    outcome does not.
    """

    required = tuple(scenario.required_scenarios)
    if not required:
        return []
    signatures = UI_REQUIRED_SCENARIO_SIGNATURES if scenario.key == "ui-revamp" else {}
    haystacks = [
        " ".join(
            str(item.get(field) or "") for field in ("name", "method", "path")
        ).casefold()
        + " "
        + json.dumps(item.get("expect_json_exact") or {}, sort_keys=True).casefold()
        for item in declared
    ]
    missing: list[str] = []
    for outcome in required:
        groups = signatures.get(outcome) or ((outcome.casefold(),),)
        if not any(
            all(term in haystack for term in group)
            for haystack in haystacks
            for group in groups
        ):
            missing.append(outcome)
    return missing


UI_MAIN = textwrap.dedent(
    """\
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    from .api import router


    def create_app() -> FastAPI:
        application = FastAPI(title="Atlas Projects API")
        application.include_router(router, prefix="/api")
        application.mount("/static", StaticFiles(directory="app/static"), name="static")

        @application.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse("app/static/index.html")

        return application


    app = create_app()
    """
)

UI_STORE = textwrap.dedent(
    """\
    from __future__ import annotations

    from copy import deepcopy


    _PROJECTS = [
        {"id": "atlas-1", "name": "Harbor Migration", "owner": "Maya", "status": "healthy", "risk": 12},
        {"id": "atlas-2", "name": "Invoice Intelligence", "owner": "Noor", "status": "at_risk", "risk": 68},
        {"id": "atlas-3", "name": "Citizen Portal", "owner": "Omar", "status": "blocked", "risk": 91},
    ]


    def list_projects() -> list[dict[str, object]]:
        return deepcopy(_PROJECTS)


    def set_status(project_id: str, status: str) -> dict[str, object] | None:
        for project in _PROJECTS:
            if project["id"] == project_id:
                project["status"] = status
                return deepcopy(project)
        return None
    """
)

UI_API = textwrap.dedent(
    """\
    from typing import Literal

    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel

    from .store import list_projects, set_status


    router = APIRouter()


    class StatusUpdate(BaseModel):
        status: Literal["healthy", "at_risk", "blocked"]


    @router.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "atlas-projects"}


    @router.get("/projects")
    def projects() -> dict[str, object]:
        values = list_projects()
        return {"items": values, "count": len(values)}


    @router.patch("/projects/{project_id}/status")
    def update_status(project_id: str, body: StatusUpdate) -> dict[str, object]:
        project = set_status(project_id, body.status)
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")
        return project
    """
)

UI_API_TEST = textwrap.dedent(
    """\
    from fastapi.testclient import TestClient

    from app.main import create_app


    def test_health_contract() -> None:
        with TestClient(create_app()) as client:
            assert client.get("/api/health").json() == {
                "status": "ok", "service": "atlas-projects"
            }


    def test_project_contract() -> None:
        with TestClient(create_app()) as client:
            response = client.get("/api/projects")
            assert response.status_code == 200
            assert response.json()["count"] == 3
    """
)

UI_SEED_FILES: dict[str, str] = {
    ".gitignore": "__pycache__/\n.env\n",
    "app/__init__.py": '"""Atlas project dashboard."""\n',
    "app/main.py": UI_MAIN,
    "app/api.py": UI_API,
    "app/store.py": UI_STORE,
    "app/static/index.html": textwrap.dedent(
        """\
        <!doctype html><html><head><title>Projects</title></head>
        <body class="legacy-shell"><h1>Projects</h1><div id="projects">Loading...</div>
        <script src="/static/app.js"></script></body></html>
        """
    ),
    "app/static/styles.css": "body { font-family: sans-serif; }\n",
    "app/static/app.js": "fetch('/api/projects').then(r => r.json()).then(console.log);\n",
    "tests/test_api_contract.py": UI_API_TEST,
    "requirements.txt": "fastapi>=0.115\npydantic>=2.10\nuvicorn>=0.34\nhttpx>=0.28\n",
    "README.md": "# Atlas Projects\n\nLegacy operations dashboard.\n",
}

UI_REQUIRED_FILES = (
    "app/static/index.html",
    "app/static/styles.css",
    "app/static/app.js",
    "tests/test_ui_contract.py",
    "README.md",
)

UI_PROTECTED_FILES = (
    "app/main.py",
    "app/api.py",
    "app/store.py",
    "tests/test_api_contract.py",
)

UI_REVAMP_REQUEST = """Fully revamp the existing Atlas Projects web UI without changing its API.

This is an existing production-shaped FastAPI project, not a greenfield build.
Inspect the API and regression tests first. Preserve the exact behavior and
bytes of app/main.py, app/api.py, app/store.py, and tests/test_api_contract.py.
Do not replace the backend, add a frontend framework, or introduce a CDN.

Modify or create exactly these application-owned files:
app/static/index.html, app/static/styles.css, app/static/app.js,
tests/test_ui_contract.py, README.md.

Build a polished responsive "Atlas Operations Console" for the existing
project data. It needs an accessible sidebar and mobile navigation, page
header, three KPI cards, searchable/filterable project table, risk/status
badges, a project status action using PATCH /api/projects/{project_id}/status,
and explicit loading, empty, success, and error states. Use semantic HTML,
visible keyboard focus, CSS custom properties, reduced-motion-safe behavior,
and no external assets. JavaScript must fetch the existing /api/projects and
/api/health contracts and must not invent endpoints or hard-code project rows.

The API's behavior is fixed and must be preserved exactly. GET /api/projects
answers {"items": [...], "count": 3} where items are exactly these three
records, each with id, name, owner, status and risk:
  atlas-1, "Harbor Migration", owner Maya, status healthy, risk 12
  atlas-2, "Invoice Intelligence", owner Noor, status at_risk, risk 68
  atlas-3, "Citizen Portal", owner Omar, status blocked, risk 91
GET /api/health answers exactly {"status": "ok", "service": "atlas-projects"}.
PATCH /api/projects/{project_id}/status accepts {"status": "..."} where status
is one of healthy, at_risk or blocked; it answers 200 with the updated project,
404 for an unknown project id, and 422 for a status outside that set.

Write acceptance scenarios that check these exactly. Where the requirement
names exact records, exact field values or an exact count, assert the exact
JSON structure rather than substrings — a substring check passes when a record
is added beside a wrong one, which is not the behavior specified above. Keep
PATCH represented as PATCH.

Add meaningful UI contract tests that protect the API URLs, accessibility
landmarks, responsive assets, and interaction controls. Update README.md with
local run/test instructions and the preserved API contract. Finish only after
the existing API regressions and new UI checks pass.
"""

UI_REPAIR_REQUEST = """Continue the Atlas UI revamp from the exact staged overlay.
Repair only the verifier findings. Preserve the protected API files byte for
byte, keep the five-file write scope, and re-run both existing API regression
tests and the new UI contract tests before finishing.
"""


def initialize_ui_revamp_project(project: Path) -> None:
    _write_seed(project, UI_SEED_FILES)


_ui_hashes = {path: _sha256(UI_SEED_FILES[path]) for path in UI_PROTECTED_FILES}
UI_REVAMP_ACCEPTANCE_SOURCE = textwrap.dedent(
    '''\
    """Host-owned Atlas UI/API regression probe."""
    from __future__ import annotations

    import hashlib
    from pathlib import Path

    from fastapi.testclient import TestClient


    root = Path.cwd()
    protected = __PROTECTED_HASHES__
    for relative, expected in protected.items():
        actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        assert actual == expected, f"protected backend regression: {relative} changed"

    html = (root / "app/static/index.html").read_text(encoding="utf-8").casefold()
    css = (root / "app/static/styles.css").read_text(encoding="utf-8").casefold()
    script = (root / "app/static/app.js").read_text(encoding="utf-8").casefold()
    readme = (root / "README.md").read_text(encoding="utf-8").casefold()
    tests = (root / "tests/test_ui_contract.py").read_text(encoding="utf-8").casefold()

    for marker in ("atlas operations console", "viewport", "stylesheet", "app.js", "search", "status"):
        assert marker in html, f"revamped UI omits {marker!r}"
    assert "legacy-shell" not in html, "legacy UI shell was not replaced"
    assert "https://" not in html + css + script, "UI depends on an external asset/CDN"
    for marker in ("--", "@media", ":focus-visible", "prefers-reduced-motion"):
        assert marker in css, f"responsive/accessibility CSS omits {marker!r}"
    for marker in ("fetch(", "/api/projects", "/api/health", "patch", "error", "loading"):
        assert marker in script, f"UI interaction script omits {marker!r}"
    assert tests.count("def test_") >= 3, "UI contract suite needs at least three tests"
    for marker in ("/api/projects", "aria", "viewport", "app.js"):
        assert marker in tests, f"UI tests omit {marker!r}"
    for marker in ("uvicorn", "pytest", "/api/projects"):
        assert marker in readme, f"README omits {marker!r} guidance"

    from app.main import create_app

    with TestClient(create_app()) as client:
        assert client.get("/api/health").json() == {"status": "ok", "service": "atlas-projects"}
        listing = client.get("/api/projects")
        assert listing.status_code == 200
        body = listing.json()
        assert body["count"] == 3
        # The exact seeded records, not a set of names and not a containment
        # check: a name set collapses duplicates, and "mentions X" is
        # satisfied by APPENDING X beside a wrong record -- the precise
        # workaround two live coder models produced when asked to correct
        # seed data. Compare id/name/status per record, order-independent
        # because the API does not promise an order.
        seeded = sorted(
            ({"id": item["id"], "name": item["name"], "status": item["status"]}
             for item in body["items"]),
            key=lambda item: item["id"],
        )
        assert seeded == [
            {"id": "atlas-1", "name": "Harbor Migration", "status": "healthy"},
            {"id": "atlas-2", "name": "Invoice Intelligence", "status": "at_risk"},
            {"id": "atlas-3", "name": "Citizen Portal", "status": "blocked"},
        ], f"seeded records are not exactly the ones the API must preserve: {seeded}"
        changed = client.patch("/api/projects/atlas-1/status", json={"status": "blocked"})
        assert changed.status_code == 200 and changed.json()["status"] == "blocked"
        assert client.patch("/api/projects/missing/status", json={"status": "healthy"}).status_code == 404
        assert client.patch("/api/projects/atlas-1/status", json={"status": "unknown"}).status_code == 422
        page = client.get("/")
        assert page.status_code == 200 and "atlas operations console" in page.text.casefold()
        assert client.get("/static/styles.css").status_code == 200
        assert client.get("/static/app.js").status_code == 200
    '''
).replace("__PROTECTED_HASHES__", repr(_ui_hashes))


REPAIR_MAIN = textwrap.dedent(
    """\
    from fastapi import FastAPI

    from .db import initialize_schema
    from .routes import router


    def create_app() -> FastAPI:
        initialize_schema()
        application = FastAPI(title="Incident Continuity API")
        application.include_router(router, prefix="/api")
        return application


    app = create_app()
    """
)

REPAIR_MODELS = textwrap.dedent(
    """\
    from typing import Literal

    from pydantic import BaseModel, Field


    Severity = Literal["low", "medium", "high", "critical"]
    IncidentStatus = Literal["open", "investigating", "resolved"]


    class IncidentCreate(BaseModel):
        title: str = Field(min_length=3, max_length=200)
        severity: Severity


    class IncidentUpdate(BaseModel):
        status: IncidentStatus
    """
)

REPAIR_SEED_FILES: dict[str, str] = {
    ".gitignore": "__pycache__/\n.env\n*.sqlite3\n",
    "app/__init__.py": '"""Incident continuity service."""\n',
    "app/main.py": REPAIR_MAIN,
    "app/models.py": REPAIR_MODELS,
    "app/db.py": textwrap.dedent(
        """\
        import os
        import sqlite3
        from pathlib import Path


        def database_path() -> Path:
            return Path(os.getenv("INCIDENT_DB_PATH", "./incidents.sqlite3"))


        def connect() -> sqlite3.Connection:
            return sqlite3.connect(database_path())


        def initialize_schema() -> None:
            with connect() as connection:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS incidents ("
                    "id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, "
                    "priority TEXT NOT NULL, status TEXT NOT NULL)"
                )
        """
    ),
    "app/repository.py": textwrap.dedent(
        """\
        from .db import connect


        def create_incident(title: str, severity: str) -> dict[str, object]:
            with connect() as connection:
                cursor = connection.execute(
                    "INSERT INTO incidents (title, severity, status) VALUES (?, ?, ?)",
                    (title, severity, "open"),
                )
                incident_id = int(cursor.lastrowid or 0)
            return get_incident(incident_id)


        def get_incident(incident_id: int) -> dict[str, object] | None:
            with connect() as connection:
                row = connection.execute(
                    "SELECT id, title, severity, status FROM incidents WHERE id = ?",
                    (incident_id,),
                ).fetchone()
            return dict(row) if row else None


        def list_incidents() -> list[dict[str, object]]:
            with connect() as connection:
                rows = connection.execute(
                    "SELECT id, title, severity, status FROM incidents ORDER BY id"
                ).fetchall()
            return [dict(row) for row in rows]


        def update_incident(incident_id: int, status: str) -> dict[str, object] | None:
            connection = connect()
            connection.execute(
                "UPDATE incidents SET status = ? WHERE id = ?", (status, incident_id)
            )
            connection.close()
            return get_incident(incident_id)
        """
    ),
    "app/routes.py": textwrap.dedent(
        """\
        from fastapi import APIRouter

        from .models import IncidentCreate, IncidentUpdate
        from .repository import create_incident, get_incident, list_incidents, update_incident


        router = APIRouter()


        @router.get("/health")
        def health() -> dict[str, str]:
            return {"status": "ok"}


        @router.post("/incidents")
        def create(body: IncidentCreate) -> dict[str, object]:
            return create_incident(body.title, body.severity)


        @router.get("/incidents")
        def listing() -> list[dict[str, object]]:
            return list_incidents()


        @router.get("/incidents/{incident_id}")
        def get(incident_id: int) -> dict[str, object] | None:
            return get_incident(incident_id)


        @router.patch("/incidents/{incident_id}")
        def update(incident_id: int, body: IncidentUpdate) -> dict[str, object] | None:
            return update_incident(incident_id, body.status)
        """
    ),
    "tests/test_incidents.py": textwrap.dedent(
        """\
        from fastapi.testclient import TestClient

        from app.main import create_app


        def test_health() -> None:
            with TestClient(create_app()) as client:
                assert client.get("/api/health").status_code == 200
        """
    ),
    "requirements.txt": "fastapi>=0.115\npydantic>=2.10\nuvicorn>=0.34\nhttpx>=0.28\n",
    "README.md": "# Incident Continuity API\n\nA deliberately fault-injected repair fixture.\n",
}

REPAIR_REQUIRED_FILES = (
    "app/db.py",
    "app/repository.py",
    "app/routes.py",
    "tests/test_incidents.py",
)

REPAIR_PROTECTED_FILES = (
    "app/main.py",
    "app/models.py",
    "requirements.txt",
)

FASTAPI_REPAIR_REQUEST = """Repair the existing fault-injected Incident Continuity FastAPI service.

This is a diagnosis and continuity task, not a rewrite. Read the existing
application and tests, reproduce the root causes, and preserve app/main.py,
app/models.py, requirements.txt, route URLs, and request schemas byte for byte.

Modify exactly these files: app/db.py, app/repository.py, app/routes.py,
tests/test_incidents.py.

Required behavior:
- configuration remains lazy through INCIDENT_DB_PATH;
- SQLite schema stores id, title, severity, and status, and row conversion is safe;
- POST /api/incidents returns 201 and persists an open incident;
- GET /api/incidents and GET /api/incidents/{id} return stable JSON;
- a missing incident returns 404, not null or 500;
- PATCH persists investigating/resolved across a fresh create_app()/TestClient;
- invalid severity/status remains a 422 schema error;
- connections and transactions are closed/committed correctly;
- regression tests cover creation, validation, missing IDs, updates, and
  persistence continuity without xfail, skip, mocks, network, or sleeps.

Fix root causes in the existing layers. Do not bypass SQLite, replace the app,
weaken validation, or hard-code successful responses. Run the real tests and
repair every blocking finding before finishing.
"""

FASTAPI_REPAIR_FOLLOWUP = """Continue the Incident Continuity repair from the exact staged overlay.
Use the newest verifier findings as the repair queue. Keep the protected files
unchanged, preserve the four-file scope, and prove the corrected status survives
a fresh application/client before finishing.
"""


def initialize_fastapi_repair_project(project: Path) -> None:
    _write_seed(project, REPAIR_SEED_FILES)


_repair_hashes = {
    path: _sha256(REPAIR_SEED_FILES[path]) for path in REPAIR_PROTECTED_FILES
}
FASTAPI_REPAIR_ACCEPTANCE_SOURCE = textwrap.dedent(
    '''\
    """Host-owned incident repair and continuity probe."""
    from __future__ import annotations

    import hashlib
    import os
    import sqlite3
    from pathlib import Path

    from fastapi.testclient import TestClient


    root = Path.cwd()
    protected = __PROTECTED_HASHES__
    for relative, expected in protected.items():
        actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        assert actual == expected, f"protected application contract changed: {relative}"

    tests = (root / "tests/test_incidents.py").read_text(encoding="utf-8").casefold()
    assert tests.count("def test_") >= 4, "incident regression suite needs at least four tests"
    assert "xfail" not in tests and "skip" not in tests, "regressions may not be disabled"
    for marker in ("201", "404", "422", "resolved", "create_app"):
        assert marker in tests, f"incident tests omit {marker!r} continuity evidence"

    database = Path("/tmp/metis-incident-repair-eval.sqlite3")
    try:
        database.unlink()
    except FileNotFoundError:
        pass
    os.environ["INCIDENT_DB_PATH"] = str(database)

    from app.main import create_app

    with TestClient(create_app()) as client:
        assert client.get("/api/health").json() == {"status": "ok"}
        invalid = client.post("/api/incidents", json={"title": "Bad severity", "severity": "urgent"})
        assert invalid.status_code == 422
        created = client.post(
            "/api/incidents",
            json={"title": "OCI extraction queue stalled", "severity": "critical"},
        )
        assert created.status_code == 201, created.text
        incident = created.json()
        incident_id = incident["id"]
        assert incident["status"] == "open" and incident["severity"] == "critical"
        missing = client.get("/api/incidents/999999")
        assert missing.status_code == 404
        updated = client.patch(
            f"/api/incidents/{incident_id}", json={"status": "resolved"}
        )
        assert updated.status_code == 200 and updated.json()["status"] == "resolved"

    with TestClient(create_app()) as restarted:
        persisted = restarted.get(f"/api/incidents/{incident_id}")
        assert persisted.status_code == 200 and persisted.json()["status"] == "resolved"
        listing = restarted.get("/api/incidents")
        assert listing.status_code == 200
        assert any(item["id"] == incident_id for item in listing.json())

    with sqlite3.connect(database) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(incidents)")}
        assert {"id", "title", "severity", "status"} <= columns
        row = connection.execute(
            "SELECT severity, status FROM incidents WHERE id = ?", (incident_id,)
        ).fetchone()
        assert row == ("critical", "resolved")
    '''
).replace("__PROTECTED_HASHES__", repr(_repair_hashes))


MERIDIAN_SCENARIO = CapabilityScenario(
    key="meridian",
    name=SCENARIO_NAME,
    slug="meridian-evidence-desk",
    request=MERIDIAN_REQUEST,
    repair_request=REPAIR_REQUEST,
    required_files=REQUIRED_FILES,
    protected_files=(),
    seed_files={},
    acceptance_source=MERIDIAN_ACCEPTANCE_SOURCE,
    probe_module="metis_eval_acceptance",
    initialize=initialize_meridian_project,
    allow_host_scaffold=True,
)

UI_REVAMP_SCENARIO = CapabilityScenario(
    key="ui-revamp",
    name="Atlas Existing-Project UI Revamp",
    slug="atlas-projects",
    request=UI_REVAMP_REQUEST,
    repair_request=UI_REPAIR_REQUEST,
    required_files=UI_REQUIRED_FILES,
    protected_files=UI_PROTECTED_FILES,
    seed_files=UI_SEED_FILES,
    acceptance_source=UI_REVAMP_ACCEPTANCE_SOURCE,
    probe_module="metis_eval_ui_revamp",
    initialize=initialize_ui_revamp_project,
    required_scenarios=UI_REQUIRED_SCENARIOS,
)

FASTAPI_REPAIR_SCENARIO = CapabilityScenario(
    key="fastapi-repair",
    name="Incident Continuity Fault Repair",
    slug="incident-continuity",
    request=FASTAPI_REPAIR_REQUEST,
    repair_request=FASTAPI_REPAIR_FOLLOWUP,
    required_files=REPAIR_REQUIRED_FILES,
    protected_files=REPAIR_PROTECTED_FILES,
    seed_files=REPAIR_SEED_FILES,
    acceptance_source=FASTAPI_REPAIR_ACCEPTANCE_SOURCE,
    probe_module="metis_eval_fastapi_repair",
    initialize=initialize_fastapi_repair_project,
)

SCENARIOS: dict[str, CapabilityScenario] = {
    item.key: item
    for item in (MERIDIAN_SCENARIO, UI_REVAMP_SCENARIO, FASTAPI_REPAIR_SCENARIO)
}

QUALIFICATION_GATE_NAMES = (
    "exact_planned_scope",
    "required_files_written",
    "clean_post_write_verification",
    "repair_continuity",
    "approved_disposable_changeset",
    "required_acceptance_probe",
)


def get_scenario(key: str) -> CapabilityScenario:
    try:
        return SCENARIOS[key]
    except KeyError as error:
        raise ValueError(f"unknown project capability scenario: {key}") from error


def qualify_scenario(
    scenario: CapabilityScenario,
    attempts: Sequence[Mapping[str, Any]],
    acceptance: Mapping[str, Any],
    *,
    approved: bool,
) -> dict[str, Any]:
    """Return fail-closed binary gates; diagnostics never average defects away."""

    first = attempts[0] if attempts else {}
    final = attempts[-1] if attempts else {}
    planned = set((first.get("plan") or {}).get("files") or [])
    required = set(scenario.required_files)
    successful = {
        str(path)
        for attempt in attempts
        for path in ((attempt.get("writes") or {}).get("unique_successful_paths") or [])
    }
    verification = final.get("verification") or {}
    approval = final.get("approval") or {}
    repair_evidence = [
        repair_attempt_evidence(attempt, planned_paths=planned)
        for attempt in attempts[1:]
    ]
    host_scaffold_paths = {
        str(path)
        for attempt in attempts
        for path in ((attempt.get("writes") or {}).get("host_scaffold_paths") or [])
    }
    gates = {
        "exact_planned_scope": bool(
            attempts
            and planned == required
            and (scenario.allow_host_scaffold or not host_scaffold_paths)
        ),
        "required_files_written": required <= successful,
        "clean_post_write_verification": bool(
            int(verification.get("attempts") or 0) > 0
            and verification.get("after_last_successful_write") is True
            and int(verification.get("blocking") or 0) == 0
            and approval.get("offered")
            and not approval.get("blocked")
        ),
        # Continuity alone is insufficient: every follow-up must also contain
        # an observed in-plan mutation and verification after its final write.
        "repair_continuity": all(all(item) for item in repair_evidence),
        "approved_disposable_changeset": bool(approved),
        "required_acceptance_probe": bool(
            acceptance.get("available") and acceptance.get("passed")
        ),
    }
    passed = all(gates.values())
    return {
        "mode": "binary",
        "passed": passed,
        "score": 100 if passed else 0,
        "gates": gates,
        "failed_gates": [name for name, value in gates.items() if not value],
    }


async def run_scenario_acceptance(
    scenario: CapabilityScenario,
    project_root: Path,
    settings: Settings,
    *,
    staged: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run one scenario probe over approved or exact pending sandbox bytes."""

    paths = [
        str(path.relative_to(project_root))
        for path in sorted(project_root.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and ".metis" not in path.relative_to(project_root).parts
    ]
    staged_requirements = (staged or {}).get("requirements.txt", {}).get("content")
    if isinstance(staged_requirements, str):
        requirements = staged_requirements
    else:
        try:
            requirements = (project_root / "requirements.txt").read_text(
                encoding="utf-8"
            )
        except OSError:
            requirements = ""
    probe_path = f"{scenario.probe_module}.py"
    try:
        verifier_overlay = acceptance_staged_overlay(
            staged,
            probe_path=probe_path,
            probe_source=scenario.acceptance_source,
        )
    except ValueError as error:
        return {
            "available": False,
            "reason": f"could not materialize exact pending acceptance overlay: {error}",
            "passed": False,
            "checks_total": 1,
            "checks_passed": 0,
            "sandbox_checks_total": 0,
            "sandbox_checks_passed": 0,
            "required_probe": {},
            "blocking_findings": [],
            "warnings": [],
            "routes": [],
            "skipped_modules": [],
        }
    service = ProjectSandboxService(settings)
    try:
        outcome = await service.verify(
            root=project_root,
            staged=verifier_overlay,
            project_paths=paths,
            requirements=requirements,
        )
    finally:
        await service.release_machine(reason="capability evaluation complete")

    check_name = f"import {scenario.probe_module}"
    required_check = next(
        (check for check in outcome.checks if check.get("name") == check_name),
        None,
    )
    blocking = [
        finding for finding in outcome.findings if finding.get("severity") != "warning"
    ]
    passed = bool(
        outcome.available
        and required_check
        and required_check.get("ok")
        and not blocking
    )
    if not outcome.available:
        reason = outcome.reason or "the sandbox acceptance probe was unavailable"
    elif required_check is None:
        reason = f"the sandbox did not execute the required {scenario.name} probe"
    elif not required_check.get("ok"):
        reason = str(required_check.get("detail") or "the required probe failed")
    elif blocking:
        reason = "the sandbox reported blocking findings outside the required probe"
    else:
        reason = ""
    return {
        "available": outcome.available,
        "reason": reason,
        "passed": passed,
        "checks_total": 1,
        "checks_passed": int(passed),
        "sandbox_checks_total": len(outcome.checks),
        "sandbox_checks_passed": sum(bool(check.get("ok")) for check in outcome.checks),
        "required_probe": dict(required_check or {}),
        "blocking_findings": blocking,
        "warnings": [
            finding
            for finding in outcome.findings
            if finding.get("severity") == "warning"
        ],
        "routes": outcome.routes,
        "skipped_modules": outcome.skipped_modules,
    }


def run_scenario_acceptance_sync(
    scenario: CapabilityScenario,
    project_root: Path,
    settings: Settings,
    *,
    staged: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    return asyncio.run(
        run_scenario_acceptance(scenario, project_root, settings, staged=staged)
    )
