"""The planner boundary, end to end, through the real production path.

Four live Run 2 attempts never reached coding, so normalization has never been
exercised past the gate: every proof of it replays a captured plan into
private helpers. This closes that with a deterministic planner instead of a
live one.

Everything below the planner is production code on one event loop: the real
control plane, the real normalization and topology gate, the real
ProjectCodingCoordinator and workspace mirror, the real sandbox-free
verification ladder, the real approval. The only fakes are the planner's reply
(the captured valid three-slice Atlas plan) and the coding engine, which
writes the bytes a coder would have written into the real disposable mirror.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
import pytest

from waqil_api.coding_contracts import (
    CodingEngineInfoV1,
    CodingEventV1,
    CodingModelRouteV1,
    CodingProviderV1,
    CodingUsageV1,
    ContinueSliceV1,
    DeleteSessionResultV1,
    RestartWithModelV1,
    SliceResultV1,
    StartSliceV1,
)
from waqil_api.config import Settings
from waqil_api.contracts import (
    AcceptanceScenarioV1,
    ProjectBuildPlanV1,
    ProjectVerticalSliceV1,
)
from waqil_api.main import create_app
from waqil_api.project_capability_eval import project_asset_id
from waqil_api.project_capability_scenarios import UI_REVAMP_REQUEST


# ── The existing project, and what the build must produce ──────────────────

SEED_MAIN = '''\
"""Atlas Projects API."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import router

_STATIC = Path(__file__).resolve().parent / "static"


def create_app() -> FastAPI:
    application = FastAPI(title="Atlas Projects API")
    application.include_router(router, prefix="/api")
    if _STATIC.is_dir():
        application.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

    @application.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(str(_STATIC / "index.html"))

    return application


app = create_app()
'''

SEED_API = '''\
"""The API contract this revamp must preserve exactly."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.store import list_projects, set_status

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
'''

SEED_STORE = '''\
"""The seeded records, which the revamp must not change."""

from __future__ import annotations

from copy import deepcopy

_PROJECTS = [
    {"id": "atlas-1", "name": "Harbor Migration", "status": "healthy"},
    {"id": "atlas-2", "name": "Invoice Intelligence", "status": "at_risk"},
]


def list_projects() -> list[dict[str, object]]:
    return deepcopy(_PROJECTS)


def set_status(project_id: str, status: str) -> dict[str, object] | None:
    for project in _PROJECTS:
        if project["id"] == project_id:
            project["status"] = status
            return deepcopy(project)
    return None
'''

SEED_API_TEST = """\
from fastapi.testclient import TestClient

from app.main import create_app


def test_health_contract() -> None:
    with TestClient(create_app()) as client:
        assert client.get("/api/health").json() == {
            "status": "ok",
            "service": "atlas-projects",
        }
"""

PROTECTED_SEED = {
    "app/main.py": SEED_MAIN,
    "app/api.py": SEED_API,
    "app/store.py": SEED_STORE,
    "tests/test_api_contract.py": SEED_API_TEST,
}

# The five files the revamp owns, in dependency order.
PLANNED = [
    "app/static/index.html",
    "app/static/styles.css",
    "app/static/app.js",
    "tests/test_ui_contract.py",
    "README.md",
]

# What the coder will write. Real bytes: the verification ladder imports and
# exercises this application for real.
BUILT = {
    "app/static/index.html": (
        "<!doctype html>\n"
        '<html lang="en">\n'
        '<head><meta charset="utf-8"><title>Atlas Operations Console</title>\n'
        '<link rel="stylesheet" href="/static/styles.css"></head>\n'
        "<body>\n"
        "  <h1>Atlas Operations Console</h1>\n"
        '  <table id="projects"></table>\n'
        '  <script src="/static/app.js"></script>\n'
        "</body>\n"
        "</html>\n"
    ),
    "app/static/styles.css": (
        ":root { --atlas-bg: #0b1020; }\n"
        "body { font-family: system-ui; background: var(--atlas-bg); }\n"
        "@media (max-width: 600px) { body { font-size: 14px; } }\n"
        ":focus-visible { outline: 2px solid #6ea8fe; }\n"
        "@media (prefers-reduced-motion: reduce) { * { animation: none; } }\n"
    ),
    "app/static/app.js": (
        '"use strict";\n'
        "async function load() {\n"
        '  const response = await fetch("/api/projects");\n'
        "  const body = await response.json();\n"
        '  const table = document.getElementById("projects");\n'
        "  table.innerHTML = body.items\n"
        "    .map((item) => `<tr><td>${item.name}</td><td>${item.status}</td></tr>`)\n"
        '    .join("");\n'
        "}\n"
        "async function setStatus(id, status) {\n"
        "  await fetch(`/api/projects/${id}/status`, {\n"
        '    method: "PATCH",\n'
        '    headers: { "Content-Type": "application/json" },\n'
        "    body: JSON.stringify({ status }),\n"
        "  });\n"
        "  await load();\n"
        "}\n"
        'document.addEventListener("DOMContentLoaded", load);\n'
    ),
    "tests/test_ui_contract.py": (
        "from pathlib import Path\n\n\n"
        "def test_page_loads_its_script_and_styles() -> None:\n"
        '    html = Path("app/static/index.html").read_text(encoding="utf-8")\n'
        '    assert "/static/app.js" in html\n'
        '    assert "/static/styles.css" in html\n\n\n'
        "def test_script_uses_the_existing_api_contract() -> None:\n"
        '    script = Path("app/static/app.js").read_text(encoding="utf-8")\n'
        '    assert "/api/projects" in script\n'
        '    assert "PATCH" in script\n'
    ),
    "README.md": (
        "# Atlas Operations Console\n\n"
        "Run with `uvicorn app.main:app`, test with `pytest`.\n\n"
        "The API contract is unchanged: `GET /api/projects`, "
        "`GET /api/health`, `PATCH /api/projects/{id}/status`.\n"
    ),
}

SCENARIOS = [
    AcceptanceScenarioV1(
        name="listing is exactly the seeded records",
        method="GET",
        path="/api/projects",
        expect_status="2xx",
        expect_json_exact={
            "count": 2,
            "items": [
                {"id": "atlas-1", "name": "Harbor Migration", "status": "healthy"},
                {"id": "atlas-2", "name": "Invoice Intelligence", "status": "at_risk"},
            ],
        },
    ),
    AcceptanceScenarioV1(
        name="health is exact",
        method="GET",
        path="/api/health",
        expect_status="2xx",
        expect_json_exact={"status": "ok", "service": "atlas-projects"},
    ),
    AcceptanceScenarioV1(
        name="patch updates the status",
        method="PATCH",
        path="/api/projects/atlas-2/status",
        body_kind="json",
        body={"status": "healthy"},
        expect_status="2xx",
        expect_json_exact={
            "id": "atlas-2",
            "name": "Invoice Intelligence",
            "status": "healthy",
        },
    ),
    AcceptanceScenarioV1(
        name="unknown project is refused",
        method="PATCH",
        path="/api/projects/atlas-999/status",
        body_kind="json",
        body={"status": "healthy"},
        expect_status="4xx",
    ),
    AcceptanceScenarioV1(
        name="invalid status is refused",
        method="PATCH",
        path="/api/projects/atlas-1/status",
        body_kind="json",
        body={"status": "not-a-status"},
        expect_status="4xx",
    ),
    AcceptanceScenarioV1(
        name="the console page serves",
        method="GET",
        path="/",
        expect_status="2xx",
        expect_contains=["atlas operations console"],
    ),
]

# The captured shape: two runtime slices, then a trailing tests-and-docs
# slice. Six scenarios across the plan -- the count that used to be refused.
THREE_SLICE_PLAN = [
    ProjectVerticalSliceV1(
        name="Console shell and styling",
        outcome="The console page loads and is styled.",
        owned_files=["app/static/index.html", "app/static/styles.css"],
        scenario_names=["the console page serves"],
    ),
    ProjectVerticalSliceV1(
        name="Console data and interaction layer",
        outcome="The page lists the seeded projects and its status action works.",
        owned_files=["app/static/app.js"],
        scenario_names=[
            "listing is exactly the seeded records",
            "health is exact",
            "patch updates the status",
            "unknown project is refused",
            "invalid status is refused",
        ],
    ),
    ProjectVerticalSliceV1(
        name="UI contract tests and documentation",
        outcome="Tests and docs cover the console.",
        owned_files=["tests/test_ui_contract.py", "README.md"],
        scenario_names=[],
    ),
]


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


async def _empty_events() -> AsyncIterator[CodingEventV1]:
    if False:  # pragma: no cover - gives this helper its async-generator shape
        yield CodingEventV1.model_construct()


class ScriptedPlanner:
    """Answers project_plan_files with the captured plan; delegates the rest.

    Only the plan is scripted. Everything else the project loop asks the model
    for -- the exploration steps that must precede planning above all -- is
    still answered by the real deterministic provider, so the run reaches the
    planner the same way a production run does.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls = 0
        self.steps = 0
        self.last_usage = {
            "prompt_tokens": 12_000,
            "completion_tokens": 800,
            "total_tokens": 12_800,
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def project_step(
        self, request: dict[str, Any], *, model_aliases: dict[str, str] | None = None
    ) -> Any:
        """Look around, so the plan gate opens the way it does in production.

        The manifest is deliberately taken AFTER the loop has explored (see
        _PLAN_AFTER_STEPS); the deterministic provider finishes immediately,
        which would end the turn before a plan was ever requested. These are
        ordinary reads of files that really exist in the seeded project.
        """

        from waqil_api.contracts import ProjectAgentStepV1, ProjectToolCallV1

        self.steps += 1
        looked_at = ["app/api.py", "app/store.py", "app/static/app.js"]
        if self.steps <= len(looked_at):
            return ProjectAgentStepV1(
                status="tool",
                tool_call=ProjectToolCallV1(
                    name="read_file", arguments={"path": looked_at[self.steps - 1]}
                ),
            )
        return ProjectAgentStepV1(
            status="complete", response="Exploration finished.", learnings=[]
        )

    async def project_plan_files(
        self, request: dict[str, Any], *, model_aliases: dict[str, str] | None = None
    ) -> ProjectBuildPlanV1:
        del request, model_aliases
        self.calls += 1
        return ProjectBuildPlanV1(
            intent="edit",
            scope="narrow",
            files=list(PLANNED),
            scenarios=list(SCENARIOS),
            slices=list(THREE_SLICE_PLAN),
        )


class BuildingEngine:
    """Writes each slice's owned files into the real disposable mirror."""

    def __init__(self) -> None:
        self.slices: list[list[str]] = []
        self.deleted: list[str] = []
        self.roots: dict[str, Path] = {}

    async def get_info(self) -> CodingEngineInfoV1:
        return CodingEngineInfoV1(
            protocolVersion="1",
            engine="clinecore",
            runtime="fake",
            sdkVersion="0.0.72",
            policyVersion="1",
            allowedTools=["editor", "read_files", "run_check", "search_codebase"],
        )

    async def start_slice(self, request: StartSliceV1) -> SliceResultV1:
        assert request.session_id is not None
        allowed = list(request.allowed_paths)
        self.slices.append(allowed)
        self.roots[request.session_id] = request.workspace_root
        for relative in allowed:
            content = BUILT.get(relative)
            if content is None:
                continue
            target = request.workspace_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        return SliceResultV1(
            sessionId=request.session_id,
            state="completed",
            finishReason="completed",
            summary="Wrote the slice.",
            iterations=4,
            toolCallCount=len(allowed),
            usage=CodingUsageV1(
                inputTokens=30_000, outputTokens=2_000, totalTokens=32_000, requests=1
            ),
            model=CodingModelRouteV1(
                providerId=request.provider.provider_id,
                modelId=request.provider.model_id,
            ),
        )

    async def continue_slice(self, request: ContinueSliceV1) -> SliceResultV1:
        return SliceResultV1(sessionId=request.session_id, state="completed")

    async def restart_with_model(self, request: RestartWithModelV1) -> SliceResultV1:
        assert request.new_session_id is not None
        return SliceResultV1(
            sessionId=request.new_session_id,
            parentSessionId=request.session_id,
            state="completed",
        )

    async def abort_slice(
        self, session_id: str, *, reason: str | None = None
    ) -> SliceResultV1:
        del reason
        return SliceResultV1(sessionId=session_id, state="aborted")

    async def restore_slice(
        self, session_id: str, *, provider: CodingProviderV1 | None = None
    ) -> SliceResultV1:
        del provider
        return SliceResultV1(sessionId=session_id, state="idle")

    def subscribe(
        self, session_id: str, *, after_cursor: int = 0
    ) -> AsyncIterator[CodingEventV1]:
        del session_id, after_cursor
        return _empty_events()

    async def drain_events(
        self, session_id: str, *, after_cursor: int = 0, limit: int = 256
    ) -> list[CodingEventV1]:
        del session_id, after_cursor, limit
        return []

    async def get_usage(self, session_id: str) -> CodingUsageV1:
        del session_id
        return CodingUsageV1()

    async def delete_session(self, session_id: str) -> DeleteSessionResultV1:
        self.deleted.append(session_id)
        return DeleteSessionResultV1(
            sessionId=session_id, deleted=self.roots.pop(session_id, None) is not None
        )

    async def close(self) -> None:
        return None


def _seed(project: Path) -> None:
    project.mkdir(parents=True, exist_ok=True)
    (project / ".gitignore").write_text("__pycache__/\n.env\n", encoding="utf-8")
    (project / "requirements.txt").write_text(
        "fastapi\nhttpx\npytest\n", encoding="utf-8"
    )
    (project / "README.md").write_text("# Atlas Projects\n", encoding="utf-8")
    (project / "app").mkdir(exist_ok=True)
    (project / "app" / "__init__.py").write_text("", encoding="utf-8")
    for relative, content in PROTECTED_SEED.items():
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    static = project / "app" / "static"
    static.mkdir(parents=True, exist_ok=True)
    (static / "index.html").write_text(
        "<!doctype html><html><body><h1>Projects</h1>"
        '<script src="/static/app.js"></script></body></html>\n',
        encoding="utf-8",
    )
    (static / "styles.css").write_text("body { font-family: sans-serif; }\n", "utf-8")
    (static / "app.js").write_text('fetch("/api/projects");\n', encoding="utf-8")
    metis = project / ".metis"
    metis.mkdir(exist_ok=True)
    (metis / "project-context.json").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "project_id": "",
                "project_name": "atlas-projects",
                "root_name": project.name,
                "revision": 1,
                "bootstrap": {
                    "summary": "Deterministic seed.",
                    "architecture": [],
                    "conventions": [],
                    "important_paths": [],
                    "verification": [],
                    "risks": [],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    """Real verification, zero inference.

    The backend is `ollama` so the networkless podman verifier actually runs
    and the acceptance scenarios are really replayed -- under the
    deterministic backend the container is skipped and `ran` is 0, which
    would make "exact acceptance passed" a claim about nothing. Every seam
    that could reach a model is either scripted (project_step,
    project_plan_files) or switched off here, so no inference happens.
    """

    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        repo_root=Path(__file__).resolve().parents[3],
        asset_roots=[tmp_path / "Projects"],
        model_backend="ollama",
        reference_runner_mode="deterministic",
        allow_test_backends=True,
        project_coding_engine="clinecore",
        # This suite exercises the frozen planner/slice path, which is
        # retained behind the flag for rollback and legacy checkpoints.
        project_build_path="planner_slices",
        project_plan_corrections=0,
        project_spec_rewrite=False,
        project_repo_map_enabled=False,
        project_orchestrator_enabled=False,
        cline_sidecar_max_rounds=2,
        project_agent_max_steps=24,
        project_verify_max_runs=6,
        project_sandbox_timeout_seconds=180,
        **overrides,
    )


def _pin_cloud_route(settings: Settings) -> None:
    """Satisfy the local-session gate without launching or calling anything.

    require_ready() short-circuits for a :cloud tag -- a hosted model has no
    weights to load -- so pinning one lets the run start while every model
    seam stays scripted or disabled.
    """

    from waqil_api.contracts import RoleChainEntryV1
    from waqil_api.model_preference import ModelPreferenceStore

    entry = RoleChainEntryV1(provider="local", model="glm-5.2:cloud")
    ModelPreferenceStore(settings).save(
        "split",
        None,
        provider="local",
        role_chains={"planner": [entry], "coder": [entry], "quality": [entry]},
    )


async def _drive(client: httpx.AsyncClient, project_id: str, prompt: str) -> str:
    conversation_id = (
        await client.post("/api/v1/conversations", json={"title": "atlas"})
    ).json()["id"]
    opened = await client.post(
        f"/api/v1/projects/{project_id}/open", json={"mode": "grok_bootstrap_local"}
    )
    assert opened.status_code == 200, opened.text
    accepted = (
        await client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json={
                "content": prompt,
                "project_id": project_id,
                "project_mode": "grok_bootstrap_local",
            },
        )
    ).json()
    assert "run_id" in accepted, accepted
    return accepted["run_id"]


async def _settle(client: httpx.AsyncClient, run_id: str) -> dict[str, Any]:
    async with asyncio.timeout(120):
        while True:
            run = (await client.get(f"/api/v1/runs/{run_id}")).json()
            if run["status"] not in {"queued", "running"}:
                return run
            await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_a_trailing_support_slice_is_normalized_built_verified_and_approved(
    tmp_path: Path,
) -> None:
    project_parent = tmp_path / "Projects"
    project = project_parent / "atlas-projects"
    _seed(project)
    settings = _settings(tmp_path)
    _pin_cloud_route(settings)
    protected_before = {
        path: _sha256((project / path).read_text(encoding="utf-8"))
        for path in PROTECTED_SEED
    }
    app = create_app(settings)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://metis.test"
        ) as client:
            plane = app.state.runtime.control_plane
            planner = ScriptedPlanner(plane.model)
            engine = BuildingEngine()
            # The only two fakes. Normalization, the topology gate, the
            # coordinator, the mirror, verification and approval are all real.
            plane.model = planner
            plane.project_coding.engine = engine

            assets = (await client.post("/api/v1/assets/scan")).json()
            project_id = project_asset_id(assets, project)
            run_id = await _drive(client, project_id, UI_REVAMP_REQUEST)
            run = await _settle(client, run_id)
            assert run["status"] == "awaiting_approval", run.get("last_error")

            events = await plane.database.list_events(run_id)
            kinds = [item.type for item in events]
            payloads: dict[str, Any] = {}
            for item in events:
                payloads.setdefault(item.type, item.payload)

            # 1. Normalization happened, and its evidence names both topologies.
            assert "project.plan_normalized" in kinds
            evidence = payloads["project.plan_normalized"]
            assert [item["name"] for item in evidence["original"]] == [
                "Console shell and styling",
                "Console data and interaction layer",
                "UI contract tests and documentation",
            ]
            assert [item["name"] for item in evidence["normalized"]] == [
                "Console shell and styling",
                "Console data and interaction layer",
            ]
            assert evidence["codes"] == ["support_slice_merged"]
            assert evidence["moved"][0]["paths"] == [
                "tests/test_ui_contract.py",
                "README.md",
            ]
            # The plan was never rejected, and never needed a correction.
            assert "project.plan_rejected" not in kinds
            assert planner.calls == 1

            # 2. Two coding sessions, over the normalized scopes. Compared as
            # sets: the write allowlist crosses the sidecar wire sorted, and
            # its order carries no meaning.
            assert [sorted(item) for item in engine.slices] == [
                sorted(["app/static/index.html", "app/static/styles.css"]),
                sorted(["app/static/app.js", "tests/test_ui_contract.py", "README.md"]),
            ]

            # 3. Both slice boundaries verified, the final one cumulatively.
            checks = [
                item.payload
                for item in events
                if item.type == "project.vertical_slice_checked"
            ]
            assert len(checks) == 2
            assert all(item["errors"] == 0 for item in checks), checks
            assert checks[-1]["final"] is True
            # The merged slice carries all six scenarios: five of its own plus
            # the shell slice's, replayed cumulatively at the final boundary.
            # Every scenario the plan declared really ran at the final
            # boundary -- `final` is what makes verification cumulative, and
            # the ladder below reports how many checks it executed.
            verified = [
                item.payload
                for item in events
                if item.type == "project.staged_verified"
            ]
            assert verified[-1]["errors"] == 0, verified[-1]
            # Cumulative: the final boundary runs strictly more checks than
            # the first, because it replays the whole plan's scenarios rather
            # than only the slice's own.
            assert verified[-1]["ran"] > verified[0]["ran"] > 0, verified
            # And all six scenarios really are in the plan that was verified.
            assert len(SCENARIOS) == 6

            # 4. Every planned file is staged, exactly.
            checkpoint = await plane.checkpointer.aget_tuple(
                plane._config(run["conversation_id"], run_id)
            )
            staged = checkpoint.checkpoint["channel_values"]["project_staged"]
            for path in PLANNED:
                assert path in staged, path
                assert str(staged[path]["content"]) == BUILT[path]

            # 5. Approval is clean, and applies those exact bytes.
            approval = await plane.database.get_pending_approval(run_id)
            assert approval is not None
            assert not approval.blocked_reason
            decided = await client.post(
                f"/api/v1/runs/{run_id}/decisions",
                json={"approval_id": approval.id, "decision": "approve"},
            )
            assert decided.status_code in {200, 202}, decided.text
            async with asyncio.timeout(120):
                while (await client.get(f"/api/v1/runs/{run_id}")).json()[
                    "status"
                ] not in {"completed", "failed", "cancelled"}:
                    await asyncio.sleep(0.05)
            assert (await client.get(f"/api/v1/runs/{run_id}")).json()[
                "status"
            ] == "completed"
            for path in PLANNED:
                assert (project / path).read_text(encoding="utf-8") == BUILT[path]

            # 6. Protected files are byte-identical, and cleanup is complete.
            for path, digest in protected_before.items():
                assert _sha256((project / path).read_text(encoding="utf-8")) == digest
            assert engine.roots == {}
            assert list((tmp_path / "data" / "coding-workspaces").glob("*")) == []


@pytest.mark.asyncio
async def test_plan_only_mode_runs_the_real_gate_and_creates_no_coding_session(
    tmp_path: Path,
) -> None:
    """The cheap planner probe: real plan, real normalization, no build."""

    project_parent = tmp_path / "Projects"
    project = project_parent / "atlas-projects"
    _seed(project)
    settings = _settings(tmp_path, project_plan_only=True)
    _pin_cloud_route(settings)
    app = create_app(settings)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://metis.test"
        ) as client:
            plane = app.state.runtime.control_plane
            planner = ScriptedPlanner(plane.model)
            engine = BuildingEngine()
            plane.model = planner
            plane.project_coding.engine = engine

            assets = (await client.post("/api/v1/assets/scan")).json()
            project_id = project_asset_id(assets, project)
            run_id = await _drive(client, project_id, UI_REVAMP_REQUEST)
            run = await _settle(client, run_id)

            events = await plane.database.list_events(run_id)
            kinds = [item.type for item in events]
            # The real planner ran, and the real normalization and gate ran.
            assert planner.calls == 1, (
                run["status"],
                run.get("last_error"),
                "|".join(kinds),
            )
            assert "project.plan_normalized" in kinds
            assert "project.build_planned" in kinds
            assert "project.plan_only_stop" in kinds
            stop = next(
                item.payload for item in events if item.type == "project.plan_only_stop"
            )
            assert stop["files"] == PLANNED
            assert stop["slices"] == [
                "Console shell and styling",
                "Console data and interaction layer",
            ]

            # And nothing below the seam happened at all.
            assert engine.slices == []
            assert not any(kind.startswith("project.coding_") for kind in kinds)
            assert await plane.project_coding.sessions.for_run(run_id) == []
            assert list((tmp_path / "data" / "coding-workspaces").glob("*")) == []
            assert run["status"] == "completed"
            for path in PLANNED:
                # Nothing was written to the real project either: the two new
                # files do not exist, and the three seeded ones are untouched.
                target = project / path
                assert (
                    not target.is_file()
                    or target.read_text(encoding="utf-8") != BUILT[path]
                ), path
