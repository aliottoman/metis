"""The whole routed-repair path, through the public API, on one event loop.

A prior live micro-canary proved the model half of an attributed repair --
the sidecar read one file, edited one file, and stopped -- but the harness
that drove it ran ControlPlane on a different event loop than the app's, so
the host never collected the result. That left the host half of the path
(collect, import, checkpoint, verify, approve, materialize, clean up)
without an end-to-end proof.

This closes that gap offline. Everything runs inside one app lifespan on one
loop: the run is submitted through the real HTTP API, the coding engine is
the only fake, and the assertions follow the durable record -- run events,
the graph checkpoint, the approval row, and the bytes on disk -- rather than
anything reconstructed after the fact.
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
from waqil_api.contracts import ApprovalRequestV1, RunStatus
from waqil_api.main import create_app
from waqil_api.project_capability_eval import project_asset_id


# ── The deterministic fixture: two verified slices, one seed-value defect ──

DB_PY_DEFECTIVE = '''\
"""SQLite persistence for the Projects Dashboard."""

from __future__ import annotations

SEED_PROJECTS = [
    ("proj-1", "Placeholder Project", "healthy"),
    ("proj-2", "Invoice Intelligence", "at_risk"),
]
'''

DB_PY_REPAIRED = DB_PY_DEFECTIVE.replace("Placeholder Project", "Harbor Migration")

MAIN_PY = '''\
"""Projects Dashboard entrypoint."""

from __future__ import annotations

from app.db import SEED_PROJECTS


def list_projects() -> list[dict[str, str]]:
    return [
        {"id": identifier, "name": name, "status": status}
        for identifier, name, status in SEED_PROJECTS
    ]


def create_app() -> dict[str, list[dict[str, str]]]:
    return {"projects": list_projects()}


app = create_app()
'''

TEST_PY = '''\
"""Workflow tests for the Projects Dashboard."""

from __future__ import annotations

from app.main import list_projects


def test_listing_names_the_seeded_projects() -> None:
    names = {project["name"] for project in list_projects()}
    assert "Harbor Migration" in names
'''

APP_JS = 'fetch("/api/projects").then((response) => response.json());\n'
INDEX_HTML = (
    "<!doctype html>\n"
    '<html lang="en">\n'
    "<head><title>Projects Dashboard</title></head>\n"
    "<body>\n"
    '  <ul id="project-list"></ul>\n'
    '  <script src="/static/app.js"></script>\n'
    "</body>\n"
    "</html>\n"
)

PLANNED = [
    "app/db.py",
    "app/main.py",
    "tests/test_workflows.py",
    "app/static/app.js",
    "app/static/index.html",
]

DEFECTIVE_CONTENT = {
    "app/db.py": DB_PY_DEFECTIVE,
    "app/main.py": MAIN_PY,
    "tests/test_workflows.py": TEST_PY,
    "app/static/app.js": APP_JS,
    "app/static/index.html": INDEX_HTML,
}

# Slice 2 owns the UI and integrates only the entrypoint, so the failing file
# below belongs to slice 1 alone and attribution is unambiguous.
SLICES = [
    {
        "name": "SQLite-backed persistence",
        "outcome": "The listing reads the seeded projects.",
        "files": PLANNED[:3],
        "owned_files": PLANNED[:3],
        "integration_files": [],
        "scenario_names": ["listing names the seeded projects"],
    },
    {
        "name": "Status workflow and UI",
        "outcome": "The dashboard renders the project list.",
        "files": ["app/static/app.js", "app/static/index.html", "app/main.py"],
        "owned_files": ["app/static/app.js", "app/static/index.html"],
        "integration_files": ["app/main.py"],
        "scenario_names": ["dashboard page serves the UI"],
    },
]

AUTHORIZED = list(SLICES[0]["files"])
UNAFFECTED = [path for path in PLANNED if path not in set(AUTHORIZED)]

ACCEPTANCE_FINDING = {
    "path": "app/db.py",
    "error": (
        "pytest tests/test_workflows.py::test_listing_names_the_seeded_projects "
        "failed against the staged project: AssertionError: "
        "'Harbor Migration' not in {'Invoice Intelligence', 'Placeholder Project'}"
    ),
    "severity": "error",
    "kind": "test",
    "rung": "runtime",
}


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _staged(content: dict[str, str]) -> dict[str, dict[str, Any]]:
    return {
        path: {
            "content": value,
            "bytes": len(value.encode("utf-8")),
            "origin": "patch",
            "base_sha256": _sha256(DEFECTIVE_CONTENT[path]),
        }
        for path, value in content.items()
    }


async def _empty_events() -> AsyncIterator[CodingEventV1]:
    if False:  # pragma: no cover - gives this helper its async-generator shape
        yield CodingEventV1.model_construct()


class RepairingEngine:
    """A sidecar that performs exactly the one edit a real repair would.

    It writes into the real disposable mirror the coordinator prepared, so
    every host-side step after the engine call -- diff, scope enforcement,
    import, checkpoint, verification, approval, materialization -- is the
    production path operating on real bytes.
    """

    def __init__(self) -> None:
        self.start_requests: list[StartSliceV1] = []
        self.deleted: list[str] = []
        self.roots: dict[str, Path] = {}
        self.write: tuple[str, str] | None = ("app/db.py", DB_PY_REPAIRED)

    async def get_info(self) -> CodingEngineInfoV1:
        return CodingEngineInfoV1(
            protocolVersion="1",
            engine="clinecore",
            runtime="fake",
            sdkVersion="0.0.86",
            policyVersion="1",
            allowedTools=["editor", "read_files", "run_check", "search_codebase"],
        )

    async def start_slice(self, request: StartSliceV1) -> SliceResultV1:
        assert request.session_id is not None
        self.start_requests.append(request)
        self.roots[request.session_id] = request.workspace_root
        if self.write is not None:
            relative, content = self.write
            target = request.workspace_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        return SliceResultV1(
            sessionId=request.session_id,
            state="completed",
            finishReason="completed",
            summary="Repaired the seeded project name.",
            iterations=3,
            toolCallCount=2,
            usage=CodingUsageV1(
                inputTokens=13_758, outputTokens=598, totalTokens=14_356, requests=1
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


def _seed_project(project: Path) -> None:
    project.mkdir(parents=True, exist_ok=True)
    (project / ".gitignore").write_text("__pycache__/\n.env\n", encoding="utf-8")
    (project / "requirements.txt").write_text("pytest\n", encoding="utf-8")
    (project / "README.md").write_text("# Projects Dashboard\n", encoding="utf-8")
    for relative, content in DEFECTIVE_CONTENT.items():
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


async def _seed_blocked_build(
    plane: Any, conversation_id: str, project_id: str
) -> tuple[str, dict[str, Any]]:
    """Park a fully verified build at a blocked approval, with no model call.

    This is the state a build reaches when every slice verified and only the
    final acceptance probe failed -- exactly what the live two-slice canary
    produced, reconstructed deterministically.
    """

    aliases = {
        "_project_id": project_id,
        "_coding_engine": "clinecore",
        "_provider": "local",
        "coder": "fake-coder",
        "_chain_coder": json.dumps([{"provider": "local", "model": "fake-coder"}]),
    }
    message = await plane.database.add_message(
        conversation_id, "user", "Build the projects dashboard."
    )
    run = await plane.database.create_run(
        conversation_id,
        message.id,
        graph_schema_version="1",
        model_aliases=aliases,
    )
    staged = _staged(DEFECTIVE_CONTENT)
    values = {
        "run_id": run.id,
        "conversation_id": conversation_id,
        "model_aliases": aliases,
        "project_staged": staged,
        "project_planned_files": list(PLANNED),
        "project_planned_slices": [dict(item) for item in SLICES],
        "project_planned_scenarios": [],
        "project_verified_prefix": len(PLANNED),
        "project_build_intent": "edit",
        "project_build_scope": "narrow",
        "project_context": {"manifest": {"file_tree": list(PLANNED)}},
        "project_repair_context": {
            "files": AUTHORIZED,
            "findings": [ACCEPTANCE_FINDING],
            "finding_signature": "seed",
            "unchanged_verifications": 0,
        },
    }
    await plane.checkpointer.aput(
        plane._config(conversation_id, run.id),
        {
            "v": 1,
            "id": run.id,
            "ts": "2026-08-11T00:00:00+00:00",
            "channel_values": values,
            "channel_versions": {},
            "versions_seen": {},
        },
        {"source": "update", "step": 1, "parents": {}},
        {},
    )
    await plane.database.create_approval(
        ApprovalRequestV1(
            id=f"apr_{run.id}",
            run_id=run.id,
            action_id="project_apply_build",
            kind="project_apply_build",
            title="Apply the staged build",
            summary="The acceptance probe failed on the seeded project name.",
            risk_level="R2",
            input_digest=_sha256(json.dumps(sorted(staged))),
            blocked_reason="acceptance: the listing omits Harbor Migration",
        )
    )
    await plane.database.set_run_status(run.id, RunStatus.AWAITING_APPROVAL)
    return run.id, staged


@pytest.mark.asyncio
async def test_a_routed_repair_is_collected_checkpointed_verified_and_applied(
    tmp_path: Path,
) -> None:
    project_parent = tmp_path / "Projects"
    project = project_parent / "projects-dashboard"
    _seed_project(project)
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        repo_root=Path(__file__).resolve().parents[3],
        asset_roots=[project_parent],
        model_backend="deterministic",
        reference_runner_mode="deterministic",
        allow_test_backends=True,
        project_coding_engine="clinecore",
        # This suite exercises the frozen planner/slice path, which is
        # retained behind the flag for rollback and legacy checkpoints.
        project_build_path="planner_slices",
        cline_sidecar_max_rounds=1,
        project_agent_max_steps=8,
    )
    app = create_app(settings)

    # One lifespan, one loop: the app's services, the HTTP calls, and this
    # test's own awaits all live here. Driving ControlPlane from a second loop
    # is what made the earlier harness hang for 900 seconds.
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://metis.test"
        ) as client:
            plane = app.state.runtime.control_plane
            assert plane is not None
            engine = RepairingEngine()
            # The ONLY fake. Everything downstream of it is production code.
            plane.project_coding.engine = engine

            assets = (await client.post("/api/v1/assets/scan")).json()
            project_id = project_asset_id(assets, project)
            conversation_id = (
                await client.post(
                    "/api/v1/conversations", json={"title": "routed repair"}
                )
            ).json()["id"]
            await client.post(
                f"/api/v1/projects/{project_id}/open",
                json={"mode": "grok_bootstrap_local"},
            )
            prior_run, before_staged = await _seed_blocked_build(
                plane, conversation_id, project_id
            )
            unaffected_before = {
                path: _sha256(str(before_staged[path]["content"]))
                for path in UNAFFECTED
            }

            # ── The follow-up that must repair one slice, not rebuild ──────
            accepted = (
                await client.post(
                    f"/api/v1/conversations/{conversation_id}/messages",
                    json={
                        "content": "The listing must name Harbor Migration.",
                        "project_id": project_id,
                        "project_mode": "grok_bootstrap_local",
                    },
                )
            ).json()
            run_id = accepted["run_id"]

            async def settled() -> dict[str, Any]:
                # A strict bound: the earlier harness defect showed up as an
                # unbounded wait, so this test refuses to hang either.
                async with asyncio.timeout(60):
                    while True:
                        run = (await client.get(f"/api/v1/runs/{run_id}")).json()
                        if run["status"] not in {"queued", "running"}:
                            return run
                        await asyncio.sleep(0.05)

            run = await settled()
            assert run["status"] == "awaiting_approval", run.get("last_error")

            # The durable record, not the SSE stream: /runs/{id}/events is a
            # long-lived stream that deliberately stays open while a run waits
            # at its approval card, so a test that read it would never return.
            events = await plane.database.list_events(run_id)
            kinds = [item.type for item in events]
            payloads = {item.type: item.payload for item in events}

            # 1. The blocked overlay was carried from the prior run.
            assert "project.staged_resumed" in kinds
            assert payloads["project.staged_resumed"]["from_run"] == prior_run
            assert payloads["project.staged_resumed"]["files"] == sorted(PLANNED)

            # 2. Exactly one slice was routed, with its exact authorized scope.
            assert "project.repair_routed" in kinds
            routed = payloads["project.repair_routed"]
            assert routed["target"] == "app/db.py"
            assert routed["slice"] == "SQLite-backed persistence"
            assert routed["attribution"] == "owned_file"
            assert routed["authorized_paths"] == AUTHORIZED

            # 3. The engine ran once, over that scope alone.
            assert len(engine.start_requests) == 1
            assert sorted(engine.start_requests[0].allowed_paths) == sorted(AUTHORIZED)

            # 4. The host collected the result and imported the changed path.
            assert "project.coding_round" in kinds
            assert payloads["project.coding_round"]["changed_paths"] == ["app/db.py"]
            assert payloads["project.coding_round"]["accepted"] is True

            # 5. The repaired slice was checked, and verification ran after
            #    the write rather than before it.
            assert "project.vertical_slice_checked" in kinds
            checked = payloads["project.vertical_slice_checked"]
            assert checked["repair"] is True
            assert checked["repair_target"] == "app/db.py"
            assert checked["errors"] == 0, payloads.get("project.staged_verified")
            assert "project.staged_verified" in kinds
            assert kinds.index("project.coding_round") < kinds.index(
                "project.staged_verified"
            )

            # 6. The graph checkpoint holds the repaired bytes, and only the
            #    attributed file moved.
            checkpoint = await plane.checkpointer.aget_tuple(
                plane._config(conversation_id, run_id)
            )
            overlay = checkpoint.checkpoint["channel_values"]["project_staged"]
            assert str(overlay["app/db.py"]["content"]) == DB_PY_REPAIRED
            for path, digest in unaffected_before.items():
                assert _sha256(str(overlay[path]["content"])) == digest
            assert not checkpoint.checkpoint["channel_values"].get(
                "project_repair_slice"
            )

            # 7. Approving applies those exact bytes to the real project.
            approval = await plane.database.get_pending_approval(run_id)
            assert approval is not None
            assert not approval.blocked_reason
            decided = await client.post(
                f"/api/v1/runs/{run_id}/decisions",
                json={"approval_id": approval.id, "decision": "approve"},
            )
            assert decided.status_code in {200, 202}, decided.text
            async with asyncio.timeout(60):
                while (await client.get(f"/api/v1/runs/{run_id}")).json()[
                    "status"
                ] not in {"completed", "failed", "cancelled"}:
                    await asyncio.sleep(0.05)
            assert (await client.get(f"/api/v1/runs/{run_id}")).json()[
                "status"
            ] == "completed"
            assert (project / "app/db.py").read_text(encoding="utf-8") == DB_PY_REPAIRED
            for path in UNAFFECTED:
                assert (project / path).read_text(
                    encoding="utf-8"
                ) == DEFECTIVE_CONTENT[path]

            # 8. No session or mirror is left behind.
            assert engine.deleted, "the repair session was never released"
            assert engine.roots == {}
            remaining = list((settings.data_dir / "coding-workspaces").glob("*"))
            assert remaining == [], remaining


@pytest.mark.asyncio
async def test_a_repair_that_writes_nothing_leaves_the_overlay_and_does_not_apply(
    tmp_path: Path,
) -> None:
    """The convergence gate, proven through the same public path.

    A session that settles cleanly without changing an authorized byte must
    not be dressed up as a fixed build: the previously verified overlay has
    to survive untouched and the real project must stay as it was.
    """

    project_parent = tmp_path / "Projects"
    project = project_parent / "projects-dashboard"
    _seed_project(project)
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        repo_root=Path(__file__).resolve().parents[3],
        asset_roots=[project_parent],
        model_backend="deterministic",
        reference_runner_mode="deterministic",
        allow_test_backends=True,
        project_coding_engine="clinecore",
        # This suite exercises the frozen planner/slice path, which is
        # retained behind the flag for rollback and legacy checkpoints.
        project_build_path="planner_slices",
        cline_sidecar_max_rounds=1,
        project_agent_max_steps=8,
    )
    app = create_app(settings)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://metis.test"
        ) as client:
            plane = app.state.runtime.control_plane
            engine = RepairingEngine()
            engine.write = None  # settles cleanly, changes nothing
            plane.project_coding.engine = engine

            assets = (await client.post("/api/v1/assets/scan")).json()
            project_id = project_asset_id(assets, project)
            conversation_id = (
                await client.post(
                    "/api/v1/conversations", json={"title": "empty repair"}
                )
            ).json()["id"]
            await client.post(
                f"/api/v1/projects/{project_id}/open",
                json={"mode": "grok_bootstrap_local"},
            )
            await _seed_blocked_build(plane, conversation_id, project_id)

            accepted = (
                await client.post(
                    f"/api/v1/conversations/{conversation_id}/messages",
                    json={
                        "content": "The listing must name Harbor Migration.",
                        "project_id": project_id,
                        "project_mode": "grok_bootstrap_local",
                    },
                )
            ).json()
            run_id = accepted["run_id"]
            async with asyncio.timeout(60):
                while (await client.get(f"/api/v1/runs/{run_id}")).json()["status"] in {
                    "queued",
                    "running",
                }:
                    await asyncio.sleep(0.05)

            # The defective bytes are still the bytes on disk: nothing was
            # applied on the strength of a repair that repaired nothing.
            assert (project / "app/db.py").read_text(
                encoding="utf-8"
            ) == DB_PY_DEFECTIVE
