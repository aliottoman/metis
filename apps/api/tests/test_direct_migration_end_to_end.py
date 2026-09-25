"""One Streamlit-to-FastAPI migration, end to end, through the real API.

Everything in this run is production code except model inference: the real
HTTP surface, the real `cline_direct` control path, the real contract
resolution, the real disposable mirror, the real safe importer, the real
verification ladder, the real approval card and the real materialization. The
only fake is a persistent coding session that writes deterministic bytes and
asks for a check, standing in for a provider.

It exists because the parts were each proved separately and that is not the
same claim. This is the causal chain: the user's own wording resolves the
protections, one session chooses its order and creates files that did not
exist, a check finds a missing dependency mid-session, the same session fixes
it, Metis verifies independently, and only then does an unblocked card appear
and a person's approval put anything on disk.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
import pytest

from waqil_api.coding_contracts import (
    CodingEngineInfoV1,
    CodingEventV1,
    CodingModelRouteV1,
    CodingUsageV1,
    ContinueSliceV1,
    DeleteSessionResultV1,
    HostCheck,
    HostCheckResultV1,
    RestartWithModelV1,
    SliceResultV1,
    StartSliceV1,
)
from waqil_api.coding_engine import CodingSessionStore
from waqil_api.config import Settings
from waqil_api.main import create_app
from waqil_api.project_capability_eval import project_asset_id
from waqil_api.project_coding_engine import ProjectCodingCoordinator


# ── The disposable Streamlit application, shaped like the real asset ───────

STREAMLIT_APP = '''\
"""DHL Swedish Invoice Extractor - Streamlit app."""

import streamlit as st

from excel_writer import build_output_workbook
from extractor import extract_invoice

st.set_page_config(page_title="DHL Invoice Extractor")
uploaded = st.file_uploader("Invoice PDF", type=["pdf"])
if uploaded is not None:
    payload = extract_invoice(uploaded.read())
    workbook, audit = build_output_workbook(payload)
    st.download_button("Download workbook", workbook, "invoices.xlsx")
'''

EXTRACTOR = '''\
"""The OCI vision call. This file must not change."""

VISION_MODEL_ID = "cohere.command-a-vision-07-2025"


def extract_invoice(pdf_bytes: bytes) -> dict:
    return {"invoice": {"invoice_number": "INV-1042"}, "shipments": []}
'''

EXCEL_WRITER = '''\
"""The Excel contract. This file must not change."""

COL_INVOICE_NUMBER = "A"
COL_CURRENCY = "AL"


def build_output_workbook(payload: dict) -> tuple[bytes, list[dict]]:
    number = payload["invoice"]["invoice_number"]
    return (f"workbook:{number}".encode("utf-8"), [{"column": COL_INVOICE_NUMBER}])
'''

SURCHARGE_MAPPING = '''\
"""Surcharge to column resolution. This file must not change."""

SURCHARGE_MAP = {"fraktkostnad": "X", "bransletillagg": "Y"}


def resolve_column(label: str) -> str | None:
    return SURCHARGE_MAP.get(label.strip().casefold())
'''

CONFIG = '''\
"""Central configuration. This file must not change."""

import os

SERVICE_ENDPOINT = os.getenv("OCI_ENDPOINT", "https://inference.example.invalid")
TEMPLATE_PATH = "assets/DHL_Template.xlsx"
'''

PROTECTED = (
    "extractor.py",
    "excel_writer.py",
    "surcharge_mapping.py",
    "assets/DHL_Template.xlsx",
    "config.py",
)

# What the session writes. `app/main.py` declares an upload route without
# python-multipart -- the exact defect a real build shipped into a second
# round -- so the mid-session check has something true to find.
FASTAPI_MAIN_BROKEN = '''\
"""DHL Swedish Invoice Extractor - FastAPI backend."""

from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from excel_writer import build_output_workbook
from extractor import extract_invoice

_STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="DHL Invoice Extractor")
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(_STATIC / "index.html"))


@app.post("/api/extract")
async def extract(file: UploadFile = File(...)) -> dict:
    payload = extract_invoice(await file.read())
    workbook, audit = build_output_workbook(payload)
    return {"rows": len(audit), "bytes": len(workbook)}
'''

INDEX_HTML = """\
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>DHL Invoice Extractor</title>
  <link rel="stylesheet" href="/static/styles.css">
  <script defer src="/static/app.js"></script>
</head>
<body><main class="page"><h1>DHL Invoice Extractor</h1></main></body>
</html>
"""

STYLES_CSS = ":root { --ink: #12131a; }\n.page { color: var(--ink); }\n"
APP_JS = '"use strict";\nconsole.info("DHL Invoice Extractor ready");\n'
REQUIREMENTS_BROKEN = "fastapi>=0.110\nuvicorn>=0.30\n"
REQUIREMENTS_FIXED = "fastapi>=0.110\nuvicorn>=0.30\npython-multipart>=0.0.9\n"

LOGIVITY_REQUEST = (
    "revamp the logivity invoice extractor so it uses the metis design "
    "language. move it off streamlit to fastapi with a static frontend, same "
    "as my other apps. dont touch extractor.py, excel_writer.py, "
    "surcharge_mapping.py, config.py or the DHL template, the extraction and "
    "excel output have to stay exactly the same."
)


def _seed(project: Path) -> None:
    (project / "assets").mkdir(parents=True, exist_ok=True)
    (project / "app.py").write_text(STREAMLIT_APP, encoding="utf-8")
    (project / "extractor.py").write_text(EXTRACTOR, encoding="utf-8")
    (project / "excel_writer.py").write_text(EXCEL_WRITER, encoding="utf-8")
    (project / "surcharge_mapping.py").write_text(SURCHARGE_MAPPING, encoding="utf-8")
    (project / "config.py").write_text(CONFIG, encoding="utf-8")
    (project / "assets" / "DHL_Template.xlsx").write_bytes(b"PK\x03\x04dhl-template")
    (project / "requirements.txt").write_text(
        "streamlit>=1.35\nopenpyxl>=3.1\n", encoding="utf-8"
    )


def _hashes(project: Path) -> dict[str, str]:
    return {
        path: hashlib.sha256((project / path).read_bytes()).hexdigest()
        for path in PROTECTED
    }


# ── The persistent fake session ────────────────────────────────────────────


class PersistentFakeCline:
    """One session that writes, checks, repairs, and remembers doing so.

    It stands in for a provider only. The order below is its own: it inspects
    nothing it was told to inspect first, writes the entrypoint before the
    frontend, and packages last -- and no manifest ever told it to.
    """

    def __init__(self) -> None:
        self.sessions: list[str] = []
        self.roots: dict[str, Path] = {}
        self.edit_order: list[str] = []
        self.checks: list[HostCheck] = []
        self.check_results: list[HostCheckResultV1] = []
        self.deleted: list[str] = []
        self._handler: Any = None

    # The host installs this exactly as it does on the real sidecar client.
    def set_check_handler(self, handler: Any) -> None:
        self._handler = handler

    async def get_info(self) -> CodingEngineInfoV1:
        return CodingEngineInfoV1(
            protocolVersion="1",
            engine="clinecore",
            runtime="fake",
            sdkVersion="0.0.86",
            policyVersion="1",
            allowedTools=["editor", "read_files", "run_check", "search_codebase"],
        )

    def _write(self, root: Path, relative: str, body: str) -> None:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
        self.edit_order.append(relative)

    async def start_slice(self, request: StartSliceV1) -> SliceResultV1:
        session_id = request.session_id or "fake-direct"
        self.sessions.append(session_id)
        root = Path(request.workspace_root)
        self.roots[session_id] = root

        # Its own order: backend first, then the page, then its assets, then
        # packaging. Nothing prescribed this.
        self._write(root, "app/main.py", FASTAPI_MAIN_BROKEN)
        self._write(root, "app/static/index.html", INDEX_HTML)
        self._write(root, "app/static/styles.css", STYLES_CSS)
        self._write(root, "app/static/app.js", APP_JS)
        self._write(root, "requirements.txt", REQUIREMENTS_BROKEN)

        # Mid-session check, in the same turn, before the round returns.
        if self._handler is not None:
            first = await self._handler(session_id, "imports")
            self.checks.append("imports")
            self.check_results.append(first)
            if not first.ok:
                # The finding came back to THIS session, and it repairs the
                # root cause rather than restating it.
                assert any("multipart" in item.detail for item in first.findings), (
                    first.findings
                )
                self._write(root, "requirements.txt", REQUIREMENTS_FIXED)
                second = await self._handler(session_id, "imports")
                self.checks.append("imports")
                self.check_results.append(second)

        return SliceResultV1(
            sessionId=session_id,
            state="completed",
            finishReason="completed",
            summary="Ported to FastAPI; checks clean.",
            iterations=6,
            toolCallCount=8,
            usage=CodingUsageV1(
                inputTokens=41_000, outputTokens=3_100, totalTokens=44_100, requests=6
            ),
            model=CodingModelRouteV1(
                providerId=request.provider.provider_id,
                modelId=request.provider.model_id,
            ),
        )

    async def continue_slice(self, request: ContinueSliceV1) -> SliceResultV1:
        raise AssertionError(
            "the migration completed in one session; a continuation means the "
            "in-session check loop did not do its job"
        )

    async def restart_with_model(self, request: RestartWithModelV1) -> SliceResultV1:
        raise AssertionError("no model fallback should be reachable in this run")

    async def restore_slice(self, session_id: str, provider: Any = None) -> Any:
        raise AssertionError("nothing should need restoring in a clean run")

    async def abort_slice(self, session_id: str) -> SliceResultV1:
        return SliceResultV1(sessionId=session_id, state="aborted")

    async def delete_session(self, session_id: str) -> DeleteSessionResultV1:
        self.deleted.append(session_id)
        return DeleteSessionResultV1(sessionId=session_id, deleted=True)

    def subscribe(self, session_id: str, *, after_cursor: int = 0) -> Any:
        async def _empty() -> AsyncIterator[CodingEventV1]:
            if False:  # pragma: no cover - an empty, closeable stream
                yield  # type: ignore[misc]

        return _empty()

    async def drain_events(
        self, session_id: str, *, after_cursor: int = 0, limit: int = 256
    ) -> list[CodingEventV1]:
        del session_id, after_cursor, limit
        return []

    async def shutdown(self) -> None:
        return None


@pytest.mark.asyncio
async def test_a_streamlit_app_migrates_to_fastapi_in_one_session(
    tmp_path: Path,
) -> None:
    projects_root = tmp_path / "Projects"
    project = projects_root / "logivity"
    project.mkdir(parents=True)
    _seed(project)
    protected_before = _hashes(project)
    streamlit_before = (project / "app.py").read_bytes()

    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        data_dir=tmp_path / "data",
        repo_root=tmp_path,
        asset_roots=[projects_root],
        model_backend="deterministic",
        allow_test_backends=True,
        project_coding_engine="clinecore",
        project_build_path="cline_direct",
        project_slices_enabled=False,
        project_run_check_budget=6,
        reference_runner_mode="deterministic",
    )
    app = create_app(settings)
    engine = PersistentFakeCline()

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://metis.test"
        ) as client:
            plane = app.state.runtime.control_plane
            # Fake ONLY inference: the coordinator, importer, verifier,
            # approval and materialization below are all production.
            plane.project_coding = ProjectCodingCoordinator(
                settings,
                engine,
                CodingSessionStore(plane.database),
                plane.projects,
                plane.events,
            )

            assets = (await client.post("/api/v1/assets/scan")).json()
            project_id = project_asset_id(assets, project)
            await client.post(
                f"/api/v1/projects/{project_id}/open",
                json={"mode": "grok_bootstrap_local"},
            )
            conversation_id = (
                await client.post(
                    "/api/v1/conversations", json={"title": "Logivity migration"}
                )
            ).json()["id"]

            accepted = (
                await client.post(
                    f"/api/v1/conversations/{conversation_id}/messages",
                    json={
                        "content": LOGIVITY_REQUEST,
                        "project_id": project_id,
                        "project_mode": "grok_bootstrap_local",
                    },
                )
            ).json()
            run_id = accepted["run_id"]

            deadline = asyncio.get_running_loop().time() + 120
            while True:
                run = (await client.get(f"/api/v1/runs/{run_id}")).json()
                if run["status"] not in {"queued", "running"}:
                    break
                assert asyncio.get_running_loop().time() < deadline, (
                    "run did not settle"
                )
                await asyncio.sleep(0.05)

            events = await plane.database.list_events(run_id)
            kinds = [item.type for item in events]

            # ── the contract, resolved from the user's own words ───────────
            contract = next(
                item.payload
                for item in events
                if item.type == "project.direct_contract"
            )
            assert sorted(contract["protected_files"]) == sorted(PROTECTED)
            assert contract["unresolved"] == []
            assert contract["writable_roots"] == ["."]

            # ── none of the retired machinery ran ─────────────────────────
            for retired in (
                "project.build_planned",
                "project.plan_normalized",
                "project.plan_synthesized",
                "project.plan_rejected",
                "project.vertical_slice_checked",
                "project.repair_routed",
                "project.agent_step",
            ):
                assert retired not in kinds, retired

            # ── one session, its own order, files that did not exist ───────
            assert len(set(engine.sessions)) == 1, engine.sessions
            assert engine.edit_order[:5] == [
                "app/main.py",
                "app/static/index.html",
                "app/static/styles.css",
                "app/static/app.js",
                "requirements.txt",
            ]
            for created in ("app/main.py", "app/static/index.html"):
                assert not (project / created).exists(), (
                    f"{created} must not exist before approval"
                )

            # ── the check found the dependency, in session, and it repaired ─
            assert engine.checks == ["imports", "imports"], engine.checks
            assert engine.check_results[0].ok is False
            assert any(
                "multipart" in item.detail for item in engine.check_results[0].findings
            )
            assert engine.check_results[1].ok is True
            assert engine.edit_order[-1] == "requirements.txt"
            check_events = [
                item.payload
                for item in events
                if item.type == "project.check_requested"
            ]
            assert [item["ok"] for item in check_events] == [False, True]

            # ── nothing on disk yet ───────────────────────────────────────
            assert _hashes(project) == protected_before
            assert (project / "app.py").read_bytes() == streamlit_before

            # ── independent verification, and an unblocked card ───────────
            assert run["status"] == "awaiting_approval", run.get("last_error")
            approval = await plane.database.get_pending_approval(run_id)
            assert approval is not None
            assert approval.kind == "project_apply_build"
            # Unblocked: an Approve button exists because the host verified it.
            assert not approval.blocked_reason, approval.blocked_reason

            checkpoint = await plane.checkpointer.aget_tuple(
                plane._config(conversation_id, run_id)
            )
            staged = dict(
                (checkpoint.checkpoint.get("channel_values") or {}).get(
                    "project_staged"
                )
                or {}
            )
            assert set(staged) & set(PROTECTED) == set()
            assert "python-multipart" in str(staged["requirements.txt"]["content"])

            # ── approval is what puts anything on disk ────────────────────
            decided = await client.post(
                f"/api/v1/runs/{run_id}/decisions",
                json={"approval_id": approval.id, "decision": "approve"},
            )
            assert decided.status_code in (200, 202), decided.text
            deadline = asyncio.get_running_loop().time() + 120
            while (await client.get(f"/api/v1/runs/{run_id}")).json()["status"] in {
                "queued",
                "running",
                "awaiting_approval",
            }:
                assert asyncio.get_running_loop().time() < deadline, "apply hung"
                await asyncio.sleep(0.05)

            # Only the intended application files materialized.
            assert (project / "app" / "main.py").is_file()
            assert (project / "app" / "static" / "index.html").is_file()
            assert (project / "app" / "static" / "styles.css").is_file()
            assert (project / "app" / "static" / "app.js").is_file()
            assert "python-multipart" in (project / "requirements.txt").read_text(
                encoding="utf-8"
            )

            # The protected files, byte for byte, after a completed migration.
            assert _hashes(project) == protected_before
            # And the extraction/Excel behaviour they carry is unchanged.
            assert 'VISION_MODEL_ID = "cohere.command-a-vision-07-2025"' in (
                project / "extractor.py"
            ).read_text(encoding="utf-8")
            assert 'COL_CURRENCY = "AL"' in (project / "excel_writer.py").read_text(
                encoding="utf-8"
            )
            assert '"fraktkostnad": "X"' in (
                project / "surcharge_mapping.py"
            ).read_text(encoding="utf-8")

            # ── cleanup ───────────────────────────────────────────────────
            sessions = await plane.project_coding.sessions.for_run(run_id)
            assert sessions, "the run must leave a durable session record"
            mirrors = list((settings.coding_workspace_dir).glob("*"))
            assert mirrors == [], f"a mirror was left behind: {mirrors}"
