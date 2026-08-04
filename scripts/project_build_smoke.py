"""Drive one real project build end to end against the pinned local model.

Deterministic-provider tests cannot see how a real model shapes its replies —
every defect in this area got through them — so a build fix is only verified
once a live model has staged files through the full HTTP and approval path.
Runs against a throwaway project and data directory; the real one is untouched.
"""

from __future__ import annotations

import ast
import json
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

from fastapi.testclient import TestClient

from waqil_api.config import Settings
from waqil_api.main import create_app

REQUEST = (
    'Build out this project from scratch: "Agent Showcase" — a multi-agent demo '
    "using the OCI Generative AI Responses API, with document extraction and a web UI. "
    "Create a FastAPI backend (Python 3.13) serving the API and a static frontend from "
    "app/static/. Create app/agents/base.py with one small Agent abstraction, and three "
    "agents: app/agents/planner.py, app/agents/extractor.py, app/agents/synthesizer.py. "
    "Create app/orchestrator.py running plan then extract then synthesize. Create "
    "app/extraction.py using pypdf for PDF and a plain read for TXT. Create app/config.py "
    "reading every OCI setting from environment variables. Also create app/main.py, "
    "requirements.txt and README.md. Write a one-line comment above every function."
)


def _timeline(data_dir: Path) -> list[tuple[str, str]]:
    """Every recorded run event, read straight from the run's own database."""
    connection = sqlite3.connect(f"file:{data_dir / 'waqil.db'}?mode=ro", uri=True)
    try:
        return [
            (str(kind), str(payload))
            for kind, payload in connection.execute(
                "select type, payload_json from run_events order by rowid"
            )
        ]
    finally:
        connection.close()


def _drive(client: TestClient, run_id: str, until: set[str], deadline: float) -> dict:
    """Poll a run until it reaches one of the wanted states, or time out."""
    while time.monotonic() < deadline:
        run = client.get(f"/api/v1/runs/{run_id}").json()
        if run["status"] in until:
            return run
        time.sleep(0.5)
    raise SystemExit(f"run never reached {until}")


def main() -> None:
    """Stage a real multi-file build, approve it, and check what reached disk."""
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen3-coder:30b"
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 1800.0
    root = Path(tempfile.mkdtemp(prefix="metis-build-smoke-"))
    project = root / "Projects" / "agent-showcase"
    project.mkdir(parents=True)
    # Something the build will not plan to write, so "from scratch" really is:
    # seeding a README the model then wants to create is a collision the test
    # invents rather than one the product has.
    (project / ".gitignore").write_text("__pycache__/\n.env\n", encoding="utf-8")

    data_dir = root / "data"
    data_dir.mkdir()
    # Point the session at the already-loaded model so the readiness gate passes
    # without this smoke run launching or evicting anything.
    (data_dir / "model_session.json").write_text(
        json.dumps(
            {"selected_model": model, "idle_timeout_seconds": 86400, "context_window": 32768}
        ),
        encoding="utf-8",
    )
    settings = Settings(
        _env_file=None,
        data_dir=data_dir,
        # The real repository, so the reviewed sandbox wrapper resolves and the
        # build's staged changeset is actually executed. Only code is read from
        # here; the project and the data directory are still throwaway.
        repo_root=Path(__file__).resolve().parents[1],
        asset_roots=[root / "Projects"],
        model_backend="ollama",
        reference_runner_mode="deterministic",
        allow_test_backends=True,
        planner_model=model,
        coder_model=model,
        quality_model=model,
    )

    started = time.monotonic()
    deadline = started + budget
    with TestClient(create_app(settings)) as client:
        session = client.get("/api/v1/model-session").json()
        print(f"model session: {session['state']} / {session['selected_model']}")
        if session["state"] not in {"ready", "busy"}:
            print(f"launching {model}…")
            launched = client.post(
                "/api/v1/model-session/launch",
                json={
                    "model": model,
                    "idle_timeout_seconds": 1800,
                    "context_window": 32768,
                },
            )
            if launched.status_code != 200:
                raise SystemExit(f"could not launch {model}: {launched.text[:300]}")
            print(f"model session: {launched.json()['state']}")

        project_id = client.post("/api/v1/assets/scan").json()[0]["id"]
        # Seed the bootstrap so opening the project does not call OCI. This run
        # is about the local build loop; the first-open summary is a cloud call
        # that has nothing to do with what is being verified here.
        metis = project / ".metis"
        metis.mkdir(exist_ok=True)
        (metis / "project-context.json").write_text(
            json.dumps(
                {
                    "schema_version": "1",
                    "project_id": project_id,
                    "project_name": "agent-showcase",
                    "root_name": project.name,
                    "revision": 1,
                    "bootstrap": {
                        "summary": "An empty project that a build is about to fill.",
                        "architecture": [],
                        "conventions": [],
                        "important_paths": [],
                        "verification": [],
                        "risks": [],
                    },
                }
            ),
            encoding="utf-8",
        )
        opened = client.post(
            f"/api/v1/projects/{project_id}/open", json={"mode": "grok_bootstrap_local"}
        )
        if opened.status_code != 200:
            raise SystemExit(f"could not open the project: {opened.text[:400]}")
        conversation_id = client.post("/api/v1/conversations", json={}).json()["id"]
        sent = client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json={
                "content": REQUEST,
                "project_id": project_id,
                "project_mode": "grok_bootstrap_local",
            },
        )
        if sent.status_code >= 300 or "run_id" not in sent.json():
            raise SystemExit(f"could not start the run: {sent.status_code} {sent.text[:400]}")
        run_id = sent.json()["run_id"]
        print(f"run {run_id} started")

        run = _drive(client, run_id, {"awaiting_approval", "completed", "failed"}, deadline)
        # Read the timeline from the database, not GET /runs/{id}/events — that
        # endpoint is an SSE stream that stays open until the run terminates,
        # and this run is waiting on the approval this script has not sent yet.
        timeline = _timeline(data_dir)
        for kind, payload in timeline:
            if kind in {
                "project.build_planned",
                "project.step_blocked",
                "project.staged_verified",
            }:
                print(f"  {kind}: {payload[:400]}")
        # Refused writes are the loop's wasted motion — one build spent 43
        # create_file calls to produce 11 files — so the count is part of the
        # result, not something to reconstruct from a database afterwards.
        results = [
            json.loads(payload)
            for kind, payload in timeline
            if kind == "project.tool_result"
        ]
        writes = [item for item in results if item.get("tool") in {"create_file", "apply_patch"}]
        refused = [item for item in writes if not item.get("ok")]
        print(
            f"  tool calls: {len(results)} ({len(writes)} writes, "
            f"{len(refused)} refused, {len(results) - len(writes)} reads)"
        )

        print(f"status after {time.monotonic() - started:.0f}s: {run['status']}")
        if run["status"] != "awaiting_approval":
            messages = client.get(
                f"/api/v1/conversations/{conversation_id}/messages"
            ).json()
            print("final message:", messages[-1]["content"][:1500])
            raise SystemExit("the build never reached an approval — nothing was staged")

        approval = next(
            item["approval"]
            for item in client.get("/api/v1/runs?status=awaiting_approval").json()
            if item["run"]["id"] == run_id
        )
        print("\n--- approval card ---")
        print(approval["summary"][:2500])
        client.post(
            f"/api/v1/runs/{run_id}/decisions",
            json={"approval_id": approval["id"], "decision": "approve"},
        )
        _drive(client, run_id, {"completed", "failed"}, deadline)

    written = sorted(
        path
        for path in project.rglob("*")
        if path.is_file() and ".metis" not in path.parts and path.name != ".gitignore"
    )
    print(f"\n--- {len(written)} file(s) on disk after {time.monotonic() - started:.0f}s ---")
    broken = 0
    for path in written:
        note = ""
        if path.suffix == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8"))
                note = "parses"
            except SyntaxError as exc:
                note = f"SYNTAX ERROR line {exc.lineno}"
                broken += 1
        print(f"  {path.relative_to(project)!s:36s} {path.stat().st_size:6d} bytes  {note}")
    print(f"\n{len(written)} written, {broken} broken")
    shutil.rmtree(root, ignore_errors=True)
    if not written or broken:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
