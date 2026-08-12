"""The optional pre-launch build step: parsing, execution, trust, persistence.

An asset may declare `launch.build` — one argv, or a list of argv steps — that
Metis runs on the host before the launch command, so an edited-but-uncompiled
frontend is built before it is served. These tests pin the parser (a broken
build must poison the launch, never silently serve a stale artifact), the
execution (runs before launch, streams to the same logs, aborts on failure or
timeout), and that the build is part of the trust fingerprint and survives a
catalog reload.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

import waqil_api.asset_library as asset_library
from waqil_api.asset_library import manifest_metadata_from_body
from waqil_api.config import Settings
from waqil_api.main import create_app


def _client(settings: Settings, root: Path) -> TestClient:
    configured = settings.model_copy(update={"asset_roots": [root]})
    return TestClient(create_app(configured))


def _write_manifest(project: Path, body: dict) -> None:
    manifest_dir = project / ".metis"
    manifest_dir.mkdir()
    (manifest_dir / "asset.json").write_text(json.dumps(body), encoding="utf-8")


# ── Parsing ──────────────────────────────────────────────────────────────────


def test_build_parses_as_one_argv_or_a_list_of_steps() -> None:
    single = manifest_metadata_from_body(
        {"launch": {"command": ["python", "app.py"], "build": ["npm", "run", "build"]}}
    )
    assert single.command == ("python", "app.py")
    assert single.build == (("npm", "run", "build"),)

    many = manifest_metadata_from_body(
        {
            "launch": {
                "command": ["python", "app.py"],
                "build": [["npm", "ci"], ["npm", "run", "build"]],
            }
        }
    )
    assert many.build == (("npm", "ci"), ("npm", "run", "build"))


def test_absent_or_empty_build_is_valid_and_the_command_still_parses() -> None:
    # Absent and an explicit empty list are equivalent: no build, still launchable.
    # (The catalog persists a buildless asset as `"build": []`, so this parse must
    # stay valid or trust would evaporate on reload.)
    for launch in (
        {"command": ["python", "app.py"]},
        {"command": ["python", "app.py"], "build": []},
    ):
        meta = manifest_metadata_from_body({"launch": launch})
        assert meta.command == ("python", "app.py")
        assert meta.build == ()


def test_a_malformed_build_poisons_the_launch_command() -> None:
    # A present-but-invalid build must leave nothing launchable behind, or the
    # asset would start and silently serve a stale, un-rebuilt artifact.
    malformed = (
        [123],  # a step that is not an argv list
        [["npm", 1]],  # a non-string token
        [["npm", "run\nbuild"]],  # a control character in a token
        [[]],  # an empty argv
        [["only", "one"], "npm run build"],  # a mix of step shapes
        [[f"tok{i}"] for i in range(9)],  # more steps than the cap
    )
    for bad in malformed:
        meta = manifest_metadata_from_body(
            {"launch": {"command": ["python", "app.py"], "build": bad}}
        )
        assert meta.command is None, bad
        assert meta.build == ()


# ── Execution ────────────────────────────────────────────────────────────────


def test_build_runs_before_launch_and_streams_to_logs(
    settings: Settings, tmp_path: Path
) -> None:
    root = tmp_path / "projects"
    root.mkdir()
    project = root / "needs-build"
    project.mkdir()
    (project / "README.md").write_text(
        "# Needs Build\n\nCompiles first.\n", encoding="utf-8"
    )
    # The launch asserts the build artifact exists, so it can only come up if the
    # build ran first, in the project directory.
    build_step = [
        "{python}",
        "-c",
        "open('built.txt','w').write('ok');print('BUILD-DONE',flush=True)",
    ]
    server_code = (
        "import http.server,os,sys;"
        "assert os.path.exists('built.txt'),'built.txt missing';"
        "http.server.ThreadingHTTPServer((sys.argv[2],int(sys.argv[1])),"
        "http.server.SimpleHTTPRequestHandler).serve_forever()"
    )
    _write_manifest(
        project,
        {
            "name": "Needs Build",
            "launch": {
                "command": ["{python}", "-u", "-c", server_code, "{port}", "{host}"],
                "build": build_step,
            },
        },
    )
    with _client(settings, root) as client:
        asset = client.post("/api/v1/assets/scan").json()[0]
        assert asset["build_command"] == [build_step]
        assert client.post(f"/api/v1/assets/{asset['id']}/approval").json()[
            "launch_approved"
        ]
        started = client.post(f"/api/v1/assets/{asset['id']}/start", json={})
        assert started.status_code == 200, started.text
        # start() returns only after the build has finished, so its side effect
        # is already on disk.
        assert (project / "built.txt").exists()

        logs = client.get(f"/api/v1/assets/{asset['id']}/logs").json()["logs"]
        assert "build 1/1" in logs
        assert "BUILD-DONE" in logs

        deadline = time.monotonic() + 5
        view = started.json()
        while view.get("status") != "running" and time.monotonic() < deadline:
            time.sleep(0.05)
            view = client.get("/api/v1/assets").json()[0]
        assert view["status"] == "running", view
        assert view["url"].startswith("http://127.0.0.1:")
        client.post(f"/api/v1/assets/{asset['id']}/stop")


def test_a_failed_build_aborts_the_launch_and_marks_failed(
    settings: Settings, tmp_path: Path
) -> None:
    root = tmp_path / "projects"
    root.mkdir()
    project = root / "bad-build"
    project.mkdir()
    (project / "README.md").write_text(
        "# Bad Build\n\nWill not compile.\n", encoding="utf-8"
    )
    _write_manifest(
        project,
        {
            "name": "Bad Build",
            "launch": {
                # If the launch ever ran it would leave this marker; it must not.
                "command": ["{python}", "-c", "open('launched.txt','w').write('x')"],
                "build": [
                    "{python}",
                    "-c",
                    "import sys;print('BOOM',flush=True);sys.exit(3)",
                ],
            },
        },
    )
    with _client(settings, root) as client:
        asset = client.post("/api/v1/assets/scan").json()[0]
        client.post(f"/api/v1/assets/{asset['id']}/approval")
        started = client.post(f"/api/v1/assets/{asset['id']}/start", json={})
        assert started.status_code == 200, started.text
        assert started.json()["status"] == "failed"
        assert started.json()["url"] is None
        assert not (project / "launched.txt").exists()
        logs = client.get(f"/api/v1/assets/{asset['id']}/logs").json()["logs"]
        assert "BOOM" in logs
        assert "build step 1 failed (exit 3)" in logs


def test_a_hung_build_step_times_out_and_aborts(
    settings: Settings, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(asset_library, "_BUILD_STEP_TIMEOUT", 0.5)
    root = tmp_path / "projects"
    root.mkdir()
    project = root / "hung-build"
    project.mkdir()
    (project / "README.md").write_text(
        "# Hung Build\n\nNever finishes.\n", encoding="utf-8"
    )
    _write_manifest(
        project,
        {
            "name": "Hung Build",
            "launch": {
                "command": ["{python}", "-c", "open('launched.txt','w').write('x')"],
                "build": ["{python}", "-c", "import time;time.sleep(30)"],
            },
        },
    )
    with _client(settings, root) as client:
        asset = client.post("/api/v1/assets/scan").json()[0]
        client.post(f"/api/v1/assets/{asset['id']}/approval")
        started = client.post(f"/api/v1/assets/{asset['id']}/start", json={})
        assert started.status_code == 200, started.text
        assert started.json()["status"] == "failed"
        assert not (project / "launched.txt").exists()
        assert (
            "timed out"
            in client.get(f"/api/v1/assets/{asset['id']}/logs").json()["logs"]
        )


# ── Trust and persistence ────────────────────────────────────────────────────


def test_editing_the_build_step_revokes_trust(
    settings: Settings, tmp_path: Path
) -> None:
    root = tmp_path / "projects"
    root.mkdir()
    project = root / "rebuilt"
    project.mkdir()
    manifest = {
        "launch": {
            "command": [sys.executable, "-c", "print('run')"],
            "build": [[sys.executable, "-c", "print('build-v1')"]],
        }
    }
    _write_manifest(project, manifest)
    with _client(settings, root) as client:
        asset = client.post("/api/v1/assets/scan").json()[0]
        assert client.post(f"/api/v1/assets/{asset['id']}/approval").status_code == 200
        # The build argv is part of the trusted fingerprint, so editing it drops
        # trust exactly as editing the launch command does.
        manifest["launch"]["build"][0][-1] = "print('build-v2')"
        (project / ".metis" / "asset.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        denied = client.post(f"/api/v1/assets/{asset['id']}/start", json={})
        assert denied.status_code == 409
        changed = client.get("/api/v1/assets").json()[0]
        assert changed["launch_approved"] is False
        assert changed["status"] == "needs_approval"


def test_a_trusted_build_survives_a_client_restart(
    settings: Settings, tmp_path: Path
) -> None:
    root = tmp_path / "projects"
    root.mkdir()
    project = root / "persist-build"
    project.mkdir()
    build = [[sys.executable, "-c", "print('build')"]]
    _write_manifest(
        project,
        {"launch": {"command": [sys.executable, "-c", "print('run')"], "build": build}},
    )
    with _client(settings, root) as client:
        asset = client.post("/api/v1/assets/scan").json()[0]
        assert client.post(f"/api/v1/assets/{asset['id']}/approval").status_code == 200

    # A fresh client reloads from the persisted catalog. The build must round-trip
    # so the recomputed fingerprint still matches the stored approval — otherwise
    # trust would silently evaporate on restart.
    with _client(settings, root) as client:
        persisted = client.get("/api/v1/assets").json()[0]
        assert persisted["launch_approved"] is True
        assert persisted["build_command"] == build
