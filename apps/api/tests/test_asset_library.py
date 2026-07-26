from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

from fastapi.testclient import TestClient

from waqil_api.config import Settings
from waqil_api.main import create_app


def _client(settings: Settings, root: Path) -> TestClient:
    configured = settings.model_copy(update={"asset_roots": [root]})
    return TestClient(create_app(configured))


def _write_manifest(project: Path, body: dict) -> None:
    manifest_dir = project / ".metis"
    manifest_dir.mkdir()
    (manifest_dir / "asset.json").write_text(json.dumps(body), encoding="utf-8")


def test_asset_scan_is_dynamic_bounded_and_metadata_only(
    settings: Settings, tmp_path: Path
) -> None:
    root = tmp_path / "projects"
    root.mkdir()
    dashboard = root / "customer-dashboard"
    dashboard.mkdir()
    (dashboard / "README.md").write_text(
        "# Customer 360\n\n"
        "A Streamlit dashboard for Oracle Cloud analytics.\n\n"
        "Configure `DAC_OCID` and then run `streamlit run app.py`.\n",
        encoding="utf-8",
    )
    (dashboard / "requirements.txt").write_text("streamlit==1.0\n", encoding="utf-8")
    (dashboard / "app.py").write_text(
        "import os\nprint(os.getenv('SECONDARY_OCID', 'metadata only'))\n",
        encoding="utf-8",
    )
    (dashboard / ".env.example").write_text("DAC_OCID=\n", encoding="utf-8")
    nested = dashboard / "not-an-asset"
    nested.mkdir()
    hidden = root / ".hidden"
    hidden.mkdir()

    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)

    with _client(settings, root) as client:
        snapshot = client.get("/api/v1/assets")
        assert snapshot.status_code == 200
        assert snapshot.json() == []

        first = client.post("/api/v1/assets/scan")
        assert first.status_code == 200
        assert len(first.json()) == 1
        asset = first.json()[0]
        assert asset["name"] == "Customer 360"
        assert asset["summary"] == "A Streamlit dashboard for Oracle Cloud analytics."
        assert asset["category"] == "Analytics"
        assert {"streamlit", "python", "dashboard", "oci"} <= set(asset["tags"])
        assert asset["framework"] == "Streamlit"
        assert asset["entrypoint"] == "app.py"
        assert asset["env_keys"] == ["DAC_OCID", "SECONDARY_OCID"]
        assert asset["launch_configured"] is False
        assert asset["launch_approved"] is False
        assert asset["launch_command"] == []
        assert asset["status"] == "unconfigured"
        stable_id = asset["id"]

        new_asset = root / "fresh-api"
        new_asset.mkdir()
        (new_asset / "README.md").write_text(
            "# Fresh API\n\nA tiny backend API.\n", encoding="utf-8"
        )

        # Runtime actions revalidate only the selected saved asset. They never
        # discover a new sibling folder in the background.
        denied = client.post(f"/api/v1/assets/{stable_id}/start", json={})
        assert denied.status_code == 409
        assert {item["name"] for item in client.get("/api/v1/assets").json()} == {
            "Customer 360"
        }
        assert client.get(f"/api/v1/assets/{stable_id}/logs").status_code == 200
        assert {item["name"] for item in client.get("/api/v1/assets").json()} == {
            "Customer 360"
        }

        rescanned = client.post("/api/v1/assets/scan")
        assert rescanned.status_code == 200
        assert {item["name"] for item in rescanned.json()} == {
            "Customer 360",
            "Fresh API",
        }
        assert next(
            item["id"] for item in rescanned.json() if item["name"] == "Customer 360"
        ) == stable_id

    # The explicit scan snapshot survives an API restart without touching the
    # project folders again.
    with _client(settings, root) as client:
        assert {item["name"] for item in client.get("/api/v1/assets").json()} == {
            "Customer 360",
            "Fresh API",
        }


def test_failed_explicit_scan_preserves_last_good_catalog(
    settings: Settings, tmp_path: Path
) -> None:
    root = tmp_path / "projects"
    root.mkdir()
    project = root / "remembered"
    project.mkdir()
    (project / "README.md").write_text(
        "# Remembered Demo\n\nA catalog entry that survives a missing root.\n",
        encoding="utf-8",
    )

    with _client(settings, root) as client:
        seeded = client.post("/api/v1/assets/scan")
        assert seeded.status_code == 200
        assert [item["name"] for item in seeded.json()] == ["Remembered Demo"]

        root.rename(tmp_path / "temporarily-unavailable")
        failed = client.post("/api/v1/assets/scan")
        assert failed.status_code == 409
        assert "saved catalog was kept" in failed.json()["detail"]
        assert [item["name"] for item in client.get("/api/v1/assets").json()] == [
            "Remembered Demo"
        ]

    # The unchanged snapshot also survives an API restart while its source root
    # remains unavailable.
    with _client(settings, root) as client:
        assert [item["name"] for item in client.get("/api/v1/assets").json()] == [
            "Remembered Demo"
        ]


def test_manifest_launch_is_loopback_redacts_secrets_and_stops_on_shutdown(
    settings: Settings, tmp_path: Path
) -> None:
    root = tmp_path / "projects"
    root.mkdir()
    project = root / "safe-server"
    project.mkdir()
    (project / "README.md").write_text(
        "# Safe Server\n\nA harmless local test server.\n", encoding="utf-8"
    )
    server_code = (
        "import http.server,os,sys;"
        "print(os.environ['DEMO_SECRET'],flush=True);"
        "http.server.ThreadingHTTPServer((sys.argv[2],int(sys.argv[1])),"
        "http.server.SimpleHTTPRequestHandler).serve_forever()"
    )
    _write_manifest(
        project,
        {
            "name": "Safe Server",
            "category": "Developer Tools",
            "tags": ["demo"],
            "env": ["DEMO_SECRET"],
            "launch": {
                "command": [
                    "{python}",
                    "-u",
                    "-c",
                    server_code,
                    "{port}",
                    "{host}",
                ]
            },
        },
    )
    secret = "secret-value-that-must-not-leak"
    process = None
    with _client(settings, root) as client:
        asset = client.post("/api/v1/assets/scan").json()[0]
        assert asset["launch_configured"] is True
        assert asset["launch_approved"] is False
        assert asset["status"] == "needs_approval"
        assert asset["launch_command"][0] == "{python}"
        denied = client.post(
            f"/api/v1/assets/{asset['id']}/start",
            json={"env": {"DEMO_SECRET": secret}},
        )
        assert denied.status_code == 409
        approved = client.post(f"/api/v1/assets/{asset['id']}/approval")
        assert approved.status_code == 200
        assert approved.json()["launch_approved"] is True
        assert approved.json()["status"] == "ready"
        started = client.post(
            f"/api/v1/assets/{asset['id']}/start",
            json={"env": {"DEMO_SECRET": secret}},
        )
        assert started.status_code == 200, started.text
        assert started.json()["status"] == "running"
        assert started.json()["url"].startswith("http://127.0.0.1:")
        assert secret not in started.text

        deadline = time.monotonic() + 5
        latest_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(started.json()["url"], timeout=0.5) as response:
                    assert response.status == 200
                break
            except Exception as exc:  # noqa: BLE001 - bounded startup polling
                latest_error = exc
                time.sleep(0.02)
        else:
            raise AssertionError(f"server did not become reachable: {latest_error}")

        deadline = time.monotonic() + 2
        logs = client.get(f"/api/v1/assets/{asset['id']}/logs")
        while "[REDACTED]" not in logs.text and time.monotonic() < deadline:
            time.sleep(0.02)
            logs = client.get(f"/api/v1/assets/{asset['id']}/logs")
        assert logs.status_code == 200
        assert secret not in logs.text
        assert "[REDACTED]" in logs.json()["logs"]

        process = client.app.state.runtime.assets._runs[asset["id"]].process
        stopped = client.post(f"/api/v1/assets/{asset['id']}/stop")
        assert stopped.status_code == 200
        assert stopped.json()["status"] == "stopped"
        assert stopped.json()["url"] is None
        assert process.returncode is not None

        # Starting it again also proves runtime shutdown owns final cleanup.
        restarted = client.post(
            f"/api/v1/assets/{asset['id']}/start",
            json={"env": {"DEMO_SECRET": secret}},
        )
        assert restarted.status_code == 200
        process = client.app.state.runtime.assets._runs[asset["id"]].process

    assert process is not None
    assert process.returncode is not None


def test_launch_rejects_unknown_malformed_traversal_and_environment_keys(
    settings: Settings, tmp_path: Path
) -> None:
    root = tmp_path / "projects"
    root.mkdir()
    approved = root / "approved"
    approved.mkdir()
    _write_manifest(
        approved,
        {
            "env_keys": ["SAFE_VALUE"],
            "launch": {"command": [sys.executable, "-c", "import time;time.sleep(10)"]},
        },
    )
    malformed = root / "malformed"
    malformed.mkdir()
    _write_manifest(malformed, {"launch": {"command": "python app.py"}})

    with _client(settings, root) as client:
        assets = {item["name"]: item for item in client.post("/api/v1/assets/scan").json()}
        assert assets["malformed"]["launch_approved"] is False
        assert assets["approved"]["launch_configured"] is True
        assert assets["approved"]["launch_approved"] is False
        assert client.post("/api/v1/assets/not-an-id/start", json={}).status_code == 409
        approved_id = assets["approved"]["id"]
        assert client.post(f"/api/v1/assets/{approved_id}/approval").status_code == 200
        assert (
            client.post(
                f"/api/v1/assets/{approved_id}/start",
                json={"env": {"PATH": "/tmp/override"}},
            ).status_code
            == 409
        )
        assert (
            client.post(
                f"/api/v1/assets/{approved_id}/start",
                json={"env": {"UNDECLARED_VALUE": "no"}},
            ).status_code
            == 409
        )
        assert (
            client.post(
                f"/api/v1/assets/{assets['malformed']['id']}/start", json={}
            ).status_code
            == 409
        )


def test_launch_approval_is_persisted_and_invalidated_by_manifest_drift(
    settings: Settings, tmp_path: Path
) -> None:
    root = tmp_path / "projects"
    root.mkdir()
    project = root / "reviewed"
    project.mkdir()
    manifest = {
        "env": ["SAFE_VALUE"],
        "launch": {"command": [sys.executable, "-c", "print('v1')"]},
    }
    _write_manifest(project, manifest)

    with _client(settings, root) as client:
        asset = client.post("/api/v1/assets/scan").json()[0]
        assert client.post(f"/api/v1/assets/{asset['id']}/approval").status_code == 200

    with _client(settings, root) as client:
        persisted = client.get("/api/v1/assets").json()[0]
        assert persisted["launch_approved"] is True

        manifest["launch"]["command"][-1] = "print('v2')"
        (project / ".metis" / "asset.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        # Selecting this already-known asset revalidates only its own recipe;
        # no catalog-wide scan is needed to revoke a stale approval.
        denied = client.post(f"/api/v1/assets/{asset['id']}/start", json={})
        assert denied.status_code == 409
        changed = client.get("/api/v1/assets").json()[0]
        assert changed["launch_configured"] is True
        assert changed["launch_approved"] is False
        assert changed["status"] == "needs_approval"


def test_starting_asset_remains_visible_and_stoppable_after_project_moves(
    settings: Settings, tmp_path: Path
) -> None:
    root = tmp_path / "projects"
    root.mkdir()
    project = root / "moving-project"
    project.mkdir()
    _write_manifest(
        project,
        {"launch": {"command": [sys.executable, "-c", "import time;time.sleep(30)"]}},
    )

    with _client(settings, root) as client:
        asset = client.post("/api/v1/assets/scan").json()[0]
        assert client.post(f"/api/v1/assets/{asset['id']}/approval").status_code == 200
        started = client.post(f"/api/v1/assets/{asset['id']}/start", json={})
        assert started.status_code == 200
        assert started.json()["status"] == "starting"
        assert started.json()["url"].startswith("http://127.0.0.1:")

        # Ordinary reads report live runtime state from the saved snapshot;
        # they must not require or trigger another projects-folder scan.
        snapshot = client.get("/api/v1/assets")
        assert snapshot.status_code == 200
        assert len(snapshot.json()) == 1
        assert snapshot.json()[0]["id"] == asset["id"]
        assert snapshot.json()[0]["status"] == "starting"
        assert snapshot.json()[0]["url"] == started.json()["url"]

        project.rename(tmp_path / "moved-project")
        logs = client.get(f"/api/v1/assets/{asset['id']}/logs")
        assert logs.status_code == 200
        stopped = client.post(f"/api/v1/assets/{asset['id']}/stop")
        assert stopped.status_code == 200
        assert stopped.json()["status"] == "stopped"
        assert stopped.json()["url"] is None

        stopped_snapshot = client.get("/api/v1/assets").json()
        assert len(stopped_snapshot) == 1
        assert stopped_snapshot[0]["status"] == "stopped"
        assert stopped_snapshot[0]["url"] is None
