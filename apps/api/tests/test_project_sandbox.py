"""The execution gate: what gets materialized, imported, and reported.

Classification is where this earns its keep. The sandbox is offline, so a
package the project legitimately declares can be missing from the image; calling
that a defect would send the model to rewrite correct code, and calling a real
undeclared import "unavailable" would let a broken build through. Both
directions are pinned here.

The container itself is exercised by one integration test that skips unless the
image is actually built, so the suite never depends on a running VM.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from waqil_api.config import Settings
from waqil_api.project_sandbox import (
    ProjectSandboxService,
    classify_envelope,
    import_order,
    materialize,
)


def _staged(files: dict[str, str]) -> dict[str, dict[str, str]]:
    """Wrap raw file text in the staged-entry shape the sandbox reads."""
    return {path: {"content": content} for path, content in files.items()}


def _succeeded(checks: list[dict[str, object]]) -> dict[str, object]:
    """A sandbox envelope that ran to completion and reported these checks."""
    return {
        "schema_version": "1",
        "status": "succeeded",
        "checks": checks,
        "routes": [],
    }


# ── Materializing what the approval would actually write ─────────────────────


def test_materialize_lays_the_overlay_over_the_project_on_disk(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "app").mkdir(parents=True)
    (project / "app" / "config.py").write_text("URL = 'old'\n", encoding="utf-8")
    (project / "app" / "keep.py").write_text("KEEP = 1\n", encoding="utf-8")
    destination = tmp_path / "copy"

    materialize(
        project,
        _staged({"app/config.py": "URL = 'new'\n", "app/main.py": "X = 2\n"}),
        destination,
    )

    assert (destination / "app" / "config.py").read_text(
        encoding="utf-8"
    ) == "URL = 'new'\n"
    assert (destination / "app" / "keep.py").read_text(encoding="utf-8") == "KEEP = 1\n"
    assert (destination / "app" / "main.py").read_text(encoding="utf-8") == "X = 2\n"
    assert (project / "app" / "config.py").read_text(
        encoding="utf-8"
    ) == "URL = 'old'\n"


def test_materialize_leaves_out_directories_that_are_not_the_project(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    (project / ".git").mkdir(parents=True)
    (project / ".git" / "HEAD").write_text("ref: main\n", encoding="utf-8")
    (project / "node_modules" / "x").mkdir(parents=True)
    (project / "node_modules" / "x" / "index.js").write_text("1\n", encoding="utf-8")
    destination = tmp_path / "copy"

    materialize(project, _staged({"app.py": "X = 1\n"}), destination)

    assert not (destination / ".git").exists()
    assert not (destination / "node_modules").exists()
    assert (destination / "app.py").is_file()


def test_materialize_refuses_a_project_past_the_sandbox_limits(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    for index in range(5):
        (project / f"file{index}.txt").write_text("x" * 100, encoding="utf-8")

    with pytest.raises(ValueError):
        materialize(project, {}, tmp_path / "copy", max_files=3)


def test_import_order_puts_the_entrypoint_first(tmp_path: Path) -> None:
    staged = _staged(
        {
            "app/agents/base.py": "",
            "app/main.py": "",
            "README.md": "",
            "app/__init__.py": "",
            "tests/test_main.py": "",
        }
    )

    modules = import_order([], staged, limit=10)
    assert modules[0] == "app.main"
    assert "app.agents.base" in modules
    assert "tests.test_main" not in modules
    assert all(not name.endswith("__init__") for name in modules)


# ── Classifying what came back ───────────────────────────────────────────────


def test_a_missing_project_module_is_a_defect() -> None:
    outcome = classify_envelope(
        _succeeded(
            [
                {
                    "name": "import app.main",
                    "kind": "import",
                    "ok": False,
                    "missing_module": "app.helpers",
                    "where": "app/main.py line 2",
                }
            ]
        ),
        staged=_staged({"app/main.py": "", "app/__init__.py": ""}),
    )

    assert outcome.available
    assert [item["severity"] for item in outcome.findings] == ["error"]
    assert outcome.findings[0]["path"] == "app/main.py"
    assert "app.helpers" in outcome.findings[0]["error"]


def test_a_declared_package_the_image_lacks_is_never_reported_as_a_defect() -> None:
    """The verifier is offline. Blaming the code for that would have the model
    rewrite something that is already correct."""
    outcome = classify_envelope(
        _succeeded(
            [
                {
                    "name": "import app.main",
                    "kind": "import",
                    "ok": False,
                    "missing_module": "polars",
                    "where": "app/main.py line 3",
                }
            ]
        ),
        staged=_staged({"app/main.py": ""}),
        requirements="fastapi\npolars==1.0.0\n",
    )

    assert [item["severity"] for item in outcome.findings] == ["warning"]
    assert outcome.skipped_modules == ["polars"]


def test_an_undeclared_third_party_import_is_a_defect() -> None:
    outcome = classify_envelope(
        _succeeded(
            [
                {
                    "name": "import app.main",
                    "kind": "import",
                    "ok": False,
                    "missing_module": "polars",
                    "where": "app/main.py line 3",
                }
            ]
        ),
        staged=_staged({"app/main.py": ""}),
        requirements="fastapi\n",
    )

    assert [item["severity"] for item in outcome.findings] == ["error"]
    assert "neither declared" in outcome.findings[0]["error"]


def test_a_route_that_raises_is_reported_where_it_raised() -> None:
    outcome = classify_envelope(
        _succeeded(
            [
                {"name": "import app.main", "kind": "import", "ok": True},
                {
                    "name": "GET /report",
                    "kind": "request",
                    "ok": False,
                    "detail": "ZeroDivisionError: division by zero",
                    "where": "app/report.py line 18",
                },
            ]
        ),
        staged=_staged({"app/main.py": "", "app/report.py": ""}),
    )

    assert outcome.findings[0]["path"] == "app/report.py"
    assert "line 18" in outcome.findings[0]["error"]
    assert "ZeroDivisionError" in outcome.findings[0]["error"]


def test_a_response_failure_is_classified_against_the_route_module() -> None:
    """Wrong statuses have no traceback; route attribution must survive host
    classification so repair targets the handler, not the entrypoint."""
    outcome = classify_envelope(
        _succeeded(
            [
                {
                    "name": "acceptance: document upload",
                    "kind": "acceptance",
                    "ok": False,
                    "detail": "POST /api/documents returned HTTP 503, expected 2xx",
                    "where": "app/routes/documents.py line 27",
                }
            ]
        ),
        staged=_staged(
            {
                "app/main.py": "",
                "app/routes/documents.py": "",
            }
        ),
    )

    (finding,) = outcome.findings
    assert finding["path"] == "app/routes/documents.py"
    assert finding["kind"] == "acceptance"
    assert "line 27" in finding["error"]


def test_a_clean_run_reports_no_findings_at_all() -> None:
    outcome = classify_envelope(
        _succeeded([{"name": "import app.main", "kind": "import", "ok": True}]),
        staged=_staged({"app/main.py": ""}),
    )

    assert outcome.available
    assert outcome.findings == []
    assert len(outcome.checks) == 1


@pytest.mark.parametrize(
    "code",
    ["PODMAN_UNAVAILABLE", "ROOTLESS_REQUIRED", "UNAPPROVED_IMAGE", "INPUT_TOO_LARGE"],
)
def test_a_sandbox_that_could_not_run_degrades_instead_of_blaming_the_code(
    code: str,
) -> None:
    outcome = classify_envelope(
        {
            "schema_version": "1",
            "status": "failed",
            "error": {"code": code, "message": "nope"},
        },
        staged=_staged({"app/main.py": ""}),
    )

    assert outcome.available is False
    assert code in outcome.reason
    assert outcome.findings == []


def test_a_project_that_hangs_on_import_is_a_defect_not_a_degrade() -> None:
    outcome = classify_envelope(
        {
            "schema_version": "1",
            "status": "failed",
            "error": {
                "code": "SANDBOX_TIMEOUT",
                "message": "sandbox exceeded 90 seconds",
            },
        },
        staged=_staged({"app/main.py": ""}),
    )

    assert outcome.available is True
    assert "did not finish importing" in outcome.findings[0]["error"]


# ── The hermetic-suite boundary ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_deterministic_backend_never_reaches_for_a_container(
    tmp_path: Path,
) -> None:
    """A suite whose result depends on whether a VM happens to be up is not a
    suite, so the test backend degrades silently rather than shelling out."""
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        repo_root=tmp_path,
        model_backend="deterministic",
        allow_test_backends=True,
    )

    outcome = await ProjectSandboxService(settings).verify(
        root=tmp_path / "project", staged=_staged({"app/main.py": "X = 1\n"})
    )

    assert outcome.available is False
    assert outcome.reason == ""


# ── Giving the VM back ───────────────────────────────────────────────────────


def _sandbox(tmp_path: Path) -> ProjectSandboxService:
    """A sandbox service with a throwaway data directory."""
    return ProjectSandboxService(
        Settings(_env_file=None, data_dir=tmp_path / "data", repo_root=tmp_path)
    )


@pytest.mark.asyncio
async def test_a_machine_metis_did_not_start_is_never_stopped(tmp_path: Path) -> None:
    """Podman is a shared tool. If it was already up it belongs to whatever else
    the user is doing, and shutting that down is not Metis's call."""
    service = _sandbox(tmp_path)
    calls: list[list[str]] = []

    def record(command, **_kwargs):
        calls.append(list(command))
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkey = pytest.MonkeyPatch()
    monkey.setattr(subprocess, "run", record)
    try:
        service._ensure_machine()  # podman info succeeds → already running
        assert await service.release_machine() is False
    finally:
        monkey.undo()

    # The binary is resolved to an absolute path now; the verb is what matters.
    assert [[Path(c[0]).name, c[1]] for c in calls] == [["podman", "info"]]


@pytest.mark.asyncio
async def test_a_machine_metis_started_is_stopped_when_it_goes_idle(
    tmp_path: Path,
) -> None:
    service = _sandbox(tmp_path)
    calls: list[list[str]] = []

    def record(command, **_kwargs):
        calls.append(list(command))
        # `podman info` fails, so the machine is down and Metis starts it.
        returncode = (
            1 if [Path(command[0]).name, command[1]] == ["podman", "info"] else 0
        )
        return SimpleNamespace(returncode=returncode, stdout=b"", stderr=b"")

    monkey = pytest.MonkeyPatch()
    monkey.setattr(subprocess, "run", record)
    try:
        service._ensure_machine()
        assert service._machine_started_here is True
        # Freshly used, so an idle release does nothing yet.
        assert await service.release_if_idle(3600) is False
        assert await service.release_if_idle(0) is False
        service._last_used -= 7200
        assert await service.release_if_idle(3600) is True
        # And it does not try to stop the same machine twice.
        assert await service.release_if_idle(3600) is False
    finally:
        monkey.undo()

    stops = [
        c for c in calls if [Path(c[0]).name, *c[1:3]] == ["podman", "machine", "stop"]
    ]
    assert len(stops) == 1


# ── The real container, when it is available ─────────────────────────────────


def _image_available() -> bool:
    """Whether the built verify image is present on this machine."""
    if shutil.which("podman") is None:
        return False
    probe = subprocess.run(
        ["podman", "image", "exists", "localhost/metis/project-verify:0.3.0"],
        capture_output=True,
        check=False,
    )
    return probe.returncode == 0


def _verify(project: Path, modules: list[str]) -> dict:
    """Run the real wrapper against a project and return its envelope."""
    runner = Path("infra/sandbox/project-verify/run_project_verify.py").resolve()
    completed = subprocess.run(
        [sys.executable, str(runner), "--project-dir", str(project)],
        input=json.dumps({"schema_version": "1", "modules": modules}).encode("utf-8"),
        capture_output=True,
        timeout=180,
        check=False,
    )
    return json.loads(completed.stdout.decode("utf-8"))


@pytest.mark.skipif(
    not _image_available(), reason="verify sandbox image is not built here"
)
def test_the_verify_image_contains_the_tested_verifier_source() -> None:
    """Do not let source-only tests certify code the production image lacks."""
    source = (
        Path(__file__).resolve().parents[3]
        / "infra"
        / "sandbox"
        / "project-verify"
        / "verify_tool.py"
    )
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    completed = subprocess.run(
        [
            "podman",
            "run",
            "--rm",
            "--pull=never",
            "--network=none",
            "--read-only",
            "--entrypoint=sha256sum",
            "localhost/metis/project-verify:0.3.0",
            "/opt/metis/verify/verify_tool.py",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    image_digest = completed.stdout.split(maxsplit=1)[0]
    assert image_digest == source_digest, (
        "the project-verify image is stale; rebuild it with "
        "infra/sandbox/project-verify/build_project_verify_image.sh"
    )


@pytest.mark.skipif(
    not _image_available(), reason="verify sandbox image is not built here"
)
def test_the_sandbox_runs_as_the_project_so_relative_paths_resolve(
    tmp_path: Path,
) -> None:
    """Caught on a live build: `StaticFiles(directory="app/static")` is ordinary
    correct code, and a verifier that imports from its own working directory
    reports it as a missing directory. A false failure costs more than the check
    is worth, so the sandbox runs *as* the project, not merely with it on the path."""
    project = tmp_path / "project"
    (project / "app" / "static").mkdir(parents=True)
    (project / "app" / "__init__.py").write_text("", encoding="utf-8")
    (project / "app" / "static" / "index.html").write_text(
        "<html></html>", encoding="utf-8"
    )
    (project / "app" / "main.py").write_text(
        "from fastapi import FastAPI\n"
        "from fastapi.staticfiles import StaticFiles\n"
        "app = FastAPI()\n"
        "app.mount('/static', StaticFiles(directory='app/static'), name='static')\n"
        "@app.get('/')\n"
        "def home():\n"
        "    return {'ok': True}\n",
        encoding="utf-8",
    )

    envelope = _verify(project, ["app.main"])

    assert envelope["status"] == "succeeded"
    assert [check for check in envelope["checks"] if not check["ok"]] == []


@pytest.mark.skipif(
    not _image_available(), reason="verify sandbox image is not built here"
)
def test_the_sandbox_runs_a_project_and_reports_a_failing_route(tmp_path: Path) -> None:
    """End to end through the real wrapper and container: a route that raises is
    caught with its file and line, which no static check could ever produce."""
    project = tmp_path / "project"
    (project / "app").mkdir(parents=True)
    (project / "app" / "__init__.py").write_text("", encoding="utf-8")
    (project / "app" / "main.py").write_text(
        "from fastapi import FastAPI\n"
        "app = FastAPI()\n"
        "@app.get('/boom')\n"
        "def boom():\n"
        "    return 1 / 0\n",
        encoding="utf-8",
    )
    runner = Path("infra/sandbox/project-verify/run_project_verify.py").resolve()

    completed = subprocess.run(
        [sys.executable, str(runner), "--project-dir", str(project)],
        input=json.dumps({"schema_version": "1", "modules": ["app.main"]}).encode(
            "utf-8"
        ),
        capture_output=True,
        timeout=180,
        check=False,
    )
    envelope = json.loads(completed.stdout.decode("utf-8"))

    assert envelope["status"] == "succeeded"
    failed = [check for check in envelope["checks"] if not check["ok"]]
    assert [check["name"] for check in failed] == ["GET /boom"]
    assert failed[0]["error_type"] == "ZeroDivisionError"
    assert failed[0]["where"] == "app/main.py line 5"


@pytest.mark.asyncio
@pytest.mark.skipif(
    not _image_available(), reason="verify sandbox image is not built here"
)
async def test_project_tests_run_against_exact_staged_bytes_without_network(
    tmp_path: Path,
) -> None:
    """The production path, not an in-process approximation: the disk version
    fails the regression, the staged overlay passes it, and the same test process
    proves Podman's network namespace has no outbound route."""
    project = tmp_path / "project"
    (project / "app").mkdir(parents=True)
    (project / "tests").mkdir()
    (project / "app" / "__init__.py").write_text("", encoding="utf-8")
    (project / "app" / "value.py").write_text(
        "def current():\n    return 'disk-bug'\n", encoding="utf-8"
    )
    (project / "tests" / "test_value.py").write_text(
        "import errno\n"
        "import socket\n"
        "from app.value import current\n"
        "\n"
        "def test_exact_staged_value():\n"
        "    assert current() == 'staged-fix'\n"
        "\n"
        "def test_network_is_absent():\n"
        "    client = socket.socket()\n"
        "    client.settimeout(0.2)\n"
        "    try:\n"
        "        status = client.connect_ex(('1.1.1.1', 80))\n"
        "    finally:\n"
        "        client.close()\n"
        "    assert status in {errno.ENETUNREACH, errno.EHOSTUNREACH}\n",
        encoding="utf-8",
    )
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        repo_root=Path(__file__).resolve().parents[3],
        project_sandbox_autostart=False,
    )
    service = ProjectSandboxService(settings)

    broken = await service.verify(
        root=project,
        staged=_staged({"app/value.py": "def current():\n    return 'still-broken'\n"}),
        project_paths=[
            "app/__init__.py",
            "app/value.py",
            "tests/test_value.py",
        ],
    )

    assert broken.available, broken.reason
    test_failures = [
        finding for finding in broken.findings if finding.get("kind") == "test"
    ]
    assert len(test_failures) == 1
    assert test_failures[0]["path"] == "app/value.py"
    assert "test_exact_staged_value" in test_failures[0]["error"]

    outcome = await service.verify(
        root=project,
        staged=_staged({"app/value.py": "def current():\n    return 'staged-fix'\n"}),
        project_paths=[
            "app/__init__.py",
            "app/value.py",
            "tests/test_value.py",
        ],
    )

    assert outcome.available, outcome.reason
    assert outcome.findings == []
    test_checks = [check for check in outcome.checks if check.get("kind") == "test"]
    assert len(test_checks) == 1
    assert test_checks[0]["ok"] is True
    assert "2 passed" in test_checks[0]["detail"]
    assert (
        (project / "app" / "value.py")
        .read_text(encoding="utf-8")
        .endswith("return 'disk-bug'\n")
    )

    # A test-only follow-up still enters pytest even though test modules are
    # deliberately excluded from the app import probe.
    staged_test = (
        (project / "tests" / "test_value.py")
        .read_text(encoding="utf-8")
        .replace("'staged-fix'", "'disk-bug'")
    )
    test_only = await service.verify(
        root=project,
        staged=_staged({"tests/test_value.py": staged_test}),
        project_paths=[
            "app/__init__.py",
            "app/value.py",
            "tests/test_value.py",
        ],
    )
    assert test_only.available, test_only.reason
    assert test_only.findings == []
    assert any(
        check.get("kind") == "test" and "2 passed" in str(check.get("detail"))
        for check in test_only.checks
    )


def test_podman_resolution_survives_a_launcher_with_a_bare_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The API inherits PATH from its launcher, which need not carry the Podman
    installer's directory. Resolution must find the binary in the known install
    locations anyway — the exact gap that left every baseline build unverified
    on a Mac with a perfectly working podman."""
    from waqil_api import project_sandbox

    fake_install = tmp_path / "opt-podman-bin"
    fake_install.mkdir()
    binary = fake_install / "podman"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)

    monkeypatch.setenv("PATH", "/usr/bin:/bin")  # a launcher-shaped PATH
    monkeypatch.setattr(project_sandbox, "_PODMAN_FALLBACK_DIRS", (str(fake_install),))

    assert project_sandbox.podman_binary() == str(binary)
    child_path = project_sandbox.podman_child_env()["PATH"]
    assert child_path.split(":")[0] == str(fake_install)
    assert "/usr/bin" in child_path  # the rest of the environment survives


def test_podman_resolution_prefers_the_callers_path_when_it_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A user who arranged their own podman on PATH keeps winning; the fallback
    directories only answer when PATH is silent."""
    from waqil_api import project_sandbox

    on_path = tmp_path / "on-path"
    on_path.mkdir()
    chosen = on_path / "podman"
    chosen.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    chosen.chmod(0o755)
    ignored = tmp_path / "fallback"
    ignored.mkdir()
    (ignored / "podman").write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    (ignored / "podman").chmod(0o755)

    monkeypatch.setenv("PATH", f"{on_path}:/usr/bin:/bin")
    monkeypatch.setattr(project_sandbox, "_PODMAN_FALLBACK_DIRS", (str(ignored),))

    assert project_sandbox.podman_binary() == str(chosen)
    # PATH already sees it, so the child environment gains no duplicate entry.
    assert project_sandbox.podman_child_env()["PATH"].split(":")[0] == str(on_path)


def test_a_machine_is_never_poked_when_podman_is_genuinely_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No binary anywhere: _ensure_machine must not attempt a start, and the
    wrapper (not the host) stays the one who reports unavailable."""
    from waqil_api import project_sandbox

    monkeypatch.setenv("PATH", str(tmp_path))  # an empty directory
    monkeypatch.setattr(
        project_sandbox, "_PODMAN_FALLBACK_DIRS", (str(tmp_path / "nowhere"),)
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(
        project_sandbox.subprocess,
        "run",
        lambda argv, **_: calls.append(argv) or SimpleNamespace(returncode=1),
    )

    service = ProjectSandboxService(Settings(model_backend="deterministic"))
    service._ensure_machine()

    assert calls == []
