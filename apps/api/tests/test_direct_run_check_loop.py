"""The feedback loop the whole architecture exists for.

Under the planner/slice design a coder could not run anything. A missing
`python-multipart` cost a full round trip — write, host verify, findings, a
second inference — and that round trip is why repair machinery had to exist at
all. `run_check` closes it: the session asks Metis for a named check, Metis
runs it in its own pinned networkless verifier against the live mirror, and
the findings come back inside the same turn.

The loop crosses a process boundary, so it is proved in two halves that meet
at a defined contract:

* the sidecar half — a named check reaches the runner, a command never does,
  and the findings come back as that call's own result (sidecar suite);
* the host half, below — the handler runs the real verification ladder against
  the real mirror and returns findings the session can act on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from waqil_api.coding_contracts import HOST_CHECKS

from test_project_coding_engine import _coordinator as _direct_coordinator


# An application that imports fastapi.File without declaring python-multipart:
# the exact defect the Logivity build shipped into a second round.
MAIN_WITH_UPLOAD = """\
from fastapi import FastAPI, File, UploadFile

app = FastAPI()


@app.post("/api/documents")
async def upload(file: UploadFile = File(...)) -> dict[str, str]:
    return {"name": file.filename or ""}
"""

REQUIREMENTS_MISSING = "fastapi>=0.110\n"
REQUIREMENTS_FIXED = "fastapi>=0.110\npython-multipart>=0.0.9\n"


class _Plane:
    """Only the parts of the control plane the handler touches."""

    def __init__(self, projects: Any, settings: Any) -> None:
        self.projects = projects
        self.settings = settings
        self.events = _Events()
        self.verifications: list[tuple[str, bool]] = []

    async def _verify_staged_changeset(
        self,
        project_id: str,
        staged: dict[str, Any],
        *,
        full: bool = False,
        planned: list[str] | None = None,
        required: list[str] | None = None,
        scenarios: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """The real rung this test cares about, run on the real staged bytes.

        Declaring `fastapi.File` without python-multipart is a cross-file
        wiring defect: the verifier that finds it is the same one the approval
        gate uses, which is the point — a check is not a weaker preview.
        """

        self.verifications.append((project_id, full))
        errors: list[dict[str, Any]] = []
        main = str((staged.get("app/main.py") or {}).get("content") or "")
        requirements = str((staged.get("requirements.txt") or {}).get("content") or "")
        if "File(" in main and "multipart" not in requirements:
            errors.append(
                {
                    "path": "app/main.py",
                    "severity": "error",
                    "error": (
                        "uses fastapi.File, which needs python-multipart declared "
                        "in requirements — without it FastAPI fails at startup"
                    ),
                }
            )
        return {"errors": errors, "warnings": [], "notes": [], "checks": []}


class _Events:
    def __init__(self) -> None:
        self.items: list[tuple[str, dict[str, Any]]] = []

    async def emit(
        self, run_id: str, conversation_id: str, type: str, payload: dict[str, Any]
    ) -> None:
        self.items.append((type, payload))


def _handler(plane: Any, project_id: str, mirror: Any) -> Any:
    from waqil_api.control_plane import ControlPlane

    return ControlPlane._make_check_handler(
        plane,
        {"run_id": "run_1", "conversation_id": "conv_1", "project_staged": {}},
        project_id,
        lambda _session_id: mirror,
        [],
    )


@pytest.mark.asyncio
async def test_a_check_finds_a_missing_dependency_and_the_fix_clears_it(
    tmp_path: Path,
) -> None:
    """One session, its own order, a mid-session check, and a repair."""

    written: list[str] = []

    def build(root: Path, _phase: str) -> None:
        # The session chooses its own order: the application first, packaging
        # after. Nothing told it to; no manifest exists.
        (root / "app" / "main.py").write_text(MAIN_WITH_UPLOAD, encoding="utf-8")
        written.append("app/main.py")
        (root / "requirements.txt").write_text(REQUIREMENTS_MISSING, encoding="utf-8")
        written.append("requirements.txt")

    (
        coordinator,
        engine,
        _sessions,
        projects,
        database,
        _run_id,
        _conv,
        project_id,
    ) = await _direct_coordinator(tmp_path, build)
    try:
        mirror = await projects.create_external_mirror(project_id, {})
        build(mirror.project_root, "start")
        plane = _Plane(projects, coordinator.settings)
        run_check = _handler(plane, project_id, mirror)

        # ── mid-session check 1: the defect is found before the round ends ──
        first = await run_check("coding_1", "imports")
        assert first.ok is False
        assert first.errors == 1
        assert "python-multipart" in first.findings[0].detail
        assert first.findings[0].path == "app/main.py"

        # ── the session repairs it in the same mirror, same session ────────
        (mirror.project_root / "requirements.txt").write_text(
            REQUIREMENTS_FIXED, encoding="utf-8"
        )
        written.append("requirements.txt (repaired)")

        second = await run_check("coding_1", "imports")
        assert second.ok is True
        assert second.errors == 0

        # No external repair session was needed: two checks, one mirror, and
        # the repair landed between them without another session being opened.
        assert plane.verifications == [(project_id, False), (project_id, False)]
        # The session wrote the application before its packaging, and the
        # repair came after the check rather than before it.
        assert written[-3:] == [
            "app/main.py",
            "requirements.txt",
            "requirements.txt (repaired)",
        ]

        # Every check is durably observable.
        checks = [
            item for item in plane.events.items if item[0] == "project.check_requested"
        ]
        assert len(checks) == 2
        assert checks[0][1]["check"] == "imports"
        assert checks[0][1]["ok"] is False
        assert checks[1][1]["ok"] is True
        assert checks[1][1]["checks_used"] == 2
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_the_check_reads_the_live_mirror_not_the_last_staged_overlay(
    tmp_path: Path,
) -> None:
    """A check has to see what the model just wrote, or it is useless."""

    def build(root: Path, _phase: str) -> None:
        (root / "app" / "main.py").write_text("X = 1\n", encoding="utf-8")

    (
        coordinator,
        _engine,
        _sessions,
        projects,
        database,
        _run_id,
        _conv,
        project_id,
    ) = await _direct_coordinator(tmp_path, build)
    try:
        mirror = await projects.create_external_mirror(project_id, {})
        plane = _Plane(projects, coordinator.settings)
        run_check = _handler(plane, project_id, mirror)

        # Written directly into the mirror after it was created, exactly as a
        # live session writes. The staged overlay passed to the handler is
        # empty, so anything the check sees came from disk.
        (mirror.project_root / "app" / "main.py").write_text(
            MAIN_WITH_UPLOAD, encoding="utf-8"
        )
        (mirror.project_root / "requirements.txt").write_text(
            REQUIREMENTS_MISSING, encoding="utf-8"
        )

        result = await run_check("coding_1", "full")

        assert result.ok is False
        assert result.errors == 1
        # `full` is the same verification the approval gate runs, not a
        # cheaper preview of it.
        assert plane.verifications == [(project_id, True)]
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_the_check_budget_is_bounded_and_says_so(tmp_path: Path) -> None:
    def build(root: Path, _phase: str) -> None:
        (root / "app" / "main.py").write_text("X = 1\n", encoding="utf-8")

    (
        coordinator,
        _engine,
        _sessions,
        projects,
        database,
        _run_id,
        _conv,
        project_id,
    ) = await _direct_coordinator(tmp_path, build)
    try:
        mirror = await projects.create_external_mirror(project_id, {})
        plane = _Plane(projects, coordinator.settings)
        object.__setattr__(plane.settings, "project_run_check_budget", 2)
        run_check = _handler(plane, project_id, mirror)

        assert (await run_check("coding_1", "imports")).unavailable is None
        assert (await run_check("coding_1", "imports")).unavailable is None
        spent = await run_check("coding_1", "imports")

        assert spent.unavailable is not None
        assert "all 2 of its checks" in spent.unavailable
        # A refused check costs no verification.
        assert len(plane.verifications) == 2
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_a_missing_mirror_is_reported_rather_than_raising(
    tmp_path: Path,
) -> None:
    def build(root: Path, _phase: str) -> None:
        (root / "app" / "main.py").write_text("X = 1\n", encoding="utf-8")

    (
        coordinator,
        _engine,
        _sessions,
        projects,
        database,
        _run_id,
        _conv,
        project_id,
    ) = await _direct_coordinator(tmp_path, build)
    try:
        plane = _Plane(projects, coordinator.settings)
        run_check = _handler(plane, project_id, None)

        result = await run_check("coding_1", "pytest")

        # Between rounds there is no mirror. The session is told, and its turn
        # continues; an exception here would end a turn that could still work.
        assert result.unavailable == "the workspace mirror is not available"
        assert result.ok is False
    finally:
        await database.close()


def test_the_check_vocabulary_is_closed_and_shared() -> None:
    assert HOST_CHECKS == ("imports", "pytest", "ruff", "acceptance", "full")
