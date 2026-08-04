"""Golden traces: known-bad build changesets that must stay caught.

Every defect in this file is one a real model build actually shipped and a green
approval card actually offered. The point is not to test a rung in isolation
(the rung suites do that) but to lock the whole finding -> verdict -> approvable
chain against regressions: if a future change — a model swap, a decode-engine
swap, a refactor — quietly stops catching one of these, a test here goes red
before a broken build reaches a user again.

Composed from the pure static rungs (syntax + wiring) exactly as the build loop
assembles them, so no model and no container are needed: the traces run in
milliseconds and never depend on a VM.
"""
from __future__ import annotations

from typing import Any

from waqil_api.control_plane import (
    _annotate_summary,
    _blocking_reason,
    _distinct_findings,
    _from_rung,
)
from waqil_api.project_wiring import staged_wiring_errors
from waqil_api.project_workspace import staged_syntax_errors


def _staged(files: dict[str, str]) -> dict[str, dict[str, str]]:
    return {path: {"content": content} for path, content in files.items()}


def _verdict(
    files: dict[str, str], *, project_paths: tuple[str, ...] = ()
) -> tuple[str | None, str, dict[str, Any]]:
    """Run the static rungs the way the loop does, and return what the user sees.

    Returns ``(blocking_reason, card_text, verification)`` — ``blocking_reason``
    is None exactly when the Approve button would be offered.
    """
    staged = _staged(files)
    result: dict[str, Any] = {"errors": [], "warnings": [], "notes": [], "checks": []}
    result["errors"].extend(_from_rung(staged_syntax_errors(staged), "syntax"))
    wiring = _from_rung(
        staged_wiring_errors(staged, project_paths=list(project_paths)), "wiring"
    )
    result["errors"].extend(item for item in wiring if item.get("severity") != "warning")
    result["warnings"].extend(item for item in wiring if item.get("severity") == "warning")
    result["errors"] = _distinct_findings(result["errors"])
    result["warnings"] = _distinct_findings(result["warnings"])
    return _blocking_reason(result), _annotate_summary("<staged file list>", result), result


# ── Trace 1: the live qwen "Agent Showcase" build that did not boot ──────────
# app/main.py mounts StaticFiles at a directory the changeset never creates
# (`STATIC = Path(__file__).parent / "static"`), so the app raises at import.
# The card used to come back green and approvable; it must not.

QWEN_NONBOOT = {
    "app/main.py": (
        "from fastapi import FastAPI\n"
        "from fastapi.staticfiles import StaticFiles\n"
        "from fastapi.responses import FileResponse\n"
        "from pathlib import Path\n"
        "app = FastAPI(title='Agent Showcase')\n"
        "STATIC = Path(__file__).parent / 'static'\n"
        "app.mount('/static', StaticFiles(directory=STATIC), name='static')\n"
        "@app.get('/')\n"
        "async def index():\n"
        "    return FileResponse(STATIC / 'index.html')\n"
    ),
    "requirements.txt": "fastapi\nuvicorn\n",
}


def test_qwen_non_booting_static_mount_is_not_approvable() -> None:
    reason, card, _ = _verdict(QWEN_NONBOOT)
    assert reason is not None  # the Approve button is withheld
    assert "StaticFiles" in reason  # blocked specifically on the mount defect,
    assert "app/static" in reason  # so removing that check would flip this trace
    assert card.startswith("⚠")


def test_the_same_build_with_the_static_directory_present_is_approvable() -> None:
    # The one difference that makes it correct: the directory it mounts exists.
    approvable = {**QWEN_NONBOOT, "app/static/index.html": "<html></html>\n"}
    reason, _, _ = _verdict(approvable)
    assert reason is None


# ── Trace 2: a grok-shaped build that boots and serves ───────────────────────
# Correct static mount + real (non-stub) handlers: the static rungs must leave
# it approvable (the ceiling is that it does REAL work, which no static rung can
# prove — that is the iteration loop's job, not this gate's).

GROK_CLEAN = {
    "app/main.py": (
        "from fastapi import FastAPI\n"
        "from fastapi.staticfiles import StaticFiles\n"
        "app = FastAPI()\n"
        "app.mount('/static', StaticFiles(directory='app/static'), name='static')\n"
        "@app.get('/api/run')\n"
        "async def run():\n"
        "    return {'ok': True}\n"
    ),
    "app/static/index.html": "<html></html>\n",
    "requirements.txt": "fastapi\nuvicorn\n",
}


def test_a_correct_build_stays_approvable() -> None:
    reason, _, _ = _verdict(GROK_CLEAN)
    assert reason is None


# ── Trace 3: an indentation slip (syntax rung) ───────────────────────────────

INDENT_ERROR = {
    "app/agents/base.py": "class Agent:\n    def run(self):\n    return 1\n",
}


def test_an_indentation_error_is_not_approvable() -> None:
    reason, card, _ = _verdict(INDENT_ERROR)
    assert reason is not None
    assert card.startswith("⚠")


# ── Trace 4: agents that are all stubs (wiring rung) ─────────────────────────

STUB_AGENTS = {
    "app/agents/planner.py": (
        "class Planner:\n"
        "    def run(self, data):\n"
        "        pass\n"
        "    def name(self):\n"
        "        pass\n"
    ),
}


def test_an_all_stub_module_is_not_approvable() -> None:
    reason, _, _ = _verdict(STUB_AGENTS)
    assert reason is not None


# ── Trace 5: StaticFiles built but never mounted (wiring rung) ───────────────
# The directory exists, so this isolates the not-mounted defect from Trace 1.

UNMOUNTED_STATIC = {
    "app/main.py": (
        "from fastapi import FastAPI\n"
        "from fastapi.staticfiles import StaticFiles\n"
        "app = FastAPI()\n"
        "files = StaticFiles(directory='app/static')\n"
    ),
    "app/static/index.html": "<html></html>\n",
}


def test_static_files_built_but_never_mounted_is_not_approvable() -> None:
    reason, _, _ = _verdict(UNMOUNTED_STATIC, project_paths=("app/static/index.html",))
    assert reason is not None
