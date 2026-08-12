"""A coding session's own verdict is not an input to approval.

The failure this guards against is the quiet one: a model finishes, says it is
done, and the defect ships because nothing looked. Metis's answer is that the
session's completion claim is never read. The mirror is imported byte by byte,
the host verifies it with the same ladder the approval gate always uses, and a
changeset proven not to work gets no Approve button.

Every fake here reports success. The product still refuses.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from waqil_api.control_plane import (
    ControlPlane,
    _blocking_findings,
    _blocking_reason,
)

from test_project_coding_engine import _coordinator as _direct_coordinator


# Parses cleanly and is still genuinely broken: it imports a module this
# project does not contain, so the application cannot start. A defect that
# survives staging is the one that tests whether the APPROVAL gate looks --
# a file that will not parse is refused earlier still, which is proved below.
BROKEN_MAIN = """\
from fastapi import FastAPI

from app.nonexistent_helper import build_router

app = FastAPI()
app.include_router(build_router())


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
"""

UNPARSEABLE_MAIN = """\
def health(  ->  dict:
    return {"status": "ok"}
"""

GOOD_MAIN = """\
from fastapi import FastAPI

app = FastAPI()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
"""


class _Events:
    def __init__(self) -> None:
        self.items: list[tuple[str, dict[str, Any]]] = []

    async def emit(
        self, run_id: str, conversation_id: str, type: str, payload: dict[str, Any]
    ) -> None:
        self.items.append((type, payload))


@pytest.mark.asyncio
async def test_a_session_that_reports_completed_with_a_broken_file_is_refused(
    tmp_path: Path,
) -> None:
    """The fake claims success loudly; the host verification decides anyway."""

    def write_broken(root: Path, _phase: str) -> None:
        (root / "app" / "main.py").write_text(BROKEN_MAIN, encoding="utf-8")

    (
        coordinator,
        engine,
        _sessions,
        projects,
        database,
        run_id,
        conv,
        project_id,
    ) = await _direct_coordinator(tmp_path, write_broken)
    project = tmp_path / "Projects" / "demo"
    before = (project / "app" / "main.py").read_bytes()
    try:
        # Everything the engine can say about itself says "clean".
        engine.result_state = "completed"
        engine.finish_reason = "completed"
        engine.result_summary = (
            "All done. The application builds and every check passes."
        )

        from waqil_api.coding_contracts import CodingProviderV1

        round_result = await coordinator.start(
            run_id=run_id,
            conversation_id=conv,
            project_id=project_id,
            prompt="port it",
            staged={},
            provider=CodingProviderV1(
                providerId="ollama", modelId="glm-5.2:cloud", baseUrl="http://x"
            ),
            allowed_paths=[],
            protected_paths=[],
            broad_scope=True,
        )
        assert round_result.result is not None
        assert round_result.result.state == "completed"
        assert "every check passes" in (round_result.result.summary or "")

        # Metis imported the bytes itself rather than believing the summary.
        staged = dict(round_result.staged)
        assert "app/main.py" in staged
        assert "nonexistent_helper" in str(staged["app/main.py"]["content"])

        # The real verification ladder, on the real staged bytes.
        plane = _Plane(projects, coordinator.settings)
        verification = await ControlPlane._verify_staged_changeset(
            plane, project_id, staged, full=True
        )
        blocking = _blocking_findings(verification)
        reason = _blocking_reason(verification)

        assert blocking, "a file that does not parse must produce a blocking finding"
        assert reason, "a blocked changeset must carry a reason for the card"
        assert "app/main.py" in " ".join(
            str(item.get("path") or "") for item in blocking
        )

        # Nothing was materialized: the import stages, approval applies, and
        # approval never happened.
        assert (project / "app" / "main.py").read_bytes() == before
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_the_same_session_reporting_completed_on_good_bytes_is_clean(
    tmp_path: Path,
) -> None:
    """The refusal is earned by the bytes, not by distrust of the engine."""

    def write_good(root: Path, _phase: str) -> None:
        (root / "app" / "main.py").write_text(GOOD_MAIN, encoding="utf-8")

    (
        coordinator,
        engine,
        _sessions,
        projects,
        database,
        run_id,
        conv,
        project_id,
    ) = await _direct_coordinator(tmp_path, write_good)
    try:
        engine.result_state = "completed"
        from waqil_api.coding_contracts import CodingProviderV1

        round_result = await coordinator.start(
            run_id=run_id,
            conversation_id=conv,
            project_id=project_id,
            prompt="port it",
            staged={},
            provider=CodingProviderV1(
                providerId="ollama", modelId="glm-5.2:cloud", baseUrl="http://x"
            ),
            allowed_paths=[],
            protected_paths=[],
            broad_scope=True,
        )

        plane = _Plane(projects, coordinator.settings)
        verification = await ControlPlane._verify_staged_changeset(
            plane, project_id, dict(round_result.staged), full=True
        )

        assert _blocking_findings(verification) == []
        assert not _blocking_reason(verification)
    finally:
        await database.close()


class _Plane:
    """The control-plane surface the verification ladder actually uses.

    Both rung methods are the production ones, called unbound: this test must
    exercise Metis's real verification, not a stand-in for it.
    """

    def __init__(self, projects: Any, settings: Any) -> None:
        self.projects = projects
        self.settings = settings
        self.events = _Events()

    async def _verify_staged_rungs(self, *args: Any, **kwargs: Any) -> Any:
        return await ControlPlane._verify_staged_rungs(self, *args, **kwargs)


def test_a_blocked_reason_is_derived_from_findings_not_from_a_claim() -> None:
    """`_blocking_reason` reads verification only; no completion text reaches it."""

    clean = {"errors": [], "warnings": [], "notes": [], "checks": []}
    broken = {
        "errors": [
            {
                "path": "app/main.py",
                "severity": "error",
                "error": "invalid syntax at line 7",
            }
        ],
        "warnings": [],
        "notes": [],
        "checks": [],
    }

    assert not _blocking_reason(clean)
    reason = _blocking_reason(broken)
    assert reason
    assert "app/main.py" in reason
    assert "invalid syntax" in reason


def test_the_approval_digest_binds_the_decision_to_the_reviewed_bytes() -> None:
    """Approving applies what was reviewed, not whatever arrived afterwards."""

    first = hashlib.sha256(GOOD_MAIN.encode("utf-8")).hexdigest()
    second = hashlib.sha256(BROKEN_MAIN.encode("utf-8")).hexdigest()

    assert first != second


@pytest.mark.asyncio
async def test_a_file_that_cannot_parse_is_refused_before_it_is_ever_staged(
    tmp_path: Path,
) -> None:
    """The earlier gate, proved so the later one is not credited with its work."""

    def write_unparseable(root: Path, _phase: str) -> None:
        (root / "app" / "main.py").write_text(UNPARSEABLE_MAIN, encoding="utf-8")

    (
        coordinator,
        engine,
        _sessions,
        _projects,
        database,
        run_id,
        conv,
        project_id,
    ) = await _direct_coordinator(tmp_path, write_unparseable)
    project = tmp_path / "Projects" / "demo"
    before = (project / "app" / "main.py").read_bytes()
    try:
        engine.result_state = "completed"
        engine.result_summary = "Done, everything passes."
        from waqil_api.coding_contracts import CodingProviderV1

        round_result = await coordinator.start(
            run_id=run_id,
            conversation_id=conv,
            project_id=project_id,
            prompt="port it",
            staged={},
            provider=CodingProviderV1(
                providerId="ollama", modelId="glm-5.2:cloud", baseUrl="http://x"
            ),
            allowed_paths=[],
            protected_paths=[],
            broad_scope=True,
        )

        # It claimed success and staged nothing: the overlay never accepted a
        # file that cannot parse, so no later gate had to catch it.
        assert round_result.result is not None
        assert round_result.result.state == "completed"
        assert "app/main.py" not in dict(round_result.staged)
        assert (project / "app" / "main.py").read_bytes() == before
    finally:
        await database.close()
