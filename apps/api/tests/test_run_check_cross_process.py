"""`run_check` across the real process boundary, in both directions.

Everything here is real except the model: the compiled TypeScript sidecar runs
as a child process, the NDJSON transport carries the frames, the host bridge
issues the outbound `hostCall`, and the Python engine answers it. The fake
runtime supplies scripted edits and check requests in place of a provider.

What this proves that the two half-tests could not: the request actually
leaves the sidecar, the answer actually arrives, and the session and workspace
survive from the first edit through the blocker to the clean recheck.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from waqil_api.coding_contracts import (
    CodingProviderV1,
    ContinueSliceV1,
    HostCheck,
    HostCheckResultV1,
    StartSliceV1,
)
from waqil_api.coding_engine import SidecarCodingEngine


REPO_ROOT = Path(__file__).resolve().parents[3]
SIDECAR_ENTRYPOINT = REPO_ROOT / "apps" / "cline-sidecar" / "dist" / "src" / "index.js"

requires_sidecar = pytest.mark.skipif(
    not SIDECAR_ENTRYPOINT.is_file(),
    reason="the TypeScript sidecar must be built before cross-language integration",
)


def _engine(data_dir: Path) -> SidecarCodingEngine:
    return SidecarCodingEngine(
        [
            "node",
            str(SIDECAR_ENTRYPOINT),
            "--stdio",
            "--runtime",
            "fake",
            "--data-dir",
            str(data_dir),
        ],
        cwd=REPO_ROOT,
        start_timeout_seconds=10,
        request_timeout_seconds=15,
        shutdown_timeout_seconds=3,
    )


def _provider() -> CodingProviderV1:
    return CodingProviderV1(
        providerId="openai-compatible",
        modelId="fake-direct",
        apiKey=SecretStr("sk-cross-process-canary-123456789"),
        baseUrl="https://models.example.invalid/v1",
    )


class _Host:
    """Metis's side of the bridge: it owns the argv and reads the workspace."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.calls: list[tuple[str, HostCheck]] = []

    async def __call__(self, session_id: str, check: HostCheck) -> HostCheckResultV1:
        self.calls.append((session_id, check))
        # The real handler runs the pinned verifier; here the same shape is
        # produced from the same input it would read -- the live workspace.
        main = (self.workspace / "app" / "main.py").read_text(encoding="utf-8")
        requirements_path = self.workspace / "requirements.txt"
        requirements = (
            requirements_path.read_text(encoding="utf-8")
            if requirements_path.is_file()
            else ""
        )
        if "UploadFile" in main and "multipart" not in requirements:
            return HostCheckResultV1(
                check=check,
                ok=False,
                errors=1,
                findings=[
                    {
                        "path": "app/main.py",
                        "severity": "error",
                        "detail": (
                            "uses fastapi.File, which needs python-multipart "
                            "declared in requirements"
                        ),
                    }
                ],
            )
        return HostCheckResultV1(check=check, ok=True, errors=0)


MAIN_WITH_UPLOAD = "from fastapi import FastAPI, File, UploadFile\\n\\napp = FastAPI()"


@requires_sidecar
@pytest.mark.asyncio
async def test_a_session_edits_checks_repairs_and_rechecks_across_the_pipe(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "app").mkdir(parents=True)
    host = _Host(workspace)
    engine = _engine(tmp_path / "sidecar-data")
    engine.set_check_handler(host)
    try:
        # ── round 1: the session writes the app and its requirements, then
        #    asks for a check. The check request travels sidecar -> host.
        started = await engine.start_slice(
            StartSliceV1(
                sessionId="metis-direct-1",
                prompt=(
                    f"[fake:write=app/main.py:{MAIN_WITH_UPLOAD}]"
                    "[fake:write=requirements.txt:fastapi>=0.110]"
                    "[fake:check=imports]"
                ),
                workspaceRoot=workspace,
                provider=_provider(),
            )
        )
        assert started.session_id == "metis-direct-1"

        # The host was actually asked, by the session that is running.
        assert host.calls == [("metis-direct-1", "imports")]
        # And the model actually received the blocker: the sidecar turned the
        # host's structured result into this session's own event.
        events = await _events(engine, "metis-direct-1")
        checks = [item for item in events if item.tool == "run_check"]
        assert checks, "the check must be observable on the session's own stream"
        assert any("1 error(s)" in (item.message or "") for item in checks)
        assert any("python-multipart" in (item.message or "") for item in checks)

        # ── round 2: the SAME session repairs the file and rechecks ─────────
        continued = await engine.continue_slice(
            ContinueSliceV1(
                sessionId="metis-direct-1",
                prompt=(
                    "[fake:write=requirements.txt:fastapi>=0.110\\npython-multipart>=0.0.9]"
                    "[fake:check=imports]"
                ),
            )
        )

        assert continued.session_id == "metis-direct-1"
        assert continued.state == "completed"
        # Two checks, both from the one session, and the second is clean.
        assert host.calls == [
            ("metis-direct-1", "imports"),
            ("metis-direct-1", "imports"),
        ]
        after = await _events(engine, "metis-direct-1")
        clean = [
            item
            for item in after
            if item.tool == "run_check" and "clean" in (item.message or "")
        ]
        assert clean, "the recheck must come back clean on the same session"

        # The workspace was retained throughout: the repair landed beside the
        # first round's file rather than in a fresh mirror.
        assert (workspace / "app" / "main.py").is_file()
        assert "python-multipart" in (workspace / "requirements.txt").read_text(
            encoding="utf-8"
        )
    finally:
        await engine.shutdown()


@requires_sidecar
@pytest.mark.asyncio
async def test_the_session_is_told_when_no_host_check_is_available(
    tmp_path: Path,
) -> None:
    """No handler installed: the turn continues with an honest answer."""

    workspace = tmp_path / "workspace"
    (workspace / "app").mkdir(parents=True)
    engine = _engine(tmp_path / "sidecar-data")
    # Deliberately not installed.
    try:
        await engine.start_slice(
            StartSliceV1(
                sessionId="metis-direct-2",
                prompt=(
                    f"[fake:write=app/main.py:{MAIN_WITH_UPLOAD}][fake:check=full]"
                ),
                workspaceRoot=workspace,
                provider=_provider(),
            )
        )

        events = await _events(engine, "metis-direct-2")
        checks = [item for item in events if item.tool == "run_check"]
        assert checks
        # An unavailable check is a fact the session is told, not a hang and
        # not an exception that ends the turn.
        assert any(item.status == "failed" for item in checks)
    finally:
        await engine.shutdown()


async def _events(engine: Any, session_id: str) -> list[Any]:
    """Every event replayed for a session so far, then stop listening."""

    collected: list[Any] = []
    stream = engine.subscribe(session_id, after_cursor=0)
    try:
        while True:
            try:
                collected.append(
                    await asyncio.wait_for(stream.__anext__(), timeout=0.5)
                )
            except (asyncio.TimeoutError, StopAsyncIteration):
                break
    finally:
        await stream.aclose()
    return collected
