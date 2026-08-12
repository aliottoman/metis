"""Cross-language contract test for Python -> compiled TypeScript sidecar.

The fake runtime exercises the real NDJSON process, strict parsers, event
broker, and lifecycle without making a provider or model call.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import SecretStr

from waqil_api.coding_contracts import (
    CodingProviderV1,
    ContinueSliceV1,
    RestartWithModelV1,
    SliceLimitsV1,
    StartSliceV1,
)
from waqil_api.coding_engine import SidecarCodingEngine


REPO_ROOT = Path(__file__).resolve().parents[3]
SIDECAR_ENTRYPOINT = REPO_ROOT / "apps" / "cline-sidecar" / "dist" / "src" / "index.js"


@pytest.mark.skipif(
    not SIDECAR_ENTRYPOINT.is_file(),
    reason="the TypeScript sidecar must be built before cross-language integration",
)
@pytest.mark.asyncio
async def test_python_client_drives_compiled_fake_sidecar_lifecycle(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    data_dir = tmp_path / "sidecar-data"
    provider = CodingProviderV1(
        providerId="openai-compatible",
        modelId="fake-broad",
        apiKey=SecretStr("sk-cross-language-canary-123456789"),
        baseUrl="https://models.example.invalid/v1",
    )
    engine = SidecarCodingEngine(
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
        start_timeout_seconds=5,
        request_timeout_seconds=5,
        shutdown_timeout_seconds=2,
    )
    try:
        info = await engine.get_info()
        assert info.protocol_version == "1"
        assert info.engine == "clinecore"
        assert info.runtime == "fake"
        assert info.sdk_version == "0.0.72"
        assert set(info.allowed_tools) == {
            "read_files",
            "search_codebase",
            "editor",
            # The wire for run_check. Enabled in policy, never executed: the
            # sidecar accepts only a bare check name and Metis runs it.
            "run_check",
        }

        budget_stopped = await engine.start_slice(
            StartSliceV1(
                sessionId="cross_language_budget",
                prompt="[fake:iterations=25]",
                workspaceRoot=workspace.resolve(),
                provider=provider,
                limits=SliceLimitsV1(maxIterations=24),
            )
        )
        assert budget_stopped.state == "failed"
        assert budget_stopped.finish_reason == "error"
        assert budget_stopped.controlled_stop_reason == "max_iterations"
        assert budget_stopped.iterations == 24
        assert budget_stopped.summary == "Agent runtime exceeded maxIterations (24)"
        assert (await engine.delete_session(budget_stopped.session_id)).deleted is True

        started = await engine.start_slice(
            StartSliceV1(
                sessionId="cross_language_build",
                prompt="Build the bounded fake slice",
                workspaceRoot=workspace.resolve(),
                provider=provider,
            )
        )
        assert started.session_id == "cross_language_build"
        assert started.state == "completed"

        stream = engine.subscribe(started.session_id)
        event = await asyncio.wait_for(anext(stream), timeout=2)
        close_stream = getattr(stream, "aclose")
        await close_stream()
        assert event.session_id == started.session_id
        assert event.cursor >= 1

        continued = await engine.continue_slice(
            ContinueSliceV1(
                sessionId=started.session_id,
                recoverySessionId="cross_language_recovery",
                prompt="Repair the exact fake finding",
                provider=provider,
                timeoutMs=5_000,
            )
        )
        assert continued.session_id == started.session_id
        assert continued.usage.requests == 2

        restarted = await engine.restart_with_model(
            RestartWithModelV1(
                sessionId=started.session_id,
                newSessionId="cross_language_repair",
                prompt="Use the repair model",
                provider=CodingProviderV1(
                    providerId="openai-compatible",
                    modelId="fake-repair",
                    apiKey=SecretStr("sk-cross-language-canary-123456789"),
                    baseUrl="https://models.example.invalid/v1",
                ),
            )
        )
        assert restarted.parent_session_id == started.session_id
        assert restarted.model is not None
        assert restarted.model.model_id == "fake-repair"

        deleted = await engine.delete_session(restarted.session_id)
        assert deleted.deleted is True
        assert deleted.deleted_session_ids == [
            restarted.session_id,
            started.session_id,
        ]
        assert not any(
            "sk-cross-language-canary-123456789"
            in path.read_text(encoding="utf-8", errors="ignore")
            for path in data_dir.rglob("*")
            if path.is_file()
        )

        shutdown = await engine.shutdown()
        assert shutdown.state == "shutting_down"
        assert engine.running is False
    finally:
        await engine.close()
