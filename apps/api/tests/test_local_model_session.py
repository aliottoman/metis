"""The on-device session lifecycle: keep_alive wire format and error surfacing.

Ollama parses keep_alive with Go's time.ParseDuration, which rejects a bare
"-1" outright. These tests pin the format so an indefinite session cannot
regress into a 400 from the runtime again.
"""
from __future__ import annotations

from typing import Any

import httpx
import pytest

from waqil_api.config import Settings
from waqil_api.local_model_session import (
    KEEP_ALIVE_FOREVER,
    LocalModelSessionError,
    LocalModelSessionManager,
    keep_alive_for,
)
from waqil_api.model_preference import ModelPreferenceStore


def _manager(tmp_path) -> LocalModelSessionManager:
    settings = Settings(data_dir=tmp_path, model_backend="ollama")
    settings.prepare_directories()
    return LocalModelSessionManager(settings, ModelPreferenceStore(settings))


def test_keep_alive_for_indefinite_carries_a_unit() -> None:
    # "-1" is what Ollama rejects; any negative duration means keep forever.
    assert keep_alive_for(86_400) == KEEP_ALIVE_FOREVER
    assert KEEP_ALIVE_FOREVER.startswith("-")
    assert KEEP_ALIVE_FOREVER[-1] in {"s", "m", "h"}
    assert keep_alive_for(90_000) == KEEP_ALIVE_FOREVER


def test_keep_alive_for_bounded_windows_is_seconds() -> None:
    assert keep_alive_for(60) == "60s"
    assert keep_alive_for(1800) == "1800s"


@pytest.mark.asyncio
async def test_launch_sends_a_parseable_keep_alive(tmp_path, monkeypatch) -> None:
    manager = _manager(tmp_path)
    sent: list[dict[str, Any]] = []

    async def fake_models() -> list[Any]:
        from waqil_api.contracts import LocalModelOptionV1

        return [LocalModelOptionV1(id="qwen", name="qwen", loaded=False)]

    async def fake_post(path: str, payload: dict[str, Any], timeout: float = 120.0) -> None:
        sent.append(payload)

    monkeypatch.setattr(manager, "models", fake_models)
    monkeypatch.setattr(manager, "_post", fake_post)

    await manager.launch("qwen", 86_400, 32_768)

    assert sent[0]["keep_alive"] == KEEP_ALIVE_FOREVER
    assert manager.keep_alive == KEEP_ALIVE_FOREVER
    assert manager.settings.ollama_keep_alive == KEEP_ALIVE_FOREVER


@pytest.mark.asyncio
async def test_post_surfaces_the_runtime_explanation(tmp_path, monkeypatch) -> None:
    manager = _manager(tmp_path)

    class FakeClient:
        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def post(self, path: str, json: dict[str, Any]) -> httpx.Response:
            return httpx.Response(
                400,
                json={"error": 'time: missing unit in duration "-1"'},
                request=httpx.Request("POST", f"http://x{path}"),
            )

    monkeypatch.setattr(
        "waqil_api.local_model_session.httpx.AsyncClient",
        lambda **_: FakeClient(),
    )

    with pytest.raises(LocalModelSessionError) as error:
        await manager._post("/api/generate", {"model": "qwen"})

    # The status line alone never explains which field the runtime rejected.
    assert "missing unit in duration" in str(error.value)
    assert "400" in str(error.value)


def test_restart_restores_the_saved_idle_window(tmp_path) -> None:
    """A restart must not revert to the unload-after-every-call default."""
    settings = Settings(data_dir=tmp_path, model_backend="ollama", ollama_keep_alive="0")
    settings.prepare_directories()
    settings.model_session_path.write_text(
        '{"selected_model": "qwen", "idle_timeout_seconds": 1800, '
        '"context_window": 65536}',
        encoding="utf-8",
    )

    manager = LocalModelSessionManager(settings, ModelPreferenceStore(settings))

    assert manager.idle_timeout_seconds == 1800
    assert settings.ollama_keep_alive == "1800s"
    assert settings.context_window == 65536


def test_a_fresh_install_keeps_the_unload_after_each_call_default(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path, model_backend="ollama", ollama_keep_alive="0")
    settings.prepare_directories()

    LocalModelSessionManager(settings, ModelPreferenceStore(settings))

    assert settings.ollama_keep_alive == "0"


async def _launched(tmp_path, monkeypatch) -> tuple[LocalModelSessionManager, list[dict[str, Any]]]:
    """A manager that owns a running model, as it is after a real launch."""
    manager = _manager(tmp_path)
    sent: list[dict[str, Any]] = []
    running = {"loaded": False}

    async def fake_models() -> list[Any]:
        from waqil_api.contracts import LocalModelOptionV1

        return [LocalModelOptionV1(id="qwen", name="qwen", loaded=running["loaded"])]

    async def fake_post(path: str, payload: dict[str, Any], timeout: float = 120.0) -> None:
        sent.append(payload)
        running["loaded"] = payload.get("keep_alive") != 0

    monkeypatch.setattr(manager, "models", fake_models)
    monkeypatch.setattr(manager, "_post", fake_post)
    await manager.launch("qwen", 86_400, 32_768)
    assert manager._owned_model == "qwen"
    sent.clear()
    return manager, sent


@pytest.mark.asyncio
async def test_shutdown_gives_back_the_weights_metis_loaded(tmp_path, monkeypatch) -> None:
    """Otherwise a closed app leaves tens of gigabytes wired for the whole
    keep_alive window — forever, when the user chose "until stopped"."""
    manager, sent = await _launched(tmp_path, monkeypatch)

    assert await manager.release_owned() is True

    assert sent == [
        {"model": "qwen", "prompt": "", "stream": False, "keep_alive": 0}
    ]
    # Idempotent: a second stop has nothing of its own to release.
    assert await manager.release_owned() is False


@pytest.mark.asyncio
async def test_a_failed_unload_never_blocks_shutdown(tmp_path, monkeypatch) -> None:
    manager, _ = await _launched(tmp_path, monkeypatch)

    async def failing(*_args: Any, **_kwargs: Any) -> None:
        raise LocalModelSessionError("Ollama is unavailable")

    monkeypatch.setattr(manager, "_post", failing)

    assert await manager.release_owned() is False


@pytest.mark.asyncio
async def test_the_model_is_released_only_once_every_window_is_gone(
    tmp_path, monkeypatch
) -> None:
    manager, sent = await _launched(tmp_path, monkeypatch)

    # A client just called, so the app is still open.
    manager.touch()
    assert await manager.release_if_idle(60) is False
    assert sent == []

    # Nothing has called for longer than the window.
    manager._last_client_activity -= 120
    assert await manager.release_if_idle(60) is True
    assert len(sent) == 1


@pytest.mark.asyncio
async def test_a_model_in_use_is_never_released_underneath_a_run(
    tmp_path, monkeypatch
) -> None:
    manager, sent = await _launched(tmp_path, monkeypatch)
    manager._last_client_activity -= 10_000

    async with manager.use("qwen"):
        assert await manager.release_if_idle(60) is False

    assert sent == []


@pytest.mark.asyncio
async def test_idle_release_can_be_turned_off(tmp_path, monkeypatch) -> None:
    manager, sent = await _launched(tmp_path, monkeypatch)
    manager._last_client_activity -= 10_000

    assert await manager.release_if_idle(0) is False
    assert sent == []


@pytest.mark.asyncio
async def test_a_model_metis_did_not_launch_is_left_alone(tmp_path, monkeypatch) -> None:
    """Someone else's model is not Metis's to unload."""
    manager = _manager(tmp_path)
    manager.selected_model = "qwen"
    manager._last_client_activity -= 10_000
    sent: list[dict[str, Any]] = []

    async def fake_post(path: str, payload: dict[str, Any], timeout: float = 120.0) -> None:
        sent.append(payload)

    monkeypatch.setattr(manager, "_post", fake_post)

    assert await manager.release_if_idle(60) is False
    assert await manager.release_owned() is False
    assert sent == []


class _StubSession:
    """Just the surface the idle watchdog touches."""

    def __init__(self) -> None:
        self.released = 0
        self.touched = 0
        self.windows: list[float] = []

    def touch(self) -> None:
        self.touched += 1

    async def release_if_idle(self, after_seconds: float) -> bool:
        self.windows.append(after_seconds)
        self.released += 1
        return True

    async def release_owned(self, *, reason: str = "shutdown") -> bool:
        self.released += 1
        return True


class _StubDatabase:
    def __init__(self, active: bool) -> None:
        self.active = active

    async def has_active_runs(self) -> bool:
        return self.active


async def _run_watchdog(active: bool, seconds: float) -> _StubSession:
    import asyncio

    from waqil_api.runtime import AppRuntime

    runtime = AppRuntime.__new__(AppRuntime)
    runtime.settings = Settings(_env_file=None, model_release_after_idle_seconds=4)
    runtime.database = _StubDatabase(active)
    runtime.model_session = _StubSession()
    # The same watchdog now hands the verify sandbox its own idle clock.
    runtime.project_sandbox = None
    task = asyncio.create_task(runtime._release_idle_model())
    await asyncio.sleep(seconds)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    return runtime.model_session


@pytest.mark.asyncio
async def test_the_watchdog_asks_to_release_while_nothing_is_running(tmp_path) -> None:
    session = await _run_watchdog(active=False, seconds=2.5)

    assert session.released >= 2
    assert session.windows and all(window == 4 for window in session.windows)


@pytest.mark.asyncio
async def test_the_watchdog_holds_the_clock_open_while_a_run_is_in_flight(tmp_path) -> None:
    """A run awaiting approval makes no model call for as long as the user
    takes to answer; releasing under it would strand the turn."""
    session = await _run_watchdog(active=True, seconds=2.5)

    assert session.released == 0
    assert session.touched >= 2


@pytest.mark.asyncio
async def test_the_watchdog_stays_off_when_the_setting_is_zero() -> None:
    import asyncio

    from waqil_api.runtime import AppRuntime

    runtime = AppRuntime.__new__(AppRuntime)
    runtime.settings = Settings(_env_file=None, model_release_after_idle_seconds=0)
    runtime.database = _StubDatabase(False)
    runtime.model_session = _StubSession()

    await asyncio.wait_for(runtime._release_idle_model(), timeout=1)

    assert runtime.model_session.released == 0
