"""Explicit, laptop-conscious lifecycle for one pinned Ollama model.

Nothing in this module starts a model implicitly. A launch is always tied to
the dedicated session endpoint (or an approval resume that the user explicitly
confirmed). Models already running before Metis starts are treated as external
and are never stopped by Metis automatically.
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any, AsyncIterator

import httpx

from .config import Settings
from .contracts import LocalModelOptionV1, LocalModelSessionV1
from .model_preference import ModelPreferenceStore


class LocalModelSessionError(RuntimeError):
    pass


def _date(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except ValueError:
        return None


class LocalModelSessionManager:
    def __init__(
        self, settings: Settings, preference: ModelPreferenceStore
    ) -> None:
        self.settings = settings
        self.preference = preference
        self._lock = asyncio.Lock()
        self._busy = 0
        self._owned_model: str | None = None
        self._state = "off"
        self._error: str | None = None
        saved = self._load()
        self.selected_model = str(saved.get("selected_model") or "") or None
        self.idle_timeout_seconds = int(
            saved.get("idle_timeout_seconds") or settings.local_model_idle_seconds
        )
        self.context_window = int(saved.get("context_window") or settings.context_window)

    @property
    def deterministic(self) -> bool:
        return self.settings.model_backend == "deterministic"

    @property
    def keep_alive(self) -> str:
        if self.idle_timeout_seconds >= 86_400:
            return "-1"
        return f"{self.idle_timeout_seconds}s"

    def _load(self) -> dict[str, Any]:
        try:
            value = json.loads(
                self.settings.model_session_path.read_text(encoding="utf-8")
            )
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save(self) -> None:
        path = self.settings.model_session_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "selected_model": self.selected_model,
                    "idle_timeout_seconds": self.idle_timeout_seconds,
                    "context_window": self.context_window,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    async def _get(self, path: str) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(
                base_url=self.settings.ollama_base_url, timeout=2.5
            ) as client:
                response = await client.get(path)
                response.raise_for_status()
                value = response.json()
                return value if isinstance(value, dict) else {}
        except (httpx.HTTPError, ValueError) as exc:
            raise LocalModelSessionError(f"Ollama is unavailable: {exc}") from exc

    async def _post(self, path: str, payload: dict[str, Any], timeout: float = 120.0) -> None:
        try:
            async with httpx.AsyncClient(
                base_url=self.settings.ollama_base_url, timeout=timeout
            ) as client:
                response = await client.post(path, json=payload)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LocalModelSessionError(f"Ollama could not update the model: {exc}") from exc

    async def models(self) -> list[LocalModelOptionV1]:
        if self.deterministic:
            return [
                LocalModelOptionV1(
                    id="deterministic", name="Deterministic test model", loaded=True,
                    owned_by_metis=True,
                )
            ]
        tags, running = await asyncio.gather(self._get("/api/tags"), self._get("/api/ps"))
        live: dict[str, dict[str, Any]] = {}
        for row in running.get("models", []):
            if isinstance(row, dict):
                name = str(row.get("name") or row.get("model") or "")
                if name:
                    live[name] = row
        result: list[LocalModelOptionV1] = []
        for row in tags.get("models", []):
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or row.get("model") or "")
            if not name:
                continue
            details = row.get("details") if isinstance(row.get("details"), dict) else {}
            loaded_row = live.get(name)
            result.append(
                LocalModelOptionV1(
                    id=name,
                    name=name,
                    size_bytes=max(0, int(row.get("size") or 0)),
                    parameter_size=str(details.get("parameter_size") or ""),
                    quantization=str(details.get("quantization_level") or ""),
                    context_length=(
                        int(loaded_row.get("context_length"))
                        if loaded_row and loaded_row.get("context_length")
                        else None
                    ),
                    loaded=loaded_row is not None,
                    expires_at=_date(loaded_row.get("expires_at")) if loaded_row else None,
                    owned_by_metis=name == self._owned_model,
                )
            )
        return sorted(result, key=lambda item: item.name.casefold())

    async def status(self, *, include_models: bool = True) -> LocalModelSessionV1:
        try:
            models = await self.models()
            selected = next(
                (item for item in models if item.id == self.selected_model), None
            )
            if self._busy:
                state = "busy"
            elif selected and selected.loaded:
                state = "ready"
            else:
                state = "off"
                if self._owned_model == self.selected_model:
                    self._owned_model = None
            self._state, self._error = state, None
            return LocalModelSessionV1(
                state=state,
                selected_model=self.selected_model,
                idle_timeout_seconds=self.idle_timeout_seconds,
                context_window=self.context_window,
                expires_at=selected.expires_at if selected else None,
                owned_by_metis=bool(
                    selected and selected.id == self._owned_model
                ),
                busy_count=self._busy,
                models=models if include_models else [],
            )
        except LocalModelSessionError as exc:
            self._state, self._error = "error", str(exc)
            return LocalModelSessionV1(
                state="error",
                selected_model=self.selected_model,
                idle_timeout_seconds=self.idle_timeout_seconds,
                context_window=self.context_window,
                busy_count=self._busy,
                error=str(exc),
            )

    async def launch(
        self, model: str, idle_timeout_seconds: int, context_window: int
    ) -> LocalModelSessionV1:
        async with self._lock:
            self._state, self._error = "loading", None
            if self.deterministic:
                model = "deterministic"
            else:
                inventory = await self.models()
                available = {item.id for item in inventory}
                if model not in available:
                    self._state = "error"
                    raise LocalModelSessionError(f"Installed model not found: {model}")
                for item in inventory:
                    if not item.loaded or item.id == model:
                        continue
                    if item.id == self._owned_model:
                        await self._post(
                            "/api/generate",
                            {
                                "model": item.id, "prompt": "", "stream": False,
                                "keep_alive": 0,
                            },
                            timeout=30.0,
                        )
                    else:
                        raise LocalModelSessionError(
                            f"{item.id} is already running outside Metis. Stop it in "
                            "Ollama before launching another large model."
                        )
                await self._post(
                    "/api/generate",
                    {
                        "model": model,
                        "prompt": "",
                        "stream": False,
                        "keep_alive": "-1"
                        if idle_timeout_seconds >= 86_400
                        else f"{idle_timeout_seconds}s",
                        "options": {"num_ctx": context_window},
                    },
                )
            self.selected_model = model
            self.idle_timeout_seconds = idle_timeout_seconds
            self.context_window = context_window
            self.settings.context_window = context_window
            self.settings.ollama_keep_alive = self.keep_alive
            self._owned_model = model
            self._state = "ready"
            self.preference.save("pinned", model, provider="local", oci_tools=[])
            self._save()
        return await self.status()

    async def stop(self, *, force: bool = False) -> LocalModelSessionV1:
        async with self._lock:
            model = self.selected_model
            if model and not self.deterministic:
                if self._owned_model != model and not force:
                    raise LocalModelSessionError(
                        "This model was launched outside Metis and was left running."
                    )
                await self._post(
                    "/api/generate",
                    {"model": model, "prompt": "", "stream": False, "keep_alive": 0},
                    timeout=30.0,
                )
            self._owned_model = None
            self._state, self._error = "off", None
        return await self.status()

    async def require_ready(self, model: str | None = None) -> None:
        if self.deterministic:
            return
        value = await self.status(include_models=False)
        expected = model or self.selected_model
        if value.state not in {"ready", "busy"} or not expected:
            raise LocalModelSessionError("Launch a local model before sending this request.")
        if expected != self.selected_model:
            raise LocalModelSessionError(
                f"This run is pinned to {expected}. Launch that exact model to continue."
            )

    async def relaunch_pinned(self, model: str) -> LocalModelSessionV1:
        if self.deterministic:
            return await self.status()
        if not model:
            raise LocalModelSessionError("The paused run has no pinned local model.")
        return await self.launch(model, self.idle_timeout_seconds, self.context_window)

    @asynccontextmanager
    async def use(self, model: str) -> AsyncIterator[None]:
        await self.require_ready(model)
        self._busy += 1
        self._state = "busy"
        try:
            yield
        finally:
            self._busy = max(0, self._busy - 1)
            self._state = "ready" if self._busy == 0 else "busy"
