"""P8 — model-authored diagram code through the broker, with safe fallback.

Exercises the real ControlPlane._author_diagram_code decision (v1 copy path,
model-authored v2 path, and the canonical fallback when the model produces
invalid code) using lightweight fakes, plus the v1->v2 builtin seed upgrade.
"""
from __future__ import annotations

import tempfile
import types
from pathlib import Path

import pytest

from waqil_api.config import Settings
from waqil_api.contracts import (
    ArchitectureComponentV1,
    ArchitectureEdgeV1,
    ArchitectureBoundaryV1,
    ArchitectureSpecV1,
    DiagramCodeV1,
    ModelResultV1,
)
from waqil_api.control_plane import ControlPlane, _extract_python_source
from waqil_api.database import Database
from waqil_api.diagram_source import canonical_diagram_source, canonical_diagram_source_v2
from waqil_api.tool_registry import REFERENCE_ARCHITECTURE_SLUG, ToolRegistry


_SPEC = ArchitectureSpecV1(
    title="Test Arch",
    components=[
        ArchitectureComponentV1(id="web", label="Web Client", kind="WebApp"),
        ArchitectureComponentV1(id="api", label="API Server", kind="ApplicationServer"),
    ],
    edges=[ArchitectureEdgeV1(source="web", target="api", label="HTTP")],
    boundaries=[ArchitectureBoundaryV1(id="cloud", label="Cloud", component_ids=["api"])],
)


class _Events:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def emit(self, run_id, conversation_id, event_type, payload=None, checkpoint_id=None):
        self.events.append((event_type, payload or {}))


class _Model:
    """Model whose broker `generate` returns a scripted reply, and whose
    `diagram_code` (v1 path) copies the canonical source."""

    def __init__(self, broker_reply: str) -> None:
        self._broker_reply = broker_reply

    async def generate(self, request, on_token=None, *, model_aliases=None):
        return ModelResultV1(model="fake", content=self._broker_reply)

    async def diagram_code(self, spec, *, model_aliases=None):
        return DiagramCodeV1(diagram_code=canonical_diagram_source(spec, ["svg", "png"]))


async def _registry_with_active():
    # Model authoring on -> the active builtin is the v2 (broker-authored) def.
    db = Database(Path(tempfile.mkdtemp()) / "w.db")
    await db.open()
    settings = types.SimpleNamespace(tool_model_authoring=True)
    registry = ToolRegistry(db, settings)
    await registry.seed_builtins()
    return db, registry


def _fake_control_plane(registry, model, events):
    # _author_diagram_code only touches self.registry / self.model / self.events.
    return types.SimpleNamespace(registry=registry, model=model, events=events)


async def _author(cp_self, state):
    return await ControlPlane._author_diagram_code(cp_self, state, _SPEC)


_STATE = {"run_id": "run_1", "conversation_id": "conv_1", "model_aliases": {}}


@pytest.mark.asyncio
async def test_model_authored_valid_code_is_used() -> None:
    db, registry = await _registry_with_active()
    # The model authors a valid (v2-canonical) program, fenced as it often is.
    valid = "```python\n" + canonical_diagram_source_v2(_SPEC, ["svg", "png"]) + "```"
    events = _Events()
    cp = _fake_control_plane(registry, _Model(valid), events)
    code, validation, profile, authored_by, reason = await _author(cp, _STATE)
    assert authored_by == "model-authored"
    assert reason is None
    assert profile == "diagrams-draw-v2"
    assert validation["status"] == "passed"
    assert "Diagram(" in code
    # The brokered call was audited.
    assert any(t == "run.broker_call" for t, _ in events.events)
    await db.close()


@pytest.mark.asyncio
async def test_invalid_model_code_falls_back_to_canonical() -> None:
    db, registry = await _registry_with_active()
    # The model returns garbage (what the deterministic backend does) -> fallback.
    events = _Events()
    cp = _fake_control_plane(registry, _Model("Local deterministic response: nonsense"), events)
    code, validation, profile, authored_by, reason = await _author(cp, _STATE)
    assert authored_by == "canonical-fallback"
    assert reason is not None  # records why it fell back
    assert profile == "diagrams-draw-v2"
    assert validation["status"] == "passed"
    # Fallback is the deterministic v2 canonical source — always valid, better layout.
    assert code == canonical_diagram_source_v2(_SPEC, ["svg", "png"])
    # The run never hard-fails; the broker call was still audited.
    assert any(t == "run.broker_call" for t, _ in events.events)
    await db.close()


@pytest.mark.asyncio
async def test_no_registry_uses_v1_copy_path() -> None:
    # Without a registry (or model access), behavior is the unchanged v1 path.
    events = _Events()
    cp = _fake_control_plane(None, _Model("ignored"), events)
    code, validation, profile, authored_by, reason = await _author(cp, _STATE)
    assert authored_by == "canonical-v1"
    assert profile == "diagrams-render-v1"
    assert code == canonical_diagram_source(_SPEC, ["svg", "png"])
    assert not any(t == "run.broker_call" for t, _ in events.events)


def test_extract_python_source_strips_fences() -> None:
    fenced = "```python\nfrom pathlib import Path\nx = 1\n```"
    assert _extract_python_source(fenced) == "from pathlib import Path\nx = 1\n"
    assert _extract_python_source("raw = 1") == "raw = 1\n"


@pytest.mark.asyncio
async def test_seed_upgrades_v1_definition_to_v2() -> None:
    from waqil_api.contracts import (
        CapabilityProfileV1,
        ModelAccessV1,
        ToolDefinitionV1,
        ToolRouteFactsV1,
    )
    from waqil_api.tool_registry import finalize

    db = Database(Path(tempfile.mkdtemp()) / "w.db")
    await db.open()
    # Simulate a pre-P8 install: an active v1 definition with no model access.
    old = finalize(
        ToolDefinitionV1(
            slug=REFERENCE_ARCHITECTURE_SLUG,
            version="1",
            name="Reference Architecture Generator",
            description="v1",
            route_facts=ToolRouteFactsV1(input_pipeline="architecture_spec"),
            capability_profile=CapabilityProfileV1(
                code_allowlist="diagrams-render-v1", model_access=ModelAccessV1()
            ),
        )
    )
    await db.upsert_tool_definition(old, activate=True)
    active = await db.get_active_tool_definition(REFERENCE_ARCHITECTURE_SLUG)
    assert active.version == "1"
    assert active.capability_profile.model_access.enabled is False

    # With authoring enabled, seed_builtins upgrades the reserved builtin slug
    # to the latest host version (v2).
    settings = types.SimpleNamespace(tool_model_authoring=True)
    registry = ToolRegistry(db, settings)
    await registry.seed_builtins()
    active = await db.get_active_tool_definition(REFERENCE_ARCHITECTURE_SLUG)
    assert active.version == "2"
    assert active.capability_profile.model_access.enabled is True
    # Idempotent — a second seed is a no-op.
    await registry.seed_builtins()
    assert len(await db.list_active_tool_definitions()) == 1
    await db.close()
