"""The planner/coder split: one model plans, another writes.

The case these exist for is on record. Given a correctly planned four-file
conversion, narrowed by the host to ONE file, a capable coder read twenty-one
times and wrote nothing — it was never short of context, it was simply free to
keep looking. Current plans make that split compact: the planner owns scope and
dependency order once, the host pins the first outstanding file, and the coder
writes it with reads closed. Legacy direction coverage remains for checkpoints
created before that compact contract.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from waqil_api.contracts import (
    ProjectAgentStepV1,
    ProjectDirectionV1,
    ProjectToolCallV1,
)
from waqil_api.control_plane import (
    _FOCUSED_READ_ALLOWANCE,
    ControlPlane,
    _directed_attention,
)
from waqil_api.model_provider import ModelProviderError, PermanentModelError


async def _noop(*args: object, **kwargs: object) -> None:
    return None


async def _empty_context(project_id: str) -> dict[str, object]:
    return {"manifest": {"file_tree": []}}


def _plane(model: object, **overrides: Any) -> ControlPlane:
    plane = object.__new__(ControlPlane)
    plane.model = model
    plane.events = SimpleNamespace(emit=_noop)
    plane.projects = SimpleNamespace(context=_empty_context, execute_staged=None)
    settings: dict[str, Any] = {
        "project_agent_max_steps": 48,
        "project_staged_max_files": 48,
        "project_spec_rewrite": False,
        "project_spec_rewrite_max_chars": 1800,
        "project_reference_enabled": False,
        "project_reference_dir": Path("/none"),
        "project_reference_max_chars": 0,
        "project_reference_max_chars_local": 0,
        "project_repo_map_enabled": False,
        "project_orchestrator_enabled": True,
        "project_orchestrator_max_attempts": 3,
    }
    settings.update(overrides)
    plane.settings = SimpleNamespace(**settings)
    plane._guard = _noop
    plane._stage = _noop
    return plane


def _state(**extra: Any) -> dict[str, Any]:
    return {
        "prompt": "Convert this app to FastAPI: app/main.py and app/static/index.html.",
        "run_id": "run_x",
        "conversation_id": "conv_x",
        "model_aliases": {"_project_id": "asset_x"},
        "project_planned_files": ["app/main.py", "app/static/index.html"],
        "project_build_intent": "build",
        **extra,
    }


# ── The controller's rules ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_orchestrator_names_the_file_and_the_coder_is_told_what_to_write() -> (
    None
):
    class Model:
        async def project_direction(self, request, *, model_aliases=None):
            assert "planned_files" in request
            return ProjectDirectionV1(
                path="app/main.py",
                instruction="Define create_app() returning FastAPI and mount appkit at /static.",
                reuse=["appkit.web.mount_appkit_static"],
            )

    plane = _plane(Model())
    direction = await ControlPlane._project_direct(
        plane, _state(), {}, {}, ["app/main.py", "app/static/index.html"], 3
    )
    assert direction["path"] == "app/main.py"
    assert "create_app()" in direction["instruction"]
    assert direction["reuse"] == ["appkit.web.mount_appkit_static"]
    assert direction["attempts"] == {"app/main.py": 1}

    brief = _directed_attention(direction)
    assert "Write ONLY app/main.py" in brief
    assert "create_app()" in brief  # the orchestrator's words, whole
    assert "appkit.web.mount_appkit_static" in brief
    assert "Reads are closed" in brief


@pytest.mark.asyncio
async def test_the_compact_manifest_routes_without_a_second_planner_call() -> None:
    class MustNotRun:
        async def project_direction(self, request, *, model_aliases=None):
            raise AssertionError("the one-shot manifest already owns this judgment")

    direction = await ControlPlane._project_direct(
        _plane(MustNotRun()),
        _state(project_plan_taken=True),
        {},
        {},
        ["app/main.py", "app/static/index.html"],
        3,
    )

    assert direction["path"] == "app/main.py"
    assert "full user request" in direction["instruction"]
    assert direction["attempts"] == {"app/main.py": 1}


@pytest.mark.asyncio
async def test_the_compact_manifest_prefetches_prior_dependencies() -> None:
    calls = 0

    class Model:
        async def project_direction(self, request, *, model_aliases=None):
            nonlocal calls
            calls += 1
            raise AssertionError("a compact plan must not make a direction call")

    state = _state(project_plan_taken=True)
    first = await ControlPlane._project_direct(
        _plane(Model()),
        state,
        {},
        {},
        ["app/config.py", "app/main.py"],
        3,
    )
    assert first["path"] == "app/config.py"
    assert "full user request" in first["instruction"]
    assert calls == 0

    second = await ControlPlane._project_direct(
        _plane(Model()),
        state,
        {},
        {"app/config.py": {"content": "VALUE = 1\n", "bytes": 10}},
        ["app/config.py", "app/main.py"],
        4,
    )
    assert second["path"] == "app/main.py"
    assert "full user request" in second["instruction"]
    assert second["read"] == ["app/config.py"]
    assert calls == 0


@pytest.mark.asyncio
async def test_an_active_repair_outranks_the_next_compact_direction() -> None:
    class Model:
        def __init__(self) -> None:
            self.attention = ""

        async def project_plan_files(self, request, *, model_aliases=None):
            raise AssertionError("the plan already exists")

        async def project_direction(self, request, *, model_aliases=None):
            raise AssertionError("an active repair must not be overwritten")

        async def project_step(self, request, *, model_aliases=None):
            self.attention = request["attention"]
            return ProjectAgentStepV1(
                status="tool",
                tool_call=ProjectToolCallV1(
                    name="apply_patch",
                    arguments={
                        "path": "app/main.py",
                        "original": "broken = True",
                        "replacement": "broken = False",
                    },
                ),
            )

    async def execute_staged(project_id, call, staged, extra):
        return {"content": staged["app/main.py"]["content"]}, None

    model = Model()
    plane = _plane(model)
    plane.projects = SimpleNamespace(
        context=_empty_context,
        execute_staged=execute_staged,
    )
    result = await ControlPlane._project_step(
        plane,
        _state(
            project_plan_taken=True,
            project_iterations=4,
            project_staged={"app/main.py": {"content": "broken = True\n", "bytes": 14}},
            project_direction={
                "path": "app/main.py",
                "instruction": "Repair the verifier's create_app failure.",
                "reuse": [],
                "read": [],
            },
            project_focus_path="app/main.py",
        ),
    )

    assert "Repair the verifier's create_app failure" in model.attention
    assert "Implement app/static/index.html" not in model.attention
    assert result["project_pending_call"]["arguments"]["path"] == "app/main.py"


@pytest.mark.asyncio
async def test_a_path_outside_the_plan_is_refused_not_obeyed() -> None:
    """Scope belongs to the plan. Directing outside it is not a decision the
    controller may follow, however confidently it is stated."""

    class Wanderer:
        async def project_direction(self, request, *, model_aliases=None):
            return ProjectDirectionV1(path="/etc/passwd", instruction="do it")

    direction = await ControlPlane._project_direct(
        _plane(Wanderer()), _state(), {}, {}, ["app/main.py"], 3
    )
    assert direction["path"] == "app/main.py"


@pytest.mark.asyncio
async def test_naming_the_same_path_again_is_a_repair_until_the_cap() -> None:
    class Repairer:
        async def project_direction(self, request, *, model_aliases=None):
            return ProjectDirectionV1(path="app/main.py", instruction="fix the import")

    plane = _plane(Repairer())
    attempts: dict[str, int] = {}
    for expected in (1, 2, 3):
        direction = await ControlPlane._project_direct(
            plane,
            _state(project_direction_attempts=attempts),
            {},
            {},
            ["app/main.py"],
            3,
        )
        assert direction["path"] == "app/main.py"
        assert direction["attempts"]["app/main.py"] == expected
        attempts = direction["attempts"]
    # Past the cap the host stops offering it — and says what it gave up on,
    # rather than letting the file quietly vanish from the menu.
    exhausted = await ControlPlane._project_direct(
        _plane(Repairer()),
        _state(project_direction_attempts=attempts),
        {},
        {},
        ["app/main.py"],
        3,
    )
    assert "path" not in exhausted
    assert [item["path"] for item in exhausted["exhausted"]] == ["app/main.py"]


@pytest.mark.asyncio
async def test_done_returns_the_turn_to_the_ordinary_loop_and_never_ends_it() -> None:
    """An orchestrator that could end a turn would be deciding scope AND
    completion — the two things this split separates."""

    class Finisher:
        async def project_direction(self, request, *, model_aliases=None):
            return ProjectDirectionV1(done=True, reason="the plan is satisfied")

    direction = await ControlPlane._project_direct(
        _plane(Finisher()), _state(), {}, {}, ["app/main.py"], 3
    )
    assert direction == {"done": True, "reason": "the plan is satisfied"}
    assert "path" not in direction


@pytest.mark.asyncio
async def test_a_staged_file_is_not_directed_again() -> None:
    class Model:
        async def project_direction(self, request, *, model_aliases=None):
            assert request["available_files"] == ["app/static/index.html"]
            return ProjectDirectionV1(
                path="app/static/index.html", instruction="write it"
            )

    staged = {"app/main.py": {"content": "x", "origin": "create", "bytes": 1}}
    direction = await ControlPlane._project_direct(
        _plane(Model()),
        _state(),
        {},
        staged,
        ["app/main.py", "app/static/index.html"],
        3,
    )
    assert direction["path"] == "app/static/index.html"


@pytest.mark.asyncio
async def test_a_lost_direction_falls_back_to_the_undirected_loop() -> None:
    """The split is an improvement, never a dependency: a lane that cannot
    answer must degrade to the loop that worked before it existed — while
    saying so, which is what the credits-exhausted run showed it must."""

    class Broken:
        async def project_direction(self, request, *, model_aliases=None):
            raise RuntimeError("the planner lane is down")

    direction = await ControlPlane._project_direct(
        _plane(Broken()), _state(), {}, {}, ["app/main.py"], 3
    )
    assert "path" not in direction  # nothing is directed …
    assert (
        direction["failed"] == "the planner lane is down"
    )  # … and why is on the record


@pytest.mark.asyncio
async def test_the_split_can_be_switched_off() -> None:
    class MustNotRun:
        async def project_direction(self, request, *, model_aliases=None):
            raise AssertionError("the orchestrator must not be consulted")

    direction = await ControlPlane._project_direct(
        _plane(MustNotRun(), project_orchestrator_enabled=False),
        _state(),
        {},
        {},
        ["app/main.py"],
        3,
    )
    assert direction == {}


# ── The coder's half of the gate ───────────────────────────────────────────


def test_a_directed_request_narrows_to_one_file_and_says_reads_are_closed() -> None:
    plane = _plane(object())
    request = ControlPlane._project_step_request(
        plane,
        _state(
            project_direction={
                "path": "app/main.py",
                "instruction": "Define create_app().",
                "reuse": ["appkit.web.page"],
                "read": [],
            },
            project_focus_path="app/main.py",
        ),
        {"manifest": {"file_tree": []}},
        [],
        {},
        3,
        ["app/main.py", "app/static/index.html"],
    )
    # One file offered, which is what pins the write target on every lane.
    assert request["files_still_to_write"] == ["app/main.py"]
    assert request["reads_closed"] is True
    assert request["reuse_existing"] == ["appkit.web.page"]
    assert "Define create_app()." in request["attention"]


@pytest.mark.asyncio
async def test_a_directed_coder_may_not_read_at_all() -> None:
    """Undirected, a narrowed turn gets a small read allowance. Directed, it
    gets none: the orchestrator named what was needed and the host fetched it,
    so a read here is the exact drift the split removes."""
    plane = _plane(object())

    async def must_not_read(project_id, call, staged, extra):
        raise AssertionError("a directed coder's read must never reach the workspace")

    plane.projects = SimpleNamespace(
        context=_empty_context, execute_staged=must_not_read
    )
    state = _state(
        project_direction={
            "path": "app/main.py",
            "instruction": "write it",
            "reuse": [],
            "read": [],
        },
        project_focus_path="app/main.py",
        project_consecutive_reads=0,
        project_trace=[],
        project_pending_call={
            "name": "read_file",
            "arguments": {"path": "app/other.py"},
        },
    )
    evidence = await ControlPlane._project_execute(plane, state)  # type: ignore[arg-type]
    trace = evidence["project_trace"]
    assert trace[-1]["result"]["ok"] is False
    assert "reads are closed" in trace[-1]["result"]["error"]
    # The undirected allowance is genuinely different, so this is a real gate
    # rather than the old one renamed.
    assert _FOCUSED_READ_ALLOWANCE > 0


@pytest.mark.asyncio
async def test_the_host_reads_for_the_coder_before_closing_reads() -> None:
    """Closing reads is only fair because the host pays for them first."""
    seen: list[str] = []

    async def execute_staged(project_id, call, staged, extra):
        seen.append(call.arguments["path"])
        return {"content": f"contents of {call.arguments['path']}"}, None

    plane = _plane(object())
    plane.projects = SimpleNamespace(
        context=_empty_context, execute_staged=execute_staged
    )
    trace = await ControlPlane._prefetch_for_coder(
        plane,
        _state(),  # type: ignore[arg-type]
        "asset_x",
        {},
        # A legacy direction may redundantly name its own target as a read.
        # It must still be fetched once, last, so trace truncation retains it.
        {"path": "app/main.py", "read": ["app/main.py", "app/models.py"]},
        [],
    )
    # References are fetched first and the target last so the newest-first
    # trace bound can never discard the bytes an edit or repair must change.
    assert seen == ["app/models.py", "app/main.py"]
    assert [entry["arguments"]["path"] for entry in trace] == seen
    assert all(entry["result"]["ok"] for entry in trace)


@pytest.mark.asyncio
async def test_a_prefetch_that_fails_is_evidence_not_an_error() -> None:
    async def execute_staged(project_id, call, staged, extra):
        raise FileNotFoundError("app/ghost.py does not exist")

    plane = _plane(object())
    plane.projects = SimpleNamespace(
        context=_empty_context, execute_staged=execute_staged
    )
    trace = await ControlPlane._prefetch_for_coder(
        plane,
        _state(),
        "asset_x",
        {},  # type: ignore[arg-type]
        {"path": "app/ghost.py", "read": []},
        [],
    )
    assert trace[-1]["result"]["ok"] is False
    assert "does not exist" in trace[-1]["result"]["error"]


# ── End to end, with a scripted orchestrator and coder ─────────────────────


@pytest.mark.asyncio
async def test_the_compact_directed_arc_writes_the_first_planned_file() -> None:
    class Model:
        def __init__(self) -> None:
            self.brief: str = ""
            self.scenarios: list[dict[str, Any]] = []

        async def project_plan_files(self, request, *, model_aliases=None):
            raise AssertionError("the plan is already taken in this state")

        async def project_direction(self, request, *, model_aliases=None):
            raise AssertionError("the compact manifest already owns ordering")

        async def project_step(self, request, *, model_aliases=None):
            self.brief = request["attention"]
            self.scenarios = request["acceptance_scenarios"]
            return ProjectAgentStepV1(
                status="tool",
                tool_call=ProjectToolCallV1(
                    name="create_file",
                    arguments={
                        "path": "app/main.py",
                        "content": "def create_app():\n    ...\n",
                    },
                ),
            )

    async def execute_staged(project_id, call, staged, extra):
        return {"content": "def mount_appkit_static(app): ...\n"}, None

    model = Model()
    plane = _plane(model)
    plane.projects = SimpleNamespace(
        context=_empty_context, execute_staged=execute_staged
    )
    result = await ControlPlane._project_step(
        plane,
        _state(
            project_plan_taken=True,
            project_iterations=3,
            project_planned_scenarios=[
                {"name": "health", "method": "GET", "path": "/health"}
            ],
        ),  # type: ignore[arg-type]
    )
    # The coder was pinned to the first dependency-ordered file without a
    # second planner round trip.
    assert "Write ONLY app/main.py" in model.brief
    assert "full user request" in model.brief
    assert model.scenarios == [{"name": "health", "method": "GET", "path": "/health"}]
    # And the controller recorded who was directed, so a repair is countable.
    assert result["project_direction"]["path"] == "app/main.py"
    assert result["project_direction_attempts"] == {"app/main.py": 1}
    assert result["project_focus_path"] == "app/main.py"


@pytest.mark.asyncio
async def test_directed_coder_receives_the_overlay_interface_map() -> None:
    """The second planned file gets exact staged dependency signatures."""
    interface_calls: list[dict[str, Any]] = []

    class Model:
        request: dict[str, Any] = {}

        async def project_direction(self, request, *, model_aliases=None):
            raise AssertionError("the compact manifest already owns ordering")

        async def project_step(self, request, *, model_aliases=None):
            self.request = request
            return ProjectAgentStepV1(
                status="tool",
                tool_call=ProjectToolCallV1(
                    name="create_file",
                    arguments={
                        "path": "app/service.py",
                        "content": "from app.contracts import Document\n",
                    },
                ),
            )

    async def repo_map(project_id, *, request, max_chars):
        return "ranked overview"

    async def interface_map(project_id, **kwargs):
        interface_calls.append(kwargs)
        return (
            "Exact interface map for target app/service.py\n"
            "app/contracts.py [earlier dependency]\n"
            "  def parse_document(payload: bytes) -> Document"
        )

    async def execute_staged(project_id, call, staged, extra):
        return {
            "content": staged.get(call.arguments["path"], {}).get("content", "")
        }, None

    model = Model()
    plane = _plane(
        model,
        project_repo_map_enabled=True,
        project_repo_map_max_chars=6_000,
        project_repo_map_max_chars_local=4_000,
    )
    plane.projects = SimpleNamespace(
        context=_empty_context,
        repo_map=repo_map,
        interface_map=interface_map,
        execute_staged=execute_staged,
    )
    staged = {
        "app/contracts.py": {
            "content": "class Document:\n    text: str\n",
            "origin": "create",
            "bytes": 30,
        }
    }
    await ControlPlane._project_step(
        plane,
        _state(
            prompt="Add app/service.py using app/contracts.py.",
            project_plan_taken=True,
            project_iterations=3,
            project_planned_files=["app/contracts.py", "app/service.py"],
            project_staged=staged,
        ),  # type: ignore[arg-type]
    )

    assert len(interface_calls) == 1
    assert interface_calls[0]["target_path"] == "app/service.py"
    assert interface_calls[0]["dependency_paths"] == ["app/contracts.py"]
    assert interface_calls[0]["staged"] == staged
    assert interface_calls[0]["max_chars"] == 4_000
    context_map = model.request["project_context"]["repo_map"]
    assert context_map.startswith("ranked overview\n\nExact interface map")
    assert "parse_document(payload: bytes) -> Document" in context_map


def test_the_planner_rung_runs_in_its_own_seat_not_the_run_s_lane() -> None:
    """`_provider` is global to a run, so without this the planner ladder was
    decorative and the orchestrator ran on whatever chat was pinned to."""
    from waqil_api.control_plane import _chain_step_aliases, _planner_chain

    aliases = {
        "_provider": "local",
        "coder": "qwen3-coder:30b",
        "_chain_planner": '[{"provider": "cline", "model": "cline-pass/glm-5.2"}]',
    }
    routed = _chain_step_aliases(aliases, _planner_chain(aliases)[0], role="planner")
    assert routed["_provider"] == "cline"
    assert routed["_cline_model"] == "cline-pass/glm-5.2"
    # The coder alias is untouched: the two seats are independent.
    assert routed["coder"] == "qwen3-coder:30b"


def test_no_planner_chain_leaves_the_run_s_lane_alone() -> None:
    from waqil_api.control_plane import _planner_chain

    aliases = {"_provider": "cohere", "coder": "x"}
    assert _planner_chain(aliases) == [{"provider": "cohere", "model": None}]


def test_a_local_planner_rung_changes_the_planner_not_the_coder() -> None:
    from waqil_api.control_plane import _chain_step_aliases, _planner_chain

    aliases = {
        "_provider": "local",
        "planner": "qwen3.6:35b-mlx",
        "coder": "qwen3-coder:30b",
        "_chain_planner": '[{"provider": "local", "model": "glm-5.2:cloud"}]',
    }
    routed = _chain_step_aliases(aliases, _planner_chain(aliases)[0], role="planner")
    assert routed["planner"] == "glm-5.2:cloud"
    assert routed["coder"] == "qwen3-coder:30b"


def test_cline_without_an_explicit_coder_chain_uses_its_coder_default() -> None:
    from waqil_api.control_plane import _coder_chain

    chain = _coder_chain(
        {
            "_provider": "cline",
            "coder": "north-mini-code-1.0:mlx-nvfp4",
            "_fallbacks_coder": ('[{"provider":"cline","model":"cline-pass/kimi-k3"}]'),
        }
    )
    assert chain == [
        {"provider": "cline", "model": None},
        {"provider": "cline", "model": "cline-pass/kimi-k3"},
    ]


@pytest.mark.asyncio
async def test_an_optional_spec_failure_does_not_exhaust_manifest_planning() -> None:
    calls: list[str] = []

    class Model:
        async def project_spec(self, request, *, model_aliases=None):
            calls.append("spec")
            raise RuntimeError("the richer optional contract was not decoded")

        async def project_plan_files(self, request, *, model_aliases=None):
            calls.append("manifest")
            return "compact plan"

    state = _state(project_planner_chain_index=0)
    plane = _plane(Model())

    with pytest.raises(RuntimeError, match="optional contract"):
        await ControlPlane._project_planner_call(plane, state, "project_spec", {})
    assert state["project_planner_chain_index"] == 0
    assert (
        await ControlPlane._project_planner_call(plane, state, "project_plan_files", {})
        == "compact plan"
    )
    assert calls == ["spec", "manifest"]


@pytest.mark.asyncio
async def test_two_refused_repairs_advance_to_the_next_clinepass_coder() -> None:
    emitted: list[tuple[str, dict[str, Any]]] = []

    async def emit(run_id, conversation_id, kind, payload):
        emitted.append((kind, payload))

    async def refuse(*args, **kwargs):
        raise RuntimeError("the patch did not match the staged bytes")

    plane = _plane(object())
    plane.events = SimpleNamespace(emit=emit)
    plane.projects = SimpleNamespace(context=_empty_context, execute_staged=refuse)
    result = await ControlPlane._project_execute(
        plane,
        _state(
            model_aliases={
                "_project_id": "asset_x",
                "_provider": "cline",
                "_fallbacks_coder": (
                    '[{"provider":"cline","model":"cline-pass/kimi-k3"}]'
                ),
            },
            project_trace=[],
            project_staged={"app/main.py": {"content": "broken = True\n", "bytes": 14}},
            project_pending_call={
                "name": "apply_patch",
                "arguments": {
                    "path": "app/main.py",
                    "original": "missing",
                    "replacement": "fixed",
                },
            },
            project_direction={
                "path": "app/main.py",
                "instruction": "Fix the verifier finding.",
            },
            project_focus_path="app/main.py",
            project_syntax_retries=1,
            project_refused_streak=1,
            project_chain_index=0,
            project_blocked_targets={},
        ),  # type: ignore[arg-type]
    )

    assert result["project_chain_index"] == 1
    assert result["project_refused_streak"] == 0
    fallback = [payload for kind, payload in emitted if kind == "run.model_fallback"]
    assert fallback[0]["to"] == "Cline (cline-pass/kimi-k3)"
    assert fallback[0]["reason"] == "repair_refused_twice"


@pytest.mark.asyncio
async def test_one_surviving_planner_rung_owns_every_planning_operation() -> None:
    """A dead primary is tried once, then spec, plan and directions stay on the
    same surviving planner instead of paying for the same failure each step."""
    calls: list[tuple[str, str]] = []

    class Model:
        async def _call(self, operation, request, *, model_aliases=None):
            model = model_aliases["_cline_model"]
            calls.append((operation, model))
            if model == "cline-pass/glm-5.2":
                raise RuntimeError("primary unavailable")
            return operation

        async def project_spec(self, request, *, model_aliases=None):
            return await self._call(
                "project_spec", request, model_aliases=model_aliases
            )

        async def project_plan_files(self, request, *, model_aliases=None):
            return await self._call(
                "project_plan_files", request, model_aliases=model_aliases
            )

        async def project_direction(self, request, *, model_aliases=None):
            return await self._call(
                "project_direction", request, model_aliases=model_aliases
            )

    plane = _plane(Model())
    state = _state(
        model_aliases={
            "_project_id": "asset_x",
            "_provider": "cline",
            "_chain_planner": (
                '[{"provider":"cline","model":"cline-pass/glm-5.2"},'
                '{"provider":"cline","model":"cline-pass/kimi-k3"}]'
            ),
        }
    )
    for operation in ("project_spec", "project_plan_files", "project_direction"):
        assert (
            await ControlPlane._project_planner_call(plane, state, operation, {})
            == operation
        )

    assert calls == [
        ("project_spec", "cline-pass/glm-5.2"),
        ("project_spec", "cline-pass/kimi-k3"),
        ("project_plan_files", "cline-pass/kimi-k3"),
        ("project_direction", "cline-pass/kimi-k3"),
    ]
    assert state["project_planner_chain_index"] == 1


@pytest.mark.asyncio
async def test_an_exhausted_planner_ladder_is_recorded_and_not_retried() -> None:
    calls: list[str] = []
    emitted: list[tuple[str, dict[str, Any]]] = []

    async def emit(run_id, conversation_id, kind, payload):
        emitted.append((kind, payload))

    class Dead:
        async def project_plan_files(self, request, *, model_aliases=None):
            model = model_aliases["_cline_model"]
            calls.append(model)
            raise TimeoutError(f"{model} timed out")

    plane = _plane(Dead())
    plane.events = SimpleNamespace(emit=emit)
    state = _state(
        model_aliases={
            "_project_id": "asset_x",
            "_provider": "cline",
            "_chain_planner": (
                '[{"provider":"cline","model":"cline-pass/glm-5.2"},'
                '{"provider":"cline","model":"cline-pass/qwen3.7-plus"}]'
            ),
        }
    )

    with pytest.raises(TimeoutError):
        await ControlPlane._project_planner_call(plane, state, "project_plan_files", {})
    assert calls == ["cline-pass/glm-5.2", "cline-pass/qwen3.7-plus"]
    assert state["project_planner_chain_index"] == 2
    assert [kind for kind, _ in emitted if kind != "run.planner_attempt"] == [
        "run.model_fallback",
        "run.model_exhausted",
    ]
    assert emitted[-1][1]["reason"] == "backend_timeout"

    with pytest.raises(PermanentModelError, match="planner model ladder is exhausted"):
        await ControlPlane._project_planner_call(plane, state, "project_plan_files", {})
    assert calls == ["cline-pass/glm-5.2", "cline-pass/qwen3.7-plus"]


@pytest.mark.asyncio
async def test_an_invalid_structured_plan_advances_to_the_next_planner_rung() -> None:
    """The exact live-canary failure shape: the first planner's structured
    output fails Pydantic validation (a schema violation, not an
    infrastructure error). The ladder must still advance to the next
    configured planner, exactly as it does for a timeout or an outage."""
    calls: list[str] = []
    emitted: list[tuple[str, dict[str, Any]]] = []

    async def emit(run_id, conversation_id, kind, payload):
        emitted.append((kind, payload))

    class Model:
        async def project_plan_files(self, request, *, model_aliases=None):
            model = model_aliases["_cline_model"]
            calls.append(model)
            if model == "cline-pass/glm-5.2":
                raise ModelProviderError(
                    "model returned invalid ProjectBuildPlanV1; initial="
                    "ValidationError: 2 validation errors for ProjectBuildPlanV1\n"
                    "slices.2.scenario_names\n  List should have at most 5 items "
                    "after validation, not 7 [type=too_long]"
                )
            return "a real plan"

    plane = _plane(Model())
    plane.events = SimpleNamespace(emit=emit)
    state = _state(
        model_aliases={
            "_project_id": "asset_x",
            "_provider": "cline",
            "_chain_planner": (
                '[{"provider":"cline","model":"cline-pass/glm-5.2"},'
                '{"provider":"cline","model":"cline-pass/deepseek-v4-pro"}]'
            ),
        }
    )
    result = await ControlPlane._project_planner_call(
        plane, state, "project_plan_files", {}
    )
    assert result == "a real plan"
    assert calls == ["cline-pass/glm-5.2", "cline-pass/deepseek-v4-pro"]
    assert [kind for kind, _ in emitted if kind != "run.planner_attempt"] == [
        "run.model_fallback"
    ]
    assert emitted[0][1]["reason"] == "invalid_planner_reply"
    assert state["project_planner_chain_index"] == 1


@pytest.mark.asyncio
async def test_an_exhausted_planner_ladder_ends_a_clinecore_turn_before_any_coding() -> (
    None
):
    """When every configured planner has failed, a ClineCore-selected turn
    must end honestly -- never fall through to the legacy per-file loop,
    and never start a coding session or offer an approval card."""
    emitted: list[tuple[str, dict[str, Any]]] = []

    async def emit(run_id, conversation_id, kind, payload):
        emitted.append((kind, payload))

    class MustNotBeCalled:
        def __getattr__(self, name: str) -> Any:
            raise AssertionError(
                f"no coding-engine method should be reached after plan failure: {name}"
            )

    plane = _plane(object())
    plane.events = SimpleNamespace(emit=emit)
    plane.project_coding = MustNotBeCalled()
    state = _state(
        model_aliases={"_project_id": "asset_x", "_coding_engine": "clinecore"},
        project_planner_chain_index=1,
    )
    plan = {
        "files": None,
        "scenarios": [],
        "slices": [],
        "intent": "",
        "scope": "",
        "taken": False,
        "plan_error": "model returned invalid ProjectBuildPlanV1; initial=...; repair=...",
    }

    result = await ControlPlane._project_plan_failure_response(plane, state, plan)

    assert result is not None
    assert "could not produce a build plan" in result["response_text"]
    assert result["project_pending_call"] == {}
    assert result["project_plan_taken"] is True
    assert "approval_request" not in result  # no approval offered
    assert [kind for kind, _ in emitted if kind != "run.planner_attempt"] == [
        "project.plan_failed"
    ]
    kind, payload = emitted[0]
    assert payload["reason"] == "planner_exhausted"
    assert payload["engine"] == "clinecore"
    assert payload["planner_chain_index"] == 1


@pytest.mark.asyncio
async def test_a_taken_plan_is_not_treated_as_a_failure() -> None:
    """A real, successfully-taken plan must return None (continue as normal),
    not be mistaken for one of the failure shapes above."""
    plane = _plane(object())
    state = _state(
        model_aliases={"_project_id": "asset_x", "_coding_engine": "clinecore"}
    )
    plan = {
        "files": ["app/main.py"],
        "scenarios": [],
        "slices": [],
        "intent": "build",
        "scope": "whole_app",
        "taken": True,
    }
    assert await ControlPlane._project_plan_failure_response(plane, state, plan) is None


@pytest.mark.asyncio
async def test_a_plan_missing_an_explicitly_required_file_ends_a_clinecore_turn() -> (
    None
):
    """Requirement #4: the final plan must cover what the request explicitly
    named, even when the plan itself parsed and named some files."""
    emitted: list[tuple[str, dict[str, Any]]] = []

    async def emit(run_id, conversation_id, kind, payload):
        emitted.append((kind, payload))

    plane = _plane(object())
    plane.events = SimpleNamespace(emit=emit)
    state = _state(
        model_aliases={"_project_id": "asset_x", "_coding_engine": "clinecore"},
        project_required_files=["app/main.py", "app/db.py", "tests/test_app.py"],
    )
    plan = {
        "files": ["app/main.py"],
        "scenarios": [],
        "slices": [],
        "intent": "build",
        "scope": "whole_app",
        "taken": True,
    }

    result = await ControlPlane._project_plan_failure_response(plane, state, plan)

    assert result is not None
    assert result["project_plan_taken"] is True
    assert "omits 2 explicitly required file(s)" in result["response_text"]
    kind, payload = emitted[0]
    assert payload["reason"] == "missing_required_files"
    assert payload["missing"] == ["app/db.py", "tests/test_app.py"]


@pytest.mark.asyncio
async def test_a_verifier_repair_cannot_manufacture_a_plan_from_nothing() -> None:
    """The exact live-canary mechanism: a verifier finding on a file that was
    never part of any accepted plan must not silently become the whole plan.
    A repair may expand an ESTABLISHED plan by one proven dependency; it must
    never create one out of a single blocked, previously-unplanned file."""
    emitted: list[tuple[str, dict[str, Any]]] = []

    async def emit(run_id, conversation_id, kind, payload):
        emitted.append((kind, payload))

    plane = _plane(object())
    plane.events = SimpleNamespace(emit=emit)
    plane.projects = SimpleNamespace(
        context=_empty_context, execute_staged=None, record_plan=None
    )
    state = _state(
        project_planned_files=[],  # no plan was ever established
        project_staged={"app/repository.py": {"content": "x = 1\n"}},
    )
    errors = [
        {
            "path": "app/repository.py",
            "error": "return-value: Incompatible return value type",
            "severity": "error",
            "kind": "",
            "rung": "typecheck",
        }
    ]

    update = await ControlPlane._staged_verify_retry(
        plane, state, iterations=5, retries=0, errors=errors
    )

    assert "project_planned_files" not in update
    assert emitted == []
    # The repair itself may still be directed at the real, broken file --
    # only the silent plan-manufacture is forbidden.
    assert update["project_write_pin"] == ["app/repository.py"]


@pytest.mark.asyncio
async def test_a_verifier_repair_may_still_expand_an_established_plan() -> None:
    """The legitimate case the guard above must not break: a real, accepted
    plan may still be expanded by exactly one verifier-proven dependency."""
    emitted: list[tuple[str, dict[str, Any]]] = []

    async def emit(run_id, conversation_id, kind, payload):
        emitted.append((kind, payload))

    plane = _plane(object())
    plane.events = SimpleNamespace(emit=emit)
    plane.projects = SimpleNamespace(
        context=_empty_context, execute_staged=None, record_plan=None
    )
    state = _state(
        project_planned_files=["app/main.py"],
        project_staged={
            "app/main.py": {"content": "x = 1\n"},
            "app/repository.py": {"content": "y = 2\n"},
        },
    )
    errors = [
        {
            "path": "app/repository.py",
            "error": "return-value: Incompatible return value type",
            "severity": "error",
            "kind": "",
            "rung": "typecheck",
        }
    ]

    update = await ControlPlane._staged_verify_retry(
        plane, state, iterations=5, retries=0, errors=errors
    )

    assert update["project_planned_files"] == ["app/main.py", "app/repository.py"]
    assert [kind for kind, _ in emitted if kind != "run.planner_attempt"] == [
        "project.plan_revised"
    ]
    assert emitted[0][1]["source"] == "verifier"


@pytest.mark.asyncio
async def test_provider_cap_skips_same_provider_planners() -> None:
    """The planner shares the coder's provider-aware fallback semantics."""
    calls: list[tuple[str, str]] = []
    emitted: list[tuple[str, dict[str, Any]]] = []

    async def emit(run_id, conversation_id, kind, payload):
        emitted.append((kind, payload))

    class CappedClineHealthyCohere:
        async def project_plan_files(self, request, *, model_aliases=None):
            provider = model_aliases["_provider"]
            model = model_aliases.get("_cline_model", "")
            calls.append((provider, model))
            if provider == "cline":
                raise ModelProviderError(
                    'Cline returned HTTP 429: {"error":{"code":'
                    '"INFERENCE_CAP_ERROR","message":"weekly Clinepass limit"}}'
                )
            return "plan"

    plane = _plane(CappedClineHealthyCohere())
    plane.events = SimpleNamespace(emit=emit)
    state = _state(
        model_aliases={
            "_project_id": "asset_x",
            "_provider": "cline",
            "_chain_planner": (
                '[{"provider":"cline","model":"cline-pass/glm-5.2"},'
                '{"provider":"cline","model":"cline-pass/qwen3.7-plus"},'
                '{"provider":"cohere","model":null}]'
            ),
        }
    )

    assert (
        await ControlPlane._project_planner_call(plane, state, "project_plan_files", {})
        == "plan"
    )
    assert calls == [
        ("cline", "cline-pass/glm-5.2"),
        ("cohere", ""),
    ]
    assert state["project_planner_chain_index"] == 2
    fallback = [payload for kind, payload in emitted if kind == "run.model_fallback"]
    assert fallback[0]["from"] == "Cline provider"
    assert fallback[0]["to"] == "Cohere Command A+"
    assert fallback[0]["skipped"] == ["Cline (cline-pass/qwen3.7-plus)"]


@pytest.mark.asyncio
async def test_provider_cap_during_optional_spec_exhausts_same_provider_once() -> None:
    """A provider cap during optional spec compilation also applies to manifest."""
    calls: list[str] = []
    emitted: list[tuple[str, dict[str, Any]]] = []

    async def emit(run_id, conversation_id, kind, payload):
        emitted.append((kind, payload))

    class Capped:
        async def project_spec(self, request, *, model_aliases=None):
            calls.append(model_aliases["_cline_model"])
            raise ModelProviderError(
                'Cline returned HTTP 429: {"error":{"code":'
                '"INFERENCE_CAP_ERROR","message":"weekly Clinepass limit"}}'
            )

    plane = _plane(Capped())
    plane.events = SimpleNamespace(emit=emit)
    state = _state(
        model_aliases={
            "_project_id": "asset_x",
            "_provider": "cline",
            "_chain_planner": (
                '[{"provider":"cline","model":"cline-pass/glm-5.2"},'
                '{"provider":"cline","model":"cline-pass/qwen3.7-plus"}]'
            ),
        }
    )

    with pytest.raises(ModelProviderError, match="INFERENCE_CAP_ERROR"):
        await ControlPlane._project_planner_call(plane, state, "project_spec", {})
    assert calls == ["cline-pass/glm-5.2"]
    assert state["project_planner_chain_index"] == 2
    exhausted = [payload for kind, payload in emitted if kind == "run.model_exhausted"]
    assert exhausted[0]["model"] == "Cline provider"
    assert exhausted[0]["provider"] == "cline"
    assert exhausted[0]["reason"] == "provider_exhausted"
    assert exhausted[0]["skipped"] == ["Cline (cline-pass/qwen3.7-plus)"]

    # The optional spec failed, but the required manifest does not spend a
    # second known-doomed provider call in the same turn.
    with pytest.raises(PermanentModelError, match="planner model ladder is exhausted"):
        await ControlPlane._project_planner_call(plane, state, "project_plan_files", {})
    assert calls == ["cline-pass/glm-5.2"]


@pytest.mark.asyncio
async def test_a_complete_manifest_is_verified_before_any_model_call() -> None:
    class MustNotRun:
        def __getattr__(self, name):
            raise AssertionError(
                f"the completed plan must verify before model call {name}"
            )

    async def clean(project_id, staged, **kwargs):
        return {"errors": [], "warnings": [], "notes": [], "checks": ["smoke"]}

    plane = _plane(MustNotRun())
    plane._verify_staged_changeset = clean
    result = await ControlPlane._project_step(
        plane,
        _state(
            project_iterations=7,
            project_staged={
                "app/main.py": {"content": "app = 1\n", "bytes": 8},
                "app/static/index.html": {"content": "<main />\n", "bytes": 9},
            },
            project_direction={},
            project_focus_path="",
        ),
    )
    assert "All 2 planned file change(s)" in result["response_text"]
    assert "passed 1 verification check(s)" in result["response_text"]


@pytest.mark.asyncio
async def test_a_complete_broken_manifest_creates_one_exact_repair() -> None:
    class MustNotRun:
        def __getattr__(self, name):
            raise AssertionError(f"verification must choose the repair before {name}")

    async def broken(project_id, staged, **kwargs):
        return {
            "errors": [
                {"path": "app/main.py", "error": "missing create_app"},
                {"path": "app/static/index.html", "error": "missing root element"},
            ],
            "warnings": [],
            "notes": [],
            "checks": [],
        }

    plane = _plane(MustNotRun())
    plane._verify_staged_changeset = broken
    result = await ControlPlane._project_step(
        plane,
        _state(
            project_iterations=7,
            project_staged={
                "app/main.py": {"content": "app = 1\n", "bytes": 8},
                "app/static/index.html": {"content": "<main />\n", "bytes": 9},
            },
            project_direction={},
            project_focus_path="",
        ),
    )
    assert result["project_direction"]["path"] == "app/main.py"
    instruction = result["project_direction"]["instruction"]
    assert "missing create_app" in instruction
    assert "missing root element" not in instruction
    assert result["project_write_pin"] == ["app/main.py"]


# ── What the live Argus revamp exposed ─────────────────────────────────────


def test_a_project_with_its_own_frontend_is_not_given_appkit() -> None:
    """A Vite/React project got six Python scaffold files seeded into it, and
    on approval they landed on disk referenced by nothing."""
    from waqil_api.control_plane import _has_own_frontend

    argus = {
        "manifest": {
            "file_tree": [
                "app.py",
                "requirements.txt",
                "web/package.json",
                "web/vite.config.js",
                "web/src/App.jsx",
                "web/src/styles.css",
            ]
        }
    }
    assert _has_own_frontend(argus) is True


def test_a_project_without_a_frontend_still_gets_appkit() -> None:
    """The Streamlit-onto-appkit conversion is the case seeding was built for,
    and a bare index.html must not be mistaken for a frontend toolchain."""
    from waqil_api.control_plane import _has_own_frontend

    streamlit = {"manifest": {"file_tree": ["app.py", "requirements.txt"]}}
    assert _has_own_frontend(streamlit) is False
    static_only = {"manifest": {"file_tree": ["app.py", "static/index.html"]}}
    assert _has_own_frontend(static_only) is False
    existing_static_app = {
        "manifest": {
            "file_tree": [
                "app/main.py",
                "app/static/index.html",
                "app/static/styles.css",
                "app/static/app.js",
            ]
        }
    }
    assert _has_own_frontend(existing_static_app) is True
    # A tree Metis itself seeded does not count as the project's own.
    seeded = {
        "manifest": {
            "file_tree": ["app.py", "appkit/web.py", "appkit/static/theme.css"]
        }
    }
    assert _has_own_frontend(seeded) is False
    assert _has_own_frontend({}) is False


@pytest.mark.asyncio
async def test_a_dead_orchestrator_is_reported_not_silently_dropped() -> None:
    """The credits ran out mid-turn, every later direction failed, and the turn
    degraded into exactly the drift the split removes — saying nothing."""
    emitted: list[tuple[str, dict[str, Any]]] = []

    async def emit(run_id, conversation_id, kind, payload):
        emitted.append((kind, payload))

    class Broke:
        async def project_direction(self, request, *, model_aliases=None):
            raise RuntimeError(
                "Cline has no credits left for anthropic/claude-opus-4.5"
            )

    plane = _plane(Broke())
    plane.events = SimpleNamespace(emit=emit)
    direction = await ControlPlane._project_direct(
        plane, _state(), {}, {}, ["app/main.py"], 3
    )
    # The fallback still happens — but it is now a fact on the record.
    assert direction["failed"].startswith("Cline has no credits")
    assert [kind for kind, _ in emitted if kind != "run.planner_attempt"] == [
        "run.model_exhausted",
        "project.direction_failed",
    ]
    assert "no credits" in emitted[-1][1]["error"]


def test_an_over_long_reuse_list_is_trimmed_not_rejected() -> None:
    """A model listed thirteen things to reuse and lost the entire direction —
    instruction included — to the thirteenth. The bound protects the brief from
    flooding; it should never cost the brief."""
    direction = ProjectDirectionV1(
        path="app/main.py",
        instruction="Define create_app().",
        reuse=[f"symbol_{index}" for index in range(20)],
        read=[f"file_{index}.py" for index in range(9)],
    )
    assert direction.path == "app/main.py"
    assert direction.instruction == "Define create_app()."
    assert len(direction.reuse) == 12  # each field's own bound …
    assert len(direction.read) == 6  # … so trimming cannot trip the limit


def test_a_long_instruction_survives() -> None:
    """Real orchestrators write 3,500-6,000 characters here, so the old 6,000
    cap sat exactly on the distribution and rejected the longest."""
    brief = "Define create_app(). " * 400  # ~8,400 characters
    assert len(ProjectDirectionV1(path="a.py", instruction=brief).instruction) == len(
        brief
    )


# ── What the Argus revamp exposed: visibility and tool count ───────────────


def test_a_directed_step_advertises_only_what_it_may_do() -> None:
    """Twelve tools were offered on a step where four were legal, and four of
    the eight illegal ones were read tools the host refuses. The coder answered
    with read_file and then spent twenty-three more steps reading."""
    from waqil_api.model_provider import project_roster

    directed = project_roster(
        {
            "build_turn": True,
            "files_still_to_write": ["web/src/styles.css"],
            "reads_closed": True,
            "plan_taken": True,
        }
    )
    names = {tool["name"] for tool in directed}
    assert names == {
        "create_file",
        "apply_patch",
        "replace_lines",
        "revise_plan",
        "finish_project_task",
    }
    # No read is even expressible.
    assert not names & {"read_file", "list_files", "search_code", "inspect_api"}
    # An ordinary step keeps the full roster.
    ordinary = project_roster({"build_turn": True, "files_still_to_write": ["a.py"]})
    assert len(ordinary) > len(directed)
    assert "read_file" in {tool["name"] for tool in ordinary}


def test_the_local_grammar_makes_a_read_unexpressible_too() -> None:
    """Both lanes must be asked the same small question."""
    from waqil_api.contracts import project_directed_schema

    schema = project_directed_schema(["web/src/styles.css"])
    tools = schema["properties"]["tool"]["enum"]
    assert set(tools) == {"create_file", "apply_patch", "replace_lines", "revise_plan"}
    assert schema["properties"]["arguments"]["properties"]["path"]["enum"] == [
        "web/src/styles.css"
    ]
    # revise_plan's own arguments are expressible, or naming it would be a trap.
    assert "files" in schema["properties"]["arguments"]["properties"]
    assert "remove_files" in schema["properties"]["arguments"]["properties"]
    assert "reason" in schema["properties"]["arguments"]["properties"]


@pytest.mark.asyncio
async def test_an_exhausted_foundation_blocks_its_dependents() -> None:
    """A dependency-ordered plan must not skip a failed foundation and direct
    its dependents as though the foundation had landed."""

    class Model:
        async def project_direction(self, request, *, model_aliases=None):
            raise AssertionError(
                "a dependent must not be offered before its foundation"
            )

    attempts = {"web/src/styles.css": 3}
    direction = await ControlPlane._project_direct(
        _plane(Model()),
        _state(
            project_direction_attempts=attempts,
            project_direction={
                "path": "web/src/styles.css",
                "instruction": "rewrite it",
            },
        ),
        {},
        {},
        ["web/src/styles.css", "web/src/App.jsx"],
        6,
    )
    assert [item["path"] for item in direction["exhausted"]] == ["web/src/styles.css"]
    assert direction["exhausted"][0]["attempts"] == 3


@pytest.mark.asyncio
async def test_a_turn_with_only_failed_files_left_ends_instead_of_drifting() -> None:
    """Twenty-three steps were spent reading a file no path could reach."""

    class MustNotRun:
        async def project_direction(self, request, *, model_aliases=None):
            raise AssertionError(
                "nothing is directable; the orchestrator must not be asked"
            )

    direction = await ControlPlane._project_direct(
        _plane(MustNotRun()),
        _state(project_direction_attempts={"web/src/styles.css": 3}),
        {},
        {},
        ["web/src/styles.css"],
        20,
    )
    assert [item["path"] for item in direction["exhausted"]] == ["web/src/styles.css"]


# ── Three defects the styles.css runs exposed ──────────────────────────────


def test_a_directed_step_is_told_reads_do_not_exist() -> None:
    """Narrowing the roster was not enough. The instructions above it describe
    reads at length, and a live directed run followed the prose over the
    roster: three directions, seven reads, nothing written."""
    from waqil_api.model_provider import PROJECT_AGENT_SYSTEM, project_system_prompt

    directed = project_system_prompt({"reads_closed": True})
    assert "THIS STEP IS DIFFERENT" in directed
    assert "none of those tools exist on this step" in directed
    # And an ordinary step is untouched — reads are how it works.
    assert project_system_prompt({}) == PROJECT_AGENT_SYSTEM


def test_a_tool_whose_bound_is_spent_is_no_longer_offered() -> None:
    """A turn spent seven consecutive steps calling revise_plan past its limit,
    collecting the same refusal each time, because the roster kept offering it."""
    from waqil_api.model_provider import project_roster

    live = {
        tool["name"]
        for tool in project_roster(
            {
                "build_turn": True,
                "files_still_to_write": ["a.css"],
                "reads_closed": True,
                "plan_taken": True,
            }
        )
    }
    spent = {
        tool["name"]
        for tool in project_roster(
            {
                "build_turn": True,
                "files_still_to_write": ["a.css"],
                "reads_closed": True,
                "plan_taken": True,
                "plan_revisions_spent": True,
            }
        )
    }
    assert "revise_plan" in live
    assert "revise_plan" not in spent
    # Writing is still possible; only the spent escape is gone.
    assert {"create_file", "apply_patch", "replace_lines"} <= spent


@pytest.mark.asyncio
async def test_a_write_outside_the_plan_is_refused() -> None:
    """Once every planned file was staged the write target was unconstrained,
    and a stylesheet repair used that freedom to edit config.py and break an
    import three files away."""
    plane = _plane(object())

    async def must_not_write(*args: object, **kwargs: object):
        raise AssertionError("a write outside the plan must never reach the workspace")

    plane.projects = SimpleNamespace(
        context=_empty_context, execute_staged=must_not_write
    )
    state = _state(
        project_planned_files=["web/src/styles.css"],
        project_trace=[],
        project_pending_call={
            "name": "apply_patch",
            "arguments": {"path": "config.py", "original": "a", "replacement": "b"},
        },
    )
    evidence = await ControlPlane._project_execute(plane, state)  # type: ignore[arg-type]
    result = evidence["project_trace"][-1]["result"]
    assert result["ok"] is False
    assert "not in this turn's plan" in result["error"]
    assert "revise_plan" in result["error"]


@pytest.mark.asyncio
async def test_a_planned_file_is_still_writable() -> None:
    """The scope gate must not become a gate on the work itself."""
    plane = _plane(object())
    written: list[str] = []

    async def execute_staged(project_id, call, staged, extra):
        written.append(call.arguments["path"])
        return {"ok": True}, {"web/src/styles.css": {"bytes": 10}}

    plane.projects = SimpleNamespace(
        context=_empty_context, execute_staged=execute_staged, style_gaps=None
    )
    state = _state(
        project_planned_files=["web/src/styles.css"],
        project_trace=[],
        project_pending_call={
            "name": "create_file",
            "arguments": {"path": "web/src/styles.css", "content": ".a{}"},
        },
    )
    await ControlPlane._project_execute(plane, state)  # type: ignore[arg-type]
    assert written == ["web/src/styles.css"]


def test_a_write_tool_that_cannot_apply_is_not_offered() -> None:
    """Directed at an existing 29KB stylesheet, a coder called create_file three
    times, was refused three times for aiming at a path that exists, and the
    turn ended having written nothing — with apply_patch unused beside it."""
    from waqil_api.contracts import project_directed_schema
    from waqil_api.model_provider import project_roster

    base = {
        "build_turn": True,
        "files_still_to_write": ["web/src/styles.css"],
        "reads_closed": True,
    }
    existing = {t["name"] for t in project_roster({**base, "target_exists": True})}
    fresh = {t["name"] for t in project_roster({**base, "target_exists": False})}
    assert (
        "create_file" not in existing and {"apply_patch", "replace_lines"} <= existing
    )
    assert "create_file" in fresh and not {"apply_patch", "replace_lines"} & fresh
    # Writing is always possible one way or the other.
    for roster in (existing, fresh):
        assert roster & {"create_file", "apply_patch", "replace_lines"}
    # The local grammar narrows identically, or the two lanes disagree.
    assert set(
        project_directed_schema(["a.css"], target_exists=True)["properties"]["tool"][
            "enum"
        ]
    ) == {"apply_patch", "replace_lines", "revise_plan"}
    assert set(
        project_directed_schema(["a.css"], target_exists=False)["properties"]["tool"][
            "enum"
        ]
    ) == {"create_file", "revise_plan"}
    # Unknown (an undirected or legacy request) keeps every write tool legal.
    assert (
        "create_file"
        in project_directed_schema(["a.css"])["properties"]["tool"]["enum"]
    )


def test_revise_plan_is_bounded_by_attempts_not_by_successes() -> None:
    """A no-op revision was free so it would not burn the budget — which made it
    free forever. A live turn spent fourteen consecutive steps re-proposing the
    same file list, refused every time, costing nothing it could run out of."""
    from waqil_api.model_provider import project_roster

    base = {
        "build_turn": True,
        "files_still_to_write": ["a.css"],
        "reads_closed": True,
        "target_exists": True,
        "plan_taken": True,
    }
    assert "revise_plan" in {t["name"] for t in project_roster(base)}
    # Spent by attempts, even when no revision ever succeeded.
    spent = {t["name"] for t in project_roster({**base, "plan_revisions_spent": True})}
    assert "revise_plan" not in spent
    assert {"apply_patch", "replace_lines"} <= spent


def test_the_coder_cannot_create_the_initial_plan() -> None:
    from waqil_api.model_provider import project_roster

    initial = {
        tool["name"]
        for tool in project_roster(
            {"build_turn": True, "files_still_to_write": [], "plan_taken": False}
        )
    }
    assert "revise_plan" not in initial
