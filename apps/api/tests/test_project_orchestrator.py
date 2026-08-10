"""The orchestrator/coder split: one model decides scope, another writes.

The case these exist for is on record. Given a correctly planned four-file
conversion, narrowed by the host to ONE file, a capable coder read twenty-one
times and wrote nothing — it was never short of context, it was simply free to
keep looking. Everything here is about removing that freedom without removing
the model's judgement: the orchestrator judges and cannot write, the coder writes
and cannot judge, and the controller between them is ordinary code.
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
async def test_the_orchestrator_names_the_file_and_the_coder_is_told_what_to_write() -> None:
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
    assert "create_app()" in brief          # the orchestrator's words, whole
    assert "appkit.web.mount_appkit_static" in brief
    assert "Reads are closed" in brief


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
            {}, {}, ["app/main.py"], 3,
        )
        assert direction["path"] == "app/main.py"
        assert direction["attempts"]["app/main.py"] == expected
        attempts = direction["attempts"]
    # Past the cap the host stops offering it — and says what it gave up on,
    # rather than letting the file quietly vanish from the menu.
    exhausted = await ControlPlane._project_direct(
        _plane(Repairer()),
        _state(project_direction_attempts=attempts),
        {}, {}, ["app/main.py"], 3,
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
            return ProjectDirectionV1(path="app/static/index.html", instruction="write it")

    staged = {"app/main.py": {"content": "x", "origin": "create", "bytes": 1}}
    direction = await ControlPlane._project_direct(
        _plane(Model()), _state(), {}, staged,
        ["app/main.py", "app/static/index.html"], 3,
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
    assert "path" not in direction          # nothing is directed …
    assert direction["failed"] == "the planner lane is down"   # … and why is on the record


@pytest.mark.asyncio
async def test_the_split_can_be_switched_off() -> None:
    class MustNotRun:
        async def project_direction(self, request, *, model_aliases=None):
            raise AssertionError("the orchestrator must not be consulted")

    direction = await ControlPlane._project_direct(
        _plane(MustNotRun(), project_orchestrator_enabled=False),
        _state(), {}, {}, ["app/main.py"], 3,
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
        [], {}, 3,
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

    plane.projects = SimpleNamespace(context=_empty_context, execute_staged=must_not_read)
    state = _state(
        project_direction={"path": "app/main.py", "instruction": "write it", "reuse": [], "read": []},
        project_focus_path="app/main.py",
        project_consecutive_reads=0,
        project_trace=[],
        project_pending_call={"name": "read_file", "arguments": {"path": "app/other.py"}},
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
    plane.projects = SimpleNamespace(context=_empty_context, execute_staged=execute_staged)
    trace = await ControlPlane._prefetch_for_coder(
        plane,
        _state(),  # type: ignore[arg-type]
        "asset_x",
        {},
        {"path": "app/main.py", "read": ["app/models.py"]},
        [],
    )
    # The target file first — a repair is impossible without it — then what the
    # orchestrator asked for.
    assert seen == ["app/main.py", "app/models.py"]
    assert [entry["arguments"]["path"] for entry in trace] == seen
    assert all(entry["result"]["ok"] for entry in trace)


@pytest.mark.asyncio
async def test_a_prefetch_that_fails_is_evidence_not_an_error() -> None:
    async def execute_staged(project_id, call, staged, extra):
        raise FileNotFoundError("app/ghost.py does not exist")

    plane = _plane(object())
    plane.projects = SimpleNamespace(context=_empty_context, execute_staged=execute_staged)
    trace = await ControlPlane._prefetch_for_coder(
        plane, _state(), "asset_x", {},  # type: ignore[arg-type]
        {"path": "app/ghost.py", "read": []},
        [],
    )
    assert trace[-1]["result"]["ok"] is False
    assert "does not exist" in trace[-1]["result"]["error"]


# ── End to end, with a scripted orchestrator and coder ─────────────────────


@pytest.mark.asyncio
async def test_the_directed_arc_writes_the_file_the_orchestrator_named() -> None:
    class Model:
        def __init__(self) -> None:
            self.brief: str = ""

        async def project_plan_files(self, request, *, model_aliases=None):
            raise AssertionError("the plan is already taken in this state")

        async def project_direction(self, request, *, model_aliases=None):
            return ProjectDirectionV1(
                path="app/main.py",
                instruction="Define create_app() and mount the static files.",
                reuse=["appkit.web.mount_appkit_static"],
                read=["appkit/web.py"],
            )

        async def project_step(self, request, *, model_aliases=None):
            self.brief = request["attention"]
            return ProjectAgentStepV1(
                status="tool",
                tool_call=ProjectToolCallV1(
                    name="create_file",
                    arguments={"path": "app/main.py", "content": "def create_app():\n    ...\n"},
                ),
            )

    async def execute_staged(project_id, call, staged, extra):
        return {"content": "def mount_appkit_static(app): ...\n"}, None

    model = Model()
    plane = _plane(model)
    plane.projects = SimpleNamespace(context=_empty_context, execute_staged=execute_staged)
    result = await ControlPlane._project_step(
        plane,
        _state(project_plan_taken=True, project_iterations=3),  # type: ignore[arg-type]
    )
    # The coder was handed the orchestrator's instruction, not a generic one.
    assert "Define create_app() and mount the static files." in model.brief
    assert "appkit.web.mount_appkit_static" in model.brief
    # And the controller recorded who was directed, so a repair is countable.
    assert result["project_direction"]["path"] == "app/main.py"
    assert result["project_direction_attempts"] == {"app/main.py": 1}
    assert result["project_focus_path"] == "app/main.py"


def test_the_orchestrator_runs_in_its_own_seat_not_the_run_s_lane() -> None:
    """`_provider` is global to a run, so without this the planner ladder was
    decorative and the orchestrator ran on whatever chat was pinned to."""
    from waqil_api.control_plane import _planner_aliases

    aliases = {
        "_provider": "local",
        "coder": "qwen3-coder:30b",
        "_chain_planner": '[{"provider": "cline", "model": "anthropic/claude-opus-4.5"}]',
    }
    routed = _planner_aliases(aliases)
    assert routed["_provider"] == "cline"
    assert routed["_cline_model"] == "anthropic/claude-opus-4.5"
    # The coder alias is untouched: the two seats are independent.
    assert routed["coder"] == "qwen3-coder:30b"


def test_no_planner_chain_leaves_the_run_s_lane_alone() -> None:
    from waqil_api.control_plane import _planner_aliases

    aliases = {"_provider": "cohere", "coder": "x"}
    assert _planner_aliases(aliases) == {"_provider": "cohere", "coder": "x"}


# ── What the live Argus revamp exposed ─────────────────────────────────────


def test_a_project_with_its_own_frontend_is_not_given_appkit() -> None:
    """A Vite/React project got six Python scaffold files seeded into it, and
    on approval they landed on disk referenced by nothing."""
    from waqil_api.control_plane import _has_own_frontend

    argus = {
        "manifest": {
            "file_tree": [
                "app.py", "requirements.txt",
                "web/package.json", "web/vite.config.js",
                "web/src/App.jsx", "web/src/styles.css",
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
    # A tree Metis itself seeded does not count as the project's own.
    seeded = {"manifest": {"file_tree": ["app.py", "appkit/web.py", "appkit/static/theme.css"]}}
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
            raise RuntimeError("Cline has no credits left for anthropic/claude-opus-4.5")

    plane = _plane(Broke())
    plane.events = SimpleNamespace(emit=emit)
    direction = await ControlPlane._project_direct(
        plane, _state(), {}, {}, ["app/main.py"], 3
    )
    # The fallback still happens — but it is now a fact on the record.
    assert direction["failed"].startswith("Cline has no credits")
    assert [kind for kind, _ in emitted] == ["project.direction_failed"]
    assert "no credits" in emitted[0][1]["error"]


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
    assert len(direction.reuse) == 12    # each field's own bound …
    assert len(direction.read) == 6      # … so trimming cannot trip the limit


def test_a_long_instruction_survives() -> None:
    """Real orchestrators write 3,500-6,000 characters here, so the old 6,000
    cap sat exactly on the distribution and rejected the longest."""
    brief = "Define create_app(). " * 400          # ~8,400 characters
    assert len(ProjectDirectionV1(path="a.py", instruction=brief).instruction) == len(brief)


# ── What the Argus revamp exposed: visibility and tool count ───────────────


def test_a_directed_step_advertises_only_what_it_may_do() -> None:
    """Twelve tools were offered on a step where four were legal, and four of
    the eight illegal ones were read tools the host refuses. The coder answered
    with read_file and then spent twenty-three more steps reading."""
    from waqil_api.model_provider import project_roster

    directed = project_roster(
        {"build_turn": True, "files_still_to_write": ["web/src/styles.css"], "reads_closed": True}
    )
    names = {tool["name"] for tool in directed}
    assert names == {
        "create_file", "apply_patch", "replace_lines", "revise_plan", "finish_project_task",
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
    assert "reason" in schema["properties"]["arguments"]["properties"]


@pytest.mark.asyncio
async def test_the_orchestrator_is_told_what_the_host_gave_up_on() -> None:
    """The stylesheet everything composed against failed three times and simply
    vanished from `available_files` — no signal distinguished "already written"
    from "we stopped trying", so eleven dependents were directed anyway."""
    seen: dict[str, Any] = {}

    class Model:
        async def project_direction(self, request, *, model_aliases=None):
            seen.update(request)
            return ProjectDirectionV1(path="web/src/App.jsx", instruction="write it")

    attempts = {"web/src/styles.css": 3}
    await ControlPlane._project_direct(
        _plane(Model()),
        _state(
            project_direction_attempts=attempts,
            project_direction={"path": "web/src/styles.css", "instruction": "rewrite it"},
        ),
        {}, {},
        ["web/src/styles.css", "web/src/App.jsx"],
        6,
    )
    blocked = seen["blocked_files"]
    assert [item["path"] for item in blocked] == ["web/src/styles.css"]
    assert blocked[0]["attempts"] == 3
    # And what it asked for last, and whether that landed.
    assert seen["last_direction"]["path"] == "web/src/styles.css"
    assert seen["last_direction"]["written"] is False


@pytest.mark.asyncio
async def test_a_turn_with_only_failed_files_left_ends_instead_of_drifting() -> None:
    """Twenty-three steps were spent reading a file no path could reach."""
    class MustNotRun:
        async def project_direction(self, request, *, model_aliases=None):
            raise AssertionError("nothing is directable; the orchestrator must not be asked")

    direction = await ControlPlane._project_direct(
        _plane(MustNotRun()),
        _state(project_direction_attempts={"web/src/styles.css": 3}),
        {}, {}, ["web/src/styles.css"], 20,
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

    live = {tool["name"] for tool in project_roster(
        {"build_turn": True, "files_still_to_write": ["a.css"], "reads_closed": True}
    )}
    spent = {tool["name"] for tool in project_roster(
        {"build_turn": True, "files_still_to_write": ["a.css"], "reads_closed": True,
         "plan_revisions_spent": True}
    )}
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

    plane.projects = SimpleNamespace(context=_empty_context, execute_staged=must_not_write)
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

    plane.projects = SimpleNamespace(context=_empty_context, execute_staged=execute_staged,
                                     style_gaps=None)
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

    base = {"build_turn": True, "files_still_to_write": ["web/src/styles.css"],
            "reads_closed": True}
    existing = {t["name"] for t in project_roster({**base, "target_exists": True})}
    fresh = {t["name"] for t in project_roster({**base, "target_exists": False})}
    assert "create_file" not in existing and {"apply_patch", "replace_lines"} <= existing
    assert "create_file" in fresh and not {"apply_patch", "replace_lines"} & fresh
    # Writing is always possible one way or the other.
    for roster in (existing, fresh):
        assert roster & {"create_file", "apply_patch", "replace_lines"}
    # The local grammar narrows identically, or the two lanes disagree.
    assert set(project_directed_schema(["a.css"], target_exists=True)["properties"]["tool"]["enum"]) == {
        "apply_patch", "replace_lines", "revise_plan"
    }
    assert set(project_directed_schema(["a.css"], target_exists=False)["properties"]["tool"]["enum"]) == {
        "create_file", "revise_plan"
    }
    # Unknown (an undirected or legacy request) keeps every write tool legal.
    assert "create_file" in project_directed_schema(["a.css"])["properties"]["tool"]["enum"]


def test_revise_plan_is_bounded_by_attempts_not_by_successes() -> None:
    """A no-op revision was free so it would not burn the budget — which made it
    free forever. A live turn spent fourteen consecutive steps re-proposing the
    same file list, refused every time, costing nothing it could run out of."""
    from waqil_api.model_provider import project_roster

    base = {"build_turn": True, "files_still_to_write": ["a.css"], "reads_closed": True,
            "target_exists": True}
    assert "revise_plan" in {t["name"] for t in project_roster(base)}
    # Spent by attempts, even when no revision ever succeeded.
    spent = {t["name"] for t in project_roster({**base, "plan_revisions_spent": True})}
    assert "revise_plan" not in spent
    assert {"apply_patch", "replace_lines"} <= spent
