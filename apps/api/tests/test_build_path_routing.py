"""Which build path an ordinary project request is frozen onto.

Two live runs on 2026-08-10 were reported as evidence about the direct path and
were nothing of the kind: their `_build_path` alias was **absent**, so
`_uses_cline_direct` read its default of `planner_slices` and every step went
through the retained legacy orchestration — eighteen `project.direction` events,
no `project.direct_contract`, no coding session, no host check. The cause was a
stale server process, not the routing; but "the code looks right" is exactly the
claim that failed, so it is pinned here instead.

Deliberately built with DEFAULT settings. A test that passes
`project_build_path="cline_direct"` proves the plumbing honours an explicit
choice and says nothing about what an ordinary request gets, which is the thing
that was actually wrong.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from waqil_api.config import Settings
from waqil_api.control_plane import ControlPlane
from waqil_api.main import create_app


def _project(root: Path) -> Path:
    """A minimal but real project the workspace will open and map."""
    project = root / "Projects" / "canary"
    (project / "app").mkdir(parents=True)
    (project / "README.md").write_text("# Canary\n", encoding="utf-8")
    (project / "requirements.txt").write_text("fastapi\n", encoding="utf-8")
    (project / "app" / "main.py").write_text(
        "def create_app():\n    return None\n", encoding="utf-8"
    )
    return project


def _settings(tmp_path: Path) -> Settings:
    projects_root = tmp_path / "Projects"
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        data_dir=tmp_path / "data",
        repo_root=tmp_path,
        asset_roots=[projects_root],
        model_backend="deterministic",
        allow_test_backends=True,
        reference_runner_mode="deterministic",
        # Everything about the build path is left at its default on purpose.
    )


@pytest.mark.asyncio
async def test_an_ordinary_project_request_is_frozen_onto_cline_direct(
    tmp_path: Path,
) -> None:
    """The UI sends a project id and a mode; the run must carry the direct path.

    Asserted on the RUN's persisted aliases rather than on settings, because the
    alias is what `_uses_cline_direct` reads on every later step — and it is what
    was missing on the two runs that were mistaken for direct-path evidence.
    """
    _project(tmp_path)
    settings = _settings(tmp_path)
    assert settings.project_coding_engine == "clinecore"
    assert settings.project_build_path == "cline_direct"

    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://metis.test"
        ) as client:
            assets = (await client.post("/api/v1/assets/scan")).json()
            project_id = next(
                item["id"] for item in assets if "canary" in item["name"].casefold()
            )
            opened = await client.post(
                f"/api/v1/projects/{project_id}/open",
                json={"mode": "grok_bootstrap_local"},
            )
            assert opened.status_code == 200

            conversation_id = (
                await client.post("/api/v1/conversations", json={"title": "routing"})
            ).json()["id"]
            accepted = (
                await client.post(
                    f"/api/v1/conversations/{conversation_id}/messages",
                    json={
                        "content": "Add a /health route to app/main.py.",
                        "project_id": project_id,
                        "project_mode": "grok_bootstrap_local",
                    },
                )
            ).json()

            plane = app.state.runtime.control_plane
            record = await plane.database.get_run_execution_record(accepted["run_id"])
            aliases = (record or {}).get("model_aliases") or {}

            # The two facts the stale-process runs could not show.
            assert aliases.get("_build_path") == "cline_direct"
            assert aliases.get("_coding_engine") == "clinecore"
            # And the predicate every later step consults agrees.
            assert ControlPlane._uses_cline_direct({"model_aliases": aliases}) is True


@pytest.mark.asyncio
async def test_a_second_message_on_the_same_conversation_stays_frozen(
    tmp_path: Path,
) -> None:
    """A follow-up carries no project fields; the session supplies them, and the
    alias must be frozen there too — that branch is a separate code path."""
    _project(tmp_path)
    app = create_app(_settings(tmp_path))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://metis.test"
        ) as client:
            assets = (await client.post("/api/v1/assets/scan")).json()
            project_id = next(
                item["id"] for item in assets if "canary" in item["name"].casefold()
            )
            await client.post(
                f"/api/v1/projects/{project_id}/open",
                json={"mode": "grok_bootstrap_local"},
            )
            conversation_id = (
                await client.post("/api/v1/conversations", json={"title": "routing"})
            ).json()["id"]
            await client.post(
                f"/api/v1/conversations/{conversation_id}/messages",
                json={
                    "content": "Add a /health route.",
                    "project_id": project_id,
                    "project_mode": "grok_bootstrap_local",
                },
            )
            # No project fields at all — the conversation's session provides them.
            follow_up = (
                await client.post(
                    f"/api/v1/conversations/{conversation_id}/messages",
                    json={"content": "Now add a /version route."},
                )
            ).json()

            plane = app.state.runtime.control_plane
            record = await plane.database.get_run_execution_record(follow_up["run_id"])
            aliases = (record or {}).get("model_aliases") or {}
            assert aliases.get("_build_path") == "cline_direct"
            assert ControlPlane._uses_cline_direct({"model_aliases": aliases}) is True


def test_an_absent_alias_reads_as_the_legacy_path() -> None:
    """The default that turned two runs into legacy runs, stated outright.

    This is not a bug — a recoverable pre-migration checkpoint must keep
    finishing on the design it started under. It is only dangerous when a NEW
    run reaches the loop without the alias, which the two tests above forbid.
    """
    assert ControlPlane._uses_cline_direct({"model_aliases": {}}) is False
    assert ControlPlane._uses_cline_direct({}) is False
    assert (
        ControlPlane._uses_cline_direct(
            {"model_aliases": {"_build_path": "planner_slices"}}
        )
        is False
    )
