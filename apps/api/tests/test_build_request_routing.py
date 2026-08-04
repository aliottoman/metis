"""Telling "draw my architecture" apart from "build these files".

Both are described with the word architecture, but only one is a picture. The
host decides this deterministically, before any model runs — a build request
routed to the diagram tool asks for a component graph and fails on the schema
minutes later, which is the failure this pins shut.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from waqil_api.config import Settings
from waqil_api.contracts import PlanEnvelopeV1, PlanningRequestV1, RiskLevel
from waqil_api.main import create_app
from waqil_api.model_provider import (
    default_routing_catalog,
    is_new_application_request,
    is_project_build_instruction,
    is_project_build_request,
    validate_plan_semantics,
)

ARCH = default_routing_catalog().architecture_tool

# The real prompt that failed: it names an architecture but asks for files.
AGENT_SHOWCASE = """Build out this project from scratch: "Agent Showcase" — a multi-agent
demo using the OCI Generative AI Responses API, with document extraction and a polished UI.
ARCHITECTURE
- FastAPI backend serving the API and a static frontend from app/static/.
- app/agents/planner.py — decomposes the user's question into steps
- app/orchestrator.py: runs plan then extract then synthesize
CODE CONVENTIONS: one-line comment above every function.
ALSO CREATE requirements.txt and README.md."""


def _request(prompt: str) -> PlanningRequestV1:
    return PlanningRequestV1(run_id="r", conversation_id="c", prompt=prompt)


def _direct() -> PlanEnvelopeV1:
    return PlanEnvelopeV1(summary="answer directly", route="direct", risk_level=RiskLevel.R0)


@pytest.mark.parametrize(
    "prompt",
    [
        AGENT_SHOWCASE,
        "create app/main.py and app/config.py for a small service",
        "scaffold a FastAPI service from scratch",
        "add a requirements.txt and a Dockerfile",
        "write the code for the parser, one-line comment above each function",
    ],
)
def test_file_writing_requests_are_not_architecture_requests(prompt: str) -> None:
    assert is_project_build_request(prompt) is True
    # The host would otherwise force the diagram tool and reject a direct plan.
    validate_plan_semantics(_direct(), _request(prompt))


@pytest.mark.parametrize(
    "prompt",
    [
        "Build an app where I upload supplier invoice images and Grok extracts the fields",
        "create a website for tracking my reading list",
        "make me an internal dashboard for customer wins",
        "spin up a small api that returns portfolio health",
        "develop a new service that syncs my Notion pages",
        # The rebuild phrasing real specs use — no indefinite article. The
        # exact historical Ledger benchmark prompt opens this way, and it got
        # strict build mode without the scaffold this classifier gates.
        'Build out this project from scratch: "Ledger" — supplier invoice intake',
        "build out the whole app from scratch with a FastAPI backend",
    ],
)
def test_generic_application_requests_enter_strict_build_mode(prompt: str) -> None:
    """The Ledger regression: no filename, no artifact — still a build."""
    assert is_new_application_request(prompt) is True
    assert is_project_build_request(prompt) is True
    assert is_project_build_instruction(prompt) is True


@pytest.mark.parametrize(
    "prompt",
    [
        "make the app faster on cold start",
        "how do I make an API call from python?",
        "what does the service do when the queue is full?",
        "add a docstring to the parser function",
        "create a plan for migrating the database",
        "create a tool that reverses text",
    ],
)
def test_non_application_requests_stay_out_of_whole_app_mode(prompt: str) -> None:
    assert is_new_application_request(prompt) is False


@pytest.mark.parametrize(
    "prompt",
    [
        "draw the architecture for our payments platform",
        "create a reference architecture diagram from the attached doc",
        "visualize the topology of the ingestion pipeline",
        "sketch the architecture for app/main.py and its dependencies",
    ],
)
def test_picture_requests_still_route_to_the_architecture_tool(prompt: str) -> None:
    """An explicit picture word wins even when source paths are mentioned."""
    with pytest.raises(ValueError, match="architecture requests require"):
        validate_plan_semantics(_direct(), _request(prompt))


@pytest.mark.parametrize(
    "prompt",
    [
        "what does app/main.py do?",
        "explain the architecture of this service",
        "is requirements.txt up to date?",
    ],
)
def test_questions_about_files_are_not_build_instructions(prompt: str) -> None:
    """Naming a file is not asking for one to be written."""
    assert is_project_build_instruction(prompt) is False


def test_build_instruction_needs_both_intent_and_a_build_signal() -> None:
    assert is_project_build_instruction(AGENT_SHOWCASE) is True
    # Creation intent alone is not enough; nothing here names files or artifacts.
    assert is_project_build_instruction("create a summary of this meeting") is False


def test_build_request_without_a_project_explains_how_to_open_one(tmp_path: Path) -> None:
    """The one answer the host can give deterministically, in one turn."""
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        repo_root=tmp_path,
        model_backend="deterministic",
        reference_runner_mode="deterministic",
        allow_test_backends=True,
    )
    with TestClient(create_app(settings)) as client:
        conversation_id = client.post("/api/v1/conversations", json={}).json()["id"]
        run_id = client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json={"content": AGENT_SHOWCASE},
        ).json()["run_id"]
        for _ in range(200):
            run = client.get(f"/api/v1/runs/{run_id}").json()
            if run["status"] in {"completed", "failed"}:
                break
            time.sleep(0.01)
        assert run["status"] == "completed"
        reply = client.get(
            f"/api/v1/conversations/{conversation_id}/messages"
        ).json()[-1]["content"]
        assert "no project is open" in reply
        assert "Project" in reply and "Assets" in reply
        # It must not have gone near the architecture tool. The events endpoint
        # is an SSE stream, so the event names are read off the `event:` lines
        # rather than from the payloads (one of which mentions diagrams).
        stream = client.get(f"/api/v1/runs/{run_id}/events").text
        names = [
            line.removeprefix("event: ").strip()
            for line in stream.splitlines()
            if line.startswith("event: ")
        ]
        assert names, "the run emitted no events"
        assert not any("architecture" in name or "reference" in name for name in names)
