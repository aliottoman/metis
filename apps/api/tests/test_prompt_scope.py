"""What the user asked, told apart from what they pasted.

The live failure: a request to file meeting notes against an account was
answered with instructions on how to open a project, because the notes quoted a
bank that was "willing to build an MVP/POC". Every routing prefilter in Metis
reads the raw prompt with a regex, so anything pasted into the chat gets a vote
on the route. These pin both halves of the fix shut — the instruction span, and
the gate that now yields to a destination the user actually named.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from waqil_api.config import Settings
from waqil_api.document_factory import is_explicit_document_request
from waqil_api.main import create_app
from waqil_api.model_provider import (
    is_explicit_toolify_request,
    is_project_build_instruction,
    is_project_build_request,
)
from waqil_api.prompt_scope import user_instruction
from waqil_api.queue_update import is_queue_update_request
from waqil_api.web_research import is_explicit_web_request

# The message that failed, near enough verbatim: nine words of instruction and a
# meeting note that mentions building things four times.
MEETING_NOTE = """Add these to the customer's notes from today's internal meeting “## Internal Meeting

* Use-case:
* Discussed the current status of the customer
* Customer overview:
* One of the bigger banks in the Ukraine
* They have their own internal fraud financial system
* They are willing to build an MVP/POC, to understand how quickly, advanced and
  secure our AI infrastructure can be to build AI agents in the middle of their
  processes and flows
* Want a beautiful demo for business aspect → Understand RAG + AI
* Next steps:
* Develop two main demos:
* Argus - Criteria B2B Onboarding
* Bank Document Extraction
* Cristina will develop the chatbot solution
* Objective is to wow the C-suite
"
"""


def test_the_instruction_is_what_survives_the_paste() -> None:
    assert user_instruction(MEETING_NOTE) == (
        "Add these to the customer's notes from today's internal meeting"
    )


def test_the_pasted_note_no_longer_reads_as_a_build_request() -> None:
    """The regression itself, at the predicate the gate consults."""
    assert is_project_build_instruction(MEETING_NOTE) is False
    # And the destination the user did name is still unmistakable, so the gate
    # would yield even if some future pattern matched the instruction anyway.
    assert is_queue_update_request(user_instruction(MEETING_NOTE)) is True


@pytest.mark.parametrize(
    "opening,closing",
    [("“", "”"), ('"', '"'), ("“", '"'), ("«", "»")],
)
def test_quote_marks_are_matched_by_kind_not_by_pair(
    opening: str, closing: str
) -> None:
    """Pasted text mixes them — the live message opened typographic and closed
    straight — and a strict pair would have missed exactly that message."""
    body = "we agreed to build an app for onboarding, and to ship it in Q3"
    prompt = f"File this against the account, from their email {opening}{body}{closing}"
    assert "build an app" not in user_instruction(prompt)


@pytest.mark.parametrize(
    "prompt",
    [
        "Summarize this thread for me:\n```\nmake me a deck of the results\n```",
        "What do you make of this?\n> we should build a new dashboard app\n> and search online for benchmarks",
        "Have a read of what they sent over\n\n## Proposal\n\nBuild an app that scores leads, and turn it into a tool",
    ],
)
def test_pasted_material_cannot_choose_the_route(prompt: str) -> None:
    """One rule, every prefilter: fenced, blockquoted and heading-led material
    is evidence, exactly as an attachment's contents already were."""
    assert is_project_build_instruction(prompt) is False
    assert is_explicit_document_request(prompt) is False
    assert is_explicit_toolify_request(prompt) is False
    assert is_explicit_web_request(prompt) is False


def test_a_short_quoted_title_stays_in_the_instruction() -> None:
    """Real build prompts name the project in quotes. Stripping that would take
    the instruction with it."""
    prompt = 'Build out this project from scratch: "Ledger" — supplier invoice intake'
    assert user_instruction(prompt) == prompt
    assert is_project_build_instruction(prompt) is True


@pytest.mark.parametrize(
    "prompt",
    [
        "Build this:\n\n## Spec\n\n- a FastAPI service, from scratch\n- a static frontend",
        "Do the following:\n```\ncreate app/main.py\n```",
        "## Spec\n\nBuild an app that tracks invoices",
    ],
)
def test_a_message_that_is_mostly_paste_is_about_the_paste(prompt: str) -> None:
    """The safety valve. An instruction that says nothing without its body is
    not one a prefilter can route on, so the whole prompt is read — which is
    how these behaved before, and how a pasted spec must keep behaving."""
    assert user_instruction(prompt) == prompt.strip()
    assert is_project_build_request(prompt) is True


def test_requirements_in_bullets_are_still_the_users_own_words() -> None:
    """Lists are how people write requirements, so they are never stripped."""
    prompt = "Build me an app with:\n- login\n- a dashboard"
    assert is_project_build_instruction(prompt) is True


def test_filing_a_pasted_note_is_not_answered_with_the_project_picker(
    tmp_path: Path,
) -> None:
    """End to end, on the message that failed: whatever the turn does with it,
    it must not be the one answer that was certainly wrong."""
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
            json={"content": MEETING_NOTE},
        ).json()["run_id"]
        for _ in range(400):
            run = client.get(f"/api/v1/runs/{run_id}").json()
            if run["status"] in {"completed", "failed"}:
                break
            time.sleep(0.01)
        assert run["status"] == "completed"
        reply = client.get(f"/api/v1/conversations/{conversation_id}/messages").json()[
            -1
        ]["content"]
        assert "no project is open" not in reply
