"""P5 — the graph-agent answer path: synthesize <-> ground_review bounded loop.

These drive a real chat turn end-to-end through the graph (plan -> synthesize ->
ground_review -> {revise|publish}) with an injected model and corpus, so the
edges, the loop counter, and the termination bound are all exercised for real."""
from __future__ import annotations

import json
import re
import time

from fastapi.testclient import TestClient

from waqil_api.contracts import KnowledgeSnippetV1, ModelResultV1
from waqil_api.main import create_app
from waqil_api.model_provider import DeterministicModelProvider

_PROMPT = "Summarise what my project does, using my own notes."


def _event_payload(events: str, event_type: str) -> dict:
    """Pull the JSON payload of the first SSE event of the given type. The SSE
    `data:` line carries the full event envelope; the domain fields are nested
    under `payload`."""
    match = re.search(rf"event: {re.escape(event_type)}\ndata: (.+)", events)
    assert match, f"no {event_type} event in stream"
    return json.loads(match.group(1))["payload"]


def _wait(client: TestClient, run_id: str, statuses: set[str]) -> dict:
    deadline = time.monotonic() + 5
    latest: dict = {}
    while time.monotonic() < deadline:
        latest = client.get(f"/api/v1/runs/{run_id}").json()
        if latest["status"] in statuses:
            return latest
        time.sleep(0.02)
    raise AssertionError(latest)


class _ScriptedModel(DeterministicModelProvider):
    """Returns whatever answer text the test scripts, tracking generate() calls.
    Inherits `plan()`, which routes a non-architecture prompt to `direct`."""

    def __init__(self, answers: list[str]) -> None:
        self._answers = answers
        self.generate_calls = 0
        self.last_request = None

    async def generate(self, request, on_token=None, *, model_aliases=None, on_reasoning=None):
        self.last_request = request
        index = min(self.generate_calls, len(self._answers) - 1)
        content = self._answers[index]
        self.generate_calls += 1
        if on_token is not None:
            await on_token(content)
        return ModelResultV1(model="scripted", content=content, fallback=False)


class _StubCorpus:
    """Injectable corpus returning a fixed snippet at a chosen relevance score."""

    def __init__(self, score: float, *, empty: bool = False) -> None:
        self._score = score
        self._empty = empty
        self.last_provider = None

    def available(self) -> bool:
        return True

    async def retrieve(self, query: str, limit=None, *, on_stage=None, provider=None):
        self.last_provider = provider
        if on_stage is not None:
            await on_stage("embedding", "Embedding your question…")
            await on_stage("reranking", "Reranking the best matches…")
        if self._empty:
            return []
        return [
            KnowledgeSnippetV1(
                source_label="proj",
                provider=provider or "local",
                rel_path="notes.md",
                symbol=None,
                start_line=1,
                text="The project is a local-first agent.",
                score=self._score,
            )
        ]


class _FailingCorpus:
    def available(self) -> bool:
        return True

    async def retrieve(self, query: str, limit=None, *, on_stage=None, provider=None):
        if on_stage is not None:
            await on_stage("embedding", "Embedding your question…")
        raise RuntimeError(
            "{'target_service': 'generative_ai_inference', 'status': 404, "
            "'opc-request-id': 'SECRET-PROVIDER-REQUEST-ID'}"
        )


def _run_turn(
    client: TestClient, model, corpus, *, knowledge_scope: str = "auto"
) -> tuple[dict, dict, str]:
    runtime = client.app.state.runtime
    runtime.control_plane.model = model
    runtime.control_plane.corpus = corpus
    conversation = client.post("/api/v1/conversations", json={}).json()
    accepted = client.post(
        f"/api/v1/conversations/{conversation['id']}/messages",
        json={
            "content": _PROMPT,
            "attachment_ids": [],
            "knowledge_scope": knowledge_scope,
        },
    ).json()
    completed = _wait(client, accepted["run_id"], {"completed", "failed"})
    events = client.get(f"/api/v1/runs/{accepted['run_id']}/events?after=0").text
    messages = client.get(
        f"/api/v1/conversations/{conversation['id']}/messages"
    ).json()
    assistant = [m for m in messages if m["role"] == "assistant"][-1]
    return completed, {"events": events}, assistant["content"]


def test_strong_retrieval_uncited_answer_triggers_one_bounded_revision(settings) -> None:
    # First answer cites nothing; strong retrieval (0.95) -> exactly one revision.
    model = _ScriptedModel(
        ["Here is a summary with no citations.", "Per my notes [1], it is a local agent."]
    )
    with TestClient(create_app(settings)) as client:
        completed, meta, content = _run_turn(client, model, _StubCorpus(score=0.95))
    assert completed["status"] == "completed", completed
    assert model.generate_calls == 2  # synthesize ran twice: original + one revision
    assert "event: answer.grounding_reviewed" in meta["events"]
    assert "[1]" in content  # the final, persisted answer is the revised cited one
    assert "**Sources**" in content


def test_cited_answer_is_accepted_without_revision(settings) -> None:
    model = _ScriptedModel(["The project is a local agent, per my notes [1]."])
    with TestClient(create_app(settings)) as client:
        completed, _, content = _run_turn(client, model, _StubCorpus(score=0.95))
    assert completed["status"] == "completed"
    assert model.generate_calls == 1  # already grounded -> no revision
    assert "[1]" in content


def test_weak_retrieval_does_not_force_a_revision(settings) -> None:
    # Uncited answer, but retrieval is weak (0.10 < threshold) -> no revision,
    # so the verifier never nags when the retrieved material was not relevant.
    model = _ScriptedModel(["A general answer with no citations."])
    with TestClient(create_app(settings)) as client:
        completed, _, content = _run_turn(client, model, _StubCorpus(score=0.10))
    assert completed["status"] == "completed"
    assert model.generate_calls == 1
    assert "[1]" not in content


def test_revision_loop_is_bounded_when_the_model_keeps_ignoring_sources(settings) -> None:
    # The model never cites; the loop must still stop at answer_max_revisions (1),
    # so synthesize runs at most twice — the bound lives in the graph, not the model.
    model = _ScriptedModel(["No citations ever."])  # same uncited answer every pass
    with TestClient(create_app(settings)) as client:
        completed, _, _ = _run_turn(client, model, _StubCorpus(score=0.95))
    assert completed["status"] == "completed"
    assert model.generate_calls == 2  # original + exactly one bounded revision


def test_relevance_gate_drops_below_threshold_snippets_from_injection(settings) -> None:
    # P6a: a snippet scoring 0.01 (< corpus_min_relevance 0.05) is gated out of
    # the answer prompt, so nothing weak is injected and no revision is provoked.
    model = _ScriptedModel(["An answer."])
    with TestClient(create_app(settings)) as client:
        completed, meta, _ = _run_turn(client, model, _StubCorpus(score=0.01))
    assert completed["status"] == "completed"
    retrieved = _event_payload(meta["events"], "context.retrieved")
    assert retrieved["knowledge_snippet_count"] == 0
    assert retrieved["knowledge_gated_out"] == 1
    assert model.generate_calls == 1  # no injection -> no grounding revision


def test_notion_only_scope_filters_retrieval_to_notion(settings) -> None:
    model = _ScriptedModel(["The project is local-first [1]."])
    corpus = _StubCorpus(score=0.95)
    with TestClient(create_app(settings)) as client:
        completed, meta, content = _run_turn(
            client, model, corpus, knowledge_scope="notion"
        )
    assert completed["status"] == "completed"
    assert corpus.last_provider == "notion"
    assert "[1]" in content
    retrieved = _event_payload(meta["events"], "context.retrieved")
    assert retrieved["knowledge_scope"] == "notion"


def test_notion_only_scope_refuses_without_support_and_skips_generation(settings) -> None:
    model = _ScriptedModel(["This should never be generated."])
    corpus = _StubCorpus(score=0.95, empty=True)
    with TestClient(create_app(settings)) as client:
        completed, _, content = _run_turn(
            client, model, corpus, knowledge_scope="notion"
        )
    assert completed["status"] == "completed"
    assert corpus.last_provider == "notion"
    assert model.generate_calls == 0
    assert "couldn't find relevant support" in content


def _run_attachment_turn(
    client: TestClient, model, corpus, *, knowledge_scope: str = "auto"
) -> tuple[dict, str, str]:
    """Ask about an attached document while retrieval also returns a strong hit —
    the shape that used to get the document answer rewritten around Notion."""
    runtime = client.app.state.runtime
    runtime.control_plane.model = model
    runtime.control_plane.corpus = corpus
    uploaded = client.post(
        "/api/v1/uploads",
        files={
            "file": (
                "launch-brief.md",
                b"The launch code is ORCHID-73.",
                "text/markdown",
            )
        },
    ).json()
    conversation = client.post("/api/v1/conversations", json={}).json()
    accepted = client.post(
        f"/api/v1/conversations/{conversation['id']}/messages",
        json={
            "content": "What is the launch code in the attached brief?",
            "attachment_ids": [uploaded["id"]],
            "knowledge_scope": knowledge_scope,
        },
    ).json()
    completed = _wait(client, accepted["run_id"], {"completed", "failed"})
    events = client.get(f"/api/v1/runs/{accepted['run_id']}/events?after=0").text
    messages = client.get(
        f"/api/v1/conversations/{conversation['id']}/messages"
    ).json()
    assistant = [m for m in messages if m["role"] == "assistant"][-1]
    return completed, events, assistant["content"]


def test_attached_document_is_citable_alongside_strong_retrieval(settings) -> None:
    # The stub corpus contributes one passage as [1], so the attached document is
    # offered as [2]. Citing it counts as grounding: no revision, and the Sources
    # list names the file rather than only the retrieved passage.
    model = _ScriptedModel(["The launch code is ORCHID-73 [2]."])
    with TestClient(create_app(settings)) as client:
        completed, events, content = _run_attachment_turn(
            client, model, _StubCorpus(score=0.95)
        )
    assert completed["status"] == "completed", completed
    assert model.generate_calls == 1  # the document answer survives as written
    prompt = model.last_request.user_prompt
    assert "[2] Attached document — launch-brief.md" in prompt
    # The number must sit on the file's own header, next to the text it labels —
    # a lone index line is easy for a smaller model to lose track of.
    assert "--- [2] launch-brief.md (untrusted attachment) ---" in prompt
    assert "**Sources**" in content
    assert "[2] Attached document — launch-brief.md" in content
    verdict = _event_payload(events, "answer.grounding_reviewed")
    assert verdict["has_attachments"] is True
    assert verdict["cited"] is True
    assert verdict["revision"] is False


def test_uncited_document_answer_is_published_without_a_revision(settings) -> None:
    # The regression that produced a Notion-only reply: an answer drawn from the
    # attachment carries no [n] marker, so the gate read it as ungrounded and sent
    # a revision that rewrote it around the retrieved passages. An attachment must
    # now opt the turn out — the document answer is published exactly as written.
    model = _ScriptedModel(
        ["The launch code is ORCHID-73.", "Per my notes [1], there is no code."]
    )
    with TestClient(create_app(settings)) as client:
        completed, events, content = _run_attachment_turn(
            client, model, _StubCorpus(score=0.95)
        )
    assert completed["status"] == "completed", completed
    assert model.generate_calls == 1  # the second, corpus-shaped answer never runs
    assert content == "The launch code is ORCHID-73."
    verdict = _event_payload(events, "answer.grounding_reviewed")
    assert verdict["has_attachments"] is True
    assert verdict["strong_retrieval"] is True  # retrieval was strong, and ignored
    assert verdict["cited"] is False
    assert verdict["revision"] is False


def test_notion_only_scope_never_asks_the_model_to_cite_a_withheld_attachment(
    settings,
) -> None:
    # Notion-only mode deliberately withholds attachments from the prompt, so the
    # attachment opt-out must not apply there: the gate still grounds the answer
    # against the Notion passages that were the only evidence the model saw.
    model = _ScriptedModel(["An answer with no citations.", "Per Notion [1]."])
    with TestClient(create_app(settings)) as client:
        completed, events, _ = _run_attachment_turn(
            client, model, _StubCorpus(score=0.95), knowledge_scope="notion"
        )
    assert completed["status"] == "completed", completed
    assert model.generate_calls == 2
    verdict = _event_payload(events, "answer.grounding_reviewed")
    assert verdict["has_attachments"] is False
    critique = model.last_request.user_prompt
    assert "highly relevant material from the user's own knowledge" in critique
    assert "attached document is the primary factual source" not in critique


def test_attachment_is_primary_evidence_when_cloud_knowledge_lookup_fails(settings) -> None:
    model = _ScriptedModel(["The attached launch code is ORCHID-73."])
    with TestClient(create_app(settings)) as client:
        runtime = client.app.state.runtime
        runtime.control_plane.model = model
        runtime.control_plane.corpus = _FailingCorpus()
        uploaded = client.post(
            "/api/v1/uploads",
            files={
                "file": (
                    "launch-brief.md",
                    b"The launch code is ORCHID-73.",
                    "text/markdown",
                )
            },
        ).json()
        conversation = client.post("/api/v1/conversations", json={}).json()
        accepted = client.post(
            f"/api/v1/conversations/{conversation['id']}/messages",
            json={
                "content": "What is the launch code in the attached brief?",
                "attachment_ids": [uploaded["id"]],
                "knowledge_scope": "auto",
            },
        ).json()
        completed = _wait(client, accepted["run_id"], {"completed", "failed"})
        events = client.get(
            f"/api/v1/runs/{accepted['run_id']}/events?after=0"
        ).text

    assert completed["status"] == "completed"
    assert model.last_request is not None
    assert "use the attachment evidence as the primary factual source" in (
        model.last_request.system_prompt
    )
    assert "<attachment-evidence>" in model.last_request.user_prompt
    assert "ORCHID-73" in model.last_request.user_prompt
    error = _event_payload(events, "context.knowledge_error")
    assert error["category"] == "model_or_region_unavailable"
    assert "Continuing with the attached document" in error["summary"]
    assert "SECRET-PROVIDER-REQUEST-ID" not in events
