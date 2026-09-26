"""End-to-end source routing with a scripted semantic decision.

The model is deliberately scripted here: these tests check that the chat graph
honors a decision, runs the right adapters, protects public queries, and bounds
the evidence follow-up. The separate human-labelled matrix is for measuring
the quality of real model decisions, which a fake model cannot establish.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from waqil_api.contracts import KnowledgeSnippetV1, ModelResultV1
from waqil_api.evidence_routing import (
    EvidencePlanV1,
    EvidenceReviewV1,
    has_official_release_coverage,
    preserve_public_component,
    safe_public_open_urls,
    safe_public_queries,
)
from waqil_api.main import create_app
from waqil_api.model_provider import DeterministicModelProvider


class _Model(DeterministicModelProvider):
    def __init__(
        self,
        *,
        sources: list[str],
        queries: list[str] | None = None,
        focus_terms: list[str] | None = None,
        verify: bool = False,
        review: EvidenceReviewV1 | None = None,
        model_name: str = "scripted",
    ) -> None:
        self.sources = sources
        self.queries = queries or []
        self.focus_terms = focus_terms or []
        self.verify = verify
        self.review = review or EvidenceReviewV1(adequate=True)
        self.model_name = model_name
        self.structured_calls: list[str] = []

    async def _structured(self, schema, **kwargs):
        del kwargs
        self.structured_calls.append(schema.__name__)
        if schema is EvidencePlanV1:
            return EvidencePlanV1(
                sources=self.sources,
                public_queries=self.queries,
                focus_terms=self.focus_terms,
                action="answer",
                needs_verification=self.verify,
            )
        if schema is EvidenceReviewV1:
            return self.review
        raise AssertionError(f"Unexpected structured call: {schema.__name__}")

    async def generate(
        self, request, on_token=None, *, model_aliases=None, on_reasoning=None
    ):
        del request, model_aliases, on_reasoning
        content = "The answer is supported by the supplied source [1]."
        if on_token is not None:
            await on_token(content)
        return ModelResultV1(model=self.model_name, content=content, fallback=False)


class _BrokenPlanner(_Model):
    async def _structured(self, schema, **kwargs):
        del schema, kwargs
        raise RuntimeError("synthetic planner outage")


class _Web:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def available(self) -> bool:
        return True

    async def retrieve(
        self,
        prompt: str,
        *,
        queries: list[str] | None = None,
        focus_terms: list[str] | None = None,
        include_prompt_urls: bool = True,
    ) -> list[KnowledgeSnippetV1]:
        self.calls.append(
            {
                "prompt": prompt,
                "queries": queries,
                "focus_terms": focus_terms,
                "include_prompt_urls": include_prompt_urls,
            }
        )
        # An explicit empty list means no public search. An explicit URL can
        # still be opened directly, exactly as the real adapter behaves.
        if queries == [] and not (include_prompt_urls and "https://" in prompt):
            return []
        return [
            KnowledgeSnippetV1(
                source_label="Official source",
                provider="web",
                rel_path="https://example.com/release",
                source_url="https://example.com/release",
                symbol=None,
                start_line=None,
                text="Official release notes describe the feature.",
                score=0.95,
            )
        ]


class _Corpus:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def available(self) -> bool:
        return True

    async def retrieve(self, query: str, limit=None, *, on_stage=None, provider=None):
        del limit, on_stage
        self.calls.append({"query": query, "provider": provider})
        return [
            KnowledgeSnippetV1(
                source_label="Private note",
                provider=provider or "notion",
                rel_path="notes.md",
                symbol=None,
                start_line=1,
                text="The private note records the user's comparison point.",
                score=0.95,
            )
        ]


def _payload(events: str, name: str) -> dict[str, Any]:
    match = re.search(rf"event: {re.escape(name)}\ndata: (.+)", events)
    assert match is not None, f"Missing {name}: {events[:2000]}"
    return json.loads(match.group(1))["payload"]


def _turn(
    client: TestClient,
    model: _Model,
    web: _Web,
    corpus: _Corpus,
    prompt: str,
    *,
    scope: str = "auto",
) -> tuple[dict[str, Any], str]:
    plane = client.app.state.runtime.control_plane
    plane.model = model
    plane.web = web
    plane.corpus = corpus
    conversation_id = client.post("/api/v1/conversations", json={}).json()["id"]
    accepted = client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={"content": prompt, "attachment_ids": [], "knowledge_scope": scope},
    ).json()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        run = client.get(f"/api/v1/runs/{accepted['run_id']}").json()
        if run["status"] in {"completed", "failed", "awaiting_approval", "awaiting_input"}:
            break
        time.sleep(0.02)
    events = client.get(f"/api/v1/runs/{accepted['run_id']}/events?after=0").text
    assert run["status"] == "completed", (run, events[-3000:])
    return _payload(events, "evidence.planned"), events


@pytest.mark.parametrize(
    "sources,query,web_calls,private_calls",
    [
        ([], None, 0, 0),
        (["private"], None, 0, 1),
        (["web"], "Cline SDK release notes", 1, 0),
        (["private", "web"], "Cline SDK release notes", 1, 1),
    ],
)
def test_semantic_source_plan_controls_real_retrieval(
    settings, sources, query, web_calls, private_calls
) -> None:
    model = _Model(sources=sources, queries=[query] if query else [])
    web, corpus = _Web(), _Corpus()
    with TestClient(create_app(settings)) as client:
        planned, events = _turn(
            client,
            model,
            web,
            corpus,
            "Compare my Cline notes with the current public SDK release.",
        )
    assert set(planned["sources"]) == set(sources)
    assert planned["method"] == "semantic"
    assert len(web.calls) == web_calls
    assert len(corpus.calls) == private_calls
    retrieved = _payload(events, "context.retrieved")
    assert retrieved["web_requested"] is ("web" in sources)
    assert retrieved["private_requested"] is ("private" in sources)


@pytest.mark.parametrize(
    "scope,expected_sources,web_calls,private_calls",
    [
        ("web", ["web"], 1, 0),
        ("notion", ["private"], 0, 1),
    ],
)
def test_explicit_source_scope_overrides_semantic_model(
    settings, scope, expected_sources, web_calls, private_calls
) -> None:
    model = _Model(sources=[])
    web, corpus = _Web(), _Corpus()
    with TestClient(create_app(settings)) as client:
        planned, _ = _turn(
            client,
            model,
            web,
            corpus,
            "Explain Python list comprehensions.",
            scope=scope,
        )
    assert planned["sources"] == expected_sources
    assert planned["method"] == "explicit_scope"
    assert "EvidencePlanV1" not in model.structured_calls
    assert model.structured_calls.count("EvidenceReviewV1") == (1 if scope == "web" else 0)
    assert len(web.calls) == web_calls
    assert len(corpus.calls) == private_calls


def test_auto_never_sends_private_phrases_as_public_queries(settings) -> None:
    model = _Model(
        sources=["private", "web"],
        queries=["my notes about secret initiative", "Cline SDK changelog"],
    )
    web, corpus = _Web(), _Corpus()
    with TestClient(create_app(settings)) as client:
        planned, _ = _turn(
            client,
            model,
            web,
            corpus,
            "Compare my notes about the secret initiative with Cline's SDK changes.",
        )
    assert planned["query_count"] == 1
    assert len(web.calls) == 1
    assert web.calls[0]["queries"] == ["Cline SDK changelog"]
    assert len(corpus.calls) == 1


def test_rejected_private_query_has_no_raw_prompt_fallback(settings) -> None:
    model = _Model(
        sources=["web"],
        queries=["our internal meeting about the secret initiative"],
    )
    web, corpus = _Web(), _Corpus()
    with TestClient(create_app(settings)) as client:
        planned, events = _turn(
            client,
            model,
            web,
            corpus,
            "Check what our internal meeting says about the secret initiative online.",
        )
    assert planned["query_count"] == 0
    assert len(web.calls) == 1
    assert web.calls[0]["queries"] == []
    assert corpus.calls == []
    assert _payload(events, "context.retrieved")["web_source_count"] == 0


def test_private_note_phrases_are_rejected_by_query_guard() -> None:
    assert safe_public_queries(
        ["my notes about Cline SDK", "our internal meeting action items"]
    ) == []
    assert safe_public_queries(["Cline SDK 0.0.86 release notes"]) == [
        "Cline SDK 0.0.86 changelog"
    ]
    assert safe_public_queries(["my release notes for Cline SDK"]) == []


def test_public_component_query_keeps_requested_harness_track() -> None:
    assert preserve_public_component(
        "What new Cline harness features were released?",
        ["Cline VS Code extension recent releases", "Cline latest changelog"],
        ["Cline", "harness", "releases"],
    ) == ["Cline SDK changelog", "Cline VS Code extension recent releases"]
    assert preserve_public_component(
        "What new Cline harness features were released?",
        ["Cline official changelog", "Cline harness new features"],
        ["Cline", "harness"],
    ) == ["Cline SDK changelog", "Cline official changelog"]
    assert preserve_public_component(
        "Compare my private SDK notes with the latest release.",
        ["latest release"],
        ["my private SDK notes"],
    ) == ["latest release"]


def test_official_component_changelog_can_satisfy_release_coverage() -> None:
    sdk = {
        "provider": "web",
        "source_label": "Cline SDK Changelog",
        "source_url": "https://github.com/cline/cline/blob/main/sdk/CHANGELOG.md",
        "text": (
            "# Cline SDK Changelog\n## 0.0.86\nLocal model recovery.\n"
            "## 0.0.85\nRetry improvements.\n"
            "## 0.0.83\nProvider-native web search enabled by default."
        ),
    }
    question = "What new features did Cline release recently for its harness?"
    assert has_official_release_coverage(question, [sdk], ["Cline", "harness"])
    assert not has_official_release_coverage(
        "What did Cline SDK 0.0.80 and 0.0.83 add?", [sdk], ["Cline", "SDK"]
    )
    generic = {**sdk, "source_url": "https://github.com/cline/cline/blob/main/CHANGELOG.md"}
    assert not has_official_release_coverage(question, [generic], ["Cline", "harness"])
    external = {**sdk, "source_url": "https://github.com/other/cline/blob/main/sdk/CHANGELOG.md"}
    assert not has_official_release_coverage(question, [external], ["Cline", "harness"])


def test_official_changelog_skips_slow_review_and_drops_aggregators(settings) -> None:
    class _OfficialWeb(_Web):
        async def retrieve(self, prompt, **kwargs):
            self.calls.append({"prompt": prompt, **kwargs})
            body = "## 0.0.86\nLocal recovery.\n## 0.0.83\nNative web search."
            urls = (
                "https://github.com/cline/cline/blob/main/sdk/CHANGELOG.md",
                "https://github.com/cline/cline/blob/main/CHANGELOG.md",
                "https://example.com/cline-summary",
            )
            return [
                KnowledgeSnippetV1(
                    source_label="Cline SDK Changelog" if index == 0 else "Release summary",
                    provider="web",
                    rel_path=url,
                    source_url=url,
                    text=body if index == 0 else "A long secondary release summary. " * 8,
                    score=0.95 - index * 0.05,
                )
                for index, url in enumerate(urls)
            ]

    model = _Model(
        sources=["web"],
        queries=["Cline recent releases"],
        focus_terms=["Cline", "harness"],
        verify=True,
    )
    web, corpus = _OfficialWeb(), _Corpus()
    with TestClient(create_app(settings)) as client:
        _, events = _turn(
            client,
            model,
            web,
            corpus,
            "What new Cline harness features were released recently?",
        )
    assert web.calls[0]["queries"][0] == "Cline SDK changelog"
    assert model.structured_calls.count("EvidenceReviewV1") == 0
    assert _payload(events, "evidence.reviewed")["method"] == "official_changelog"
    assert _payload(events, "context.retrieved")["web_source_count"] == 2


def test_lossy_official_changelog_is_reviewed_with_older_entries(settings) -> None:
    late_feature = (
        "Provider-native web search is enabled by default for supported models."
    )

    class _LongOfficialWeb(_Web):
        async def retrieve(self, prompt, **kwargs):
            result = await super().retrieve(prompt, **kwargs)
            result[0].source_url = (
                "https://github.com/example/sdk/blob/main/sdk/CHANGELOG.md"
            )
            result[0].rel_path = result[0].source_url
            result[0].text = (
                "# Example SDK Changelog\n## 0.0.86\n"
                + "- New agent session capability.\n" * 45
                + "## 0.0.83\n- "
                + late_feature
                + "\n[18 other release entries omitted from this excerpt]"
            )
            return result

    class _CaptureReview(_Model):
        reviewed_sources = ""
        answer_prompt = ""

        async def _structured(self, schema, **kwargs):
            if schema is EvidenceReviewV1:
                self.reviewed_sources = kwargs["user_prompt"]
            return await super()._structured(schema, **kwargs)

        async def generate(self, request, on_token=None, *, model_aliases=None, on_reasoning=None):
            self.answer_prompt = request.user_prompt
            return await super().generate(
                request, on_token, model_aliases=model_aliases,
                on_reasoning=on_reasoning,
            )

    model = _CaptureReview(
        sources=["web"],
        queries=["Example SDK changelog"],
        focus_terms=["example", "sdk"],
        verify=True,
    )
    web, corpus = _LongOfficialWeb(), _Corpus()
    with TestClient(create_app(settings)) as client:
        _, events = _turn(
            client, model, web, corpus,
            "What new Example SDK features were released recently?",
        )
    reviewed = _payload(events, "evidence.reviewed")
    assert reviewed["method"] == "model"
    assert reviewed["review_calls"] == 1
    assert late_feature in model.reviewed_sources
    assert late_feature in model.answer_prompt
    assert "Scan all release sections" in model.answer_prompt
    assert corpus.calls == []


def test_user_supplied_public_url_is_opened_even_if_model_misses_web(settings) -> None:
    model = _Model(sources=[])
    web, corpus = _Web(), _Corpus()
    with TestClient(create_app(settings)) as client:
        planned, _ = _turn(
            client,
            model,
            web,
            corpus,
            "Summarize https://example.com/release for me.",
        )
    assert planned["sources"] == ["web"]
    assert len(web.calls) == 1
    assert web.calls[0]["queries"] == []
    assert corpus.calls == []


def test_inadequate_web_evidence_gets_one_bounded_followup(settings) -> None:
    model = _Model(
        sources=["web"],
        queries=["Cline SDK 0.0.86 changelog"],
        verify=True,
        review=EvidenceReviewV1(
            adequate=False,
            followup_queries=["Cline SDK 0.0.83 native web search"],
            focus_terms=["0.0.83"],
        ),
    )
    web, corpus = _Web(), _Corpus()
    with TestClient(create_app(settings)) as client:
        _, events = _turn(
            client,
            model,
            web,
            corpus,
            "What did Cline SDK 0.0.83 and 0.0.86 add?",
        )
    assert model.structured_calls.count("EvidenceReviewV1") == 1
    assert len(web.calls) == 2
    assert web.calls[1]["queries"] == ["Cline SDK 0.0.83 native web search"]
    assert web.calls[1]["include_prompt_urls"] is False
    assert _payload(events, "evidence.reviewed")["followup_source_count"] == 1
    assert corpus.calls == []


def test_model_can_only_open_urls_from_public_search_results() -> None:
    snippets = [
        {
            "provider": "web",
            "source_url": "https://example.com/release",
            "text": "A public release excerpt.",
        },
        {
            "provider": "web",
            "source_url": "http://127.0.0.1/private",
            "text": "Unsafe URL.",
        },
        {
            "provider": "notion",
            "source_url": "https://example.com/private",
            "text": "Private document.",
        },
    ]
    assert safe_public_open_urls(
        [
            "https://example.com/release",
            "https://example.com/elsewhere?secret=123",
            "http://127.0.0.1/private",
            "https://example.com/private",
        ],
        snippets,
    ) == ["https://example.com/release"]


def test_inadequate_excerpt_opens_only_existing_source(settings) -> None:
    model = _Model(
        sources=["web"],
        queries=["Cline SDK changelog"],
        verify=True,
        review=EvidenceReviewV1(
            adequate=False,
            open_urls=[
                "https://example.com/release",
                "https://example.com/elsewhere?private=1",
            ],
        ),
    )
    web, corpus = _Web(), _Corpus()
    with TestClient(create_app(settings)) as client:
        _, events = _turn(client, model, web, corpus, "What changed in Cline SDK?")
    assert len(web.calls) == 2
    assert web.calls[1]["prompt"] == "Read https://example.com/release"
    assert web.calls[1]["queries"] == []
    assert _payload(events, "evidence.reviewed")["opened_url_count"] == 1
    assert corpus.calls == []


def test_opened_page_replaces_short_search_excerpt_for_same_url(settings) -> None:
    class _CaptureModel(_Model):
        answer_prompt = ""

        async def generate(self, request, on_token=None, *, model_aliases=None, on_reasoning=None):
            self.answer_prompt = request.user_prompt
            return await super().generate(
                request, on_token, model_aliases=model_aliases,
                on_reasoning=on_reasoning,
            )

    class _SameUrlWeb(_Web):
        async def retrieve(self, prompt, **kwargs):
            result = await super().retrieve(prompt, **kwargs)
            if len(self.calls) == 2:
                result[0].text = "Brief search excerpt."
            elif len(self.calls) == 3:
                result[0].text = (
                    "Full opened page says Cline SDK 0.0.83 enabled native web search."
                )
            return result

    model = _CaptureModel(
        sources=["web"],
        queries=["Cline SDK changelog"],
        verify=True,
        review=EvidenceReviewV1(
            adequate=False,
            followup_queries=["Cline SDK native web search"],
            open_urls=["https://example.com/release"],
        ),
    )
    web, corpus = _SameUrlWeb(), _Corpus()
    with TestClient(create_app(settings)) as client:
        _turn(client, model, web, corpus, "What changed in Cline SDK?")
    assert "Full opened page says Cline SDK" in model.answer_prompt
    assert "Brief search excerpt" not in model.answer_prompt
    assert len(web.calls) == 3
    assert corpus.calls == []


def test_new_followup_passage_gets_second_coverage_review(settings) -> None:
    class _ReviewTwice(_Model):
        async def _structured(self, schema, **kwargs):
            if schema is EvidenceReviewV1:
                self.structured_calls.append(schema.__name__)
                if self.structured_calls.count(schema.__name__) == 1:
                    return EvidenceReviewV1(
                        adequate=False,
                        followup_queries=["Cline SDK 0.0.83 native web search"],
                    )
                return EvidenceReviewV1(adequate=True)
            return await super()._structured(schema, **kwargs)

    class _NewSourceWeb(_Web):
        async def retrieve(self, prompt, **kwargs):
            result = await super().retrieve(prompt, **kwargs)
            if len(self.calls) == 2:
                result[0].source_url = "https://example.com/sdk-changelog"
                result[0].rel_path = result[0].source_url
                result[0].text = "Cline SDK 0.0.83 enabled native web search."
            return result

    model = _ReviewTwice(
        sources=["web"], queries=["Cline SDK changelog"], verify=True
    )
    web, corpus = _NewSourceWeb(), _Corpus()
    with TestClient(create_app(settings)) as client:
        _, events = _turn(client, model, web, corpus, "What changed in Cline SDK?")
    reviewed = _payload(events, "evidence.reviewed")
    assert reviewed["adequate"] is True
    assert reviewed["review_calls"] == 2
    assert reviewed["followup_search_calls"] == 1
    assert _payload(events, "context.retrieved")["web_source_count"] == 2
    assert corpus.calls == []


def test_followup_official_changelog_ends_research_and_drops_aggregator(settings) -> None:
    class _OfficialSecond(_Web):
        async def retrieve(self, prompt, **kwargs):
            result = await super().retrieve(prompt, **kwargs)
            if len(self.calls) == 2:
                result[0].source_url = (
                    "https://github.com/cline/cline/blob/main/sdk/CHANGELOG.md"
                )
                result[0].rel_path = result[0].source_url
                result[0].text = (
                    "## 0.0.86\nLocal recovery.\n"
                    "## 0.0.83\nProvider-native web search enabled."
                )
            return result

    model = _Model(
        sources=["web"],
        queries=["Cline SDK changelog"],
        focus_terms=["Cline", "SDK"],
        verify=True,
        review=EvidenceReviewV1(
            adequate=False,
            followup_queries=["Cline SDK official changelog"],
        ),
    )
    web, corpus = _OfficialSecond(), _Corpus()
    with TestClient(create_app(settings)) as client:
        _, events = _turn(
            client, model, web, corpus,
            "What features did Cline SDK 0.0.83 and 0.0.86 release?",
        )
    reviewed = _payload(events, "evidence.reviewed")
    assert reviewed["adequate"] is True
    assert reviewed["review_calls"] == 1
    assert _payload(events, "context.retrieved")["web_source_count"] == 1
    assert len(web.calls) == 2
    assert corpus.calls == []


def test_web_research_stops_after_two_refinement_rounds(settings) -> None:
    class _PersistentGap(_Model):
        async def _structured(self, schema, **kwargs):
            if schema is EvidenceReviewV1:
                self.structured_calls.append(schema.__name__)
                number = self.structured_calls.count(schema.__name__)
                return EvidenceReviewV1(
                    adequate=False,
                    followup_queries=[f"Cline SDK 0.0.{80 + number} changelog"],
                )
            return await super()._structured(schema, **kwargs)

    class _NovelWeb(_Web):
        async def retrieve(self, prompt, **kwargs):
            result = await super().retrieve(prompt, **kwargs)
            if result:
                number = len(self.calls)
                result[0].source_url = f"https://example.com/release-{number}"
                result[0].rel_path = result[0].source_url
                result[0].text = f"Release {number} describes a separate public feature."
            return result

    model = _PersistentGap(
        sources=["web"], queries=["Cline SDK changelog"], verify=True
    )
    web, corpus = _NovelWeb(), _Corpus()
    with TestClient(create_app(settings)) as client:
        _, events = _turn(client, model, web, corpus, "What changed in Cline SDK?")
    reviewed = _payload(events, "evidence.reviewed")
    assert reviewed["review_calls"] == 2
    assert reviewed["followup_search_calls"] == 2
    assert reviewed["adequate"] is None
    assert len(web.calls) == 3  # Initial search and no more than two follow-ups.
    assert corpus.calls == []


def test_initial_web_search_timeout_still_delivers_reply(settings, monkeypatch) -> None:
    from waqil_api import control_plane

    class _SlowWeb(_Web):
        async def retrieve(self, prompt, **kwargs):
            await asyncio.sleep(0.2)
            return await super().retrieve(prompt, **kwargs)

    monkeypatch.setattr(control_plane, "_WEB_RESEARCH_INITIAL_SECONDS", 0.01)
    model = _Model(
        sources=["web"], queries=["Cline SDK changelog"], verify=True
    )
    web, corpus = _SlowWeb(), _Corpus()
    with TestClient(create_app(settings)) as client:
        _, events = _turn(client, model, web, corpus, "What changed in Cline SDK?")
    error = _payload(events, "context.knowledge_error")
    assert error["category"] == "web_search_failed"
    assert error["error_type"] == "TimeoutError"
    assert _payload(events, "context.retrieved")["web_source_count"] == 0
    assert model.structured_calls.count("EvidenceReviewV1") == 0
    assert corpus.calls == []


def test_web_research_review_timeout_preserves_initial_sources(settings, monkeypatch) -> None:
    from waqil_api import control_plane

    class _SlowReview(_Model):
        async def _structured(self, schema, **kwargs):
            if schema is EvidenceReviewV1:
                await asyncio.sleep(0.2)
            return await super()._structured(schema, **kwargs)

    monkeypatch.setattr(control_plane, "_WEB_RESEARCH_REVIEW_SECONDS", 0.01)
    model = _SlowReview(
        sources=["web"], queries=["Cline SDK changelog"], verify=True
    )
    web, corpus = _Web(), _Corpus()
    with TestClient(create_app(settings)) as client:
        _, events = _turn(client, model, web, corpus, "What changed in Cline SDK?")
    reviewed = _payload(events, "evidence.reviewed")
    assert reviewed["timed_out"] is True
    assert reviewed["review_calls"] == 0
    assert _payload(events, "context.retrieved")["web_source_count"] == 1
    assert len(web.calls) == 1
    assert corpus.calls == []


def test_model_response_event_exposes_only_safe_latency_metrics(settings) -> None:
    class _TimedModel(_Model):
        async def generate(self, request, on_token=None, *, model_aliases=None, on_reasoning=None):
            result = await super().generate(
                request, on_token, model_aliases=model_aliases,
                on_reasoning=on_reasoning,
            )
            result.structured = {
                "provider": "cline",
                "generation_seconds": 1.23456,
                "response_headers_seconds": 0.25,
                "first_event_seconds": 0.5,
                "first_text_seconds": 0.75,
                "first_reasoning_seconds": 0.6,
                "input_characters": 100,
                "reasoning_characters": 120,
                "output_characters": 20,
                "max_completion_tokens": 512,
                "streamed": True,
                "raw_prompt": "must never appear in events",
                "api_key": "must never appear in events",
                "other_seconds": float("inf"),
            }
            return result

    model = _TimedModel(sources=[], model_name="cline-pass/qwen3.7-plus")
    web, corpus = _Web(), _Corpus()
    with TestClient(create_app(settings)) as client:
        _, events = _turn(client, model, web, corpus, "Explain Python lists.")
    response = _payload(events, "model.response")
    assert response["timings"] == {
        "generation_seconds": 1.235,
        "response_headers_seconds": 0.25,
        "first_event_seconds": 0.5,
        "first_text_seconds": 0.75,
        "first_reasoning_seconds": 0.6,
        "input_characters": 100,
        "reasoning_characters": 120,
        "output_characters": 20,
        "max_completion_tokens": 512,
        "streamed": True,
    }
    assert "must never appear in events" not in events


def test_response_event_identifies_clinepass_as_cline(settings) -> None:
    # The actual gateway reply does not include structured provider metadata.
    # Its old event fell back to "local", misleading the activity panel even
    # when the answer came from ClinePass.
    settings.cline_api_key = "synthetic-test-key"
    model = _Model(sources=[], model_name="cline-pass/qwen3.7-plus")
    web, corpus = _Web(), _Corpus()
    with TestClient(create_app(settings)) as client:
        client.app.state.runtime.model_preference.save("split", None, provider="cline")
        _, events = _turn(
            client,
            model,
            web,
            corpus,
            "Explain Python list comprehensions.",
        )
    assert _payload(events, "model.response")["provider"] == "cline"


def test_auto_planner_failure_keeps_public_requirement_without_raw_query(
    settings,
) -> None:
    model = _BrokenPlanner(sources=[])
    web, corpus = _Web(), _Corpus()
    with TestClient(create_app(settings)) as client:
        planned, events = _turn(
            client,
            model,
            web,
            corpus,
            "What new Cline SDK features shipped recently?",
        )
    assert _payload(events, "evidence.plan_failed")["error_type"] == "RuntimeError"
    assert planned["method"] == "planner_error"
    assert planned["sources"] == ["web"]
    assert planned["query_count"] == 0
    assert len(web.calls) == 1
    assert web.calls[0]["queries"] == []
    assert corpus.calls == []
    assert _payload(events, "context.retrieved")["web_source_count"] == 0
    assert "couldn't form a safe public search" in events


@pytest.mark.parametrize(
    "scope,expected_sources,web_calls,private_calls",
    [
        ("web", ["web"], 1, 0),
        ("notion", ["private"], 0, 1),
    ],
)
def test_explicit_source_scope_survives_planner_failure(
    settings, scope, expected_sources, web_calls, private_calls
) -> None:
    model = _BrokenPlanner(sources=[])
    web, corpus = _Web(), _Corpus()
    with TestClient(create_app(settings)) as client:
        planned, events = _turn(
            client,
            model,
            web,
            corpus,
            "Explain Python list comprehensions.",
            scope=scope,
        )
    assert planned["method"] == "explicit_scope"
    assert planned["sources"] == expected_sources
    assert "event: evidence.plan_failed" not in events
    assert len(web.calls) == web_calls
    assert len(corpus.calls) == private_calls
