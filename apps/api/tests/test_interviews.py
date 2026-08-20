"""Five questions, one verdict, and the arithmetic nobody else is allowed to do.

Most of these tests guard two boundaries. The credential boundary: the
ElevenLabs API key mints the token server-side and must never appear in
anything the browser receives. The honesty boundary: the model submits five
criterion scores and Metis computes the overall — so the score spoken in the
debrief is the score stored, a retried submission cannot rewrite a verdict
already heard, and an interview too short to score produces no number at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from waqil_api.config import Settings
from waqil_api.contracts import InterviewEvaluationV1
from waqil_api.interview_delivery import (
    candidate_speaker,
    delivery_metrics,
    spoken_words,
)
from waqil_api.interviews import (
    MIN_SCORED_QUESTIONS,
    NO_FOCUS_AREAS,
    QUESTION_LIMIT,
    SCORE_WEIGHTS,
    STANDARD_OBJECTIVE,
    build_scorecard,
    overall_score,
    recommendation_for,
)
from waqil_api.main import create_app

REPO_ROOT = Path(__file__).resolve().parents[3]

READY = dict(
    elevenlabs_api_key="sk-canary-interviews",
    interview_elevenlabs_agent_id="agent_interview_1",
)

CONTEXT = {
    "job_title": "Senior Data Engineer",
    "company_name": "Batelco",
    "job_description": "Own the lakehouse. Requires Spark, Airflow, and judgment.",
    "interview_type": "technical",
}


def _settings(tmp_path: Path, **overrides) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        repo_root=REPO_ROOT,
        model_backend="deterministic",
        reference_runner_mode="deterministic",
        allow_test_backends=True,
        **overrides,
    )


class FakeResponse:
    def __init__(self, *, status_code: int = 200, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        if self._payload is None:
            raise ValueError("not JSON")
        return self._payload


class FakeTokenClient:
    """Answers the token mint and records exactly what was asked."""

    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.sent: dict = {}

    async def get(self, url: str, **kwargs) -> FakeResponse:
        self.sent = {"url": url, **kwargs}
        return self.response


class FakeSpeech:
    available = True

    def __init__(self, client: FakeTokenClient) -> None:
        self._client_instance = client

    async def _client(self) -> FakeTokenClient:
        return self._client_instance


class FakeRouter:
    def __init__(self, speech: FakeSpeech) -> None:
        self.elevenlabs = speech


def _wire_token(app, token: str = "conv-token-1") -> FakeTokenClient:
    """Point the interview service at a fake ElevenLabs that returns a token."""
    client = FakeTokenClient(FakeResponse(payload={"token": token}))
    app.state.runtime.interviews.model = FakeRouter(FakeSpeech(client))
    return client


def _evaluation(**overrides) -> dict:
    body = {
        "specific_evidence": 8,
        "role_depth": 7,
        "relevance": 7,
        "structure": 6,
        "communication": 7,
        "verdict": "Solid on evidence, thin on tradeoffs.",
        "strongest_answer_quote": "We cut the nightly run from six hours to forty minutes.",
        "strongest_answer_reason": "A number, a before, and an after.",
        "improvements": [
            {
                "what_happened": "Hid behind 'we' on the migration story.",
                "evidence": "\"We decided to move to Airflow\" — no personal contribution named.",
                "why_it_hurt": "The interviewer cannot credit work they cannot attribute.",
                "better_approach": "Name the part that was yours before describing the team's.",
            },
            {
                "what_happened": "Answered the scenario with generalities.",
                "evidence": "\"I would look at the data and figure it out.\"",
                "why_it_hurt": "The question tested judgment; nothing specific was judged.",
                "better_approach": "Commit to one concrete first step and defend it.",
            },
            {
                "what_happened": "Ran long on question two.",
                "evidence": "The answer continued for three topics past the question.",
                "why_it_hurt": "The point about ownership was buried.",
                "better_approach": "Land the point, then stop.",
            },
        ],
        "drill": "Rewrite the migration story with 'I' as the subject of every sentence. Ten minutes.",
        "completed_question_count": 5,
        "incomplete": False,
    }
    body.update(overrides)
    return body


def _ev(**overrides) -> InterviewEvaluationV1:
    return InterviewEvaluationV1.model_validate(_evaluation(**overrides))


# -- the score formula, deterministically -----------------------------------


def test_the_weights_are_the_specified_percentages() -> None:
    assert SCORE_WEIGHTS == {
        "specific_evidence": 25,
        "role_depth": 25,
        "relevance": 20,
        "structure": 15,
        "communication": 15,
    }
    assert sum(SCORE_WEIGHTS.values()) == 100


def test_the_formula_weights_evidence_more_than_polish() -> None:
    evidence_heavy = _ev(
        specific_evidence=10, role_depth=6, relevance=6, structure=6, communication=6
    )
    structure_heavy = _ev(
        specific_evidence=6, role_depth=6, relevance=6, structure=10, communication=6
    )
    assert overall_score(evidence_heavy) == 7.0
    assert overall_score(structure_heavy) == 6.6


def test_recommendation_thresholds_at_their_exact_boundaries() -> None:
    # 8,8,7,7,7 → exactly 7.50: the lowest score that advances.
    assert overall_score(
        _ev(specific_evidence=8, role_depth=8, relevance=7, structure=7, communication=7)
    ) == 7.5
    assert recommendation_for(7.5) == "advance"
    # 8,7,7,8,7 → exactly 7.40: the top of borderline.
    assert overall_score(
        _ev(specific_evidence=8, role_depth=7, relevance=7, structure=8, communication=7)
    ) == 7.4
    assert recommendation_for(7.4) == "borderline"
    assert recommendation_for(6.0) == "borderline"
    # 6,6,7,5,5 → exactly 5.90: below borderline.
    assert overall_score(
        _ev(specific_evidence=6, role_depth=6, relevance=7, structure=5, communication=5)
    ) == 5.9
    assert recommendation_for(5.9) == "do_not_advance"
    assert overall_score(_ev(specific_evidence=10, role_depth=10, relevance=10, structure=10, communication=10)) == 10.0
    assert overall_score(_ev(specific_evidence=1, role_depth=1, relevance=1, structure=1, communication=1)) == 1.0


def test_half_point_boundaries_round_half_up() -> None:
    # 7,7,8,7,8 → 7.35 exactly; half-up gives 7.4, never 7.3.
    assert overall_score(
        _ev(specific_evidence=7, role_depth=7, relevance=8, structure=7, communication=8)
    ) == 7.4


def test_a_short_interview_gets_no_number_and_three_answers_get_a_provisional_one() -> None:
    unscorable = build_scorecard(_ev(incomplete=True, completed_question_count=2))
    assert unscorable.overall_score is None
    assert unscorable.recommendation is None
    assert unscorable.provisional is True

    provisional = build_scorecard(
        _ev(incomplete=True, completed_question_count=MIN_SCORED_QUESTIONS)
    )
    assert provisional.overall_score == 7.1
    assert provisional.recommendation == "borderline"
    assert provisional.provisional is True

    complete = build_scorecard(_ev())
    assert complete.provisional is False
    assert complete.overall_score == 7.1


# -- starting a session ------------------------------------------------------


def test_every_context_field_is_required_to_start(tmp_path) -> None:
    app = create_app(_settings(tmp_path, **READY))
    with TestClient(app) as client:
        _wire_token(app)
        for missing in ("job_title", "company_name", "job_description", "interview_type"):
            body = {key: value for key, value in CONTEXT.items() if key != missing}
            response = client.post("/api/v1/interviews/sessions", json=body)
            assert response.status_code == 422, missing


def test_the_interview_type_is_restricted_to_the_three_rounds(tmp_path) -> None:
    app = create_app(_settings(tmp_path, **READY))
    with TestClient(app) as client:
        _wire_token(app)
        for bad in ("panel", "executive", "", "HR"):
            response = client.post(
                "/api/v1/interviews/sessions", json={**CONTEXT, "interview_type": bad}
            )
            assert response.status_code == 422, bad
        for good in ("hr_recruiter", "hiring_manager", "technical"):
            response = client.post(
                "/api/v1/interviews/sessions", json={**CONTEXT, "interview_type": good}
            )
            assert response.status_code == 201, good


def test_the_browser_cannot_set_the_question_limit(tmp_path) -> None:
    # extra="forbid" makes the attempt a 422 rather than a silent ignore.
    app = create_app(_settings(tmp_path, **READY))
    with TestClient(app) as client:
        _wire_token(app)
        response = client.post(
            "/api/v1/interviews/sessions", json={**CONTEXT, "question_limit": 1}
        )
        assert response.status_code == 422


def test_start_passes_every_dynamic_variable_and_fixes_the_limit_at_five(tmp_path) -> None:
    app = create_app(_settings(tmp_path, **READY))
    with TestClient(app) as client:
        minted = _wire_token(app)
        response = client.post("/api/v1/interviews/sessions", json=CONTEXT)
        assert response.status_code == 201
        body = response.json()
        variables = body["dynamic_variables"]
        assert set(variables) == {
            "job_title",
            "company_name",
            "job_description",
            "interview_type",
            "question_limit",
            "metis_interview_session_id",
            "interview_objective",
            "focus_areas",
        }
        assert variables["question_limit"] == str(QUESTION_LIMIT) == "5"
        assert variables["interview_type"] == "technical"
        assert variables["metis_interview_session_id"] == body["session"]["id"]
        # Every referenced variable must be supplied every time, so an unset
        # objective travels as words, never as an absent key.
        assert variables["interview_objective"] == STANDARD_OBJECTIVE
        assert variables["focus_areas"] == NO_FOCUS_AREAS
        assert body["conversation_token"] == "conv-token-1"
        assert body["session"]["question_limit"] == 5
        assert body["session"]["status"] == "active"
        # The mint asked for the interview agent, not the voice one.
        assert minted.sent["url"] == "/v1/convai/conversation/token"
        assert minted.sent["params"]["agent_id"] == "agent_interview_1"


def test_the_objective_and_focus_areas_travel_to_the_agent_and_persist(tmp_path) -> None:
    app = create_app(_settings(tmp_path, **READY))
    with TestClient(app) as client:
        _wire_token(app)
        response = client.post(
            "/api/v1/interviews/sessions",
            json={
                **CONTEXT,
                "interview_objective": "Grill me on why ElevenLabs over a custom stack.",
                "focus_areas": ["architecture trade-offs", "  why ElevenLabs  ", ""],
            },
        )
        assert response.status_code == 201
        body = response.json()
        variables = body["dynamic_variables"]
        assert variables["interview_objective"] == (
            "Grill me on why ElevenLabs over a custom stack."
        )
        # Whitespace items are dropped, the rest join into one spoken string.
        assert variables["focus_areas"] == "architecture trade-offs; why ElevenLabs"
        session = body["session"]
        assert session["interview_objective"] == (
            "Grill me on why ElevenLabs over a custom stack."
        )
        assert session["focus_areas"] == ["architecture trade-offs", "why ElevenLabs"]
        fetched = client.get(f"/api/v1/interviews/sessions/{session['id']}").json()
        assert fetched["focus_areas"] == ["architecture trade-offs", "why ElevenLabs"]


def test_focus_areas_are_short_phrases_and_few(tmp_path) -> None:
    app = create_app(_settings(tmp_path, **READY))
    with TestClient(app) as client:
        _wire_token(app)
        too_long = client.post(
            "/api/v1/interviews/sessions",
            json={**CONTEXT, "focus_areas": ["x" * 121]},
        )
        assert too_long.status_code == 422
        too_many = client.post(
            "/api/v1/interviews/sessions",
            json={**CONTEXT, "focus_areas": [f"area {n}" for n in range(7)]},
        )
        assert too_many.status_code == 422


def test_the_api_key_never_reaches_the_browser(tmp_path) -> None:
    app = create_app(_settings(tmp_path, **READY))
    with TestClient(app) as client:
        _wire_token(app)
        availability = client.get("/api/v1/interviews/availability")
        assert READY["elevenlabs_api_key"] not in availability.text
        started = client.post("/api/v1/interviews/sessions", json=CONTEXT)
        assert READY["elevenlabs_api_key"] not in started.text
        fetched = client.get(
            f"/api/v1/interviews/sessions/{started.json()['session']['id']}"
        )
        assert READY["elevenlabs_api_key"] not in fetched.text


def test_availability_names_each_missing_piece_without_leaking_secrets(tmp_path) -> None:
    app = create_app(_settings(tmp_path, elevenlabs_api_key="sk-canary-0003"))
    with TestClient(app) as client:
        # A key is configured (the fake speech provider is live); only the
        # agent id is missing, and the reason must name exactly that.
        _wire_token(app)
        body = client.get("/api/v1/interviews/availability").json()
        assert body["available"] is False
        assert "WAQIL_INTERVIEW_ELEVENLABS_AGENT_ID" in body["reason"]
        assert "sk-canary-0003" not in json.dumps(body)

    bare = create_app(_settings(tmp_path / "bare"))
    with TestClient(bare) as client:
        body = client.get("/api/v1/interviews/availability").json()
        assert body["available"] is False
        assert len(body["missing"]) == 2
        assert any("WAQIL_ELEVENLABS_API_KEY" in item for item in body["missing"])
        assert any(
            "WAQIL_INTERVIEW_ELEVENLABS_AGENT_ID" in item for item in body["missing"]
        )


def test_an_unconfigured_agent_refuses_the_start_with_the_variable_name(tmp_path) -> None:
    app = create_app(_settings(tmp_path, elevenlabs_api_key="sk-x"))
    with TestClient(app) as client:
        _wire_token(app)
        response = client.post("/api/v1/interviews/sessions", json=CONTEXT)
        assert response.status_code == 503
        assert "WAQIL_INTERVIEW_ELEVENLABS_AGENT_ID" in response.json()["detail"]


def test_a_refused_mint_becomes_a_502_and_a_failed_session(tmp_path) -> None:
    app = create_app(_settings(tmp_path, **READY))
    with TestClient(app) as client:
        service = app.state.runtime.interviews
        service.model = FakeRouter(FakeSpeech(FakeTokenClient(FakeResponse(status_code=401))))
        response = client.post("/api/v1/interviews/sessions", json=CONTEXT)
        assert response.status_code == 502
        assert "ElevenLabs" in response.json()["detail"]


# -- the transcript ----------------------------------------------------------


def _started(client, app) -> str:
    _wire_token(app)
    response = client.post("/api/v1/interviews/sessions", json=CONTEXT)
    assert response.status_code == 201
    return response.json()["session"]["id"]


def test_turns_keep_their_order_and_roles_and_absorb_resends(tmp_path) -> None:
    app = create_app(_settings(tmp_path, **READY))
    with TestClient(app) as client:
        session_id = _started(client, app)
        first = client.post(
            f"/api/v1/interviews/sessions/{session_id}/turns",
            json={
                "turns": [
                    {"ordinal": 1, "role": "user", "text": "I led the migration."},
                    {"ordinal": 0, "role": "agent", "text": "Question one. Walk me through it."},
                ]
            },
        )
        assert first.status_code == 200
        assert first.json()["stored"] == 2
        second = client.post(
            f"/api/v1/interviews/sessions/{session_id}/turns",
            json={
                "turns": [
                    # A re-send of ordinal 1 plus one new line.
                    {"ordinal": 1, "role": "user", "text": "I led the migration."},
                    {"ordinal": 2, "role": "agent", "text": "Question two. Your part, specifically?"},
                ]
            },
        )
        assert second.json()["stored"] == 3
        session = client.get(f"/api/v1/interviews/sessions/{session_id}").json()
        assert [turn["ordinal"] for turn in session["turns"]] == [0, 1, 2]
        assert [turn["role"] for turn in session["turns"]] == ["agent", "user", "agent"]
        assert session["turns"][2]["text"].startswith("Question two.")


def test_turns_for_a_missing_session_are_refused(tmp_path) -> None:
    app = create_app(_settings(tmp_path, **READY))
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/interviews/sessions/ivw_missing/turns",
            json={"turns": [{"ordinal": 0, "role": "agent", "text": "Question one."}]},
        )
        assert response.status_code == 404


def test_the_provider_conversation_id_attaches_once_reported(tmp_path) -> None:
    app = create_app(_settings(tmp_path, **READY))
    with TestClient(app) as client:
        session_id = _started(client, app)
        response = client.patch(
            f"/api/v1/interviews/sessions/{session_id}",
            json={"provider_conversation_id": "conv_abc123"},
        )
        assert response.status_code == 200
        assert response.json()["provider_conversation_id"] == "conv_abc123"


# -- the evaluation ----------------------------------------------------------


def test_the_backend_calculates_the_score_the_model_never_submits_one(tmp_path) -> None:
    app = create_app(_settings(tmp_path, **READY))
    with TestClient(app) as client:
        session_id = _started(client, app)
        # The submission carries no overall score field at all; sending one is refused.
        refused = client.post(
            f"/api/v1/interviews/sessions/{session_id}/evaluation",
            json={**_evaluation(), "overall_score": 9.9},
        )
        assert refused.status_code == 422
        response = client.post(
            f"/api/v1/interviews/sessions/{session_id}/evaluation",
            json=_evaluation(),
        )
        assert response.status_code == 200
        scorecard = response.json()
        assert scorecard["overall_score"] == 7.1
        assert scorecard["recommendation"] == "borderline"
        assert scorecard["provisional"] is False
        session = client.get(f"/api/v1/interviews/sessions/{session_id}").json()
        assert session["status"] == "complete"
        assert session["scorecard"]["overall_score"] == 7.1


def test_malformed_and_out_of_range_evaluations_are_rejected(tmp_path) -> None:
    app = create_app(_settings(tmp_path, **READY))
    with TestClient(app) as client:
        session_id = _started(client, app)
        rejected = [
            _evaluation(specific_evidence=0),
            _evaluation(communication=11),
            _evaluation(verdict=""),
            _evaluation(improvements=_evaluation()["improvements"][:2]),
            _evaluation(improvements=_evaluation()["improvements"] * 2),
            _evaluation(completed_question_count=6),
            # Flags that disagree with each other.
            _evaluation(incomplete=False, completed_question_count=3),
            _evaluation(incomplete=True, completed_question_count=5),
        ]
        for body in rejected:
            response = client.post(
                f"/api/v1/interviews/sessions/{session_id}/evaluation", json=body
            )
            assert response.status_code == 422, body
        # Nothing was stored by any of the refusals.
        session = client.get(f"/api/v1/interviews/sessions/{session_id}").json()
        assert session["scorecard"] is None
        assert session["status"] == "active"


def test_a_retried_submission_returns_the_verdict_already_heard(tmp_path) -> None:
    app = create_app(_settings(tmp_path, **READY))
    with TestClient(app) as client:
        session_id = _started(client, app)
        first = client.post(
            f"/api/v1/interviews/sessions/{session_id}/evaluation", json=_evaluation()
        ).json()
        second = client.post(
            f"/api/v1/interviews/sessions/{session_id}/evaluation",
            json=_evaluation(specific_evidence=10, role_depth=10, relevance=10,
                             structure=10, communication=10),
        ).json()
        assert second["overall_score"] == first["overall_score"] == 7.1
        assert second["evaluation"]["specific_evidence"] == 8


def test_an_early_exit_with_three_answers_is_provisional_and_marked_ended_early(tmp_path) -> None:
    app = create_app(_settings(tmp_path, **READY))
    with TestClient(app) as client:
        session_id = _started(client, app)
        response = client.post(
            f"/api/v1/interviews/sessions/{session_id}/evaluation",
            json=_evaluation(incomplete=True, completed_question_count=3),
        )
        scorecard = response.json()
        assert scorecard["provisional"] is True
        assert scorecard["overall_score"] == 7.1
        session = client.get(f"/api/v1/interviews/sessions/{session_id}").json()
        assert session["status"] == "ended_early"


def test_two_answers_store_the_evaluation_but_refuse_to_score_it(tmp_path) -> None:
    app = create_app(_settings(tmp_path, **READY))
    with TestClient(app) as client:
        session_id = _started(client, app)
        scorecard = client.post(
            f"/api/v1/interviews/sessions/{session_id}/evaluation",
            json=_evaluation(incomplete=True, completed_question_count=2),
        ).json()
        assert scorecard["overall_score"] is None
        assert scorecard["recommendation"] is None
        assert scorecard["provisional"] is True


# -- teardown ----------------------------------------------------------------


def test_ending_a_session_is_idempotent_and_keeps_the_first_reason(tmp_path) -> None:
    app = create_app(_settings(tmp_path, **READY))
    with TestClient(app) as client:
        session_id = _started(client, app)
        first = client.post(
            f"/api/v1/interviews/sessions/{session_id}/end",
            json={"reason": "ended_early"},
        )
        assert first.status_code == 200
        assert first.json()["status"] == "ended_early"
        # pagehide, unmount and an explicit stop may all fire; none conflicts.
        second = client.post(
            f"/api/v1/interviews/sessions/{session_id}/end",
            json={"reason": "failed"},
        )
        assert second.status_code == 200
        assert second.json()["status"] == "ended_early"


def test_the_evaluation_still_lands_after_a_teardown_end(tmp_path) -> None:
    # The tool call races the disconnect. The transcript is already stored and
    # the session row still exists, so the verdict must not be lost to the race.
    app = create_app(_settings(tmp_path, **READY))
    with TestClient(app) as client:
        session_id = _started(client, app)
        client.post(
            f"/api/v1/interviews/sessions/{session_id}/end",
            json={"reason": "ended_early"},
        )
        response = client.post(
            f"/api/v1/interviews/sessions/{session_id}/evaluation",
            json=_evaluation(incomplete=True, completed_question_count=4),
        )
        assert response.status_code == 200
        assert response.json()["provisional"] is True


def test_deleting_a_session_is_idempotent(tmp_path) -> None:
    app = create_app(_settings(tmp_path, **READY))
    with TestClient(app) as client:
        session_id = _started(client, app)
        assert client.delete(f"/api/v1/interviews/sessions/{session_id}").status_code == 204
        assert client.get(f"/api/v1/interviews/sessions/{session_id}").status_code == 404
        # Already gone is success, not an error.
        assert client.delete(f"/api/v1/interviews/sessions/{session_id}").status_code == 204


# -- delivery analysis --------------------------------------------------------


def _word(text: str, start: float, end: float, speaker: str) -> dict:
    return {"text": text, "start": start, "end": end, "speaker_id": speaker, "type": "word"}


def _scribe_payload() -> dict:
    # Chiron asks; the candidate answers with two "um"s, one hedge, and one
    # 2.7-second silence in the middle of their own answer.
    agent = [
        _word("Question", 0.0, 0.4, "speaker_0"),
        _word("one.", 0.5, 0.8, "speaker_0"),
        _word("Walk", 0.9, 1.1, "speaker_0"),
        _word("me", 1.2, 1.3, "speaker_0"),
        _word("through", 1.4, 1.7, "speaker_0"),
        _word("it.", 1.8, 2.0, "speaker_0"),
    ]
    candidate = [
        _word("Um,", 3.0, 3.2, "speaker_1"),
        _word("I", 3.3, 3.4, "speaker_1"),
        _word("think", 3.5, 3.8, "speaker_1"),
        _word("we", 3.9, 4.0, "speaker_1"),
        _word("cut", 4.1, 4.3, "speaker_1"),
        _word("the", 4.4, 4.5, "speaker_1"),
        _word("nightly", 4.6, 5.0, "speaker_1"),
        _word("run", 5.1, 5.3, "speaker_1"),
        _word("um", 8.0, 8.2, "speaker_1"),
        _word("to", 8.3, 8.4, "speaker_1"),
        _word("forty", 8.5, 8.8, "speaker_1"),
        _word("minutes", 8.9, 9.4, "speaker_1"),
    ]
    spacing = [{"text": " ", "type": "spacing"}]
    return {"words": agent + spacing + candidate, "text": "…"}


def test_delivery_metrics_count_fillers_hedges_and_real_pauses() -> None:
    words = spoken_words(_scribe_payload())
    speaker = candidate_speaker(
        words,
        agent_texts=["Question one. Walk me through it."],
        user_texts=["Um, I think we cut the nightly run to forty minutes."],
    )
    assert speaker == "speaker_1"
    metrics = delivery_metrics(words, speaker)
    assert metrics["candidate_word_count"] == 12
    assert metrics["filler_breakdown"] == {"um": 2}
    assert metrics["filler_count"] == 2
    assert metrics["hedging_breakdown"] == {"i think": 1}
    # The 2.7s silence sits inside the candidate's own answer — a pause. The
    # gap while Chiron asked the question is not one.
    assert metrics["long_pause_count"] == 1
    assert metrics["longest_pause_seconds"] == 2.7
    # Talk time is the two speech runs: 3.0→5.3 and 8.0→9.4 = 3.7 seconds.
    assert metrics["candidate_talk_seconds"] == 3.7
    assert metrics["words_per_minute"] == pytest.approx(194.6, abs=0.1)
    assert metrics["filler_rate_per_100_words"] == pytest.approx(16.7, abs=0.1)


def test_a_silence_while_the_interviewer_speaks_is_not_a_pause() -> None:
    words = [
        _word("So", 0.0, 0.2, "cand"),
        _word("yes.", 0.3, 0.5, "cand"),
        # Chiron talks from 1.0 to 5.5; the candidate resumes at 6.0.
        _word("And", 1.0, 1.2, "agent"),
        _word("why?", 5.0, 5.5, "agent"),
        _word("Because", 6.0, 6.4, "cand"),
        _word("it", 6.5, 6.6, "cand"),
        _word("worked.", 6.7, 7.0, "cand"),
    ]
    metrics = delivery_metrics(words, "cand")
    assert metrics["long_pause_count"] == 0
    assert metrics["longest_pause_seconds"] == 0.0


class FakeDeliveryClient:
    """Serves the conversation lookup, the recording, and records the calls."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def get(self, url: str, **kwargs) -> FakeResponse:
        self.calls.append(url)
        if url.endswith("/audio"):
            response = FakeResponse(payload=None)
            response.headers = {"content-type": "audio/mpeg"}
            response.content = b"mp3-bytes"
            return response
        if "/v1/convai/conversations/" in url:
            return FakeResponse(payload={"status": "done", "has_audio": True})
        return FakeResponse(payload={"token": "conv-token-1"})


class FakeDeliverySpeech:
    available = True

    def __init__(self, client) -> None:
        self._client_instance = client
        self.transcribed: dict = {}

    async def _client(self):
        return self._client_instance

    async def transcribe_meeting(self, audio: bytes, filename: str, media_type: str) -> dict:
        self.transcribed = {
            "audio": audio, "filename": filename, "media_type": media_type
        }
        return _scribe_payload()


def _wire_delivery(app) -> FakeDeliverySpeech:
    speech = FakeDeliverySpeech(FakeDeliveryClient())
    app.state.runtime.interviews.model = FakeRouter(speech)
    app.state.runtime.interviews.delivery_poll_seconds = 0.0
    return speech


def test_delivery_analysis_measures_the_recording_and_stores_the_metrics(tmp_path) -> None:
    app = create_app(_settings(tmp_path, **READY))
    with TestClient(app) as client:
        session_id = _started(client, app)
        client.post(
            f"/api/v1/interviews/sessions/{session_id}/turns",
            json={
                "turns": [
                    {"ordinal": 0, "role": "agent", "text": "Question one. Walk me through it."},
                    {"ordinal": 1, "role": "user", "text": "Um, I think we cut the nightly run to forty minutes."},
                ]
            },
        )
        client.patch(
            f"/api/v1/interviews/sessions/{session_id}",
            json={"provider_conversation_id": "conv_abc123"},
        )
        speech = _wire_delivery(app)
        client.post(
            f"/api/v1/interviews/sessions/{session_id}/end",
            json={"reason": "ended_early"},
        )
        response = client.post(f"/api/v1/interviews/sessions/{session_id}/delivery")
        assert response.status_code == 200
        session = response.json()
        assert session["delivery_stage"] == "ready"
        delivery = session["delivery"]
        assert delivery["filler_count"] == 2
        assert delivery["filler_breakdown"] == {"um": 2}
        assert delivery["hedging_count"] == 1
        assert delivery["long_pause_count"] == 1
        assert delivery["note"] == ""
        # The recording went to Scribe as the bytes ElevenLabs returned.
        assert speech.transcribed["audio"] == b"mp3-bytes"
        assert speech.transcribed["filename"] == "interview.mp3"
        # The scorecard was never touched by the analysis.
        assert session["scorecard"] is None


def test_delivery_without_a_conversation_or_recording_is_unavailable_not_failed(tmp_path) -> None:
    app = create_app(_settings(tmp_path, **READY))
    with TestClient(app) as client:
        # No conversation id was ever attached: nothing to fetch, ever.
        session_id = _started(client, app)
        _wire_delivery(app)
        client.post(
            f"/api/v1/interviews/sessions/{session_id}/end",
            json={"reason": "ended_early"},
        )
        response = client.post(f"/api/v1/interviews/sessions/{session_id}/delivery")
        assert response.status_code == 200
        assert response.json()["delivery_stage"] == "unavailable"
        assert response.json()["delivery"] is None

        # A finalized conversation with recording disabled: also unavailable.
        second_id = _started(client, app)
        client.patch(
            f"/api/v1/interviews/sessions/{second_id}",
            json={"provider_conversation_id": "conv_noaudio"},
        )
        speech = _wire_delivery(app)

        async def no_audio(url: str, **kwargs) -> FakeResponse:
            return FakeResponse(payload={"status": "done", "has_audio": False})

        speech._client_instance.get = no_audio
        client.post(
            f"/api/v1/interviews/sessions/{second_id}/end",
            json={"reason": "ended_early"},
        )
        response = client.post(f"/api/v1/interviews/sessions/{second_id}/delivery")
        assert response.json()["delivery_stage"] == "unavailable"


def test_delivery_waits_for_the_call_to_end(tmp_path) -> None:
    # A premature poke must not analyze — the call is live, there is no
    # recording — and must not stamp any stage that would suppress the
    # automatic post-call run.
    app = create_app(_settings(tmp_path, **READY))
    with TestClient(app) as client:
        session_id = _started(client, app)
        client.patch(
            f"/api/v1/interviews/sessions/{session_id}",
            json={"provider_conversation_id": "conv_abc123"},
        )
        speech = _wire_delivery(app)
        response = client.post(f"/api/v1/interviews/sessions/{session_id}/delivery")
        assert response.status_code == 200
        assert response.json()["status"] == "active"
        assert response.json()["delivery_stage"] == ""
        assert speech.transcribed == {}


def test_a_recording_not_ready_yet_is_failed_and_retryable_never_unavailable(tmp_path) -> None:
    # 'unavailable' is terminal and nothing retries it, so it is reserved for
    # a conversation finalized without audio. A recording still processing
    # when the poll budget runs out lands 'failed' — and a later retry works.
    app = create_app(_settings(tmp_path, **READY))
    with TestClient(app) as client:
        session_id = _started(client, app)
        client.patch(
            f"/api/v1/interviews/sessions/{session_id}",
            json={"provider_conversation_id": "conv_slow"},
        )
        speech = _wire_delivery(app)
        app.state.runtime.interviews.delivery_poll_attempts = 2

        async def still_processing(url: str, **kwargs) -> FakeResponse:
            return FakeResponse(payload={"status": "processing", "has_audio": False})

        speech._client_instance.get = still_processing
        client.post(
            f"/api/v1/interviews/sessions/{session_id}/end",
            json={"reason": "ended_early"},
        )
        response = client.post(f"/api/v1/interviews/sessions/{session_id}/delivery")
        assert response.json()["delivery_stage"] == "failed"

        # The recording finalizes; the retry measures it.
        fresh = _wire_delivery(app)
        app.state.runtime.interviews.delivery_poll_attempts = 2
        retried = client.post(f"/api/v1/interviews/sessions/{session_id}/delivery")
        assert retried.json()["delivery_stage"] == "ready"
        assert fresh.transcribed["audio"] == b"mp3-bytes"


def test_the_conversation_id_alphabet_is_pinned(tmp_path) -> None:
    # The id is spliced into provider URL paths later; an id that could
    # carry a slash or query never gets stored.
    app = create_app(_settings(tmp_path, **READY))
    with TestClient(app) as client:
        session_id = _started(client, app)
        for bad in ("conv/../evil", "conv?x=1", "conv abc", "conv#f"):
            response = client.patch(
                f"/api/v1/interviews/sessions/{session_id}",
                json={"provider_conversation_id": bad},
            )
            assert response.status_code == 422, bad


def test_words_without_timings_count_but_never_invent_pauses() -> None:
    # Scribe's timings are nullable: an untimed word still counts as a word,
    # but coercing its null start to 0.0 would open a phantom hours-long
    # pause. It must stay out of the run/pause accounting entirely.
    words = [
        {"text": "Um,", "start": None, "end": None, "speaker_id": "cand", "type": "word"},
        _word("I", 100.0, 100.1, "cand"),
        _word("shipped", 100.2, 100.6, "cand"),
        _word("it.", 100.7, 101.0, "cand"),
    ]
    metrics = delivery_metrics(words, "cand")
    assert metrics["candidate_word_count"] == 4
    assert metrics["filler_breakdown"] == {"um": 1}
    assert metrics["long_pause_count"] == 0
    assert metrics["longest_pause_seconds"] == 0.0
    assert metrics["candidate_talk_seconds"] == 1.0


def test_delivery_for_a_missing_session_is_a_404(tmp_path) -> None:
    app = create_app(_settings(tmp_path, **READY))
    with TestClient(app) as client:
        _wire_delivery(app)
        response = client.post("/api/v1/interviews/sessions/ivw_missing/delivery")
        assert response.status_code == 404


# -- the workflow specification ---------------------------------------------


def _workflow() -> dict:
    path = REPO_ROOT / "docs" / "interviews-workflow.json"
    return json.loads(path.read_text())


def test_every_route_has_exactly_five_question_nodes() -> None:
    spec = _workflow()
    questions = [node for node in spec["nodes"] if node["kind"] == "question"]
    assert len(questions) == 15
    for route, node_ids in spec["routes"].items():
        assert len(node_ids) == 5, route
        on_route = [node for node in questions if node["route"] == route]
        assert sorted(node["ordinal"] for node in on_route) == [1, 2, 3, 4, 5], route
        assert [node["id"] for node in sorted(on_route, key=lambda n: n["ordinal"])] == node_ids


def test_the_graph_walks_each_route_to_the_shared_evaluation() -> None:
    spec = _workflow()
    edges = {(edge["from"], edge["to"]) for edge in spec["edges"]}
    for route, node_ids in spec["routes"].items():
        assert ("router", node_ids[0]) in edges, route
        for earlier, later in zip(node_ids, node_ids[1:]):
            assert (earlier, later) in edges, (route, earlier)
        assert (node_ids[-1], "evaluation") in edges, route
        # Stopping early is reachable from every question except the last,
        # where the interview is over anyway.
        for node_id in node_ids[:-1]:
            assert (node_id, "evaluation") in edges, (route, node_id)
    assert ("evaluation", "end") in edges


def test_the_router_dispatches_on_the_dynamic_variable_not_a_spoken_answer() -> None:
    spec = _workflow()
    assert spec["entry"] == "router"
    assert set(spec["routes"]) == {"hr_recruiter", "hiring_manager", "technical"}
    assert "question_limit" in spec["dynamic_variables"]
    assert "metis_interview_session_id" in spec["dynamic_variables"]
    router_edges = [edge for edge in spec["edges"] if edge["from"] == "router"]
    assert len(router_edges) == 3
    for edge in router_edges:
        assert "{{interview_type}}" in edge["condition"]


def test_every_question_node_announces_its_number() -> None:
    # "Question N" opens every spoken question; the page's progress counter
    # and the five-question guarantee both hang off that phrasing.
    spec = _workflow()
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    for node in spec["nodes"]:
        if node["kind"] != "question":
            continue
        assert f"Question {words[node['ordinal']]}" in node["prompt"], node["id"]


def test_the_spec_supplies_the_objective_and_bounds_the_follow_ups() -> None:
    spec = _workflow()
    # Both steering variables are declared, so the API must supply them —
    # a referenced-but-missing variable kills the conversation at start.
    assert "interview_objective" in spec["dynamic_variables"]
    assert "focus_areas" in spec["dynamic_variables"]
    # Every question node allows probes but caps them at two, and never lets
    # a probe carry a question number — the counter depends on that.
    for node in spec["nodes"]:
        if node["kind"] != "question":
            continue
        assert "unnumbered follow-up probes" in node["prompt"], node["id"]
        assert "at most two" in node["prompt"] or "two short" in node["prompt"], node["id"]
    # Advance edges wait for the probes, so a probe can never be answered by
    # the next node's question.
    for edge in spec["edges"]:
        if edge["from"] == "router" or edge["to"] == "end":
            continue
        if "want to stop" in edge["condition"]:
            continue
        assert "follow-up probes" in edge["condition"], (edge["from"], edge["to"])
