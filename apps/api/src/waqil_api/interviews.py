"""Five spoken questions, an honest verdict, and the arithmetic held server-side.

The interview agent (Chiron) runs entirely on ElevenLabs' hosted model — no
Custom LLM, no tunnel, no lease. Metis' whole job here is three things: mint
the one-conversation WebRTC token so the API key never reaches the browser,
persist the transcript the browser heard, and do the scoring arithmetic. The
model submits five criterion scores through a blocking client tool and Metis
computes the overall score and recommendation, so the number Chiron speaks in
the debrief is always the number stored in the scorecard — a model that did
its own arithmetic would sometimes speak a different one.
"""

from __future__ import annotations

from typing import Any, Literal

from .config import Settings
from .contracts import (
    InterviewContextV1,
    InterviewEvaluationV1,
    InterviewScorecardV1,
    InterviewSessionStartV1,
    InterviewSessionV1,
    InterviewTurnV1,
    utc_now,
)
from .model_provider import ModelProviderError

# Every interview is exactly five questions. Fixed here rather than accepted
# from the browser, because a client that could set the limit could also ask
# for a one-question interview and still receive a full-weight verdict.
QUESTION_LIMIT = 5

# The fewest answered questions an early exit can still be scored on. Below
# this the evaluation stores, but no number is computed: two answers are not
# an interview, and an honest system says so instead of extrapolating.
MIN_SCORED_QUESTIONS = 3

# How much each criterion moves the overall score, in percent. Integers on
# purpose: the arithmetic below stays exact, so a boundary like 7.45 rounds
# the same way on every machine. Evidence and role depth carry half the
# weight between them because they are what an interviewer actually hires on;
# delivery polish is real but secondary.
SCORE_WEIGHTS: dict[str, int] = {
    "specific_evidence": 25,
    "role_depth": 25,
    "relevance": 20,
    "structure": 15,
    "communication": 15,
}

# Recommendation thresholds over the rounded overall score.
ADVANCE_THRESHOLD = 7.5
BORDERLINE_THRESHOLD = 6.0

Recommendation = Literal["advance", "borderline", "do_not_advance"]


class InterviewError(RuntimeError):
    """The message is user-facing."""


# Computes the weighted overall score from a submitted evaluation, to one decimal.
def overall_score(evaluation: InterviewEvaluationV1) -> float:
    # Integer hundredths of a point, then half-up to tenths: exact, portable,
    # and immune to binary-float surprises at the .x5 boundaries.
    hundredths = sum(
        weight * getattr(evaluation, criterion)
        for criterion, weight in SCORE_WEIGHTS.items()
    )
    return ((hundredths + 5) // 10) / 10


# Maps a rounded overall score onto the recommendation the UI and agent speak.
def recommendation_for(score: float) -> Recommendation:
    if score >= ADVANCE_THRESHOLD:
        return "advance"
    if score >= BORDERLINE_THRESHOLD:
        return "borderline"
    return "do_not_advance"


# Builds the stored scorecard: the submission plus what Metis computed from it.
def build_scorecard(evaluation: InterviewEvaluationV1) -> InterviewScorecardV1:
    scorable = (
        not evaluation.incomplete
        or evaluation.completed_question_count >= MIN_SCORED_QUESTIONS
    )
    score = overall_score(evaluation) if scorable else None
    return InterviewScorecardV1(
        evaluation=evaluation,
        overall_score=score,
        recommendation=recommendation_for(score) if score is not None else None,
        provisional=evaluation.incomplete,
        created_at=utc_now(),
    )


class InterviewService:
    """Sessions, transcripts, and the scorecard the evaluation tool writes."""

    def __init__(self, settings: Settings, database: Any, *, model: Any = None) -> None:
        self.settings = settings
        self.database = database
        self.model = model

    # -- availability ------------------------------------------------------

    # Names every missing configuration item, in the owner's language.
    def missing_configuration(self) -> list[str]:
        missing: list[str] = []
        speech = getattr(self.model, "elevenlabs", None)
        if speech is None or not speech.available:
            missing.append("Set WAQIL_ELEVENLABS_API_KEY (an ElevenLabs API key).")
        if not self.settings.interview_elevenlabs_agent_id.strip():
            missing.append(
                "Create the Chiron interview agent per "
                "docs/interviews-elevenlabs-agent.md, then set "
                "WAQIL_INTERVIEW_ELEVENLABS_AGENT_ID."
            )
        return missing

    # Why an interview cannot start; empty string means it can.
    def unavailable_reason(self) -> str:
        missing = self.missing_configuration()
        return missing[0] if missing else ""

    # -- session lifecycle -------------------------------------------------

    async def start(self, context: InterviewContextV1) -> InterviewSessionStartV1:
        """Create the session row, mint the token, and fix the variables.

        The row is written before the provider is called so a refused mint
        leaves a 'failed' session that can explain itself. question_limit is
        stamped here — the browser sends the interview context and nothing
        about the interview's shape.
        """
        reason = self.unavailable_reason()
        if reason:
            raise InterviewError(reason)
        row = await self.database.create_interview_session(
            job_title=context.job_title,
            company_name=context.company_name,
            job_description=context.job_description,
            interview_type=context.interview_type,
            question_limit=QUESTION_LIMIT,
        )
        try:
            token = await self._conversation_token()
        except ModelProviderError:
            await self.database.end_interview_session(row["id"], status="failed")
            raise
        session = await self._session(row["id"])
        return InterviewSessionStartV1(
            session=session,
            conversation_token=token,
            dynamic_variables={
                "job_title": context.job_title.strip(),
                "company_name": context.company_name.strip(),
                "job_description": context.job_description.strip(),
                "interview_type": context.interview_type,
                "question_limit": str(QUESTION_LIMIT),
                "metis_interview_session_id": row["id"],
            },
        )

    async def _conversation_token(self) -> str:
        """A short-lived WebRTC conversation token for the interview agent.

        Minted server-side for the same reason voice mode's is: the
        alternative is handing the browser an API key that opens the whole
        ElevenLabs account. Borrows the speech provider's httpx client so
        there is exactly one key holder and one timeout policy.
        """
        speech = getattr(self.model, "elevenlabs", None)
        if speech is None or not speech.available:
            raise ModelProviderError("Interviews need WAQIL_ELEVENLABS_API_KEY")
        client = await speech._client()
        try:
            response = await client.get(
                "/v1/convai/conversation/token",
                params={"agent_id": self.settings.interview_elevenlabs_agent_id.strip()},
            )
        except Exception as exc:  # noqa: BLE001 - network errors become model errors
            raise ModelProviderError(
                f"Could not reach ElevenLabs: {str(exc)[:200]}"
            ) from exc
        if response.status_code >= 400:
            raise ModelProviderError(
                f"ElevenLabs refused the conversation: HTTP {response.status_code}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise ModelProviderError("ElevenLabs returned a non-JSON reply") from exc
        token = str((payload or {}).get("token") or "").strip()
        if not token:
            raise ModelProviderError("ElevenLabs returned no conversation token")
        return token

    # -- reads and writes --------------------------------------------------

    async def _session(self, session_id: str) -> InterviewSessionV1:
        detail = await self.database.interview_session_detail(session_id)
        if detail is None:
            raise KeyError(session_id)
        return _to_contract(detail)

    async def session(self, session_id: str) -> InterviewSessionV1 | None:
        """The session and its transcript, or None when it does not exist."""
        detail = await self.database.interview_session_detail(session_id)
        return _to_contract(detail) if detail is not None else None

    async def evaluate(
        self, session_id: str, evaluation: InterviewEvaluationV1
    ) -> InterviewScorecardV1 | None:
        """Store the verdict once and return what is stored, always.

        The blocking client tool may retry; a second submission returns the
        scorecard the candidate already heard rather than a new one.
        """
        scorecard = build_scorecard(evaluation)
        status = "ended_early" if evaluation.incomplete else "complete"
        row = await self.database.save_interview_scorecard(
            session_id,
            scorecard_json=scorecard.model_dump_json(),
            status=status,
        )
        if row is None:
            return None
        return InterviewScorecardV1.model_validate_json(row["scorecard_json"])


# Maps a database detail dict onto the session contract.
def _to_contract(detail: dict[str, Any]) -> InterviewSessionV1:
    session = detail["session"]
    scorecard_json = session.get("scorecard_json")
    return InterviewSessionV1(
        id=session["id"],
        provider_conversation_id=session["provider_conversation_id"],
        job_title=session["job_title"],
        company_name=session["company_name"],
        job_description=session["job_description"],
        interview_type=session["interview_type"],
        question_limit=session["question_limit"],
        status=session["status"],
        turns=[
            InterviewTurnV1(
                id=turn["id"],
                ordinal=turn["ordinal"],
                role=turn["role"],
                text=turn["text"],
                created_at=turn["created_at"],
            )
            for turn in detail["turns"]
        ],
        scorecard=(
            InterviewScorecardV1.model_validate_json(scorecard_json)
            if scorecard_json
            else None
        ),
        created_at=session["created_at"],
        updated_at=session["updated_at"],
    )
