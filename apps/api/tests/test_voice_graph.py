"""The restricted voice path, and the things it must never be able to do.

Most of these tests prove an absence, which is the only honest way to test a
boundary. A refusal is not "the model said no" — it is that retrieval never
ran, no account was resolved, and no model was called, so there was never a
decision for anything to get wrong. The fakes here count their own calls for
exactly that reason.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from waqil_api.config import Settings
from waqil_api.contracts import (
    AttentionFeedV1,
    AttentionItemV1,
    CustomerAccountV1,
    KnowledgeSnippetV1,
    ModelResultV1,
)
from waqil_api.speech_preference import SpeechPreferenceStore
from waqil_api.voice_graph import (
    NO_ANSWER,
    VOICE_DETAIL_OUTPUT_TOKENS,
    VOICE_EVIDENCE_SNIPPETS,
    VOICE_HISTORY_TURNS,
    VOICE_OUTPUT_TOKENS,
    VOICE_PERMISSIONS,
    VOICE_RETRIEVAL,
    WITHHELD_ANSWER,
    WITHHELD_FIGURE,
    VoiceGraph,
    VoiceTurn,
)
from waqil_api.voice_intents import BUILD_REFUSAL


def _settings(tmp_path: Path, **overrides) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        repo_root=Path(__file__).resolve().parents[3],
        allow_test_backends=True,
        model_backend="deterministic",
        **overrides,
    )


class CountingModel:
    """A provider that streams a fixed answer and records every request.

    The answer arrives in small chunks that cut across sentence boundaries,
    the way a real token stream does. `probe` is called after every chunk so
    a test can look at what the listener had heard by then.
    """

    def __init__(
        self, text: str = "The sizing went out on Tuesday.", *, chunk: int = 7
    ) -> None:
        self.text = text
        self.chunk = chunk
        self.calls: list[dict] = []
        self.probe = None

    async def generate(self, request, on_token=None, *, model_aliases=None, on_reasoning=None):
        self.calls.append(
            {
                "system_prompt": request.system_prompt,
                "user_prompt": request.user_prompt,
                "max_output_tokens": request.max_output_tokens,
                "reasoning": request.reasoning,
                "model_aliases": model_aliases,
            }
        )
        for start in range(0, len(self.text), self.chunk):
            if on_token is not None:
                await on_token(self.text[start : start + self.chunk])
            if self.probe is not None:
                self.probe()
        return ModelResultV1(model="fake", content=self.text)


class CountingCorpus:
    def __init__(self, snippets: list[KnowledgeSnippetV1] | None = None) -> None:
        self.snippets = snippets or []
        self.calls = 0

    async def retrieve(self, query, limit=None, **kwargs):
        self.calls += 1
        return list(self.snippets)


class CountingAnswers:
    def __init__(self, snippets: list[KnowledgeSnippetV1] | None = None) -> None:
        self.snippets = snippets or []
        self.calls = 0

    async def retrieve(self, query, *, top_k=3):
        self.calls += 1
        return list(self.snippets)


class CountingCustomers:
    def __init__(self, accounts: list[CustomerAccountV1] | None = None) -> None:
        self._accounts = accounts or []
        self.account_calls = 0
        self.evidence_calls: list[str] = []

    async def accounts(self):
        self.account_calls += 1
        return list(self._accounts)

    async def evidence(self, account_id, *, compact=False):
        self.evidence_calls.append(account_id)
        return [
            KnowledgeSnippetV1(
                source_label="Customer record",
                provider="customer",
                rel_path=f"{account_id}/fact/1",
                text="The workshop is on Thursday.",
                score=1.0,
            )
        ]


class CountingAttention:
    def __init__(self) -> None:
        self.calls = 0

    async def feed(self, top=3):
        self.calls += 1
        return AttentionFeedV1(
            generated_at=datetime.now(UTC),
            total=2,
            top=[
                AttentionItemV1(
                    key="customer_action:1",
                    kind="customer_action",
                    kind_label="Commitment",
                    title="Send the sizing to Batelco",
                    overdue=True,
                )
            ],
        )


def _account(account_id: str, name: str, *aliases: str) -> CustomerAccountV1:
    now = datetime.now(UTC)
    return CustomerAccountV1(
        id=account_id,
        name=name,
        aliases=list(aliases),
        created_at=now,
        updated_at=now,
    )


def _graph(tmp_path, **parts) -> tuple[VoiceGraph, dict]:
    settings = _settings(tmp_path)
    built = {
        "model": parts.get("model") or CountingModel(),
        "corpus": parts.get("corpus") or CountingCorpus(),
        "answers": parts.get("answers") or CountingAnswers(),
        "customers": parts.get("customers") or CountingCustomers(),
        "attention": parts.get("attention") or CountingAttention(),
    }
    graph = VoiceGraph(
        settings,
        speech_preference=SpeechPreferenceStore(settings),
        **built,
    )
    return graph, built


def _turn(said: str, **overrides) -> VoiceTurn:
    return VoiceTurn(
        transcript=said,
        voice_session_id=overrides.pop("voice_session_id", "vs_1"),
        turn_id=overrides.pop("turn_id", "t_1"),
        **overrides,
    )


# -- what voice is not given -------------------------------------------------


def test_the_graph_holds_no_registry_sandbox_project_or_mutation_handle(
    tmp_path,
) -> None:
    """The boundary, asserted as an absence rather than as a disabled flag.

    Nothing here was turned off. These were never passed in, so there is no
    branch that reaches them and no setting that could bring one back.
    """
    graph, _ = _graph(tmp_path)
    forbidden = (
        "registry",
        "tools",
        "sandbox",
        "project_sandbox",
        "projects",
        "coding_engine",
        "coding_sessions",
        "broker",
        "tool_model",
        "database",
        "blobs",
        "checkpointer",
        "memory_index",
        "web",
        "run_history",
    )
    for name in forbidden:
        assert not hasattr(graph, name), f"the voice graph must not hold {name}"


def test_the_retrieval_profile_excludes_what_voice_may_not_read() -> None:
    assert VOICE_RETRIEVAL.tool_catalog is False
    assert VOICE_RETRIEVAL.project_context is False
    assert VOICE_RETRIEVAL.attachments is False
    assert VOICE_RETRIEVAL.memory_harvest is False
    assert VOICE_RETRIEVAL.web_research is False
    assert VOICE_RETRIEVAL.corpus and VOICE_RETRIEVAL.customer_evidence


def test_the_ceiling_carries_no_execution_network_or_tool_permission() -> None:
    names = {permission.value for permission in VOICE_PERMISSIONS}
    assert names == {"conversation:respond", "read:run-inputs", "model:broker"}


# -- refusals ----------------------------------------------------------------


@pytest.mark.parametrize(
    "said",
    [
        "Build me a tool that summarises meeting notes",
        "build it",
        "Turn this into a tool",
        "Create a new application for tracking wins",
        "Rebuild the ledger app",
    ],
)
@pytest.mark.asyncio
async def test_a_build_request_never_reaches_retrieval_or_a_model(
    tmp_path, said
) -> None:
    graph, parts = _graph(tmp_path)
    rendition = await graph.answer(_turn(said))

    assert rendition.intent == "refuse_build"
    assert rendition.spoken == BUILD_REFUSAL
    # The verbatim utterance travels with the refusal so the composer can be
    # prefilled with what was actually said.
    assert rendition.transcript == said
    assert parts["model"].calls == []
    assert parts["corpus"].calls == 0
    assert parts["answers"].calls == 0
    assert parts["customers"].account_calls == 0
    assert parts["attention"].calls == 0


@pytest.mark.parametrize(
    ("said", "expected"),
    [
        ("Approve the run that's waiting", "Approvals aren't something"),
        ("Delete the note about Batelco", "won't delete or overwrite"),
        ("Mark that action done", "won't delete or overwrite"),
        ("Read the file at ~/Projects/notes.md", "can't reach files"),
        ("Run the verification script", "can't run code"),
        ("Switch to the Grok model", "can't switch models"),
    ],
)
@pytest.mark.asyncio
async def test_each_protected_intent_has_a_fixed_answer_and_no_graph_entry(
    tmp_path, said, expected
) -> None:
    graph, parts = _graph(tmp_path)
    rendition = await graph.answer(_turn(said))

    assert rendition.intent == "refuse_protected"
    assert expected in rendition.spoken
    assert parts["model"].calls == []
    assert parts["corpus"].calls == 0
    assert parts["customers"].account_calls == 0


@pytest.mark.asyncio
async def test_activating_a_tool_is_refused_by_the_shared_build_classifier(
    tmp_path,
) -> None:
    """Which refusal it earns is the existing classifier's call, not ours.

    `_BUILD_PATTERNS` already claims "activate ... tool", so this lands on the
    build refusal rather than the tool one. Pinned as behavior rather than
    corrected: a private second opinion about what counts as a build request
    is exactly the gap the shared classifier exists to prevent.
    """
    graph, parts = _graph(tmp_path)
    rendition = await graph.answer(_turn("Activate the sizing tool"))
    assert rendition.intent in ("refuse_build", "refuse_protected")
    assert parts["model"].calls == []


# -- reading -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_ordinary_question_retrieves_and_says_what_it_shows(
    tmp_path,
) -> None:
    corpus = CountingCorpus(
        [
            KnowledgeSnippetV1(
                source_label="Service request",
                provider="notion",
                rel_path="sr-9912.md",
                text="The sizing was sent on Tuesday.",
                score=0.8,
            )
        ]
    )
    graph, parts = _graph(tmp_path, corpus=corpus)
    rendition = await graph.answer(_turn("When did the sizing go out?"))

    assert rendition.intent == "read"
    # One text for the ear and the eye; the sources sit beside it as cards.
    assert rendition.spoken == "The sizing went out on Tuesday."
    assert rendition.written == rendition.spoken
    assert rendition.spoken_fallback is False
    assert [item.label for item in rendition.citations] == ["Service request"]
    assert corpus.calls == 1
    call = parts["model"].calls[0]
    assert len(parts["model"].calls) == 1
    # No thinking first: the answer has to land inside a conversational pause.
    assert call["reasoning"] is False


@pytest.mark.asyncio
async def test_sentences_are_spoken_while_the_model_is_still_writing(
    tmp_path,
) -> None:
    """The point of streaming, asserted from the listener's side.

    By the time the model has finished its second sentence, the first one has
    already been handed to the caller — not held until the reply is whole.
    """
    model = CountingModel("The sizing went out on Tuesday. Batelco confirmed it.")
    heard: list[str] = []
    heard_by_chunk: list[int] = []
    model.probe = lambda: heard_by_chunk.append(len(heard))

    async def listen(sentence: str) -> None:
        heard.append(sentence)

    graph, _ = _graph(tmp_path, model=model)
    rendition = await graph.answer(
        _turn("When did the sizing go out?"), on_spoken=listen
    )

    assert heard == ["The sizing went out on Tuesday.", "Batelco confirmed it."]
    assert rendition.spoken == " ".join(heard)
    # The first sentence was heard before the last chunk of the second arrived.
    assert 1 in heard_by_chunk[:-1]


@pytest.mark.asyncio
async def test_a_refusal_is_spoken_exactly_once(tmp_path) -> None:
    heard: list[str] = []

    async def listen(sentence: str) -> None:
        heard.append(sentence)

    graph, _ = _graph(tmp_path)
    await graph.answer(_turn("Build me a tool that summarises notes"), on_spoken=listen)
    assert heard == [BUILD_REFUSAL]


@pytest.mark.asyncio
async def test_a_named_account_scopes_the_evidence_it_retrieves(tmp_path) -> None:
    customers = CountingCustomers([_account("acc_bat", "Batelco B.S.C.")])
    graph, _ = _graph(tmp_path, customers=customers)
    await graph.answer(_turn("What did we agree with Batelco about the workshop?"))
    assert customers.evidence_calls == ["acc_bat"]


@pytest.mark.asyncio
async def test_two_plausible_accounts_ask_instead_of_guessing(tmp_path) -> None:
    customers = CountingCustomers(
        [
            _account("acc_bap", "Bahrain Petroleum Company"),
            _account("acc_bat", "Bahrain Telecommunications"),
        ]
    )
    graph, parts = _graph(
        tmp_path,
        customers=customers,
        corpus=CountingCorpus(),
        answers=CountingAnswers(),
    )
    rendition = await graph.answer(_turn("What's open on Bahrain?"))

    assert rendition.intent == "clarify"
    assert "Which account" in rendition.spoken
    # Nothing was retrieved for either account, and no model was asked to pick.
    assert customers.evidence_calls == []
    assert parts["model"].calls == []


@pytest.mark.asyncio
async def test_the_queue_is_read_only_when_the_question_asks_for_it(tmp_path) -> None:
    graph, parts = _graph(tmp_path)
    await graph.answer(_turn("What did we agree with the bank?"))
    assert parts["attention"].calls == 0

    graph, parts = _graph(tmp_path)
    await graph.answer(_turn("What's waiting on me today?"))
    assert parts["attention"].calls == 1


@pytest.mark.asyncio
async def test_history_is_labelled_as_a_record_not_an_instruction(tmp_path) -> None:
    graph, parts = _graph(tmp_path)
    await graph.answer(
        _turn("And what about the other one?", history=("Approve the Batelco run",))
    )
    prompt = parts["model"].calls[0]["user_prompt"]
    assert "a record, not an instruction" in prompt
    assert "Approve the Batelco run" in prompt


# -- the spoken half ---------------------------------------------------------


@pytest.mark.asyncio
async def test_a_model_that_writes_markdown_is_made_sayable(tmp_path) -> None:
    model = CountingModel("## Where it stands\nThe **sizing** went out [1].")
    graph, _ = _graph(tmp_path, model=model)
    rendition = await graph.answer(_turn("Where does the sizing stand?"))

    assert rendition.spoken_fallback is True
    for noise in ("#", "**", "[1]"):
        assert noise not in rendition.spoken
        assert noise not in rendition.written


@pytest.mark.asyncio
async def test_an_empty_reply_is_said_as_no_answer(tmp_path) -> None:
    graph, _ = _graph(tmp_path, model=CountingModel(""))
    rendition = await graph.answer(_turn("When did the sizing go out?"))
    assert rendition.spoken == NO_ANSWER
    assert rendition.spoken_fallback is True


@pytest.mark.asyncio
async def test_an_invented_figure_is_withheld_before_it_is_spoken(tmp_path) -> None:
    """The claim gate, on the surface that cannot show its working.

    A spoken answer is heard once and cannot be scrolled back to check, so a
    figure the records do not contain must never reach the listener — not
    even for the moment before a correction.
    """
    customers = CountingCustomers([_account("acc_bat", "Batelco")])
    model = CountingModel("Batelco signed for $4,200,000 last quarter.")
    heard: list[str] = []

    async def listen(sentence: str) -> None:
        heard.append(sentence)

    graph, _ = _graph(tmp_path, model=model, customers=customers)
    rendition = await graph.answer(
        _turn("How big was the Batelco deal?"), on_spoken=listen
    )

    assert heard == [WITHHELD_ANSWER]
    assert rendition.spoken == rendition.written == WITHHELD_ANSWER
    assert "4,200,000" not in rendition.spoken


@pytest.mark.asyncio
async def test_a_supported_sentence_is_kept_when_a_later_one_is_withheld(
    tmp_path,
) -> None:
    customers = CountingCustomers([_account("acc_bat", "Batelco")])
    model = CountingModel(
        "The workshop is on Thursday. Batelco signed for $4,200,000 last quarter."
    )
    graph, _ = _graph(tmp_path, model=model, customers=customers)
    rendition = await graph.answer(_turn("What's the latest with Batelco?"))

    assert rendition.spoken == f"The workshop is on Thursday. {WITHHELD_FIGURE}"
    assert rendition.spoken_fallback is True


# -- brevity -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_ordinary_turn_stays_inside_the_brief_budget(tmp_path) -> None:
    """Brevity is the default, and it is enforced rather than requested.

    A spoken answer is two to four sentences. Everything sent to reach it is
    paid for on this turn and again on the next one through the history, so
    the budget is sized for what the ear can hold, not for what the context
    window would allow.
    """
    corpus = CountingCorpus(
        [
            KnowledgeSnippetV1(
                source_label=f"Source {index}",
                provider="local",
                rel_path=f"doc-{index}.md",
                text="x" * 4_000,
                score=0.5,
            )
            for index in range(20)
        ]
    )
    graph, parts = _graph(tmp_path, corpus=corpus)
    await graph.answer(
        _turn(
            "When did the sizing go out?",
            history=tuple(f"They: {'y' * 2_000}" for _ in range(20)),
        )
    )
    call = parts["model"].calls[0]
    prompt = call["user_prompt"]
    assert prompt.count("] Source") <= VOICE_EVIDENCE_SNIPPETS
    assert len(prompt) < 8_000, "one spoken turn should not cost an essay"
    assert call["max_output_tokens"] == VOICE_OUTPUT_TOKENS
    # Only the tail of the history travels, and each line is clipped.
    assert prompt.count("They: yyy") <= VOICE_HISTORY_TURNS


@pytest.mark.parametrize(
    "said",
    [
        "Walk me through the Batelco account in detail",
        "Tell me more about the sizing",
        "Give me the long version",
        "Break that down for me",
    ],
)
@pytest.mark.asyncio
async def test_asking_for_detail_raises_both_ceilings_for_that_turn(
    tmp_path, said
) -> None:
    graph, parts = _graph(tmp_path)
    await graph.answer(_turn(said))
    call = parts["model"].calls[0]
    assert call["max_output_tokens"] == VOICE_DETAIL_OUTPUT_TOKENS
    assert "asked for detail" in call["system_prompt"]


@pytest.mark.asyncio
async def test_a_long_spoken_answer_is_cut_to_four_sentences_unless_asked(
    tmp_path,
) -> None:
    """Where "brief unless asked" is actually enforced.

    A model told to be brief is often brief. A model held to four sentences
    always is — and the turn where someone asked for the long version gets it
    because they asked, not because the model felt expansive.
    """
    long_answer = " ".join(f"Sentence number {index}." for index in range(1, 9))

    graph, _ = _graph(tmp_path, model=CountingModel(long_answer))
    brief = await graph.answer(_turn("Where does the sizing stand?"))
    assert brief.spoken.count(".") == 4
    assert brief.spoken_fallback is True

    graph, _ = _graph(tmp_path, model=CountingModel(long_answer))
    detailed = await graph.answer(_turn("Walk me through where the sizing stands"))
    assert detailed.spoken.count(".") == 8


# -- the ceiling -------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_ceiling_is_evaluated_before_and_after_the_model_replies(
    tmp_path,
) -> None:
    graph, _ = _graph(tmp_path)
    assert graph.last_outcome is None
    await graph.answer(_turn("When did the sizing go out?"))
    # The last evaluation is the one made after the model finished speaking.
    assert graph.last_outcome.action == "voice.answer"
    assert graph.last_outcome.disposition.value == "allow"
    assert graph.last_outcome.declared_risk.value == "R2"


@pytest.mark.asyncio
async def test_a_turn_without_a_session_identity_is_refused(tmp_path) -> None:
    graph, _ = _graph(tmp_path)
    with pytest.raises(ValueError, match="session and a turn identity"):
        await graph.answer(_turn("anything", voice_session_id=""))


@pytest.mark.asyncio
async def test_a_failed_retrieval_arm_costs_itself_and_not_the_turn(
    tmp_path,
) -> None:
    class BrokenCorpus(CountingCorpus):
        async def retrieve(self, query, limit=None, **kwargs):
            self.calls += 1
            raise RuntimeError("the index is being rebuilt")

    corpus = BrokenCorpus()
    graph, parts = _graph(tmp_path, corpus=corpus)
    rendition = await graph.answer(_turn("When did the sizing go out?"))

    assert corpus.calls == 1
    assert rendition.intent == "read"
    assert len(parts["model"].calls) == 1
