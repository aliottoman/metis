"""Delivery metrics measured from the call audio, not estimated from a hunch.

The live agent transcript is cleaned before the model sees it — the "um"s are
gone before anyone could count them. So delivery analysis is a second pass:
the call recording goes through verbatim speech-to-text (Scribe keeps fillers
and false starts by default), the candidate's words are separated from the
interviewer's by diarization, and everything reported here is a count over
those words. No number in this module is a judgment call; judgments stay in
the scorecard, which a model wrote and Metis weighed.
"""

from __future__ import annotations

from typing import Any

# One utterance of any of these is a filler. Single tokens are matched after
# punctuation is stripped; phrases are matched as consecutive tokens. The set
# is deliberately conservative — "like" is absent because counting the
# preposition would inflate the number, and an inflated count is an invented
# delivery problem, the exact thing the evaluation prompt forbids.
FILLER_TOKENS = frozenset(
    {"um", "uh", "erm", "uhm", "umm", "uhh", "hmm", "mhm", "er"}
)
FILLER_PHRASES: tuple[tuple[str, ...], ...] = (
    ("you", "know"),
    ("i", "mean"),
    ("sort", "of"),
    ("kind", "of"),
)

# Hedges soften a claim without adding information. Counted separately from
# fillers because the fix is different: fillers are pacing, hedges are
# confidence.
HEDGE_PHRASES: tuple[tuple[str, ...], ...] = (
    ("i", "think"),
    ("i", "guess"),
    ("i", "believe"),
    ("i", "feel", "like"),
    ("maybe",),
    ("probably",),
    ("possibly",),
    ("hopefully",),
    ("not", "sure"),
)

# A silence this long inside the candidate's own answer counts as a pause.
LONG_PAUSE_SECONDS = 2.0


def _normalize(token: str) -> str:
    return "".join(ch for ch in token.lower() if ch.isalpha() or ch == "'")


def spoken_words(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """The Scribe payload's spoken words, in the order they were spoken.

    Spacing and audio-event entries are dropped — they carry no words to
    count. Provider order is kept as-is (timings are nullable, so sorting by
    them would yank an untimed word out of its phrase), the same trust
    `turns_from_scribe` extends to the meetings payload.
    """
    return [
        item
        for item in payload.get("words") or []
        if isinstance(item, dict) and item.get("type", "word") == "word"
    ]


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _speaker_texts(words: list[dict[str, Any]]) -> dict[str, list[str]]:
    by_speaker: dict[str, list[str]] = {}
    for item in words:
        speaker = str(item.get("speaker_id") or "speaker_0")
        by_speaker.setdefault(speaker, []).append(_normalize(str(item.get("text", ""))))
    return by_speaker


def _containment(tokens: list[str], corpus: frozenset[str]) -> float:
    spoken = [token for token in tokens if token]
    if not spoken or not corpus:
        return 0.0
    return sum(1 for token in spoken if token in corpus) / len(spoken)


def candidate_speaker(
    words: list[dict[str, Any]],
    agent_texts: list[str],
    user_texts: list[str],
) -> str | None:
    """Which diarized speaker is the candidate.

    The stored transcript already knows who said what — the browser labeled
    every line user or agent as it was heard. Each diarized voice is scored by
    how much of what it said appears in each side of that transcript, and the
    candidate is the voice that reads like the user side, not the agent side.
    """
    by_speaker = _speaker_texts(words)
    if not by_speaker:
        return None
    if len(by_speaker) == 1:
        return next(iter(by_speaker))
    agent_corpus = frozenset(
        _normalize(token) for text in agent_texts for token in text.split()
    )
    user_corpus = frozenset(
        _normalize(token) for text in user_texts for token in text.split()
    )
    return max(
        by_speaker,
        key=lambda speaker: (
            _containment(by_speaker[speaker], user_corpus)
            - _containment(by_speaker[speaker], agent_corpus),
            len(by_speaker[speaker]),
        ),
    )


def _count_phrases(
    tokens: list[str], phrases: tuple[tuple[str, ...], ...]
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for phrase in phrases:
        size = len(phrase)
        found = sum(
            1
            for start in range(len(tokens) - size + 1)
            if tuple(tokens[start : start + size]) == phrase
        )
        if found:
            counts[" ".join(phrase)] = found
    return counts


def delivery_metrics(
    words: list[dict[str, Any]],
    candidate: str,
) -> dict[str, Any]:
    """Every metric the Delivery section shows, counted from the words.

    Talk time sums the candidate's speech runs (a run breaks at a long
    silence); a silence only counts as a pause when nobody else was speaking
    in it — waiting through the interviewer's question is not hesitating.
    """
    mine = [
        item
        for item in words
        if str(item.get("speaker_id") or "speaker_0") == candidate
    ]
    # Scribe's timings are nullable. A word without them still counts as a
    # word, but it cannot take part in run/pause accounting — a null start
    # coerced to 0.0 would open a phantom hours-long pause at the call's
    # beginning and inflate talk time to match.
    timed = [
        item
        for item in mine
        if item.get("start") is not None and item.get("end") is not None
    ]
    timed_others = [
        item
        for item in words
        if str(item.get("speaker_id") or "speaker_0") != candidate
        and item.get("start") is not None
        and item.get("end") is not None
    ]
    tokens = [_normalize(str(item.get("text", ""))) for item in mine]
    tokens = [token for token in tokens if token]

    filler_breakdown: dict[str, int] = {}
    for token in tokens:
        if token in FILLER_TOKENS:
            filler_breakdown[token] = filler_breakdown.get(token, 0) + 1
    for phrase, count in _count_phrases(tokens, FILLER_PHRASES).items():
        filler_breakdown[phrase] = count
    filler_count = sum(filler_breakdown.values())

    hedging_breakdown = _count_phrases(tokens, HEDGE_PHRASES)
    hedging_count = sum(hedging_breakdown.values())

    talk_seconds = 0.0
    long_pause_count = 0
    longest_pause = 0.0
    run_start: float | None = None
    previous_end: float | None = None
    for item in timed:
        start, end = _number(item.get("start")), _number(item.get("end"))
        if run_start is None:
            run_start, previous_end = start, end
            continue
        gap = start - (previous_end or start)
        if gap > LONG_PAUSE_SECONDS:
            talk_seconds += (previous_end or run_start) - run_start
            if not _someone_spoke_between(timed_others, previous_end or start, start):
                long_pause_count += 1
                longest_pause = max(longest_pause, gap)
            run_start = start
        previous_end = max(previous_end or end, end)
    if run_start is not None and previous_end is not None:
        talk_seconds += previous_end - run_start

    word_count = len(tokens)
    words_per_minute = (
        (word_count / talk_seconds) * 60 if talk_seconds > 0 else 0.0
    )
    return {
        "candidate_word_count": word_count,
        "candidate_talk_seconds": round(talk_seconds, 1),
        "words_per_minute": round(words_per_minute, 1),
        "filler_count": filler_count,
        "filler_rate_per_100_words": (
            round((filler_count / word_count) * 100, 1) if word_count else 0.0
        ),
        "filler_breakdown": filler_breakdown,
        "hedging_count": hedging_count,
        "hedging_breakdown": hedging_breakdown,
        "long_pause_count": long_pause_count,
        "longest_pause_seconds": round(longest_pause, 1),
    }


def _someone_spoke_between(
    others: list[dict[str, Any]], gap_start: float, gap_end: float
) -> bool:
    return any(
        _number(item.get("start")) < gap_end
        and _number(item.get("end")) > gap_start
        for item in others
    )
