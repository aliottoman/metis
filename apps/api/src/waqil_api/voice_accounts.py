"""Which account was that, said out loud.

The scale here exists because there wasn't one. `/customers/search` is a
SQLite substring scan whose match is a boolean, the entity graph carries no
weight, and the `confidence` floats on facts and memory proposals score a
model's belief in a *claim*, not the quality of a *match*. Retrieval scores
are rank-derived and not comparable between queries, so none of them can carry
an absolute threshold. Setting one on any of them would have been a number
with nothing behind it.

So this is a deterministic name-match score, not a model float. Which customer
this is, is a string-identity question, and a deterministic scale stays
auditable and stable: the same utterance scores the same way today and next
month, and when it picks wrong you can read exactly why.

    1.00  exact case-folded match on the name or a registered alias
    0.92  exact after normalizing punctuation, spacing, and legal suffixes
    0.75  token-set overlap or a bounded edit distance, scaled to 0.90
    <0.75 a substring or a single shared token, and nothing more

An account links itself only when the transcript names it essentially exactly
and nothing else comes close. Both thresholds are constants so the margin can
be tightened after real use without hunting for a literal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

# Decided in docs/metis-voice-mode-brief.md, Appendix A.1. Auto-resolve only
# at or above the score, and only when the runner-up trails by the margin —
# a near-miss spelling, or two plausible accounts, becomes a question instead.
AUTO_LINK_SCORE = 0.90
AUTO_LINK_MARGIN = 0.15

# Corporate furniture. Stripped for the normalized comparison only: "Batelco
# BSC" and "Batelco" are the same company, and nobody says the suffix aloud.
_LEGAL_SUFFIXES = frozenset(
    {
        "bsc",
        "co",
        "corp",
        "corporation",
        "gmbh",
        "inc",
        "incorporated",
        "llc",
        "llp",
        "ltd",
        "limited",
        "nv",
        "plc",
        "pjsc",
        "psc",
        "sa",
        "saog",
        "sarl",
        "spa",
        "wll",
    }
)

# Dots vanish rather than splitting a word, so "B.S.C." normalizes to the
# suffix "bsc" and is dropped whole. Splitting on it instead left three
# one-letter tokens that no suffix list can recognize, and "Batelco B.S.C."
# scored as a partial match against someone saying "Batelco".
_DOTS = re.compile(r"\.(?=\w)|(?<=\w)\.")
_PUNCTUATION = re.compile(r"[^\w\s]+", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")
# Words too short or too common to mean an account on their own. Without this
# an account called "The Group" scores against half of everything said.
_MIN_TOKEN_CHARS = 3


@dataclass(frozen=True, slots=True)
class AccountMatch:
    """One account's claim on an utterance."""

    account_id: str
    name: str
    score: float


@dataclass(frozen=True, slots=True)
class AccountResolution:
    """Who the utterance meant, and whether that is certain enough to act on."""

    best: AccountMatch | None = None
    runner_up: AccountMatch | None = None
    # True only when the score clears the threshold AND the runner-up trails by
    # the margin. Everything else is a question, never a guess.
    decisive: bool = False
    # Set when the answer came from the page the user is looking at rather than
    # from anything they said. A receipt still has to name the account.
    from_scope: bool = False
    candidates: tuple[AccountMatch, ...] = field(default_factory=tuple)

    @property
    def ambiguous(self) -> bool:
        """More than one account is plausible, so nothing may be committed."""
        return not self.decisive and self.best is not None


def _fold(value: str) -> str:
    return _WHITESPACE.sub(" ", (value or "").casefold()).strip()


def _normalize(value: str) -> str:
    """Case, punctuation, spacing and legal suffixes all removed."""
    folded = _PUNCTUATION.sub(" ", _DOTS.sub("", _fold(value)))
    tokens = [
        token for token in folded.split() if token and token not in _LEGAL_SUFFIXES
    ]
    return " ".join(tokens)


def _contains_phrase(haystack: str, needle: str) -> bool:
    """Whole-token containment, so "SAP" never matches inside "sapphire"."""
    if not needle:
        return False
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack) is not None


def _edit_distance(left: str, right: str, *, ceiling: int = 2) -> int:
    """Levenshtein, abandoned once it is past the ceiling we care about."""
    if abs(len(left) - len(right)) > ceiling:
        return ceiling + 1
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        if min(current) > ceiling:
            return ceiling + 1
        previous = current
    return previous[-1]


def score_name(candidate: str, utterance: str) -> float:
    """How strongly one name or alias is present in what was said."""
    folded_candidate, folded_utterance = _fold(candidate), _fold(utterance)
    if not folded_candidate or not folded_utterance:
        return 0.0
    if folded_candidate == folded_utterance or _contains_phrase(
        folded_utterance, folded_candidate
    ):
        return 1.00

    normal_candidate = _normalize(candidate)
    normal_utterance = _normalize(utterance)
    if not normal_candidate:
        return 0.0
    if normal_candidate == normal_utterance or _contains_phrase(
        normal_utterance, normal_candidate
    ):
        return 0.92

    candidate_tokens = [
        token for token in normal_candidate.split() if len(token) >= _MIN_TOKEN_CHARS
    ]
    utterance_tokens = [
        token for token in normal_utterance.split() if len(token) >= _MIN_TOKEN_CHARS
    ]
    if not candidate_tokens or not utterance_tokens:
        return 0.0

    spoken = set(utterance_tokens)
    present = [token for token in candidate_tokens if token in spoken]
    coverage = len(present) / len(candidate_tokens)
    if coverage == 1.0:
        # Every word of the name is there, out of order or interrupted: "the
        # Bahrain Petroleum account" for "Bahrain Petroleum Company".
        return 0.90

    # A bounded misspelling of a name long enough for the distance to mean
    # something. "Batelko" for "Batelco" is this; "AB" for "AC" is not. A
    # distance of zero is deliberately excluded: that is a token the coverage
    # test already counted, and promoting it here would score one shared word
    # out of three as a near-match when the table calls it overlap.
    near = any(
        len(token) >= 5 and 1 <= _edit_distance(token, spoken_token) <= 2
        for token in candidate_tokens
        for spoken_token in utterance_tokens
    )
    if near:
        return 0.80 if coverage < 0.5 else 0.85

    if coverage >= 0.5:
        # Half the name, exactly said. Real, and nowhere near enough to act on.
        return 0.75
    return 0.60 if present else 0.0


def score_account(name: str, aliases: Iterable[str], utterance: str) -> float:
    """The best claim this account can make, across its name and every alias."""
    return max(
        (score_name(candidate, utterance) for candidate in (name, *aliases)),
        default=0.0,
    )


def resolve_account(
    accounts: Sequence[tuple[str, str, Iterable[str]]],
    utterance: str,
    *,
    scoped_account_id: str | None = None,
) -> AccountResolution:
    """Which account this utterance is about.

    `accounts` is `(id, name, aliases)` per account. `scoped_account_id` is the
    page the user is looking at, which supplies the answer only when what they
    said names no account at all — an utterance that explicitly names a
    different or ambiguous account is never resolved to whatever happens to be
    on screen, because "add a note to Batelco" typed on the BAPCO page means
    Batelco or it means ask.
    """
    scored = sorted(
        (
            AccountMatch(account_id, name, score_account(name, aliases, utterance))
            for account_id, name, aliases in accounts
        ),
        key=lambda match: (-match.score, match.name.casefold()),
    )
    named = tuple(match for match in scored if match.score > 0.0)
    best = named[0] if named else None
    runner_up = named[1] if len(named) > 1 else None

    if best is None:
        scope = next(
            (
                AccountMatch(account_id, name, 1.00)
                for account_id, name, _ in accounts
                if scoped_account_id and account_id == scoped_account_id
            ),
            None,
        )
        # Decisive, but flagged: the scope is a real answer to "which account",
        # and the receipt still has to say which one it landed on.
        return AccountResolution(
            best=scope, decisive=scope is not None, from_scope=scope is not None
        )

    margin = best.score - (runner_up.score if runner_up else 0.0)
    decisive = best.score >= AUTO_LINK_SCORE and margin >= AUTO_LINK_MARGIN
    return AccountResolution(
        best=best,
        runner_up=runner_up,
        decisive=decisive,
        candidates=named[:4],
    )
