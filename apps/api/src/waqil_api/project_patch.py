"""Finding the block a patch means to replace.

``apply_patch`` asks the model to quote the current text exactly and then
swaps that text for a replacement. Exactness is what makes the tool safe: an
ambiguous quote is refused rather than guessed at, so a patch can never land
somewhere the model did not look.

Exactness is also what made it unusable for local models. A live repair turn
spent six consecutive attempts on one line and lost every one of them to a
single space — the file held ``Decimal("0"))  # just the sum`` and the model
quoted ``Decimal("0"))   # just the sum``. The quote was semantically perfect
and byte-wrong, the turn ended without fixing a defect it had diagnosed
correctly, and the same shape accounted for every failed write across two
models and eight attempts.

So the match is tried in tiers, from strictest to loosest, and the first tier
that finds a candidate decides. What never relaxes is the ambiguity rule:
a tier that matches more than one place in the file is refused outright, at
that tier, rather than falling through to a looser one that might match fewer
places by accident. The looser tiers only ever forgive whitespace — never a
character of code — and the tier that matched travels back with the result so
the model, the trace, and the approval card all say how the block was found.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

# How a block was located, in the order the tiers are tried. "exact" is the
# original contract; the rest forgive progressively more whitespace.
EXACT = "exact"
TRAILING = "ignoring trailing whitespace"
INDENT = "ignoring indentation"
SPACING = "ignoring whitespace"


@dataclass(frozen=True)
class PatchMatch:
    """Where the block is, and the replacement text fitted to that location."""

    start: int
    end: int
    replacement: str
    how: str

    @property
    def fuzzy(self) -> bool:
        return self.how != EXACT


@dataclass(frozen=True)
class PatchProblem:
    """Why no single block could be chosen. ``count`` is 0 or more than 1."""

    count: int
    how: str


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip())


def _shift(line: str, delta: int) -> str:
    """Move one replacement line by the indent delta the file actually uses.

    Blank lines are left alone: padding them turns a clean file into one with
    trailing whitespace on every empty line inside the patched block.
    """
    if not line.strip() or delta == 0:
        return line
    if delta > 0:
        return " " * delta + line
    return line[min(-delta, _indent_of(line)) :]


# Each tier is a per-line normalizer. Whitespace only — no tier may make two
# different sequences of code characters compare equal.
_TIERS: tuple[tuple[str, Callable[[str], str]], ...] = (
    (TRAILING, lambda line: line.rstrip()),
    (INDENT, lambda line: line.strip()),
    (SPACING, lambda line: " ".join(line.split())),
)


def locate_patch(
    text: str, original: str, replacement: str
) -> PatchMatch | PatchProblem:
    """The one place ``original`` refers to, or why there is no single place.

    Fuzzy tiers match whole lines only. A partial-line quote that is not
    byte-exact stays a refusal: relaxing mid-line matching would let a
    replacement land inside an expression the model never quoted in full.
    """
    exact_count = text.count(original)
    if exact_count == 1:
        start = text.index(original)
        return PatchMatch(start, start + len(original), replacement, EXACT)
    if exact_count > 1:
        # Ambiguous exactly: a looser tier can only be more ambiguous, and
        # the model's own fix is to quote more surrounding lines.
        return PatchProblem(exact_count, EXACT)

    original_lines = original.splitlines()
    if not original_lines:
        return PatchProblem(0, EXACT)

    lines = text.splitlines(keepends=True)
    offsets: list[int] = []
    position = 0
    for line in lines:
        offsets.append(position)
        position += len(line)
    offsets.append(position)

    width = len(original_lines)
    for how, normalize in _TIERS:
        wanted = [normalize(line) for line in original_lines]
        windows = [
            index
            for index in range(len(lines) - width + 1)
            if [normalize(line) for line in lines[index : index + width]] == wanted
        ]
        if not windows:
            continue
        if len(windows) > 1:
            return PatchProblem(len(windows), how)
        index = windows[0]
        start, end = offsets[index], offsets[index + width]
        # The file's indentation wins over the model's. It quoted the block
        # from memory; the file is what has to keep parsing.
        delta = _indent_of(lines[index]) - _indent_of(original_lines[0])
        fitted = "\n".join(_shift(line, delta) for line in replacement.splitlines())
        if text[start:end].endswith("\n") and not fitted.endswith("\n"):
            fitted += "\n"
        return PatchMatch(start, end, fitted, how)

    return PatchProblem(0, EXACT)
