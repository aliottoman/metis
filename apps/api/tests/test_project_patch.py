"""Whitespace may be forgiven when locating a patch. Code may not.

The headline case is measured, not invented: a live repair turn lost six
consecutive attempts on one line because the file held two spaces before a
trailing comment and the model quoted three. Every other test here exists to
bound that forgiveness — ambiguity still refuses, code characters still have
to match, and the file's own indentation still wins.
"""

from __future__ import annotations

from waqil_api.project_patch import (
    EXACT,
    INDENT,
    SPACING,
    TRAILING,
    PatchMatch,
    PatchProblem,
    locate_patch,
)

# The real file text and the real quote, byte for byte, from the run that
# spent six attempts here. They differ by one space before the comment.
FILE = (
    "def check(items):\n"
    "    line_totals_sum = Decimal(0)\n"
    "    for lt in items:\n"
    "        line_totals_sum += lt\n"
    "\n"
    '    expected_subtotal = sum_money(line_totals_sum, Decimal("0"))  # just the sum\n'
    "    return expected_subtotal\n"
)
QUOTED = (
    "        line_totals_sum += lt\n"
    "\n"
    '    expected_subtotal = sum_money(line_totals_sum, Decimal("0"))   # just the sum'
)


def _apply(text: str, match: PatchMatch) -> str:
    return text[: match.start] + match.replacement + text[match.end :]


def test_the_six_attempt_failure_now_matches() -> None:
    replacement = (
        "        line_totals_sum += lt\n"
        "\n"
        "    expected_subtotal = sum_money(item_totals)  # one argument\n"
    )
    match = locate_patch(FILE, QUOTED, replacement)
    assert isinstance(match, PatchMatch)
    assert match.how == SPACING
    assert match.fuzzy is True
    updated = _apply(FILE, match)
    assert "sum_money(item_totals)" in updated
    assert "sum_money(line_totals_sum" not in updated
    # Everything outside the quoted block is untouched.
    assert updated.startswith("def check(items):\n    line_totals_sum = Decimal(0)\n")
    assert updated.endswith("    return expected_subtotal\n")


def test_exact_still_wins_and_is_reported_as_exact() -> None:
    quoted = '    expected_subtotal = sum_money(line_totals_sum, Decimal("0"))  # just the sum'
    match = locate_patch(FILE, quoted, "    expected_subtotal = total()")
    assert isinstance(match, PatchMatch)
    assert match.how == EXACT
    assert match.fuzzy is False


def test_trailing_whitespace_is_forgiven_before_looser_tiers() -> None:
    text = "a = 1\nb = 2\n"
    match = locate_patch(text, "b = 2   ", "b = 3")
    assert isinstance(match, PatchMatch)
    assert match.how == TRAILING


def test_indentation_is_forgiven_and_the_file_wins() -> None:
    # The quote is deeper than the file, so it cannot match as a substring —
    # under-quoting the indent would, since "    x" lives inside "        x".
    text = "def f():\n    shallow = 1\n"
    match = locate_patch(text, "        shallow = 1", "        shallow = 2")
    assert isinstance(match, PatchMatch)
    assert match.how == INDENT
    # The replacement is re-fitted to the file's real indentation, so the
    # result still parses rather than inheriting the model's guess.
    assert _apply(text, match) == "def f():\n    shallow = 2\n"


def test_ambiguity_refuses_at_every_tier() -> None:
    exact_twice = "x = 1\nx = 1\n"
    problem = locate_patch(exact_twice, "x = 1", "x = 2")
    assert isinstance(problem, PatchProblem)
    assert problem.count == 2
    assert problem.how == EXACT

    # Unique nowhere exactly, but two whitespace-equal candidates.
    spacing_twice = "x  = 1\nx   = 1\n"
    problem = locate_patch(spacing_twice, "x = 1", "x = 2")
    assert isinstance(problem, PatchProblem)
    assert problem.count == 2


def test_code_characters_are_never_forgiven() -> None:
    # Different identifier, different argument, different case. None is a
    # substring of the real line, so no tier may reach any of them.
    for quoted in (
        "total_x = sum_money(x)",
        "subtotal = sum_money(y)",
        "SUBTOTAL = SUM_MONEY(x)",
    ):
        problem = locate_patch("subtotal = sum_money(x)\n", quoted, "z = 1")
        assert isinstance(problem, PatchProblem), quoted
        assert problem.count == 0


def test_a_partial_line_quote_that_is_not_exact_stays_refused() -> None:
    """Fuzzy tiers match whole lines only: a replacement must never land
    inside an expression the model did not quote in full."""
    problem = locate_patch("value = compute(a, b)\n", "compute(a,b)", "compute(a)")
    assert isinstance(problem, PatchProblem)


def test_blank_lines_are_not_padded_when_reindenting() -> None:
    text = "def f():\n        a = 1\n\n        b = 2\n"
    match = locate_patch(text, "    a = 1\n\n    b = 2", "    a = 9\n\n    b = 8")
    assert isinstance(match, PatchMatch)
    updated = _apply(text, match)
    assert updated == "def f():\n        a = 9\n\n        b = 8\n"
    assert "    \n" not in updated  # the blank line stayed genuinely blank


def test_an_empty_quote_is_a_problem_not_a_match() -> None:
    assert isinstance(locate_patch("a = 1\n", "", "b = 2"), PatchProblem)
