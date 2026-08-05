"""A repair that hands back worse work than it received has to say so.

Measured end to end with Grok: blocking findings went 4 -> 2 -> 2 on the
build, 1 -> 2 -> 1 on the first repair, and 1 -> 1 -> 12 on the second, which
spent its retry budget mid-regression and surfaced a card reporting thirteen
problems. The card had no memory that the changeset it started from had one,
so the only signal that anything had gone backwards was a bigger number.

Warned, not auto-rejected: a repair that fixes a masking defect legitimately
uncovers problems that were always there, which is a judgement about the
project's code rather than something a counter can settle.
"""

from __future__ import annotations

from typing import Any

from waqil_api.control_plane import _note_regression, blocking_count


def _verification(count: int) -> dict[str, Any]:
    return {
        "errors": [
            {"rung": "typecheck", "path": f"app/f{i}.py", "error": f"F821: undefined name x{i}"}
            for i in range(count)
        ]
    }


def _reason(count: int) -> str:
    return (
        f"{count} problem(s) would stop this project working — app/rules.py: "
        "call-arg: Too many arguments. Send a follow-up to fix it, or reject."
    )


def test_blocking_count_reads_back_the_hosts_own_wording() -> None:
    assert blocking_count(_reason(13)) == 13
    assert blocking_count(_reason(1)) == 1
    assert blocking_count("something else entirely") == 0
    assert blocking_count("") == 0
    assert blocking_count(None) == 0  # type: ignore[arg-type]


def test_a_worse_repair_is_named_with_both_numbers() -> None:
    noted = _note_regression(_reason(13), prior=1, verification=_verification(13))
    assert noted is not None
    assert "worse than the changeset it started from" in noted
    assert "which had 1" in noted
    # And it points at the way back, since the earlier card is still pending.
    assert "leaves that earlier one pending" in noted


def test_progress_and_parity_are_not_flagged() -> None:
    # Fewer findings than it inherited: real progress, no warning.
    assert "worse" not in str(_note_regression(_reason(2), prior=7, verification=_verification(2)))
    # The same count is not a regression either.
    assert "worse" not in str(_note_regression(_reason(3), prior=3, verification=_verification(3)))


def test_a_clean_repair_stays_clean() -> None:
    """No blocking reason means an Approve button; nothing may add prose to it."""
    assert _note_regression(None, prior=9, verification={"errors": []}) is None


def test_a_first_build_has_nothing_to_regress_against() -> None:
    """prior=0 is a build that carried no changeset, not a perfect one."""
    noted = _note_regression(_reason(5), prior=0, verification=_verification(5))
    assert noted == _reason(5)


def test_only_blocking_findings_count_toward_the_comparison() -> None:
    """Warnings never blocked, so they must not manufacture a regression."""
    verification = {
        "errors": [
            {"rung": "runtime", "path": "app/main.py", "error": "GET /x failed: HTTP 500"},
            {"rung": "runtime", "path": "app/main.py", "error": "slow import"},
        ]
    }
    # Neither runtime finding is provable, so the blocking count is 0.
    assert "worse" not in str(_note_regression(_reason(2), prior=1, verification=verification))
