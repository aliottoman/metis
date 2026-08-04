"""Money arithmetic that never touches binary floating point.

Two failure families motivate every rule here, both observed in real
generated builds: an exact one-cent tolerance check failing on float drift,
and missing values coerced to zero so "unknown" silently became "wrong".
Hence: Decimal end to end, quantized to cents, and None stays None — a
missing number is a finding to report, not a zero to add.
"""

from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal, InvalidOperation

CENT = Decimal("0.01")


def to_money(value: object) -> Decimal | None:
    """A value as a Decimal quantized to cents, or None when absent/unreadable.

    Floats pass through str() so 0.1 becomes Decimal("0.1"), not the binary
    neighbour the float actually stores.
    """
    if value is None:
        return None
    if isinstance(value, Decimal):
        try:
            return value.quantize(CENT)
        except InvalidOperation:
            return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return Decimal(text).quantize(CENT)
    except (InvalidOperation, ValueError):
        return None


def sum_money(values: Iterable[object]) -> Decimal | None:
    """A total that propagates missing: any None value makes the sum None."""
    total = Decimal("0")
    for value in values:
        money = to_money(value)
        if money is None:
            return None
        total += money
    return total.quantize(CENT)


def within_cents(a: object, b: object, cents: int = 1) -> bool | None:
    """Whether two amounts agree within an inclusive cent tolerance.

    None when either side is missing — the caller reports a missing value,
    which is a different finding from a mismatched one.
    """
    left, right = to_money(a), to_money(b)
    if left is None or right is None:
        return None
    return abs(left - right) <= CENT * cents


def within_percent(
    expected: object, actual: object, percent: object, *, inclusive: bool = True
) -> bool | None:
    """Whether actual is within percent of expected, Decimal end to end.

    Inclusive means exactly-at-the-boundary passes (a 5% policy accepts a
    5.00% variance and rejects 5% plus one cent). None when a side is missing.
    """
    expected_money, actual_money = to_money(expected), to_money(actual)
    if expected_money is None or actual_money is None:
        return None
    try:
        allowance = abs(expected_money) * Decimal(str(percent)) / Decimal("100")
    except (InvalidOperation, ValueError):
        return None
    difference = abs(actual_money - expected_money)
    return difference <= allowance if inclusive else difference < allowance
