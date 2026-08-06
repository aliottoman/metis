"""Factual integrity for customer answers.

The failure this pins: an account whose entire record was one recorded win
produced an answer claiming 4× faster deployment, 60% lower cost, 95%+
adoption, a multi-year agreement, and a quotation from its CTO. None of it
was in the record, and nothing stopped it — the account's context arrived as
one prose block, so no claim could be cited, and the grounding gate measures
citations against retrieved evidence it never had.
"""
from __future__ import annotations

import pytest

from waqil_api.control_plane import _numbers_in, _source_display, _unsupported_claims

# The DynaAI record as it actually stands: one win, nothing else.
DYNAAI_EVIDENCE = (
    "Win: Gemma4 2xH200s DAC. Yearly ARR: $110,000. "
    "Shape: Model Import DAC (2xH200). Services: Generative AI Services, DAC"
)

FABRICATED_ANSWER = (
    "DynaAI achieved 4× faster model deployment [1] and a 60% reduction in "
    "infrastructure costs, with 95%+ adoption across teams. They signed a "
    'multi-year agreement worth $110,000 annually. Their CTO said, "Metis '
    'transformed how our engineering organisation ships models to production."'
)


def test_every_fabricated_claim_is_caught() -> None:
    found = _unsupported_claims(FABRICATED_ANSWER, DYNAAI_EVIDENCE)
    assert "60%" in found
    assert "95%" in found
    assert "4×" in found
    assert "multi-year" in found
    assert any(item.startswith("a quotation") for item in found)


def test_an_answer_within_the_record_passes_clean() -> None:
    truthful = (
        "DynaAI runs a Gemma4 2xH200s Model Import DAC [1], recorded at "
        "$110,000 yearly ARR [1], covering Generative AI Services and DAC [1]."
    )
    assert _unsupported_claims(truthful, DYNAAI_EVIDENCE) == []


@pytest.mark.parametrize(
    "answer",
    [
        "They run a 2xH200 shape [1].",  # an identifier, not a multiplier
        "1. First point\n2. Second point",  # ordinals
        "See [1] and [2] for detail.",  # citation markers
        "The DAC is live and the team is happy.",  # ordinary prose
    ],
)
def test_ordinary_answers_are_never_flagged(answer: str) -> None:
    assert _unsupported_claims(answer, DYNAAI_EVIDENCE) == []


def test_digits_inside_identifiers_are_not_known_quantities() -> None:
    """"Gemma4" must not make 4 a supported figure, or "4× faster" reads as
    grounded in a record that says nothing of the kind."""
    assert 4.0 not in _numbers_in("Gemma4 2xH200s")
    assert 110_000.0 in _numbers_in("Yearly ARR: $110,000")


def test_money_scales_are_normalized() -> None:
    assert _unsupported_claims("worth $110,000 [1]", DYNAAI_EVIDENCE) == []
    assert _unsupported_claims("worth $2M [1]", DYNAAI_EVIDENCE) == ["$2M"]


def test_no_evidence_means_no_claim_check() -> None:
    """Without an account record there is nothing to check against, and every
    figure would be reported as unsupported."""
    assert _unsupported_claims(FABRICATED_ANSWER, "") == []


def test_customer_sources_display_as_account_and_record() -> None:
    display = _source_display(
        {
            "provider": "customer",
            "rel_path": "DynaAI#win_abc123",
            "symbol": "Gemma4 2xH200s DAC",
        }
    )
    assert display == "DynaAI › Gemma4 2xH200s DAC"


async def test_evidence_turns_the_record_into_citable_sources() -> None:
    """Wins were loaded into the account detail and then never reached the
    model — the omission at the root of the fabrication."""
    from waqil_api.contracts import (
        CustomerAccountDetailV1,
        CustomerAccountV1,
        CustomerWinV1,
    )
    from waqil_api.customer_intelligence import CustomerIntelligenceService

    now = "2026-08-01T00:00:00Z"
    detail = CustomerAccountDetailV1(
        account=CustomerAccountV1(
            id="cust_" + "0" * 20, name="DynaAI", created_at=now, updated_at=now
        ),
        wins=[
            CustomerWinV1(
                id="win_1",
                account_id="cust_" + "0" * 20,
                title="Gemma4 2xH200s DAC",
                services=["Generative AI Services", "DAC"],
                dac_shape="Model Import DAC (2xH200)",
                yearly_arr=110_000.0,
                created_at=now,
                updated_at=now,
            )
        ],
    )
    service = CustomerIntelligenceService.__new__(CustomerIntelligenceService)

    async def account(_: str) -> CustomerAccountDetailV1:
        return detail

    service.account = account  # type: ignore[method-assign]
    evidence = await service.evidence("cust_" + "0" * 20)
    assert len(evidence) == 1
    win = evidence[0]
    assert win.provider == "customer"
    assert win.source_label == "Recorded win"
    assert win.score == 1.0
    assert "110,000" in win.text
    assert "Model Import DAC (2xH200)" in win.text
    # And the figure it carries is exactly the one the claim gate will accept.
    assert _unsupported_claims("ARR is $110,000 [1].", win.text) == []
