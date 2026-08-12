"""What a failed coding step and a token count must tell us afterwards.

A live Logivity session recorded nine `editor:failed` events with empty
message fields. A failed edit and a refused edit looked identical from the
outside, so three expensive runs could not say why the model wrote nothing.
The same session reported 496,623 input tokens for 12,182 output in one round,
and nothing in the record could say whether that was context being re-sent
every iteration or read back from a cache.

These tests hold the host half of both answers: the durable event carries the
sidecar's diagnostics through unchanged, and a cumulative usage snapshot is
published with the delta that makes compounding visible.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from waqil_api.coding_contracts import CodingEventV1, CodingUsageV1


def _event(**overrides: Any) -> CodingEventV1:
    payload: dict[str, Any] = {
        "sessionId": "coding_1",
        "cursor": 1,
        "type": "tool",
        "status": "tool_call_failed",
        "tool": "editor",
        "occurredAt": datetime.now(UTC).isoformat(),
    }
    payload.update(overrides)
    return CodingEventV1.model_validate(payload)


def _session(session_id: str = "coding_1") -> Any:
    """Only what the emitter reads.

    A full CodingSessionV1 validates its workspace digest against a real file
    manifest; building one here would test the snapshot contract rather than
    the diagnostics path this file is about.
    """

    return SimpleNamespace(id=session_id, run_id="run_1", conversation_id="conv_1")


class _Sink:
    """Captures exactly what would become a durable run event."""

    def __init__(self) -> None:
        self.emitted: list[tuple[str, dict[str, Any]]] = []

    async def emit(
        self, run_id: str, conversation_id: str, type: str, payload: dict[str, Any]
    ) -> None:
        self.emitted.append((type, payload))


def _coordinator(sink: _Sink) -> Any:
    from waqil_api.project_coding_engine import ProjectCodingCoordinator

    coordinator = ProjectCodingCoordinator.__new__(ProjectCodingCoordinator)
    coordinator.events = sink  # type: ignore[attr-defined]
    coordinator._usage_marks = {}  # type: ignore[attr-defined]
    return coordinator


# ── The failure reason survives to the durable event ───────────────────────


@pytest.mark.asyncio
async def test_a_failed_editor_reaches_the_timeline_with_its_reason() -> None:
    sink = _Sink()
    coordinator = _coordinator(sink)

    await coordinator._emit_coding_event(
        _session(),
        _event(
            message="old_text did not match the file at README.md",
            failureClass="EditApplyError",
            path="README.md",
            iteration=3,
        ),
    )

    kind, payload = sink.emitted[-1]
    assert kind == "project.coding_event"
    assert payload["tool"] == "editor"
    assert payload["message"] == "old_text did not match the file at README.md"
    assert payload["failure_class"] == "EditApplyError"
    assert payload["path"] == "README.md"
    assert payload["iteration"] == 3


@pytest.mark.asyncio
async def test_a_failure_without_sdk_text_is_still_never_blank() -> None:
    # The sidecar substitutes an explicit sentence; the host must not drop it
    # back to "" on the way to the timeline.
    sink = _Sink()
    coordinator = _coordinator(sink)

    await coordinator._emit_coding_event(
        _session(),
        _event(message="SDK returned a failed tool result without diagnostic text"),
    )

    _, payload = sink.emitted[-1]
    assert payload["message"].startswith("SDK returned a failed tool result")


def test_the_event_contract_bounds_the_diagnostic_fields() -> None:
    # Redaction happens in the sidecar; the host still refuses to persist an
    # unbounded message or an invented failure class.
    with pytest.raises(ValueError):
        _event(message="x" * 5_000)
    with pytest.raises(ValueError):
        _event(failureClass="y" * 200)


# ── Usage says whether context is compounding ──────────────────────────────


@pytest.mark.asyncio
async def test_a_cumulative_snapshot_publishes_its_delta() -> None:
    sink = _Sink()
    coordinator = _coordinator(sink)
    session = _session()

    for cursor, (input_tokens, output_tokens) in enumerate(
        [(49_307, 802), (129_764, 4_938)], start=1
    ):
        await coordinator._emit_coding_event(
            session,
            _event(
                cursor=cursor,
                type="usage",
                status="usage",
                tool="",
                usageScope="cumulative",
                usage={
                    "inputTokens": input_tokens,
                    "outputTokens": output_tokens,
                    "totalTokens": input_tokens + output_tokens,
                    "requests": cursor,
                    "cacheReadTokens": 10_000 * cursor,
                },
            ),
        )

    first = sink.emitted[0][1]
    second = sink.emitted[1][1]

    assert first["usage_scope"] == "cumulative"
    # The first reading is its own increase -- reporting zero would claim the
    # context arrived free.
    assert first["usage_delta"]["first_snapshot"] is True
    assert first["usage_delta"]["inputTokens"] == 49_307

    assert second["usage_delta"]["first_snapshot"] is False
    assert second["usage_delta"]["inputTokens"] == 129_764 - 49_307
    assert second["usage_delta"]["outputTokens"] == 4_938 - 802
    assert second["usage_delta"]["cacheReadTokens"] == 10_000
    # The cumulative total is published unchanged beside the delta.
    assert second["usage"]["totalTokens"] == 129_764 + 4_938


@pytest.mark.asyncio
async def test_a_restarted_session_does_not_difference_against_its_ancestor() -> None:
    """restartWithModel opens a new session id; its series starts fresh.

    Differencing a recovery child against its parent's larger totals would
    otherwise clamp every delta to zero and hide the child's real cost.
    """

    sink = _Sink()
    coordinator = _coordinator(sink)
    snapshot = {
        "inputTokens": 500_000,
        "outputTokens": 12_000,
        "totalTokens": 512_000,
        "requests": 1,
    }

    await coordinator._emit_coding_event(
        _session("coding_parent"),
        _event(
            cursor=1, type="usage", tool="", usageScope="cumulative", usage=snapshot
        ),
    )
    await coordinator._emit_coding_event(
        _session("coding_recovery_child"),
        _event(
            cursor=1,
            type="usage",
            tool="",
            usageScope="cumulative",
            usage={
                "inputTokens": 40_000,
                "outputTokens": 900,
                "totalTokens": 40_900,
                "requests": 1,
            },
        ),
    )

    child = sink.emitted[-1][1]
    assert child["usage_delta"]["first_snapshot"] is True
    assert child["usage_delta"]["totalTokens"] == 40_900


@pytest.mark.asyncio
async def test_a_counter_that_goes_backwards_never_reports_a_negative_delta() -> None:
    sink = _Sink()
    coordinator = _coordinator(sink)
    session = _session()

    for cursor, total in enumerate([100_000, 10_000], start=1):
        await coordinator._emit_coding_event(
            session,
            _event(
                cursor=cursor,
                type="usage",
                tool="",
                usageScope="cumulative",
                usage={
                    "inputTokens": total,
                    "outputTokens": 0,
                    "totalTokens": total,
                    "requests": 1,
                },
            ),
        )

    assert sink.emitted[-1][1]["usage_delta"]["totalTokens"] == 0


@pytest.mark.asyncio
async def test_an_event_without_usage_publishes_no_delta() -> None:
    sink = _Sink()
    coordinator = _coordinator(sink)

    await coordinator._emit_coding_event(_session(), _event())

    assert sink.emitted[-1][1]["usage_delta"] is None
    assert sink.emitted[-1][1]["usage"] is None


def test_absent_cache_counters_stay_absent_rather_than_becoming_zero() -> None:
    usage = CodingUsageV1.model_validate(
        {"inputTokens": 10, "outputTokens": 2, "totalTokens": 12, "requests": 1}
    )

    # None means "the provider did not report it". Zero would be a claim.
    assert usage.cache_read_tokens is None
    assert usage.cache_write_tokens is None
    assert usage.model_dump(mode="json")["cacheReadTokens"] is None


def test_cache_counters_round_trip_through_the_wire_contract() -> None:
    usage = CodingUsageV1.model_validate(
        {
            "inputTokens": 49_307,
            "outputTokens": 802,
            "totalTokens": 50_109,
            "requests": 3,
            "cacheReadTokens": 40_000,
            "cacheWriteTokens": 1_200,
            "costUsd": 0.42,
        }
    )

    assert usage.cache_read_tokens == 40_000
    assert usage.cache_write_tokens == 1_200
    assert usage.cost_usd == 0.42
