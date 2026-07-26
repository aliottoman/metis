from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from waqil_api.contracts import RiskLevel
from waqil_api.policy import (
    ExecutionBoundary,
    PolicyDisposition,
    PolicyEngine,
    PolicyPermission,
    PolicyRequest,
    PolicyViolation,
)


def _evaluate(**kwargs):
    return PolicyEngine().evaluate(PolicyRequest.from_raw(**kwargs))


def _wait(client: TestClient, run_id: str, statuses: set[str]) -> dict:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        run = client.get(f"/api/v1/runs/{run_id}").json()
        if run["status"] in statuses:
            return run
        time.sleep(0.02)
    raise AssertionError(f"run {run_id} did not reach {statuses}")


def test_r0_direct_and_r2_sandbox_flows_are_allowed() -> None:
    direct = _evaluate(
        action="conversation.respond",
        declared_risk=RiskLevel.R0,
        additional_permissions=(PolicyPermission.CONVERSATION_RESPONSE,),
    )
    assert direct.disposition == PolicyDisposition.ALLOW

    sandbox = _evaluate(
        action="tool.execute",
        declared_risk=RiskLevel.R2,
        permissions=("network:none", "read:run-inputs", "write:run-artifacts"),
        additional_permissions=(PolicyPermission.SANDBOX_EXECUTION,),
        execution_boundary=ExecutionBoundary.SANDBOXED,
    )
    assert sandbox.disposition == PolicyDisposition.ALLOW
    assert sandbox.required_risk == RiskLevel.R2


def test_permissions_cannot_exceed_the_declared_risk() -> None:
    outcome = _evaluate(
        action="underdeclared-write",
        declared_risk=RiskLevel.R1,
        permissions=("write:run-artifacts",),
    )
    assert outcome.disposition == PolicyDisposition.DENY
    assert outcome.required_risk == RiskLevel.R2
    with pytest.raises(PolicyViolation, match="below required R2"):
        outcome.enforce()


@pytest.mark.parametrize(
    "permission",
    ["tool:activate", "network:access", "dependency:change", "filesystem:wider"],
)
def test_r3_capabilities_require_explicit_approval(permission: str) -> None:
    pending = _evaluate(
        action="r3-action",
        declared_risk=RiskLevel.R3,
        permissions=(permission,),
    )
    assert pending.disposition == PolicyDisposition.REQUIRE_APPROVAL
    pending.require_approval()

    approved = _evaluate(
        action="r3-action",
        declared_risk=RiskLevel.R3,
        permissions=(permission,),
        approval_granted=True,
    )
    assert approved.disposition == PolicyDisposition.ALLOW


@pytest.mark.parametrize(
    "permission",
    [
        "execute:unsandboxed",
        "secrets:read",
        "write:system",
        "privilege:escalate",
    ],
)
def test_r4_capabilities_are_denied_even_with_approval(permission: str) -> None:
    outcome = _evaluate(
        action="forbidden-action",
        declared_risk=RiskLevel.R4,
        permissions=(permission,),
        approval_granted=True,
    )
    assert outcome.disposition == PolicyDisposition.DENY
    assert outcome.required_risk == RiskLevel.R4


def test_unsandboxed_boundaries_and_unknown_permissions_fail_closed() -> None:
    unsandboxed = _evaluate(
        action="host-execution",
        declared_risk=RiskLevel.R2,
        additional_permissions=(PolicyPermission.SANDBOX_EXECUTION,),
        execution_boundary=ExecutionBoundary.UNSANDBOXED,
        approval_granted=True,
    )
    assert unsandboxed.disposition == PolicyDisposition.DENY
    assert "unsandboxed execution" in "; ".join(unsandboxed.reasons)

    unknown = _evaluate(
        action="unknown-capability",
        declared_risk=RiskLevel.R3,
        permissions=("telepathy:write",),
        approval_granted=True,
    )
    assert unknown.disposition == PolicyDisposition.DENY
    assert unknown.required_risk == RiskLevel.R0
    assert "unknown permissions fail closed" in "; ".join(unknown.reasons)


def test_root_graph_applies_direct_sandbox_and_activation_policy_gates(
    client: TestClient,
) -> None:
    conversation = client.post("/api/v1/conversations", json={}).json()
    direct = client.post(
        f"/api/v1/conversations/{conversation['id']}/messages",
        json={"content": "Hello", "attachment_ids": []},
    ).json()
    assert _wait(client, direct["run_id"], {"completed", "failed"})["status"] == (
        "completed"
    )
    direct_events = client.get(
        f"/api/v1/runs/{direct['run_id']}/events?after=0"
    ).text
    assert '"action":"conversation.respond"' in direct_events
    assert '"disposition":"allow"' in direct_events

    candidate = client.post(
        f"/api/v1/conversations/{conversation['id']}/messages",
        json={"content": "Build an architecture diagram", "attachment_ids": []},
    ).json()
    waiting = _wait(client, candidate["run_id"], {"awaiting_approval", "failed"})
    assert waiting["status"] == "awaiting_approval", waiting
    recovery = client.get("/api/v1/runs?status=awaiting_approval").json()
    approval = next(
        item["approval"]
        for item in recovery
        if item["run"]["id"] == candidate["run_id"]
    )
    assert approval["risk_level"] == "R3"
    assert "tool:activate" in approval["permissions"]
    resumed = client.post(
        f"/api/v1/runs/{candidate['run_id']}/decisions",
        json={
            "approval_id": approval["id"],
            "decision": "approve",
            "reason": "Policy integration test",
        },
    )
    assert resumed.status_code == 200
    completed = _wait(client, candidate["run_id"], {"completed", "failed"})
    assert completed["status"] == "completed", completed
    events = client.get(
        f"/api/v1/runs/{candidate['run_id']}/events?after=0"
    ).text
    assert '"action":"tool.execute"' in events
    assert '"action":"tool.activate"' in events
    assert '"disposition":"require_approval"' in events
    assert '"approval_granted":true' in events
