from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable

from .contracts import RiskLevel


class PolicyPermission(StrEnum):
    """Canonical capabilities understood by the trusted control plane."""

    NO_NETWORK = "network:none"
    CONVERSATION_RESPONSE = "conversation:respond"
    READ_RUN_INPUTS = "read:run-inputs"
    READ_GRANTED_WORKSPACE = "read:granted-workspace"
    WRITE_RUN_ARTIFACTS = "write:run-artifacts"
    SANDBOX_EXECUTION = "execute:sandboxed"
    MODEL_BROKER = "model:broker"
    TOOL_DEFINITION = "tool:define"
    TOOL_ACTIVATION = "tool:activate"
    DEPENDENCY_CHANGE = "dependency:change"
    NETWORK_ACCESS = "network:access"
    WIDER_FILESYSTEM = "filesystem:wider"
    UNSANDBOXED_EXECUTION = "execute:unsandboxed"
    SECRET_ACCESS = "secrets:read"
    SYSTEM_WRITE = "write:system"
    PRIVILEGE_ESCALATION = "privilege:escalate"


class ExecutionBoundary(StrEnum):
    NONE = "none"
    SANDBOXED = "sandboxed"
    UNSANDBOXED = "unsandboxed"


class PolicyDisposition(StrEnum):
    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


class PolicyViolation(RuntimeError):
    """The requested action violates a non-overridable policy rule."""


class PolicyApprovalRequired(PolicyViolation):
    """The action is valid only after an explicit human approval."""


_RISK_RANK = {
    RiskLevel.R0: 0,
    RiskLevel.R1: 1,
    RiskLevel.R2: 2,
    RiskLevel.R3: 3,
    RiskLevel.R4: 4,
}

_MINIMUM_RISK = {
    PolicyPermission.NO_NETWORK: RiskLevel.R0,
    PolicyPermission.CONVERSATION_RESPONSE: RiskLevel.R0,
    PolicyPermission.READ_RUN_INPUTS: RiskLevel.R1,
    PolicyPermission.READ_GRANTED_WORKSPACE: RiskLevel.R1,
    PolicyPermission.WRITE_RUN_ARTIFACTS: RiskLevel.R2,
    PolicyPermission.SANDBOX_EXECUTION: RiskLevel.R2,
    PolicyPermission.MODEL_BROKER: RiskLevel.R2,
    PolicyPermission.TOOL_DEFINITION: RiskLevel.R3,
    PolicyPermission.TOOL_ACTIVATION: RiskLevel.R3,
    PolicyPermission.DEPENDENCY_CHANGE: RiskLevel.R3,
    PolicyPermission.NETWORK_ACCESS: RiskLevel.R3,
    PolicyPermission.WIDER_FILESYSTEM: RiskLevel.R3,
    PolicyPermission.UNSANDBOXED_EXECUTION: RiskLevel.R4,
    PolicyPermission.SECRET_ACCESS: RiskLevel.R4,
    PolicyPermission.SYSTEM_WRITE: RiskLevel.R4,
    PolicyPermission.PRIVILEGE_ESCALATION: RiskLevel.R4,
}

_CATEGORICALLY_DENIED = frozenset(
    {
        PolicyPermission.UNSANDBOXED_EXECUTION,
        PolicyPermission.SECRET_ACCESS,
        PolicyPermission.SYSTEM_WRITE,
        PolicyPermission.PRIVILEGE_ESCALATION,
    }
)


def _parse_permission(value: str) -> PolicyPermission | None:
    normalized = value.strip().lower()
    aliases = {
        permission.value: permission for permission in PolicyPermission
    } | {
        "activate:tool": PolicyPermission.TOOL_ACTIVATION,
        "network:disabled": PolicyPermission.NO_NETWORK,
        "network:off": PolicyPermission.NO_NETWORK,
        "read:uploads": PolicyPermission.READ_RUN_INPUTS,
        "sandbox:execute": PolicyPermission.SANDBOX_EXECUTION,
    }
    if normalized in aliases:
        return aliases[normalized]
    if normalized.startswith("secret") or normalized.startswith("credential"):
        return PolicyPermission.SECRET_ACCESS
    if "system" in normalized and normalized.startswith("write"):
        return PolicyPermission.SYSTEM_WRITE
    if normalized.startswith("privilege") or normalized.startswith("capabilities:"):
        return PolicyPermission.PRIVILEGE_ESCALATION
    if normalized.startswith("shell:") or (
        normalized.startswith("execute:")
        and any(token in normalized for token in ("host", "unsandboxed"))
    ):
        return PolicyPermission.UNSANDBOXED_EXECUTION
    if normalized.startswith("model:"):
        return PolicyPermission.MODEL_BROKER
    if normalized.startswith("network:"):
        return PolicyPermission.NETWORK_ACCESS
    if normalized.startswith("dependency:") or normalized.startswith("dependencies:"):
        return PolicyPermission.DEPENDENCY_CHANGE
    if normalized.startswith("read:") or normalized.startswith("write:"):
        return PolicyPermission.WIDER_FILESYSTEM
    return None


@dataclass(frozen=True, slots=True)
class PolicyRequest:
    action: str
    declared_risk: RiskLevel
    permissions: frozenset[PolicyPermission]
    unknown_permissions: tuple[str, ...] = ()
    execution_boundary: ExecutionBoundary = ExecutionBoundary.NONE
    approval_granted: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "declared_risk", RiskLevel(self.declared_risk))
        object.__setattr__(
            self, "execution_boundary", ExecutionBoundary(self.execution_boundary)
        )

    @classmethod
    def from_raw(
        cls,
        *,
        action: str,
        declared_risk: RiskLevel | str,
        permissions: Iterable[str] = (),
        additional_permissions: Iterable[PolicyPermission] = (),
        execution_boundary: ExecutionBoundary = ExecutionBoundary.NONE,
        approval_granted: bool = False,
    ) -> PolicyRequest:
        parsed = set(additional_permissions)
        unknown: list[str] = []
        for raw in permissions:
            permission = _parse_permission(raw)
            if permission is None:
                unknown.append(raw)
            else:
                parsed.add(permission)
        return cls(
            action=action,
            declared_risk=RiskLevel(declared_risk),
            permissions=frozenset(parsed),
            unknown_permissions=tuple(sorted(set(unknown))),
            execution_boundary=execution_boundary,
            approval_granted=approval_granted,
        )


@dataclass(frozen=True, slots=True)
class PolicyOutcome:
    action: str
    disposition: PolicyDisposition
    declared_risk: RiskLevel
    required_risk: RiskLevel
    reasons: tuple[str, ...] = ()

    def enforce(self) -> None:
        detail = "; ".join(self.reasons) or self.disposition.value
        if self.disposition == PolicyDisposition.DENY:
            raise PolicyViolation(f"policy denied {self.action}: {detail}")
        if self.disposition == PolicyDisposition.REQUIRE_APPROVAL:
            raise PolicyApprovalRequired(
                f"policy requires explicit approval for {self.action}: {detail}"
            )

    def require_approval(self) -> None:
        if self.disposition == PolicyDisposition.REQUIRE_APPROVAL:
            return
        self.enforce()
        raise PolicyViolation(
            f"policy expected an approval boundary for {self.action}, but allowed it"
        )


class PolicyEngine:
    """Fail-closed R0-R4 policy evaluation independent of model output."""

    def evaluate(self, request: PolicyRequest) -> PolicyOutcome:
        required_risk = RiskLevel.R0
        if request.permissions:
            required_risk = max(
                (_MINIMUM_RISK[item] for item in request.permissions),
                key=lambda level: _RISK_RANK[level],
            )

        reasons: list[str] = []
        forbidden = sorted(
            (item.value for item in request.permissions & _CATEGORICALLY_DENIED)
        )
        if request.declared_risk == RiskLevel.R4:
            reasons.append("R4 actions are categorically denied")
        if request.execution_boundary == ExecutionBoundary.UNSANDBOXED:
            reasons.append("unsandboxed execution is categorically denied")
        if forbidden:
            reasons.append("categorically denied permissions: " + ", ".join(forbidden))
        if request.unknown_permissions:
            reasons.append(
                "unknown permissions fail closed: "
                + ", ".join(request.unknown_permissions)
            )
        if (
            PolicyPermission.SANDBOX_EXECUTION in request.permissions
            and request.execution_boundary != ExecutionBoundary.SANDBOXED
        ):
            reasons.append("sandbox execution requires a verified sandbox boundary")
        if _RISK_RANK[request.declared_risk] < _RISK_RANK[required_risk]:
            reasons.append(
                f"declared {request.declared_risk} is below required {required_risk}"
            )
        if reasons:
            return PolicyOutcome(
                action=request.action,
                disposition=PolicyDisposition.DENY,
                declared_risk=request.declared_risk,
                required_risk=RiskLevel.R4
                if forbidden
                or request.declared_risk == RiskLevel.R4
                or request.execution_boundary == ExecutionBoundary.UNSANDBOXED
                else required_risk,
                reasons=tuple(reasons),
            )

        if (
            request.declared_risk == RiskLevel.R3 or required_risk == RiskLevel.R3
        ) and not request.approval_granted:
            return PolicyOutcome(
                action=request.action,
                disposition=PolicyDisposition.REQUIRE_APPROVAL,
                declared_risk=request.declared_risk,
                required_risk=RiskLevel.R3,
                reasons=("R3 actions require explicit human approval",),
            )

        return PolicyOutcome(
            action=request.action,
            disposition=PolicyDisposition.ALLOW,
            declared_risk=request.declared_risk,
            required_risk=required_risk,
        )
