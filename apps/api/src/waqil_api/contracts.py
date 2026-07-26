from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


def _as_text(value: Any) -> str:
    """Coerce a value to text — local models sometimes return a JSON object where a
    descriptive string was asked for. Serialize rather than crash the run."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AssetStatus(StrEnum):
    UNCONFIGURED = "unconfigured"
    NEEDS_APPROVAL = "needs_approval"
    READY = "ready"
    STARTING = "starting"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"


class ToolState(StrEnum):
    DRAFT = "draft"
    QUARANTINED = "quarantined"
    EVALUATED = "evaluated"
    APPROVED = "approved"
    ACTIVE = "active"
    DEPRECATED = "deprecated"


class ProposalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    DRAFT = "draft"


class Decision(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"
    DRAFT = "draft"


class RiskLevel(StrEnum):
    R0 = "R0"
    R1 = "R1"
    R2 = "R2"
    R3 = "R3"
    R4 = "R4"


class ArtifactRefV1(Contract):
    id: str
    filename: str
    media_type: str
    size: int = Field(ge=0)
    sha256: str
    download_url: str


class UploadV1(Contract):
    id: str
    filename: str
    media_type: str
    size: int = Field(ge=0)
    sha256: str
    created_at: datetime


class AssetV1(Contract):
    id: str
    name: str = Field(min_length=1, max_length=120)
    summary: str = Field(min_length=1, max_length=240)
    category: str = Field(min_length=1, max_length=60)
    tags: list[str] = Field(default_factory=list, max_length=16)
    framework: str | None = Field(default=None, max_length=60)
    entrypoint: str | None = Field(default=None, max_length=240)
    env_keys: list[str] = Field(default_factory=list, max_length=64)
    launch_configured: bool = False
    launch_approved: bool = False
    launch_command: list[str] = Field(default_factory=list, max_length=32)
    status: AssetStatus
    url: str | None = None


class AssetStartV1(Contract):
    env: dict[str, str] = Field(default_factory=dict, max_length=64)

    @field_validator("env")
    @classmethod
    def bounded_environment(cls, value: dict[str, str]) -> dict[str, str]:
        for key, item in value.items():
            if len(key) > 64 or len(item) > 16_384 or "\x00" in item:
                raise ValueError("asset environment contains an invalid key or value")
        return value


class AssetLogsV1(Contract):
    asset_id: str
    status: AssetStatus
    logs: str = Field(default="", max_length=65_536)
    truncated: bool = False
    return_code: int | None = None


class ConversationCreateV1(Contract):
    title: str | None = Field(default=None, max_length=200)


class ConversationV1(Contract):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime


class MessageCreateV1(Contract):
    content: str = Field(min_length=1, max_length=20_000)
    attachment_ids: list[str] = Field(default_factory=list, max_length=20)
    project_id: str | None = Field(default=None, pattern=r"^asset_[a-f0-9]{20}$")
    project_mode: Literal["grok_bootstrap_local", "grok_continuous"] | None = None
    knowledge_scope: Literal["auto", "notion"] = "auto"


class MessageV1(Contract):
    id: str
    conversation_id: str
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    attachment_ids: list[str] = Field(default_factory=list)
    run_id: str | None = None
    created_at: datetime


class MessageAcceptedV1(Contract):
    message_id: str
    run_id: str
    status: RunStatus


class RunV1(Contract):
    id: str
    conversation_id: str
    user_message_id: str
    status: RunStatus
    graph_schema_version: str
    cancel_requested: bool
    result: dict[str, Any] | None = None
    last_error: str | None = None
    created_at: datetime
    updated_at: datetime


class PlanningRequestV1(Contract):
    run_id: str
    conversation_id: str
    prompt: str
    attachment_ids: list[str] = Field(default_factory=list)
    untrusted_attachment_excerpt: str = Field(default="", max_length=12_000)
    untrusted_attachment_signals: list[
        Literal[
            "project_documentation",
            "software_components",
            "component_relationships",
            "deployment_configuration",
            "source_configuration",
        ]
    ] = Field(default_factory=list, max_length=5)
    attachment_excerpt_truncated: bool = False
    memories: list[str] = Field(default_factory=list)
    conversation_summary: str = Field(default="", max_length=8_000)
    recent_messages: list[dict[str, str]] = Field(default_factory=list, max_length=20)
    active_tools: list[dict[str, Any]] = Field(default_factory=list)
    # The catalog the planner may route to. Identity and intent only, never
    # capabilities; the host still disposes route and risk.
    tool_catalog: list[dict[str, Any]] = Field(default_factory=list, max_length=64)


class PlanStepV1(Contract):
    id: str
    title: str
    description: str
    kind: Literal["respond", "tool", "build_tool", "validate"]


class PlanEnvelopeV1(Contract):
    schema_version: Literal["1"] = "1"
    summary: str
    route: Literal["direct", "existing_tool", "tool_factory", "tool_definition"]
    tool_slug: str | None = None
    risk_level: RiskLevel = RiskLevel.R0
    steps: list[PlanStepV1] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class ModelRequestV1(Contract):
    role: Literal["planner", "coder", "reviewer"] = "planner"
    system_prompt: str
    user_prompt: str
    response_schema: dict[str, Any] | None = None
    temperature: float = Field(default=0.1, ge=0, le=2)


class ModelResultV1(Contract):
    model: str
    content: str
    structured: dict[str, Any] | None = None
    fallback: bool = False


# ── Project workspaces ───────────────────────────────────────────────────────


class ProjectWorkspaceV1(Contract):
    id: str
    name: str
    summary: str
    framework: str | None = None
    initialized: bool = False
    manifest_revision: int = Field(default=0, ge=0)
    file_count: int = Field(default=0, ge=0)
    metis_md_path: str = ".metis/METIS.md"
    updated_at: datetime | None = None


class ProjectOpenV1(Contract):
    mode: Literal["grok_bootstrap_local", "grok_continuous"]


class ConversationProjectV1(Contract):
    conversation_id: str
    project_id: str
    mode: Literal["grok_bootstrap_local", "grok_continuous"]
    updated_at: datetime


class ProjectBootstrapV1(Contract):
    summary: str = Field(max_length=2_000)
    architecture: list[str] = Field(default_factory=list, max_length=24)
    conventions: list[str] = Field(default_factory=list, max_length=24)
    important_paths: list[str] = Field(default_factory=list, max_length=48)
    verification: list[str] = Field(default_factory=list, max_length=24)
    risks: list[str] = Field(default_factory=list, max_length=24)


class ProjectToolCallV1(Contract):
    name: Literal[
        "list_files",
        "search_code",
        "read_file",
        "apply_patch",
        "create_file",
    ]
    arguments: dict[str, Any] = Field(default_factory=dict)


class ProjectAgentStepV1(Contract):
    status: Literal["tool", "complete"]
    response: str = Field(default="", max_length=40_000)
    tool_call: ProjectToolCallV1 | None = None
    learnings: list[str] = Field(default_factory=list, max_length=16)

    @model_validator(mode="after")
    def valid_project_step(self) -> "ProjectAgentStepV1":
        if self.status == "tool" and self.tool_call is None:
            raise ValueError("a project tool step requires tool_call")
        if self.status == "complete" and not self.response.strip():
            raise ValueError("a completed project step requires a response")
        return self


class DiagramCodeV1(Contract):
    """North's constrained source-code result for the architecture sandbox."""

    schema_version: Literal["1"] = "1"
    diagram_code: str = Field(min_length=1, max_length=100_000)

    @field_validator("diagram_code")
    @classmethod
    def validate_diagram_code(cls, value: str) -> str:
        if "\x00" in value or "\r" in value:
            raise ValueError("diagram_code must use LF line endings and contain no NUL bytes")
        if len(value.encode("utf-8")) > 100_000:
            raise ValueError("diagram_code exceeds 100000 UTF-8 bytes")
        return value


class ToolManifestV1(Contract):
    schema_version: Literal["1"] = "1"
    slug: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    name: str
    description: str
    version: str
    entrypoint: str
    runner_image: str | None = None
    risk_level: RiskLevel
    permissions: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    content_hash: str


class ToolInvocationV1(Contract):
    invocation_id: str
    run_id: str
    tool_slug: str
    tool_version: str
    tool_content_hash: str
    input_digest: str
    arguments: dict[str, Any]


class ToolResultV1(Contract):
    invocation_id: str
    status: Literal["succeeded", "failed", "cancelled"]
    output: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[ArtifactRefV1] = Field(default_factory=list)
    logs: str = ""
    error: str | None = None


class EvalCaseV1(Contract):
    id: str
    name: str
    input: dict[str, Any]
    expected_properties: list[str]


class EvalResultV1(Contract):
    case_id: str
    passed: bool
    checks: dict[str, bool] = Field(default_factory=dict)
    message: str = ""


class EvalReportV1(Contract):
    passed: bool
    score: float = Field(ge=0, le=1)
    results: list[EvalResultV1]
    static_checks: dict[str, bool] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class ApprovalRequestV1(Contract):
    id: str
    run_id: str
    action_id: str
    kind: Literal[
        "activate_tool",
        "define_tool",
        "activate_definition",
        "filesystem",
        "project_write",
        "network",
        "dependency",
    ]
    title: str
    summary: str
    risk_level: RiskLevel
    proposal_id: str | None = None
    tool_version_id: str | None = None
    input_digest: str
    permissions: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class ApprovalDecisionV1(Contract):
    approval_id: str | None = None
    decision: Decision
    reason: str | None = Field(default=None, max_length=2000)


class RecoverableRunV1(Contract):
    run: RunV1
    approval: ApprovalRequestV1 | None = None


class RunEventV1(Contract):
    id: str
    sequence: int = Field(ge=1)
    run_id: str
    thread_id: str
    checkpoint_id: str | None = None
    type: str
    timestamp: datetime
    payload: dict[str, Any] = Field(default_factory=dict)


class ToolVersionV1(Contract):
    id: str
    tool_id: str
    version: str
    state: ToolState
    content_hash: str
    manifest: ToolManifestV1
    eval_report: EvalReportV1 | None = None
    source_run_id: str | None = None
    created_at: datetime


class ToolSourceFileV1(Contract):
    path: str
    sha256: str
    size: int = Field(ge=0)
    content: str


class ToolVersionEvidenceV1(Contract):
    tool_id: str
    version_id: str
    state: ToolState
    content_hash: str
    manifest: ToolManifestV1
    eval_report: EvalReportV1 | None = None
    bundle_verified: bool
    files: list[ToolSourceFileV1] = Field(default_factory=list)
    evidence_truncated: bool = False
    compared_to_version_id: str | None = None
    source_diff: str = ""


class ToolV1(Contract):
    id: str
    slug: str
    name: str
    description: str
    active_version_id: str | None = None
    created_at: datetime
    updated_at: datetime


class ToolProposalV1(Contract):
    id: str
    tool_id: str
    tool_version_id: str
    source_run_id: str
    status: ProposalStatus
    risk_level: RiskLevel
    summary: str
    created_at: datetime
    decided_at: datetime | None = None


class ToolImprovementProposalV1(Contract):
    id: str
    source_run_id: str
    tool_id: str
    tool_version_id: str
    content_hash: str
    correction: str
    regression_eval: EvalCaseV1
    status: ProposalStatus
    created_at: datetime
    decision_reason: str | None = None
    decided_at: datetime | None = None
    outcome: Literal["revision_queued", "revision_activated", "rejected"] | None = None
    revision_request_id: str | None = None
    target_version_id: str | None = None


class ToolRevisionRequestV1(Contract):
    id: str
    proposal_id: str
    tool_id: str
    base_version_id: str
    base_content_hash: str
    correction: str
    regression_eval: EvalCaseV1
    status: Literal["queued"] = "queued"
    created_at: datetime


class ToolImprovementDecisionV1(Contract):
    decision: Literal["approve", "reject"]
    idempotency_key: str = Field(min_length=8, max_length=200)
    reason: str = Field(min_length=1, max_length=2000)
    target_version_id: str | None = None


class ToolImprovementDecisionResultV1(Contract):
    proposal: ToolImprovementProposalV1
    outcome: Literal["revision_queued", "revision_activated", "rejected"]
    revision_request: ToolRevisionRequestV1 | None = None
    activated_version_id: str | None = None
    prior_version_id: str | None = None


class ToolImprovementEvidenceV1(Contract):
    proposal: ToolImprovementProposalV1
    base_version: ToolVersionEvidenceV1
    eligible_revisions: list[ToolVersionEvidenceV1] = Field(default_factory=list)


class ProposalDecisionV1(Contract):
    reason: str | None = Field(default=None, max_length=2000)


class ToolVersionActivationV1(Contract):
    idempotency_key: str = Field(min_length=8, max_length=200)
    reason: str = Field(min_length=1, max_length=2000)


class MemoryProposalV1(Contract):
    id: str
    kind: Literal["user", "project", "skill"]
    content: str
    source_run_id: str | None = None
    confidence: float = Field(ge=0, le=1)
    status: ProposalStatus
    created_at: datetime
    decided_at: datetime | None = None


class MemoryProposalCreateV1(Contract):
    kind: Literal["user", "project", "skill"] = "project"
    content: str = Field(min_length=3, max_length=20_000)
    source_run_id: str | None = Field(default=None, max_length=200)


class MemoryDecisionV1(Contract):
    decision: Decision
    reason: str | None = Field(default=None, max_length=2000)


class FeedbackV1(Contract):
    run_id: str
    rating: Literal["positive", "negative"]
    correction: str | None = Field(default=None, max_length=20_000)


CorpusKind = Literal["code", "docs", "notes", "mixed"]
CorpusStatus = Literal["pending", "indexing", "indexed", "error", "revoked"]


class CorpusSourceCreateV1(Contract):
    root_path: str = Field(min_length=1, max_length=4096)
    label: str | None = Field(default=None, max_length=120)
    kind: CorpusKind = "mixed"


class CorpusConsentDecisionV1(Contract):
    """Grant or revoke cloud-embedding consent for one source. Granting is the
    egress boundary: only after this does any text from the source leave the Mac."""

    consent: bool
    reason: str | None = Field(default=None, max_length=2000)


class CorpusSourceV1(Contract):
    id: str
    root_path: str
    label: str
    kind: CorpusKind
    provider: Literal["local", "notion"] = "local"
    consent: bool
    status: CorpusStatus
    file_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    last_indexed_at: datetime | None = None
    last_error: str | None = None
    created_at: datetime
    updated_at: datetime


class CorpusReindexResultV1(Contract):
    source_id: str
    status: CorpusStatus
    files_indexed: int = Field(ge=0)
    files_skipped: int = Field(ge=0)
    files_removed: int = Field(ge=0)
    chunks: int = Field(ge=0)
    message: str = ""


class NotionConnectionUpdateV1(Contract):
    """A read-only connection configuration.

    ``access_token=None`` preserves an already stored token, so the UI never
    needs to read a secret back from the API.
    """

    access_token: str | None = Field(default=None, min_length=10, max_length=500)
    # Empty means every page explicitly shared with the Notion connection.
    # IDs optionally narrow the mirror to selected page trees.
    root_page_ids: list[str] = Field(default_factory=list, max_length=50)
    label: str = Field(default="Notion", min_length=1, max_length=120)


class NotionConnectionV1(Contract):
    configured: bool
    token_configured: bool
    root_page_ids: list[str] = Field(default_factory=list)
    label: str = "Notion"
    source: CorpusSourceV1 | None = None
    last_synced_at: datetime | None = None
    page_count: int = Field(default=0, ge=0)
    last_error: str | None = None


class NotionSyncResultV1(Contract):
    pages_fetched: int = Field(ge=0)
    pages_written: int = Field(ge=0)
    pages_removed: int = Field(ge=0)
    source: CorpusSourceV1
    index_result: CorpusReindexResultV1 | None = None
    message: str = ""


class KnowledgeSnippetV1(Contract):
    """One retrieved passage of the user's own knowledge, with its provenance."""

    source_label: str
    provider: Literal["local", "notion"] = "local"
    rel_path: str
    symbol: str | None = None
    start_line: int | None = None
    text: str
    score: float


class PersonalProfileV1(Contract):
    content: str = Field(default="", max_length=16_000)
    characters: int = Field(default=0, ge=0)
    updated_at: datetime | None = None


class PersonalProfileUpdateV1(Contract):
    content: str = Field(default="", max_length=16_000)


class ModelPreferenceV1(Contract):
    mode: Literal["split", "pinned"] = "split"
    model: str | None = None
    provider: Literal["local", "oci"] = "local"
    oci_tools: list[Literal["x_search", "code_interpreter"]] = Field(
        default_factory=lambda: ["code_interpreter"], max_length=2
    )
    oci_available: bool = False


class ModelPreferenceUpdateV1(Contract):
    mode: Literal["split", "pinned"]
    model: str | None = Field(default=None, max_length=200)
    provider: Literal["local", "oci"] = "local"
    oci_tools: list[Literal["x_search", "code_interpreter"]] = Field(
        default_factory=lambda: ["code_interpreter"], max_length=2
    )


# A tool is a data record, not code. The registry stores immutable versioned
# definitions; the planner only ever sees the catalog.


class ModelAccessV1(Contract):
    """A tool's *runtime* model budget. Off by default. Prompts are pinned
    templates chosen at definition time; a tool fills parameters, never authors
    a system prompt, so it cannot repurpose the model."""

    enabled: bool = False
    roles: list[Literal["planner", "coder", "reviewer"]] = Field(default_factory=list)
    max_calls_per_run: int = Field(default=0, ge=0, le=16)
    max_tokens_per_call: int = Field(default=0, ge=0, le=8_192)
    prompt_templates: dict[str, str] = Field(default_factory=dict)


class CapabilityProfileV1(Contract):
    """Every grant a tool has, declared up front and immutable after approval.
    `code_allowlist` / `runtime_allowlists` name AST profiles that live in
    reviewed host code — a definition references them by name, never inline."""

    code_allowlist: str
    runtime_allowlists: dict[str, str] = Field(default_factory=dict)
    model_access: ModelAccessV1 = Field(default_factory=ModelAccessV1)
    filesystem: Literal["run-io"] = "run-io"
    network: Literal["none"] = "none"
    max_runtime_seconds: int = Field(default=150, ge=1, le=600)
    max_artifact_bytes: int = Field(default=10_000_000, ge=1)


class ToolRouteFactsV1(Contract):
    """Deterministic routing facts the host applies when this tool matches — the
    values that used to be hardcoded literals in the planner normalizer."""

    existing_risk: RiskLevel = RiskLevel.R2
    factory_risk: RiskLevel = RiskLevel.R3
    input_pipeline: Literal["none", "attachment_text", "architecture_spec"] = "none"


class ToolDefinitionV1(Contract):
    schema_version: Literal["1"] = "1"
    slug: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    version: str
    name: str
    description: str
    # The reviewed host archetype this definition was hardened from. It owns the
    # capability profile and eval fixtures; the model never picks it.
    archetype: str = ""
    # Pinned authoring prompt, frozen at Gate-1. Authored code is stored per-build.
    author_system_prompt: str = ""
    intent_examples: list[str] = Field(default_factory=list, max_length=16)
    input_contract: dict[str, Any] = Field(default_factory=dict)
    output_contract: dict[str, Any] = Field(default_factory=dict)
    route_facts: ToolRouteFactsV1 = Field(default_factory=ToolRouteFactsV1)
    capability_profile: CapabilityProfileV1
    status: Literal["draft", "proposed", "defined", "retired"] = "defined"
    content_hash: str = ""
    created_at: datetime = Field(default_factory=utc_now)


class ToolDefinitionDraftV1(Contract):
    """A model-proposed new tool, pre-Gate-1. The host validates and hardens this
    into a ToolDefinitionV1 (assigning the capability profile from a safe menu);
    the model never sets its own capabilities."""

    name: str = Field(max_length=120)
    description: str = Field(max_length=2_000)
    intent: str = Field(default="", max_length=2_000)
    requested_capabilities: list[str] = Field(default_factory=list, max_length=16)
    input_sketch: str = Field(default="", max_length=2_000)
    output_sketch: str = Field(default="", max_length=2_000)

    # Coerce and bound these free-text hints rather than fail the whole draft.
    @field_validator("description", "intent", "input_sketch", "output_sketch", mode="before")
    @classmethod
    def _coerce_text_field(cls, value: Any) -> str:
        return _as_text(value)[:2_000]

    @field_validator("name", mode="before")
    @classmethod
    def _coerce_name(cls, value: Any) -> str:
        return (_as_text(value).strip() or "Custom Tool")[:120]

    @field_validator("requested_capabilities", mode="before")
    @classmethod
    def _coerce_caps(cls, value: Any) -> list[str]:
        if isinstance(value, list):
            return [_as_text(item)[:200] for item in value][:16]
        if value in (None, ""):
            return []
        return [_as_text(value)[:200]]


class ToolDefinitionProposalV1(Contract):
    """Gate-1 record: a drafted definition awaiting human approval of its
    *capabilities*. Approval promotes the definition to `defined` (buildable);
    rejection tombstones it. Mirrors ToolProposalV1 (Gate-2 over built versions)."""

    id: str
    definition_id: str
    slug: str
    version: str
    status: ProposalStatus
    risk_level: RiskLevel
    summary: str
    source_run_id: str | None = None
    decision_reason: str | None = None
    created_at: datetime
    decided_at: datetime | None = None


class ToolDefinitionBuildV1(Contract):
    """Gate-2 record for a declarative tool: a `defined` tool that the factory
    built and evaluated (hermetic scripted-broker eval). Activation pins this
    immutable, content-hashed build as the runnable definition version."""

    id: str
    definition_id: str
    slug: str
    version: str
    content_hash: str
    status: Literal["evaluated", "active", "rejected", "superseded"]
    eval_report: EvalReportV1 | None = None
    # The authored implementation this build pins. Empty for declarative tools.
    implementation: str = ""
    code_review: dict[str, Any] | None = None
    source_run_id: str | None = None
    created_at: datetime
    decided_at: datetime | None = None


class ToolCodeReviewV1(Contract):
    """Result of the optional OCI Grok review of freshly-authored tool code. The
    host AST-gate still validates whatever code is used — Grok can improve or flag
    but never widens capabilities; an unsafe verdict blocks the build."""

    reviewed: bool
    reviewer: str = ""
    safe: bool = True
    improved: bool = False
    reasons: list[str] = Field(default_factory=list)


class ToolDefinitionRecordV1(Contract):
    """Registry-browser view: a definition plus its host-derived lifecycle state.
    `runnable` means an active tool exists to serve `existing_tool`; `buildable`
    means it is `defined` but not yet built/active (serves `tool_factory`)."""

    definition: ToolDefinitionV1
    active: bool = False
    runnable: bool = False
    buildable: bool = False
    disabled: bool = False
    pending_definition_proposal: bool = False
    pending_build: bool = False


class CorpusSearchV1(Contract):
    query: str = Field(min_length=1, max_length=4_000)
    limit: int | None = Field(default=None, ge=1, le=50)


# ── Code graph (Graph-RAG Stage 1) ───────────────────────────────────────────


class CodeGraphDefinitionV1(Contract):
    kind: str
    name: str
    qualname: str
    rel_path: str
    start_line: int
    end_line: int
    source_label: str


class CodeGraphCallerV1(Contract):
    """A site that calls the looked-up symbol (matched by name)."""

    caller: str
    rel_path: str
    line: int
    dst_raw: str
    source_label: str


class CodeGraphCalleeV1(Contract):
    """A symbol the looked-up definition calls."""

    dst_name: str
    dst_raw: str
    rel_path: str
    line: int
    source_label: str


class CodeGraphImportV1(Contract):
    rel_path: str
    dst_raw: str
    line: int
    source_label: str


class CodeGraphLookupV1(Contract):
    """Everything the deterministic graph knows about one symbol name. Name-based
    matching means results can span files; callers/callees are one hop."""

    name: str
    definitions: list[CodeGraphDefinitionV1] = Field(default_factory=list)
    callers: list[CodeGraphCallerV1] = Field(default_factory=list)
    callees: list[CodeGraphCalleeV1] = Field(default_factory=list)
    imports: list[CodeGraphImportV1] = Field(default_factory=list)


class CodeGraphStatsV1(Contract):
    node_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)
    nodes_by_kind: dict[str, int] = Field(default_factory=dict)
    edges_by_kind: dict[str, int] = Field(default_factory=dict)


# ── Entity graph (Graph-RAG Stage 2) ─────────────────────────────────────────


class EntityGraphStatsV1(Contract):
    node_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)
    nodes_by_kind: dict[str, int] = Field(default_factory=dict)


class EntityRelationOutV1(Contract):
    relation: str
    dst_name: str
    rel_path: str
    source_label: str


class EntityRelationInV1(Contract):
    relation: str
    src_name: str
    rel_path: str
    source_label: str


class EntityGraphLookupV1(Contract):
    """An entity's kinds and its one-hop relationships in both directions."""

    name: str
    kinds: list[str] = Field(default_factory=list)
    relations_out: list[EntityRelationOutV1] = Field(default_factory=list)
    relations_in: list[EntityRelationInV1] = Field(default_factory=list)


class ArchitectureComponentV1(Contract):
    id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    label: str = Field(min_length=1, max_length=120)
    kind: str = Field(min_length=1, max_length=64)


class ArchitectureEdgeV1(Contract):
    source: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    target: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    label: str = Field(default="", max_length=120)


class ArchitectureBoundaryV1(Contract):
    id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    label: str = Field(min_length=1, max_length=120)
    component_ids: list[str] = Field(min_length=1, max_length=64)


class ArchitectureSpecV1(Contract):
    title: str = Field(min_length=1, max_length=160)
    provider: Literal[
        "generic", "aws", "azure", "gcp", "oci", "kubernetes", "onprem", "hybrid"
    ] = "generic"
    direction: Literal["LR", "RL", "TB", "BT"] = "LR"
    components: list[ArchitectureComponentV1] = Field(min_length=1, max_length=64)
    edges: list[ArchitectureEdgeV1] = Field(default_factory=list, max_length=256)
    boundaries: list[ArchitectureBoundaryV1] = Field(default_factory=list, max_length=16)
    assumptions: list[str] = Field(default_factory=list, max_length=32)
    unresolved_ambiguities: list[str] = Field(default_factory=list, max_length=32)

    @field_validator("edges")
    @classmethod
    def validate_edges(cls, value: list[ArchitectureEdgeV1]) -> list[ArchitectureEdgeV1]:
        return value

    @field_validator("assumptions", "unresolved_ambiguities")
    @classmethod
    def validate_notes(cls, value: list[str]) -> list[str]:
        if any(not item or len(item) > 500 for item in value):
            raise ValueError("architecture notes must contain 1 to 500 characters")
        return value

    @model_validator(mode="after")
    def validate_references(self) -> "ArchitectureSpecV1":
        component_ids = [component.id for component in self.components]
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("component IDs must be unique")
        known = set(component_ids)
        for edge in self.edges:
            if edge.source not in known or edge.target not in known:
                raise ValueError("edges must reference declared components")
        boundary_ids = [boundary.id for boundary in self.boundaries]
        if len(boundary_ids) != len(set(boundary_ids)):
            raise ValueError("boundary IDs must be unique")
        assigned: set[str] = set()
        for boundary in self.boundaries:
            members = boundary.component_ids
            if len(members) != len(set(members)):
                raise ValueError("boundary component IDs must be unique")
            if not set(members).issubset(known):
                raise ValueError("boundaries must reference declared components")
            if assigned.intersection(members):
                raise ValueError("components may belong to only one boundary")
            assigned.update(members)
        return self


class HealthV1(Contract):
    status: Literal["ok", "degraded"]
    version: str
    database: bool
    checkpoints: bool
    model_backend: str
    reference_runner: str
    details: dict[str, Any] = Field(default_factory=dict)
