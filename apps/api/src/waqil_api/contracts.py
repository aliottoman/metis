from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
    ValidationInfo,
)


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


# JSON-Schema keywords that bound a *value* rather than describe the *shape* of
# a reply. Pydantic enforces every one of them when the reply is validated, so
# dropping them from a decoding grammar loses no guarantee — see grammar_schema.
_VALUE_CONSTRAINT_KEYWORDS = frozenset(
    {
        "maxLength",
        "minLength",
        "pattern",
        "format",
        "maxItems",
        "minItems",
        "uniqueItems",
        "maxProperties",
        "minProperties",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
    }
)

# Structural keywords a grammar compiler has to interpret rather than ignore.
# They are legal, but each one has broken a real backend, so a schema headed for
# local decode is checked for them (see grammar_risks) instead of trusting them.
_GRAMMAR_RISK_KEYWORDS = ("$ref", "anyOf", "allOf", "oneOf", "not", "if")


def _const_to_enum(node: Any) -> None:
    """Rewrite JSON-Schema ``const`` as a one-element ``enum``, in place, recursively."""
    if isinstance(node, dict):
        if "const" in node:
            # Always drop const; keep an existing enum if one is somehow present.
            node.setdefault("enum", [node["const"]])
            node.pop("const")
        for value in node.values():
            _const_to_enum(value)
    elif isinstance(node, list):
        for item in node:
            _const_to_enum(item)


def grammar_schema(contract: type[BaseModel]) -> dict[str, Any]:
    """The shape-only JSON Schema used to constrain a local model's decode.

    A local backend compiles this schema into a decoding grammar, and that
    compiler is far less forgiving than a JSON-Schema validator. Three separate
    outages were the same mistake — handing it a keyword it could not compile. A
    nested ``anyOf``/``$ref`` collapsed MLX output to empty; ``const`` made
    llama.cpp reject the request outright; and a ``maxLength`` of 2000 or more
    still does, with HTTP 400 "failed to parse grammar" on every project step,
    which the host could only read as the model replying unintelligibly.

    Patching one keyword at a time never converged, because the underlying
    mistake is sending a *validation* contract to a *decoder*. So the grammar is
    now told only what shape to produce — types, properties, required keys,
    enums, items — and every keyword that merely bounds a value is dropped.
    Nothing is actually unenforced: the reply is validated against the full
    contract on arrival, where the bounds still apply. The trade is deliberate,
    and lopsided in our favour: overrunning a bound costs one repair round-trip,
    while a schema the compiler rejects costs the entire turn.
    """
    return _project_for_grammar(contract.model_json_schema())


def _project_for_grammar(node: Any) -> Any:
    """Copy one JSON-Schema node, keeping only what a decoding grammar can use."""
    if isinstance(node, dict):
        projected = {
            key: _project_for_grammar(value)
            for key, value in node.items()
            if key not in _VALUE_CONSTRAINT_KEYWORDS
        }
        if "const" in projected:
            # The identical constraint, in the spelling every backend compiles.
            projected.setdefault("enum", [projected.pop("const")])
        return projected
    if isinstance(node, list):
        return [_project_for_grammar(item) for item in node]
    return node


def value_constraints(schema: dict[str, Any], path: str = "") -> list[str]:
    """Locations of value-bounding keywords still present in a schema.

    A projected schema must report none: every one of these is a keyword some
    grammar compiler has to interpret, and ``maxLength`` alone took out five of
    the eight schemas the local path decodes with.
    """
    found: list[str] = []
    if isinstance(schema, dict):
        for keyword in sorted(_VALUE_CONSTRAINT_KEYWORDS & set(schema)):
            found.append(f"{path or '.'}:{keyword}")
        for key, value in schema.items():
            found.extend(value_constraints(value, f"{path}.{key}"))
    elif isinstance(schema, list):
        for index, item in enumerate(schema):
            found.extend(value_constraints(item, f"{path}[{index}]"))
    return found


def grammar_risks(schema: dict[str, Any], path: str = "") -> list[str]:
    """Locations of keywords a grammar compiler has historically mishandled.

    Empty means the schema is flat enough that no backend has to resolve a
    reference or choose between branches to build its grammar. This is an
    assertion for tests and the preflight, not a transform: a risk here is a
    schema that needs rethinking, not one the host can quietly rewrite.
    """
    found: list[str] = []
    if isinstance(schema, dict):
        for keyword in _GRAMMAR_RISK_KEYWORDS:
            if keyword in schema:
                found.append(f"{path or '.'}:{keyword}")
        for key, value in schema.items():
            found.extend(grammar_risks(value, f"{path}.{key}"))
    elif isinstance(schema, list):
        for index, item in enumerate(schema):
            found.extend(grammar_risks(item, f"{path}[{index}]"))
    return found


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: Any, handler: Any) -> Any:
        # No contract emits ``const``, in any direction. grammar_schema would
        # rewrite it for local decode anyway, but the OCI Responses path sends
        # ``model_json_schema()`` straight through under strict mode, and that
        # strict subset has never included ``const`` — so a single-value Literal
        # has to arrive as a one-element enum there too. Identical constraint,
        # universally understood spelling.
        schema = handler(core_schema)
        _const_to_enum(handler.resolve_ref_schema(schema))
        return schema


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    # The model paused the turn to ask the user a question (ask_user). Like
    # AWAITING_APPROVAL this is NOT terminal: the SSE stream stays open across
    # the pause so the answer resumes the same run on the same event stream.
    AWAITING_INPUT = "awaiting_input"
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


# Who leads each bounded project step after the initial repository map. The
# map itself is not a mode: it is one call, made once, by whichever cloud
# provider is configured. These choose the driver of the loop that follows.
#   grok_bootstrap_local  North runs project turns on-device.
#   grok_continuous       Grok leads every step, through OCI.
#   cohere_continuous     Command A+ leads every step, through the Cohere key.
# Spelled once here because the same three strings are a request field, an
# open-project body, a stored session, and a SQLite CHECK constraint, and the
# four drifted apart the last time a mode was added.
ProjectModeV1 = Literal["grok_bootstrap_local", "grok_continuous", "cohere_continuous"]
PROJECT_MODES: frozenset[str] = frozenset(get_args(ProjectModeV1))


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


class TranscriptV1(Contract):
    """One dictation turned into text. Carries no id: nothing is stored."""

    text: str


class AssetEnvVarV1(Contract):
    """One variable declared by the project's own .env file.

    Presence only: `is_set` says a non-empty value exists on disk, and the value
    itself is never serialized. Values travel inward, never back out.
    """

    key: str = Field(min_length=1, max_length=64)
    is_set: bool = False
    sensitive: bool = False


class AssetV1(Contract):
    id: str
    name: str = Field(min_length=1, max_length=120)
    summary: str = Field(min_length=1, max_length=240)
    category: str = Field(min_length=1, max_length=60)
    tags: list[str] = Field(default_factory=list, max_length=16)
    framework: str | None = Field(default=None, max_length=60)
    entrypoint: str | None = Field(default=None, max_length=240)
    env_keys: list[str] = Field(default_factory=list, max_length=64)
    env_file: list[AssetEnvVarV1] = Field(default_factory=list, max_length=64)
    env_file_present: bool = False
    launch_configured: bool = False
    launch_approved: bool = False
    launch_command: list[str] = Field(default_factory=list, max_length=32)
    build_command: list[list[str]] = Field(default_factory=list, max_length=8)
    status: AssetStatus
    url: str | None = None


class AssetRecipeV1(Contract):
    """A model-drafted launch recipe for one asset, flat by design.

    This is the wire shape the selected model fills through one structured
    call — nested `launch.command` objects decode less reliably than flat fields,
    so the endpoint assembles the real .metis/asset.json body from these.
    """

    entrypoint: str | None = Field(default=None, max_length=240)
    launch_command: list[str] = Field(min_length=1, max_length=32)
    build_command: list[str] | None = Field(default=None, max_length=32)
    launch_path: str = Field(default="", max_length=240)
    env_keys: list[str] = Field(default_factory=list, max_length=64)


class AssetCreateV1(Contract):
    """A request to start a brand-new, empty project folder from the picker.

    The bound is the folder name only: where it lands is always the first
    configured projects root, decided by the host — a request can never choose
    a directory.
    """

    name: str = Field(min_length=1, max_length=64)


class AssetEnvUpdateV1(Contract):
    values: dict[str, str] = Field(default_factory=dict, max_length=64)

    @field_validator("values")
    @classmethod
    def bounded_values(cls, value: dict[str, str]) -> dict[str, str]:
        for key, item in value.items():
            if len(key) > 64 or len(item) > 16_384:
                raise ValueError("asset environment contains an invalid key or value")
            if any(character in item for character in ("\x00", "\r", "\n")):
                raise ValueError("asset environment values must be single-line text")
        return value


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
    project_mode: ProjectModeV1 | None = None
    knowledge_scope: Literal["auto", "notion", "web"] = "auto"
    customer_id: str | None = Field(default=None, pattern=r"^cust_[a-f0-9]{20}$")


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
    route: Literal[
        "direct",
        "existing_tool",
        "tool_factory",
        "tool_definition",
        "document",
        "queue_update",
        "ask_user",
    ]
    tool_slug: str | None = None
    risk_level: RiskLevel = RiskLevel.R0
    steps: list[PlanStepV1] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    # Only for route == "ask_user": the single clarifying question the planner
    # pauses the turn to ask, and up to eight offered choices. Empty options
    # means the card is a pure free-text prompt. Both stay None/empty on every
    # other route, so adding them changes no existing plan.
    question: str | None = Field(default=None, max_length=2000)
    options: list[str] = Field(default_factory=list, max_length=8)


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


class ProjectProtectionsV1(Contract):
    """Per-project files a coding session may read but never change.

    Identities and patterns only: the bytes behind them are hashed per run at
    admission, so a file edited between runs cannot inherit a stale hash.
    """

    protected_files: list[str] = Field(default_factory=list, max_length=256)


class ProjectOpenV1(Contract):
    mode: ProjectModeV1


class ConversationProjectV1(Contract):
    conversation_id: str
    project_id: str
    mode: ProjectModeV1
    updated_at: datetime


class ProjectBootstrapV1(Contract):
    summary: str = Field(max_length=2_000)
    architecture: list[str] = Field(default_factory=list, max_length=24)
    conventions: list[str] = Field(default_factory=list, max_length=24)
    important_paths: list[str] = Field(default_factory=list, max_length=48)
    verification: list[str] = Field(default_factory=list, max_length=24)
    risks: list[str] = Field(default_factory=list, max_length=24)


class ProjectCheckV1(Contract):
    """One reviewed verification command a project declares for itself."""

    name: str = Field(min_length=1, max_length=32)
    command: list[str] = Field(default_factory=list, max_length=32)
    description: str = Field(default="", max_length=240)
    explanation: str = Field(default="", max_length=600)
    timeout_seconds: int = Field(default=300, ge=1, le=1_800)


class ProjectVerificationV1(Contract):
    """A project's verification recipe and its approval state.

    `explanation` and `boundary` carry the plain-English account of what
    approval authorizes, so the decision never depends on reading argv.
    """

    project_id: str
    configured: bool = False
    approved: bool = False
    fingerprint: str | None = None
    checks: list[ProjectCheckV1] = Field(default_factory=list, max_length=12)
    explanation: str = Field(default="", max_length=8_000)
    boundary: str = Field(default="", max_length=1_200)
    error: str | None = Field(default=None, max_length=400)


class ProjectToolCallV1(Contract):
    name: Literal[
        "list_files",
        "search_code",
        "read_file",
        "apply_patch",
        "replace_lines",
        "create_file",
        "run_check",
        "inspect_api",
        "ask_user",
        "respond",
        "revise_plan",
    ]
    arguments: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def coerce_argument_key(cls, value: Any) -> Any:
        """Accept the argument-key synonyms coder models actually emit.

        A live run died on ``{"name": "read_file", "parameters": {...}}`` —
        three retries, same key, turn over. The intent is unambiguous, so the
        synonym is renamed rather than rejected. Shape only: the tool name
        Literal and the workspace's own bounds still decide what may run, and
        an *unrecognized* extra key still fails exactly as before.
        """
        if not isinstance(value, dict) or isinstance(value.get("arguments"), dict):
            return value
        payload = dict(value)
        for synonym in ("parameters", "params", "args", "inputs", "input"):
            candidate = payload.get(synonym)
            if isinstance(candidate, dict):
                payload.pop(synonym)
                payload["arguments"] = candidate
                return payload
        return value


class ProjectAgentStepV1(Contract):
    status: Literal["tool", "complete"]
    response: str = Field(default="", max_length=40_000)
    tool_call: ProjectToolCallV1 | None = None
    # Further READ-ONLY calls that arrived in the same reply, to run in the
    # same step. The tool-calling transports have always been handed these —
    # a hosted model asked to look around commonly returns a listing and two
    # reads at once — and they were dropped on the floor: every provider
    # returned on the first call it found. Each discarded read then cost a
    # whole round-trip to ask for again, out of a 48-step budget.
    #
    # Reads only, and only when EVERY call in the reply is a read: a batch is
    # executed in order with no model step between its members, so anything
    # whose result should change the next call — a write, a check — must stay
    # exclusive. The refusal breaker, the write pin and the staging overlay
    # all assume one write per step, and this leaves that untouched.
    extra_calls: list[ProjectToolCallV1] = Field(default_factory=list, max_length=3)
    learnings: list[str] = Field(default_factory=list, max_length=16)

    @field_validator("extra_calls")
    @classmethod
    def batched_calls_are_reads_only(
        cls, value: list[ProjectToolCallV1]
    ) -> list[ProjectToolCallV1]:
        """Enforce what the comment above has always asserted.

        The batch executes in order with no model step between its members and
        reports ``staged: False`` for every one, so a write riding along here
        would bypass the write pin, the one-write-per-step rule, and the
        read-only-before-a-plan gate. This was documented and never checked;
        a non-read is dropped rather than rejecting the whole step, because
        the step's own first call may be perfectly good.
        """

        return [item for item in value if item.name in READ_ONLY_PROJECT_TOOLS]

    @model_validator(mode="before")
    @classmethod
    def coerce_step_envelope(cls, value: Any) -> Any:
        """Accept the collapsed shapes a coder model reaches for.

        Small local models routinely return the tool call itself —
        ``{"name": "list_files", "arguments": {...}}`` — instead of the
        envelope that carries it. The intent is unambiguous, so the host
        reshapes it rather than spending a repair round-trip and then failing
        the turn. This widens the accepted *shape*, never the authority: the
        tool name is still checked against the fixed Literal below, so an
        invented tool is rejected exactly as before.
        """
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        status = payload.get("status")
        name = payload.get("name") or payload.get("tool")
        if (
            "tool_call" not in payload
            and isinstance(name, str)
            and name
            and status in (None, "tool")
        ):
            arguments = payload.get("arguments")
            if not isinstance(arguments, dict):
                arguments = {}
                for synonym in ("args", "parameters", "params", "inputs", "input"):
                    candidate = payload.get(synonym)
                    if isinstance(candidate, dict):
                        arguments = candidate
                        break
            payload = {
                key: item
                for key, item in payload.items()
                if key in {"response", "learnings"}
            }
            payload["tool_call"] = {"name": name, "arguments": arguments}
        if "status" not in payload:
            payload["status"] = "tool" if payload.get("tool_call") else "complete"
        return payload

    @model_validator(mode="after")
    def valid_project_step(self) -> "ProjectAgentStepV1":
        if self.status == "tool" and self.tool_call is None:
            raise ValueError("a project tool step requires tool_call")
        if self.status == "complete" and not self.response.strip():
            raise ValueError("a completed project step requires a response")
        return self


# Every argument key the six project tools accept, as one flat table. The
# grammar shows this closed set for `arguments`, and that alone is what stops
# the model inventing keys: against a free-form object it produced
# {"path","patch"} for apply_patch on every single attempt, and against this one
# it produces {"path","original","replacement"} with a verbatim block. The host
# was requiring argument names it never told the model — only Grok, which gets
# real function schemas, was ever shown them.
#
# Deliberately one flat table rather than a per-tool model: a $ref or a
# discriminated union is the exact construct that collapsed MLX decode to empty
# output, which is what the wire contracts below were written to avoid.
PROJECT_TOOL_ARGUMENT_PROPERTIES: dict[str, dict[str, Any]] = {
    "path": {"type": "string"},
    "content": {"type": "string"},
    "original": {"type": "string"},
    "replacement": {"type": "string"},
    "query": {"type": "string"},
    "case_sensitive": {"type": "boolean"},
    "name": {"type": "string"},
    "module": {"type": "string"},
    "symbol": {"type": "string"},
    "start_line": {"type": "integer"},
    "end_line": {"type": "integer"},
    "limit": {"type": "integer"},
    "expect": {"type": "string"},
    # The talk channel (ask_user / respond). Flat like everything above: a
    # nested per-tool model is the construct that collapsed MLX decode.
    "question": {"type": "string"},
    "options": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
    "message": {"type": "string"},
    # revise_plan. `files` is the proposed manifest, so it carries the same
    # bound the manifest contract does. Omissions are retained by the host;
    # `remove_files` is the explicit, evidence-backed removal channel.
    "files": {"type": "array", "items": {"type": "string"}, "maxItems": 24},
    "remove_files": {
        "type": "array",
        "items": {"type": "string"},
        "maxItems": 24,
    },
    "reason": {"type": "string"},
}

# What each tool actually needs, mirroring the host's own refusals so the
# grammar and the workspace cannot disagree about what a valid call looks like.
# Tools that only ever look. Everything else can put bytes in the staged
# overlay and is therefore subject to the plan gate in the control plane.
READ_ONLY_PROJECT_TOOLS = frozenset(
    {"list_files", "search_code", "read_file", "inspect_api"}
)


PROJECT_TOOL_REQUIRED_ARGUMENTS: dict[str, list[str]] = {
    "list_files": [],
    "search_code": ["query"],
    "read_file": ["path"],
    "create_file": ["path", "content"],
    "apply_patch": ["path", "original", "replacement"],
    "replace_lines": ["path", "start_line", "end_line", "replacement"],
    "run_check": ["name"],
    "inspect_api": ["module"],
    # The talk channel. Before these, "speak to the user" and "the work is
    # done" shared one exit (status=complete), so a model that wanted to ask a
    # clarifying question could only fabricate a finish — the host then judged
    # that finish by keyword-matching the USER'S prompt, and a "revamp"-phrased
    # build read as conversation. ask_user pauses the turn on one question and
    # resumes with the answer as this call's result; respond answers without
    # claiming any build work happened.
    "ask_user": ["question"],
    "respond": ["message"],
    # The plan channel. A manifest is taken once and then holds the turn
    # accountable to it — including by narrowing create_file's path to an enum
    # of what is still owed. That is right when the plan is right and a trap
    # when it is not: a reskin planned against paths the project does not have
    # left the model with no legal write it believed in, and three replies the
    # backend refused to generate ended the turn. revise_plan is the way out
    # that is not "give up": the model states the corrected file list and what
    # the evidence showed, the host re-gates on the new list.
    "revise_plan": ["files", "reason"],
}

# A deliberately out-of-range inclusive end is safe: the workspace clamps it
# to the current file length. This lets the recovery grammar require a complete
# replacement without first spending another model step rediscovering a line
# count the host already owns.
WHOLE_FILE_END_LINE = 1_000_000

# The roster's talk tools: host affordances the loop routes itself (a pause on
# a question, an answer published as the turn's response). They are on the
# roster so every transport can call them, and they must never reach the
# workspace executor, which only knows files and checks.
PROJECT_TALK_TOOLS: frozenset[str] = frozenset({"ask_user", "respond"})

# Tools that only look: they stage nothing, run nothing, and cannot change
# what a later call in the same reply would have wanted to do. That is what
# makes them safe to execute as a batch inside one step. run_check is
# deliberately absent — it executes a command — and so is every write.
PROJECT_READ_TOOLS: frozenset[str] = frozenset(
    {"list_files", "search_code", "read_file", "inspect_api"}
)

# The same rule for the plan channel: revise_plan rewrites the turn's manifest
# in graph state and never touches a file.
PROJECT_PLAN_TOOLS: frozenset[str] = frozenset({"revise_plan"})

# Every roster tool the loop answers itself. The workspace refuses all of them
# by name, so a routing bug is loud instead of arriving as "unsupported tool".
PROJECT_HOST_TOOLS: frozenset[str] = PROJECT_TALK_TOOLS | PROJECT_PLAN_TOOLS

# Merged into the emitted schema for `arguments`, replacing the open
# `additionalProperties: true` a dict[str, Any] would otherwise produce. The
# Python type stays a plain dict so the argument-synonym coercion below still
# runs — the grammar narrows what is generated, not what the host will accept.
_CLOSED_TOOL_ARGUMENTS: dict[str, Any] = {
    "properties": PROJECT_TOOL_ARGUMENT_PROPERTIES,
    "additionalProperties": False,
}


# Listed per tool rather than derived, because "every key this tool does not
# require" is not the same thing as "every key it accepts" — offering
# create_file an `original` invites exactly the confusion this table exists to
# remove.
PROJECT_TOOL_OPTIONAL_ARGUMENTS: dict[str, list[str]] = {
    "list_files": ["path", "limit"],
    "search_code": ["case_sensitive", "limit"],
    "read_file": ["start_line", "end_line"],
    "create_file": [],
    "apply_patch": [],
    "replace_lines": ["expect"],
    "run_check": [],
    "inspect_api": ["symbol"],
    "ask_user": ["options"],
    "respond": [],
    "revise_plan": ["remove_files"],
}

# One line per tool saying what it is for, in the terms the model has to get
# right. The local model is given no function schemas at all — the system prompt
# names the tools and never their arguments — so this is the only place it can
# learn that a patch is an exact-match replacement rather than a diff.
_PROJECT_TOOL_NOTES: dict[str, str] = {
    "list_files": 'List project files. Send path "" for the whole project.',
    "search_code": "Find an exact string in the project's readable text.",
    "read_file": "Read one file. The text returned is verbatim, safe to copy.",
    "create_file": (
        "Create a new file. content must be the complete file text and cannot be "
        "empty. Refuses to overwrite; use apply_patch to change a file that exists "
        "or is already staged."
    ),
    "apply_patch": (
        "Replace one block of an existing or staged file. original must appear "
        "exactly once in the current file text, copied verbatim; replacement is "
        "what it becomes. This is not a diff — do not send patch text. If your "
        "original keeps failing to match, switch to replace_lines."
    ),
    "replace_lines": (
        "Replace lines start_line through end_line (1-indexed, inclusive) of an "
        "existing or staged file with replacement — no exact quoting needed. "
        "read_file the range first: its start_line/end_line confirm your "
        "coordinates, and passing a short distinctive substring of the doomed "
        "block as expect makes a mis-aimed range refuse instead of landing. To "
        "rewrite a whole file, replace lines 1 through its last line."
    ),
    "run_check": "Run one declared verification check by name.",
    "inspect_api": (
        "Look up an INSTALLED library before you write against it: its real "
        "exported names, and the real signature of one function or class. Use it "
        "whenever you are about to call an API you have not verified — a keyword "
        "argument that does not exist parses perfectly and fails at runtime. "
        'module is an import path such as "openai"; symbol is optional. It '
        "reads libraries, never this project's own files — use read_file for those."
    ),
    "ask_user": (
        "Pause the turn to ask the user ONE clarifying question; their answer "
        "comes back as this call's result and the same turn continues. Use it "
        "when the request is genuinely ambiguous in a way that changes what you "
        "would build, or when the user invited questions. Prefer sensible "
        "defaults over asking; never ask more than once per turn. options may "
        "offer up to 6 short choices."
    ),
    "respond": (
        "Answer the user without claiming any build work happened. Use it for "
        "questions about the project, explanations, and status — a respond turn "
        "is complete in itself and stages nothing."
    ),
    "revise_plan": (
        "Correct this turn's file manifest when what you read contradicts it — "
        "a planned path the project does not have, a framework that makes it "
        "wrong, work that turns out to need different files. Omitted prior "
        "paths remain committed. files proposes the corrected order and any "
        "additions; remove_files explicitly names invalid unstaged paths to "
        "drop, and reason gives the repository evidence. Use this instead of "
        "writing a file you believe is wrong or finishing to escape the plan."
    ),
}


def project_tool_catalog() -> list[dict[str, Any]]:
    """The tool reference sent to a local model, which sees no function schemas."""
    return [
        {
            "name": name,
            "required_arguments": list(required),
            "optional_arguments": list(PROJECT_TOOL_OPTIONAL_ARGUMENTS[name]),
            "note": _PROJECT_TOOL_NOTES[name],
        }
        for name, required in PROJECT_TOOL_REQUIRED_ARGUMENTS.items()
    ]


def project_step_retry_schema(tool: str) -> dict[str, Any]:
    """A flat step schema pinned to one tool, carrying its exact required keys.

    Used for the step immediately after the host refused a call for the shape of
    its arguments. The general schema lists every key the six tools share, so a
    model can still omit one this particular tool needs; narrowing to the tool it
    just got wrong makes the omission ungrammatical for one step. Finishing is
    also removed — the model is mid-correction, not done.
    """
    schema = grammar_schema(ProjectAgentStepWireV1)
    properties = dict(schema["properties"])
    properties["status"] = {"type": "string", "enum": ["tool"]}
    properties["tool"] = {"type": "string", "enum": [tool]}
    properties["arguments"] = {
        "type": "object",
        "additionalProperties": False,
        "properties": PROJECT_TOOL_ARGUMENT_PROPERTIES,
        "required": list(PROJECT_TOOL_REQUIRED_ARGUMENTS[tool]),
    }
    return {
        **schema,
        "properties": properties,
        "required": ["status", "tool", "arguments"],
    }


def project_whole_file_schema(path: str) -> dict[str, Any]:
    """One exact recovery move: replace the complete current file.

    Exact blocks and guessed line ranges are the two edit strategies that can
    keep failing without changing the overlay. After either is refused, this
    schema makes the safer fallback structural: the path and whole-file range
    are fixed, leaving the model to provide only the corrected complete text.
    """
    schema = project_step_retry_schema("replace_lines")
    properties = dict(schema["properties"])
    arguments = dict(properties["arguments"])
    argument_properties = dict(arguments["properties"])
    argument_properties["path"] = {"type": "string", "enum": [path]}
    argument_properties["start_line"] = {"type": "integer", "enum": [1]}
    argument_properties["end_line"] = {
        "type": "integer",
        "enum": [WHOLE_FILE_END_LINE],
    }
    arguments["properties"] = argument_properties
    properties["arguments"] = arguments
    return {**schema, "properties": properties}


def project_directed_schema(
    paths: list[str], *, target_exists: bool | None = None
) -> dict[str, Any]:
    """The grammar for a step the orchestrator directed: write it, or say why.

    The tool-calling lanes get this narrowing as a smaller roster; the local
    lane gets it as a grammar, so both are asked the same small question. What
    is unexpressible here is every read — which the host refuses on a directed
    step anyway, so the model was previously free to generate a call whose only
    possible outcome was a refusal, and a live run showed it doing exactly that.

    revise_plan stays legal. "This instruction cannot be carried out" is a real
    answer, and the alternative to letting the model say it is watching it write
    something wrong.
    """
    schema = project_write_schema(paths)
    properties = dict(schema["properties"])
    writes = ["create_file", "apply_patch", "replace_lines"]
    if target_exists is True:
        # create_file is refused on a path that exists; making it unexpressible
        # is the difference between three wasted steps and a patch.
        writes = ["apply_patch", "replace_lines"]
    elif target_exists is False:
        writes = ["create_file"]
    properties["tool"] = {"type": "string", "enum": [*writes, "revise_plan"]}
    arguments = dict(properties["arguments"])
    argument_properties = dict(arguments["properties"])
    # revise_plan's own arguments, or naming it in the enum would make it
    # grammatically legal and semantically impossible to fill in.
    argument_properties["files"] = PROJECT_TOOL_ARGUMENT_PROPERTIES["files"]
    argument_properties["remove_files"] = PROJECT_TOOL_ARGUMENT_PROPERTIES[
        "remove_files"
    ]
    argument_properties["reason"] = PROJECT_TOOL_ARGUMENT_PROPERTIES["reason"]
    arguments["properties"] = argument_properties
    properties["arguments"] = arguments
    return {**schema, "properties": properties}


def project_write_schema(paths: list[str]) -> dict[str, Any]:
    """A flat step schema whose write target must be a file the build still owes.

    Used for the step after a create_file was refused for aiming at a path that
    already exists. Prose does not fix this: one live build spent 43 create_file
    calls to produce 11 files, re-sending paths it had already staged, and
    another spent eight consecutive steps on a single one. The host knows the
    manifest and the overlay, so it makes the wrong target ungrammatical — the
    same move that took apply_patch from 0/4 to 4/4 correct calls.

    Deliberately not narrowed to create_file alone. The refused path stays in the
    enum and both revision tools stay legal, so a model that meant to *revise*
    the file it just wrote can still do exactly that; what becomes unexpressible
    is only the thing it was measurably getting wrong. Flat, with no `$ref` or
    `anyOf`, so the MLX grammar-collapse protection holds.
    """
    schema = grammar_schema(ProjectAgentStepWireV1)
    properties = dict(schema["properties"])
    properties["status"] = {"type": "string", "enum": ["tool"]}
    properties["tool"] = {
        "type": "string",
        "enum": ["create_file", "apply_patch", "replace_lines"],
    }
    properties["arguments"] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "path": {"type": "string", "enum": list(paths)},
            "content": {"type": "string"},
            "original": {"type": "string"},
            "replacement": {"type": "string"},
            "start_line": {"type": "integer"},
            "end_line": {"type": "integer"},
            "expect": {"type": "string"},
        },
        "required": ["path"],
    }
    return {
        **schema,
        "properties": properties,
        "required": ["status", "tool", "arguments"],
    }


class ProjectSpecV1(Contract):
    """A loose build request compiled into the prescriptive spec that builds well.

    Measured on the same model, same day, same pipeline: a conversational
    build prompt produced 38 blocking findings; its prescriptive rewrite —
    named files, named routes, explicit stack and rules — produced 11. The
    spec is derived context, never a replacement for intent: the original
    request stays the source of truth, and every product decision the user
    did not state is confessed in ``assumptions`` rather than smuggled in.
    """

    spec: str = Field(min_length=1, max_length=8_000)
    assumptions: list[str] = Field(default_factory=list, max_length=8)


# Big enough for a seeded listing or a record's full field set, small enough
# that the plan, the checkpoint, and the sandbox request all stay bounded.
_MAX_EXPECT_JSON_CHARS = 4_000


class AcceptanceScenarioV1(Contract):
    """One machine-checkable claim about what the built app must do.

    The verification ladder proves *structure* — parses, imports, serves —
    and real builds have passed every rung while being non-functional
    demoware. A scenario is the spec's own claim made executable: a method, a
    path, the smallest body that exercises the claim, and what the response
    must look like. The sandbox runs them exactly as written; the card
    reports the ones that failed. Deliberately flat and enum-driven so the
    local grammar can carry it.
    """

    name: str = Field(min_length=1, max_length=120)
    # Keep this enum in parity with the networkless verifier. Collapsing an
    # edit verb to POST changes the claim being tested (and can turn a correct
    # PATCH-only route into a false 405), while collapsing it to GET can make a
    # mutating workflow look healthy without exercising it at all.
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"] = "GET"
    # The request line, query string included — the sandbox replays it verbatim.
    # A route with required query parameters answers 422 to a bare path, so
    # "/convert" where "/convert?value=0&direction=c-to-f" was meant reports a
    # correct app as broken.
    path: str = Field(min_length=1, max_length=300)
    # What rides in the request: nothing, the JSON object in `body`, or one of
    # the verifier's deterministic multipart fixtures. ``text_upload`` is the
    # credentialless TXT path; ``executable_upload`` is a harmless fixed binary
    # signature used only to prove refusal; ``image_upload`` deliberately
    # exercises image handling and may therefore reach an external adapter.
    body_kind: Literal[
        "none",
        "json",
        "text_upload",
        "executable_upload",
        "image_upload",
    ] = "none"
    body: dict[str, Any] = Field(default_factory=dict)
    # "2xx_or_4xx" is the resilient default: it proves the route is alive and
    # validating without guessing which side of validation a minimal body
    # lands on. Only a 5xx or an unhandled exception fails it.
    expect_status: Literal["2xx", "4xx", "2xx_or_4xx"] = "2xx_or_4xx"
    # Substrings the response text must contain, when the claim is about
    # content — extracted fields present, a verdict named, a total computed.
    # Case-insensitive and satisfiable by extra data, which is right for a
    # textual claim and wrong for an exact one: two live models passed a
    # containment check by *appending* a record next to the wrong one they
    # were asked to fix. Use expect_json_exact whenever the request names
    # exact records, counts, field values, or a exact list.
    expect_contains: list[str] = Field(default_factory=list, max_length=8)
    # The response body parsed as JSON must equal this structure exactly.
    # Object keys compare independent of order; arrays keep their order unless
    # json_match says otherwise. Extra keys and extra array elements are
    # failures — that is the whole point. None means the scenario makes no
    # exact claim, which keeps every scenario written before this field
    # behaving exactly as it did.
    expect_json_exact: dict[str, Any] | list[Any] | None = None
    # A closed enum, never inference: "exact" compares arrays positionally,
    # "unordered_array" compares them as multisets (identical members and
    # multiplicity) at every depth. A scenario that needs order-independence
    # must say so, so an exact assertion can never be silently weakened into
    # a set comparison by a planner that found ordering inconvenient.
    json_match: Literal["exact", "unordered_array"] = "exact"

    @field_validator("expect_json_exact")
    @classmethod
    def validate_expect_json_exact(cls, value: Any) -> Any:
        """Keep the assertion small enough to carry and cheap to compare.

        This rides the same local grammar and checkpoint as the rest of the
        plan, and it is replayed inside the networkless sandbox, so an
        unbounded structure is both a context cost and a verifier cost.
        """

        if value is None:
            return None
        try:
            encoded = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ValueError("expect_json_exact must be JSON-serializable") from error
        if len(encoded) > _MAX_EXPECT_JSON_CHARS:
            raise ValueError(
                "expect_json_exact must be at most "
                f"{_MAX_EXPECT_JSON_CHARS} serialized characters"
            )
        return value


# One number, shared by every consumer of a slice's scenario_names. It is the
# PLAN's scenario bound: a slice may legitimately name every scenario the plan
# declared, and a separate, smaller per-slice cap was a silent policy that
# rejected a correct planner reply for naming six of its own six scenarios.
MAX_PLAN_SCENARIOS = 8
# How many files one vertical slice may own and write. The single source for
# the bound: project_slices, plan validation and the sidecar's write-scope wire
# format all agree with this number, and the host synthesizes a whole-manifest
# slice only up to it.
MAX_SLICE_FILES = 8


_ENTRYPOINT_BASENAMES = frozenset(
    {
        "main.py",
        "app.py",
        "server.py",
        "asgi.py",
        "wsgi.py",
        "index.js",
        "index.ts",
        "server.js",
        "server.ts",
        "app.js",
        "app.ts",
    }
)


class ProjectVerticalSliceV1(Contract):
    """One independently buildable, verifier-gated product outcome.

    Files remain globally dependency ordered.  A slice groups one contiguous
    range into a small end-to-end outcome so the coding harness can finish and
    verify useful behaviour without carrying the entire build conversation.
    """

    name: str = Field(min_length=1, max_length=120)
    outcome: str = Field(min_length=1, max_length=600)
    files: list[str] = Field(
        default_factory=list, min_length=1, max_length=MAX_SLICE_FILES
    )
    # owned_files/integration_files are the outcome-based split beneath
    # `files`: a slice that owns nothing but re-opens an earlier slice's file
    # to wire it up isn't a horizontal layer, it's the integration step a
    # true vertical slice needs. Both optional so an older planner reply
    # (which knows only `files`) keeps working exactly as before -- see
    # bind_slice_ownership below, which fills them in from `files` when
    # absent and validates them against `files` when present.
    owned_files: list[str] = Field(default_factory=list)
    integration_files: list[str] = Field(default_factory=list)
    # No max_length here: harmless planner excess (a model naming more
    # scenarios than the host keeps) must be recoverable, not a rejection
    # that discards an otherwise valid multi-file plan. validate_scenario_names
    # runs in "before" mode so it can normalize and trim an oversized list
    # itself, ahead of any core-schema length check that would reject it
    # outright. The five-item host limit and the filter against declared
    # scenario names are enforced together in ProjectBuildPlanV1, which is
    # the only place both a slice's names and the declared scenario list are
    # both in view.
    scenario_names: list[str] = Field(default_factory=list)

    @field_validator("files")
    @classmethod
    def validate_files(cls, value: list[str]) -> list[str]:
        return ProjectBuildPlanV1.validate_files(value)

    @field_validator("owned_files", "integration_files")
    @classmethod
    def validate_ownership_paths(cls, value: list[str]) -> list[str]:
        return ProjectBuildPlanV1.validate_files(value)

    @field_validator("scenario_names", mode="before")
    @classmethod
    def validate_scenario_names(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        seen: list[str] = []
        for item in value:
            name = " ".join(str(item or "").split())[:120]
            if name and name not in seen:
                seen.append(name)
        return seen

    @model_validator(mode="before")
    @classmethod
    def derive_files_from_ownership(cls, value: Any) -> Any:
        """Let the planner author ownership, and derive `files` from it.

        `files` is the union of owned_files and integration_files, so asking a
        model to restate it is asking it to keep two lists in agreement for no
        benefit — and a live planner failed exactly there, returning slices
        whose combined list did not match its own ownership split. Where
        ownership is given it is authoritative and the combined list is
        derived canonically; a redundant model-authored `files` is ignored
        rather than reconciled. A reply that names only `files` (older
        providers, scripted callers, checkpoints) is untouched.
        """

        if not isinstance(value, dict):
            return value
        owned = [
            str(path) for path in value.get("owned_files") or [] if str(path).strip()
        ]
        integration = [
            str(path)
            for path in value.get("integration_files") or []
            if str(path).strip()
        ]
        if owned or integration:
            return {**value, "files": owned + integration}
        return value

    @model_validator(mode="after")
    def bind_slice_ownership(self) -> ProjectVerticalSliceV1:
        declared = set(self.owned_files) | set(self.integration_files)
        if not declared:
            # An older-style reply that only ever named `files`: the whole
            # slice is its own, exclusive work -- exactly today's behavior.
            self.owned_files = list(self.files)
            self.integration_files = []
            return self
        # `files` was derived from these two lists above, so they agree by
        # construction. What still has to hold is that they are disjoint: a
        # file cannot be both this slice's own work and an earlier slice's
        # file it is merely allowed to extend.
        overlap = set(self.owned_files) & set(self.integration_files)
        if overlap:
            raise ValueError(
                f"a file cannot be both owned and an integration file in the "
                f"same slice: {', '.join(sorted(overlap))}"
            )
        return self


class ProjectBuildPlanV1(Contract):
    """The files a build turn commits to writing, named before it starts.

    Completion used to mean only "the model says it is done", and the sole
    guard was whether *anything* had been staged — so a build asked for
    eighteen files could stage five, declare success, and be believed. Fixing
    the list up front turns that into a checkable claim: the host keeps
    ``complete`` out of the grammar until every planned file exists in the
    overlay, and the model is answering against its own plan rather than the
    host's guess at one.

    ``intent`` and ``scope`` are the same move applied to classification. What
    kind of turn this is used to be decided by host regexes over the user's
    wording — which read "what does app/main.py do?" as a build because it
    contains a path, and read a whole-application rewire as narrow because it
    lacked an indefinite article. The model answering here has the request AND
    the repository map in front of it, so it is simply better placed to say.
    The regexes survive as the fallback for a provider that cannot answer.
    """

    # What this turn is. "build" and "edit" both write files and both earn the
    # strict finish gate; only "question" turns it off, and a question that
    # names files is still a question.
    intent: Literal["build", "edit", "question"] = "build"
    # Whether this stands up a whole application (which earns the appkit
    # scaffold) or works inside one that already exists.
    scope: Literal["whole_app", "narrow"] = "narrow"
    files: list[str] = Field(default_factory=list, max_length=24)
    # The acceptance scenarios that make "done" checkable against the spec
    # rather than against the model's summary. Optional: an empty list keeps
    # the ladder exactly as it was.
    scenarios: list[AcceptanceScenarioV1] = Field(
        default_factory=list, max_length=MAX_PLAN_SCENARIOS
    )
    # Preferred execution units for ClineCore. Each valid slice is contiguous
    # in ``files`` and contains at most six paths. Older providers/checkpoints
    # may omit this; the host then derives conservative bounded slices.
    slices: list[ProjectVerticalSliceV1] = Field(default_factory=list, max_length=8)

    @field_validator("files")
    @classmethod
    def validate_files(cls, value: list[str]) -> list[str]:
        """Keep only plausible project-relative paths, in the order given.

        The "./" prefix is removed a prefix at a time, not with lstrip: lstrip
        takes a character set, so it would turn ".gitignore" into "gitignore" —
        a planned path that could then never equal the key the overlay stores,
        leaving the gate waiting forever for a file that was already written.
        """
        seen: list[str] = []
        for item in value:
            path = " ".join(str(item or "").split())[:400]
            while path.startswith("./"):
                path = path[2:]
            if not path or path.startswith("/") or ".." in path or path in seen:
                continue
            seen.append(path)
        return seen

    @model_validator(mode="after")
    def bind_slice_scenario_names(self) -> ProjectBuildPlanV1:
        """Recover harmless excess in a slice's scenario_names, in place.

        A slice naming a scenario the plan never declared is harmless excess
        that must not discard an otherwise valid multi-file plan, so unknown
        names are filtered out. Every DECLARED name is kept: a slice may
        legitimately be responsible for all of the plan's scenarios, and the
        old five-item trim silently dropped commitments the planner made —
        which is what refused a correct six-scenario Atlas plan.

        Deduplicated, order preserved, bounded only by the plan's own
        scenario limit, which the declared list already satisfies.
        """
        if not self.slices:
            return self
        declared = {scenario.name for scenario in self.scenarios}
        for slice_ in self.slices:
            slice_.scenario_names = list(
                dict.fromkeys(
                    name for name in slice_.scenario_names if name in declared
                )
            )[:MAX_PLAN_SCENARIOS]
        return self

    @model_validator(mode="after")
    def bind_slice_ownership_and_entrypoint(self) -> ProjectBuildPlanV1:
        """Reject a declared multi-slice plan that cannot actually be built as
        independent vertical slices, falling back to the host's safe
        deterministic chunking rather than executing a plan that would either
        violate an earlier slice's immutability or never produce anything
        runnable.

        Two provable failure shapes, both cleared the same way — to an empty
        slice list, which vertical_build_slices (project_slices.py) already
        treats as "no valid declared plan, chunk the dependency order
        instead":

        - An integration_files reference to a file no STRICTLY EARLIER slice
          owns. Without this check, "integration" would just be a second name
          for "edit whatever you want": the whole point of the exclusive-
          ownership default (ProjectVerticalSliceV1.bind_slice_ownership) is
          that a later slice touches an earlier file only where the earlier
          slice explicitly allowed it.
        - A whole-application plan whose first slice owns or integrates no
          recognizable application entrypoint. The measured failure this
          closes: a "Foundation: config, database, models" first slice with
          a Health-route scenario attached that could not actually run yet,
          because app/main.py was planned four slices later. A bootstrap
          slice is acceptable only when it can produce a checkable runnable
          outcome, such as the application importing and serving /health.
        """
        if not self.slices:
            return self
        # The top-level manifest is authoritative. Now that each slice's
        # combined `files` is DERIVED from its ownership, the ownership split
        # is the only thing standing between the planner and what gets built,
        # so it must reproduce that manifest exactly -- same paths, same
        # count, same order. Anything that loses, duplicates, invents or
        # reorders a planned file is cleared to the safe deterministic
        # partition rather than executed.
        owned_sequence = [path for item in self.slices for path in item.owned_files]
        if owned_sequence != self.files:
            self.slices = []
            return self
        owned_so_far: set[str] = set()
        for slice_ in self.slices:
            if set(slice_.integration_files) - owned_so_far:
                self.slices = []
                return self
            owned_so_far |= set(slice_.owned_files)
        if self.scope == "whole_app":
            first = self.slices[0]
            first_files = set(first.owned_files) | set(first.integration_files)
            if not any(
                path.rsplit("/", 1)[-1] in _ENTRYPOINT_BASENAMES for path in first_files
            ):
                self.slices = []
        return self


class ProjectDirectionV1(Contract):
    """One file's instruction, written by the orchestrator for the coder.

    This is the hard gate between deciding scope and writing code, expressed as
    a contract. The orchestrator answers here and has no tools at all, so it
    *cannot* write; the coder receives this and is given one path with reads
    closed, so it cannot decide what the turn is about. Measured on the case
    that motivated it: a model handed the whole four-file conversion planned it
    correctly, was narrowed to one file, and still read to the ceiling without
    writing — it was never short of context, it was free to keep choosing to
    look. Nothing here gives it that choice.

    Flat on purpose, like every other locally-decoded contract: no ``$ref``, no
    ``anyOf``, because the nested forms are what collapse MLX decode.
    """

    # Which file the coder must write next. The host checks it against the plan
    # and against a per-file attempt cap, so naming the same path twice is a
    # repair rather than an escape — and naming a path outside the plan is
    # refused rather than obeyed.
    path: str = ""
    # What to write, imperatively. This replaces the fixed sentence the host
    # used to send, which could say "write this file" and never what it should
    # contain.
    # Measured, not guessed: real orchestrators write 3,500–6,000 characters
    # here, so a 6,000 cap sat exactly on the distribution and rejected the
    # longest — and a rejected direction is indistinguishable from a model that
    # cannot follow the schema. The brief IS the coder's entire knowledge of the
    # work, so headroom is cheaper than truncation.
    instruction: str = Field(default="", max_length=16_000)
    # Symbols and modules the project already has that this file must compose
    # rather than reinvent — the repo map's whole purpose, made specific.
    reuse: list[str] = Field(default_factory=list, max_length=12)
    # Files the HOST should fetch and hand to the coder. The point of the split
    # is that the coder does not spend steps looking: if something must be read,
    # the orchestrator says so and the host pays for it once.
    read: list[str] = Field(default_factory=list, max_length=6)
    # The plan is satisfied, or cannot be carried further. Ends the directed
    # arc and hands the turn back to the ordinary loop to finish honestly.
    done: bool = False
    # Why — shown in the timeline, and the reason a `done` before the plan is
    # complete is legible rather than mysterious.
    reason: str = Field(default="", max_length=2_000)

    @field_validator("reuse", "read", mode="before")
    @classmethod
    def bound_lists(cls, value: Any, info: ValidationInfo) -> Any:
        """Trim an over-long list instead of rejecting the whole direction.

        The bounds exist so one reply cannot flood the coder's brief, and the
        thirteenth item is genuinely not worth a step. But `max_length` REJECTS,
        and a rejected direction is indistinguishable from a model that cannot
        follow the schema — measured, when a model listed thirteen things to
        reuse and lost the entire instruction along with the thirteenth. Trim
        what is over the line; keep everything the bound was protecting.
        """
        if isinstance(value, list):
            # Each field's own bound, so trimming can never itself trip the
            # limit it is protecting.
            return value[: 12 if info.field_name == "reuse" else 6]
        return value

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        path = " ".join(str(value or "").split())[:400]
        while path.startswith("./"):
            path = path[2:]
        if path.startswith("/") or ".." in path:
            return ""
        return path


class ProjectAgentStepWireV1(Contract):
    """The flat wire form of a project step, shaped for grammar-constrained decode.

    A local model's structured output is constrained by a GBNF grammar the
    runtime derives from this schema. The nested ``tool_call`` union in
    ``ProjectAgentStepV1`` becomes an ``anyOf`` over a ``$ref`` — and on the MLX
    backend that grammar collapses a non-trivial reply to empty output, which is
    what ended real build turns after three "unreadable" steps. This form is
    deliberately flat: no ``$ref``, no ``anyOf``, so the grammar is simple enough
    for the model to satisfy, and the tool name is a closed enum the grammar
    itself enforces. The host converts it back to the nested step it uses
    everywhere else, so nothing downstream sees the wire shape.
    """

    status: Literal["tool", "complete"]
    # "" is how a completion carries no tool; the real tool names are the
    # only other values the grammar will emit. A test pins this list to
    # PROJECT_TOOL_REQUIRED_ARGUMENTS — a tool present in one and not the
    # other is advertised-but-unusable, which is how inspect_api spent months
    # implemented, catalogued, and impossible for any provider to call.
    tool: Literal[
        "",
        "list_files",
        "search_code",
        "read_file",
        "apply_patch",
        "replace_lines",
        "create_file",
        "run_check",
        "inspect_api",
        "ask_user",
        "respond",
        "revise_plan",
    ] = ""
    arguments: dict[str, Any] = Field(
        default_factory=dict, json_schema_extra=_CLOSED_TOOL_ARGUMENTS
    )
    response: str = Field(default="", max_length=40_000)
    learnings: list[str] = Field(default_factory=list, max_length=16)

    def to_step(self) -> "ProjectAgentStepV1":
        """The nested ``ProjectAgentStepV1`` this flat reply stands for."""
        if self.status == "tool":
            if not self.tool:
                raise ValueError("a project tool step requires a tool name")
            return ProjectAgentStepV1(
                status="tool",
                response=self.response,
                learnings=self.learnings,
                tool_call=ProjectToolCallV1(name=self.tool, arguments=self.arguments),
            )
        if not self.response.strip():
            raise ValueError("a completed project step requires a response")
        return ProjectAgentStepV1(
            status="complete", response=self.response, learnings=self.learnings
        )


class ProjectBuildStepWireV1(Contract):
    """The build-turn narrowing of the flat wire step: a completion is unexpressible.

    ``ProjectAgentStepWireV1`` lets the model finish on token one — ``status`` is
    the first field and ``status="complete"`` needs nothing but a non-empty
    ``response`` (``to_step`` above), so a grammar-constrained decoder can satisfy
    a build request by *describing* files it never wrote. The rule against that
    lived only in prose the decoder is free to ignore, which is why the failure
    reproduced even on a non-reasoning model.

    This schema removes the empty-completion branch from the grammar itself, for
    the one population where it is the bug: a build instruction that has staged
    nothing. ``status`` is pinned to ``"tool"`` and ``tool`` drops its empty
    member, so the model's first legal object must name a real tool and carry
    arguments — it has to write or inspect a file before it can talk about one.
    The shape stays flat (no ``$ref``/``anyOf``) so the MLX grammar-collapse
    protection documented on ``ProjectAgentStepWireV1`` carries over, and the
    permissive schema returns as soon as one file is staged, where finishing is
    legitimately allowed again.
    """

    status: Literal["tool"] = "tool"
    # ask_user is here — a mid-build model may genuinely be blocked on the
    # user — but respond is deliberately NOT: a prose exit from a build turn
    # is exactly the escape this narrowed grammar exists to close off.
    # revise_plan IS here, and belongs here most of all: this is the grammar a
    # build wears while it still owes files, so it is exactly where a model
    # that has just proved the plan wrong needs a legal move that is not
    # writing the wrong file.
    tool: Literal[
        "list_files",
        "search_code",
        "read_file",
        "apply_patch",
        "replace_lines",
        "create_file",
        "run_check",
        "inspect_api",
        "ask_user",
        "revise_plan",
    ]
    arguments: dict[str, Any] = Field(
        default_factory=dict, json_schema_extra=_CLOSED_TOOL_ARGUMENTS
    )
    response: str = Field(default="", max_length=40_000)
    learnings: list[str] = Field(default_factory=list, max_length=16)

    def to_step(self) -> "ProjectAgentStepV1":
        """The nested tool step this build-turn reply stands for."""
        return ProjectAgentStepV1(
            status="tool",
            response=self.response,
            learnings=self.learnings,
            tool_call=ProjectToolCallV1(name=self.tool, arguments=self.arguments),
        )


class DiagramCodeV1(Contract):
    """North's constrained source-code result for the architecture sandbox."""

    schema_version: Literal["1"] = "1"
    diagram_code: str = Field(min_length=1, max_length=100_000)

    @field_validator("diagram_code")
    @classmethod
    def validate_diagram_code(cls, value: str) -> str:
        if "\x00" in value or "\r" in value:
            raise ValueError(
                "diagram_code must use LF line endings and contain no NUL bytes"
            )
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
        # A single legacy per-file write. New project turns stage their writes
        # and raise one project_apply_build instead; the kind survives so an
        # approval persisted before the upgrade still resumes.
        "project_write",
        "project_apply_build",
        "project_verify",
        "network",
        "dependency",
        # Closing out work from the chat. The record only ever changes through
        # this gate: the model proposes, the user approves, the host applies.
        "queue_update",
    ]
    title: str
    summary: str
    risk_level: RiskLevel
    proposal_id: str | None = None
    tool_version_id: str | None = None
    input_digest: str
    permissions: list[str] = Field(default_factory=list)
    # Set when the host has already proven this action cannot work — a staged
    # build with a file that will not parse or will not import. Approving is
    # refused while it is set, because a warning at the top of a card is only
    # as strong as the attention of whoever is reading it, and the one that
    # mattered was approved three minutes after it was raised. Rejecting and
    # sending a follow-up stay available. Optional and defaulted so approvals
    # persisted before this field round-trip unchanged.
    blocked_reason: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class ApprovalDecisionV1(Contract):
    approval_id: str | None = None
    decision: Decision
    reason: str | None = Field(default=None, max_length=2000)


class ElicitationRequestV1(Contract):
    """A question the model paused the run to ask (ask_user), shown as the inline
    card near the composer. Mirrors ApprovalRequestV1's lifecycle: persisted,
    emitted as an `elicitation.requested` run event, and resolved through one
    answer endpoint that resumes the same turn with the answer as the tool result.
    """

    id: str
    run_id: str
    question: str = Field(min_length=1, max_length=2000)
    # The offered choices. Empty is allowed (a pure free-text prompt). The card
    # renders one button per option.
    options: list[str] = Field(default_factory=list, max_length=8)
    # Whether the card also offers a free-text field ("Other…"). When there are
    # no options this is forced on by the endpoint that builds the request.
    allow_text: bool = True
    created_at: datetime = Field(default_factory=utc_now)


class ElicitationAnswerV1(Contract):
    """The user's reply to an ElicitationRequestV1. At least one of option/text
    must be present; the resume endpoint enforces that and feeds it back as the
    ask_user tool result."""

    elicitation_id: str | None = None
    option: str | None = Field(default=None, max_length=2000)
    text: str | None = Field(default=None, max_length=4000)


class RecoverableRunV1(Contract):
    run: RunV1
    approval: ApprovalRequestV1 | None = None
    elicitation: ElicitationRequestV1 | None = None


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


class MemoryCandidateV1(Contract):
    """A durable fact a finished run suggests remembering.

    A candidate is a *proposal*, never an activation: it still travels the
    existing approve-first path before it can influence a later turn.
    """

    content: str = Field(min_length=8, max_length=600)
    kind: Literal["user", "project", "skill"] = "project"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class MemoryHarvestV1(Contract):
    candidates: list[MemoryCandidateV1] = Field(default_factory=list, max_length=10)


class MemoryConsentV1(Contract):
    """Opt long-term memory into cloud embedding, or withdraw and purge it."""

    consent: bool = False
    reason: str | None = Field(default=None, max_length=2000)


class MemoryIndexStatusV1(Contract):
    """Whether memory is retrieved by meaning or by keyword, and why.

    `semantic` is the honest answer to "is this actually on": consent alone is
    not enough if the cloud path is unreachable or nothing has been embedded yet.
    """

    consent: bool = False
    consent_reason: str | None = Field(default=None, max_length=2000)
    cloud_available: bool = False
    semantic: bool = False
    active: int = Field(default=0, ge=0)
    embedded: int = Field(default=0, ge=0)


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
    provider: Literal["local", "notion", "web", "customer", "answer"] = "local"
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


class RoleChainEntryV1(Contract):
    """One rung of a role's model ladder: a lane, and the model on it.

    ``model`` matters on the local (Ollama) lane, which serves many models,
    and on the Cline lane, where naming one overrides that seat's configured
    default; the OCI and Cohere lanes each run their configured model and
    ignore it. Kept as two fields rather than a parsed string so the UI can
    render lanes and models as separate controls.
    """

    provider: Literal["local", "oci", "cohere", "cline"] = "local"
    model: str | None = Field(default=None, max_length=200)


# The three routing roles a preference may chain. Spelled once: the store
# validates against it, resolve_aliases iterates it, and the UI renders it.
MODEL_ROLES: tuple[str, ...] = ("planner", "coder", "quality")


class ModelPreferenceV1(Contract):
    mode: Literal["split", "pinned"] = "split"
    model: str | None = None
    provider: Literal["local", "oci", "cohere", "cline"] = "local"
    oci_tools: list[Literal["x_search", "code_interpreter"]] = Field(
        default_factory=lambda: ["code_interpreter"], max_length=2
    )
    # Per-role fallback ladders, first entry primary. Empty means "no explicit
    # chain": the role runs exactly as it always has, plus the synthesized
    # safety fallback resolve_aliases documents. Bounded: a ladder longer than
    # four is a configuration smell, not resilience.
    role_chains: dict[str, list[RoleChainEntryV1]] = Field(default_factory=dict)
    oci_available: bool = False
    cohere_available: bool = False
    cline_available: bool = False
    # Subscription-backed models the backend has verified against the project
    # tool contract. Paid Cline gateway routes are still valid explicit chain
    # entries, but are not advertised as ClinePass choices.
    cline_models: list[str] = Field(default_factory=list, max_length=32)

    @field_validator("role_chains")
    @classmethod
    def validate_role_chains(
        cls, value: dict[str, list[RoleChainEntryV1]]
    ) -> dict[str, list[RoleChainEntryV1]]:
        for role, chain in value.items():
            if role not in MODEL_ROLES:
                raise ValueError(f"unknown model role: {role}")
            if len(chain) > 4:
                raise ValueError(f"{role} chain is longer than 4 entries")
        return value


class ModelPreferenceUpdateV1(Contract):
    mode: Literal["split", "pinned"]
    model: str | None = Field(default=None, max_length=200)
    provider: Literal["local", "oci", "cohere", "cline"] = "local"
    oci_tools: list[Literal["x_search", "code_interpreter"]] = Field(
        default_factory=lambda: ["code_interpreter"], max_length=2
    )
    # None means "leave the stored chains as they are"; an empty dict clears
    # them. The distinction lets the existing UI save mode/model without
    # knowing chains exist.
    role_chains: dict[str, list[RoleChainEntryV1]] | None = None


class SpeechPreferenceV1(Contract):
    """How Metis listens, and how it speaks back.

    Kept apart from the model preference because these are different
    decisions: dictating through one service says nothing about which model
    should answer a chat message. The availability flags travel with the
    choice so a provider with no key behind it reads as unavailable rather
    than as something that saves and then fails.
    """

    stt_provider: Literal["cohere", "elevenlabs"] = "cohere"
    # Read the exact record back and wait for a yes before committing a voice
    # write. Off by default: the visual receipt and its undo are the standing
    # safety mechanism, and a spoken read-back on every append taxes the
    # ordinary case to catch the rare one.
    spoken_confirmation: bool = False
    # Which model reasons during a voice conversation, from the server-owned
    # allowlist below. Separate from the chat model on purpose — the model
    # that should answer a spoken question in two seconds is not the one that
    # should write a build plan.
    voice_model: str = ""
    cohere_available: bool = False
    elevenlabs_available: bool = False
    voice_models: list[str] = Field(default_factory=list, max_length=16)


class SpeechPreferenceUpdateV1(Contract):
    stt_provider: Literal["cohere", "elevenlabs"]
    spoken_confirmation: bool = False
    # None keeps the stored model; "" clears it back to the configured default.
    voice_model: str | None = Field(default=None, max_length=200)


# What one voice turn can be. `read` and `clarify` are the answering intents;
# the refusals are decided before any model is called, and `navigation` is the
# one thing a client tool may do (open a page — never mutate).
VoiceIntent = Literal[
    "read",
    "clarify",
    "navigation",
    "customer_append",
    "refuse_build",
    "refuse_protected",
]

VoiceRecordType = Literal["note", "fact", "action", "person", "win"]


class VoiceWriteCandidateV1(Contract):
    """The only thing a model may propose for a voice write.

    Fields, and not one of them decisive. The record type was chosen by the
    host from the utterance, the account was resolved by the host, and the
    transcript excerpt is copied by the host verbatim — what the model
    contributes is the shape of the text, which the host then validates and
    bounds. There is no account, no record type, and no id on this contract,
    because a model that could name any of those could redirect a write.
    """

    title: str = Field(default="", max_length=400)
    body: str = Field(default="", max_length=2_000)
    kind: str = Field(default="", max_length=400)
    content: str = Field(default="", max_length=400)
    description: str = Field(default="", max_length=400)
    owner: str = Field(default="", max_length=400)
    name: str = Field(default="", max_length=400)
    role: str = Field(default="", max_length=400)
    organization: str = Field(default="", max_length=400)
    brief: str = Field(default="", max_length=400)


class VoiceWriteReceiptV1(Contract):
    """The immutable record of one thing voice added.

    It survives an undo. A reversal sets `undone_at` and nothing else, because
    "this was added and then taken back" is a different fact from "this never
    happened", and only one of them is true.
    """

    id: str
    record_type: VoiceRecordType
    record_id: str
    account_id: str
    account_name: str
    source: Literal["voice"] = "voice"
    voice_session_id: str = ""
    provider_conversation_id: str = ""
    turn_id: str = ""
    run_id: str = ""
    transcript_excerpt: str = Field(default="", max_length=8_000)
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    undone_at: datetime | None = None
    undone_by: str = ""
    # What the card says. Quotes the committed text rather than describing it,
    # so a wrong record is visibly wrong at a glance.
    summary: str = Field(default="", max_length=400)
    # Single-use, short-lived, and bound to this receipt. Returned once, with
    # the write; a receipt read back later carries no token.
    undo_token: str = Field(default="", max_length=120)


class MeetingWordV1(Contract):
    """One word and when it was said. `None` where the provider gave nothing."""

    text: str = Field(default="", max_length=200)
    start: float | None = None
    end: float | None = None


class MeetingTurnV1(Contract):
    """One contiguous stretch of one speaker.

    `original_text` is what the provider heard and is never overwritten, so a
    correction can always be read against it.
    """

    id: str
    ordinal: int = Field(default=0, ge=0)
    speaker_id: str = ""
    text: str = ""
    original_text: str = ""
    start_seconds: float = 0.0
    end_seconds: float = 0.0
    words: list[MeetingWordV1] = Field(default_factory=list)
    corrected_at: datetime | None = None


class MeetingSpeakerV1(Contract):
    speaker_id: str
    display_name: str = Field(default="", max_length=120)
    person_id: str | None = None


class MeetingProposalV1(Contract):
    """Something the meeting suggests, which nobody has agreed to yet.

    Every one carries the turn and the seconds that produced it, so accepting
    it is a four-second check against the audio rather than an act of faith.
    """

    id: str
    kind: Literal["account_link", "action", "decision"]
    payload: dict[str, Any] = Field(default_factory=dict)
    status: Literal["proposed", "accepted", "rejected"] = "proposed"
    score: float | None = None
    evidence_turn_id: str | None = None
    evidence_start: float | None = None
    evidence_end: float | None = None
    created_at: datetime
    decided_at: datetime | None = None


class MeetingEventV1(Contract):
    """One observable step of the ingestion job."""

    stage: str
    message: str = ""
    created_at: datetime


class MeetingV1(Contract):
    id: str
    title: str = ""
    audio_sha256: str = ""
    audio_filename: str = ""
    audio_media_type: str = ""
    audio_bytes: int = Field(default=0, ge=0)
    isolated_sha256: str | None = None
    duration_seconds: float | None = None
    language: str = ""
    transcript: str = ""
    provider_request_id: str = ""
    stage: Literal[
        "uploaded", "isolating", "transcribing", "analyzing", "ready", "failed"
    ] = "uploaded"
    error: str = ""
    attempts: int = Field(default=0, ge=0)
    account_id: str | None = None
    linked_at: datetime | None = None
    link_score: float | None = None
    created_at: datetime
    updated_at: datetime


class MeetingDetailV1(Contract):
    meeting: MeetingV1
    turns: list[MeetingTurnV1] = Field(default_factory=list)
    speakers: list[MeetingSpeakerV1] = Field(default_factory=list)
    proposals: list[MeetingProposalV1] = Field(default_factory=list)
    events: list[MeetingEventV1] = Field(default_factory=list)
    # People on the linked account, offered as speaker names. Suggestions only:
    # naming a speaker is a person's judgment about a voice they recognize.
    suggested_names: list[str] = Field(default_factory=list, max_length=40)


class MeetingSpeakerUpdateV1(Contract):
    display_name: str = Field(default="", max_length=120)
    person_id: str | None = None


class MeetingTurnCorrectionV1(Contract):
    text: str = Field(min_length=1, max_length=4_000)


class MeetingProposalDecisionV1(Contract):
    status: Literal["accepted", "rejected"]


class InterviewContextV1(Contract):
    """Everything the setup screen collects, and all the agent may ever know.

    The job description is pasted, untrusted text. It travels to the agent as
    a dynamic variable whose prompt marks it as source material to quote from,
    never instructions to follow — the same posture pasted text takes
    everywhere else in Metis.
    """

    job_title: str = Field(min_length=1, max_length=200)
    company_name: str = Field(min_length=1, max_length=200)
    job_description: str = Field(min_length=1, max_length=30_000)
    interview_type: Literal["hr_recruiter", "hiring_manager", "technical"]


class InterviewTurnInV1(Contract):
    """One transcript line as the browser heard it, ordered by ordinal."""

    ordinal: int = Field(ge=0, le=9_999)
    role: Literal["user", "agent"]
    text: str = Field(min_length=1, max_length=8_000)


class InterviewTurnsAppendV1(Contract):
    turns: list[InterviewTurnInV1] = Field(min_length=1, max_length=200)


class InterviewTurnV1(Contract):
    id: str
    ordinal: int
    role: Literal["user", "agent"]
    text: str
    created_at: datetime


class InterviewTurnsReceiptV1(Contract):
    """How many turns the session now holds. Re-sends are absorbed, not doubled."""

    stored: int = Field(ge=0)


class InterviewImprovementV1(Contract):
    what_happened: str = Field(min_length=1, max_length=2_000)
    evidence: str = Field(min_length=1, max_length=2_000)
    why_it_hurt: str = Field(min_length=1, max_length=2_000)
    better_approach: str = Field(min_length=1, max_length=2_000)


class InterviewEvaluationV1(Contract):
    """What the agent's blocking tool submits. Deliberately no overall score:
    the model hands over five criterion scores and Metis does the arithmetic,
    so the number spoken in the debrief is always the number stored here.
    """

    specific_evidence: int = Field(ge=1, le=10)
    role_depth: int = Field(ge=1, le=10)
    relevance: int = Field(ge=1, le=10)
    structure: int = Field(ge=1, le=10)
    communication: int = Field(ge=1, le=10)
    verdict: str = Field(min_length=1, max_length=500)
    strongest_answer_quote: str = Field(min_length=1, max_length=2_000)
    strongest_answer_reason: str = Field(min_length=1, max_length=2_000)
    improvements: list[InterviewImprovementV1] = Field(min_length=3, max_length=3)
    drill: str = Field(min_length=1, max_length=2_000)
    completed_question_count: int = Field(ge=0, le=5)
    incomplete: bool = False

    @model_validator(mode="after")
    def _flags_must_agree(self) -> "InterviewEvaluationV1":
        # A "complete" evaluation of fewer than five answers is the exact
        # inconsistency that lets an early exit masquerade as a full round.
        if not self.incomplete and self.completed_question_count != 5:
            raise ValueError(
                "a complete interview has five answered questions; "
                "mark it incomplete or fix completed_question_count"
            )
        if self.incomplete and self.completed_question_count >= 5:
            raise ValueError(
                "an interview with five answered questions is not incomplete"
            )
        return self


class InterviewScorecardV1(Contract):
    """The stored verdict: the submitted evaluation plus what Metis computed.

    overall_score and recommendation are None when fewer than three questions
    were answered — too little evidence to score, and no number is honest.
    """

    evaluation: InterviewEvaluationV1
    overall_score: float | None = Field(default=None, ge=1.0, le=10.0)
    recommendation: Literal["advance", "borderline", "do_not_advance"] | None = None
    provisional: bool = False
    created_at: datetime


class InterviewSessionV1(Contract):
    id: str
    provider_conversation_id: str = Field(default="", max_length=200)
    job_title: str
    company_name: str
    job_description: str
    interview_type: Literal["hr_recruiter", "hiring_manager", "technical"]
    question_limit: int = 5
    status: Literal["active", "complete", "ended_early", "failed"]
    turns: list[InterviewTurnV1] = Field(default_factory=list)
    scorecard: InterviewScorecardV1 | None = None
    created_at: datetime
    updated_at: datetime


class InterviewSessionStartV1(Contract):
    """The one response carrying the short-lived conversation credential.

    A WebRTC conversation token minted server-side and returned once; the
    ElevenLabs API key never leaves the trusted process. The dynamic
    variables ride alongside because the SDK sends them at session start —
    question_limit is fixed here, not accepted from the browser.
    """

    session: InterviewSessionV1
    conversation_token: str = Field(default="", max_length=4_000)
    dynamic_variables: dict[str, str] = Field(default_factory=dict)


class InterviewSessionUpdateV1(Contract):
    """The one field the browser learns after start: ElevenLabs' own id."""

    provider_conversation_id: str = Field(min_length=1, max_length=200)


class InterviewEndV1(Contract):
    reason: Literal["completed", "ended_early", "failed"] = "ended_early"


class InterviewAvailabilityV1(Contract):
    """Whether an interview can start, and every missing piece when it cannot."""

    available: bool = False
    reason: str = Field(default="", max_length=400)
    missing: list[str] = Field(default_factory=list, max_length=8)


class VoiceUndoV1(Contract):
    """Reverse exactly the record one receipt created.

    The receipt id and its token are the whole request. There is deliberately
    no record type and no record id: a caller that could name those could
    point undo at a row voice never wrote.
    """

    undo_token: str = Field(min_length=1, max_length=120)


class VoiceCitationV1(Contract):
    """One source behind a spoken answer, in the form a receipt can show."""

    label: str = Field(min_length=1, max_length=200)
    reference: str = Field(default="", max_length=400)
    provider: Literal["local", "notion", "web", "customer", "answer"] = "local"


class VoiceAnswerV1(Contract):
    """The only structured reply the voice model is allowed to author.

    Two strings and nothing else. There is no action name, no tool, no record
    and no permission on this contract, so a model cannot widen what the turn
    is allowed to do by what it returns — the ceiling is re-evaluated after
    this comes back, and there is nothing here for it to disagree with.
    """

    written: str = Field(default="", max_length=20_000)
    spoken: str = Field(default="", max_length=1_200)


class VoiceRenditionV1(Contract):
    """One voice turn's two outputs, and the provenance of both.

    `written` is the full grounded answer; it travels to the browser over the
    loopback stream and lands in the conversation. `spoken` is the short
    rendition, and it is the *only* text that goes back through the speech
    provider. They are separate fields rather than one truncated string
    because they answer different questions: what can be read, and what can be
    heard once, without scrolling back.
    """

    written: str = Field(default="", max_length=20_000)
    spoken: str = Field(default="", max_length=1_200)
    citations: list[VoiceCitationV1] = Field(default_factory=list, max_length=12)
    intent: VoiceIntent = "read"
    # Present only on a committed append, and the reason the surface can show a
    # card with an Undo on it the moment the words stop.
    write: VoiceWriteReceiptV1 | None = None
    # The finalized utterance this answered, verbatim. Carried so a refusal can
    # hand the composer exactly what was said rather than a paraphrase.
    transcript: str = Field(default="", max_length=8_000)
    voice_session_id: str = ""
    turn_id: str = ""
    run_id: str = ""
    # Set when the answer was shortened or normalized by the host rather than
    # written that way, so the surface can be honest about which it is showing.
    spoken_fallback: bool = False


class VoiceSessionV1(Contract):
    """One open voice session, as the browser sees it.

    Deliberately without the signed conversation URL: that is minted once, at
    start, and returned once. A status poll that kept handing it back would
    turn a short-lived credential into an ambient one.
    """

    id: str
    conversation_id: str
    state: Literal["starting", "live", "ending", "ended", "failed"] = "starting"
    provider_conversation_id: str = ""
    started_at: datetime
    lease_expires_at: datetime
    ended_at: datetime | None = None
    # Why it ended, in the user's language: "you stopped it", "it timed out".
    reason: str = ""
    spoken_confirmation: bool = False
    voice_model: str = ""
    # Billable wall-clock so far. Counted by the host from its own lease, not
    # from anything the browser reports.
    elapsed_seconds: float = Field(default=0.0, ge=0.0)


class VoiceSessionStartV1(Contract):
    """The one response that carries the short-lived conversation credential.

    A WebRTC conversation token, not a signed WebSocket URL — the two are
    different endpoints on the provider and the SDK accepts one or the other,
    never both. It is minted server-side, returned once, and expires on its
    own; the API key behind it never leaves the trusted process.
    """

    session: VoiceSessionV1
    conversation_token: str = Field(default="", max_length=4_000)
    lease_seconds: int = Field(default=60, ge=10, le=3_600)


class VoiceAvailabilityV1(Contract):
    """Whether voice can start, and the plain-language reason when it cannot."""

    available: bool = False
    reason: str = Field(default="", max_length=400)
    # The Custom LLM URL to paste into the ElevenLabs agent, assembled from the
    # configured tunnel hostname. Shown rather than guessed at: getting this
    # wrong is the most likely single reason a correctly-built voice mode does
    # not answer, and the value is a public hostname, not a secret.
    custom_llm_url: str = Field(default="", max_length=400)
    public_model_alias: str = Field(default="", max_length=120)
    # All unmet prerequisites, in setup order. ``reason`` remains for older
    # clients and is the first item in this list.
    missing: list[str] = Field(default_factory=list, max_length=12)


class VoiceTurnRequestV1(Contract):
    """What the isolated ingress forwards: one utterance and its context.

    Notice what is absent. There is no model, no system prompt, no account, no
    permission and no action name — nothing the far end of a tunnel could set
    that would change what this turn is allowed to do.
    """

    provider_conversation_id: str = Field(default="", max_length=120)
    # The browser puts this into ElevenLabs' custom-LLM extra body. It binds a
    # provider call to the exact tab that opened it instead of guessing by age.
    metis_session_id: str = Field(default="", max_length=120)
    transcript: str = Field(min_length=1, max_length=8_000)
    history: list[str] = Field(default_factory=list, max_length=12)


class VoicePostCallV1(Contract):
    """A verified ElevenLabs post-call webhook, as the ingress relays it."""

    event_type: str = Field(default="post_call", max_length=80)
    provider_conversation_id: str = Field(default="", max_length=120)
    body_sha256: str = Field(default="", max_length=64)
    # The provider's own payload, kept as untrusted evidence. Nothing in here
    # becomes a customer fact, an action, a memory or an account link without
    # a person accepting it.
    payload: dict[str, Any] = Field(default_factory=dict)


class LocalModelOptionV1(Contract):
    id: str
    name: str
    size_bytes: int = Field(default=0, ge=0)
    parameter_size: str = ""
    quantization: str = ""
    context_length: int | None = Field(default=None, ge=0)
    loaded: bool = False
    # What this model actually occupies right now, reported by the runtime. It
    # starts at the weight size and grows as the KV cache fills, so it is the
    # honest number to show rather than an estimate from the file size.
    resident_bytes: int = Field(default=0, ge=0)
    expires_at: datetime | None = None
    owned_by_metis: bool = False


class LocalModelSessionV1(Contract):
    state: Literal["off", "loading", "ready", "busy", "error"] = "off"
    selected_model: str | None = None
    idle_timeout_seconds: int = Field(default=300, ge=60, le=86_400)
    context_window: int = Field(default=32_768, ge=4_096, le=262_144)
    expires_at: datetime | None = None
    owned_by_metis: bool = False
    busy_count: int = Field(default=0, ge=0)
    error: str | None = None
    # Live unified-memory footprint of everything Ollama has loaded, against the
    # machine's total, so the cost of a running model is visible while it runs.
    resident_bytes: int = Field(default=0, ge=0)
    total_memory_bytes: int = Field(default=0, ge=0)
    models: list[LocalModelOptionV1] = Field(default_factory=list)


class LocalModelSessionLaunchV1(Contract):
    model: str = Field(min_length=1, max_length=200)
    idle_timeout_seconds: Literal[60, 300, 900, 1800, 86400] = 300
    context_window: Literal[8192, 16384, 32768, 65536, 131072] = 32768


class LocalModelSessionStopV1(Contract):
    force: bool = False


class CustomerAccountCreateV1(Contract):
    name: str = Field(min_length=1, max_length=200)
    aliases: list[str] = Field(default_factory=list, max_length=20)
    industry: str = Field(default="", max_length=120)
    region: str = Field(default="", max_length=120)


class CustomerAccountUpdateV1(Contract):
    name: str = Field(min_length=1, max_length=200)
    aliases: list[str] = Field(default_factory=list, max_length=20)
    industry: str = Field(default="", max_length=120)
    region: str = Field(default="", max_length=120)
    status: Literal["active", "paused", "archived"] = "active"


class CustomerAccountV1(Contract):
    id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    industry: str = ""
    region: str = ""
    status: Literal["active", "paused", "archived"] = "active"
    open_actions: int = Field(default=0, ge=0)
    pending_notes: int = Field(default=0, ge=0)
    wins: int = Field(default=0, ge=0)
    last_interaction_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class CustomerEvidenceV1(Contract):
    quote: str = Field(default="", max_length=1_000)
    source_id: str | None = None
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)


class CustomerPersonExtractV1(Contract):
    name: str = Field(min_length=1, max_length=160)
    role: str = Field(default="", max_length=160)
    organization: str = Field(default="", max_length=160)
    evidence: CustomerEvidenceV1 = Field(default_factory=CustomerEvidenceV1)


class CustomerPersonV1(CustomerPersonExtractV1):
    """A saved contact. Carries the row id an extraction has no reason to know,
    because editing or removing one addresses the record rather than the name."""

    id: str


class CustomerPersonUpsertV1(Contract):
    name: str = Field(min_length=1, max_length=160)
    role: str = Field(default="", max_length=160)
    organization: str = Field(default="", max_length=160)


CustomerFactKind = Literal[
    "requirement",
    "decision",
    "use_case",
    "risk",
    "question",
    "constraint",
    "model",
    "dac_note",
    "other",
]
CustomerFactStatus = Literal["active", "superseded", "disputed"]


class CustomerFactExtractV1(Contract):
    kind: CustomerFactKind = "other"
    content: str = Field(min_length=1, max_length=4_000)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    evidence: CustomerEvidenceV1 = Field(default_factory=CustomerEvidenceV1)


class CustomerActionExtractV1(Contract):
    description: str = Field(min_length=1, max_length=2_000)
    owner: str = Field(default="", max_length=160)
    due_at: datetime | None = None
    evidence: CustomerEvidenceV1 = Field(default_factory=CustomerEvidenceV1)


class CustomerExtractionV1(Contract):
    summary: str = Field(default="", max_length=4_000)
    occurred_at: datetime | None = None
    people: list[CustomerPersonExtractV1] = Field(default_factory=list, max_length=50)
    facts: list[CustomerFactExtractV1] = Field(default_factory=list, max_length=100)
    actions: list[CustomerActionExtractV1] = Field(default_factory=list, max_length=100)


class CustomerCaptureV1(Contract):
    account_id: str = Field(pattern=r"^cust_[a-f0-9]{20}$")
    title: str = Field(min_length=1, max_length=240)
    content: str = Field(min_length=1, max_length=100_000)
    source_kind: Literal["note", "meeting", "chat", "notion", "attachment"] = "note"
    source_ref: str = Field(default="", max_length=2_000)
    occurred_at: datetime | None = None


class CustomerSourceV1(Contract):
    id: str
    account_id: str
    source_kind: str
    title: str
    content: str
    source_ref: str = ""
    occurred_at: datetime | None = None
    status: Literal["waiting", "review", "saved", "duplicate"]
    created_at: datetime
    updated_at: datetime


class CustomerUpdateProposalV1(Contract):
    id: str
    source_id: str
    account_id: str
    status: Literal["review", "approved", "rejected"]
    extraction: CustomerExtractionV1
    model: str
    prompt_version: str
    created_at: datetime
    decided_at: datetime | None = None


class CustomerProposalSaveV1(Contract):
    extraction: CustomerExtractionV1


class CustomerActionV1(Contract):
    id: str
    account_id: str
    # Populated on the cross-account queue, where a description without the
    # customer it belongs to is not actionable. Empty on account-scoped reads.
    account_name: str = ""
    interaction_id: str | None = None
    description: str
    owner: str = ""
    due_at: datetime | None = None
    status: Literal["open", "done", "cancelled"] = "open"
    evidence: CustomerEvidenceV1 = Field(default_factory=CustomerEvidenceV1)
    created_at: datetime
    updated_at: datetime


class CustomerActionStatusV1(Contract):
    status: Literal["open", "done", "cancelled"]


class CustomerActionCreateV1(Contract):
    description: str = Field(min_length=1, max_length=2_000)
    owner: str = Field(default="", max_length=160)
    due_at: datetime | None = None


class CustomerActionEditV1(CustomerActionCreateV1):
    status: Literal["open", "done", "cancelled"] = "open"


class CustomerInteractionV1(Contract):
    id: str
    account_id: str
    source_id: str
    title: str
    occurred_at: datetime
    summary: str
    created_at: datetime


class CustomerFactV1(Contract):
    id: str
    account_id: str
    interaction_id: str | None = None
    kind: str
    content: str
    status: CustomerFactStatus = "active"
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: CustomerEvidenceV1 = Field(default_factory=CustomerEvidenceV1)
    created_at: datetime


class CustomerFactCreateV1(Contract):
    """A fact the user writes themselves.

    It carries no evidence quote because there is no note behind it, and full
    confidence because a person asserted it rather than a model inferring it.
    """

    kind: CustomerFactKind = "other"
    content: str = Field(min_length=1, max_length=4_000)


class CustomerFactEditV1(CustomerFactCreateV1):
    status: CustomerFactStatus = "active"


class CustomerSourceUpdateV1(Contract):
    title: str = Field(min_length=1, max_length=240)
    content: str = Field(min_length=1, max_length=100_000)
    source_kind: Literal["note", "meeting", "chat", "notion", "attachment"] = "note"
    occurred_at: datetime | None = None


class CustomerNoteCreateV1(Contract):
    title: str = Field(default="", max_length=240)
    body: str = Field(min_length=1, max_length=100_000)
    pinned: bool = False
    # A note written in the workbench is 'manual'; one saved out of a scoped
    # conversation is 'chat', so its provenance survives the round trip.
    origin: Literal["manual", "chat"] = "manual"
    origin_ref: str = Field(default="", max_length=2_000)


class CustomerNoteUpdateV1(Contract):
    title: str = Field(default="", max_length=240)
    body: str = Field(min_length=1, max_length=100_000)
    pinned: bool = False


class CustomerNoteV1(Contract):
    id: str
    account_id: str
    title: str = ""
    body: str
    pinned: bool = False
    origin: Literal["manual", "chat"] = "manual"
    origin_ref: str = ""
    created_at: datetime
    updated_at: datetime


class CustomerSearchHitV1(Contract):
    kind: Literal["account", "note", "fact", "action", "win", "source"]
    id: str
    account_id: str
    account_name: str
    title: str
    snippet: str = ""
    occurred_at: datetime | None = None


class CustomerSearchResultV1(Contract):
    query: str
    hits: list[CustomerSearchHitV1] = Field(default_factory=list)
    # True when the store held more matches than the limit returned, so the UI
    # can say "showing the first N" instead of implying the list is exhaustive.
    truncated: bool = False


class WinValuationLineV1(Contract):
    """One billable component of an estimated win, priced by the host."""

    sku: str
    part_number: str | None = None
    name: str = ""
    unit: str = ""
    quantity: float = Field(default=0.0, ge=0)
    utilization: float = Field(default=1.0, ge=0, le=1)
    rate: float = Field(default=0.0, ge=0)
    rate_verified: bool = False
    yearly_amount: float = Field(default=0.0, ge=0)
    basis: str = ""
    why: str = ""


class WinValuationV1(Contract):
    id: str
    win_id: str
    estimated_yearly_arr: float | None = None
    currency: str = "USD"
    lines: list[WinValuationLineV1] = Field(default_factory=list)
    explanation: str = ""
    confidence: Literal["low", "medium", "high"] = "low"
    # SKUs the model named that carry no rate — a real component nobody has
    # priced yet, rather than something to silently value at zero.
    unpriced: list[str] = Field(default_factory=list)
    rates_verified: bool = False
    model_used: str | None = None
    prompt_version: str = ""
    status: Literal["proposed", "accepted", "dismissed"] = "proposed"
    created_at: datetime
    updated_at: datetime


class WinValuationAcceptV1(Contract):
    yearly_arr: float | None = Field(default=None, ge=0)


class SkuRateV1(Contract):
    key: str
    part_number: str | None = None
    unit: str = ""
    value: float = Field(default=0.0, ge=0)
    label: str = ""
    verified: bool = False
    aliases: list[str] = Field(default_factory=list)
    note: str = ""


class SkuRateCardV1(Contract):
    currency: str = "USD"
    hours_per_year: int = Field(default=8760, gt=0)
    source_urls: list[str] = Field(default_factory=list)
    rates: list[SkuRateV1] = Field(default_factory=list)
    catalog_size: int = Field(default=0, ge=0)


class SkuRateUpdateV1(Contract):
    key: str
    value: float | None = Field(default=None, ge=0)
    verified: bool | None = None


class SkuRateCardUpdateV1(Contract):
    updates: list[SkuRateUpdateV1] = Field(default_factory=list, max_length=200)


class CustomerWinCreateV1(Contract):
    title: str = Field(min_length=1, max_length=240)
    brief: str = Field(default="", max_length=4_000)
    services: list[str] = Field(default_factory=list, max_length=10)
    dac_shape: str = Field(default="", max_length=240)
    yearly_arr: float | None = Field(default=None, ge=0)
    won_at: datetime | None = None
    source_ref: str = Field(default="", max_length=2_000)


class CustomerWinUpdateV1(CustomerWinCreateV1):
    pass


class CustomerWinV1(Contract):
    id: str
    account_id: str
    account_name: str = ""
    title: str
    brief: str = ""
    services: list[str] = Field(default_factory=list)
    dac_shape: str = ""
    yearly_arr: float | None = None
    won_at: datetime | None = None
    source_ref: str = ""
    # The estimate, when one has been run. Never the win's value — that stays
    # `yearly_arr`, and only an accepted estimate is written through to it.
    valuation: WinValuationV1 | None = None
    created_at: datetime
    updated_at: datetime


class CustomerAccountDetailV1(Contract):
    account: CustomerAccountV1
    interactions: list[CustomerInteractionV1] = Field(default_factory=list)
    facts: list[CustomerFactV1] = Field(default_factory=list)
    actions: list[CustomerActionV1] = Field(default_factory=list)
    people: list[CustomerPersonV1] = Field(default_factory=list)
    sources: list[CustomerSourceV1] = Field(default_factory=list)
    wins: list[CustomerWinV1] = Field(default_factory=list)
    notes: list[CustomerNoteV1] = Field(default_factory=list)


class CustomerDashboardV1(Contract):
    active_accounts: int = Field(default=0, ge=0)
    open_actions: int = Field(default=0, ge=0)
    overdue_actions: int = Field(default=0, ge=0)
    waiting_notes: int = Field(default=0, ge=0)
    total_wins: int = Field(default=0, ge=0)
    dac_wins: int = Field(default=0, ge=0)
    total_yearly_arr: float = Field(default=0.0, ge=0)
    wins_by_service: dict[str, int] = Field(default_factory=dict)
    recent_accounts: list[CustomerAccountV1] = Field(default_factory=list)
    priority_actions: list[CustomerActionV1] = Field(default_factory=list)
    recent_wins: list[CustomerWinV1] = Field(default_factory=list)


class CustomerOutputRequestV1(Contract):
    kind: Literal["activity_tracker", "meeting_brief", "follow_up", "internal_update"]
    interaction_id: str | None = None


class CustomerOutputV1(Contract):
    id: str
    account_id: str
    kind: str
    content: str
    tracker_url: str = ""
    created_at: datetime


class CustomerSettingsV1(Contract):
    tracker_url: str = Field(default="", max_length=2_000)
    activity_template: str = Field(default="", max_length=20_000)
    updated_at: datetime | None = None


class CustomerSettingsUpdateV1(Contract):
    tracker_url: str = Field(default="", max_length=2_000)
    activity_template: str = Field(default="", max_length=20_000)


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
    @field_validator(
        "description", "intent", "input_sketch", "output_sketch", mode="before"
    )
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
    boundaries: list[ArchitectureBoundaryV1] = Field(
        default_factory=list, max_length=16
    )
    assumptions: list[str] = Field(default_factory=list, max_length=32)
    unresolved_ambiguities: list[str] = Field(default_factory=list, max_length=32)

    @field_validator("edges")
    @classmethod
    def validate_edges(
        cls, value: list[ArchitectureEdgeV1]
    ) -> list[ArchitectureEdgeV1]:
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


# ── Dedicated AI Cluster sizing ──────────────────────────────────────────────


class DacGpuV1(Contract):
    key: str
    label: str
    memory_gb: float
    memory_bandwidth_gb_s: float
    dense_bf16_tflops: float
    dense_fp8_tflops: float | None = None
    supports_fp8: bool = False


class DacShapeV1(Contract):
    key: str
    gpu: str
    gpu_count: int
    ai_units: float
    total_memory_gb: float
    importable: bool = True


class DacModelV1(Contract):
    id: str
    family: str
    capability: str
    validated_shapes: list[str] = Field(default_factory=list)
    benchmarked_shapes: list[str] = Field(default_factory=list)
    supported: bool = True
    unsupported_reason: str | None = None
    config_source: str | None = None
    architecture: dict[str, Any] | None = None


class DacCatalogV1(Contract):
    models: list[DacModelV1]
    shapes: list[DacShapeV1]
    gpus: list[DacGpuV1]
    quantizations: list[str]
    pricing: dict[str, Any]
    provenance: dict[str, Any]


class DacVramBreakdownV1(Contract):
    weights_gb: float
    kv_cache_gb: float
    activations_gb: float
    overhead_gb: float
    total_gb: float
    capacity_gb: float
    usable_gb: float
    utilization: float
    status: Literal["okay", "moderate", "high", "very_high", "insufficient"]
    fits: bool
    max_concurrency: int


class DacPerformanceV1(Contract):
    ttft_s: float
    inference_speed_tps: float
    token_throughput_tps: float
    request_latency_s: float
    request_throughput_rps: float
    request_throughput_rpm: float
    total_throughput_tps: float
    concurrency: int
    prompt_tokens: int
    response_tokens: int


class DacConfidenceV1(Contract):
    tier: Literal["measured", "interpolated", "modeled"]
    error_margin: float | None = None
    reason: str


class DacEstimateRequestV1(Contract):
    model_id: str
    shape: str
    units: int = Field(default=1, ge=1, le=16)
    prompt_tokens: int = Field(default=2000, ge=1, le=2_000_000)
    response_tokens: int = Field(default=200, ge=1, le=131_072)
    concurrency: int = Field(default=1, ge=1, le=4096)
    quantization: str | None = None
    kv_quantization: str | None = None
    hours: float = Field(default=744.0, gt=0, le=100_000)
    price_per_ai_unit_hour: float | None = Field(default=None, ge=0)


class DacEstimateV1(Contract):
    model_id: str
    shape: str
    units: int
    oracle_validated: bool
    minimum_shape: str | None = None
    vram: DacVramBreakdownV1
    performance: DacPerformanceV1
    cost: dict[str, Any]
    confidence: DacConfidenceV1
    published: dict[str, Any] | None = None
    notes: list[str] = Field(default_factory=list)


class DacOptimizeRequestV1(Contract):
    model_id: str
    prompt_tokens: int = Field(default=2000, ge=1, le=2_000_000)
    response_tokens: int = Field(default=200, ge=1, le=131_072)
    concurrency: int = Field(default=8, ge=1, le=4096)
    max_ttft_s: float | None = Field(default=None, gt=0)
    max_request_latency_s: float | None = Field(default=None, gt=0)
    min_inference_speed_tps: float | None = Field(default=None, gt=0)
    min_request_throughput_rps: float | None = Field(default=None, gt=0)
    quantization: str | None = None
    kv_quantization: str | None = None
    hours: float = Field(default=744.0, gt=0, le=100_000)
    price_per_ai_unit_hour: float | None = Field(default=None, ge=0)
    validated_only: bool = True
    max_units: int = Field(default=8, ge=1, le=16)


class DacOptionV1(Contract):
    shape: str
    gpu: str
    gpu_count: int
    units: int
    oracle_validated: bool
    vram: DacVramBreakdownV1
    performance: DacPerformanceV1
    cost: dict[str, Any]
    meets_sla: bool
    unmet: list[str] = Field(default_factory=list)


class DacOptimizeResultV1(Contract):
    model_id: str
    options: list[DacOptionV1]
    confidence: DacConfidenceV1
    considered: int
    notes: list[str] = Field(default_factory=list)


class DacRecommendRequestV1(Contract):
    use_case: str = Field(min_length=1, max_length=4000)
    concurrency: int = Field(default=8, ge=1, le=4096)
    prompt_tokens: int = Field(default=2000, ge=1, le=2_000_000)
    response_tokens: int = Field(default=200, ge=1, le=131_072)
    max_request_latency_s: float | None = Field(default=None, gt=0)
    capability: str | None = None
    limit: int = Field(default=5, ge=1, le=20)


class DacCandidateV1(Contract):
    model_id: str
    family: str
    capability: str
    score: float
    shape: str | None = None
    units: int = 1
    performance: DacPerformanceV1 | None = None
    cost: dict[str, Any] | None = None
    meets_sla: bool = False
    rationale: str | None = None


class DacRecommendationV1(Contract):
    use_case: str
    candidates: list[DacCandidateV1]
    summary: str | None = None
    model_used: str | None = None
    model_backed: bool = False
    notes: list[str] = Field(default_factory=list)


class DocumentSectionV1(Contract):
    """One section of a document: a slide in a deck, a block on a page.

    Deliberately flat. Every field is a plain string or a list of strings, so
    the smallest model in the lineup can fill it without nesting mistakes —
    the renderer, not the model, decides what any of it looks like.
    """

    heading: str = Field(default="", max_length=140)
    # Prose, and any table, as GitHub-style markdown. Tables deliberately have
    # no typed fields of their own: asking a model to fill parallel column/row
    # arrays makes Command A+ produce tool arguments its own platform then
    # rejects, while every model writes a markdown table fluently. The
    # renderer parses them out, so the fragile half stays in tested host code.
    body: str = Field(default="", max_length=2_000)
    bullets: list[str] = Field(default_factory=list, max_length=8)
    notes: str = Field(default="", max_length=1_000)


class DocumentOutlineV1(Contract):
    """The model's entire contribution to a generated document: its content.

    No styling, no layout, no code — those belong to the renderer, which is
    first-party and tested. This split is what lets file generation work on
    every provider rather than only the strongest one.
    """

    title: str = Field(default="", max_length=160)
    subtitle: str = Field(default="", max_length=240)
    sections: list[DocumentSectionV1] = Field(default_factory=list, max_length=16)
    sources: list[str] = Field(default_factory=list, max_length=12)


class AttentionItemV1(Contract):
    """One thing waiting on the user, from whichever workbench holds it."""

    key: str
    kind: Literal[
        "run_approval",
        "customer_action",
        "customer_note",
        "tool_proposal",
        "memory",
        "answer_atom",
        "asset_trust",
        "stale_source",
    ]
    kind_label: str = ""
    title: str
    detail: str = ""
    href: str = ""
    account_id: str | None = None
    due_at: datetime | None = None
    created_at: datetime | None = None
    overdue: bool = False
    # Consequence, not recency: what it costs to leave this until tomorrow.
    priority: float = 0.0
    deferred_until: datetime | None = None


class AttentionFeedV1(Contract):
    generated_at: datetime
    items: list[AttentionItemV1] = Field(default_factory=list)
    # The headline. If it says three things need you, these are the three.
    top: list[AttentionItemV1] = Field(default_factory=list)
    # Snoozed items travel separately so `items` always means "live work",
    # and a deferred thing can still be brought back without a second call.
    deferred_items: list[AttentionItemV1] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    total: int = 0
    deferred: int = 0


class AttentionDeferV1(Contract):
    key: str = Field(min_length=1, max_length=200)
    kind: str = Field(min_length=1, max_length=40)
    days: int = Field(default=7, ge=1, le=365)
    reason: str = Field(default="", max_length=400)


class ActionResolutionV1(Contract):
    """One open action the user's message says is finished.

    `action_id` is chosen from a supplied list, never composed: the host
    rejects any id it did not offer, so a model cannot close a commitment it
    has invented or one belonging to another account."""

    action_id: str = Field(min_length=1, max_length=80)
    note: str = Field(default="", max_length=600)


class NewActionV1(Contract):
    """A follow-up the message creates."""

    description: str = Field(min_length=1, max_length=2_000)
    account_id: str = Field(default="", max_length=80)
    owner: str = Field(default="", max_length=160)
    due_at: datetime | None = None


class CapturedNoteV1(Contract):
    """A note a message asks to file against an account.

    Deliberately carries no body. The note's content is the user's own words,
    set verbatim by the host and never regenerated by the model — an exact
    identifier (an SR number, a response id, a branch code) that a model
    "tidied" into a near-miss is evidence turned wrong. The model only says
    which account it belongs to and proposes a title.
    """

    account_id: str = Field(min_length=1, max_length=80)
    title: str = Field(min_length=1, max_length=240)


class TidiedNoteV1(Contract):
    """A reformatted note body, produced only when the user asks for cleanup.

    Reformatting only: the host verifies every identifier in `body` survives
    verbatim in the original before it is stored, and falls back to the raw
    text otherwise, so cleanup can never alter a fact."""

    body: str = Field(default="", max_length=20_000)


class QueueUpdateV1(Contract):
    """What a message would change about the customer record: file a note,
    close commitments it settles, open the ones it creates.

    Proposal-only. Nothing here touches a record until the approval it is
    attached to is granted."""

    summary: str = Field(default="", max_length=1_000)
    note: CapturedNoteV1 | None = None
    completed: list[ActionResolutionV1] = Field(default_factory=list, max_length=25)
    new_actions: list[NewActionV1] = Field(default_factory=list, max_length=25)
    unmatched: list[str] = Field(default_factory=list, max_length=10)


class CustomerToolCallV1(Contract):
    """One call the customer agent chose from its catalog.

    The fields are flat and typed on purpose. An earlier version carried a
    free-form ``arguments: dict`` envelope; Cohere's tool validation cannot
    schema-check an open dict and rejected every routed turn with
    HALLUCINATED_ALL_TOOL_CALLS, so the router silently fell back to answering.
    Naming the two metadata fields the tools actually take keeps the schema
    fully defined for every provider. They are metadata the model may author —
    a note title, which proposal to apply — never the user's evidence, which the
    host supplies verbatim (see ``CapturedNoteV1``). A field is simply left blank
    when the chosen tool has no use for it.
    """

    name: Literal[
        "answer_about_account",
        "file_note",
        "apply_extraction",
        "record_activity",
        "record_win",
        "generate_tracker",
    ]
    # file_note / record_win: a short label for the note or win.
    title: str = Field(default="", max_length=240)
    # apply_extraction: which pending proposal to commit.
    proposal_id: str = Field(default="", max_length=80)


class CustomerAgentStepV1(Contract):
    """The customer agent's plan for one message: the ordered tools to run.

    One planning call, not a multi-round loop — because every write the agent
    can make is user-gated (an apply card, a filed-note confirmation), the model
    never needs one write's result to choose the next. So it names every call a
    message warrants at once ("file this note and log a win" is two calls, in
    order) and the host runs them. An empty list means "just answer" — the safe
    default when nothing actionable was asked.
    """

    calls: list[CustomerToolCallV1] = Field(default_factory=list, max_length=8)


class AttentionBatchV1(Contract):
    """Decide several queued items in one call."""

    keys: list[str] = Field(min_length=1, max_length=100)
    decision: Literal["approve", "reject", "defer"]
    days: int = Field(default=7, ge=1, le=365)
    reason: str = Field(default="", max_length=400)


class AttentionBatchResultV1(Contract):
    # Honest about partial success: a key whose kind has no one-click decision
    # is reported, never silently counted as done.
    applied: list[str] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)
    feed: AttentionFeedV1


class MorningBriefV1(Contract):
    """The day's opening summary.

    Every number and name here is counted from the records by the host. The
    model is given those facts and writes only the connecting sentences, so a
    brief can read well without ever being able to overstate what happened.
    """

    generated_at: datetime
    since: datetime
    # Host-computed, never model-written.
    changed: list[str] = Field(default_factory=list)
    focus: list[AttentionItemV1] = Field(default_factory=list)
    waiting_total: int = 0
    # Model-written prose over the facts above; empty when no model is reachable,
    # and the brief is still complete without it.
    narrative: str = ""
    recommendation: str = ""


class AnswerAtomV1(Contract):
    """One thing you have already worked out, kept as reusable knowledge."""

    question: str = Field(min_length=1, max_length=400)
    paraphrases: list[str] = Field(default_factory=list, max_length=6)
    answer: str = Field(min_length=1, max_length=4_000)
    citations: list[str] = Field(default_factory=list, max_length=12)
    entities: list[str] = Field(default_factory=list, max_length=12)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ToolInputRestatementV1(Contract):
    """A request relabelled into the parameter names one tool declares.

    Labels only. Every value in it is checked against the user's own message
    before the tool sees it, so a restatement that changed a figure is thrown
    away rather than run."""

    lines: list[str] = Field(default_factory=list, max_length=12)


class AnswerAtomHarvestV1(Contract):
    """What a finished run offered the bank.

    Bounded at two on purpose: a bank that proposes five atoms a run becomes
    two hundred pending reviews in a week, and then it is never opened."""

    atoms: list[AnswerAtomV1] = Field(default_factory=list, max_length=2)
