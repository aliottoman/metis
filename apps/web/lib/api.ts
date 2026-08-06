import type {
  AssetEnvVar,
  AssetLogsV1,
  AssetV1,
  AttachmentRef,
  ChatMessage,
  CodeGraphLookup,
  DacCatalog,
  DacEstimate,
  DacEstimateRequest,
  DacOptimizeRequest,
  DacOptimizeResult,
  DacRecommendRequest,
  DacRecommendation,
  CodeGraphStats,
  EntityGraphLookup,
  EntityGraphStats,
  ConversationDetail,
  ConversationSummary,
  CustomerAccount,
  CustomerAccountDetail,
  CustomerAction,
  CustomerDashboard,
  CustomerExtraction,
  CustomerFact,
  CustomerNote,
  CustomerOutput,
  CustomerPerson,
  CustomerProposal,
  CustomerSearchResult,
  CustomerSettings,
  CustomerSource,
  CustomerWin,
  SkuRateCard,
  WinValuation,
  CorpusHealth,
  CorpusReindexResult,
  CorpusSource,
  HealthSnapshot,
  KnowledgeSnippet,
  KnowledgeScope,
  MemoryIndexStatus,
  MemoryProposal,
  ModelHealth,
  ModelPreference,
  LocalModelSession,
  NotionConnection,
  NotionSyncResult,
  ConversationProject,
  PersonalProfile,
  ProjectMode,
  ProjectVerification,
  ProjectWorkspace,
  RecoverableRun,
  RiskLevel,
  RunHandle,
  ToolDefinitionBuild,
  ToolDefinitionProposal,
  ToolDefinitionRecord,
  EvalReport,
  ModelAccess,
  CapabilityProfile,
  ToolRouteFacts,
  ToolDefinition,
  ToolImprovementProposal,
  ToolImprovementDecisionResult,
  ToolImprovementEvidence,
  ToolRecord,
  ToolVersion,
  ToolVersionEvidence,
} from "@/lib/types";
import { latestRunId } from "./run-history.ts";

export const API_BASE = (process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
const API_PREFIX = "/api/v1";

export class ApiError extends Error {
  public readonly status: number;
  public readonly detail?: unknown;

  constructor(message: string, status: number, detail?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : {};
}

function unwrap(value: unknown): unknown {
  const record = asRecord(value);
  return record.data ?? record.result ?? value;
}

function listFrom(value: unknown, ...keys: string[]): unknown[] {
  const unwrapped = unwrap(value);
  if (Array.isArray(unwrapped)) return unwrapped;
  const record = asRecord(unwrapped);
  for (const key of [...keys, "items", "results"]) {
    if (Array.isArray(record[key])) return record[key] as unknown[];
  }
  return [];
}

function stringValue(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : value == null ? fallback : String(value);
}

function numberValue(value: unknown): number | undefined {
  const result = Number(value);
  return Number.isFinite(result) ? result : undefined;
}

function apiUrl(path: string): string {
  if (/^https?:\/\//.test(path)) return path;
  return `${API_BASE}${path.startsWith("/") ? path : `/${path}`}`;
}

async function readError(response: Response): Promise<unknown> {
  const text = await response.text();
  if (!text) return undefined;
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return text;
  }
}

/**
 * A readable sentence out of FastAPI's 422 body.
 *
 * FastAPI reports validation failures as an array of objects, which the generic
 * string coercion renders as "[object Object]" — the user sees that a request
 * failed but not which field or why. Returns undefined for every other error
 * shape so the existing handling stays in charge.
 */
function validationMessage(detail: unknown): string | undefined {
  if (!Array.isArray(detail) || !detail.length) return undefined;
  const parts = detail
    .map((entry) => {
      const item = asRecord(entry);
      const message = typeof item.msg === "string" ? item.msg : undefined;
      if (!message) return undefined;
      const location = Array.isArray(item.loc)
        ? item.loc.filter((part) => part !== "body").join(".")
        : "";
      return location ? `${location}: ${message}` : message;
    })
    .filter((part): part is string => Boolean(part));
  return parts.length ? parts.join("; ") : undefined;
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  if (init?.body && !(init.body instanceof FormData) && !headers.has("content-type")) {
    headers.set("content-type", "application/json");
  }
  headers.set("accept", "application/json");

  const response = await fetch(apiUrl(path), {
    ...init,
    headers,
    cache: "no-store",
  });
  if (!response.ok) {
    const detail = await readError(response);
    const record = asRecord(detail);
    throw new ApiError(
      validationMessage(record.detail) ??
        stringValue(record.detail ?? record.message, `Request failed (${response.status})`),
      response.status,
      detail,
    );
  }
  if (response.status === 204) return undefined as T;
  const text = await response.text();
  return (text ? JSON.parse(text) : undefined) as T;
}

function normalizeConversation(value: unknown): ConversationSummary {
  const item = asRecord(value);
  return {
    id: stringValue(item.id ?? item.conversation_id ?? item.thread_id),
    title: stringValue(item.title, "New conversation"),
    created_at: item.created_at ? stringValue(item.created_at) : undefined,
    updated_at: item.updated_at ? stringValue(item.updated_at) : undefined,
    last_message: item.last_message ? stringValue(item.last_message) : undefined,
  };
}

function normalizeAttachment(value: unknown): AttachmentRef {
  const item = asRecord(value);
  return {
    id: stringValue(item.id ?? item.upload_id ?? item.artifact_id),
    name: stringValue(item.name ?? item.filename, "Attachment"),
    media_type: item.media_type ? stringValue(item.media_type) : item.content_type ? stringValue(item.content_type) : undefined,
    size: numberValue(item.size ?? item.size_bytes),
    sha256: item.sha256 ? stringValue(item.sha256) : undefined,
  };
}

function normalizeMessage(value: unknown): ChatMessage {
  const item = asRecord(value);
  return {
    id: stringValue(item.id ?? item.message_id, crypto.randomUUID()),
    role: item.role === "user" || item.role === "system" ? item.role : "assistant",
    content: stringValue(item.content ?? item.text),
    run_id: item.run_id ? stringValue(item.run_id) : item.runId ? stringValue(item.runId) : undefined,
    created_at: item.created_at ? stringValue(item.created_at) : undefined,
    attachments: listFrom(item.attachments).map(normalizeAttachment),
  };
}

export async function listConversations(): Promise<ConversationSummary[]> {
  const response = await request<unknown>(`${API_PREFIX}/conversations`);
  return listFrom(response, "conversations").map(normalizeConversation).filter((item) => item.id);
}

export async function createConversation(title?: string): Promise<ConversationSummary> {
  const response = await request<unknown>(`${API_PREFIX}/conversations`, {
    method: "POST",
    body: JSON.stringify(title ? { title } : {}),
  });
  return normalizeConversation(unwrap(response));
}

export async function deleteConversation(id: string): Promise<void> {
  await request<void>(`${API_PREFIX}/conversations/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
}

export async function getConversation(id: string): Promise<ConversationDetail> {
  const [conversationResponse, messagesResponse] = await Promise.all([
    request<unknown>(`${API_PREFIX}/conversations/${encodeURIComponent(id)}`),
    request<unknown>(`${API_PREFIX}/conversations/${encodeURIComponent(id)}/messages`),
  ]);
  const conversation = normalizeConversation(unwrap(conversationResponse));
  const messages = listFrom(messagesResponse, "messages").map(normalizeMessage);
  return {
    ...conversation,
    id: conversation.id || id,
    messages,
    latest_run_id: latestRunId(messages),
  };
}

export async function uploadFile(file: File): Promise<AttachmentRef> {
  const body = new FormData();
  body.append("file", file, file.name);
  const response = await request<unknown>(`${API_PREFIX}/uploads`, { method: "POST", body });
  return normalizeAttachment(unwrap(response));
}

/**
 * Retire a message and everything after it, then re-read the thread.
 *
 * What makes an edited message an edit rather than a follow-up question: the
 * turns that came after the original leave the model's view. The API marks
 * them rather than deleting them, so the run history stays auditable.
 */
export async function rewindConversation(
  conversationId: string,
  messageId: string,
): Promise<ConversationDetail> {
  await request<unknown>(
    `${API_PREFIX}/conversations/${encodeURIComponent(conversationId)}/messages/${encodeURIComponent(messageId)}/rewind`,
    { method: "POST" },
  );
  return getConversation(conversationId);
}

/**
 * One dictated clip turned into composer text, via Cohere Transcribe.
 *
 * Nothing is stored on either side: this is not an upload, and what comes
 * back is a draft the user still edits before sending.
 */
export async function transcribeAudio(audio: Blob, filename = "dictation.webm"): Promise<string> {
  const body = new FormData();
  body.append("file", audio, filename);
  const response = await request<unknown>(`${API_PREFIX}/transcribe`, { method: "POST", body });
  const payload = unwrap(response) as { text?: unknown };
  return typeof payload.text === "string" ? payload.text : "";
}

export async function sendMessage(
  conversationId: string,
  content: string,
  attachments: AttachmentRef[],
  project?: { id: string; mode: ProjectMode } | null,
  knowledgeScope: KnowledgeScope = "auto",
  customerId?: string | null,
): Promise<RunHandle> {
  const response = await request<unknown>(
    `${API_PREFIX}/conversations/${encodeURIComponent(conversationId)}/messages`,
    {
      method: "POST",
      body: JSON.stringify({
        content,
        attachment_ids: attachments.map((item) => item.id),
        knowledge_scope: knowledgeScope,
        ...(customerId ? { customer_id: customerId } : {}),
        ...(project === undefined
          ? {}
          : project
            ? { project_id: project.id, project_mode: project.mode }
            : { project_id: null, project_mode: null }),
      }),
    },
  );
  const value = asRecord(unwrap(response));
  return {
    run_id: stringValue(value.run_id ?? value.id),
    // The id the message was actually stored under. The composer shows an
    // optimistic message immediately, under a client-side id that exists
    // nowhere on the server — anything that later addresses that message by
    // id (rewinding a thread to it) needs this to swap in.
    message_id: value.message_id ? stringValue(value.message_id) : undefined,
    conversation_id: value.conversation_id ? stringValue(value.conversation_id) : conversationId,
    status: value.status ? stringValue(value.status) : undefined,
  };
}

function normalizeProjectWorkspace(value: unknown): ProjectWorkspace {
  const item = asRecord(value);
  return {
    id: stringValue(item.id ?? item.project_id),
    name: stringValue(item.name, "Untitled project"),
    summary: stringValue(item.summary, "No project summary available."),
    framework: item.framework == null ? null : stringValue(item.framework),
    initialized: item.initialized === true,
    manifestRevision: numberValue(item.manifestRevision ?? item.manifest_revision) ?? 0,
    fileCount: numberValue(item.fileCount ?? item.file_count) ?? 0,
    metisMdPath: stringValue(item.metisMdPath ?? item.metis_md_path, ".metis/METIS.md"),
    updatedAt: item.updatedAt || item.updated_at ? stringValue(item.updatedAt ?? item.updated_at) : null,
  };
}

export async function listProjectWorkspaces(): Promise<ProjectWorkspace[]> {
  return listFrom(await request<unknown>(`${API_PREFIX}/projects`), "projects")
    .map(normalizeProjectWorkspace)
    .filter((project) => project.id);
}

export async function openProjectWorkspace(
  projectId: string,
  mode: ProjectMode,
): Promise<ProjectWorkspace> {
  return normalizeProjectWorkspace(
    unwrap(await request<unknown>(`${API_PREFIX}/projects/${encodeURIComponent(projectId)}/open`, {
      method: "POST",
      body: JSON.stringify({ mode }),
    })),
  );
}

function normalizeMemoryIndexStatus(value: unknown): MemoryIndexStatus {
  const item = asRecord(unwrap(value));
  return {
    consent: item.consent === true,
    consentReason: item.consentReason || item.consent_reason
      ? stringValue(item.consentReason ?? item.consent_reason)
      : null,
    cloudAvailable: (item.cloudAvailable ?? item.cloud_available) === true,
    semantic: item.semantic === true,
    active: numberValue(item.active) ?? 0,
    embedded: numberValue(item.embedded) ?? 0,
  };
}

export async function getMemoryIndexStatus(): Promise<MemoryIndexStatus> {
  return normalizeMemoryIndexStatus(await request<unknown>(`${API_PREFIX}/memory/index`));
}

export async function setMemoryIndexConsent(consent: boolean): Promise<MemoryIndexStatus> {
  return normalizeMemoryIndexStatus(
    await request<unknown>(`${API_PREFIX}/memory/index/consent`, {
      method: "POST",
      body: JSON.stringify({ consent }),
    }),
  );
}

function normalizeProjectVerification(
  value: unknown,
  projectId: string,
): ProjectVerification {
  const item = asRecord(unwrap(value));
  return {
    projectId: stringValue(item.projectId ?? item.project_id, projectId),
    configured: item.configured === true,
    approved: item.approved === true,
    fingerprint: item.fingerprint == null ? null : stringValue(item.fingerprint),
    checks: listFrom(item.checks, "checks").map((raw) => {
      const check = asRecord(raw);
      return {
        name: stringValue(check.name),
        command: Array.isArray(check.command) ? check.command.map((part) => stringValue(part)) : [],
        description: stringValue(check.description, ""),
        explanation: stringValue(check.explanation, ""),
        timeoutSeconds: numberValue(check.timeoutSeconds ?? check.timeout_seconds) ?? 300,
      };
    }),
    explanation: stringValue(item.explanation, ""),
    boundary: stringValue(item.boundary, ""),
    error: item.error == null ? null : stringValue(item.error),
  };
}

export async function getProjectVerification(projectId: string): Promise<ProjectVerification> {
  return normalizeProjectVerification(
    await request<unknown>(
      `${API_PREFIX}/projects/${encodeURIComponent(projectId)}/verification`,
    ),
    projectId,
  );
}

export async function decideProjectVerification(
  projectId: string,
  decision: "approve" | "revoke",
): Promise<ProjectVerification> {
  return normalizeProjectVerification(
    await request<unknown>(
      `${API_PREFIX}/projects/${encodeURIComponent(projectId)}/verification/${decision}`,
      { method: "POST" },
    ),
    projectId,
  );
}

export async function getConversationProject(
  conversationId: string,
): Promise<ConversationProject | null> {
  const value = await request<unknown>(
    `${API_PREFIX}/conversations/${encodeURIComponent(conversationId)}/project`,
  );
  if (value == null) return null;
  const item = asRecord(unwrap(value));
  const projectId = stringValue(item.projectId ?? item.project_id);
  const mode = item.mode === "grok_continuous" ? "grok_continuous" : "grok_bootstrap_local";
  return projectId ? {
    conversationId: stringValue(item.conversationId ?? item.conversation_id, conversationId),
    projectId,
    mode,
    updatedAt: item.updatedAt || item.updated_at ? stringValue(item.updatedAt ?? item.updated_at) : undefined,
  } : null;
}

export async function decideRun(
  runId: string,
  approvalId: string,
  decision: "approve" | "reject",
  reason?: string,
): Promise<void> {
  await request(`${API_PREFIX}/runs/${encodeURIComponent(runId)}/decisions`, {
    method: "POST",
    body: JSON.stringify({ approval_id: approvalId, decision, reason }),
  });
}

export async function cancelRun(runId: string): Promise<void> {
  await request(`${API_PREFIX}/runs/${encodeURIComponent(runId)}/cancel`, { method: "POST" });
}

/** A finished run's own record, including whatever it produced.
 *
 * Artifacts otherwise reach the UI only as live events, so a generated file
 * disappeared the moment the conversation was reopened — which is exactly
 * when someone comes back for the deck they asked for. */
export async function getRunResult(runId: string): Promise<Record<string, unknown>> {
  const run = await request<{ result?: Record<string, unknown> | null }>(
    `${API_PREFIX}/runs/${encodeURIComponent(runId)}`,
  );
  return run.result ?? {};
}

function normalizeApproval(value: unknown): RecoverableRun["approval"] {
  if (!value || typeof value !== "object") return null;
  const item = asRecord(value);
  const id = stringValue(item.id ?? item.approval_id);
  if (!id) return null;
  return {
    id,
    run_id: item.run_id ? stringValue(item.run_id) : undefined,
    title: stringValue(item.title ?? item.action, "Approval required"),
    summary: stringValue(item.summary ?? item.description, "Metis is waiting for your decision."),
    risk_level: item.risk_level ? stringValue(item.risk_level) as RiskLevel : undefined,
    permissions: listFrom(item.permissions).map((permission) => stringValue(permission)),
    action_digest: item.action_digest ? stringValue(item.action_digest) : item.input_digest ? stringValue(item.input_digest) : undefined,
    status: "pending",
  };
}

export async function listRecoverableRuns(): Promise<RecoverableRun[]> {
  const response = await request<unknown>(`${API_PREFIX}/runs?status=awaiting_approval`);
  return listFrom(response, "runs").flatMap((value) => {
    const envelope = asRecord(value);
    const run = asRecord(envelope.run);
    const id = stringValue(run.id ?? run.run_id);
    const conversationId = stringValue(run.conversation_id ?? run.thread_id);
    if (!id || !conversationId) return [];
    return [{
      run: {
        id,
        conversation_id: conversationId,
        user_message_id: run.user_message_id ? stringValue(run.user_message_id) : undefined,
        status: stringValue(run.status, "awaiting_approval"),
        graph_schema_version: run.graph_schema_version ? stringValue(run.graph_schema_version) : undefined,
        cancel_requested: typeof run.cancel_requested === "boolean" ? run.cancel_requested : undefined,
        result: run.result && typeof run.result === "object" ? run.result as Record<string, unknown> : null,
        last_error: run.last_error == null ? null : stringValue(run.last_error),
        created_at: run.created_at ? stringValue(run.created_at) : undefined,
        updated_at: run.updated_at ? stringValue(run.updated_at) : undefined,
      },
      approval: normalizeApproval(envelope.approval),
    }];
  });
}

export async function submitFeedback(
  runId: string,
  rating: "positive" | "negative",
  correction?: string,
): Promise<{ id: string; memory_proposal_id?: string | null }> {
  return request(`${API_PREFIX}/runs/${encodeURIComponent(runId)}/feedback`, {
    method: "POST",
    body: JSON.stringify({ run_id: runId, rating, correction: correction || null }),
  });
}

export function runEventsUrl(runId: string, after: number): string {
  return apiUrl(`${API_PREFIX}/runs/${encodeURIComponent(runId)}/events?after=${Math.max(0, after)}`);
}

// ── Tool Factory v2: declarative tool definitions ────────────────────────────

function stringRecord(value: unknown): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [key, entry] of Object.entries(asRecord(value))) {
    out[key] = stringValue(entry);
  }
  return out;
}

function boolRecord(value: unknown): Record<string, boolean> {
  const out: Record<string, boolean> = {};
  for (const [key, entry] of Object.entries(asRecord(value))) {
    out[key] = entry === true;
  }
  return out;
}

function normalizeModelAccess(value: unknown): ModelAccess {
  const item = asRecord(value);
  const valid = new Set(["planner", "coder", "reviewer"]);
  const roles = listFrom(item.roles)
    .map((role) => stringValue(role))
    .filter((role) => valid.has(role)) as ModelAccess["roles"];
  return {
    enabled: item.enabled === true,
    roles,
    max_calls_per_run: numberValue(item.max_calls_per_run) ?? 0,
    max_tokens_per_call: numberValue(item.max_tokens_per_call) ?? 0,
    prompt_templates: stringRecord(item.prompt_templates),
  };
}

function normalizeCapabilityProfile(value: unknown): CapabilityProfile {
  const item = asRecord(value);
  return {
    code_allowlist: stringValue(item.code_allowlist),
    runtime_allowlists: stringRecord(item.runtime_allowlists),
    model_access: normalizeModelAccess(item.model_access),
    filesystem: stringValue(item.filesystem, "run-io") as CapabilityProfile["filesystem"],
    network: stringValue(item.network, "none") as CapabilityProfile["network"],
    max_runtime_seconds: numberValue(item.max_runtime_seconds) ?? 150,
    max_artifact_bytes: numberValue(item.max_artifact_bytes) ?? 10_000_000,
  };
}

function normalizeRouteFacts(value: unknown): ToolRouteFacts {
  const item = asRecord(value);
  return {
    existing_risk: stringValue(item.existing_risk, "R2") as RiskLevel,
    factory_risk: stringValue(item.factory_risk, "R3") as RiskLevel,
    input_pipeline: stringValue(item.input_pipeline, "none") as ToolRouteFacts["input_pipeline"],
  };
}

function normalizeToolDefinition(value: unknown): ToolDefinition {
  const item = asRecord(value);
  return {
    schema_version: item.schema_version ? stringValue(item.schema_version) : undefined,
    slug: stringValue(item.slug),
    version: stringValue(item.version),
    name: stringValue(item.name, "Untitled definition"),
    description: stringValue(item.description),
    archetype: stringValue(item.archetype),
    intent_examples: listFrom(item.intent_examples).map((example) => stringValue(example)),
    input_contract: asRecord(item.input_contract),
    output_contract: asRecord(item.output_contract),
    route_facts: normalizeRouteFacts(item.route_facts),
    capability_profile: normalizeCapabilityProfile(item.capability_profile),
    status: stringValue(item.status, "defined") as ToolDefinition["status"],
    content_hash: stringValue(item.content_hash),
    created_at: item.created_at ? stringValue(item.created_at) : undefined,
  };
}

function normalizeToolDefinitionRecord(value: unknown): ToolDefinitionRecord {
  const item = asRecord(value);
  return {
    definition: normalizeToolDefinition(item.definition ?? item),
    active: item.active === true,
    runnable: item.runnable === true,
    buildable: item.buildable === true,
    disabled: item.disabled === true,
    pending_definition_proposal: item.pending_definition_proposal === true,
    pending_build: item.pending_build === true,
  };
}

function normalizeToolDefinitionProposal(value: unknown): ToolDefinitionProposal {
  const item = asRecord(value);
  return {
    id: stringValue(item.id ?? item.proposal_id),
    definition_id: stringValue(item.definition_id),
    slug: stringValue(item.slug),
    version: stringValue(item.version),
    status: stringValue(item.status, "pending") as ToolDefinitionProposal["status"],
    risk_level: item.risk_level ? (stringValue(item.risk_level) as RiskLevel) : undefined,
    summary: stringValue(item.summary),
    source_run_id: item.source_run_id == null ? null : stringValue(item.source_run_id),
    decision_reason: item.decision_reason == null ? null : stringValue(item.decision_reason),
    created_at: item.created_at ? stringValue(item.created_at) : undefined,
    decided_at: item.decided_at == null ? null : stringValue(item.decided_at),
  };
}

function normalizeEvalReport(value: unknown): EvalReport | null {
  if (!value || typeof value !== "object") return null;
  const item = asRecord(value);
  return {
    passed: item.passed === true,
    score: numberValue(item.score) ?? 0,
    results: listFrom(item.results).map((result) => {
      const entry = asRecord(result);
      return {
        case_id: stringValue(entry.case_id ?? entry.id),
        passed: entry.passed === true,
        checks: boolRecord(entry.checks),
        message: stringValue(entry.message),
      };
    }),
    static_checks: boolRecord(item.static_checks),
    created_at: item.created_at ? stringValue(item.created_at) : undefined,
  };
}

function normalizeToolDefinitionBuild(value: unknown): ToolDefinitionBuild {
  const item = asRecord(value);
  return {
    id: stringValue(item.id ?? item.build_id),
    definition_id: stringValue(item.definition_id),
    slug: stringValue(item.slug),
    version: stringValue(item.version),
    content_hash: stringValue(item.content_hash),
    status: stringValue(item.status, "evaluated") as ToolDefinitionBuild["status"],
    eval_report: normalizeEvalReport(item.eval_report),
    source_run_id: item.source_run_id == null ? null : stringValue(item.source_run_id),
    created_at: item.created_at ? stringValue(item.created_at) : undefined,
    decided_at: item.decided_at == null ? null : stringValue(item.decided_at),
  };
}

export async function listToolDefinitions(): Promise<ToolDefinitionRecord[]> {
  const response = await request<unknown>(`${API_PREFIX}/tool-definitions`);
  return listFrom(response, "definitions", "records")
    .map(normalizeToolDefinitionRecord)
    .filter((item) => item.definition.slug || item.definition.name);
}

export async function listToolDefinitionProposals(status?: string): Promise<ToolDefinitionProposal[]> {
  const query = status ? `?status=${encodeURIComponent(status)}` : "";
  const response = await request<unknown>(`${API_PREFIX}/tool-definition-proposals${query}`);
  return listFrom(response, "proposals").map(normalizeToolDefinitionProposal).filter((item) => item.id);
}

export async function listToolDefinitionBuilds(status?: string): Promise<ToolDefinitionBuild[]> {
  const query = status ? `?status=${encodeURIComponent(status)}` : "";
  const response = await request<unknown>(`${API_PREFIX}/tool-definition-builds${query}`);
  return listFrom(response, "builds").map(normalizeToolDefinitionBuild).filter((item) => item.id);
}

function normalizeTool(value: unknown): ToolRecord {
  const item = asRecord(value);
  const evaluation = asRecord(item.evaluation ?? item.eval_report);
  return {
    id: stringValue(item.id ?? item.tool_id),
    proposal_id: item.proposal_id ? stringValue(item.proposal_id) : undefined,
    name: stringValue(item.name, "Unnamed capability"),
    description: stringValue(item.description ?? item.summary),
    state: (stringValue(item.state ?? item.status, item.active_version_id ? "active" : "draft") as ToolRecord["state"]),
    risk_level: item.risk_level ? (stringValue(item.risk_level) as ToolRecord["risk_level"]) : undefined,
    permissions: listFrom(item.permissions).map((permission) => stringValue(permission)),
    active_version: item.active_version ? stringValue(item.active_version) : item.active_version_id ? stringValue(item.active_version_id) : undefined,
    latest_version: item.latest_version ? stringValue(item.latest_version) : item.version ? stringValue(item.version) : undefined,
    content_hash: item.content_hash ? stringValue(item.content_hash) : undefined,
    created_at: item.created_at ? stringValue(item.created_at) : undefined,
    updated_at: item.updated_at ? stringValue(item.updated_at) : undefined,
    evaluation: Object.keys(evaluation).length
      ? {
          passed: numberValue(evaluation.passed) ?? 0,
          failed: numberValue(evaluation.failed) ?? 0,
          total: numberValue(evaluation.total) ?? 0,
          score: numberValue(evaluation.score),
        }
      : undefined,
  };
}

export async function listTools(): Promise<ToolRecord[]> {
  const [toolsResponse, proposalsResponse] = await Promise.all([
    request<unknown>(`${API_PREFIX}/tools`),
    request<unknown>(`${API_PREFIX}/tool-proposals?status=`),
  ]);
  const tools = listFrom(toolsResponse, "tools").map(normalizeTool).filter((item) => item.id);
  const byId = new Map(tools.map((tool) => [tool.id, tool]));
  const proposalSeen = new Set<string>();
  for (const rawProposal of listFrom(proposalsResponse, "proposals")) {
    const proposal = asRecord(rawProposal);
    const toolId = stringValue(proposal.tool_id);
    if (proposalSeen.has(toolId)) continue;
    proposalSeen.add(toolId);
    const existing = byId.get(toolId);
    const status = stringValue(proposal.status, "pending");
    const mappedState: ToolRecord["state"] = status === "pending" ? "evaluated" : status === "rejected" ? "rejected" : status === "approved" ? "approved" : "draft";
    byId.set(toolId || stringValue(proposal.id), {
      ...(existing ?? {
        id: toolId || stringValue(proposal.id),
        name: "Capability proposal",
        description: stringValue(proposal.summary),
        state: mappedState,
      }),
      proposal_id: stringValue(proposal.id),
      description: existing?.description || stringValue(proposal.summary),
      state: status === "pending" ? "evaluated" : existing?.active_version ? "active" : mappedState,
      risk_level: proposal.risk_level ? stringValue(proposal.risk_level) as ToolRecord["risk_level"] : existing?.risk_level,
      created_at: proposal.created_at ? stringValue(proposal.created_at) : existing?.created_at,
    });
  }
  return [...byId.values()];
}

export async function listToolVersions(toolId: string): Promise<ToolVersion[]> {
  const response = await request<unknown>(`${API_PREFIX}/tools/${encodeURIComponent(toolId)}/versions`);
  return listFrom(response, "versions").map((value) => {
    const item = asRecord(value);
    const report = asRecord(item.eval_report ?? item.evaluation);
    const results = listFrom(report.results).map(asRecord);
    const manifest = asRecord(item.manifest);
    return {
      id: stringValue(item.id ?? item.version_id),
      version: stringValue(item.version),
      state: stringValue(item.state ?? item.status, "draft") as ToolVersion["state"],
      content_hash: item.content_hash ? stringValue(item.content_hash) : undefined,
      created_at: item.created_at ? stringValue(item.created_at) : undefined,
      risk_level: manifest.risk_level ? stringValue(manifest.risk_level) as ToolVersion["risk_level"] : undefined,
      permissions: listFrom(manifest.permissions).map((permission) => stringValue(permission)),
      evaluation: Object.keys(report).length ? {
        passed: results.filter((result) => result.passed === true).length,
        failed: results.filter((result) => result.passed !== true).length,
        total: results.length,
        score: numberValue(report.score),
      } : undefined,
    };
  });
}

export async function activateToolVersion(
  toolId: string,
  versionId: string,
  idempotencyKey: string,
  reason: string,
): Promise<{ tool_id: string; active_version_id: string; prior_version_id?: string | null }> {
  return request(`${API_PREFIX}/tools/${encodeURIComponent(toolId)}/versions/${encodeURIComponent(versionId)}/activate`, {
    method: "POST",
    body: JSON.stringify({ idempotency_key: idempotencyKey, reason }),
  });
}

export async function decideToolProposal(proposalId: string, decision: "approve" | "reject"): Promise<void> {
  await request(`${API_PREFIX}/tool-proposals/${encodeURIComponent(proposalId)}/${decision}`, {
    method: "POST",
    body: JSON.stringify({}),
  });
}

export async function listToolImprovementProposals(): Promise<ToolImprovementProposal[]> {
  const response = await request<unknown>(`${API_PREFIX}/tool-improvement-proposals?status=`);
  return listFrom(response, "proposals").flatMap((value) => {
    const item = asRecord(value);
    const regression = asRecord(item.regression_eval);
    const id = stringValue(item.id ?? item.proposal_id);
    if (!id) return [];
    return [{
      id,
      source_run_id: stringValue(item.source_run_id),
      tool_id: stringValue(item.tool_id),
      tool_version_id: stringValue(item.tool_version_id),
      content_hash: stringValue(item.content_hash),
      correction: stringValue(item.correction),
      regression_eval: {
        id: stringValue(regression.id),
        name: stringValue(regression.name, "Correction regression"),
        input: asRecord(regression.input),
        expected_properties: listFrom(regression.expected_properties).map((property) => stringValue(property)),
      },
      status: stringValue(item.status, "pending"),
      created_at: item.created_at ? stringValue(item.created_at) : undefined,
      decision_reason: item.decision_reason == null ? null : stringValue(item.decision_reason),
      decided_at: item.decided_at == null ? null : stringValue(item.decided_at),
      outcome: item.outcome == null ? null : stringValue(item.outcome) as ToolImprovementProposal["outcome"],
      revision_request_id: item.revision_request_id == null ? null : stringValue(item.revision_request_id),
      target_version_id: item.target_version_id == null ? null : stringValue(item.target_version_id),
    }];
  });
}

export async function getToolVersionEvidence(
  toolId: string,
  versionId: string,
): Promise<ToolVersionEvidence> {
  return request<ToolVersionEvidence>(
    `${API_PREFIX}/tools/${encodeURIComponent(toolId)}/versions/${encodeURIComponent(versionId)}/evidence`,
  );
}

export async function getToolProposalEvidence(
  proposalId: string,
): Promise<ToolVersionEvidence> {
  return request<ToolVersionEvidence>(
    `${API_PREFIX}/tool-proposals/${encodeURIComponent(proposalId)}/evidence`,
  );
}

export async function getToolImprovementEvidence(
  proposalId: string,
): Promise<ToolImprovementEvidence> {
  return request<ToolImprovementEvidence>(
    `${API_PREFIX}/tool-improvement-proposals/${encodeURIComponent(proposalId)}/evidence`,
  );
}

export async function decideToolImprovement(
  proposalId: string,
  decision: "approve" | "reject",
  idempotencyKey: string,
  reason: string,
  targetVersionId?: string,
): Promise<ToolImprovementDecisionResult> {
  return request<ToolImprovementDecisionResult>(
    `${API_PREFIX}/tool-improvement-proposals/${encodeURIComponent(proposalId)}/decision`,
    {
      method: "POST",
      body: JSON.stringify({
        decision,
        idempotency_key: idempotencyKey,
        reason,
        ...(targetVersionId ? { target_version_id: targetVersionId } : {}),
      }),
    },
  );
}

function normalizeMemory(value: unknown): MemoryProposal {
  const item = asRecord(value);
  return {
    id: stringValue(item.id ?? item.proposal_id),
    kind: stringValue(item.kind ?? item.memory_type, "project"),
    content: stringValue(item.content ?? item.proposed_content),
    rationale: item.rationale ? stringValue(item.rationale) : undefined,
    confidence: numberValue(item.confidence),
    status: stringValue(item.status, "pending") as MemoryProposal["status"],
    source_run_id: item.source_run_id ? stringValue(item.source_run_id) : undefined,
    created_at: item.created_at ? stringValue(item.created_at) : undefined,
  };
}

export async function listMemoryProposals(): Promise<MemoryProposal[]> {
  const response = await request<unknown>(`${API_PREFIX}/memory/proposals?status=`);
  return listFrom(response, "proposals", "memories").map(normalizeMemory).filter((item) => item.id);
}

export async function createMemoryProposal(
  kind: "user" | "project" | "skill",
  content: string,
  sourceRunId?: string,
): Promise<MemoryProposal> {
  return normalizeMemory(
    await request<unknown>(`${API_PREFIX}/memory/proposals`, {
      method: "POST",
      body: JSON.stringify({
        kind,
        content,
        source_run_id: sourceRunId || null,
      }),
    }),
  );
}

export async function decideMemoryProposal(
  proposalId: string,
  decision: "approve" | "reject",
): Promise<void> {
  await request(`${API_PREFIX}/memory/proposals/${encodeURIComponent(proposalId)}/decision`, {
    method: "POST",
    body: JSON.stringify({ decision }),
  });
}

function normalizeModel(value: unknown): ModelHealth {
  const item = asRecord(value);
  const rawStatus = stringValue(item.status, item.loaded ? "online" : "unknown");
  return {
    id: stringValue(item.id ?? item.model ?? item.name),
    label: stringValue(item.label ?? item.name ?? item.model, "Local model"),
    role: item.role ? stringValue(item.role) : undefined,
    status: (["online", "offline", "loading", "unknown"].includes(rawStatus) ? rawStatus : "unknown") as ModelHealth["status"],
    context_window: numberValue(item.context_window ?? item.num_ctx),
    loaded: typeof item.loaded === "boolean" ? item.loaded : undefined,
    latency_ms: numberValue(item.latency_ms),
  };
}

export async function getHealth(): Promise<HealthSnapshot> {
  let response: unknown;
  try {
    response = await request<unknown>(`${API_PREFIX}/health`);
  } catch (error) {
    if (!(error instanceof ApiError) || error.status !== 404) throw error;
    response = await request<unknown>("/health");
  }
  const item = asRecord(unwrap(response));
  const ollama = asRecord(item.ollama);
  const details = asRecord(item.details);
  const model = asRecord(details.model);
  const modelReachable = typeof model.reachable === "boolean" ? model.reachable : undefined;
  const configuredModels = asRecord(model.configured);
  const configuredAvailable = asRecord(model.configured_available);
  const models = Object.entries(configuredModels).map(([role, name]) => normalizeModel({
    id: name,
    name,
    role,
    status: configuredAvailable[role] === true ? "online" : modelReachable ? "offline" : "unknown",
  }));
  return {
    status: stringValue(item.status, "unknown"),
    version: item.version ? stringValue(item.version) : undefined,
    database: typeof item.database === "boolean" ? (item.database ? "ok" : "unavailable") : item.database ? stringValue(item.database) : undefined,
    sandbox: item.reference_runner ? stringValue(item.reference_runner) : item.sandbox ? stringValue(item.sandbox) : undefined,
    ollama: Object.keys(ollama).length
      ? { status: ollama.status ? stringValue(ollama.status) : undefined, url: ollama.url ? stringValue(ollama.url) : undefined }
      : { status: modelReachable ? "ok" : "unavailable", url: model.base_url ? stringValue(model.base_url) : undefined },
    models: models.length
      ? models
      : listFrom(item.models).length
        ? listFrom(item.models).map(normalizeModel)
        : [],
  };
}

export function artifactUrl(artifactId: string, explicitUrl?: string): string {
  return explicitUrl ? apiUrl(explicitUrl) : apiUrl(`${API_PREFIX}/artifacts/${encodeURIComponent(artifactId)}`);
}

function normalizeCorpusSource(value: unknown): CorpusSource {
  const item = asRecord(value);
  return {
    id: stringValue(item.id),
    root_path: stringValue(item.root_path),
    label: stringValue(item.label, "Source"),
    kind: stringValue(item.kind, "mixed") as CorpusSource["kind"],
    provider: stringValue(item.provider, "local") as CorpusSource["provider"],
    consent: item.consent === true,
    status: stringValue(item.status, "pending") as CorpusSource["status"],
    file_count: numberValue(item.file_count) ?? 0,
    chunk_count: numberValue(item.chunk_count) ?? 0,
    last_indexed_at: item.last_indexed_at == null ? null : stringValue(item.last_indexed_at),
    last_error: item.last_error == null ? null : stringValue(item.last_error),
  };
}

function normalizeNotionConnection(value: unknown): NotionConnection {
  const item = asRecord(unwrap(value));
  return {
    configured: item.configured === true,
    token_configured: item.token_configured === true,
    root_page_ids: listFrom(item.root_page_ids).map((value) => stringValue(value)).filter(Boolean),
    label: stringValue(item.label, "Notion"),
    source: item.source ? normalizeCorpusSource(item.source) : null,
    last_synced_at: item.last_synced_at == null ? null : stringValue(item.last_synced_at),
    page_count: numberValue(item.page_count) ?? 0,
    last_error: item.last_error == null ? null : stringValue(item.last_error),
  };
}

export async function getNotionConnection(): Promise<NotionConnection> {
  return normalizeNotionConnection(
    await request<unknown>(`${API_PREFIX}/corpus/notion`),
  );
}

export async function saveNotionConnection(input: {
  accessToken?: string;
  rootPageIds: string[];
  label: string;
}): Promise<NotionConnection> {
  const response = await request<unknown>(`${API_PREFIX}/corpus/notion`, {
    method: "PUT",
    body: JSON.stringify({
      ...(input.accessToken ? { access_token: input.accessToken } : {}),
      root_page_ids: input.rootPageIds,
      label: input.label,
    }),
  });
  return normalizeNotionConnection(response);
}

export async function syncNotion(): Promise<NotionSyncResult> {
  const item = asRecord(unwrap(await request<unknown>(`${API_PREFIX}/corpus/notion/sync`, {
    method: "POST",
  })));
  return {
    pages_fetched: numberValue(item.pages_fetched) ?? 0,
    pages_written: numberValue(item.pages_written) ?? 0,
    pages_removed: numberValue(item.pages_removed) ?? 0,
    source: normalizeCorpusSource(item.source),
    index_result: item.index_result as NotionSyncResult["index_result"],
    message: stringValue(item.message),
  };
}

export async function getCorpusHealth(): Promise<CorpusHealth> {
  const item = asRecord(await request<unknown>(`${API_PREFIX}/corpus/status`));
  return {
    available: item.available === true,
    cloud_embeddings_enabled: item.cloud_embeddings_enabled === true,
    embed_model: stringValue(item.embed_model),
    rerank_model: stringValue(item.rerank_model),
    entity_graph_enabled: item.entity_graph_enabled === true,
  };
}

export async function listCorpusSources(): Promise<CorpusSource[]> {
  const response = await request<unknown>(`${API_PREFIX}/corpus/sources`);
  return listFrom(response, "sources").map(normalizeCorpusSource).filter((item) => item.id);
}

export async function createCorpusSource(
  rootPath: string,
  label: string,
  kind: CorpusSource["kind"],
): Promise<CorpusSource> {
  const response = await request<unknown>(`${API_PREFIX}/corpus/sources`, {
    method: "POST",
    body: JSON.stringify({ root_path: rootPath, label: label || null, kind }),
  });
  return normalizeCorpusSource(unwrap(response));
}

export async function setCorpusConsent(
  id: string,
  consent: boolean,
  reason?: string,
): Promise<CorpusSource> {
  const response = await request<unknown>(
    `${API_PREFIX}/corpus/sources/${encodeURIComponent(id)}/consent`,
    { method: "POST", body: JSON.stringify({ consent, reason: reason || null }) },
  );
  return normalizeCorpusSource(unwrap(response));
}

export async function reindexCorpusSource(id: string): Promise<CorpusReindexResult> {
  const response = asRecord(
    await request<unknown>(
      `${API_PREFIX}/corpus/sources/${encodeURIComponent(id)}/reindex`,
      { method: "POST" },
    ),
  );
  return {
    source_id: stringValue(response.source_id, id),
    status: stringValue(response.status, "indexed") as CorpusReindexResult["status"],
    files_indexed: numberValue(response.files_indexed) ?? 0,
    files_skipped: numberValue(response.files_skipped) ?? 0,
    files_removed: numberValue(response.files_removed) ?? 0,
    chunks: numberValue(response.chunks) ?? 0,
    message: stringValue(response.message),
  };
}

export async function deleteCorpusSource(id: string): Promise<void> {
  await request(`${API_PREFIX}/corpus/sources/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
}

function normalizeSnippet(value: unknown): KnowledgeSnippet {
  const item = asRecord(value);
  return {
    source_label: stringValue(item.source_label),
    provider: stringValue(item.provider, "local") as KnowledgeSnippet["provider"],
    rel_path: stringValue(item.rel_path),
    symbol: item.symbol == null ? null : stringValue(item.symbol),
    start_line: numberValue(item.start_line) ?? null,
    text: stringValue(item.text),
    score: numberValue(item.score) ?? 0,
  };
}

export async function searchCorpus(
  query: string,
  limit?: number,
): Promise<KnowledgeSnippet[]> {
  const response = await request<unknown>(`${API_PREFIX}/corpus/search`, {
    method: "POST",
    body: JSON.stringify({ query, ...(limit ? { limit } : {}) }),
  });
  return listFrom(response).map(normalizeSnippet);
}

function normalizeProfile(value: unknown): PersonalProfile {
  const item = asRecord(value);
  return {
    content: stringValue(item.content),
    characters: numberValue(item.characters) ?? 0,
    updated_at: item.updated_at == null ? null : stringValue(item.updated_at),
  };
}

export async function getProfile(): Promise<PersonalProfile> {
  return normalizeProfile(await request<unknown>(`${API_PREFIX}/profile`));
}

export async function saveProfile(content: string): Promise<PersonalProfile> {
  return normalizeProfile(
    await request<unknown>(`${API_PREFIX}/profile`, {
      method: "PUT",
      body: JSON.stringify({ content }),
    }),
  );
}

function normalizeModelPreference(value: unknown): ModelPreference {
  const item = asRecord(value);
  return {
    mode: item.mode === "pinned" ? "pinned" : "split",
    model: item.model == null ? null : stringValue(item.model),
    provider: item.provider === "oci" ? "oci" : item.provider === "cohere" ? "cohere" : "local",
    oci_tools: listFrom(item.oci_tools)
      .map((entry) => stringValue(entry))
      .filter((entry): entry is "x_search" | "code_interpreter" => entry === "x_search" || entry === "code_interpreter"),
    oci_available: item.oci_available === true,
    cohere_available: item.cohere_available === true,
  };
}

export async function getModelPreference(): Promise<ModelPreference> {
  return normalizeModelPreference(await request<unknown>(`${API_PREFIX}/settings/model`));
}

export async function setModelPreference(
  mode: "split" | "pinned",
  model: string | null,
  provider: "local" | "oci" | "cohere" = "local",
  ociTools: Array<"x_search" | "code_interpreter"> = ["code_interpreter"],
): Promise<ModelPreference> {
  return normalizeModelPreference(
    await request<unknown>(`${API_PREFIX}/settings/model`, {
      method: "PUT",
      body: JSON.stringify({ mode, model, provider, oci_tools: ociTools }),
    }),
  );
}

export async function getLocalModelSession(): Promise<LocalModelSession> {
  return request<LocalModelSession>(`${API_PREFIX}/model-session`);
}

export async function launchLocalModel(
  model: string,
  idleTimeoutSeconds: LocalModelSession["idle_timeout_seconds"] = 300,
  contextWindow: LocalModelSession["context_window"] = 32768,
): Promise<LocalModelSession> {
  return request<LocalModelSession>(`${API_PREFIX}/model-session/launch`, {
    method: "POST",
    body: JSON.stringify({
      model,
      idle_timeout_seconds: idleTimeoutSeconds,
      context_window: contextWindow,
    }),
  });
}

export async function stopLocalModel(force = true): Promise<LocalModelSession> {
  return request<LocalModelSession>(`${API_PREFIX}/model-session/stop`, {
    method: "POST",
    body: JSON.stringify({ force }),
  });
}

export async function listCustomers(): Promise<CustomerAccount[]> {
  return request<CustomerAccount[]>(`${API_PREFIX}/customers`);
}

export async function createCustomer(input: {
  name: string;
  aliases?: string[];
  industry?: string;
  region?: string;
}): Promise<CustomerAccount> {
  return request<CustomerAccount>(`${API_PREFIX}/customers`, {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export async function updateCustomer(
  id: string,
  input: {
    name: string;
    aliases?: string[];
    industry?: string;
    region?: string;
    status?: CustomerAccount["status"];
  },
): Promise<CustomerAccount> {
  return request<CustomerAccount>(`${API_PREFIX}/customers/${encodeURIComponent(id)}`, {
    method: "PUT",
    body: JSON.stringify(input),
  });
}

export async function deleteCustomer(id: string): Promise<void> {
  await request<void>(`${API_PREFIX}/customers/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
}

/** Search every customer record at once — accounts, notes, facts, actions,
 *  wins, and captured sources. */
export async function searchCustomerRecords(
  query: string,
  limit = 40,
): Promise<CustomerSearchResult> {
  const params = new URLSearchParams({ q: query, limit: String(limit) });
  return request<CustomerSearchResult>(`${API_PREFIX}/customers/search?${params}`);
}

export async function getCustomer(id: string): Promise<CustomerAccountDetail> {
  return request<CustomerAccountDetail>(
    `${API_PREFIX}/customers/${encodeURIComponent(id)}`,
  );
}

export async function getCustomerDashboard(): Promise<CustomerDashboard> {
  return request<CustomerDashboard>(`${API_PREFIX}/customers/dashboard`);
}

export async function captureCustomerSource(input: {
  account_id: string;
  title: string;
  content: string;
  source_kind?: "note" | "meeting" | "chat" | "notion" | "attachment";
  source_ref?: string;
  occurred_at?: string | null;
}): Promise<CustomerSource> {
  return request<CustomerSource>(`${API_PREFIX}/customers/sources`, {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export async function analyzeCustomerSource(sourceId: string): Promise<CustomerProposal> {
  return request<CustomerProposal>(
    `${API_PREFIX}/customers/sources/${encodeURIComponent(sourceId)}/analyze`,
    { method: "POST" },
  );
}

export async function saveCustomerProposal(
  proposalId: string,
  extraction: CustomerExtraction,
): Promise<CustomerProposal> {
  return request<CustomerProposal>(
    `${API_PREFIX}/customers/proposals/${encodeURIComponent(proposalId)}/save`,
    {
      method: "PUT",
      body: JSON.stringify({ extraction }),
    },
  );
}

export interface CustomerWinInput {
  title: string;
  brief?: string;
  services?: string[];
  dac_shape?: string;
  yearly_arr?: number | null;
  won_at?: string | null;
  source_ref?: string;
}

export async function createCustomerWin(
  accountId: string,
  input: CustomerWinInput,
): Promise<CustomerWin> {
  return request<CustomerWin>(
    `${API_PREFIX}/customers/${encodeURIComponent(accountId)}/wins`,
    { method: "POST", body: JSON.stringify(input) },
  );
}

export async function updateCustomerWin(
  winId: string,
  input: CustomerWinInput,
): Promise<CustomerWin> {
  return request<CustomerWin>(
    `${API_PREFIX}/customers/wins/${encodeURIComponent(winId)}`,
    { method: "PUT", body: JSON.stringify(input) },
  );
}

export async function deleteCustomerWin(winId: string): Promise<void> {
  await request<void>(`${API_PREFIX}/customers/wins/${encodeURIComponent(winId)}`, {
    method: "DELETE",
  });
}

/** Estimate a win's yearly value from the account's notes.
 *
 *  Always re-runnable, and never destructive: the result is stored beside the
 *  win as a proposal, and only reaches `yearly_arr` through `acceptWinValuation`.
 */
export async function estimateWinValuation(winId: string): Promise<WinValuation> {
  return request<WinValuation>(
    `${API_PREFIX}/customers/wins/${encodeURIComponent(winId)}/valuation`,
    { method: "POST" },
  );
}

export async function acceptWinValuation(
  winId: string,
  yearlyArr?: number | null,
): Promise<WinValuation> {
  return request<WinValuation>(
    `${API_PREFIX}/customers/wins/${encodeURIComponent(winId)}/valuation/accept`,
    {
      method: "POST",
      body: JSON.stringify({ yearly_arr: yearlyArr ?? null }),
    },
  );
}

export async function dismissWinValuation(winId: string): Promise<WinValuation> {
  return request<WinValuation>(
    `${API_PREFIX}/customers/wins/${encodeURIComponent(winId)}/valuation/dismiss`,
    { method: "POST" },
  );
}

export async function getSkuRates(): Promise<SkuRateCard> {
  return request<SkuRateCard>(`${API_PREFIX}/sku-rates`);
}

export async function saveSkuRates(
  updates: Array<{ key: string; value?: number; verified?: boolean }>,
): Promise<SkuRateCard> {
  return request<SkuRateCard>(`${API_PREFIX}/sku-rates`, {
    method: "PUT",
    body: JSON.stringify({ updates }),
  });
}

export async function updateCustomerAction(
  actionId: string,
  status: CustomerAction["status"],
): Promise<CustomerAction> {
  return request<CustomerAction>(
    `${API_PREFIX}/customers/actions/${encodeURIComponent(actionId)}`,
    { method: "PATCH", body: JSON.stringify({ status }) },
  );
}

// ── Hand edits ──────────────────────────────────────────────────────────────
// The extraction path proposes records; these let the user write and correct
// the same records directly, with no model in the loop.

export interface CustomerActionInput {
  description: string;
  owner?: string;
  due_at?: string | null;
}

export async function createCustomerAction(
  accountId: string,
  input: CustomerActionInput,
): Promise<CustomerAction> {
  return request<CustomerAction>(
    `${API_PREFIX}/customers/${encodeURIComponent(accountId)}/actions`,
    { method: "POST", body: JSON.stringify(input) },
  );
}

export async function editCustomerAction(
  actionId: string,
  input: CustomerActionInput & { status: CustomerAction["status"] },
): Promise<CustomerAction> {
  return request<CustomerAction>(
    `${API_PREFIX}/customers/actions/${encodeURIComponent(actionId)}`,
    { method: "PUT", body: JSON.stringify(input) },
  );
}

export async function deleteCustomerAction(actionId: string): Promise<void> {
  await request<void>(
    `${API_PREFIX}/customers/actions/${encodeURIComponent(actionId)}`,
    { method: "DELETE" },
  );
}

export async function createCustomerFact(
  accountId: string,
  input: { kind: string; content: string },
): Promise<CustomerFact> {
  return request<CustomerFact>(
    `${API_PREFIX}/customers/${encodeURIComponent(accountId)}/facts`,
    { method: "POST", body: JSON.stringify(input) },
  );
}

export async function updateCustomerFact(
  factId: string,
  input: { kind: string; content: string; status: CustomerFact["status"] },
): Promise<CustomerFact> {
  return request<CustomerFact>(
    `${API_PREFIX}/customers/facts/${encodeURIComponent(factId)}`,
    { method: "PUT", body: JSON.stringify(input) },
  );
}

export async function deleteCustomerFact(factId: string): Promise<void> {
  await request<void>(`${API_PREFIX}/customers/facts/${encodeURIComponent(factId)}`, {
    method: "DELETE",
  });
}

export interface CustomerPersonInput {
  name: string;
  role?: string;
  organization?: string;
}

export async function addCustomerPerson(
  accountId: string,
  input: CustomerPersonInput,
): Promise<CustomerPerson> {
  return request<CustomerPerson>(
    `${API_PREFIX}/customers/${encodeURIComponent(accountId)}/people`,
    { method: "POST", body: JSON.stringify(input) },
  );
}

export async function updateCustomerPerson(
  personId: string,
  input: CustomerPersonInput,
): Promise<CustomerPerson> {
  return request<CustomerPerson>(
    `${API_PREFIX}/customers/people/${encodeURIComponent(personId)}`,
    { method: "PUT", body: JSON.stringify(input) },
  );
}

export async function deleteCustomerPerson(personId: string): Promise<void> {
  await request<void>(`${API_PREFIX}/customers/people/${encodeURIComponent(personId)}`, {
    method: "DELETE",
  });
}

export async function updateCustomerSource(
  sourceId: string,
  input: {
    title: string;
    content: string;
    source_kind?: CustomerSource["source_kind"];
    occurred_at?: string | null;
  },
): Promise<CustomerSource> {
  return request<CustomerSource>(
    `${API_PREFIX}/customers/sources/${encodeURIComponent(sourceId)}`,
    { method: "PUT", body: JSON.stringify(input) },
  );
}

export async function deleteCustomerSource(sourceId: string): Promise<void> {
  await request<void>(`${API_PREFIX}/customers/sources/${encodeURIComponent(sourceId)}`, {
    method: "DELETE",
  });
}

export interface CustomerNoteInput {
  body: string;
  title?: string;
  pinned?: boolean;
  origin?: CustomerNote["origin"];
  origin_ref?: string;
}

/** Write a note straight onto an account: no model, no review queue. */
export async function createCustomerNote(
  accountId: string,
  input: CustomerNoteInput,
): Promise<CustomerNote> {
  return request<CustomerNote>(
    `${API_PREFIX}/customers/${encodeURIComponent(accountId)}/notes`,
    { method: "POST", body: JSON.stringify(input) },
  );
}

export async function updateCustomerNote(
  noteId: string,
  input: { title: string; body: string; pinned: boolean },
): Promise<CustomerNote> {
  return request<CustomerNote>(
    `${API_PREFIX}/customers/notes/${encodeURIComponent(noteId)}`,
    { method: "PUT", body: JSON.stringify(input) },
  );
}

export async function deleteCustomerNote(noteId: string): Promise<void> {
  await request<void>(`${API_PREFIX}/customers/notes/${encodeURIComponent(noteId)}`, {
    method: "DELETE",
  });
}

export async function getCustomerSettings(): Promise<CustomerSettings> {
  return request<CustomerSettings>(`${API_PREFIX}/customer-settings`);
}

export async function saveCustomerSettings(
  settings: Pick<CustomerSettings, "tracker_url" | "activity_template">,
): Promise<CustomerSettings> {
  return request<CustomerSettings>(`${API_PREFIX}/customer-settings`, {
    method: "PUT",
    body: JSON.stringify(settings),
  });
}

export async function createCustomerOutput(
  accountId: string,
  interactionId?: string | null,
): Promise<CustomerOutput> {
  return request<CustomerOutput>(
    `${API_PREFIX}/customers/${encodeURIComponent(accountId)}/outputs`,
    {
      method: "POST",
      body: JSON.stringify({
        kind: "activity_tracker",
        interaction_id: interactionId || null,
      }),
    },
  );
}

function numberRecord(value: unknown): Record<string, number> {
  const out: Record<string, number> = {};
  for (const [key, entry] of Object.entries(asRecord(value))) {
    const parsed = numberValue(entry);
    if (parsed != null) out[key] = parsed;
  }
  return out;
}

export async function getCodeGraphStats(): Promise<CodeGraphStats> {
  const item = asRecord(await request<unknown>(`${API_PREFIX}/corpus/graph/stats`));
  return {
    node_count: numberValue(item.node_count) ?? 0,
    edge_count: numberValue(item.edge_count) ?? 0,
    nodes_by_kind: numberRecord(item.nodes_by_kind),
    edges_by_kind: numberRecord(item.edges_by_kind),
  };
}

export async function lookupCodeGraphSymbol(name: string): Promise<CodeGraphLookup> {
  const item = asRecord(
    await request<unknown>(`${API_PREFIX}/corpus/graph/symbol/${encodeURIComponent(name)}`),
  );
  return {
    name: stringValue(item.name, name),
    definitions: listFrom(item.definitions).map((value) => {
      const entry = asRecord(value);
      return {
        kind: stringValue(entry.kind),
        name: stringValue(entry.name),
        qualname: stringValue(entry.qualname),
        rel_path: stringValue(entry.rel_path),
        start_line: numberValue(entry.start_line) ?? 0,
        end_line: numberValue(entry.end_line) ?? 0,
        source_label: stringValue(entry.source_label),
      };
    }),
    callers: listFrom(item.callers).map((value) => {
      const entry = asRecord(value);
      return {
        caller: stringValue(entry.caller),
        rel_path: stringValue(entry.rel_path),
        line: numberValue(entry.line) ?? 0,
        dst_raw: stringValue(entry.dst_raw),
        source_label: stringValue(entry.source_label),
      };
    }),
    callees: listFrom(item.callees).map((value) => {
      const entry = asRecord(value);
      return {
        dst_name: stringValue(entry.dst_name),
        dst_raw: stringValue(entry.dst_raw),
        rel_path: stringValue(entry.rel_path),
        line: numberValue(entry.line) ?? 0,
        source_label: stringValue(entry.source_label),
      };
    }),
    imports: listFrom(item.imports).map((value) => {
      const entry = asRecord(value);
      return {
        rel_path: stringValue(entry.rel_path),
        dst_raw: stringValue(entry.dst_raw),
        line: numberValue(entry.line) ?? 0,
        source_label: stringValue(entry.source_label),
      };
    }),
  };
}

export async function getEntityGraphStats(): Promise<EntityGraphStats> {
  const item = asRecord(await request<unknown>(`${API_PREFIX}/corpus/entities/stats`));
  return {
    node_count: numberValue(item.node_count) ?? 0,
    edge_count: numberValue(item.edge_count) ?? 0,
    nodes_by_kind: numberRecord(item.nodes_by_kind),
  };
}

export async function lookupEntity(name: string): Promise<EntityGraphLookup> {
  const item = asRecord(
    await request<unknown>(`${API_PREFIX}/corpus/entities/${encodeURIComponent(name)}`),
  );
  return {
    name: stringValue(item.name, name),
    kinds: listFrom(item.kinds).map((value) => stringValue(value)),
    relations_out: listFrom(item.relations_out).map((value) => {
      const entry = asRecord(value);
      return {
        relation: stringValue(entry.relation),
        dst_name: stringValue(entry.dst_name),
        rel_path: stringValue(entry.rel_path),
        source_label: stringValue(entry.source_label),
      };
    }),
    relations_in: listFrom(item.relations_in).map((value) => {
      const entry = asRecord(value);
      return {
        relation: stringValue(entry.relation),
        src_name: stringValue(entry.src_name),
        rel_path: stringValue(entry.rel_path),
        source_label: stringValue(entry.source_label),
      };
    }),
  };
}

// ── Local project asset library ─────────────────────────────────────────────

/** Normalize both the public camelCase contract and legacy snake_case fields. */
export function normalizeAsset(value: unknown): AssetV1 {
  const item = asRecord(value);
  const rawUrl = item.url ?? item.launch_url;
  return {
    id: stringValue(item.id ?? item.assetId ?? item.asset_id),
    name: stringValue(item.name, "Untitled asset"),
    summary: stringValue(item.summary ?? item.description, "No project summary is available yet."),
    category: stringValue(item.category, "Uncategorized"),
    tags: listFrom(item.tags).map((tag) => stringValue(tag)).filter(Boolean),
    framework: stringValue(item.framework, "Unknown"),
    entrypoint: stringValue(item.entrypoint),
    status: stringValue(item.status, "discovered"),
    launchConfigured: item.launchConfigured === true || item.launch_configured === true,
    launchApproved: item.launchApproved === true || item.launch_approved === true,
    launchCommand: listFrom(item.launchCommand ?? item.launch_command).map((part) => stringValue(part)),
    envKeys: listFrom(item.envKeys ?? item.env_keys).map((key) => stringValue(key)).filter(Boolean),
    envFile: listFrom(item.envFile ?? item.env_file)
      .map(normalizeAssetEnvVar)
      .filter((variable) => variable.key),
    envFilePresent: item.envFilePresent === true || item.env_file_present === true,
    url: rawUrl == null || rawUrl === "" ? null : stringValue(rawUrl),
  };
}

function normalizeAssetEnvVar(value: unknown): AssetEnvVar {
  const item = asRecord(value);
  return {
    key: stringValue(item.key),
    isSet: item.isSet === true || item.is_set === true,
    sensitive: item.sensitive === true,
  };
}

function normalizeAssetList(value: unknown): AssetV1[] {
  return listFrom(value, "assets").map(normalizeAsset).filter((asset) => asset.id);
}

export async function listAssets(): Promise<AssetV1[]> {
  return normalizeAssetList(await request<unknown>(`${API_PREFIX}/assets`));
}

export async function scanAssets(): Promise<AssetV1[]> {
  return normalizeAssetList(
    await request<unknown>(`${API_PREFIX}/assets/scan`, { method: "POST" }),
  );
}

/** Create a brand-new empty project folder in the configured projects root.
 *  The returned catalog id doubles as the project workspace id. */
export async function createProjectAsset(name: string): Promise<AssetV1> {
  const response = await request<unknown>(`${API_PREFIX}/assets/create`, {
    method: "POST",
    body: JSON.stringify({ name }),
  });
  return normalizeAsset(unwrap(response));
}

export async function startAsset(
  assetId: string,
  env: Record<string, string>,
): Promise<AssetV1> {
  const response = await request<unknown>(
    `${API_PREFIX}/assets/${encodeURIComponent(assetId)}/start`,
    { method: "POST", body: JSON.stringify({ env }) },
  );
  return normalizeAsset(unwrap(response));
}

/** Persist runtime values into the project's own .env file. Values only ever
 *  travel outbound: the response reports presence, never content. */
export async function saveAssetEnv(
  assetId: string,
  values: Record<string, string>,
): Promise<AssetV1> {
  const response = await request<unknown>(
    `${API_PREFIX}/assets/${encodeURIComponent(assetId)}/env`,
    { method: "PUT", body: JSON.stringify({ values }) },
  );
  return normalizeAsset(unwrap(response));
}

export async function approveAsset(assetId: string): Promise<AssetV1> {
  const response = await request<unknown>(
    `${API_PREFIX}/assets/${encodeURIComponent(assetId)}/approval`,
    { method: "POST" },
  );
  return normalizeAsset(unwrap(response));
}

/**
 * Draft .metis/asset.json with Command A+ for an asset that has none.
 *
 * Generation is not trust: the result lands as launch_configured and NOT
 * launch_approved, so the fingerprint review still gates the first start.
 */
export async function generateAssetRecipe(assetId: string): Promise<AssetV1> {
  const response = await request<unknown>(
    `${API_PREFIX}/assets/${encodeURIComponent(assetId)}/manifest/generate`,
    { method: "POST" },
  );
  return normalizeAsset(unwrap(response));
}

export async function revokeAssetApproval(assetId: string): Promise<AssetV1> {
  const response = await request<unknown>(
    `${API_PREFIX}/assets/${encodeURIComponent(assetId)}/approval`,
    { method: "DELETE" },
  );
  return normalizeAsset(unwrap(response));
}

export async function stopAsset(assetId: string): Promise<AssetV1> {
  const response = await request<unknown>(
    `${API_PREFIX}/assets/${encodeURIComponent(assetId)}/stop`,
    { method: "POST" },
  );
  return normalizeAsset(unwrap(response));
}

export async function getAssetLogs(assetId: string): Promise<AssetLogsV1> {
  const response = asRecord(
    unwrap(await request<unknown>(`${API_PREFIX}/assets/${encodeURIComponent(assetId)}/logs`)),
  );
  const rawLogs = response.logs;
  return {
    assetId: stringValue(response.assetId ?? response.asset_id, assetId),
    logs: Array.isArray(rawLogs)
      ? rawLogs.map((line) => stringValue(line)).join("\n")
      : stringValue(rawLogs),
  };
}

/* ── Dedicated AI Cluster sizing ──────────────────────────────────────────── */

export async function getDacCatalog(): Promise<DacCatalog> {
  return request<DacCatalog>(`${API_PREFIX}/dac/catalog`);
}

export async function estimateDac(body: DacEstimateRequest): Promise<DacEstimate> {
  return request<DacEstimate>(`${API_PREFIX}/dac/estimate`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function optimizeDac(body: DacOptimizeRequest): Promise<DacOptimizeResult> {
  return request<DacOptimizeResult>(`${API_PREFIX}/dac/optimize`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function recommendDac(body: DacRecommendRequest): Promise<DacRecommendation> {
  return request<DacRecommendation>(`${API_PREFIX}/dac/recommend`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}
