export type RiskLevel = "R0" | "R1" | "R2" | "R3" | "R4";

export interface ConversationSummary {
  id: string;
  title: string;
  created_at?: string;
  updated_at?: string;
  last_message?: string;
}

export interface AttachmentRef {
  id: string;
  name: string;
  media_type?: string;
  size?: number;
  sha256?: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  run_id?: string;
  created_at?: string;
  attachments?: AttachmentRef[];
  streaming?: boolean;
  failed?: boolean;
}

export interface ArtifactRef {
  id: string;
  name: string;
  media_type?: string;
  size?: number;
  sha256?: string;
  download_url?: string;
}

export interface ApprovalRequest {
  id: string;
  run_id?: string;
  title: string;
  summary: string;
  risk_level?: RiskLevel;
  permissions?: string[];
  action_digest?: string;
  status?: "pending" | "approved" | "rejected";
}

export interface RunEventV1 {
  id: string;
  sequence: number;
  run_id: string;
  thread_id?: string;
  checkpoint_id?: string;
  type: string;
  timestamp?: string;
  payload: Record<string, unknown>;
}

export interface RunHandle {
  run_id: string;
  conversation_id?: string;
  status?: string;
}

export interface RunRecord {
  id: string;
  conversation_id: string;
  user_message_id?: string;
  status: string;
  graph_schema_version?: string;
  cancel_requested?: boolean;
  result?: Record<string, unknown> | null;
  last_error?: string | null;
  created_at?: string;
  updated_at?: string;
}

export interface RecoverableRun {
  run: RunRecord;
  approval: ApprovalRequest | null;
}

export interface EvalSummary {
  passed: number;
  failed: number;
  total: number;
  score?: number;
}

export interface ToolVersion {
  id: string;
  version: string;
  state: "draft" | "quarantined" | "evaluated" | "approved" | "active" | "deprecated";
  content_hash?: string;
  created_at?: string;
  evaluation?: EvalSummary;
  risk_level?: RiskLevel;
  permissions?: string[];
}

export interface ToolRecord {
  id: string;
  proposal_id?: string;
  name: string;
  description: string;
  state: ToolVersion["state"] | "pending" | "rejected";
  risk_level?: RiskLevel;
  permissions?: string[];
  active_version?: string;
  latest_version?: string;
  content_hash?: string;
  created_at?: string;
  updated_at?: string;
  evaluation?: EvalSummary;
}

export interface EvalCase {
  id: string;
  name: string;
  input: Record<string, unknown>;
  expected_properties: string[];
}

export interface ToolImprovementProposal {
  id: string;
  source_run_id: string;
  tool_id: string;
  tool_version_id: string;
  content_hash: string;
  correction: string;
  regression_eval: EvalCase;
  status: "pending" | "approved" | "rejected" | "draft" | string;
  created_at?: string;
  decision_reason?: string | null;
  decided_at?: string | null;
  outcome?: "revision_queued" | "revision_activated" | "rejected" | null;
  revision_request_id?: string | null;
  target_version_id?: string | null;
}

export interface ToolEvidenceFile {
  path: string;
  sha256: string;
  size: number;
  content: string;
}

export interface ToolVersionEvidence {
  tool_id: string;
  version_id: string;
  state: ToolVersion["state"];
  content_hash: string;
  manifest: Record<string, unknown> & { version?: string; name?: string };
  eval_report?: Record<string, unknown> | null;
  bundle_verified: boolean;
  files: ToolEvidenceFile[];
  evidence_truncated: boolean;
  compared_to_version_id?: string | null;
  source_diff: string;
}

export interface ToolImprovementEvidence {
  proposal: ToolImprovementProposal;
  base_version: ToolVersionEvidence;
  eligible_revisions: ToolVersionEvidence[];
}

export interface ToolRevisionRequest {
  id: string;
  proposal_id: string;
  tool_id: string;
  base_version_id: string;
  base_content_hash: string;
  correction: string;
  regression_eval: EvalCase;
  status: "queued";
  created_at: string;
}

export interface ToolImprovementDecisionResult {
  proposal: ToolImprovementProposal;
  outcome: "revision_queued" | "revision_activated" | "rejected";
  revision_request?: ToolRevisionRequest | null;
  activated_version_id?: string | null;
  prior_version_id?: string | null;
}

// ── Tool Factory v2: declarative tool definitions (mirror contracts.py) ───────

export interface ModelAccess {
  enabled: boolean;
  roles: Array<"planner" | "coder" | "reviewer">;
  max_calls_per_run: number;
  max_tokens_per_call: number;
  prompt_templates: Record<string, string>;
}

export interface CapabilityProfile {
  code_allowlist: string;
  runtime_allowlists: Record<string, string>;
  model_access: ModelAccess;
  filesystem: "run-io";
  network: "none";
  max_runtime_seconds: number;
  max_artifact_bytes: number;
}

export interface ToolRouteFacts {
  existing_risk: RiskLevel;
  factory_risk: RiskLevel;
  input_pipeline: "none" | "attachment_text" | "architecture_spec";
}

export interface ToolDefinition {
  schema_version?: string;
  slug: string;
  version: string;
  name: string;
  description: string;
  archetype: string;
  intent_examples: string[];
  input_contract: Record<string, unknown>;
  output_contract: Record<string, unknown>;
  route_facts: ToolRouteFacts;
  capability_profile: CapabilityProfile;
  status: "draft" | "proposed" | "defined" | "retired";
  content_hash: string;
  created_at?: string;
}

export interface ToolDefinitionRecord {
  definition: ToolDefinition;
  active: boolean;
  runnable: boolean;
  buildable: boolean;
  disabled: boolean;
  pending_definition_proposal: boolean;
  pending_build: boolean;
}

export interface ToolDefinitionProposal {
  id: string;
  definition_id: string;
  slug: string;
  version: string;
  status: "pending" | "approved" | "rejected" | "draft" | string;
  risk_level?: RiskLevel;
  summary: string;
  source_run_id?: string | null;
  decision_reason?: string | null;
  created_at?: string;
  decided_at?: string | null;
}

export interface EvalResult {
  case_id: string;
  passed: boolean;
  checks: Record<string, boolean>;
  message: string;
}

export interface EvalReport {
  passed: boolean;
  score: number;
  results: EvalResult[];
  static_checks: Record<string, boolean>;
  created_at?: string;
}

export interface ToolDefinitionBuild {
  id: string;
  definition_id: string;
  slug: string;
  version: string;
  content_hash: string;
  status: "evaluated" | "active" | "rejected" | "superseded" | string;
  eval_report?: EvalReport | null;
  source_run_id?: string | null;
  created_at?: string;
  decided_at?: string | null;
}

export interface MemoryProposal {
  id: string;
  kind: "user" | "project" | "conversation" | "skill" | string;
  content: string;
  rationale?: string;
  confidence?: number;
  status: "pending" | "approved" | "rejected";
  source_run_id?: string;
  created_at?: string;
}

export interface ModelHealth {
  id: string;
  label: string;
  role?: string;
  status: "online" | "offline" | "loading" | "unknown";
  context_window?: number;
  loaded?: boolean;
  latency_ms?: number;
}

export interface HealthSnapshot {
  status: "ok" | "degraded" | "offline" | string;
  version?: string;
  ollama?: {
    status?: string;
    url?: string;
  };
  models?: ModelHealth[];
  database?: string;
  sandbox?: string;
}

export interface ConversationDetail extends ConversationSummary {
  messages: ChatMessage[];
  latest_run_id?: string;
}

export type CorpusKind = "code" | "docs" | "notes" | "mixed";
export type KnowledgeScope = "auto" | "notion";
export type CorpusStatus =
  | "pending"
  | "indexing"
  | "indexed"
  | "error"
  | "revoked";

export interface CorpusSource {
  id: string;
  root_path: string;
  label: string;
  kind: CorpusKind;
  provider: "local" | "notion";
  consent: boolean;
  status: CorpusStatus;
  file_count: number;
  chunk_count: number;
  last_indexed_at?: string | null;
  last_error?: string | null;
}

export interface CorpusHealth {
  available: boolean;
  cloud_embeddings_enabled: boolean;
  embed_model: string;
  rerank_model: string;
  entity_graph_enabled: boolean;
}

export interface CorpusReindexResult {
  source_id: string;
  status: CorpusStatus;
  files_indexed: number;
  files_skipped: number;
  files_removed: number;
  chunks: number;
  message: string;
}

export interface KnowledgeSnippet {
  source_label: string;
  provider: "local" | "notion";
  rel_path: string;
  symbol?: string | null;
  start_line?: number | null;
  text: string;
  score: number;
}

export interface NotionConnection {
  configured: boolean;
  token_configured: boolean;
  root_page_ids: string[];
  label: string;
  source: CorpusSource | null;
  last_synced_at?: string | null;
  page_count: number;
  last_error?: string | null;
}

export interface NotionSyncResult {
  pages_fetched: number;
  pages_written: number;
  pages_removed: number;
  source: CorpusSource;
  index_result: CorpusReindexResult | null;
  message: string;
}

export interface CodeGraphStats {
  node_count: number;
  edge_count: number;
  nodes_by_kind: Record<string, number>;
  edges_by_kind: Record<string, number>;
}

export interface CodeGraphDefinition {
  kind: string;
  name: string;
  qualname: string;
  rel_path: string;
  start_line: number;
  end_line: number;
  source_label: string;
}

export interface CodeGraphCaller {
  caller: string;
  rel_path: string;
  line: number;
  dst_raw: string;
  source_label: string;
}

export interface CodeGraphCallee {
  dst_name: string;
  dst_raw: string;
  rel_path: string;
  line: number;
  source_label: string;
}

export interface CodeGraphImport {
  rel_path: string;
  dst_raw: string;
  line: number;
  source_label: string;
}

export interface CodeGraphLookup {
  name: string;
  definitions: CodeGraphDefinition[];
  callers: CodeGraphCaller[];
  callees: CodeGraphCallee[];
  imports: CodeGraphImport[];
}

export interface EntityGraphStats {
  node_count: number;
  edge_count: number;
  nodes_by_kind: Record<string, number>;
}

export interface EntityRelationOut {
  relation: string;
  dst_name: string;
  rel_path: string;
  source_label: string;
}

export interface EntityRelationIn {
  relation: string;
  src_name: string;
  rel_path: string;
  source_label: string;
}

export interface EntityGraphLookup {
  name: string;
  kinds: string[];
  relations_out: EntityRelationOut[];
  relations_in: EntityRelationIn[];
}

export interface PersonalProfile {
  content: string;
  characters: number;
  updated_at?: string | null;
}

export interface ModelPreference {
  mode: "split" | "pinned";
  model: string | null;
  provider: "local" | "oci";
  oci_tools: Array<"x_search" | "code_interpreter">;
  oci_available: boolean;
}

export type ProjectMode = "grok_bootstrap_local" | "grok_continuous";

export interface ProjectWorkspace {
  id: string;
  name: string;
  summary: string;
  framework: string | null;
  initialized: boolean;
  manifestRevision: number;
  fileCount: number;
  metisMdPath: string;
  updatedAt: string | null;
}

export interface ConversationProject {
  conversationId: string;
  projectId: string;
  mode: ProjectMode;
  updatedAt?: string;
}

/** A project discovered by Metis and exposed through the local asset runner. */
export interface AssetV1 {
  id: string;
  name: string;
  summary: string;
  category: string;
  tags: string[];
  framework: string;
  entrypoint: string;
  status: string;
  launchConfigured: boolean;
  launchApproved: boolean;
  launchCommand: string[];
  envKeys: string[];
  url: string | null;
}

export interface AssetLogsV1 {
  assetId: string;
  logs: string;
}
