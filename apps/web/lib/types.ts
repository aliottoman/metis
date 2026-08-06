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
  // The model's own thinking for this run, streamed on a separate channel and
  // never merged into `content`. Live-only: it is not persisted with the message.
  reasoning?: string;
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
  // Set when the host proved this action cannot work. Approve is withheld while
  // it is present; the API refuses the same decision independently.
  blocked_reason?: string;
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
  /** Where the user's message was stored, so the optimistic id can be replaced. */
  message_id?: string;
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
  provider: "local" | "oci" | "cohere";
  oci_tools: Array<"x_search" | "code_interpreter">;
  oci_available: boolean;
  cohere_available: boolean;
}

export interface LocalModelOption {
  id: string;
  name: string;
  size_bytes: number;
  parameter_size: string;
  quantization: string;
  context_length: number | null;
  loaded: boolean;
  resident_bytes: number;
  expires_at: string | null;
  owned_by_metis: boolean;
}

export interface LocalModelSession {
  state: "off" | "loading" | "ready" | "busy" | "error";
  selected_model: string | null;
  idle_timeout_seconds: 60 | 300 | 900 | 1800 | 86400;
  context_window: 8192 | 16384 | 32768 | 65536 | 131072;
  expires_at: string | null;
  owned_by_metis: boolean;
  busy_count: number;
  error: string | null;
  resident_bytes: number;
  total_memory_bytes: number;
  models: LocalModelOption[];
}

export interface CustomerAccount {
  id: string;
  name: string;
  aliases: string[];
  industry: string;
  region: string;
  status: "active" | "paused" | "archived";
  open_actions: number;
  pending_notes: number;
  wins: number;
  last_interaction_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface WinValuationLine {
  sku: string;
  part_number: string | null;
  name: string;
  unit: string;
  quantity: number;
  utilization: number;
  rate: number;
  rate_verified: boolean;
  yearly_amount: number;
  basis: string;
  why: string;
}

export interface WinValuation {
  id: string;
  win_id: string;
  estimated_yearly_arr: number | null;
  currency: string;
  lines: WinValuationLine[];
  explanation: string;
  confidence: "low" | "medium" | "high";
  unpriced: string[];
  rates_verified: boolean;
  model_used: string | null;
  prompt_version: string;
  status: "proposed" | "accepted" | "dismissed";
  created_at: string;
  updated_at: string;
}

export interface CustomerWin {
  id: string;
  account_id: string;
  account_name: string;
  title: string;
  brief: string;
  services: string[];
  dac_shape: string;
  yearly_arr: number | null;
  won_at: string | null;
  source_ref: string;
  /** The estimate, when one has been run. Never the win's value — only an
   *  accepted estimate is written through to `yearly_arr`. */
  valuation: WinValuation | null;
  created_at: string;
  updated_at: string;
}

export interface SkuRate {
  key: string;
  part_number: string | null;
  unit: string;
  value: number;
  label: string;
  verified: boolean;
  aliases: string[];
  note: string;
}

export interface SkuRateCard {
  currency: string;
  hours_per_year: number;
  source_urls: string[];
  rates: SkuRate[];
  catalog_size: number;
}

export interface CustomerEvidence {
  quote: string;
  source_id: string | null;
  line_start: number | null;
  line_end: number | null;
}

/** A person as a model proposed them — no row yet, so no id. */
export interface CustomerPersonExtract {
  name: string;
  role: string;
  organization: string;
  evidence: CustomerEvidence;
}

/** A saved contact. The id is what edit and delete address, so a rename stays
 *  a rename rather than becoming a new person. */
export interface CustomerPerson extends CustomerPersonExtract {
  id: string;
}

export interface CustomerFact {
  id: string;
  account_id: string;
  interaction_id: string | null;
  kind: string;
  content: string;
  status: "active" | "superseded" | "disputed";
  confidence: number;
  evidence: CustomerEvidence;
  created_at: string;
}

export interface CustomerAction {
  id: string;
  account_id: string;
  /** Set on the cross-account attention queue; empty on account-scoped reads. */
  account_name: string;
  interaction_id: string | null;
  description: string;
  owner: string;
  due_at: string | null;
  status: "open" | "done" | "cancelled";
  evidence: CustomerEvidence;
  created_at: string;
  updated_at: string;
}

export interface CustomerInteraction {
  id: string;
  account_id: string;
  source_id: string;
  title: string;
  occurred_at: string;
  summary: string;
  created_at: string;
}

export interface CustomerSource {
  id: string;
  account_id: string;
  source_kind: string;
  title: string;
  content: string;
  source_ref: string;
  occurred_at: string | null;
  status: "waiting" | "review" | "saved" | "duplicate";
  created_at: string;
  updated_at: string;
}

export interface CustomerNote {
  id: string;
  account_id: string;
  title: string;
  body: string;
  /** Pinned notes are the ones handed to a customer-scoped conversation. */
  pinned: boolean;
  origin: "manual" | "chat";
  origin_ref: string;
  created_at: string;
  updated_at: string;
}

export interface CustomerSearchHit {
  kind: "account" | "note" | "fact" | "action" | "win" | "source";
  id: string;
  account_id: string;
  account_name: string;
  title: string;
  snippet: string;
  occurred_at: string | null;
}

export interface CustomerSearchResult {
  query: string;
  hits: CustomerSearchHit[];
  /** True when more matched than were returned, so the UI never implies
   *  the list is exhaustive. */
  truncated: boolean;
}

export interface CustomerExtraction {
  summary: string;
  occurred_at: string | null;
  people: CustomerPersonExtract[];
  facts: Array<{
    kind: string;
    content: string;
    confidence: number;
    evidence: CustomerEvidence;
  }>;
  actions: Array<{
    description: string;
    owner: string;
    due_at: string | null;
    evidence: CustomerEvidence;
  }>;
}

export interface CustomerProposal {
  id: string;
  source_id: string;
  account_id: string;
  status: "review" | "approved" | "rejected";
  extraction: CustomerExtraction;
  model: string;
  prompt_version: string;
  created_at: string;
  decided_at: string | null;
}

export interface CustomerAccountDetail {
  account: CustomerAccount;
  interactions: CustomerInteraction[];
  facts: CustomerFact[];
  actions: CustomerAction[];
  people: CustomerPerson[];
  sources: CustomerSource[];
  wins: CustomerWin[];
  notes: CustomerNote[];
}

export interface CustomerDashboard {
  active_accounts: number;
  open_actions: number;
  overdue_actions: number;
  waiting_notes: number;
  total_wins: number;
  dac_wins: number;
  total_yearly_arr: number;
  wins_by_service: Record<string, number>;
  recent_accounts: CustomerAccount[];
  priority_actions: CustomerAction[];
  recent_wins: CustomerWin[];
}

export interface CustomerOutput {
  id: string;
  account_id: string;
  kind: string;
  content: string;
  tracker_url: string;
  created_at: string;
}

export interface CustomerSettings {
  tracker_url: string;
  activity_template: string;
  updated_at: string | null;
}

/**
 * Who leads each bounded project step after the initial repository map.
 * `grok_bootstrap_local` runs them on-device; the two continuous modes hand
 * every step to their own cloud provider. Mirrors ProjectModeV1 in contracts.py.
 */
export type ProjectMode = "grok_bootstrap_local" | "grok_continuous" | "cohere_continuous";

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

export interface MemoryIndexStatus {
  consent: boolean;
  consentReason: string | null;
  cloudAvailable: boolean;
  /** True only when consent, the cloud path, and embedded vectors all line up. */
  semantic: boolean;
  active: number;
  embedded: number;
}

export interface ProjectCheck {
  name: string;
  command: string[];
  description: string;
  /** Plain-English account of what this command does, derived from its argv. */
  explanation: string;
  timeoutSeconds: number;
}

export interface ProjectVerification {
  projectId: string;
  configured: boolean;
  approved: boolean;
  fingerprint: string | null;
  checks: ProjectCheck[];
  explanation: string;
  boundary: string;
  error: string | null;
}

export interface ConversationProject {
  conversationId: string;
  projectId: string;
  mode: ProjectMode;
  updatedAt?: string;
}

/**
 * One variable from the project's own .env file. `isSet` reports that a value
 * exists on disk — the value itself never leaves the API.
 */
export interface AssetEnvVar {
  key: string;
  isSet: boolean;
  sensitive: boolean;
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
  envFile: AssetEnvVar[];
  envFilePresent: boolean;
  url: string | null;
}

export interface AssetLogsV1 {
  assetId: string;
  logs: string;
}

/* ── Dedicated AI Cluster sizing ────────────────────────────────────────────
 *
 * These mirror the API's Dac*V1 contracts field-for-field, in snake_case. The
 * rest of this file uses camelCase because those payloads are normalized on the
 * way in; sizing responses are deeply nested computed numbers with no identity
 * or lifecycle, so they are passed through as the API returns them rather than
 * maintaining a rename map that could silently drop a field.
 */

export type DacConfidenceTier = "measured" | "interpolated" | "modeled";

export interface DacGpu {
  key: string;
  label: string;
  memory_gb: number;
  memory_bandwidth_gb_s: number;
  dense_bf16_tflops: number;
  dense_fp8_tflops: number | null;
  supports_fp8: boolean;
}

export interface DacShape {
  key: string;
  gpu: string;
  gpu_count: number;
  ai_units: number;
  total_memory_gb: number;
  importable: boolean;
}

export interface DacModel {
  id: string;
  family: string;
  capability: string;
  validated_shapes: string[];
  benchmarked_shapes: string[];
  supported: boolean;
  unsupported_reason: string | null;
  config_source: string | null;
  architecture: Record<string, unknown> | null;
}

export interface DacCatalog {
  models: DacModel[];
  shapes: DacShape[];
  gpus: DacGpu[];
  quantizations: string[];
  pricing: Record<string, unknown>;
  provenance: Record<string, unknown>;
}

export interface DacVramBreakdown {
  weights_gb: number;
  kv_cache_gb: number;
  activations_gb: number;
  overhead_gb: number;
  total_gb: number;
  capacity_gb: number;
  usable_gb: number;
  utilization: number;
  status: "okay" | "moderate" | "high" | "very_high" | "insufficient";
  fits: boolean;
  max_concurrency: number;
}

export interface DacPerformance {
  ttft_s: number;
  inference_speed_tps: number;
  token_throughput_tps: number;
  request_latency_s: number;
  request_throughput_rps: number;
  request_throughput_rpm: number;
  total_throughput_tps: number;
  concurrency: number;
  prompt_tokens: number;
  response_tokens: number;
}

export interface DacConfidence {
  tier: DacConfidenceTier;
  error_margin: number | null;
  reason: string;
}

export interface DacCost {
  ai_units_per_unit: number;
  units: number;
  hours: number;
  unit_hours: number;
  billed_unit_hours: number;
  minimum_unit_hours: number;
  cost: number;
}

export interface DacEstimateRequest {
  model_id: string;
  shape: string;
  units?: number;
  prompt_tokens?: number;
  response_tokens?: number;
  concurrency?: number;
  quantization?: string | null;
  kv_quantization?: string | null;
  hours?: number;
  price_per_ai_unit_hour?: number | null;
}

export interface DacEstimate {
  model_id: string;
  shape: string;
  units: number;
  oracle_validated: boolean;
  minimum_shape: string | null;
  vram: DacVramBreakdown;
  performance: DacPerformance;
  cost: DacCost;
  confidence: DacConfidence;
  published: Record<string, unknown> | null;
  notes: string[];
}

export interface DacOptimizeRequest {
  model_id: string;
  prompt_tokens?: number;
  response_tokens?: number;
  concurrency?: number;
  max_ttft_s?: number | null;
  max_request_latency_s?: number | null;
  min_inference_speed_tps?: number | null;
  min_request_throughput_rps?: number | null;
  quantization?: string | null;
  hours?: number;
  price_per_ai_unit_hour?: number | null;
  validated_only?: boolean;
  max_units?: number;
}

export interface DacOption {
  shape: string;
  gpu: string;
  gpu_count: number;
  units: number;
  oracle_validated: boolean;
  vram: DacVramBreakdown;
  performance: DacPerformance;
  cost: DacCost;
  meets_sla: boolean;
  unmet: string[];
}

export interface DacOptimizeResult {
  model_id: string;
  options: DacOption[];
  confidence: DacConfidence;
  considered: number;
  notes: string[];
}

export interface DacRecommendRequest {
  use_case: string;
  concurrency?: number;
  prompt_tokens?: number;
  response_tokens?: number;
  max_request_latency_s?: number | null;
  capability?: string | null;
  limit?: number;
}

export interface DacCandidate {
  model_id: string;
  family: string;
  capability: string;
  score: number;
  shape: string | null;
  units: number;
  performance: DacPerformance | null;
  cost: DacCost | null;
  meets_sla: boolean;
  rationale: string | null;
}

export interface DacRecommendation {
  use_case: string;
  candidates: DacCandidate[];
  summary: string | null;
  model_used: string | null;
  model_backed: boolean;
  notes: string[];
}
