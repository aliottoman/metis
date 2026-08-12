import { randomUUID } from "node:crypto";
import { chmod, lstat, mkdir, realpath } from "node:fs/promises";
import { isAbsolute, join, parse, relative, resolve, sep } from "node:path";

import type {
  AgentResult,
  ClineCore,
  ClineCoreStartInput,
  CoreSessionEvent,
  SessionRecord,
  StartSessionResult,
} from "@cline/sdk";
import { getClineDefaultSystemPrompt } from "@cline/sdk";

import {
  MAX_ALLOWED_PATHS,
  RpcFault,
  type AbortSliceParams,
  type ControlledStopReason,
  type ContinueSliceParams,
  type DeleteSessionResult,
  type ModelRoute,
  type ProviderInput,
  type RestartWithModelParams,
  type RestoreSliceParams,
  type SessionParams,
  type SliceLimits,
  type SliceResult,
  type SliceState,
  type StartSliceParams,
  type Usage,
  zeroUsage,
} from "./protocol.js";
import { EventedRuntime } from "./runtime.js";
import {
  IterationTracker,
  SecretRedactor,
  findSecretLeaks,
  sanitizeCoreEvent,
} from "./sanitize.js";
import { HostBridge } from "./host-bridge.js";
import {
  FAIL_CLOSED_TOOL_POLICIES,
  createFailClosedHooks,
  decideToolApproval,
  type CheckRunner,
  type SafeToolObserver,
} from "./security.js";
import { createRunCheckTool } from "./run-check-tool.js";

const DEFAULT_MAX_ITERATIONS = 32;
const DEFAULT_TIMEOUT_MS = 15 * 60 * 1000;
const DEFAULT_MAX_TOKENS_PER_TURN = 32_768;
const EDITOR_SAFE_TEXT_LIMIT = 5_500;
const RECOVERY_COMPACTION_MESSAGE_THRESHOLD = 12;

type InitialMessage = NonNullable<
  ClineCoreStartInput["initialMessages"]
>[number] & {
  id?: string;
  metrics?: {
    inputTokens?: number;
    outputTokens?: number;
    cacheReadTokens?: number;
    cacheWriteTokens?: number;
    cost?: number;
  };
  modelInfo?: {
    id: string;
    provider: string;
    family?: string;
  };
  ts?: number;
};

const RECOVERY_HANDOFF =
  "Metis compacted the prior implementation turns before this continuation. " +
  "The original request above and the current workspace files are authoritative. " +
  "Read only the files needed by the new continuation prompt, make the smallest complete repair, " +
  "and leave final verification to Metis.";

interface SessionMetadata {
  requests: number;
  maxIterations: number;
  model: ModelRoute;
  workspaceRoot: string;
  systemPrompt: string;
  parentSessionId?: string;
  /** See StartSliceParams.allowedPaths; undefined means no slice restriction was requested. */
  allowedPaths?: readonly string[];
  /** See StartSliceParams.broadScope. Persisted so a restore rebuilds the same hooks. */
  broadScope?: boolean;
  /**
   * See StartSliceParams.protectedPaths. Persisted for the same reason as the
   * two above: a model-switch fork builds a NEW session, and a fork that
   * forgot the deny-list would be admitted under a weaker contract than the
   * run froze. Metis re-checks every imported byte independently either way;
   * this keeps the two layers agreeing.
   */
  protectedPaths?: readonly string[];
}

interface TerminationEvidence {
  finishReason?: string;
  iterations: number;
  text: string;
}

/**
 * Normalize one pinned SDK 0.0.72 defect without rewriting its raw result.
 *
 * The agents package throws this exact error after executing the configured
 * number of iterations, then its catch path reports the generic finish reason
 * `error`. All three facts are required so an arbitrary model/runtime error at
 * or near the limit can never acquire controlled-stop semantics.
 */
export function controlledStopReason(
  result: TerminationEvidence | undefined,
  configuredMaxIterations: number,
): ControlledStopReason | undefined {
  if (result?.finishReason === "max_iterations") return "max_iterations";
  if (
    result?.finishReason === "error" &&
    result.iterations === configuredMaxIterations &&
    result.text ===
      `Agent runtime exceeded maxIterations (${configuredMaxIterations})`
  ) {
    return "max_iterations";
  }
  return undefined;
}

let configuredDataDirectory: string | undefined;

function isWithin(root: string, candidate: string): boolean {
  const result = relative(root, candidate);
  return (
    result === "" ||
    (!result.startsWith(`..${sep}`) && result !== ".." && !isAbsolute(result))
  );
}

function mapState(
  status: string | undefined,
  finishReason?: string,
): SliceState {
  if (finishReason === "completed") return "completed";
  if (finishReason === "aborted") return "aborted";
  if (finishReason !== undefined) return "failed";
  switch (status) {
    case "pending":
      return "starting";
    case "running":
      return "running";
    case "idle":
      return "idle";
    case "completed":
      return "completed";
    case "cancelled":
      return "aborted";
    default:
      return "failed";
  }
}

function isSdkSessionNotFound(error: unknown): boolean {
  if (error === null || typeof error !== "object") return false;
  const candidate = error as { code?: unknown; message?: unknown };
  return (
    candidate.code === "session_not_found" ||
    (typeof candidate.message === "string" &&
      candidate.message.toLowerCase().includes("session not found"))
  );
}

function resultUsage(result: AgentResult | undefined, requests: number): Usage {
  if (result === undefined) return { ...zeroUsage(), requests };
  const inputTokens = Math.max(0, result.usage.inputTokens);
  const outputTokens = Math.max(0, result.usage.outputTokens);
  const usage: Usage = {
    inputTokens,
    outputTokens,
    totalTokens: inputTokens + outputTokens,
    requests,
  };
  if (
    result.usage.totalCost !== undefined &&
    Number.isFinite(result.usage.totalCost)
  ) {
    usage.costUsd = Math.max(0, result.usage.totalCost);
  }
  return usage;
}

function accumulatedUsage(
  value:
    | {
        aggregateUsage?: {
          inputTokens: number;
          outputTokens: number;
          totalCost: number;
        };
        usage?: {
          inputTokens: number;
          outputTokens: number;
          totalCost: number;
        };
      }
    | undefined,
  requests: number,
): Usage {
  const source = value?.aggregateUsage ?? value?.usage;
  if (source === undefined) return { ...zeroUsage(), requests };
  const inputTokens = Math.max(0, source.inputTokens);
  const outputTokens = Math.max(0, source.outputTokens);
  return {
    inputTokens,
    outputTokens,
    totalTokens: inputTokens + outputTokens,
    costUsd: Math.max(0, source.totalCost),
    requests,
  };
}

function executionConfig(limits: SliceLimits | undefined): {
  maxConsecutiveMistakes: number;
  loopDetection: { softThreshold: number; hardThreshold: number };
  reminderAfterIterations: number;
  reminderText: string;
  maxTokensPerTurn: number;
  apiTimeoutMs: number;
} {
  const maxIterations = limits?.maxIterations ?? DEFAULT_MAX_ITERATIONS;
  return {
    maxConsecutiveMistakes: 3,
    loopDetection: { softThreshold: 2, hardThreshold: 3 },
    // The real UI canaries spent all 24 turns on file tools after already
    // producing most/all planned files. Intervene around 60% of the budget,
    // before the final planned file can be crowded out by polish and rereads.
    reminderAfterIterations: Math.max(1, Math.floor(maxIterations * 0.6)),
    reminderText:
      `You are past 60% of Metis's bounded tool-iteration budget. Complete every remaining planned file now. ` +
      `Every editor old_text/new_text value must be at most ${EDITOR_SAFE_TEXT_LIMIT} characters. ` +
      "Batch independent reads, avoid confirmation-only rereads, prefer the smallest complete implementation, " +
      "then return a concise summary without another tool call. Metis will run the independent checks.",
    maxTokensPerTurn: limits?.maxTokensPerTurn ?? DEFAULT_MAX_TOKENS_PER_TURN,
    apiTimeoutMs: Math.min(
      limits?.timeoutMs ?? DEFAULT_TIMEOUT_MS,
      5 * 60 * 1000,
    ),
  };
}

/**
 * Keep Cline's own path/tool-use contract, then narrow it to Metis' local
 * execution boundary. The stock prompt is what tells models the canonical
 * working directory and that SDK file tools require absolute paths.
 */
export function buildMetisSystemPrompt(
  workspaceRoot: string,
  providerId: string,
  systemPrompt?: string,
  allowedPaths?: readonly string[],
): string {
  const clinePrompt = getClineDefaultSystemPrompt({
    workspaceRoot,
    providerId,
  });
  const taskInstructions =
    systemPrompt === undefined
      ? ""
      : `\n\n# Metis task-specific instructions\n\n${systemPrompt}`;
  const scopeLine =
    allowedPaths === undefined
      ? ""
      : allowedPaths.length === 0
        ? "\n- This round is read-only: the editor tool is not available for any file."
        : `\n- The editor tool may create or edit only these exact files: ${allowedPaths.join(", ")}. ` +
          "Every other file, including one an earlier slice already wrote, is read-only for this round " +
          "— use read_files or search_codebase for context, never the editor.";
  const exampleRelativePath = allowedPaths?.[0];
  const exampleLine =
    exampleRelativePath === undefined
      ? ""
      : `\n- Concretely: to read or edit \`${exampleRelativePath}\`, pass the absolute path ` +
        `\`${workspaceRoot}/${exampleRelativePath}\`. Never assume the workspace root is "/" or any ` +
        "other path — a path outside the exact root above is denied, wasting a turn that a search cannot recover.";
  const metisPolicy = `# Metis execution boundary (highest priority)

- The exact disposable workspace root is: ${workspaceRoot}
- Work only inside that root. Paths passed to read_files and editor must be absolute paths inside it.${exampleLine}
- Use only read_files, search_codebase, and editor. Other tools mentioned by the general Cline prompt are unavailable.
- Never use shell commands, network access, MCP, plugins, skills, questions, subagents, or teams.
- Never read or edit .git, .cline, .metis, or secret-bearing .env files, and never traverse symbolic links.
- Batch all independent initial reads into one read_files call. Do not reread a file merely to confirm an edit; Metis verifies the final workspace.
- The editor hard-rejects any individual old_text or new_text longer than ${EDITOR_SAFE_TEXT_LIMIT} characters. Partition a large file before the first editor call and keep every chunk within that limit. Never retry a rejected oversized payload unchanged.
- Implement the smallest complete vertical slice. Finish every explicitly planned file before optional polish or extra tests.
- Make cohesive code changes and finish with a concise summary.${scopeLine}`;
  return `${clinePrompt}${taskInstructions}\n\n${metisPolicy}`;
}

function finiteMetric(value: number | undefined): number | undefined {
  return value !== undefined && Number.isFinite(value) && value >= 0
    ? value
    : undefined;
}

/**
 * Keep restart continuity without replaying a verbose tool transcript to a new
 * model on every repair iteration. The original request carries the contract;
 * the current mirror and the new repair prompt carry the state. Aggregated
 * assistant metrics are retained so child usage remains cumulative and the
 * host's ancestry-delta accounting stays exact.
 */
export function compactRecoveryMessages(
  messages: NonNullable<ClineCoreStartInput["initialMessages"]>,
): NonNullable<ClineCoreStartInput["initialMessages"]> {
  if (messages.length <= RECOVERY_COMPACTION_MESSAGE_THRESHOLD) return messages;
  const typed = messages as InitialMessage[];
  const originalRequest = typed.find(
    (message) =>
      message.role === "user" &&
      (typeof message.content === "string" ||
        message.content.some(
          (part: { type?: string; text?: string }) =>
            part.type === "text" && (part.text?.trim().length ?? 0) > 0,
        )),
  );
  const assistantMessages = typed.filter(
    (message) => message.role === "assistant",
  );
  const summaryTemplate = assistantMessages.at(-1);
  if (originalRequest === undefined || summaryTemplate === undefined)
    return messages;
  if (
    assistantMessages.some(
      (message) =>
        finiteMetric(message.metrics?.inputTokens) === undefined ||
        finiteMetric(message.metrics?.outputTokens) === undefined,
    )
  ) {
    // Unknown metrics would make a compact child's cumulative usage smaller
    // than its parent and break exact lineage-delta accounting. Fail open to
    // the full transcript for this uncommon persistence shape.
    return messages;
  }

  const metricKeys = [
    "inputTokens",
    "outputTokens",
    "cacheReadTokens",
    "cacheWriteTokens",
    "cost",
  ] as const;
  const metrics: NonNullable<InitialMessage["metrics"]> = {};
  for (const key of metricKeys) {
    const values = assistantMessages
      .map((message) => finiteMetric(message.metrics?.[key]))
      .filter((value): value is number => value !== undefined);
    if (values.length > 0)
      metrics[key] = values.reduce((total, value) => total + value, 0);
  }

  const compactSummary: InitialMessage = {
    ...summaryTemplate,
    content: [{ type: "text", text: RECOVERY_HANDOFF }],
    metrics,
  };
  return [originalRequest, compactSummary];
}

interface BuildStartInputOptions {
  sessionId: string;
  prompt: string;
  workspaceRoot: string;
  provider: ProviderInput;
  systemPrompt: string;
  limits?: SliceLimits;
  initialMessages?: ClineCoreStartInput["initialMessages"];
  parentSessionId?: string;
  toolObserver?: SafeToolObserver;
  allowedPaths?: readonly string[];
  protectedPaths?: readonly string[];
  /** Runs one Metis-owned check; absent means checks are not offered. */
  runCheck?: CheckRunner;
  /** See StartSliceParams.broadScope. */
  broadScope?: boolean;
}

/** Pure, exported for contract tests so safety/limit wiring cannot silently drift. */
export function buildClineStartInput(
  options: BuildStartInputOptions,
): ClineCoreStartInput {
  const maxIterations = options.limits?.maxIterations ?? DEFAULT_MAX_ITERATIONS;
  return {
    prompt: options.prompt,
    source: "sdk",
    interactive: false,
    ...(options.initialMessages === undefined
      ? {}
      : { initialMessages: options.initialMessages }),
    sessionMetadata: {
      metisSidecarProtocol: 1,
      requests: 1,
      maxIterations,
      ...(options.parentSessionId === undefined
        ? {}
        : { parentSessionId: options.parentSessionId }),
    },
    config: {
      sessionId: options.sessionId,
      providerId: options.provider.providerId,
      modelId: options.provider.modelId,
      ...(options.provider.apiKey === undefined
        ? {}
        : { apiKey: options.provider.apiKey }),
      ...(options.provider.baseUrl === undefined
        ? {}
        : { baseUrl: options.provider.baseUrl }),
      ...(options.provider.headers === undefined
        ? {}
        : { headers: options.provider.headers }),
      systemPrompt: options.systemPrompt,
      cwd: options.workspaceRoot,
      workspaceRoot: options.workspaceRoot,
      enableTools: true,
      enableSpawnAgent: false,
      enableAgentTeams: false,
      disableMcpSettingsTools: true,
      maxIterations,
      maxTokensPerTurn:
        options.limits?.maxTokensPerTurn ?? DEFAULT_MAX_TOKENS_PER_TURN,
      skills: [],
      checkpoint: { enabled: false },
      execution: executionConfig(options.limits),
    },
    localRuntime: {
      hooks: createFailClosedHooks({
        workspaceRoot: options.workspaceRoot,
        ...(options.allowedPaths === undefined
          ? {}
          : { allowedPaths: new Set(options.allowedPaths) }),
        ...(options.toolObserver === undefined
          ? {}
          : { observer: options.toolObserver }),
        ...(options.runCheck === undefined
          ? {}
          : { runCheck: options.runCheck }),
        ...(options.broadScope === undefined
          ? {}
          : { broadScope: options.broadScope }),
      }),
      // The one custom tool Metis registers. Present only when this session
      // actually has a host to ask: a `run_check` the model can call but Metis
      // cannot answer would be worse than not offering it at all.
      ...(options.runCheck === undefined
        ? {}
        : {
            extraTools: [
              createRunCheckTool({
                runCheck: options.runCheck,
                ...(options.toolObserver === undefined
                  ? {}
                  : { observer: options.toolObserver }),
              }),
            ],
          }),
      configExtensions: [],
    },
    toolPolicies: { ...FAIL_CLOSED_TOOL_POLICIES },
    capabilities: { requestToolApproval: decideToolApproval },
  };
}

async function validateWorkspace(workspaceRoot: string): Promise<string> {
  const stat = await lstat(workspaceRoot).catch((error: unknown) => {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      throw new RpcFault(
        "INVALID_REQUEST",
        "The disposable workspace does not exist",
      );
    }
    throw error;
  });
  if (!stat.isDirectory() || stat.isSymbolicLink()) {
    throw new RpcFault(
      "INVALID_REQUEST",
      "The disposable workspace must be a real directory",
    );
  }
  const canonical = await realpath(workspaceRoot);
  if (canonical === parse(canonical).root) {
    throw new RpcFault(
      "INVALID_REQUEST",
      "The disposable workspace must not be a filesystem root",
    );
  }
  return canonical;
}

async function configureStorage(dataDirectory: string): Promise<string> {
  if (!isAbsolute(dataDirectory))
    throw new Error("Cline dataDirectory must be absolute");
  const requested = resolve(dataDirectory);
  if (requested === parse(requested).root)
    throw new Error("Cline dataDirectory must not be a filesystem root");
  await mkdir(requested, { recursive: true, mode: 0o700 });
  const stat = await lstat(requested);
  if (!stat.isDirectory() || stat.isSymbolicLink()) {
    throw new Error(
      "Cline dataDirectory must be a real directory, not a symlink",
    );
  }
  await chmod(requested, 0o700);
  const canonical = await realpath(requested);
  if (
    configuredDataDirectory !== undefined &&
    configuredDataDirectory !== canonical
  ) {
    throw new Error(
      "A process cannot host Cline runtimes with different storage roots",
    );
  }
  configuredDataDirectory = canonical;
  process.env.CLINE_DIR = canonical;
  process.env.CLINE_DATA_DIR = canonical;
  process.env.CLINE_SESSION_DATA_DIR = join(canonical, "sessions");
  process.env.CLINE_DB_DATA_DIR = join(canonical, "db");
  return canonical;
}

export class ClineRuntime extends EventedRuntime {
  readonly runtimeMode = "cline" as const;
  readonly #core: ClineCore;
  readonly #dataDirectory: string;
  readonly #redactor: SecretRedactor;
  // Counts agent iterations and remembers the last usage snapshot per
  // session, so a cumulative counter can be reported with its delta.
  readonly #iterations = new IterationTracker();
  // Set by the server once stdio is up: run_check needs a way to reach
  // Metis, and nothing else in this runtime talks upstream.
  #bridge: HostBridge | undefined;
  readonly #metadata = new Map<string, SessionMetadata>();
  readonly #busy = new Set<string>();
  readonly #unsubscribe: () => void;
  #shuttingDown = false;

  private constructor(
    core: ClineCore,
    dataDirectory: string,
    redactor: SecretRedactor,
  ) {
    super();
    this.#core = core;
    this.#dataDirectory = dataDirectory;
    this.#redactor = redactor;
    this.#unsubscribe = core.subscribe((event: CoreSessionEvent) => {
      const sanitized = sanitizeCoreEvent(
        event,
        this.#redactor,
        this.#iterations,
      );
      if (sanitized !== undefined) this.emit(sanitized);
    });
  }

  /** Install the outbound channel `run_check` uses. */
  attachHost(bridge: HostBridge | undefined): void {
    this.#bridge = bridge;
  }

  static async create(
    dataDirectory: string,
    redactor = new SecretRedactor(),
    options: { fetch?: typeof fetch } = {},
  ): Promise<ClineRuntime> {
    const canonicalDataDirectory = await configureStorage(dataDirectory);
    const { ClineCore } = await import("@cline/sdk");
    const core = await ClineCore.create({
      clientName: "metis-cline-sidecar",
      distinctId: "metis-local-sidecar",
      backendMode: "local",
      automation: false,
      toolPolicies: { ...FAIL_CLOSED_TOOL_POLICIES },
      capabilities: { requestToolApproval: decideToolApproval },
      ...(options.fetch === undefined ? {} : { fetch: options.fetch }),
    });
    return new ClineRuntime(core, canonicalDataDirectory, redactor);
  }

  #ensureAvailable(): void {
    if (this.#shuttingDown)
      throw new RpcFault("SHUTTING_DOWN", "Runtime is shutting down");
  }

  #requireNotBusy(sessionId: string): void {
    if (this.#busy.has(sessionId)) {
      throw new RpcFault(
        "SESSION_BUSY",
        `Session ${sessionId} already has an active turn`,
      );
    }
  }

  async #withTimeout<T>(
    sessionId: string,
    timeoutMs: number,
    operation: Promise<T>,
  ): Promise<T> {
    let timer: NodeJS.Timeout | undefined;
    const timeout = new Promise<never>((_resolve, reject) => {
      timer = setTimeout(() => {
        void this.#core
          .abort(sessionId, "Metis wall-time limit exceeded")
          .finally(() => {
            reject(
              new RpcFault(
                "RUNTIME_ERROR",
                `Session exceeded its ${timeoutMs} ms wall-time limit`,
              ),
            );
          });
      }, timeoutMs);
      timer.unref();
    });
    try {
      return await Promise.race([operation, timeout]);
    } finally {
      if (timer !== undefined) clearTimeout(timer);
    }
  }

  async #assertArtifactsOwnedAndSecretFree(
    sessionId: string,
    artifactPaths: readonly string[],
    workspaceRoot: string,
  ): Promise<void> {
    const outside = artifactPaths.filter(
      (path) => !isWithin(this.#dataDirectory, resolve(path)),
    );
    if (outside.length > 0) {
      await this.#core.delete(sessionId).catch(() => false);
      throw new RpcFault(
        "POLICY_DENIED",
        "Cline attempted to persist session artifacts outside Metis-owned storage",
      );
    }
    // Scan both roots: Cline's own transcript/session storage, and the
    // disposable workspace mirror the model actually edits. The provider
    // credential is never placed in the model's own context, so a leak into
    // the mirror is not an expected path -- but the canary must still catch
    // it if a defect anywhere ever put the literal key into a written file.
    const secrets = this.#redactor.values();
    const [ownedLeaks, workspaceLeaks] = await Promise.all([
      findSecretLeaks(this.#dataDirectory, secrets),
      findSecretLeaks(workspaceRoot, secrets),
    ]);
    const leaks = [
      ...ownedLeaks,
      ...workspaceLeaks.map((file) => `workspace:${file}`),
    ];
    if (leaks.length > 0) {
      await this.#core.delete(sessionId).catch(() => false);
      throw new RpcFault(
        "POLICY_DENIED",
        "Credential material was detected in Cline session storage; the session was deleted",
        { files: leaks },
      );
    }
  }

  #remember(
    sessionId: string,
    workspaceRoot: string,
    provider: ProviderInput,
    systemPrompt: string,
    requests: number,
    maxIterations: number,
    parentSessionId?: string,
    allowedPaths?: readonly string[],
    broadScope?: boolean,
    protectedPaths?: readonly string[],
  ): SessionMetadata {
    const metadata: SessionMetadata = {
      requests,
      maxIterations,
      model: { providerId: provider.providerId, modelId: provider.modelId },
      workspaceRoot,
      systemPrompt,
      ...(parentSessionId === undefined ? {} : { parentSessionId }),
      ...(allowedPaths === undefined ? {} : { allowedPaths }),
      ...(broadScope === undefined ? {} : { broadScope }),
      ...(protectedPaths === undefined ? {} : { protectedPaths }),
    };
    this.#metadata.set(sessionId, metadata);
    return metadata;
  }

  async #persistRequestCount(
    sessionId: string,
    metadata: SessionMetadata,
  ): Promise<void> {
    await this.#core.update(sessionId, {
      metadata: {
        metisSidecarProtocol: 1,
        requests: metadata.requests,
        maxIterations: metadata.maxIterations,
        ...(metadata.parentSessionId === undefined
          ? {}
          : { parentSessionId: metadata.parentSessionId }),
        ...(metadata.allowedPaths === undefined
          ? {}
          : { allowedPaths: [...metadata.allowedPaths] }),
        ...(metadata.broadScope === undefined
          ? {}
          : { broadScope: metadata.broadScope }),
        ...(metadata.protectedPaths === undefined
          ? {}
          : { protectedPaths: [...metadata.protectedPaths] }),
      },
    });
  }

  async #metadataFor(session: SessionRecord): Promise<SessionMetadata> {
    const existing = this.#metadata.get(session.sessionId);
    if (existing !== undefined) return existing;
    const persistedRequests = session.metadata?.requests;
    const persistedMaxIterations = session.metadata?.maxIterations;
    const persistedParent = session.metadata?.parentSessionId;
    const persistedAllowedPaths = session.metadata?.allowedPaths;
    const rawProtectedPaths = session.metadata?.protectedPaths;
    const persistedProtectedPaths =
      Array.isArray(rawProtectedPaths) &&
      rawProtectedPaths.every((item: unknown) => typeof item === "string")
        ? (rawProtectedPaths as string[])
        : undefined;
    const allowedPaths =
      Array.isArray(persistedAllowedPaths) &&
      persistedAllowedPaths.length <= MAX_ALLOWED_PATHS &&
      persistedAllowedPaths.every((item: unknown) => typeof item === "string")
        ? (persistedAllowedPaths as string[])
        : undefined;
    const metadata: SessionMetadata = {
      requests:
        typeof persistedRequests === "number" &&
        Number.isSafeInteger(persistedRequests)
          ? Math.max(0, persistedRequests)
          : 0,
      maxIterations:
        typeof persistedMaxIterations === "number" &&
        Number.isSafeInteger(persistedMaxIterations) &&
        persistedMaxIterations >= 1 &&
        persistedMaxIterations <= 200
          ? persistedMaxIterations
          : DEFAULT_MAX_ITERATIONS,
      model: { providerId: session.provider, modelId: session.model },
      workspaceRoot: session.workspaceRoot,
      systemPrompt: buildMetisSystemPrompt(
        session.workspaceRoot,
        session.provider,
        undefined,
        allowedPaths,
      ),
      ...(typeof persistedParent === "string"
        ? { parentSessionId: persistedParent }
        : {}),
      ...(allowedPaths === undefined ? {} : { allowedPaths }),
      ...(typeof session.metadata?.broadScope === "boolean"
        ? { broadScope: session.metadata.broadScope }
        : {}),
      ...(persistedProtectedPaths === undefined
        ? {}
        : { protectedPaths: persistedProtectedPaths }),
    };
    this.#metadata.set(session.sessionId, metadata);
    return metadata;
  }

  async #requireSession(sessionId: string): Promise<SessionRecord> {
    const session = await this.#core.get(sessionId);
    if (session === undefined) {
      throw new RpcFault(
        "SESSION_NOT_FOUND",
        `Session ${sessionId} was not found`,
      );
    }
    return session;
  }

  async #sliceResult(
    sessionId: string,
    result: AgentResult | undefined,
    fallbackStatus?: string,
  ): Promise<SliceResult> {
    const session = await this.#requireSession(sessionId);
    const metadata = await this.#metadataFor(session);
    const usage = await this.getUsage({ sessionId }).catch(() =>
      resultUsage(result, metadata.requests),
    );
    const summary =
      result?.text === undefined
        ? undefined
        : this.#redactor.redact(result.text);
    const normalizedStop = controlledStopReason(result, metadata.maxIterations);
    return {
      sessionId,
      state: mapState(fallbackStatus ?? session.status, result?.finishReason),
      ...(result?.finishReason === undefined
        ? {}
        : { finishReason: result.finishReason }),
      ...(normalizedStop === undefined
        ? {}
        : { controlledStopReason: normalizedStop }),
      ...(result === undefined
        ? {}
        : {
            iterations: result.iterations,
            toolCallCount: result.toolCalls.length,
          }),
      usage,
      model: { ...metadata.model },
      ...(summary === undefined || summary.length === 0 ? {} : { summary }),
      ...(metadata.parentSessionId === undefined
        ? {}
        : { parentSessionId: metadata.parentSessionId }),
    };
  }

  async startSlice(params: StartSliceParams): Promise<SliceResult> {
    this.#ensureAvailable();
    const sessionId = params.sessionId ?? `metis-${randomUUID()}`;
    this.#requireNotBusy(sessionId);
    if ((await this.#core.get(sessionId)) !== undefined) {
      throw new RpcFault("SESSION_BUSY", `Session ${sessionId} already exists`);
    }
    const workspaceRoot = await validateWorkspace(params.workspaceRoot);
    const systemPrompt = buildMetisSystemPrompt(
      workspaceRoot,
      params.provider.providerId,
      params.systemPrompt,
      params.allowedPaths,
    );
    this.#redactor.addProvider(params.provider);
    this.#busy.add(sessionId);
    this.emit({ sessionId, type: "state", status: "starting" });
    try {
      const operation = this.#core.start(
        buildClineStartInput({
          sessionId,
          prompt: params.prompt,
          workspaceRoot,
          provider: params.provider,
          systemPrompt,
          ...(params.limits === undefined ? {} : { limits: params.limits }),
          ...(params.allowedPaths === undefined
            ? {}
            : { allowedPaths: params.allowedPaths }),
          ...(params.protectedPaths === undefined
            ? {}
            : { protectedPaths: params.protectedPaths }),
          ...(params.broadScope === undefined
            ? {}
            : { broadScope: params.broadScope }),
          ...(this.#bridge === undefined
            ? {}
            : {
                runCheck: (check: string) =>
                  this.#bridge!.runCheck(sessionId, check as never),
              }),
          toolObserver: ({ tool, status, reasonCode }) => {
            this.emit({
              sessionId,
              type: "tool",
              tool,
              status,
              ...(reasonCode === undefined ? {} : { reasonCode }),
            });
          },
        }),
      ) as Promise<StartSessionResult>;
      const started = await this.#withTimeout(
        sessionId,
        params.limits?.timeoutMs ?? DEFAULT_TIMEOUT_MS,
        operation,
      );
      this.#remember(
        sessionId,
        workspaceRoot,
        params.provider,
        systemPrompt,
        1,
        params.limits?.maxIterations ?? DEFAULT_MAX_ITERATIONS,
        undefined,
        params.allowedPaths,
        params.broadScope,
        params.protectedPaths,
      );
      await this.#assertArtifactsOwnedAndSecretFree(
        sessionId,
        [started.manifestPath, started.messagesPath],
        workspaceRoot,
      );
      const answer = await this.#sliceResult(
        sessionId,
        started.result,
        started.manifest.status,
      );
      this.emit({ sessionId, type: "usage", usage: answer.usage });
      return answer;
    } finally {
      this.#busy.delete(sessionId);
    }
  }

  async continueSlice(params: ContinueSliceParams): Promise<SliceResult> {
    this.#ensureAvailable();
    this.#requireNotBusy(params.sessionId);
    const session = await this.#requireSession(params.sessionId);
    const metadata = await this.#metadataFor(session);
    if (params.provider !== undefined) {
      if (
        params.provider.providerId !== metadata.model.providerId ||
        params.provider.modelId !== metadata.model.modelId
      ) {
        throw new RpcFault(
          "POLICY_DENIED",
          "Use restartWithModel to change a session model",
        );
      }
      this.#redactor.addProvider(params.provider);
      try {
        await this.#core.updateSessionConnection(
          params.sessionId,
          params.provider,
        );
      } catch (error) {
        if (!isSdkSessionNotFound(error)) throw error;
        if (params.recoverySessionId === undefined) {
          throw new RpcFault(
            "INVALID_REQUEST",
            "A host-chosen recoverySessionId is required to recover a persisted continuation",
          );
        }
        // ClineCore persists transcripts but not a live runtime across process
        // restarts. Fork under the identity supplied by Metis, then apply the
        // pending prompt exactly once.
        return this.restartWithModel({
          sessionId: params.sessionId,
          newSessionId: params.recoverySessionId,
          prompt: params.prompt,
          provider: params.provider,
          limits: {
            maxIterations: metadata.maxIterations,
            ...(params.timeoutMs === undefined
              ? {}
              : { timeoutMs: params.timeoutMs }),
          },
        });
      }
    }
    this.#busy.add(params.sessionId);
    try {
      metadata.requests += 1;
      await this.#persistRequestCount(params.sessionId, metadata);
      const operation = this.#core.send({
        sessionId: params.sessionId,
        prompt: params.prompt,
        timeoutMs: params.timeoutMs ?? DEFAULT_TIMEOUT_MS,
      });
      let result: AgentResult | undefined;
      try {
        result = await this.#withTimeout(
          params.sessionId,
          params.timeoutMs ?? DEFAULT_TIMEOUT_MS,
          operation,
        );
      } catch (error) {
        if (!isSdkSessionNotFound(error) || params.provider === undefined)
          throw error;
        if (params.recoverySessionId === undefined) {
          throw new RpcFault(
            "INVALID_REQUEST",
            "A host-chosen recoverySessionId is required to recover a persisted continuation",
          );
        }
        this.#busy.delete(params.sessionId);
        return this.restartWithModel({
          sessionId: params.sessionId,
          newSessionId: params.recoverySessionId,
          prompt: params.prompt,
          provider: params.provider,
          limits: {
            maxIterations: metadata.maxIterations,
            ...(params.timeoutMs === undefined
              ? {}
              : { timeoutMs: params.timeoutMs }),
          },
        });
      }
      const answer = await this.#sliceResult(params.sessionId, result);
      this.emit({
        sessionId: params.sessionId,
        type: "usage",
        usage: answer.usage,
      });
      return answer;
    } finally {
      this.#busy.delete(params.sessionId);
    }
  }

  async abortSlice(params: AbortSliceParams): Promise<SliceResult> {
    this.#ensureAvailable();
    await this.#requireSession(params.sessionId);
    await this.#core.abort(
      params.sessionId,
      params.reason ?? "Aborted by Metis",
    );
    const answer = await this.#sliceResult(
      params.sessionId,
      undefined,
      "cancelled",
    );
    return {
      ...answer,
      state: "aborted",
      ...(params.reason === undefined ? {} : { summary: params.reason }),
    };
  }

  async restoreSlice(params: RestoreSliceParams): Promise<SliceResult> {
    this.#ensureAvailable();
    const session = await this.#requireSession(params.sessionId);
    const metadata = await this.#metadataFor(session);
    if (params.provider !== undefined) {
      if (
        params.provider.providerId !== metadata.model.providerId ||
        params.provider.modelId !== metadata.model.modelId
      ) {
        throw new RpcFault(
          "POLICY_DENIED",
          "Use restartWithModel to change a session model",
        );
      }
      this.#redactor.addProvider(params.provider);
      try {
        await this.#core.updateSessionConnection(
          params.sessionId,
          params.provider,
        );
      } catch (error) {
        if (!isSdkSessionNotFound(error)) throw error;
        this.emit({
          sessionId: params.sessionId,
          type: "notice",
          status: "persisted_session",
          message:
            "Provider will be reattached by forking persisted messages on continuation",
        });
      }
    }
    return this.#sliceResult(params.sessionId, undefined, session.status);
  }

  async restartWithModel(params: RestartWithModelParams): Promise<SliceResult> {
    this.#ensureAvailable();
    this.#requireNotBusy(params.sessionId);
    const parent = await this.#requireSession(params.sessionId);
    const parentMetadata = await this.#metadataFor(parent);
    const newSessionId = params.newSessionId ?? `metis-${randomUUID()}`;
    this.#requireNotBusy(newSessionId);
    if ((await this.#core.get(newSessionId)) !== undefined) {
      throw new RpcFault(
        "SESSION_BUSY",
        `Session ${newSessionId} already exists`,
      );
    }
    const workspaceRoot = await validateWorkspace(parentMetadata.workspaceRoot);
    const initialMessages = compactRecoveryMessages(
      await this.#core.readLiveMessages(params.sessionId),
    );
    // A repair fork never widens scope on its own: an explicit host-supplied
    // list is honored, but a fork with no opinion of its own (including every
    // internal continueSlice fallback below) inherits the parent slice's
    // exact allowlist rather than defaulting to unrestricted.
    const allowedPaths = params.allowedPaths ?? parentMetadata.allowedPaths;
    this.#redactor.addProvider(params.provider);
    this.#busy.add(newSessionId);
    try {
      const operation = this.#core.start(
        buildClineStartInput({
          sessionId: newSessionId,
          prompt: params.prompt,
          workspaceRoot,
          provider: params.provider,
          systemPrompt: parentMetadata.systemPrompt,
          initialMessages,
          parentSessionId: params.sessionId,
          ...(params.limits === undefined ? {} : { limits: params.limits }),
          ...(allowedPaths === undefined ? {} : { allowedPaths }),
          // A fork inherits the parent's scope mode for the same reason it
          // inherits its allowlist: the fork is the same work continuing, not
          // a new admission.
          ...(parentMetadata.broadScope === undefined
            ? {}
            : { broadScope: parentMetadata.broadScope }),
          ...(parentMetadata.protectedPaths === undefined
            ? {}
            : { protectedPaths: parentMetadata.protectedPaths }),
          ...(this.#bridge === undefined
            ? {}
            : {
                runCheck: (check: string) =>
                  this.#bridge!.runCheck(newSessionId, check as never),
              }),
          toolObserver: ({ tool, status, reasonCode }) => {
            this.emit({
              sessionId: newSessionId,
              type: "tool",
              tool,
              status,
              ...(reasonCode === undefined ? {} : { reasonCode }),
            });
          },
        }),
      ) as Promise<StartSessionResult>;
      const started = await this.#withTimeout(
        newSessionId,
        params.limits?.timeoutMs ?? DEFAULT_TIMEOUT_MS,
        operation,
      );
      this.#remember(
        newSessionId,
        workspaceRoot,
        params.provider,
        parentMetadata.systemPrompt,
        1,
        params.limits?.maxIterations ?? DEFAULT_MAX_ITERATIONS,
        params.sessionId,
        allowedPaths,
        parentMetadata.broadScope,
        parentMetadata.protectedPaths,
      );
      await this.#assertArtifactsOwnedAndSecretFree(
        newSessionId,
        [started.manifestPath, started.messagesPath],
        workspaceRoot,
      );
      const answer = await this.#sliceResult(
        newSessionId,
        started.result,
        started.manifest.status,
      );
      return { ...answer, parentSessionId: params.sessionId };
    } finally {
      this.#busy.delete(newSessionId);
    }
  }

  async getUsage(params: SessionParams): Promise<Usage> {
    const session = await this.#requireSession(params.sessionId);
    const metadata = await this.#metadataFor(session);
    const usage = await this.#core.getAccumulatedUsage(params.sessionId);
    return accumulatedUsage(usage, metadata.requests);
  }

  async deleteSession(params: SessionParams): Promise<DeleteSessionResult> {
    const deletedSessionIds: string[] = [];
    const seen = new Set<string>();
    let current: string | undefined = params.sessionId;
    while (current !== undefined && deletedSessionIds.length < 64) {
      if (seen.has(current)) {
        throw new RpcFault(
          "RUNTIME_ERROR",
          "Coding-session ancestry contains a cycle",
        );
      }
      seen.add(current);
      // A deleted session can never be differenced against again; dropping it
      // also keeps the tracker from growing for the process's lifetime.
      this.#iterations.forget(current);
      this.#requireNotBusy(current);
      const session = await this.#core.get(current);
      if (session === undefined) break;
      const metadata = await this.#metadataFor(session);
      const parent = metadata.parentSessionId;
      if (session.status === "running" || session.status === "pending") {
        await this.#core.stop(current);
      }
      if (await this.#core.delete(current)) deletedSessionIds.push(current);
      this.#metadata.delete(current);
      current = parent;
    }
    return {
      sessionId: params.sessionId,
      deleted: deletedSessionIds.includes(params.sessionId),
      deletedSessionIds,
    };
  }

  async shutdown(): Promise<void> {
    if (this.#shuttingDown) return;
    this.#shuttingDown = true;
    this.#unsubscribe();
    await this.#core.dispose("Metis sidecar shutdown");
  }
}
