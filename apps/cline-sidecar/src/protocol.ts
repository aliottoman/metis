import { isAbsolute, normalize, parse, resolve } from "node:path";

export const RPC_VERSION = "1" as const;
export const ENGINE_NAME = "clinecore" as const;
export const CLINE_SDK_VERSION = "0.0.72" as const;
export const TOOL_POLICY_VERSION = "1" as const;
export const MAX_FRAME_BYTES = 1024 * 1024;
export const MAX_PROMPT_BYTES = 256 * 1024;
export const MAX_SYSTEM_PROMPT_BYTES = 64 * 1024;

export const TOOL_DENIAL_REASON_CODES = [
  "tool_not_allowed",
  "invalid_search_scope",
  "missing_path",
  "invalid_path",
  "outside_workspace",
  "protected_path",
  "symlink_path",
  "outside_slice",
  "stuck_inspection_loop",
  // The model asked to run something instead of naming one of Metis's checks.
  "unknown_check",
  "command_not_permitted",
] as const;

/** Mirrors project_slices.MAX_SLICE_FILES / coding_contracts.MAX_ALLOWED_PATHS. */
export const MAX_ALLOWED_PATHS = 8;

/**
 * Files the session may read but must never change.
 *
 * A broad-scope session cannot be expressed as an 8-entry allowlist, so the
 * direct path supplies a deny-list instead: everything in the project is
 * writable except these. Metis refuses the same paths again at import, so a
 * bypass here changes nothing that reaches disk.
 */
export const MAX_PROTECTED_PATHS = 256;

export type ToolDenialReasonCode = (typeof TOOL_DENIAL_REASON_CODES)[number];

/**
 * The complete set of checks a model may ask Metis to run.
 *
 * The model selects a NAME. It never supplies a command, arguments, a path or
 * a shell string: Metis owns the argv and runs it in its pinned networkless
 * verifier against the disposable mirror. Nothing is executed in this process.
 */
export const HOST_CHECKS = [
  "imports",
  "pytest",
  "ruff",
  "acceptance",
  "full",
] as const;

export type HostCheck = (typeof HOST_CHECKS)[number];

export function isHostCheck(value: unknown): value is HostCheck {
  return (
    typeof value === "string" &&
    (HOST_CHECKS as readonly string[]).includes(value)
  );
}

/** One finding returned by a host check. Bounded and already redacted. */
export interface HostCheckFinding {
  path: string;
  severity: "error" | "warning";
  detail: string;
}

export interface HostCheckResult {
  check: HostCheck;
  ok: boolean;
  /** Why the check could not run at all, as opposed to running and failing. */
  unavailable?: string;
  errors: number;
  warnings: number;
  findings: HostCheckFinding[];
  durationMs?: number;
  truncated?: boolean;
}

/**
 * A request travelling sidecar → host on the same pipe as everything else.
 *
 * The host answers with a frame carrying the same id and a `hostResult`. The
 * id space is disjoint from RPC request ids, so neither side can confuse a
 * reply to its own request with a request from the other.
 */
export interface HostCallRequest {
  version: typeof RPC_VERSION;
  method: "hostCall";
  id: string;
  params: {
    sessionId: string;
    call: "runCheck";
    check: HostCheck;
  };
}

export interface HostCallResponse {
  version: typeof RPC_VERSION;
  id: string;
  hostResult?: HostCheckResult;
  hostError?: { message: string };
}

export function hostCallRequest(
  id: string,
  sessionId: string,
  check: HostCheck,
): HostCallRequest {
  return {
    version: RPC_VERSION,
    method: "hostCall",
    id,
    params: { sessionId, call: "runCheck", check },
  };
}

export function parseHostCallResponse(
  value: unknown,
): HostCallResponse | undefined {
  if (value === null || typeof value !== "object" || Array.isArray(value))
    return undefined;
  const candidate = value as Record<string, unknown>;
  if (typeof candidate.id !== "string") return undefined;
  if (!("hostResult" in candidate) && !("hostError" in candidate))
    return undefined;
  return candidate as unknown as HostCallResponse;
}

export const RPC_METHODS = [
  "startSlice",
  "continueSlice",
  "abortSlice",
  "restoreSlice",
  "restartWithModel",
  "subscribe",
  "getUsage",
  "deleteSession",
  "getInfo",
  "shutdown",
] as const;

export type RpcMethod = (typeof RPC_METHODS)[number];
export type RpcId = string;
export type SliceState =
  "starting" | "running" | "idle" | "completed" | "aborted" | "failed";

export type SliceFinishReason =
  "completed" | "aborted" | "error" | "mistake_limit" | "max_iterations";

/** Host-owned interpretation of a controlled SDK stop; raw finishReason remains intact. */
export type ControlledStopReason = "max_iterations";

export interface ModelRoute {
  providerId: string;
  modelId: string;
}

export interface ProviderInput extends ModelRoute {
  apiKey?: string;
  baseUrl?: string;
  headers?: Record<string, string>;
}

export interface SliceLimits {
  maxIterations?: number;
  timeoutMs?: number;
  maxTokensPerTurn?: number;
}

export interface StartSliceParams {
  sessionId?: string;
  prompt: string;
  systemPrompt?: string;
  workspaceRoot: string;
  provider: ProviderInput;
  limits?: SliceLimits;
  /**
   * The exact files this slice may create or edit, workspace-relative.
   * Absent means no slice-scoped write restriction is enforced by the
   * sidecar itself (Metis's independent host-side diff check still applies
   * either way); present-and-empty means a read-only round.
   */
  allowedPaths?: string[];
  /** Files readable but never writable; see MAX_PROTECTED_PATHS. */
  protectedPaths?: string[];
  /**
   * The `cline_direct` path, where one session owns the whole authorized
   * scope. Disables the sliced path's fixed inspection-count stop; every
   * path/security denial and hard limit is unchanged. Absent means false, so
   * a frozen sliced checkpoint keeps exactly the behaviour it started under.
   */
  broadScope?: boolean;
}

export interface ContinueSliceParams {
  sessionId: string;
  /** Host-chosen identity for a transcript fork after a sidecar restart. */
  recoverySessionId?: string;
  prompt: string;
  timeoutMs?: number;
  /** Rehydrates an in-memory provider connection after a sidecar restart. */
  provider?: ProviderInput;
}

export interface AbortSliceParams {
  sessionId: string;
  reason?: string;
}

export interface SessionParams {
  sessionId: string;
}

export interface RestoreSliceParams extends SessionParams {
  /** Required by the host after process restart because credentials are never persisted. */
  provider?: ProviderInput;
}

export interface SubscriptionParams extends SessionParams {
  afterCursor?: number;
}

export interface RestartWithModelParams {
  sessionId: string;
  newSessionId?: string;
  prompt: string;
  provider: ProviderInput;
  limits?: SliceLimits;
  /** Carried forward from the parent slice; see StartSliceParams.allowedPaths. */
  allowedPaths?: string[];
}

export type EmptyParams = Record<string, never>;

export interface RpcParamsByMethod {
  startSlice: StartSliceParams;
  continueSlice: ContinueSliceParams;
  abortSlice: AbortSliceParams;
  restoreSlice: RestoreSliceParams;
  restartWithModel: RestartWithModelParams;
  subscribe: SubscriptionParams;
  getUsage: SessionParams;
  deleteSession: SessionParams;
  getInfo: EmptyParams;
  shutdown: EmptyParams;
}

export interface RpcRequest<M extends RpcMethod = RpcMethod> {
  version: typeof RPC_VERSION;
  id: RpcId;
  method: M;
  params: RpcParamsByMethod[M];
}

export interface Usage {
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  costUsd?: number;
  requests: number;
  /**
   * Prompt-cache counters, when the provider reports them. Absent means "not
   * reported", never "zero" -- the difference decides whether a 496k-input
   * round was re-sending its context or reading it back from cache.
   */
  cacheReadTokens?: number;
  cacheWriteTokens?: number;
}

export interface SliceResult {
  sessionId: string;
  state: SliceState;
  /** Exact Cline runtime stop reason; absent for restored/in-flight sessions. */
  finishReason?: SliceFinishReason;
  /** Narrow host normalization backed by configured-limit and exact SDK evidence. */
  controlledStopReason?: ControlledStopReason;
  /** Aggregate run shape only; tool arguments and contents never cross this boundary. */
  iterations?: number;
  toolCallCount?: number;
  summary?: string;
  usage: Usage;
  model?: ModelRoute;
  parentSessionId?: string;
}

export interface SubscriptionResult {
  sessionId: string;
  cursor: number;
  oldestCursor: number;
  replayed: number;
  overflowed: boolean;
}

export interface DeleteSessionResult {
  sessionId: string;
  deleted: boolean;
  /** The current session and every private ancestor removed with it. */
  deletedSessionIds?: string[];
}

export interface SidecarInfo {
  protocolVersion: typeof RPC_VERSION;
  engine: typeof ENGINE_NAME;
  runtime: "cline" | "fake";
  sdkVersion: typeof CLINE_SDK_VERSION;
  policyVersion: typeof TOOL_POLICY_VERSION;
  allowedTools: string[];
}

export interface SanitizedSessionEvent {
  sessionId: string;
  cursor: number;
  type: string;
  status?: string;
  tool?: string;
  /** Fixed, non-sensitive classification; present only for denied tool calls. */
  reasonCode?: ToolDenialReasonCode;
  path?: string;
  message?: string;
  usage?: Usage;
  /**
   * Whether `usage` counts this event alone or everything since the session
   * began. Cline reports session totals, so a consumer that sums usage events
   * would multiply the bill; the delta is computed from the previous snapshot
   * instead.
   */
  usageScope?: "cumulative" | "incremental";
  /** Coarse SDK failure classification, e.g. an error name or code. */
  failureClass?: string;
  /** 1-based agent iteration this event belongs to, when observable. */
  iteration?: number;
  occurredAt: string;
}

export interface RpcSuccess {
  version: typeof RPC_VERSION;
  id: RpcId;
  result: unknown;
}

export interface RpcErrorBody {
  code:
    | "INVALID_REQUEST"
    | "METHOD_NOT_FOUND"
    | "SESSION_NOT_FOUND"
    | "SESSION_BUSY"
    | "POLICY_DENIED"
    | "RUNTIME_ERROR"
    | "FRAME_TOO_LARGE"
    | "SHUTTING_DOWN";
  message: string;
  data?: Record<string, unknown>;
}

export interface RpcFailure {
  version: typeof RPC_VERSION;
  id: RpcId;
  error: RpcErrorBody;
}

export interface RpcNotification {
  version: typeof RPC_VERSION;
  method: "event";
  params: SanitizedSessionEvent;
}

export type RpcEnvelope = RpcSuccess | RpcFailure | RpcNotification;

export class RpcFault extends Error {
  readonly code: RpcErrorBody["code"];
  readonly data?: Record<string, unknown>;

  constructor(
    code: RpcErrorBody["code"],
    message: string,
    data?: Record<string, unknown>,
  ) {
    super(message);
    this.name = "RpcFault";
    this.code = code;
    if (data !== undefined) this.data = data;
  }
}

type JsonObject = Record<string, unknown>;

function isObject(value: unknown): value is JsonObject {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function exactKeys(
  value: JsonObject,
  allowed: readonly string[],
  label: string,
): void {
  const allowedKeys = new Set(allowed);
  const unexpected = Object.keys(value).filter((key) => !allowedKeys.has(key));
  if (unexpected.length > 0) {
    throw new RpcFault(
      "INVALID_REQUEST",
      `${label} contains unsupported field(s): ${unexpected.join(", ")}`,
    );
  }
}

function requiredString(
  value: unknown,
  label: string,
  maxBytes: number,
  options: { trim?: boolean } = {},
): string {
  if (typeof value !== "string") {
    throw new RpcFault("INVALID_REQUEST", `${label} must be a string`);
  }
  const candidate = options.trim === false ? value : value.trim();
  if (candidate.length === 0) {
    throw new RpcFault("INVALID_REQUEST", `${label} must not be empty`);
  }
  if (candidate.includes("\0")) {
    throw new RpcFault(
      "INVALID_REQUEST",
      `${label} must not contain NUL bytes`,
    );
  }
  if (Buffer.byteLength(candidate, "utf8") > maxBytes) {
    throw new RpcFault("INVALID_REQUEST", `${label} exceeds ${maxBytes} bytes`);
  }
  return candidate;
}

function optionalString(
  value: unknown,
  label: string,
  maxBytes: number,
  options: { trim?: boolean } = {},
): string | undefined {
  if (value === undefined) return undefined;
  return requiredString(value, label, maxBytes, options);
}

function sessionId(value: unknown, label = "params.sessionId"): string {
  const parsed = requiredString(value, label, 128);
  if (!/^[A-Za-z0-9][A-Za-z0-9._:-]*$/.test(parsed)) {
    throw new RpcFault(
      "INVALID_REQUEST",
      `${label} may contain only letters, digits, period, underscore, colon, and hyphen`,
    );
  }
  return parsed;
}

function positiveInteger(
  value: unknown,
  label: string,
  minimum: number,
  maximum: number,
): number | undefined {
  if (value === undefined) return undefined;
  if (
    !Number.isInteger(value) ||
    (value as number) < minimum ||
    (value as number) > maximum
  ) {
    throw new RpcFault(
      "INVALID_REQUEST",
      `${label} must be an integer from ${minimum} through ${maximum}`,
    );
  }
  return value as number;
}

function nonnegativeInteger(value: unknown, label: string): number | undefined {
  if (value === undefined) return undefined;
  if (!Number.isSafeInteger(value) || (value as number) < 0) {
    throw new RpcFault(
      "INVALID_REQUEST",
      `${label} must be a nonnegative safe integer`,
    );
  }
  return value as number;
}

function parseProvider(value: unknown): ProviderInput {
  if (!isObject(value)) {
    throw new RpcFault("INVALID_REQUEST", "params.provider must be an object");
  }
  exactKeys(
    value,
    ["providerId", "modelId", "apiKey", "baseUrl", "headers"],
    "params.provider",
  );
  const provider: ProviderInput = {
    providerId: requiredString(
      value.providerId,
      "params.provider.providerId",
      128,
    ),
    modelId: requiredString(value.modelId, "params.provider.modelId", 256),
  };
  const apiKey = optionalString(
    value.apiKey,
    "params.provider.apiKey",
    16 * 1024,
    {
      trim: false,
    },
  );
  if (apiKey !== undefined) provider.apiKey = apiKey;
  if (value.baseUrl !== undefined) {
    const rawUrl = requiredString(
      value.baseUrl,
      "params.provider.baseUrl",
      2048,
    );
    let parsedUrl: URL;
    try {
      parsedUrl = new URL(rawUrl);
    } catch {
      throw new RpcFault(
        "INVALID_REQUEST",
        "params.provider.baseUrl must be a valid URL",
      );
    }
    if (
      !new Set(["http:", "https:"]).has(parsedUrl.protocol) ||
      parsedUrl.username ||
      parsedUrl.password
    ) {
      throw new RpcFault(
        "INVALID_REQUEST",
        "params.provider.baseUrl must be HTTP(S) and must not contain credentials",
      );
    }
    provider.baseUrl = parsedUrl.toString().replace(/\/$/, "");
  }
  if (value.headers !== undefined) {
    if (!isObject(value.headers) || Object.keys(value.headers).length > 32) {
      throw new RpcFault(
        "INVALID_REQUEST",
        "params.provider.headers must contain at most 32 fields",
      );
    }
    const headers: Record<string, string> = {};
    for (const [name, rawValue] of Object.entries(value.headers)) {
      if (!/^[!#$%&'*+.^_`|~0-9A-Za-z-]+$/.test(name)) {
        throw new RpcFault(
          "INVALID_REQUEST",
          `Invalid provider header name: ${name}`,
        );
      }
      headers[name] = requiredString(
        rawValue,
        `params.provider.headers.${name}`,
        16 * 1024,
        {
          trim: false,
        },
      );
    }
    provider.headers = headers;
  }
  return provider;
}

function parseLimits(value: unknown): SliceLimits | undefined {
  if (value === undefined) return undefined;
  if (!isObject(value)) {
    throw new RpcFault("INVALID_REQUEST", "params.limits must be an object");
  }
  exactKeys(
    value,
    ["maxIterations", "timeoutMs", "maxTokensPerTurn"],
    "params.limits",
  );
  const limits: SliceLimits = {};
  const maxIterations = positiveInteger(
    value.maxIterations,
    "params.limits.maxIterations",
    1,
    200,
  );
  const timeoutMs = positiveInteger(
    value.timeoutMs,
    "params.limits.timeoutMs",
    1000,
    3_600_000,
  );
  const maxTokensPerTurn = positiveInteger(
    value.maxTokensPerTurn,
    "params.limits.maxTokensPerTurn",
    256,
    262_144,
  );
  if (maxIterations !== undefined) limits.maxIterations = maxIterations;
  if (timeoutMs !== undefined) limits.timeoutMs = timeoutMs;
  if (maxTokensPerTurn !== undefined)
    limits.maxTokensPerTurn = maxTokensPerTurn;
  return limits;
}

/** A bounded, workspace-relative POSIX path — no absolute paths, no `..`/`.`. */
function relativePath(value: unknown, label: string): string {
  const candidate = requiredString(value, label, 1024).replaceAll("\\", "/");
  const stripped = candidate.startsWith("./") ? candidate.slice(2) : candidate;
  const parts = stripped.split("/");
  if (
    stripped.startsWith("/") ||
    parts.some((part) => part === "" || part === "." || part === "..")
  ) {
    throw new RpcFault(
      "INVALID_REQUEST",
      `${label} must be a safe relative path`,
    );
  }
  return stripped;
}

function parseAllowedPaths(
  value: unknown,
  label: string,
  limit: number = MAX_ALLOWED_PATHS,
): string[] | undefined {
  if (value === undefined) return undefined;
  if (!Array.isArray(value) || value.length > limit) {
    throw new RpcFault(
      "INVALID_REQUEST",
      `${label} must be an array of at most ${limit} paths`,
    );
  }
  return value.map((item, index) => relativePath(item, `${label}[${index}]`));
}

function parseWorkspaceRoot(value: unknown): string {
  const candidate = requiredString(value, "params.workspaceRoot", 4096);
  if (!isAbsolute(candidate)) {
    throw new RpcFault(
      "INVALID_REQUEST",
      "params.workspaceRoot must be an absolute path",
    );
  }
  const resolved = resolve(normalize(candidate));
  if (resolved === parse(resolved).root) {
    throw new RpcFault(
      "INVALID_REQUEST",
      "params.workspaceRoot must not be a filesystem root",
    );
  }
  return resolved;
}

function parseSessionParams(value: unknown): SessionParams {
  if (!isObject(value))
    throw new RpcFault("INVALID_REQUEST", "params must be an object");
  exactKeys(value, ["sessionId"], "params");
  return { sessionId: sessionId(value.sessionId) };
}

function parseParams(
  method: RpcMethod,
  raw: unknown,
): RpcParamsByMethod[RpcMethod] {
  if (!isObject(raw))
    throw new RpcFault("INVALID_REQUEST", "params must be an object");
  switch (method) {
    case "startSlice": {
      exactKeys(
        raw,
        [
          "sessionId",
          "prompt",
          "systemPrompt",
          "workspaceRoot",
          "provider",
          "limits",
          "allowedPaths",
          "protectedPaths",
          "broadScope",
        ],
        "params",
      );
      const parsed: StartSliceParams = {
        prompt: requiredString(raw.prompt, "params.prompt", MAX_PROMPT_BYTES, {
          trim: false,
        }),
        workspaceRoot: parseWorkspaceRoot(raw.workspaceRoot),
        provider: parseProvider(raw.provider),
      };
      if (raw.sessionId !== undefined)
        parsed.sessionId = sessionId(raw.sessionId);
      const systemPrompt = optionalString(
        raw.systemPrompt,
        "params.systemPrompt",
        MAX_SYSTEM_PROMPT_BYTES,
        { trim: false },
      );
      if (systemPrompt !== undefined) parsed.systemPrompt = systemPrompt;
      const limits = parseLimits(raw.limits);
      if (limits !== undefined) parsed.limits = limits;
      const allowedPaths = parseAllowedPaths(
        raw.allowedPaths,
        "params.allowedPaths",
      );
      if (allowedPaths !== undefined) parsed.allowedPaths = allowedPaths;
      const protectedPaths = parseAllowedPaths(
        raw.protectedPaths,
        "params.protectedPaths",
        MAX_PROTECTED_PATHS,
      );
      if (protectedPaths !== undefined) parsed.protectedPaths = protectedPaths;
      if (raw.broadScope !== undefined) {
        if (typeof raw.broadScope !== "boolean") {
          throw new RpcFault(
            "INVALID_REQUEST",
            "params.broadScope must be a boolean",
          );
        }
        parsed.broadScope = raw.broadScope;
      }
      return parsed;
    }
    case "continueSlice": {
      exactKeys(
        raw,
        ["sessionId", "recoverySessionId", "prompt", "timeoutMs", "provider"],
        "params",
      );
      const parsed: ContinueSliceParams = {
        sessionId: sessionId(raw.sessionId),
        prompt: requiredString(raw.prompt, "params.prompt", MAX_PROMPT_BYTES, {
          trim: false,
        }),
      };
      if (raw.recoverySessionId !== undefined) {
        parsed.recoverySessionId = sessionId(
          raw.recoverySessionId,
          "params.recoverySessionId",
        );
      }
      const timeoutMs = positiveInteger(
        raw.timeoutMs,
        "params.timeoutMs",
        1000,
        3_600_000,
      );
      if (timeoutMs !== undefined) parsed.timeoutMs = timeoutMs;
      if (raw.provider !== undefined)
        parsed.provider = parseProvider(raw.provider);
      return parsed;
    }
    case "abortSlice": {
      exactKeys(raw, ["sessionId", "reason"], "params");
      const parsed: AbortSliceParams = { sessionId: sessionId(raw.sessionId) };
      const reason = optionalString(raw.reason, "params.reason", 2048, {
        trim: false,
      });
      if (reason !== undefined) parsed.reason = reason;
      return parsed;
    }
    case "getUsage":
    case "deleteSession":
      return parseSessionParams(raw);
    case "restoreSlice": {
      exactKeys(raw, ["sessionId", "provider"], "params");
      const parsed: RestoreSliceParams = {
        sessionId: sessionId(raw.sessionId),
      };
      if (raw.provider !== undefined)
        parsed.provider = parseProvider(raw.provider);
      return parsed;
    }
    case "subscribe": {
      exactKeys(raw, ["sessionId", "afterCursor"], "params");
      const parsed: SubscriptionParams = {
        sessionId: sessionId(raw.sessionId),
      };
      const afterCursor = nonnegativeInteger(
        raw.afterCursor,
        "params.afterCursor",
      );
      if (afterCursor !== undefined) parsed.afterCursor = afterCursor;
      return parsed;
    }
    case "restartWithModel": {
      exactKeys(
        raw,
        [
          "sessionId",
          "newSessionId",
          "prompt",
          "provider",
          "limits",
          "allowedPaths",
        ],
        "params",
      );
      const parsed: RestartWithModelParams = {
        sessionId: sessionId(raw.sessionId),
        prompt: requiredString(raw.prompt, "params.prompt", MAX_PROMPT_BYTES, {
          trim: false,
        }),
        provider: parseProvider(raw.provider),
      };
      if (raw.newSessionId !== undefined) {
        parsed.newSessionId = sessionId(
          raw.newSessionId,
          "params.newSessionId",
        );
      }
      const limits = parseLimits(raw.limits);
      if (limits !== undefined) parsed.limits = limits;
      const allowedPaths = parseAllowedPaths(
        raw.allowedPaths,
        "params.allowedPaths",
      );
      if (allowedPaths !== undefined) parsed.allowedPaths = allowedPaths;
      return parsed;
    }
    case "getInfo":
    case "shutdown":
      exactKeys(raw, [], "params");
      return {};
  }
}

export function parseRpcRequest(value: unknown): RpcRequest {
  if (!isObject(value))
    throw new RpcFault("INVALID_REQUEST", "RPC frame must be an object");
  exactKeys(value, ["version", "id", "method", "params"], "request");
  if (value.version !== RPC_VERSION) {
    throw new RpcFault(
      "INVALID_REQUEST",
      `Unsupported RPC version; expected ${RPC_VERSION}`,
    );
  }
  const id = requiredString(value.id, "request.id", 128);
  if (
    typeof value.method !== "string" ||
    !RPC_METHODS.includes(value.method as RpcMethod)
  ) {
    throw new RpcFault("METHOD_NOT_FOUND", "Unknown RPC method");
  }
  const method = value.method as RpcMethod;
  return {
    version: RPC_VERSION,
    id,
    method,
    params: parseParams(method, value.params),
  } as RpcRequest;
}

export function success(id: RpcId, result: unknown): RpcSuccess {
  return { version: RPC_VERSION, id, result };
}

export function failure(id: RpcId, fault: RpcFault): RpcFailure {
  const error: RpcErrorBody = { code: fault.code, message: fault.message };
  if (fault.data !== undefined) error.data = fault.data;
  return { version: RPC_VERSION, id, error };
}

export function eventNotification(
  params: SanitizedSessionEvent,
): RpcNotification {
  return { version: RPC_VERSION, method: "event", params };
}

export function zeroUsage(): Usage {
  return { inputTokens: 0, outputTokens: 0, totalTokens: 0, requests: 0 };
}
