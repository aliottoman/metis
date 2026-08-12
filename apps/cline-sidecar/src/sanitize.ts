import { readdir, readFile } from "node:fs/promises";
import { relative, resolve } from "node:path";

import type { CoreSessionEvent } from "@cline/sdk";

import type { ProviderInput, ToolDenialReasonCode, Usage } from "./protocol.js";

const MAX_EVENT_STRING_BYTES = 4096;
const SECRET_HEADER = /(authorization|api[-_]?key|token|cookie|secret)/i;
const GENERIC_SECRET_PATTERNS = [
  /\bBearer\s+[A-Za-z0-9._~+/=-]{8,}/gi,
  /\b(?:sk|key|token)[-_][A-Za-z0-9._~+/=-]{12,}/gi,
  /(["']?(?:api[_-]?key|access[_-]?token|refresh[_-]?token|authorization)["']?\s*[:=]\s*["']?)[^\s,"'}]{8,}/gi,
];

export interface RuntimeEvent {
  sessionId: string;
  type: string;
  status?: string;
  tool?: string;
  reasonCode?: ToolDenialReasonCode;
  path?: string;
  message?: string;
  usage?: Usage;
  usageScope?: "cumulative" | "incremental";
  failureClass?: string;
  iteration?: number;
}

/**
 * Text emitted when the SDK reports a failure and carries no diagnostic of its
 * own. A blank message is indistinguishable from "we never looked", which is
 * exactly the ambiguity that made nine failed editor calls unexplainable.
 */
export const NO_SDK_DIAGNOSTIC =
  "SDK returned a failed tool result without diagnostic text";

function bounded(value: string, maxBytes = MAX_EVENT_STRING_BYTES): string {
  if (Buffer.byteLength(value, "utf8") <= maxBytes) return value;
  const buffer = Buffer.from(value, "utf8");
  return `${buffer.subarray(0, maxBytes).toString("utf8")}…`;
}

export class SecretRedactor {
  readonly #secrets = new Set<string>();

  add(secret: string | undefined): void {
    if (secret !== undefined && secret.length >= 8) this.#secrets.add(secret);
  }

  addProvider(provider: ProviderInput): void {
    this.add(provider.apiKey);
    for (const [name, value] of Object.entries(provider.headers ?? {})) {
      if (SECRET_HEADER.test(name)) this.add(value);
    }
  }

  values(): readonly string[] {
    return [...this.#secrets];
  }

  redact(value: string): string {
    let sanitized = value;
    for (const secret of this.#secrets)
      sanitized = sanitized.replaceAll(secret, "[REDACTED]");
    for (const pattern of GENERIC_SECRET_PATTERNS) {
      sanitized = sanitized.replace(
        pattern,
        (_match, prefix: string | undefined) =>
          prefix === undefined ? "[REDACTED]" : `${prefix}[REDACTED]`,
      );
    }
    return bounded(sanitized);
  }
}

function stringValue(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function objectValue(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

function numberValue(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
    ? value
    : undefined;
}

/** Read the first present counter among several plausible SDK spellings. */
function counter(
  source: Record<string, unknown> | undefined,
  ...names: readonly string[]
): number | undefined {
  for (const name of names) {
    const value = numberValue(source?.[name]);
    if (value !== undefined) return value;
  }
  return undefined;
}

const FAILURE_MARKERS = ["fail", "error", "reject", "denied", "abort"];

/** Whether an SDK agent event describes something that did not succeed. */
function isFailure(
  sdkType: string,
  sdkEvent: Record<string, unknown> | undefined,
  toolCall: Record<string, unknown> | undefined,
): boolean {
  const lowered = sdkType.toLowerCase();
  if (FAILURE_MARKERS.some((marker) => lowered.includes(marker))) return true;
  if (sdkEvent?.isError === true || toolCall?.isError === true) return true;
  if (sdkEvent?.success === false || toolCall?.success === false) return true;
  const status = stringValue(sdkEvent?.status) ?? stringValue(toolCall?.status);
  return (
    status !== undefined &&
    FAILURE_MARKERS.some((m) => status.toLowerCase().includes(m))
  );
}

/**
 * Pull diagnostic text out of an SDK event without assuming its shape.
 *
 * The Cline event contract has already moved once under us (usage counters
 * were nested, then top level), and `@cline/core` types are not resolvable
 * here, so every plausible location is probed rather than one being trusted.
 * Order matters only in that the most specific wins.
 */
function failureText(
  sdkEvent: Record<string, unknown> | undefined,
  toolCall: Record<string, unknown> | undefined,
): string | undefined {
  const error = objectValue(sdkEvent?.error) ?? objectValue(toolCall?.error);
  const result = objectValue(toolCall?.result);
  return (
    stringValue(error?.message) ??
    stringValue(error?.detail) ??
    stringValue(sdkEvent?.error) ??
    stringValue(toolCall?.error) ??
    stringValue(sdkEvent?.errorMessage) ??
    stringValue(toolCall?.errorMessage) ??
    stringValue(result?.error) ??
    stringValue(result?.message) ??
    stringValue(sdkEvent?.message) ??
    stringValue(toolCall?.message) ??
    stringValue(sdkEvent?.detail) ??
    stringValue(sdkEvent?.reason) ??
    stringValue(toolCall?.result)
  );
}

/** A coarse, non-sensitive label for the failure, when the SDK names one. */
function failureClass(
  sdkEvent: Record<string, unknown> | undefined,
  toolCall: Record<string, unknown> | undefined,
): string | undefined {
  const error = objectValue(sdkEvent?.error) ?? objectValue(toolCall?.error);
  return (
    stringValue(error?.name) ??
    stringValue(error?.code) ??
    stringValue(error?.type) ??
    stringValue(sdkEvent?.errorCode) ??
    stringValue(toolCall?.errorCode)
  );
}

/**
 * Session-scoped agent iteration counter.
 *
 * Cline's usage counters are session totals, so "did context grow per
 * iteration" cannot be answered from them alone -- the snapshots have to be
 * attributable to an iteration and differenced. This owns both: it counts
 * `iteration_start` events and remembers the last usage it saw, so each
 * snapshot can carry a non-negative delta.
 */
export class IterationTracker {
  readonly #iterations = new Map<string, number>();
  readonly #lastUsage = new Map<string, Usage>();

  iteration(sessionId: string): number {
    return this.#iterations.get(sessionId) ?? 0;
  }

  advance(sessionId: string): number {
    const next = this.iteration(sessionId) + 1;
    this.#iterations.set(sessionId, next);
    return next;
  }

  /** The increase since the previous snapshot, clamped at zero. */
  delta(sessionId: string, usage: Usage): Usage {
    const previous = this.#lastUsage.get(sessionId);
    this.#lastUsage.set(sessionId, usage);
    if (previous === undefined) return usage;
    const step = (now: number, before: number): number =>
      Math.max(0, now - before);
    const delta: Usage = {
      inputTokens: step(usage.inputTokens, previous.inputTokens),
      outputTokens: step(usage.outputTokens, previous.outputTokens),
      totalTokens: step(usage.totalTokens, previous.totalTokens),
      requests: step(usage.requests, previous.requests),
    };
    if (usage.cacheReadTokens !== undefined) {
      delta.cacheReadTokens = step(
        usage.cacheReadTokens,
        previous.cacheReadTokens ?? 0,
      );
    }
    if (usage.cacheWriteTokens !== undefined) {
      delta.cacheWriteTokens = step(
        usage.cacheWriteTokens,
        previous.cacheWriteTokens ?? 0,
      );
    }
    if (usage.costUsd !== undefined) {
      delta.costUsd = step(usage.costUsd, previous.costUsd ?? 0);
    }
    return delta;
  }

  /** A restarted or restored session must not difference against its ancestor. */
  forget(sessionId: string): void {
    this.#iterations.delete(sessionId);
    this.#lastUsage.delete(sessionId);
  }
}

const TRANSIENT_AGENT_EVENT_TYPES = new Set([
  "content_start",
  "content_update",
  "content_end",
]);

/**
 * Project a broad SDK event onto the deliberately narrow, bounded wire event.
 *
 * Cline emits each streamed token twice: once as a raw JSON `chunk`, then as a
 * structured content event. Neither is useful operational progress, while a
 * realistic 24-turn edit produced more than 500 persisted events and evicted
 * the beginning of its own replay. Drop those fragments; retain state, tool,
 * iteration, usage, notice, error, and completion evidence.
 */
export function sanitizeCoreEvent(
  event: CoreSessionEvent,
  redactor: SecretRedactor,
  tracker?: IterationTracker,
): RuntimeEvent | undefined {
  const payload = event.payload;
  switch (event.type) {
    case "chunk":
      if (payload.stream === "agent") return undefined;
      return {
        sessionId: payload.sessionId,
        type: "text",
        status: payload.stream,
        message: redactor.redact(payload.chunk),
      };
    case "status":
      return {
        sessionId: payload.sessionId,
        type: "state",
        status: redactor.redact(payload.status),
      };
    case "ended": {
      // A session that ends in error used to record the word "error" and
      // nothing else, which is how a provider-overload message stayed
      // unreadable through an entire investigation.
      const failed = payload.reason !== "completed";
      const ending = payload as unknown as Record<string, unknown>;
      // Only a failure is interrogated for diagnostics. A clean end already
      // says everything it has to say in `status`, and repeating "completed"
      // as a message is noise, not evidence.
      const endedText = failed ? failureText(ending, undefined) : undefined;
      const endedClass = failed ? failureClass(ending, undefined) : undefined;
      return {
        sessionId: payload.sessionId,
        type: payload.reason === "completed" ? "completed" : "ended",
        status: redactor.redact(payload.reason),
        ...(endedClass === undefined
          ? {}
          : { failureClass: redactor.redact(endedClass) }),
        ...(endedText !== undefined
          ? { message: redactor.redact(endedText) }
          : failed
            ? { message: NO_SDK_DIAGNOSTIC }
            : {}),
      };
    }
    case "hook":
      return {
        sessionId: payload.sessionId,
        type: "tool",
        status: payload.hookEventName,
        ...(payload.toolName === undefined
          ? {}
          : { tool: redactor.redact(payload.toolName) }),
      };
    case "session_snapshot":
      return {
        sessionId: payload.sessionId,
        type: "state",
        status: redactor.redact(String(payload.snapshot.status)),
      };
    case "agent_event": {
      const sdkEvent = objectValue(payload.event);
      const sdkType = stringValue(sdkEvent?.type) ?? "agent_event";
      if (TRANSIENT_AGENT_EVENT_TYPES.has(sdkType)) return undefined;
      const toolCall = objectValue(sdkEvent?.toolCall);
      const sessionId = payload.sessionId;
      if (sdkType.includes("iteration_start")) tracker?.advance(sessionId);
      const iteration = tracker?.iteration(sessionId);

      // AgentUsageEvent exposes counters at the top level. Accept the legacy
      // nested form as well, because both have existed in Cline SDK events.
      const counters = objectValue(sdkEvent?.usage) ?? sdkEvent;
      const inputTokens =
        counter(counters, "inputTokens", "promptTokens", "input_tokens") ?? 0;
      const outputTokens =
        counter(
          counters,
          "outputTokens",
          "completionTokens",
          "output_tokens",
        ) ?? 0;
      const cacheReadTokens = counter(
        counters,
        "cacheReadTokens",
        "cacheReadInputTokens",
        "cache_read_input_tokens",
      );
      const cacheWriteTokens = counter(
        counters,
        "cacheWriteTokens",
        "cacheCreationInputTokens",
        "cache_creation_input_tokens",
      );
      const costUsd = counter(counters, "totalCost", "costUsd", "cost");
      const requests =
        counter(counters, "requests", "requestCount", "apiRequests") ?? 0;
      const carriesUsage =
        sdkType.includes("usage") || inputTokens > 0 || outputTokens > 0;
      // Cline reports session totals, so the raw snapshot is cumulative and
      // the delta is what answers "is context compounding".
      const cumulative: Usage = {
        inputTokens,
        outputTokens,
        // Prefer the SDK's own total; fall back to the parts, never below them.
        totalTokens: Math.max(
          counter(counters, "totalTokens", "total_tokens") ?? 0,
          inputTokens + outputTokens,
        ),
        requests,
        ...(cacheReadTokens === undefined ? {} : { cacheReadTokens }),
        ...(cacheWriteTokens === undefined ? {} : { cacheWriteTokens }),
        ...(costUsd === undefined ? {} : { costUsd }),
      };

      const failed = isFailure(sdkType, sdkEvent, toolCall);
      const rawText = failureText(sdkEvent, toolCall);
      const rawClass = failureClass(sdkEvent, toolCall);
      const toolName =
        stringValue(toolCall?.toolName) ?? stringValue(sdkEvent?.toolName);
      const path = stringValue(toolCall?.path) ?? stringValue(sdkEvent?.path);

      return {
        sessionId,
        type: sdkType.includes("tool")
          ? "tool"
          : sdkType.includes("usage")
            ? "usage"
            : "progress",
        status: redactor.redact(sdkType),
        ...(toolName === undefined ? {} : { tool: redactor.redact(toolName) }),
        // `path` is only ever the tool's own already-scoped target, which the
        // event policy already permits alongside denials.
        ...(path === undefined ? {} : { path: redactor.redact(path) }),
        ...(rawClass === undefined
          ? {}
          : { failureClass: redactor.redact(rawClass) }),
        // A failure always carries text: the SDK's own if it gave any, an
        // explicit statement that it gave none otherwise.
        ...(rawText !== undefined
          ? { message: redactor.redact(rawText) }
          : failed
            ? { message: NO_SDK_DIAGNOSTIC }
            : {}),
        ...(carriesUsage
          ? {
              usage: cumulative,
              usageScope: "cumulative" as const,
            }
          : {}),
        ...(iteration === undefined || iteration === 0 ? {} : { iteration }),
      };
    }
    case "pending_prompts":
      return {
        sessionId: payload.sessionId,
        type: "notice",
        status: "pending_input",
        message: `${payload.prompts.length} pending prompt(s)`,
      };
    case "pending_prompt_submitted":
      return {
        sessionId: payload.sessionId,
        type: "notice",
        status: "prompt_submitted",
      };
    case "team_progress":
      return {
        sessionId: payload.sessionId,
        type: "policy_violation",
        status: "team_event_blocked",
        message: "A team event was observed although team tools are disabled",
      };
  }
  const fallback = event as CoreSessionEvent & {
    payload: { sessionId: string };
  };
  return {
    sessionId: fallback.payload.sessionId,
    type: "notice",
    status: "unsupported_sdk_event",
  };
}

export async function findSecretLeaks(
  root: string,
  secrets: readonly string[],
): Promise<string[]> {
  const needles = secrets
    .filter((secret) => secret.length >= 8)
    .map((secret) => Buffer.from(secret));
  if (needles.length === 0) return [];
  const leaks: string[] = [];
  const pending = [resolve(root)];
  let scannedBytes = 0;
  const maxTotalBytes = 256 * 1024 * 1024;

  while (pending.length > 0) {
    const directory = pending.pop();
    if (directory === undefined) break;
    let entries;
    try {
      entries = await readdir(directory, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const entry of entries) {
      const path = resolve(directory, entry.name);
      if (entry.isSymbolicLink()) continue;
      if (entry.isDirectory()) {
        pending.push(path);
        continue;
      }
      if (!entry.isFile()) continue;
      let contents: Buffer;
      try {
        contents = await readFile(path);
      } catch (error) {
        // Atomic journal writes may rename a temporary file after readdir.
        if ((error as NodeJS.ErrnoException).code === "ENOENT") continue;
        throw error;
      }
      scannedBytes += contents.byteLength;
      if (scannedBytes > maxTotalBytes) {
        throw new Error("Secret-leak scan exceeded its 256 MiB safety bound");
      }
      if (needles.some((needle) => contents.includes(needle))) {
        leaks.push(relative(root, path));
      }
    }
  }
  return leaks.sort();
}
