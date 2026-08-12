import { EventBroker } from "./event-broker.js";
import {
  CLINE_SDK_VERSION,
  ENGINE_NAME,
  RPC_VERSION,
  RpcFault,
  TOOL_POLICY_VERSION,
  type AbortSliceParams,
  type ContinueSliceParams,
  type RestartWithModelParams,
  type RestoreSliceParams,
  type RpcRequest,
  type SessionParams,
  type StartSliceParams,
  type SubscriptionParams,
} from "./protocol.js";
import type { SliceRuntime } from "./runtime.js";
import type { SanitizedSessionEvent } from "./protocol.js";
import { SecretRedactor } from "./sanitize.js";
import { ALLOWED_TOOLS } from "./security.js";

export interface RpcClient {
  readonly subscriptions: Set<string>;
  deliver(event: SanitizedSessionEvent): void;
}

export class RpcService {
  readonly #runtime: SliceRuntime;
  readonly #events: EventBroker;
  readonly #unsubscribeRuntime: () => void;
  readonly #redactor: SecretRedactor;
  #shuttingDown = false;

  constructor(
    runtime: SliceRuntime,
    dataDirectory: string,
    maxReplayEvents = 256,
    redactor = new SecretRedactor(),
  ) {
    this.#runtime = runtime;
    this.#events = new EventBroker(dataDirectory, maxReplayEvents);
    this.#redactor = redactor;
    this.#unsubscribeRuntime = runtime.subscribe((event) => {
      void this.#events.append(event).catch((error: unknown) => {
        // stdout is reserved for RPC. Operational diagnostics are safe on stderr.
        process.stderr.write(
          `Metis sidecar event persistence failed: ${safeErrorMessage(error)}\n`,
        );
      });
    });
  }

  async dispatch(request: RpcRequest, client: RpcClient): Promise<unknown> {
    if (this.#shuttingDown && request.method !== "shutdown") {
      throw new RpcFault("SHUTTING_DOWN", "Sidecar is shutting down");
    }
    switch (request.method) {
      case "startSlice":
        return this.#runtime.startSlice(request.params as StartSliceParams);
      case "continueSlice":
        return this.#runtime.continueSlice(
          request.params as ContinueSliceParams,
        );
      case "abortSlice":
        return this.#runtime.abortSlice(request.params as AbortSliceParams);
      case "restoreSlice":
        return this.#runtime.restoreSlice(request.params as RestoreSliceParams);
      case "restartWithModel":
        return this.#runtime.restartWithModel(
          request.params as RestartWithModelParams,
        );
      case "getUsage":
        return this.#runtime.getUsage(request.params as SessionParams);
      case "deleteSession": {
        const params = request.params as SessionParams;
        const result = await this.#runtime.deleteSession(params);
        // Clear the requested replay journal even when Cline already removed
        // the SDK record (for example after a fail-closed credential canary).
        for (const sessionId of new Set([
          params.sessionId,
          ...(result.deletedSessionIds ?? []),
        ])) {
          await this.#events.delete(sessionId);
        }
        return result;
      }
      case "getInfo":
        return {
          protocolVersion: RPC_VERSION,
          engine: ENGINE_NAME,
          runtime: this.#runtime.runtimeMode,
          sdkVersion: CLINE_SDK_VERSION,
          policyVersion: TOOL_POLICY_VERSION,
          allowedTools: [...ALLOWED_TOOLS].sort(),
        };
      case "subscribe": {
        const params = request.params as SubscriptionParams;
        client.subscriptions.add(params.sessionId);
        return this.#events.subscribe(
          params.sessionId,
          params.afterCursor ?? 0,
          client.deliver,
        );
      }
      case "shutdown":
        if (!this.#shuttingDown) {
          this.#shuttingDown = true;
          this.#unsubscribeRuntime();
          await this.#runtime.shutdown();
        }
        return { state: "shutting_down" };
    }
  }

  disconnect(client: RpcClient): void {
    for (const sessionId of client.subscriptions) {
      this.#events.unsubscribe(sessionId, client.deliver);
    }
    client.subscriptions.clear();
  }

  normalizeError(error: unknown): RpcFault {
    return normalizeFault(error, this.#redactor);
  }
}

export function safeErrorMessage(error: unknown): string {
  if (error instanceof RpcFault) return error.message;
  if (error instanceof Error) return error.message.slice(0, 2048);
  return "Unknown runtime error";
}

export function normalizeFault(
  error: unknown,
  redactor = new SecretRedactor(),
): RpcFault {
  if (error instanceof RpcFault) return error;
  const message = redactor
    .redact(safeErrorMessage(error))
    .replace(/\bBearer\s+\S+/gi, "[REDACTED]")
    .replace(/\b(?:sk|key|token)[-_][A-Za-z0-9._~+/=-]{12,}/gi, "[REDACTED]");
  return new RpcFault("RUNTIME_ERROR", message);
}
