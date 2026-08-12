import { randomUUID } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";

import type { HostBridge } from "./host-bridge.js";

import {
  RpcFault,
  type AbortSliceParams,
  type ContinueSliceParams,
  type ControlledStopReason,
  type DeleteSessionResult,
  type ModelRoute,
  type RestartWithModelParams,
  type RestoreSliceParams,
  type SessionParams,
  type SliceResult,
  type SliceFinishReason,
  type SliceState,
  type StartSliceParams,
  type Usage,
  zeroUsage,
} from "./protocol.js";
import { EventedRuntime } from "./runtime.js";
import { SecretRedactor } from "./sanitize.js";

interface FakeSession {
  sessionId: string;
  parentSessionId?: string;
  state: SliceState;
  finishReason?: SliceFinishReason;
  controlledStopReason?: ControlledStopReason;
  summary?: string;
  usage: Usage;
  model: ModelRoute;
  workspaceRoot: string;
}

function copyUsage(usage: Usage): Usage {
  return { ...usage };
}

export class FakeRuntime extends EventedRuntime {
  readonly runtimeMode = "fake" as const;
  readonly #sessions = new Map<string, FakeSession>();
  readonly #redactor: SecretRedactor;
  #bridge: HostBridge | undefined;
  #shuttingDown = false;

  /** Installed by the server, exactly as the real runtime's is. */
  attachHost(bridge: HostBridge | undefined): void {
    this.#bridge = bridge;
  }

  /**
   * Scripted edits and checks, so the cross-language test can drive a real
   * session through a real check without a provider.
   *
   * `[fake:write=path:contents]` writes a file into the workspace.
   * `[fake:check=name]` asks the host for that check and reports the verdict.
   */
  async #script(
    sessionId: string,
    params: {
      prompt: string;
      workspaceRoot: string;
    },
  ): Promise<void> {
    for (const match of params.prompt.matchAll(
      /\[fake:write=([^:\]]+):([^\]]*)\]/g,
    )) {
      const target = resolve(params.workspaceRoot, match[1] ?? "");
      await mkdir(dirname(target), { recursive: true });
      await writeFile(
        target,
        `${(match[2] ?? "").replaceAll("\\n", "\n")}\n`,
        "utf8",
      );
      this.emit({
        sessionId,
        type: "tool",
        tool: "editor",
        status: "finished",
      });
    }
    for (const match of params.prompt.matchAll(/\[fake:check=([a-z]+)\]/g)) {
      const name = match[1] ?? "full";
      this.emit({
        sessionId,
        type: "tool",
        tool: "run_check",
        status: "started",
      });
      const result = await this.#bridge?.runCheck(sessionId, name as never);
      this.emit({
        sessionId,
        type: "tool",
        tool: "run_check",
        // `unavailable` arrives as JSON null when the host reports a result
        // it could produce; `=== undefined` alone called every clean check a
        // failure.
        status:
          result === undefined || result.unavailable != null
            ? "failed"
            : "finished",
        message:
          result === undefined
            ? "no host bridge"
            : `${result.check}:${result.ok ? "clean" : `${result.errors} error(s)`}` +
              (result.findings[0] ? ` :: ${result.findings[0].detail}` : ""),
      });
    }
  }

  constructor(redactor = new SecretRedactor()) {
    super();
    this.#redactor = redactor;
  }

  #requireSession(sessionId: string): FakeSession {
    const session = this.#sessions.get(sessionId);
    if (session === undefined) {
      throw new RpcFault(
        "SESSION_NOT_FOUND",
        `Session ${sessionId} was not found`,
      );
    }
    return session;
  }

  #result(session: FakeSession): SliceResult {
    return {
      sessionId: session.sessionId,
      state: session.state,
      ...(session.finishReason === undefined
        ? {}
        : { finishReason: session.finishReason }),
      ...(session.controlledStopReason === undefined
        ? {}
        : { controlledStopReason: session.controlledStopReason }),
      iterations: session.usage.requests,
      toolCallCount: 0,
      usage: copyUsage(session.usage),
      model: { ...session.model },
      ...(session.summary === undefined ? {} : { summary: session.summary }),
      ...(session.parentSessionId === undefined
        ? {}
        : { parentSessionId: session.parentSessionId }),
    };
  }

  async startSlice(params: StartSliceParams): Promise<SliceResult> {
    if (this.#shuttingDown)
      throw new RpcFault("SHUTTING_DOWN", "Runtime is shutting down");
    this.#redactor.addProvider(params.provider);
    const sessionId = params.sessionId ?? `fake-${randomUUID()}`;
    if (this.#sessions.has(sessionId)) {
      throw new RpcFault("SESSION_BUSY", `Session ${sessionId} already exists`);
    }
    const session: FakeSession = {
      sessionId,
      state: "running",
      usage: zeroUsage(),
      model: {
        providerId: params.provider.providerId,
        modelId: params.provider.modelId,
      },
      workspaceRoot: params.workspaceRoot,
    };
    this.#sessions.set(sessionId, session);
    this.emit({ sessionId, type: "state", status: "running" });
    this.emit({
      sessionId,
      type: "text",
      message: this.#redactor.redact(`Fake runtime accepted ${params.prompt}`),
    });
    await this.#script(sessionId, params);

    const requestedIterations = Number(
      params.prompt.match(/\[fake:iterations=(\d+)\]/)?.[1] ?? 1,
    );
    const maxIterations = params.limits?.maxIterations ?? 32;
    const executedIterations = Math.min(requestedIterations, maxIterations);
    session.usage.inputTokens += 11 * executedIterations;
    session.usage.outputTokens += 3 * executedIterations;
    session.usage.totalTokens =
      session.usage.inputTokens + session.usage.outputTokens;
    session.usage.requests += executedIterations;
    if (requestedIterations > maxIterations) {
      session.state = "failed";
      // Match the pinned SDK 0.0.72 contract: preserve its generic raw reason
      // while exposing Metis's independently-derived controlled stop code.
      session.finishReason = "error";
      session.controlledStopReason = "max_iterations";
      session.summary = `Agent runtime exceeded maxIterations (${maxIterations})`;
      this.emit({
        sessionId,
        type: "error",
        status: "max_iterations",
        message: session.summary,
      });
      return this.#result(session);
    }
    const requestedDuration = Number(
      params.prompt.match(/\[fake:duration=(\d+)\]/)?.[1] ?? 0,
    );
    if (
      params.limits?.timeoutMs !== undefined &&
      requestedDuration > params.limits.timeoutMs
    ) {
      session.state = "failed";
      session.finishReason = "error";
      session.summary = `Stopped at the wall-time limit (${params.limits.timeoutMs} ms)`;
      this.emit({
        sessionId,
        type: "error",
        status: "timeout",
        message: session.summary,
      });
      return this.#result(session);
    }
    if (params.prompt.includes("[fake:hold]")) return this.#result(session);
    if (params.prompt.includes("[fake:error]")) {
      session.state = "failed";
      session.finishReason = "error";
      session.summary = "Deterministic fake failure";
      this.emit({
        sessionId,
        type: "error",
        status: "failed",
        message: session.summary,
      });
      return this.#result(session);
    }
    session.state = "completed";
    session.finishReason = "completed";
    session.summary = "Deterministic fake completion";
    this.emit({ sessionId, type: "usage", usage: copyUsage(session.usage) });
    this.emit({ sessionId, type: "completed", status: "completed" });
    return this.#result(session);
  }

  async continueSlice(params: ContinueSliceParams): Promise<SliceResult> {
    const session = this.#requireSession(params.sessionId);
    if (session.state === "running") {
      throw new RpcFault(
        "SESSION_BUSY",
        `Session ${params.sessionId} is already running`,
      );
    }
    session.state = "running";
    delete session.finishReason;
    delete session.controlledStopReason;
    this.emit({
      sessionId: session.sessionId,
      type: "state",
      status: "running",
    });
    // Same session, same workspace: a continuation edits and checks exactly
    // where the first round left off.
    await this.#script(session.sessionId, {
      prompt: params.prompt,
      workspaceRoot: session.workspaceRoot,
    });
    session.usage.inputTokens += 7;
    session.usage.outputTokens += 2;
    session.usage.totalTokens =
      session.usage.inputTokens + session.usage.outputTokens;
    session.usage.requests += 1;
    session.state = params.prompt.includes("[fake:error]")
      ? "failed"
      : "completed";
    session.finishReason = session.state === "failed" ? "error" : "completed";
    session.summary =
      session.state === "failed"
        ? "Deterministic fake failure"
        : "Deterministic fake continuation";
    this.emit({
      sessionId: session.sessionId,
      type: session.state === "failed" ? "error" : "completed",
      status: session.state,
      message: this.#redactor.redact(session.summary),
      usage: copyUsage(session.usage),
    });
    return this.#result(session);
  }

  async abortSlice(params: AbortSliceParams): Promise<SliceResult> {
    const session = this.#requireSession(params.sessionId);
    session.state = "aborted";
    session.finishReason = "aborted";
    delete session.controlledStopReason;
    session.summary = params.reason ?? "Aborted by Metis";
    this.emit({
      sessionId: session.sessionId,
      type: "ended",
      status: "aborted",
      message: this.#redactor.redact(session.summary),
    });
    return this.#result(session);
  }

  async restoreSlice(params: RestoreSliceParams): Promise<SliceResult> {
    const session = this.#requireSession(params.sessionId);
    if (
      params.provider !== undefined &&
      (params.provider.providerId !== session.model.providerId ||
        params.provider.modelId !== session.model.modelId)
    ) {
      throw new RpcFault(
        "POLICY_DENIED",
        "Use restartWithModel to change a session model",
      );
    }
    if (params.provider !== undefined)
      this.#redactor.addProvider(params.provider);
    if (session.state === "running") {
      session.state = "idle";
      delete session.finishReason;
      delete session.controlledStopReason;
    }
    this.emit({
      sessionId: session.sessionId,
      type: "state",
      status: session.state,
    });
    return this.#result(session);
  }

  async restartWithModel(params: RestartWithModelParams): Promise<SliceResult> {
    const parent = this.#requireSession(params.sessionId);
    const newSessionId = params.newSessionId ?? `fake-${randomUUID()}`;
    if (this.#sessions.has(newSessionId)) {
      throw new RpcFault(
        "SESSION_BUSY",
        `Session ${newSessionId} already exists`,
      );
    }
    const result = await this.startSlice({
      sessionId: newSessionId,
      prompt: params.prompt,
      workspaceRoot: parent.workspaceRoot,
      provider: params.provider,
      ...(params.limits === undefined ? {} : { limits: params.limits }),
    });
    const session = this.#requireSession(newSessionId);
    session.parentSessionId = parent.sessionId;
    return { ...result, parentSessionId: parent.sessionId };
  }

  async getUsage(params: SessionParams): Promise<Usage> {
    return copyUsage(this.#requireSession(params.sessionId).usage);
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
      const session = this.#sessions.get(current);
      if (session === undefined) break;
      const parent = session.parentSessionId;
      if (this.#sessions.delete(current)) deletedSessionIds.push(current);
      current = parent;
    }
    return {
      sessionId: params.sessionId,
      deleted: deletedSessionIds.includes(params.sessionId),
      deletedSessionIds,
    };
  }

  async shutdown(): Promise<void> {
    this.#shuttingDown = true;
    for (const session of this.#sessions.values()) {
      if (session.state === "running") session.state = "aborted";
    }
  }
}
