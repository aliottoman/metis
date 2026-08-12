import { randomUUID } from "node:crypto";

import {
  hostCallRequest,
  type HostCallRequest,
  type HostCheck,
  type HostCheckResult,
} from "./protocol.js";

/**
 * The sidecar's one outbound channel to Metis.
 *
 * Everything else on this pipe flows host → sidecar. `run_check` is the
 * exception: the model asks for a named check, and Metis is the only party
 * that may run one — it owns the argv, the pinned networkless container, and
 * the mirror. So the request travels back up the same stdio pipe and this
 * class matches the answer to the caller.
 *
 * Deliberately narrow. There is exactly one call type, its parameter is one
 * name from a closed enum, and a call that is not answered inside its timeout
 * resolves to an explicit "unavailable" result rather than hanging a turn.
 */
export class HostBridge {
  readonly #write: (frame: HostCallRequest) => void;
  readonly #timeoutMs: number;
  readonly #pending = new Map<
    string,
    { resolve: (value: HostCheckResult) => void; timer: NodeJS.Timeout }
  >();
  #closed = false;

  constructor(write: (frame: HostCallRequest) => void, timeoutMs = 300_000) {
    this.#write = write;
    this.#timeoutMs = timeoutMs;
  }

  get pendingCount(): number {
    return this.#pending.size;
  }

  /**
   * Ask Metis to run one named check and wait for its findings.
   *
   * Never rejects: a check that cannot be run is a fact the model should be
   * told, not an exception that ends its turn.
   */
  async runCheck(
    sessionId: string,
    check: HostCheck,
  ): Promise<HostCheckResult> {
    if (this.#closed)
      return unavailable(check, "the host connection is closed");
    const id = `hc_${randomUUID()}`;
    return new Promise<HostCheckResult>((resolve) => {
      const timer = setTimeout(() => {
        this.#pending.delete(id);
        resolve(
          unavailable(check, "the host did not answer within its time limit"),
        );
      }, this.#timeoutMs);
      // Node keeps the process alive for a pending timer; a check must never
      // be the reason the sidecar refuses to exit.
      timer.unref?.();
      this.#pending.set(id, { resolve, timer });
      try {
        this.#write(hostCallRequest(id, sessionId, check));
      } catch (error) {
        clearTimeout(timer);
        this.#pending.delete(id);
        resolve(
          unavailable(check, `the request could not be sent: ${String(error)}`),
        );
      }
    });
  }

  /** Deliver a host answer. Returns false when the id is not ours. */
  settle(
    id: string,
    result: HostCheckResult | undefined,
    error?: string,
  ): boolean {
    const waiting = this.#pending.get(id);
    if (waiting === undefined) return false;
    this.#pending.delete(id);
    clearTimeout(waiting.timer);
    waiting.resolve(
      result ?? unavailable("full", error ?? "the host reported no result"),
    );
    return true;
  }

  /**
   * Fail every outstanding call. Called when the pipe dies: a model waiting on
   * a check that can no longer be answered has to be told so, or the turn
   * blocks until its own iteration budget runs out.
   */
  close(reason = "the host connection ended"): void {
    this.#closed = true;
    for (const [id, waiting] of this.#pending) {
      clearTimeout(waiting.timer);
      waiting.resolve(unavailable("full", reason));
      this.#pending.delete(id);
    }
  }
}

function unavailable(check: HostCheck, reason: string): HostCheckResult {
  return {
    check,
    ok: false,
    unavailable: reason,
    errors: 0,
    warnings: 0,
    findings: [],
  };
}
