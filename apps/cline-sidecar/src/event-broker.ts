import { mkdir, readFile, rename, unlink, writeFile } from "node:fs/promises";
import { join } from "node:path";

import { TOOL_DENIAL_REASON_CODES } from "./protocol.js";
import type { SanitizedSessionEvent, SubscriptionResult } from "./protocol.js";
import type { RuntimeEvent } from "./sanitize.js";

type EventListener = (event: SanitizedSessionEvent) => void;

interface SessionBuffer {
  events: SanitizedSessionEvent[];
  nextCursor: number;
}

const TOOL_DENIAL_REASON_CODE_SET = new Set<string>(TOOL_DENIAL_REASON_CODES);

export interface ReplaySubscription extends SubscriptionResult {}

function isEvent(
  value: unknown,
  sessionId: string,
): value is SanitizedSessionEvent {
  if (value === null || typeof value !== "object" || Array.isArray(value))
    return false;
  const event = value as Partial<SanitizedSessionEvent>;
  return (
    event.sessionId === sessionId &&
    Number.isSafeInteger(event.cursor) &&
    (event.cursor ?? 0) > 0 &&
    typeof event.type === "string" &&
    (event.reasonCode === undefined ||
      (event.status === "denied" &&
        TOOL_DENIAL_REASON_CODE_SET.has(event.reasonCode))) &&
    typeof event.occurredAt === "string"
  );
}

/** Bounded, sanitized event replay with monotonic cursors persisted under Metis storage. */
export class EventBroker {
  readonly #eventDirectory: string;
  readonly #maxEvents: number;
  readonly #buffers = new Map<string, SessionBuffer>();
  readonly #listeners = new Map<string, Set<EventListener>>();
  readonly #chains = new Map<string, Promise<void>>();

  constructor(dataDirectory: string, maxEvents = 256) {
    if (!Number.isInteger(maxEvents) || maxEvents < 2 || maxEvents > 4096) {
      throw new Error("maxEvents must be an integer from 2 through 4096");
    }
    this.#eventDirectory = join(dataDirectory, "metis-events-v1");
    this.#maxEvents = maxEvents;
  }

  #filePath(sessionId: string): string {
    return join(this.#eventDirectory, `${sessionId}.jsonl`);
  }

  async #load(sessionId: string): Promise<SessionBuffer> {
    const existing = this.#buffers.get(sessionId);
    if (existing !== undefined) return existing;
    let events: SanitizedSessionEvent[] = [];
    try {
      const contents = await readFile(this.#filePath(sessionId), "utf8");
      events = contents
        .split("\n")
        .filter(Boolean)
        .flatMap((line) => {
          try {
            const parsed: unknown = JSON.parse(line);
            return isEvent(parsed, sessionId) ? [parsed] : [];
          } catch {
            return [];
          }
        })
        .sort((left, right) => left.cursor - right.cursor)
        .slice(-this.#maxEvents);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    }
    const buffer = {
      events,
      nextCursor: (events.at(-1)?.cursor ?? 0) + 1,
    };
    this.#buffers.set(sessionId, buffer);
    return buffer;
  }

  async #persist(
    sessionId: string,
    events: readonly SanitizedSessionEvent[],
  ): Promise<void> {
    await mkdir(this.#eventDirectory, { recursive: true, mode: 0o700 });
    const destination = this.#filePath(sessionId);
    const temporary = `${destination}.${process.pid}.tmp`;
    const body = events.map((event) => JSON.stringify(event)).join("\n");
    await writeFile(temporary, body.length === 0 ? "" : `${body}\n`, {
      mode: 0o600,
    });
    await rename(temporary, destination);
  }

  async #exclusive<T>(
    sessionId: string,
    operation: () => Promise<T>,
  ): Promise<T> {
    const previous = this.#chains.get(sessionId) ?? Promise.resolve();
    let release: (() => void) | undefined;
    const current = new Promise<void>((resolve) => {
      release = resolve;
    });
    const queued = previous.then(() => current);
    this.#chains.set(sessionId, queued);
    await previous;
    try {
      return await operation();
    } finally {
      release?.();
      if (this.#chains.get(sessionId) === queued)
        this.#chains.delete(sessionId);
    }
  }

  async append(draft: RuntimeEvent): Promise<SanitizedSessionEvent> {
    const event = await this.#exclusive(draft.sessionId, async () => {
      const buffer = await this.#load(draft.sessionId);
      const persisted: SanitizedSessionEvent = {
        ...draft,
        cursor: buffer.nextCursor,
        occurredAt: new Date().toISOString(),
      };
      buffer.nextCursor += 1;
      buffer.events.push(persisted);
      if (buffer.events.length > this.#maxEvents)
        buffer.events.splice(0, buffer.events.length - this.#maxEvents);
      await this.#persist(draft.sessionId, buffer.events);
      return persisted;
    });
    for (const listener of this.#listeners.get(draft.sessionId) ?? [])
      listener(event);
    return event;
  }

  async subscribe(
    sessionId: string,
    afterCursor: number,
    listener: EventListener,
  ): Promise<ReplaySubscription> {
    return this.#exclusive(sessionId, async () => {
      const listeners =
        this.#listeners.get(sessionId) ?? new Set<EventListener>();
      listeners.add(listener);
      this.#listeners.set(sessionId, listeners);
      const buffer = await this.#load(sessionId);
      const cursor = buffer.events.at(-1)?.cursor ?? 0;
      const oldestCursor = buffer.events.at(0)?.cursor ?? cursor;
      const overflowed =
        buffer.events.length > 0 && afterCursor < oldestCursor - 1;
      const events = buffer.events.filter(
        (event) => event.cursor > afterCursor,
      );
      if (overflowed) {
        events.unshift({
          sessionId,
          cursor: Math.max(0, oldestCursor - 1),
          type: "replay_overflow",
          status: "history_truncated",
          message: `Events before cursor ${oldestCursor} are no longer retained`,
          occurredAt: new Date().toISOString(),
        });
      }
      for (const event of events) listener(event);
      return {
        sessionId,
        cursor,
        oldestCursor,
        replayed: events.length,
        overflowed,
      };
    });
  }

  unsubscribe(sessionId: string, listener: EventListener): void {
    const listeners = this.#listeners.get(sessionId);
    listeners?.delete(listener);
    if (listeners?.size === 0) this.#listeners.delete(sessionId);
  }

  async cursor(sessionId: string): Promise<number> {
    return this.#exclusive(
      sessionId,
      async () => (await this.#load(sessionId)).nextCursor - 1,
    );
  }

  async delete(sessionId: string): Promise<void> {
    await this.#exclusive(sessionId, async () => {
      this.#buffers.delete(sessionId);
      this.#listeners.delete(sessionId);
      try {
        await unlink(this.#filePath(sessionId));
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
      }
    });
  }
}
