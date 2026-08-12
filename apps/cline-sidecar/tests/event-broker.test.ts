import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { EventBroker } from "../src/event-broker.js";
import type { SanitizedSessionEvent } from "../src/protocol.js";

test("replay survives broker restart with monotonic cursors and bounded overflow", async () => {
  const dataDirectory = await mkdtemp(join(tmpdir(), "metis-events-"));
  try {
    const writer = new EventBroker(dataDirectory, 3);
    for (let index = 1; index <= 5; index += 1) {
      await writer.append({ sessionId: "session-1", type: "progress", message: `event-${index}` });
    }

    const reader = new EventBroker(dataDirectory, 3);
    const replay: SanitizedSessionEvent[] = [];
    const subscription = await reader.subscribe("session-1", 0, (event) => replay.push(event));
    assert.equal(subscription.cursor, 5);
    assert.equal(subscription.oldestCursor, 3);
    assert.equal(subscription.overflowed, true);
    assert.deepEqual(
      replay.map((event) => event.cursor),
      [2, 3, 4, 5],
    );
    assert.equal(replay[0]?.type, "replay_overflow");

    await reader.append({ sessionId: "session-1", type: "progress", message: "event-6" });
    assert.equal(replay.at(-1)?.cursor, 6);
  } finally {
    await rm(dataDirectory, { recursive: true, force: true });
  }
});

test("replay preserves a bounded tool denial code without sensitive details", async () => {
  const dataDirectory = await mkdtemp(join(tmpdir(), "metis-denial-events-"));
  try {
    const writer = new EventBroker(dataDirectory, 3);
    await writer.append({
      sessionId: "session-denied",
      type: "tool",
      tool: "read_files",
      status: "denied",
      reasonCode: "outside_workspace",
    });

    const replay: SanitizedSessionEvent[] = [];
    const reader = new EventBroker(dataDirectory, 3);
    await reader.subscribe("session-denied", 0, (event) => replay.push(event));
    assert.equal(replay[0]?.reasonCode, "outside_workspace");
    assert.equal(JSON.stringify(replay).includes("path"), false);
    assert.equal(JSON.stringify(replay).includes("reason\""), false);
  } finally {
    await rm(dataDirectory, { recursive: true, force: true });
  }
});

test("replay drops a denial event with an unrecognized reason code", async () => {
  const dataDirectory = await mkdtemp(join(tmpdir(), "metis-invalid-denial-event-"));
  try {
    const eventDirectory = join(dataDirectory, "metis-events-v1");
    await mkdir(eventDirectory);
    await writeFile(
      join(eventDirectory, "session-invalid.jsonl"),
      `${JSON.stringify({
        sessionId: "session-invalid",
        cursor: 1,
        type: "tool",
        tool: "read_files",
        status: "denied",
        reasonCode: "path=/private/secret.txt",
        occurredAt: new Date().toISOString(),
      })}\n`,
    );

    const replay: SanitizedSessionEvent[] = [];
    const reader = new EventBroker(dataDirectory, 3);
    const subscription = await reader.subscribe("session-invalid", 0, (event) => replay.push(event));
    assert.equal(subscription.replayed, 0);
    assert.deepEqual(replay, []);
  } finally {
    await rm(dataDirectory, { recursive: true, force: true });
  }
});
