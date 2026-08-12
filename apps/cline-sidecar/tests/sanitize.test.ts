import assert from "node:assert/strict";
import { test } from "node:test";

import {
  IterationTracker,
  NO_SDK_DIAGNOSTIC,
  SecretRedactor,
  sanitizeCoreEvent,
} from "../src/sanitize.js";

test("raw chunks and structured content fragments are not persisted as progress events", () => {
  const redactor = new SecretRedactor();
  assert.equal(
    sanitizeCoreEvent(
      {
        type: "chunk",
        payload: {
          sessionId: "session-1",
          stream: "agent",
          chunk: JSON.stringify({
            type: "content_start",
            input: { new_text: "large model-generated source" },
          }),
        },
      } as never,
      redactor,
    ),
    undefined,
  );
  for (const type of ["content_start", "content_update", "content_end"]) {
    assert.equal(
      sanitizeCoreEvent(
        {
          type: "agent_event",
          payload: { sessionId: "session-1", event: { type } },
        } as never,
        redactor,
      ),
      undefined,
    );
  }
  assert.deepEqual(
    sanitizeCoreEvent(
      {
        type: "chunk",
        payload: {
          sessionId: "session-1",
          stream: "stderr",
          chunk: "bounded runtime diagnostic",
        },
      } as never,
      redactor,
    ),
    {
      sessionId: "session-1",
      type: "text",
      status: "stderr",
      message: "bounded runtime diagnostic",
    },
  );
});

test("iteration, usage, tool-hook, state, and completion evidence remains observable", () => {
  const redactor = new SecretRedactor();
  const iteration = sanitizeCoreEvent(
    {
      type: "agent_event",
      payload: {
        sessionId: "session-1",
        event: { type: "iteration_end", iteration: 3, hadToolCalls: true },
      },
    } as never,
    redactor,
  );
  assert.deepEqual(iteration, {
    sessionId: "session-1",
    type: "progress",
    status: "iteration_end",
  });

  const usage = sanitizeCoreEvent(
    {
      type: "agent_event",
      payload: {
        sessionId: "session-1",
        event: { type: "usage", inputTokens: 120, outputTokens: 30 },
      },
    } as never,
    redactor,
  );
  assert.deepEqual(usage, {
    sessionId: "session-1",
    type: "usage",
    status: "usage",
    // Cline reports session totals; the snapshot now says so explicitly so a
    // consumer cannot sum snapshots into a bill that was never charged.
    usageScope: "cumulative",
    usage: {
      inputTokens: 120,
      outputTokens: 30,
      totalTokens: 150,
      requests: 0,
    },
  });

  assert.deepEqual(
    sanitizeCoreEvent(
      {
        type: "hook",
        payload: {
          sessionId: "session-1",
          hookEventName: "finished",
          toolName: "editor",
        },
      } as never,
      redactor,
    ),
    {
      sessionId: "session-1",
      type: "tool",
      status: "finished",
      tool: "editor",
    },
  );
  assert.deepEqual(
    sanitizeCoreEvent(
      {
        type: "status",
        payload: { sessionId: "session-1", status: "running" },
      } as never,
      redactor,
    ),
    { sessionId: "session-1", type: "state", status: "running" },
  );
  assert.deepEqual(
    sanitizeCoreEvent(
      {
        type: "ended",
        payload: { sessionId: "session-1", reason: "completed" },
      } as never,
      redactor,
    ),
    { sessionId: "session-1", type: "completed", status: "completed" },
  );
});

// ── Failed tool calls must arrive with their reason ────────────────────────
//
// Nine `editor:failed` events were recorded across a live Logivity session
// with empty message fields, which made a failed edit indistinguishable from
// a refused one and left an expensive investigation unable to say why the
// model wrote nothing.

test("a failed editor call carries the SDK's own error text", () => {
  const event = sanitizeCoreEvent(
    {
      type: "agent_event",
      payload: {
        sessionId: "session-1",
        event: {
          type: "tool_call_failed",
          toolCall: { toolName: "editor", path: "README.md" },
          error: {
            name: "EditApplyError",
            message: "old_text did not match the file at README.md",
          },
        },
      },
    } as never,
    new SecretRedactor(),
  );

  assert.equal(event?.type, "tool");
  assert.equal(event?.tool, "editor");
  assert.equal(event?.path, "README.md");
  assert.equal(event?.failureClass, "EditApplyError");
  assert.equal(event?.message, "old_text did not match the file at README.md");
});

test("a failed editor call with no SDK text says so explicitly", () => {
  const event = sanitizeCoreEvent(
    {
      type: "agent_event",
      payload: {
        sessionId: "session-1",
        event: { type: "tool_call_failed", toolCall: { toolName: "editor" } },
      },
    } as never,
    new SecretRedactor(),
  );

  // Never blank: blank is what "we never looked" also looks like.
  assert.equal(event?.message, NO_SDK_DIAGNOSTIC);
  assert.equal(event?.tool, "editor");
});

test("failure is recognised from a result flag, not only from the event name", () => {
  const event = sanitizeCoreEvent(
    {
      type: "agent_event",
      payload: {
        sessionId: "session-1",
        event: {
          type: "tool_call_finished",
          toolCall: {
            toolName: "editor",
            isError: true,
            result: { error: "disk full" },
          },
        },
      },
    } as never,
    new SecretRedactor(),
  );

  assert.equal(event?.message, "disk full");
});

test("a successful tool call gains no invented failure text", () => {
  const event = sanitizeCoreEvent(
    {
      type: "agent_event",
      payload: {
        sessionId: "session-1",
        event: {
          type: "tool_call_finished",
          toolCall: { toolName: "read_files" },
        },
      },
    } as never,
    new SecretRedactor(),
  );

  assert.equal(event?.message, undefined);
  assert.equal(event?.failureClass, undefined);
});

test("secrets inside failure text are redacted and the text stays bounded", () => {
  const redactor = new SecretRedactor();
  redactor.add("super-secret-token-value");
  const event = sanitizeCoreEvent(
    {
      type: "agent_event",
      payload: {
        sessionId: "session-1",
        event: {
          type: "tool_call_failed",
          toolCall: { toolName: "editor" },
          error: {
            message:
              "auth failed with super-secret-token-value and Bearer abcdefghijklmnop " +
              "x".repeat(9000),
          },
        },
      },
    } as never,
    redactor,
  );

  assert.ok(event?.message);
  assert.ok(!event.message.includes("super-secret-token-value"));
  assert.ok(event.message.includes("[REDACTED]"));
  assert.ok(!event.message.includes("Bearer abcdefghijklmnop"));
  assert.ok(Buffer.byteLength(event.message, "utf8") <= 4200);
});

test("a session that ends in error no longer records only the word error", () => {
  const event = sanitizeCoreEvent(
    {
      type: "ended",
      payload: {
        sessionId: "session-1",
        reason: "error",
        error: { message: "model 'kimi-k2.7-code' is temporarily overloaded" },
      },
    } as never,
    new SecretRedactor(),
  );

  assert.equal(event?.type, "ended");
  assert.equal(
    event?.message,
    "model 'kimi-k2.7-code' is temporarily overloaded",
  );
});

// ── Usage has to answer "is context compounding" ───────────────────────────

test("usage carries cache counters, cost and requests when the SDK reports them", () => {
  const event = sanitizeCoreEvent(
    {
      type: "agent_event",
      payload: {
        sessionId: "session-1",
        event: {
          type: "usage",
          usage: {
            inputTokens: 49_307,
            outputTokens: 802,
            totalTokens: 50_109,
            cacheReadTokens: 40_000,
            cacheWriteTokens: 1_200,
            totalCost: 0.42,
            requests: 3,
          },
        },
      },
    } as never,
    new SecretRedactor(),
  );

  assert.equal(event?.usageScope, "cumulative");
  assert.deepEqual(event?.usage, {
    inputTokens: 49_307,
    outputTokens: 802,
    totalTokens: 50_109,
    requests: 3,
    cacheReadTokens: 40_000,
    cacheWriteTokens: 1_200,
    costUsd: 0.42,
  });
});

test("absent cache counters are omitted rather than reported as zero", () => {
  const event = sanitizeCoreEvent(
    {
      type: "agent_event",
      payload: {
        sessionId: "session-1",
        event: { type: "usage", usage: { inputTokens: 10, outputTokens: 2 } },
      },
    } as never,
    new SecretRedactor(),
  );

  assert.equal(event?.usage?.cacheReadTokens, undefined);
  assert.equal(event?.usage?.cacheWriteTokens, undefined);
  assert.equal(event?.usage?.costUsd, undefined);
  // The total is never allowed below its own parts.
  assert.equal(event?.usage?.totalTokens, 12);
});

test("iterations are counted and stamped onto the events between them", () => {
  const tracker = new IterationTracker();
  const redactor = new SecretRedactor();
  const emit = (type: string, usage?: Record<string, unknown>) =>
    sanitizeCoreEvent(
      {
        type: "agent_event",
        payload: {
          sessionId: "session-1",
          event: { type, ...(usage ? { usage } : {}) },
        },
      } as never,
      redactor,
      tracker,
    );

  assert.equal(emit("iteration_start")?.iteration, 1);
  assert.equal(
    emit("usage", { inputTokens: 100, outputTokens: 10 })?.iteration,
    1,
  );
  assert.equal(emit("iteration_start")?.iteration, 2);
  assert.equal(
    emit("usage", { inputTokens: 400, outputTokens: 40 })?.iteration,
    2,
  );
});

test("a cumulative snapshot yields a non-negative delta, and a new session restarts it", () => {
  const tracker = new IterationTracker();
  const first = tracker.delta("session-1", {
    inputTokens: 100,
    outputTokens: 10,
    totalTokens: 110,
    requests: 1,
  });
  // No prior snapshot: the first reading is its own increase, not zero.
  assert.equal(first.totalTokens, 110);

  const second = tracker.delta("session-1", {
    inputTokens: 400,
    outputTokens: 40,
    totalTokens: 440,
    requests: 2,
  });
  assert.equal(second.inputTokens, 300);
  assert.equal(second.totalTokens, 330);

  // A restart-with-model opens a new session id; it must not difference
  // against its ancestor and produce a negative or double-counted step.
  const restarted = tracker.delta("session-2", {
    inputTokens: 50,
    outputTokens: 5,
    totalTokens: 55,
    requests: 1,
  });
  assert.equal(restarted.totalTokens, 55);

  // A counter that goes backwards clamps at zero rather than going negative.
  const rewound = tracker.delta("session-1", {
    inputTokens: 10,
    outputTokens: 1,
    totalTokens: 11,
    requests: 1,
  });
  assert.equal(rewound.totalTokens, 0);
  assert.equal(rewound.inputTokens, 0);
});
