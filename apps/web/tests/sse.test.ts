import assert from "node:assert/strict";
import test from "node:test";

import { appendRunActivityEvent, normalizeRunEvent, parseSseBuffer } from "../lib/sse.ts";

test("parses CRLF frames, comments, named events, and multiline data", () => {
  const input = ": heartbeat\r\nid: 7\r\nevent: model.delta\r\ndata: {\"delta\":\"hello\"}\r\ndata: world\r\n\r\npartial";
  const result = parseSseBuffer(input);

  assert.equal(result.frames.length, 1);
  assert.deepEqual(result.frames[0], {
    id: "7",
    event: "model.delta",
    data: "{\"delta\":\"hello\"}\nworld",
    retry: undefined,
  });
  assert.equal(result.rest, "partial");
});

test("keeps an incomplete frame for the next network chunk", () => {
  const first = parseSseBuffer("id: 1\ndata: {\"type\":\"run.");
  assert.equal(first.frames.length, 0);

  const second = parseSseBuffer(`${first.rest}completed\"}\n\n`);
  assert.equal(second.frames.length, 1);
  assert.equal(second.frames[0]?.data, '{"type":"run.completed"}');
});

test("normalizes alternate event envelopes without losing payload", () => {
  const event = normalizeRunEvent(
    { seq: 4, event: "tool.started", runId: "run-1", payload: { tool: "diagram" } },
    {},
    "fallback",
    1,
  );

  assert.equal(event.sequence, 4);
  assert.equal(event.type, "tool.started");
  assert.equal(event.run_id, "run-1");
  assert.deepEqual(event.payload, { tool: "diagram" });
});

test("streamed tokens do not displace useful task activity", () => {
  const approval = normalizeRunEvent(
    { id: "approval", sequence: 1, type: "approval.required", payload: {} },
    {}, "run-1", 1,
  );
  let events = appendRunActivityEvent([], approval);
  for (let sequence = 2; sequence < 1002; sequence += 1) {
    events = appendRunActivityEvent(events, normalizeRunEvent(
      { id: `token-${sequence}`, sequence, type: "message.delta", payload: { delta: "x" } },
      {}, "run-1", sequence,
    ));
  }
  assert.equal(events.length, 1);
  assert.equal(events[0]?.id, "approval");
  assert.equal(appendRunActivityEvent(events, normalizeRunEvent(
    { id: "done", sequence: 1002, type: "run.completed", payload: {} },
    {}, "run-1", 1002,
  )).length, 2);
});
