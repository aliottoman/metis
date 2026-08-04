import assert from "node:assert/strict";
import test from "node:test";

import {
  latestRunId,
  mergeAssistantReasoning,
  mergeAssistantRunEvent,
  messageBelongsToRun,
} from "../lib/run-history.ts";
import type { ChatMessage } from "../lib/types.ts";

test("finds the most recent persisted run across conversation messages", () => {
  const messages: ChatMessage[] = [
    { id: "one", role: "user", content: "one", run_id: "run_1" },
    { id: "two", role: "assistant", content: "done", run_id: "run_1" },
    { id: "three", role: "user", content: "two", run_id: "run_2" },
  ];
  assert.equal(latestRunId(messages), "run_2");
});

test("SSE replay does not duplicate or overwrite a hydrated final assistant message", () => {
  const messages: ChatMessage[] = [
    { id: "msg_saved", role: "assistant", content: "Canonical final response", run_id: "run_7" },
  ];
  const afterDelta = mergeAssistantRunEvent(
    messages,
    { run_id: "run_7", type: "message.delta" },
    "Canonical ",
  );
  const afterCompleted = mergeAssistantRunEvent(
    afterDelta,
    { run_id: "run_7", type: "run.completed" },
    "Canonical final response",
  );

  assert.equal(afterCompleted.length, 1);
  assert.equal(afterCompleted[0]?.id, "msg_saved");
  assert.equal(afterCompleted[0]?.content, "Canonical final response");
  assert.equal(messageBelongsToRun(afterCompleted[0]!, "run_7"), true);
});

test("live response events create and complete one run-linked assistant message", () => {
  const streaming = mergeAssistantRunEvent(
    [],
    { run_id: "run_live", type: "message.delta" },
    "Working",
  );
  const completed = mergeAssistantRunEvent(
    streaming,
    { run_id: "run_live", type: "run.completed" },
    "Finished",
  );

  assert.equal(completed.length, 1);
  assert.deepEqual(completed[0], {
    id: "assistant-run_live",
    run_id: "run_live",
    role: "assistant",
    content: "Finished",
    streaming: false,
  });
});

test("thinking accumulates on its own field and never joins the answer text", () => {
  const thinking = mergeAssistantReasoning([], "run_think", "Considering the ");
  const more = mergeAssistantReasoning(thinking, "run_think", "retrieved passages.");
  const answered = mergeAssistantRunEvent(
    more,
    { run_id: "run_think", type: "message.delta" },
    "Here is the answer.",
  );

  assert.equal(answered.length, 1);
  assert.equal(answered[0]?.reasoning, "Considering the retrieved passages.");
  assert.equal(answered[0]?.content, "Here is the answer.");
});

test("an empty thinking delta leaves the message untouched", () => {
  const messages: ChatMessage[] = [
    { id: "assistant-run_x", role: "assistant", content: "", run_id: "run_x", reasoning: "kept" },
  ];
  const merged = mergeAssistantReasoning(messages, "run_x", "");
  assert.equal(merged[0]?.reasoning, "kept");
});
