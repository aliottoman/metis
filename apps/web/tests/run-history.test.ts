import assert from "node:assert/strict";
import test from "node:test";

import {
  latestRunId,
  mergeAssistantReasoning,
  mergeAssistantRunEvent,
  mergeHydratedConversationMessages,
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

test("a conversation snapshot keeps text that streamed while it was loading", () => {
  const hydrated: ChatMessage[] = [{ id: "user_saved", role: "user", content: "Question", run_id: "run_live" }];
  const live: ChatMessage[] = [
    { id: "assistant-old", role: "assistant", content: "Wrong conversation", run_id: "run_old", streaming: true },
    { id: "assistant-run_live", role: "assistant", content: "The answer so far", run_id: "run_live", streaming: true },
  ];

  const merged = mergeHydratedConversationMessages(hydrated, live, "run_live");
  assert.deepEqual(merged.map((message) => message.id), ["user_saved", "assistant-run_live"]);
  assert.equal(merged[1]?.content, "The answer so far");
});

test("a persisted answer replaces the temporary streamed copy on hydration", () => {
  const hydrated: ChatMessage[] = [{ id: "assistant_saved", role: "assistant", content: "Complete answer", run_id: "run_live" }];
  const live: ChatMessage[] = [{ id: "assistant-run_live", role: "assistant", content: "Complete ans", run_id: "run_live", streaming: true }];
  assert.deepEqual(mergeHydratedConversationMessages(hydrated, live, "run_live"), hydrated);
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

test("stopping a run settles its placeholder and keeps any partial reply", () => {
  const partial = mergeAssistantRunEvent(
    [{ id: "assistant-run_stop", role: "assistant", run_id: "run_stop", content: "A partial answer", streaming: true }],
    { run_id: "run_stop", type: "run.cancelled" },
  );
  assert.equal(partial[0]?.content, "A partial answer");
  assert.equal(partial[0]?.streaming, false);

  const empty = mergeAssistantRunEvent(
    [{ id: "assistant-run_empty", role: "assistant", run_id: "run_empty", content: "", streaming: true }],
    { run_id: "run_empty", type: "run.cancelled" },
  );
  assert.equal(empty[0]?.content, "Response stopped.");
  assert.equal(empty[0]?.streaming, false);
});

test("a failed run keeps the partial reply available for review", () => {
  const messages = mergeAssistantRunEvent(
    [{ id: "assistant-run_failed", role: "assistant", run_id: "run_failed", content: "Useful partial text", streaming: true }],
    { run_id: "run_failed", type: "run.failed" },
  );
  assert.equal(messages[0]?.content, "Useful partial text");
  assert.equal(messages[0]?.failed, true);
  assert.equal(messages[0]?.streaming, false);
});

test("failed search or project steps do not fail the assistant response", () => {
  const pending: ChatMessage[] = [
    { id: "assistant-run_work", role: "assistant", run_id: "run_work", content: "", streaming: true },
  ];
  const afterSearch = mergeAssistantRunEvent(pending, { run_id: "run_work", type: "web_search_failed" }, "Search unavailable");
  const afterProject = mergeAssistantRunEvent(afterSearch, { run_id: "run_work", type: "project.coding_failed" }, "Check failed");
  const answered = mergeAssistantRunEvent(afterProject, { run_id: "run_work", type: "message.delta" }, "Here is what I found.");

  assert.equal(afterSearch[0]?.failed, undefined);
  assert.equal(afterProject[0]?.failed, undefined);
  assert.equal(answered[0]?.content, "Here is what I found.");
  assert.equal(answered[0]?.streaming, true);
});
