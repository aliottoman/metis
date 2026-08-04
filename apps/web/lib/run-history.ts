import type { ChatMessage, RunEventV1 } from "./types";

export function latestRunId(messages: readonly ChatMessage[]): string | undefined {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const runId = messages[index]?.run_id;
    if (runId) return runId;
  }
  return undefined;
}

export function messageBelongsToRun(message: ChatMessage, runId: string | null): boolean {
  if (!runId || message.role !== "assistant") return false;
  return message.run_id === runId || message.id === `assistant-${runId}`;
}

/**
 * Append a thinking delta to the run's assistant message. Reasoning arrives on
 * its own event type and is kept in its own field, so it can be read alongside
 * the answer but never becomes part of it.
 */
export function mergeAssistantReasoning(
  messages: readonly ChatMessage[],
  runId: string,
  delta: string,
): ChatMessage[] {
  if (!delta) return [...messages];
  const existingIndex = messages.findIndex((message) => messageBelongsToRun(message, runId));
  if (existingIndex < 0) {
    return [...messages, {
      id: `assistant-${runId}`,
      run_id: runId,
      role: "assistant",
      content: "",
      reasoning: delta,
      streaming: true,
    }];
  }
  return messages.map((message, index) => index === existingIndex
    ? { ...message, run_id: runId, reasoning: `${message.reasoning ?? ""}${delta}` }
    : message);
}

/**
 * Merge response events without duplicating an assistant message that was
 * already hydrated from the durable conversation store. SSE replay is still
 * used for the timeline and artifacts; persisted final text remains canonical.
 */
export function mergeAssistantRunEvent(
  messages: readonly ChatMessage[],
  event: Pick<RunEventV1, "run_id" | "type">,
  text?: string,
): ChatMessage[] {
  const type = event.type.toLowerCase();
  const existingIndex = messages.findIndex((message) => messageBelongsToRun(message, event.run_id));
  const existing = existingIndex >= 0 ? messages[existingIndex] : undefined;
  const isHydratedFinal = Boolean(existing && !existing.streaming && existing.content);

  if (type.includes("delta") && text) {
    if (isHydratedFinal) return [...messages];
    if (!existing) {
      return [...messages, {
        id: `assistant-${event.run_id}`,
        run_id: event.run_id,
        role: "assistant",
        content: text,
        streaming: true,
      }];
    }
    return messages.map((message, index) => index === existingIndex
      ? { ...message, run_id: event.run_id, content: `${message.content}${text}`, streaming: true }
      : message);
  }

  if (["assistant.message", "message.completed", "run.completed", "completed"].includes(type)) {
    if (isHydratedFinal) return [...messages];
    if (!existing && text) {
      return [...messages, {
        id: `assistant-${event.run_id}`,
        run_id: event.run_id,
        role: "assistant",
        content: text,
      }];
    }
    if (!existing) return [...messages];
    return messages.map((message, index) => index === existingIndex
      ? { ...message, run_id: event.run_id, content: text || message.content, streaming: false }
      : message);
  }

  if (type.includes("failed")) {
    const failure = text ?? "The run stopped before it could finish.";
    if (!existing) {
      return [...messages, {
        id: `assistant-${event.run_id}`,
        run_id: event.run_id,
        role: "assistant",
        content: failure,
        failed: true,
      }];
    }
    return messages.map((message, index) => index === existingIndex
      ? { ...message, run_id: event.run_id, content: failure, streaming: false, failed: true }
      : message);
  }

  return [...messages];
}
