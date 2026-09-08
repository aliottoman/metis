"use client";

// Follows a run through the event stream the API already keeps, so the shell
// learns that a run finished or is waiting on you the moment it happens,
// without asking every few seconds. The stream closes itself once the run is
// terminal; one read of the record then confirms the final word.

import { getRunRecord, runEventsUrl } from "@/lib/api";
import type { ConversationRunState } from "@/lib/run-indicators";
import { parseSseBuffer } from "@/lib/sse";

const STATE_BY_EVENT: Record<string, ConversationRunState> = {
  "run.completed": "done",
  "run.failed": "failed",
  "run.cancelled": "cancelled",
  "run.awaiting_approval": "attention",
  "run.awaiting_input": "attention",
  "run.suspended": "attention",
  "run.recovered_awaiting_approval": "attention",
  "approval.required": "attention",
  "elicitation.requested": "attention",
  "run.started": "working",
  "run.resumed": "working",
  "run.recovered": "working",
  "approval.decided": "working",
  "elicitation.answered": "working",
};

const TERMINAL = new Set<ConversationRunState>(["done", "failed", "cancelled"]);

export function stateOfRunStatus(status: string): ConversationRunState {
  if (status === "completed") return "done";
  if (status === "failed") return "failed";
  if (status === "cancelled") return "cancelled";
  if (status === "awaiting_approval" || status === "awaiting_input") return "attention";
  return "working";
}

/** Start following a run. Returns the function that stops following it. */
export function watchRun(runId: string, onState: (state: ConversationRunState) => void): () => void {
  const controller = new AbortController();
  let stopped = false;

  const confirm = async (): Promise<boolean> => {
    try {
      const run = await getRunRecord(runId);
      if (stopped) return true;
      const state = stateOfRunStatus(run.status);
      onState(state);
      return TERMINAL.has(state);
    } catch {
      return false;
    }
  };

  const follow = async () => {
    let delay = 1000;
    while (!stopped) {
      try {
        const response = await fetch(runEventsUrl(runId, 0), {
          headers: { accept: "text/event-stream" },
          cache: "no-store",
          signal: controller.signal,
        });
        if (!response.ok || !response.body) throw new Error(`stream ${response.status}`);
        delay = 1000;
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        while (!stopped) {
          const { done, value } = await reader.read();
          buffer += decoder.decode(value, { stream: !done });
          const parsed = parseSseBuffer(buffer);
          buffer = parsed.rest;
          for (const frame of parsed.frames) {
            const state = frame.event ? STATE_BY_EVENT[frame.event] : undefined;
            if (state) onState(state);
          }
          if (done) break;
        }
        // The server ends the stream when the run is terminal; confirm it.
        if (stopped || (await confirm())) return;
      } catch {
        if (stopped || controller.signal.aborted) return;
        if (await confirm()) return;
      }
      await new Promise((resolve) => window.setTimeout(resolve, delay));
      delay = Math.min(delay * 2, 30_000);
    }
  };

  void follow();
  return () => {
    stopped = true;
    controller.abort();
  };
}
