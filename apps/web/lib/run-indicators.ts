"use client";

export type ConversationRunState = "working" | "attention" | "done" | "failed" | "cancelled";

export interface ConversationRunIndicator {
  conversationId: string;
  runId: string;
  title: string;
  state: ConversationRunState;
  unread: boolean;
  updatedAt: string;
}

const STORAGE_KEY = "metis.conversation-runs.v1";
export const RUN_INDICATORS_CHANGED_EVENT = "metis:run-indicators-changed";

export function readRunIndicators(): Record<string, ConversationRunIndicator> {
  if (typeof window === "undefined") return {};
  try {
    const parsed = JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? "{}") as unknown;
    return parsed && typeof parsed === "object" ? parsed as Record<string, ConversationRunIndicator> : {};
  } catch {
    return {};
  }
}

function write(indicators: Record<string, ConversationRunIndicator>): void {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(indicators));
  window.dispatchEvent(new Event(RUN_INDICATORS_CHANGED_EVENT));
}

export function trackConversationRun(input: { conversationId: string; runId: string; title: string }): void {
  if (typeof window === "undefined") return;
  const current = readRunIndicators();
  current[input.conversationId] = {
    conversationId: input.conversationId,
    runId: input.runId,
    title: input.title || "Conversation",
    state: "working",
    unread: false,
    updatedAt: new Date().toISOString(),
  };
  write(current);
}

export function updateConversationRun(
  runId: string,
  state: ConversationRunState,
  unread: boolean,
): void {
  if (typeof window === "undefined") return;
  const current = readRunIndicators();
  const item = Object.values(current).find((candidate) => candidate.runId === runId);
  if (!item) return;
  current[item.conversationId] = { ...item, state, unread, updatedAt: new Date().toISOString() };
  write(current);
}

export function acknowledgeConversationRun(conversationId: string): void {
  if (typeof window === "undefined") return;
  const current = readRunIndicators();
  const item = current[conversationId];
  if (!item?.unread) return;
  current[conversationId] = { ...item, unread: false };
  write(current);
}
