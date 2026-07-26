"use client";

import type { ConversationSummary } from "@/lib/types";

const STORAGE_KEY = "metis.recent-conversations.v1";
export const CONVERSATIONS_CHANGED_EVENT = "metis:conversations-changed";

export function readRecentConversations(): ConversationSummary[] {
  if (typeof window === "undefined") return [];
  try {
    const parsed = JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? "[]") as unknown;
    return Array.isArray(parsed) ? (parsed as ConversationSummary[]).filter((item) => item?.id) : [];
  } catch {
    return [];
  }
}

export function rememberConversation(conversation: ConversationSummary): void {
  if (typeof window === "undefined" || !conversation.id) return;
  const current = readRecentConversations().filter((item) => item.id !== conversation.id);
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify([conversation, ...current].slice(0, 40)));
  window.dispatchEvent(new Event(CONVERSATIONS_CHANGED_EVENT));
}
