import type { AttachmentRef } from "./types.ts";

export type ChatDraft = { content: string; attachments: AttachmentRef[] };

/** Restoring a queued or failed message must preserve anything typed after it. */
export function restoreChatDraft(previous: ChatDraft, current: ChatDraft): ChatDraft {
  return {
    content: [previous.content, current.content].filter((text) => text.trim()).join("\n\n"),
    attachments: Array.from(new Map([...previous.attachments, ...current.attachments].map((item) => [item.id, item])).values()),
  };
}

/** Explicit messages carry their own files; a generated prompt has none. */
export function outgoingChatDraft(current: ChatDraft, overrideContent?: string, overrideAttachments?: AttachmentRef[]): ChatDraft {
  return typeof overrideContent === "string"
    ? { content: overrideContent.trim(), attachments: overrideAttachments ?? [] }
    : { content: current.content.trim(), attachments: current.attachments };
}
