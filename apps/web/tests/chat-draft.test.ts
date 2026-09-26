import assert from "node:assert/strict";
import test from "node:test";

import { outgoingChatDraft, restoreChatDraft } from "../lib/chat-draft.ts";
import type { AttachmentRef } from "../lib/types.ts";

const document = { id: "document", name: "brief.pdf", media_type: "application/pdf" } as AttachmentRef;
const image = { id: "image", name: "screen.png", media_type: "image/png" } as AttachmentRef;

test("a queued attachment-only message carries its files into the request", () => {
  const draft = { content: "Still drafting", attachments: [image] };
  assert.deepEqual(outgoingChatDraft(draft, "", [document]), { content: "", attachments: [document] });
  assert.deepEqual(draft, { content: "Still drafting", attachments: [image] });
});

test("a generated prompt does not consume unrelated draft attachments", () => {
  assert.deepEqual(outgoingChatDraft({ content: "Unsent", attachments: [image] }, " Make a tool "), { content: "Make a tool", attachments: [] });
});

test("undoing a queue restores both messages and deduplicates attachments", () => {
  assert.deepEqual(restoreChatDraft(
    { content: "Queued message", attachments: [document] },
    { content: "New draft", attachments: [image, document] },
  ), { content: "Queued message\n\nNew draft", attachments: [document, image] });
});

test("restoring an attachment-only message does not add blank lines", () => {
  assert.deepEqual(restoreChatDraft(
    { content: "", attachments: [document] },
    { content: "New draft", attachments: [] },
  ), { content: "New draft", attachments: [document] });
});
