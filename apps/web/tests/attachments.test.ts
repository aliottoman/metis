import assert from "node:assert/strict";
import test from "node:test";

import { attachmentBadge, CHAT_ATTACHMENT_ACCEPT } from "../lib/attachments.ts";

test("the chat picker includes the supported raster image formats", () => {
  for (const value of [
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
  ]) {
    assert.ok(CHAT_ATTACHMENT_ACCEPT.split(",").includes(value), `${value} is accepted`);
  }
  for (const value of [".svg", ".heic", ".tiff"]) {
    assert.ok(!CHAT_ATTACHMENT_ACCEPT.split(",").includes(value), `${value} is excluded`);
  }
});

test("image attachments are visibly distinguished from text files", () => {
  assert.equal(attachmentBadge({ id: "one", name: "diagram.png", media_type: "image/png" }), "IMG");
  assert.equal(attachmentBadge({ id: "two", name: "PHOTO.JPEG" }), "IMG");
  assert.equal(attachmentBadge({ id: "three", name: "notes.md", media_type: "text/markdown" }), "TXT");
  assert.equal(attachmentBadge({ id: "four", name: "brief.pdf", media_type: "application/pdf" }), "PDF");
});
