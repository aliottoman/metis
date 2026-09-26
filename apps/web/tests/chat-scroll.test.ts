import assert from "node:assert/strict";
import test from "node:test";

import { nextThreadFollowing } from "../lib/chat-scroll.ts";

test("an upward scroll pauses live following even close to the end", () => {
  assert.equal(nextThreadFollowing(true, 500, 480, 20, "scroll"), false);
  assert.equal(nextThreadFollowing(false, 480, 480, 20, "resize"), false);
});

test("scrolling back to the bottom resumes following", () => {
  assert.equal(nextThreadFollowing(false, 400, 470, 20, "scroll"), true);
  assert.equal(nextThreadFollowing(false, 400, 450, 140, "scroll"), false);
});
