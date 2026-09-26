import assert from "node:assert/strict";
import test from "node:test";
import { accountNoteDrafts, createNoteDraftStore, hasNoteContent, sameNoteDraft } from "../components/customers/note-drafts.ts";

const original = { title: "Discovery", body: "Confirm the launch window.", pinned: true };

test("a note save updates the replacement notebook after account navigation", () => {
  const store = createNoteDraftStore();
  store.setCreating(original);
  const unsubscribe = store.subscribe(() => {});
  unsubscribe();
  let notifications = 0;
  store.subscribe(() => { notifications += 1; });
  store.clearCreating(original);
  assert.equal(notifications, 1);
  assert.equal(hasNoteContent(store.getSnapshot().creating), false);
});

test("a late save never discards newer writing or another note's draft", () => {
  const store = createNoteDraftStore();
  store.setCreating(original);
  store.setEditing("other", original);
  store.setCreating({ ...original, body: "A new unsaved thought." });
  store.clearCreating(original);
  assert.equal(store.getSnapshot().creating.body, "A new unsaved thought.");
  assert.deepEqual(store.getSnapshot().editing.other, original);
  store.setEditing("note", original);
  store.setEditing("note", { ...original, pinned: false });
  store.clearEditing("note", original);
  assert.equal(store.getSnapshot().editing.note.pinned, false);
  store.clearEditing("note", { ...original, pinned: false });
  assert.equal(store.getSnapshot().editing.note, undefined);
  assert.deepEqual(store.getSnapshot().editing.other, original);
});

test("drafts follow their own account and survive reopening it", () => {
  accountNoteDrafts("draft-test-a").setCreating(original);
  assert.deepEqual(accountNoteDrafts("draft-test-a").getSnapshot().creating, original);
  assert.equal(hasNoteContent(accountNoteDrafts("draft-test-b").getSnapshot().creating), false);
  assert.equal(sameNoteDraft(original, { ...original }), true);
  assert.equal(sameNoteDraft(original, { ...original, body: "Changed" }), false);
  accountNoteDrafts("draft-test-a").clearCreating();
});
