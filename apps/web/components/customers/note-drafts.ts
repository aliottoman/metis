export type NoteDraft = { title: string; body: string; pinned: boolean };
export type NoteDrafts = { creating: NoteDraft; editing: Record<string, NoteDraft> };
export const emptyNote = (): NoteDraft => ({ title: "", body: "", pinned: false });
export const hasNoteContent = (draft: NoteDraft) => Boolean(draft.title || draft.body || draft.pinned);
export const sameNoteDraft = (left: NoteDraft, right: NoteDraft) => left.title === right.title && left.body === right.body && left.pinned === right.pinned;

// A save can finish after a notebook unmounts. Every new subscriber must see
// that completion, and a late completion may only clear the draft it saved.
export function createNoteDraftStore() {
  let snapshot: NoteDrafts = { creating: emptyNote(), editing: {} };
  const listeners = new Set<() => void>();
  const publish = (next: NoteDrafts) => { snapshot = next; for (const listener of listeners) listener(); };
  return {
    getSnapshot: () => snapshot,
    subscribe: (listener: () => void) => { listeners.add(listener); return () => { listeners.delete(listener); }; },
    setCreating: (creating: NoteDraft) => publish({ ...snapshot, creating }),
    setEditing: (id: string, draft: NoteDraft) => publish({ ...snapshot, editing: { ...snapshot.editing, [id]: draft } }),
    clearCreating: (saved?: NoteDraft) => {
      if (saved && !sameNoteDraft(snapshot.creating, saved)) return;
      publish({ ...snapshot, creating: emptyNote() });
    },
    clearEditing: (id: string, saved?: NoteDraft) => {
      if (saved && (!snapshot.editing[id] || !sameNoteDraft(snapshot.editing[id], saved))) return;
      const editing = { ...snapshot.editing }; delete editing[id];
      publish({ ...snapshot, editing });
    },
  };
}

const accounts = new Map<string, ReturnType<typeof createNoteDraftStore>>();
export function accountNoteDrafts(accountId: string) {
  let store = accounts.get(accountId);
  if (!store) { store = createNoteDraftStore(); accounts.set(accountId, store); }
  return store;
}
