"use client";

import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import { FileText, NotebookPen, Pencil, Pin, Plus, Search, Trash2 } from "lucide-react";
import { createCustomerNote, deleteCustomerNote, updateCustomerNote } from "@/lib/api";
import { when } from "@/lib/customers";
import { accountNoteDrafts, hasNoteContent, sameNoteDraft, type NoteDraft } from "@/components/customers/note-drafts";
import type { Customers } from "@/hooks/use-customers";
import type { CustomerAccountDetail, CustomerNote } from "@/lib/types";

function NoteEditor({ draft, onChange, onSave, onClose, onDiscard, busy, creating }: { draft: NoteDraft; onChange: (value: NoteDraft) => void; onSave: () => void; onClose: () => void; onDiscard: () => void; busy: boolean; creating: boolean }) {
  const titleInput = useRef<HTMLInputElement>(null);
  useEffect(() => { if (!titleInput.current?.closest("[hidden]")) titleInput.current?.focus(); }, []);
  return <form className="customer-note-editor" onSubmit={(event) => { event.preventDefault(); if (draft.body.trim() && !busy) onSave(); }} onKeyDown={(event) => { if ((event.metaKey || event.ctrlKey) && event.key === "Enter") { event.preventDefault(); if (draft.body.trim() && !busy) onSave(); } }}>
    <header><span className="ui-eyebrow">{creating ? "New account note" : "Editing note"}</span><span>Draft · not saved yet</span></header>
    <label className="customer-note-title-field"><span className="sr-only">Note title</span><input ref={titleInput} value={draft.title} disabled={busy} onChange={(event) => onChange({ ...draft, title: event.target.value })} placeholder="Give this note a title…" /></label>
    <label className="customer-note-body-field"><span className="sr-only">Note body</span><textarea value={draft.body} disabled={busy} onChange={(event) => onChange({ ...draft, body: event.target.value })} placeholder={"What should you remember about this account?\n\nCapture the context, decisions, or details you’ll want close at hand."} /></label>
    <label className="customer-note-pin-field"><input type="checkbox" checked={draft.pinned} disabled={busy} onChange={(event) => onChange({ ...draft, pinned: event.target.checked })} /><span><strong>Pin to account context</strong><small>Include this note in conversations scoped to this account.</small></span></label>
    <footer><button type="button" className="ui-btn is-quiet is-sm is-danger-text" disabled={busy} onClick={onDiscard}>Discard</button><span className="customer-note-word-count">{draft.body.trim() ? draft.body.trim().split(/\s+/).length : 0} words</span><button type="button" className="ui-btn is-quiet" disabled={busy} onClick={onClose}>Keep draft</button><button type="submit" className="ui-btn is-primary" disabled={busy || !draft.body.trim()}>{busy ? "Saving…" : "Save note"}</button></footer>
  </form>;
}

export function Notes({ c, detail, composerOpen, setComposerOpen }: { c: Customers; detail: CustomerAccountDetail; composerOpen: boolean; setComposerOpen: (open: boolean) => void }) {
  const accountId = detail.account.id;
  const [draftStore] = useState(() => accountNoteDrafts(accountId));
  const drafts = useSyncExternalStore(draftStore.subscribe, draftStore.getSnapshot, draftStore.getSnapshot);
  const mounted = useRef(true);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  const [selectedId, setSelectedId] = useState<string | null>(c.requestedNote);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [pinnedOnly, setPinnedOnly] = useState(false);
  const hasDraft = hasNoteContent(drafts.creating);
  const unfinished = hasDraft || Object.entries(drafts.editing).some(([id, draft]) => { const saved = detail.notes.find((item) => item.id === id); return !saved || !sameNoteDraft(draft, saved); });
  useEffect(() => {
    if (!unfinished) return;
    const warn = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = ""; };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [unfinished]);
  useEffect(() => { if (c.requestedNote) { setSelectedId(c.requestedNote); setEditingId(null); setComposerOpen(false); setQuery(""); setPinnedOnly(false); } }, [c.requestedNote, setComposerOpen]);
  const notes = useMemo(() => [...detail.notes].filter((note) => (!pinnedOnly || note.pinned) && (!query.trim() || `${note.title} ${note.body}`.toLowerCase().includes(query.trim().toLowerCase()))).sort((a, b) => Number(b.pinned) - Number(a.pinned) || b.updated_at.localeCompare(a.updated_at)), [detail.notes, pinnedOnly, query]);
  const note = notes.find((item) => item.id === selectedId) ?? notes[0];
  const edit = note && editingId === note.id ? drafts.editing[note.id] : null;
  const openNote = (next: CustomerNote) => { setSelectedId(next.id); setEditingId(null); setComposerOpen(false); c.select(accountId, "notes", { kind: "note", id: next.id }); };
  const discard = (id?: string) => {
    const draft = id ? drafts.editing[id] : drafts.creating;
    const saved = id ? detail.notes.find((item) => item.id === id) : null;
    if (draft && (saved ? !sameNoteDraft(draft, saved) : hasNoteContent(draft)) && !window.confirm("Discard this unsaved draft?")) return;
    if (id) { draftStore.clearEditing(id); setEditingId(null); }
    else { draftStore.clearCreating(); setComposerOpen(false); }
  };
  const saveNew = () => {
    let created: CustomerNote | undefined;
    void c.run("new-note", async () => { created = await createCustomerNote(accountId, { title: drafts.creating.title.trim(), body: drafts.creating.body.trim(), pinned: drafts.creating.pinned }); }, drafts.creating.pinned ? "Note saved and pinned." : "Note saved.").then((ok) => {
      if (!ok) return;
      draftStore.clearCreating(drafts.creating);
      if (!mounted.current) return;
      setComposerOpen(false); setQuery(""); setPinnedOnly(false); if (created) setSelectedId(created.id);
    });
  };
  const saveEdit = (current: CustomerNote, draft: NoteDraft) => void c.run(current.id, () => updateCustomerNote(current.id, { title: draft.title.trim(), body: draft.body.trim(), pinned: draft.pinned }), "Note updated.").then((ok) => {
    if (!ok) return;
    draftStore.clearEditing(current.id, draft); if (mounted.current) setEditingId(null);
  });

  return <div className="customer-notes">
    <div className="customer-section-heading"><div><h2>Account notes</h2><p>Your working memory for {detail.account.name}. Pin the context you want in every conversation.</p></div><button type="button" className="ui-btn" onClick={() => setComposerOpen(true)}><Plus size={15} aria-hidden="true" />New note</button></div>
    <div className="customer-notebook">
      <aside className="customer-note-index" aria-label="Account notes">
        <label className="workspace-search"><Search size={15} aria-hidden="true" /><input type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Find a note…" aria-label="Search account notes" /></label>
        <div className="customer-note-filters" role="group" aria-label="Filter notes"><button type="button" className={`ui-chip${!pinnedOnly ? " is-accent" : ""}`} aria-pressed={!pinnedOnly} onClick={() => setPinnedOnly(false)}>All <small>{detail.notes.length}</small></button><button type="button" className={`ui-chip${pinnedOnly ? " is-accent" : ""}`} aria-pressed={pinnedOnly} onClick={() => setPinnedOnly(true)}><Pin size={12} aria-hidden="true" />Pinned <small>{detail.notes.filter((item) => item.pinned).length}</small></button></div>
        {hasDraft || composerOpen ? <button type="button" className={`customer-note-list-item is-draft${composerOpen ? " is-selected" : ""}`} onClick={() => setComposerOpen(true)}><strong><NotebookPen size={14} aria-hidden="true" />{drafts.creating.title || "Untitled draft"}</strong><small>Continue writing · not saved</small></button> : null}
        <div className="customer-note-list">{notes.map((item) => <button key={item.id} type="button" className={`customer-note-list-item${!composerOpen && note?.id === item.id ? " is-selected" : ""}`} aria-current={!composerOpen && note?.id === item.id ? "true" : undefined} onClick={() => openNote(item)}><strong>{item.pinned ? <Pin size={13} aria-label="Pinned" /> : null}{item.title || "Untitled note"}</strong><span>{item.body}</span><small>{when(item.updated_at)}{drafts.editing[item.id] && !sameNoteDraft(drafts.editing[item.id], item) ? " · Draft changes" : ""}</small></button>)}</div>
        {!notes.length ? <p className="customer-note-index-empty">{detail.notes.length ? "No notes match this view." : "Your notes will appear here."}</p> : null}
      </aside>
      <div className="customer-note-page">
        {composerOpen ? <NoteEditor draft={drafts.creating} onChange={draftStore.setCreating} onSave={saveNew} onClose={() => setComposerOpen(false)} onDiscard={() => discard()} busy={Boolean(c.busy)} creating />
          : note && edit ? <NoteEditor draft={edit} onChange={(next) => draftStore.setEditing(note.id, next)} onSave={() => saveEdit(note, edit)} onClose={() => setEditingId(null)} onDiscard={() => discard(note.id)} busy={Boolean(c.busy)} creating={false} />
          : note ? <article className="customer-note-reading" id={`note-${note.id}`}><header><span className="ui-eyebrow">{note.pinned ? <><Pin size={12} aria-hidden="true" />Pinned to context</> : <><FileText size={12} aria-hidden="true" />Account note</>}</span><button type="button" className="ui-btn is-sm" disabled={Boolean(c.busy)} onClick={() => { draftStore.setEditing(note.id, drafts.editing[note.id] ?? { title: note.title, body: note.body, pinned: note.pinned }); setEditingId(note.id); }}><Pencil size={13} aria-hidden="true" />{drafts.editing[note.id] && !sameNoteDraft(drafts.editing[note.id], note) ? "Resume editing" : "Edit note"}</button></header><h3>{note.title || "Untitled note"}</h3><div className="customer-note-metadata">Updated {when(note.updated_at)}{note.origin === "chat" ? " · Saved from a conversation" : " · Written by you"}</div><div className="customer-note-body">{note.body}</div><footer><button type="button" className="ui-btn is-quiet is-sm" disabled={Boolean(c.busy)} onClick={() => void c.run(note.id, () => updateCustomerNote(note.id, { title: note.title, body: note.body, pinned: !note.pinned }), note.pinned ? "Note unpinned." : "Note pinned to account context.")}><Pin size={13} aria-hidden="true" />{note.pinned ? "Unpin note" : "Pin to context"}</button><button type="button" className="ui-btn is-quiet is-sm is-danger-text" disabled={Boolean(c.busy)} onClick={() => { if (!window.confirm("Delete this note?")) return; void c.run(note.id, () => deleteCustomerNote(note.id), "Note deleted.").then((ok) => { if (ok) { draftStore.clearEditing(note.id); } }); }}><Trash2 size={13} aria-hidden="true" />Delete</button></footer></article>
          : <div className="customer-note-blank"><NotebookPen size={34} aria-hidden="true" /><h3>{detail.notes.length ? "Find your next note" : "A little context goes a long way"}</h3><p>{detail.notes.length ? "Try another search or show all notes to keep reading." : "Keep useful details, ideas, and working notes here. Nothing is analyzed unless you choose to capture a source."}</p><button type="button" className="ui-btn is-primary" onClick={() => { if (detail.notes.length) { setQuery(""); setPinnedOnly(false); } else setComposerOpen(true); }}>{detail.notes.length ? "Show all notes" : "Write your first note"}</button></div>}
      </div>
    </div>
  </div>;
}
