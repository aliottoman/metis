"use client";

// One account's records, a section per tab. Each section keeps its own
// drafts; the detail remounts on account change, so nothing carries over.

import { useState, type ReactNode } from "react";

import { WinValuation } from "@/components/customers/win-valuation";
import { createCustomerAction, createCustomerFact, createCustomerNote, createCustomerOutput, deleteCustomerAction, deleteCustomerFact, deleteCustomerNote, deleteCustomerPerson, deleteCustomerSource, deleteCustomerWin, editCustomerAction, addCustomerPerson, saveCustomerSettings, updateCustomerAction, updateCustomerFact, updateCustomerNote, updateCustomerPerson, updateCustomerSource } from "@/lib/api";
import { actionMeta, dateInputValue, FACT_KINDS, instantFromDateInput, isOverdue, SOURCE_KINDS, TECHNICAL_KINDS, usd, when, winDay } from "@/lib/customers";
import type { Customers } from "@/hooks/use-customers";
import type { CustomerAccountDetail, CustomerAction, CustomerFact, CustomerNote, CustomerOutput, CustomerPerson, CustomerSource } from "@/lib/types";

type Props = { c: Customers; detail: CustomerAccountDetail };

/** A list's top line: a sentence about it and the one button that adds to it. */
function ListHead({ text, adding, onToggle, label, children }: { text: ReactNode; adding: boolean; onToggle: () => void; label: string; children?: ReactNode }) {
  return (
    <div className="records-head">
      <p>{text}</p>
      {children}
      <button type="button" className={`ui-btn is-sm${adding ? "" : " is-primary"}`} onClick={onToggle}>{adding ? "Close" : label}</button>
    </div>
  );
}

/** The Cancel / Save pair under an inline form. */
function FormActions({ onCancel, onSave, canSave, saving, label = "Save" }: { onCancel: () => void; onSave: () => void; canSave: boolean; saving: boolean; label?: string }) {
  return (
    <div className="records-form-actions">
      <button type="button" className="ui-btn is-quiet is-sm" onClick={onCancel}>Cancel</button>
      <button type="button" className="ui-btn is-primary is-sm" disabled={!canSave || saving} onClick={onSave}>{saving ? "Saving…" : label}</button>
    </div>
  );
}

function RowActions({ onEdit, onDelete, busy, children }: { onEdit: () => void; onDelete: () => void; busy: boolean; children?: ReactNode }) {
  return (
    <div className="records-row-actions">
      {children}
      <button type="button" className="ui-btn is-quiet is-sm" onClick={onEdit}>Edit</button>
      <button type="button" className="ui-btn is-quiet is-sm is-danger-text" disabled={busy} onClick={onDelete}>Delete</button>
    </div>
  );
}

function Empty({ children }: { children: ReactNode }) {
  return <p className="records-empty">{children}</p>;
}

// ── Overview ───────────────────────────────────────────────────────────────

export function Overview({ c, detail }: Props) {
  const open = detail.actions.filter((item) => item.status === "open");
  const recent = [...detail.sources].sort((a, b) => b.created_at.localeCompare(a.created_at)).slice(0, 4);
  const pinned = detail.notes.filter((note) => note.pinned);
  return (
    <div className="overview">
      <dl className="overview-facts">
        <div><dt>Last interaction</dt><dd>{when(detail.account.last_interaction_at)}</dd></div>
        <div><dt>Open actions</dt><dd>{detail.account.open_actions}</dd></div>
        <div><dt>Facts</dt><dd>{detail.facts.length}</dd></div>
        <div><dt>Notes</dt><dd>{detail.notes.length}</dd></div>
      </dl>
      {open.length ? (
        <section className="overview-card">
          <h3>Next actions</h3>
          {open.slice(0, 5).map((action) => (
            <label key={action.id} className="overview-action">
              <input type="checkbox" checked={false} disabled={c.busy === action.id} onChange={() => void c.run(action.id, () => updateCustomerAction(action.id, "done"))} />
              <span>{action.description}<small>{actionMeta(action)}</small></span>
            </label>
          ))}
        </section>
      ) : null}
      {detail.facts.length ? (
        <section className="overview-card">
          <h3>Latest understanding</h3>
          {detail.facts.slice(0, 6).map((fact) => <p key={fact.id}><b>{fact.kind.replace("_", " ")}</b>{fact.content}</p>)}
        </section>
      ) : null}
      {recent.length ? (
        <section className="overview-card">
          <h3>Recently captured</h3>
          {recent.map((source) => (
            <p key={source.id}><b>{source.source_kind}</b>{source.title || source.content.slice(0, 120)}{source.status === "waiting" ? <em> · waiting for analysis</em> : source.status === "review" ? <em> · ready to review</em> : null}</p>
          ))}
        </section>
      ) : null}
      {pinned.length ? (
        <section className="overview-card">
          <h3>Pinned notes</h3>
          {pinned.map((note) => <p key={note.id}><b>{note.title || "note"}</b>{note.body.slice(0, 220)}{note.body.length > 220 ? "…" : ""}</p>)}
        </section>
      ) : null}
      {!open.length && !detail.facts.length && !recent.length ? <Empty>Nothing recorded yet. Capture a note or add a fact to start.</Empty> : null}
    </div>
  );
}

// ── Notes ──────────────────────────────────────────────────────────────────

type NoteDraft = { title: string; body: string; pinned: boolean };
const EMPTY_NOTE: NoteDraft = { title: "", body: "", pinned: false };

function NoteForm({ draft, onChange, onCancel, onSave, saving }: { draft: NoteDraft; onChange: (draft: NoteDraft) => void; onCancel: () => void; onSave: () => void; saving: boolean }) {
  return (
    <article className="records-form">
      <input value={draft.title} onChange={(event) => onChange({ ...draft, title: event.target.value })} placeholder="Title (optional)" aria-label="Note title" />
      <textarea value={draft.body} onChange={(event) => onChange({ ...draft, body: event.target.value })} placeholder="What should this account always carry with it?" aria-label="Note body" />
      <label className="records-check"><input type="checkbox" checked={draft.pinned} onChange={(event) => onChange({ ...draft, pinned: event.target.checked })} /> Pin to account context</label>
      <FormActions onCancel={onCancel} onSave={onSave} canSave={Boolean(draft.body.trim())} saving={saving} />
    </article>
  );
}

export function Notes({ c, detail, composerOpen, setComposerOpen }: Props & { composerOpen: boolean; setComposerOpen: (open: boolean) => void }) {
  const [draft, setDraft] = useState(EMPTY_NOTE);
  const [editing, setEditing] = useState<{ id: string; draft: NoteDraft } | null>(null);
  const id = detail.account.id;
  const save = (note: CustomerNote, next: NoteDraft, notice?: string) => c.run(note.id, () => updateCustomerNote(note.id, { title: next.title.trim(), body: next.body.trim(), pinned: next.pinned }), notice).then((ok) => ok && setEditing(null));
  return (
    <section className="records">
      <ListHead text="Notes are yours: written directly, never analyzed. Pin one to hand it to every conversation scoped to this account." adding={composerOpen} onToggle={() => setComposerOpen(!composerOpen)} label="New note" />
      {composerOpen ? (
        <NoteForm draft={draft} onChange={setDraft} onCancel={() => { setComposerOpen(false); setDraft(EMPTY_NOTE); }} saving={c.busy === "new-note"} onSave={() => void c.run("new-note", () => createCustomerNote(id, { title: draft.title.trim(), body: draft.body.trim(), pinned: draft.pinned }), draft.pinned ? "Note saved and pinned." : "Note saved.").then((ok) => { if (ok) { setDraft(EMPTY_NOTE); setComposerOpen(false); } })} />
      ) : null}
      {detail.notes.map((note) => editing?.id === note.id ? (
        <NoteForm key={note.id} draft={editing.draft} onChange={(next) => setEditing({ id: note.id, draft: next })} onCancel={() => setEditing(null)} saving={c.busy === note.id} onSave={() => void save(note, editing.draft)} />
      ) : (
        <article key={note.id} className={`records-row${note.pinned ? " is-pinned" : ""}`}>
          <div>
            <strong>{note.title || "Note"}</strong>
            <small>{when(note.updated_at)}{note.origin === "chat" ? " · saved from a conversation" : ""}{note.pinned ? " · pinned" : ""}</small>
            <p>{note.body}</p>
          </div>
          <RowActions busy={c.busy === note.id} onEdit={() => setEditing({ id: note.id, draft: { title: note.title, body: note.body, pinned: note.pinned } })} onDelete={() => window.confirm("Delete this note?") && void c.run(note.id, () => deleteCustomerNote(note.id))}>
            <button type="button" className="ui-btn is-quiet is-sm" disabled={c.busy === note.id} onClick={() => void save(note, { title: note.title, body: note.body, pinned: !note.pinned })}>{note.pinned ? "Unpin" : "Pin"}</button>
          </RowActions>
        </article>
      ))}
      {!detail.notes.length && !composerOpen ? <Empty>No notes on this account yet.</Empty> : null}
    </section>
  );
}

// ── Actions ────────────────────────────────────────────────────────────────

type ActionDraft = { description: string; owner: string; due: string; status: CustomerAction["status"] };
const EMPTY_ACTION: ActionDraft = { description: "", owner: "", due: "", status: "open" };

function ActionForm({ draft, onChange, onCancel, onSave, saving, withStatus, label }: { draft: ActionDraft; onChange: (draft: ActionDraft) => void; onCancel: () => void; onSave: () => void; saving: boolean; withStatus?: boolean; label?: string }) {
  return (
    <div className="records-form is-inline">
      <input value={draft.description} onChange={(event) => onChange({ ...draft, description: event.target.value })} placeholder="What needs doing" aria-label="Action description" />
      <input value={draft.owner} onChange={(event) => onChange({ ...draft, owner: event.target.value })} placeholder="Owner" aria-label="Action owner" />
      <input type="date" value={draft.due} onChange={(event) => onChange({ ...draft, due: event.target.value })} aria-label="Due date" />
      {withStatus ? (
        <select value={draft.status} onChange={(event) => onChange({ ...draft, status: event.target.value as ActionDraft["status"] })} aria-label="Action status">
          <option value="open">Open</option><option value="done">Done</option><option value="cancelled">Cancelled</option>
        </select>
      ) : null}
      <FormActions onCancel={onCancel} onSave={onSave} canSave={Boolean(draft.description.trim())} saving={saving} label={label} />
    </div>
  );
}

export function Actions({ c, detail }: Props) {
  const [draft, setDraft] = useState<ActionDraft | null>(null);
  const [editing, setEditing] = useState<{ id: string; draft: ActionDraft } | null>(null);
  const open = detail.actions.filter((item) => item.status === "open").length;
  return (
    <section className="records">
      <ListHead text={detail.actions.length ? `${open} open of ${detail.actions.length}.` : "No actions captured yet."} adding={Boolean(draft)} onToggle={() => setDraft(draft ? null : EMPTY_ACTION)} label="Add action" />
      {draft ? (
        <ActionForm draft={draft} onChange={setDraft} onCancel={() => setDraft(null)} saving={c.busy === "new-action"} label="Add" onSave={() => void c.run("new-action", () => createCustomerAction(detail.account.id, { description: draft.description.trim(), owner: draft.owner.trim(), due_at: instantFromDateInput(draft.due) })).then((ok) => ok && setDraft(null))} />
      ) : null}
      {detail.actions.map((action) => editing?.id === action.id ? (
        <ActionForm key={action.id} draft={editing.draft} withStatus onChange={(next) => setEditing({ id: action.id, draft: next })} onCancel={() => setEditing(null)} saving={c.busy === action.id} onSave={() => void c.run(action.id, () => editCustomerAction(action.id, { description: editing.draft.description.trim(), owner: editing.draft.owner.trim(), due_at: instantFromDateInput(editing.draft.due), status: editing.draft.status })).then((ok) => ok && setEditing(null))} />
      ) : (
        <article key={action.id} id={`action-${action.id}`} className={`records-row is-action${action.status !== "open" ? " is-done" : ""}${isOverdue(action) ? " is-overdue" : ""}`}>
          <label>
            <input type="checkbox" checked={action.status === "done"} disabled={c.busy === action.id} onChange={() => void c.run(action.id, () => updateCustomerAction(action.id, action.status === "open" ? "done" : "open"))} />
            <span><strong>{action.description}</strong><small>{actionMeta(action)}</small></span>
          </label>
          <RowActions busy={c.busy === action.id} onEdit={() => setEditing({ id: action.id, draft: { description: action.description, owner: action.owner, due: dateInputValue(action.due_at), status: action.status } })} onDelete={() => window.confirm("Delete this action?") && void c.run(action.id, () => deleteCustomerAction(action.id))} />
        </article>
      ))}
    </section>
  );
}

// ── Wins ───────────────────────────────────────────────────────────────────

export function Wins({ c, detail }: Props) {
  return (
    <section className="records">
      <ListHead text={detail.wins.length ? `${detail.wins.length} ${detail.wins.length === 1 ? "win" : "wins"} recorded for ${detail.account.name}.` : "No wins recorded for this account yet."} adding={false} onToggle={() => c.setDialog({ kind: "win" })} label="Record win" />
      {detail.wins.map((win) => (
        <article key={win.id} className="records-row is-win">
          <div>
            <strong>{win.title}</strong>
            <small>{winDay(win.won_at || win.created_at)}{win.yearly_arr !== null ? ` · ${usd(win.yearly_arr)} yearly ARR` : ""}{win.dac_shape ? ` · DAC: ${win.dac_shape}` : ""}</small>
            {win.services.length ? <div className="records-chips">{win.services.map((service) => <span key={service} className="ui-chip">{service}</span>)}</div> : null}
            {win.brief ? <p>{win.brief}</p> : null}
            <WinValuation win={win} c={c} />
          </div>
          <RowActions busy={c.busy === win.id} onEdit={() => c.setDialog({ kind: "win", win })} onDelete={() => window.confirm("Remove this win from the tracker?") && void c.run(win.id, () => deleteCustomerWin(win.id))} />
        </article>
      ))}
    </section>
  );
}

// ── Facts ──────────────────────────────────────────────────────────────────

type FactDraft = { kind: string; content: string; status: CustomerFact["status"] };
const EMPTY_FACT: FactDraft = { kind: "requirement", content: "", status: "active" };
const FACT_FILTERS: Array<[string, string]> = [["all", "All"], ["technical", "Technical"], ["decision", "Decisions"], ["risk", "Risks"], ["question", "Questions"]];

function FactForm({ draft, onChange, onCancel, onSave, saving, withStatus, label }: { draft: FactDraft; onChange: (draft: FactDraft) => void; onCancel: () => void; onSave: () => void; saving: boolean; withStatus?: boolean; label?: string }) {
  return (
    <div className="records-form">
      <div className="records-form-row">
        <select value={draft.kind} onChange={(event) => onChange({ ...draft, kind: event.target.value })} aria-label="Fact kind">{FACT_KINDS.map((kind) => <option key={kind} value={kind}>{kind.replace("_", " ")}</option>)}</select>
        {withStatus ? (
          <select value={draft.status} onChange={(event) => onChange({ ...draft, status: event.target.value as FactDraft["status"] })} aria-label="Fact status">
            <option value="active">Active</option><option value="disputed">Disputed</option><option value="superseded">Superseded</option>
          </select>
        ) : null}
      </div>
      <textarea value={draft.content} onChange={(event) => onChange({ ...draft, content: event.target.value })} placeholder="What is true about this account?" aria-label="Fact content" />
      <FormActions onCancel={onCancel} onSave={onSave} canSave={Boolean(draft.content.trim())} saving={saving} label={label} />
    </div>
  );
}

export function Facts({ c, detail }: Props) {
  const [filter, setFilter] = useState("all");
  const [draft, setDraft] = useState<FactDraft | null>(null);
  const [editing, setEditing] = useState<{ id: string; draft: FactDraft } | null>(null);
  const visible = detail.facts.filter((fact) => filter === "all" || (filter === "technical" ? TECHNICAL_KINDS.includes(fact.kind) : fact.kind === filter));
  return (
    <section className="records">
      <ListHead text="" adding={Boolean(draft)} onToggle={() => setDraft(draft ? null : EMPTY_FACT)} label="Add fact">
        <div className="records-filters" role="group" aria-label="Filter facts">
          {FACT_FILTERS.map(([value, label]) => <button key={value} type="button" className={`ui-chip${filter === value ? " is-accent" : ""}`} aria-pressed={filter === value} onClick={() => setFilter(value)}>{label}</button>)}
        </div>
      </ListHead>
      {draft ? <FactForm draft={draft} onChange={setDraft} onCancel={() => setDraft(null)} saving={c.busy === "new-fact"} label="Add" onSave={() => void c.run("new-fact", () => createCustomerFact(detail.account.id, { kind: draft.kind, content: draft.content.trim() })).then((ok) => ok && setDraft(null))} /> : null}
      {visible.map((fact) => editing?.id === fact.id ? (
        <FactForm key={fact.id} draft={editing.draft} withStatus onChange={(next) => setEditing({ id: fact.id, draft: next })} onCancel={() => setEditing(null)} saving={c.busy === fact.id} onSave={() => void c.run(fact.id, () => updateCustomerFact(fact.id, { kind: editing.draft.kind, content: editing.draft.content.trim(), status: editing.draft.status })).then((ok) => ok && setEditing(null))} />
      ) : (
        <article key={fact.id} className={`records-row${fact.status !== "active" ? ` is-${fact.status}` : ""}`}>
          <div>
            <small>{fact.kind.replace("_", " ")}{fact.status !== "active" ? ` · ${fact.status}` : ""} · {Math.round(fact.confidence * 100)}% confidence</small>
            <p>{fact.content}</p>
            <small>{fact.evidence.quote ? `“${fact.evidence.quote}”` : fact.interaction_id ? "source linked" : "written by you"}</small>
          </div>
          <RowActions busy={c.busy === fact.id} onEdit={() => setEditing({ id: fact.id, draft: { kind: fact.kind, content: fact.content, status: fact.status } })} onDelete={() => window.confirm("Delete this fact?") && void c.run(fact.id, () => deleteCustomerFact(fact.id))} />
        </article>
      ))}
      {!visible.length ? <Empty>{detail.facts.length ? "No facts of that kind yet." : "No facts saved yet. Add one, or analyze a captured note."}</Empty> : null}
    </section>
  );
}

// ── People ─────────────────────────────────────────────────────────────────

type PersonDraft = { name: string; role: string; organization: string };
const EMPTY_PERSON: PersonDraft = { name: "", role: "", organization: "" };

function PersonForm({ draft, onChange, onCancel, onSave, saving, label }: { draft: PersonDraft; onChange: (draft: PersonDraft) => void; onCancel: () => void; onSave: () => void; saving: boolean; label?: string }) {
  return (
    <div className="records-form is-inline">
      <input value={draft.name} onChange={(event) => onChange({ ...draft, name: event.target.value })} placeholder="Name" aria-label="Contact name" />
      <input value={draft.role} onChange={(event) => onChange({ ...draft, role: event.target.value })} placeholder="Role" aria-label="Contact role" />
      <input value={draft.organization} onChange={(event) => onChange({ ...draft, organization: event.target.value })} placeholder="Organization" aria-label="Contact organization" />
      <FormActions onCancel={onCancel} onSave={onSave} canSave={Boolean(draft.name.trim())} saving={saving} label={label} />
    </div>
  );
}

export function People({ c, detail }: Props) {
  const [draft, setDraft] = useState<PersonDraft | null>(null);
  const [editing, setEditing] = useState<{ id: string; draft: PersonDraft } | null>(null);
  const clean = (person: PersonDraft) => ({ name: person.name.trim(), role: person.role.trim(), organization: person.organization.trim() });
  return (
    <section className="records">
      <ListHead text={detail.people.length ? `${detail.people.length} ${detail.people.length === 1 ? "contact" : "contacts"}.` : "No people captured yet."} adding={Boolean(draft)} onToggle={() => setDraft(draft ? null : EMPTY_PERSON)} label="Add contact" />
      {draft ? <PersonForm draft={draft} onChange={setDraft} onCancel={() => setDraft(null)} saving={c.busy === "new-person"} label="Add" onSave={() => void c.run("new-person", () => addCustomerPerson(detail.account.id, clean(draft))).then((ok) => ok && setDraft(null))} /> : null}
      <div className="records-grid">
        {detail.people.map((person: CustomerPerson) => editing?.id === person.id ? (
          <PersonForm key={person.id} draft={editing.draft} onChange={(next) => setEditing({ id: person.id, draft: next })} onCancel={() => setEditing(null)} saving={c.busy === person.id} onSave={() => void c.run(person.id, () => updateCustomerPerson(person.id, clean(editing.draft))).then((ok) => ok && setEditing(null))} />
        ) : (
          <article key={person.id} className="records-row is-person">
            <i aria-hidden="true">{person.name.slice(0, 1).toUpperCase()}</i>
            <div><strong>{person.name}</strong><small>{person.role || "Role not captured"}{person.organization ? ` · ${person.organization}` : ""}</small></div>
            <RowActions busy={c.busy === person.id} onEdit={() => setEditing({ id: person.id, draft: { name: person.name, role: person.role, organization: person.organization } })} onDelete={() => window.confirm("Remove this contact?") && void c.run(person.id, () => deleteCustomerPerson(person.id))} />
          </article>
        ))}
      </div>
    </section>
  );
}

// ── Sources ────────────────────────────────────────────────────────────────

type SourceDraft = { title: string; content: string; source_kind: string };
const SOURCE_STATUS: Record<CustomerSource["status"], string> = { waiting: "Waiting for analysis", review: "Ready to review", saved: "Analyzed", duplicate: "Duplicate" };

export function Sources({ c, detail }: Props) {
  const [editing, setEditing] = useState<{ id: string; draft: SourceDraft } | null>(null);
  return (
    <section className="records">
      {detail.sources.map((source) => editing?.id === source.id ? (
        <div key={source.id} id={`source-${source.id}`} className="records-form">
          <div className="records-form-row">
            <input value={editing.draft.title} onChange={(event) => setEditing({ id: source.id, draft: { ...editing.draft, title: event.target.value } })} aria-label="Note title" />
            <select value={editing.draft.source_kind} onChange={(event) => setEditing({ id: source.id, draft: { ...editing.draft, source_kind: event.target.value } })} aria-label="Note type">{SOURCE_KINDS.map((kind) => <option key={kind} value={kind}>{kind}</option>)}</select>
          </div>
          <textarea value={editing.draft.content} onChange={(event) => setEditing({ id: source.id, draft: { ...editing.draft, content: event.target.value } })} aria-label="Note content" />
          <FormActions onCancel={() => setEditing(null)} canSave={Boolean(editing.draft.title.trim() && editing.draft.content.trim())} saving={c.busy === source.id} onSave={() => void c.run(source.id, () => updateCustomerSource(source.id, { title: editing.draft.title.trim(), content: editing.draft.content.trim(), source_kind: editing.draft.source_kind as CustomerSource["source_kind"] }), "Note corrected. Facts and actions already saved from it are unchanged.").then((ok) => ok && setEditing(null))} />
        </div>
      ) : (
        <article key={source.id} id={`source-${source.id}`} className="records-row is-source">
          <div>
            <small>{source.source_kind} · {when(source.occurred_at || source.created_at)} · <em className={`source-${source.status}`}>{SOURCE_STATUS[source.status]}</em></small>
            <strong>{source.title}</strong>
            <p>{source.content}</p>
          </div>
          <RowActions busy={c.busy === source.id} onEdit={() => setEditing({ id: source.id, draft: { title: source.title, content: source.content, source_kind: source.source_kind } })} onDelete={() => window.confirm("Delete this captured note? It leaves the timeline with it; facts and actions already saved from it remain.") && void c.run(source.id, () => deleteCustomerSource(source.id))}>
            {source.status === "waiting" ? <button type="button" className="ui-btn is-sm" disabled={!c.modelReady || c.busy === source.id} onClick={() => void c.openProposal(source.id, true)}>{c.modelReady ? (c.busy === source.id ? "Analyzing…" : "Analyze") : "Launch model to analyze"}</button>
              : source.status === "review" ? <button type="button" className="ui-btn is-sm" disabled={c.busy === source.id} onClick={() => void c.openProposal(source.id, false)}>{c.busy === source.id ? "Opening…" : "Review update"}</button> : null}
          </RowActions>
        </article>
      ))}
      {!detail.sources.length ? <Empty>Capture a note to start the source trail.</Empty> : null}
    </section>
  );
}

// ── Timeline and outputs ───────────────────────────────────────────────────

export function Timeline({ detail }: { detail: CustomerAccountDetail }) {
  return (
    <section className="records">
      {detail.interactions.map((item) => (
        <article key={item.id} className="records-row is-event"><time>{when(item.occurred_at)}</time><div><strong>{item.title}</strong><p>{item.summary}</p></div></article>
      ))}
      {!detail.interactions.length ? <Empty>No saved interactions yet.</Empty> : null}
    </section>
  );
}

export function Outputs({ c, detail }: Props) {
  const [output, setOutput] = useState<CustomerOutput | null>(null);
  const copyAndOpen = () => {
    if (!output) return;
    void navigator.clipboard.writeText(output.content);
    const target = output.tracker_url || c.settings.tracker_url;
    if (target) window.open(target, "_blank", "noopener,noreferrer");
    c.toast(target ? "Markdown copied. Paste it into the tracker tab." : "Markdown copied. Add the tracker URL to open it at the same time.");
  };
  return (
    <section className="records">
      <div className="records-form is-inline">
        <input value={c.settings.tracker_url} onChange={(event) => c.setSettings({ ...c.settings, tracker_url: event.target.value })} placeholder="https://company.example/activity" aria-label="Activity tracker URL" />
        <button type="button" className="ui-btn is-sm" disabled={c.busy === "settings"} onClick={() => void c.run("settings", async () => c.setSettings(await saveCustomerSettings(c.settings)), "Tracker link saved.")}>Save link</button>
        <button type="button" className="ui-btn is-primary is-sm" disabled={c.busy === "output" || !detail.interactions.length} onClick={() => void c.run("output", async () => setOutput(await createCustomerOutput(detail.account.id, detail.interactions[0]?.id)))}>{c.busy === "output" ? "Building…" : "Generate tracker update"}</button>
      </div>
      {output ? (
        <article className="records-output">
          <header><strong>Activity tracker update</strong><button type="button" className="ui-btn is-primary is-sm" onClick={copyAndOpen}>Copy &amp; open tracker</button></header>
          <pre>{output.content}</pre>
        </article>
      ) : null}
    </section>
  );
}
