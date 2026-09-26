"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { BookOpen, Brain, Check, ChevronDown, FileText, Plus, RefreshCw, Search, ShieldCheck, Sparkles, X } from "lucide-react";

import { MarkdownContent } from "@/components/markdown-content";
import { SelectMenu } from "@/components/select-menu";
import { Notice } from "@/components/ui/notice";
import { PageHeader } from "@/components/ui/page-header";
import { Skeleton } from "@/components/ui/skeleton";

import {
  createMemoryProposal,
  decideMemoryProposal,
  getMemoryIndexStatus,
  listMemoryProposals,
  setMemoryIndexConsent,
} from "@/lib/api";
import type { MemoryIndexStatus, MemoryProposal } from "@/lib/types";

type MemoryFilter = "pending" | "approved" | "rejected" | "all";
const MEMORY_KINDS = { user: "Personal preference", project: "Project fact", skill: "Working rule" };

export function MemoryCenter() {
  const [proposals, setProposals] = useState<MemoryProposal[]>([]);
  const [filter, setFilter] = useState<MemoryFilter>("pending");
  const [loading, setLoading] = useState(true);
  const [loaded, setLoaded] = useState(false);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [draftKind, setDraftKind] = useState<"user" | "project" | "skill">("project");
  const [savingDraft, setSavingDraft] = useState(false);
  const [index, setIndex] = useState<MemoryIndexStatus | null>(null);
  const [indexBusy, setIndexBusy] = useState(false);
  const [query, setQuery] = useState("");
  const [kindFilter, setKindFilter] = useState("all");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [composerOpen, setComposerOpen] = useState(false);
  const draftInput = useRef<HTMLTextAreaElement>(null);
  const newMemoryButton = useRef<HTMLButtonElement>(null);
  const collection = useRef<HTMLDivElement>(null);
  const restoreReviewFocus = useRef(false);
  const decisionBusy = useRef(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [items, status] = await Promise.all([
        listMemoryProposals(),
        getMemoryIndexStatus().catch(() => null),
      ]);
      setProposals(items);
      setIndex(status);
      setLoaded(true);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Could not load memory proposals.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => void load(), [load]);

  const visible = useMemo(
    () => proposals.filter((proposal) => (filter === "all" || proposal.status === filter) && (kindFilter === "all" || proposal.kind === kindFilter) && `${proposal.content} ${proposal.rationale ?? ""}`.toLowerCase().includes(query.trim().toLowerCase())),
    [filter, kindFilter, proposals, query],
  );
  const selected = visible.find((proposal) => proposal.id === selectedId) ?? visible[0] ?? null;
  const pendingCount = proposals.filter((proposal) => proposal.status === "pending").length;
  const approvedCount = proposals.filter((proposal) => proposal.status === "approved").length;
  const openComposer = () => { setComposerOpen(true); window.setTimeout(() => draftInput.current?.focus(), 0); };
  useEffect(() => {
    if (!restoreReviewFocus.current) return;
    restoreReviewFocus.current = false;
    const next = collection.current?.querySelector<HTMLButtonElement>('.reference-note-row[aria-pressed="true"]');
    (next ?? newMemoryButton.current)?.focus();
  }, [proposals, selected?.id]);

  async function decide(proposal: MemoryProposal, decision: "approve" | "reject") {
    if (decisionBusy.current) return;
    decisionBusy.current = true;
    setBusyId(proposal.id);
    setError(null);
    try {
      await decideMemoryProposal(proposal.id, decision);
      restoreReviewFocus.current = true;
      setProposals((current) => current.map((item) => item.id === proposal.id ? { ...item, status: decision === "approve" ? "approved" : "rejected" } : item));
    } catch (actionError) {
      setError(actionError instanceof Error ? actionError.message : "Could not update this memory proposal.");
    } finally {
      decisionBusy.current = false;
      setBusyId(null);
    }
  }

  async function proposeMemory() {
    const content = draft.trim();
    if (!content || savingDraft) return;
    setSavingDraft(true);
    setError(null);
    try {
      const proposal = await createMemoryProposal(draftKind, content);
      restoreReviewFocus.current = true;
      setProposals((current) => [proposal, ...current]);
      setDraft("");
      setFilter("pending");
      setQuery("");
      setKindFilter("all");
      setSelectedId(proposal.id);
      setComposerOpen(false);
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "Could not create the memory proposal.");
    } finally {
      setSavingDraft(false);
    }
  }

  async function toggleSemantic(consent: boolean) {
    setIndexBusy(true);
    setError(null);
    try {
      setIndex(await setMemoryIndexConsent(consent));
    } catch (toggleError) {
      setError(toggleError instanceof Error ? toggleError.message : "Could not change memory search.");
    } finally {
      setIndexBusy(false);
    }
  }

  // Consent alone does not mean semantic search is running: the cloud path may
  // be unconfigured or nothing embedded yet. Say which of those it is.
  const indexDetail = !index?.consent
    ? "Memories are matched by keyword, so a question phrased differently than the memory can miss it. Turning this on embeds your approved memories with Cohere so paraphrases match."
    : index.semantic
      ? `Searching ${index.embedded.toLocaleString()} of ${index.active.toLocaleString()} approved memories by meaning.`
      : index.cloudAvailable
        ? "Consent granted. Nothing is embedded yet — approve a memory, or refresh in a moment."
        : "Consent granted, but cloud retrieval is unavailable, so search has fallen back to keywords.";

  return (
    <div className="workspacePage memoryPage reference-workspace memory-workspace">
      <PageHeader
        icon={<Brain size={23} aria-hidden="true" />}
        eyebrow="The context you choose to keep"
        title="Memory"
        lede="A collection of useful facts, preferences, and working rules. Review a note once, carry it into future conversations."
        actions={<><button className="ui-btn" type="button" onClick={() => void load()} disabled={loading || busyId !== null || savingDraft || indexBusy}><RefreshCw size={14} aria-hidden="true" />{loading ? "Refreshing…" : "Refresh"}</button><button ref={newMemoryButton} className="ui-btn is-primary" type="button" onClick={openComposer}><Plus size={15} aria-hidden="true" />New memory</button></>}
      />

      {error ? <Notice kind="error" title={loaded ? "The change could not be completed" : "Memory unavailable"} action={!loaded ? "Try again" : undefined} onAction={() => void load()} onDismiss={loaded ? () => setError(null) : undefined}><p>{error}</p></Notice> : null}

      <div className="reference-summary" aria-label="Memory overview">
        <div><BookOpen size={17} aria-hidden="true" /><span><strong>{loaded ? approvedCount : "—"}</strong> remembered</span></div>
        <div><Sparkles size={17} aria-hidden="true" /><span><strong>{loaded ? pendingCount : "—"}</strong> awaiting review</span></div>
        <div><ShieldCheck size={17} aria-hidden="true" /><span>Only <strong>approved notes</strong> are used</span></div>
      </div>

      {composerOpen ? <section className="memoryComposer reference-composer" aria-labelledby="memory-composer-title">
        <div className="reference-composer-heading"><span className="reference-source-mark"><FileText size={20} aria-hidden="true" /></span><div><span className="reference-kicker">A little context goes a long way</span><h2 id="memory-composer-title">Write a new memory</h2><p>Keep one durable fact or preference. You’ll review it before it becomes part of future conversations.</p></div><button className="ui-btn is-quiet is-sm" type="button" disabled={savingDraft} aria-label="Close new memory editor" onClick={() => { setComposerOpen(false); newMemoryButton.current?.focus(); }}><X size={16} aria-hidden="true" /></button></div>
        <textarea ref={draftInput} aria-label="Memory to propose" value={draft} onChange={(event) => setDraft(event.target.value)} maxLength={20000} disabled={savingDraft} placeholder="For example: Keep project updates concise, with decisions and next steps first." />
        <div className="memoryComposerActions">
          <SelectMenu className="memoryKindSelect" label="Kind" value={draftKind} disabled={savingDraft} onChange={(value) => setDraftKind(value as "user" | "project" | "skill")} options={Object.entries(MEMORY_KINDS).map(([value, label]) => ({ value, label }))} />
          <span>{draft.trim().length.toLocaleString()} / 20,000</span>
          <button className="ui-btn is-primary" type="button" disabled={draft.trim().length < 3 || savingDraft} onClick={() => void proposeMemory()}>{savingDraft ? "Creating…" : "Create for review"}</button>
        </div>
      </section> : draft.trim() ? <button className="reference-draft-return" type="button" onClick={openComposer}><FileText size={15} aria-hidden="true" />You have an unfinished note<span>Continue writing</span></button> : null}

      <div className="reference-collection-heading"><div><span className="reference-kicker">Your collection</span><h2>{filter === "pending" ? "Ready for your review" : filter === "approved" ? "Notes Metis remembers" : filter === "rejected" ? "Notes you passed on" : "All memory notes"}</h2></div><span>{visible.length} {visible.length === 1 ? "note" : "notes"}</span></div>
      <div className="reference-memory-toolbar">
        <div className="filterTabs reference-memory-tabs" role="tablist" aria-label="Memory status filter" onKeyDown={(event) => {
          const filters: MemoryFilter[] = ["pending", "approved", "rejected", "all"];
          const current = filters.indexOf(filter);
          const next = event.key === "ArrowRight" ? (current + 1) % filters.length : event.key === "ArrowLeft" ? (current + filters.length - 1) % filters.length : event.key === "Home" ? 0 : event.key === "End" ? filters.length - 1 : -1;
          if (next < 0) return;
          event.preventDefault();
          setFilter(filters[next]!);
          event.currentTarget.querySelector<HTMLButtonElement>(`#memory-tab-${filters[next]}`)?.focus();
        }}>
          {(["pending", "approved", "rejected", "all"] as MemoryFilter[]).map((item) => <button key={item} id={`memory-tab-${item}`} role="tab" type="button" aria-selected={filter === item} aria-controls="memory-proposals" tabIndex={filter === item ? 0 : -1} onClick={() => setFilter(item)}>{item === "pending" ? "To review" : item === "approved" ? "Remembered" : item === "rejected" ? "Passed on" : "All notes"}<span>{item === "all" ? proposals.length : proposals.filter((proposal) => proposal.status === item).length}</span></button>)}
        </div>
        <div className="reference-tools"><label className="reference-search"><Search size={16} aria-hidden="true" /><input value={query} onChange={(event) => setQuery(event.target.value)} aria-label="Search memory notes" placeholder="Find a fact or preference…" type="search" /></label><SelectMenu className="reference-kind-filter" label="Memory type" hideLabel value={kindFilter} onChange={setKindFilter} options={[{ value: "all", label: "All types" }, ...Array.from(new Set([...Object.keys(MEMORY_KINDS), ...proposals.map((proposal) => proposal.kind)])).map((value) => ({ value, label: MEMORY_KINDS[value as keyof typeof MEMORY_KINDS] ?? (value === "conversation" ? "Conversation note" : value) }))]} /></div>
      </div>

      <div ref={collection} id="memory-proposals" className="reference-memory-collection" role="tabpanel" aria-labelledby={`memory-tab-${filter}`} aria-busy={loading} tabIndex={0}>
        {loading && !proposals.length ? <Skeleton rows={3} height={110} /> : null}
        {!loading && loaded && !visible.length ? <div className="reference-empty"><BookOpen size={28} aria-hidden="true" /><h3>{query.trim() || kindFilter !== "all" ? "No notes match these filters" : filter === "approved" ? "Your collection starts with a review" : filter === "rejected" ? "Nothing passed on" : "Room for your next useful note"}</h3><p>{query.trim() || kindFilter !== "all" ? "Try another phrase or include all memory types." : filter === "approved" ? "Approve a note from To review to use it in future conversations." : filter === "rejected" ? "Notes you choose not to keep will remain here for reference." : "Write a preference or save a useful correction from a conversation."}</p>{query.trim() || kindFilter !== "all" ? <button className="ui-btn is-sm" type="button" onClick={() => { setQuery(""); setKindFilter("all"); }}>Clear filters</button> : <button className="ui-btn is-sm" type="button" onClick={filter === "approved" && pendingCount ? () => setFilter("pending") : openComposer}><Plus size={13} aria-hidden="true" />{filter === "approved" && pendingCount ? "Review pending notes" : "Write a memory"}</button>}</div> : null}
        {selected ? <div className="reference-memory-layout">
          <div className="reference-note-list" aria-label="Memory notes">
            {visible.map((proposal) => <button key={proposal.id} type="button" className={`reference-note-row${selected.id === proposal.id ? " is-selected" : ""}`} aria-pressed={selected.id === proposal.id} aria-controls="memory-note-reader" onClick={() => setSelectedId(proposal.id)}><span className="reference-note-row-top"><span className="reference-kicker">{MEMORY_KINDS[proposal.kind as keyof typeof MEMORY_KINDS] ?? proposal.kind}</span>{proposal.status === "approved" ? <Check size={13} aria-label="Remembered" /> : <span className={`reference-note-state is-${proposal.status}`} aria-label={proposal.status} />}</span><strong>{proposal.content}</strong><span className="reference-note-date">{proposal.created_at ? new Date(proposal.created_at).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" }) : "Recently proposed"}</span></button>)}
          </div>
          <article id="memory-note-reader" className="reference-note-reader" aria-label="Selected memory note">
            <header><span className="reference-source-mark"><FileText size={20} aria-hidden="true" /></span><div><span className="reference-kicker">{MEMORY_KINDS[selected.kind as keyof typeof MEMORY_KINDS] ?? selected.kind}</span><h3>{selected.status === "pending" ? "A note to remember" : selected.status === "approved" ? "Part of your memory" : "Kept for reference"}</h3></div><span className={`statusPill status-${selected.status}`}>{selected.status === "approved" ? "Remembered" : selected.status === "pending" ? "For review" : "Passed on"}</span></header>
            <div className="reference-note-body"><MarkdownContent content={selected.content} /></div>
            {selected.rationale ? <div className="reference-note-rationale"><Sparkles size={15} aria-hidden="true" /><div><strong>Why this may help</strong><p>{selected.rationale}</p></div></div> : null}
            <dl className="reference-note-facts"><div><dt>Proposed</dt><dd>{selected.created_at ? new Date(selected.created_at).toLocaleDateString() : "Recently"}</dd></div><div><dt>Confidence</dt><dd>{selected.confidence == null ? "Not scored" : `${Math.round(selected.confidence * 100)}%`}</dd></div><div><dt>Source</dt><dd title={selected.source_run_id ?? undefined}>{selected.source_run_id ? "Conversation" : "Not recorded"}</dd></div></dl>
            {selected.status === "pending" ? <footer className="reference-note-actions"><p>This note becomes available only after you approve it.</p><div><button className="ui-btn is-quiet" type="button" disabled={busyId !== null} onClick={() => void decide(selected, "reject")}>Pass on this</button><button className="ui-btn is-primary" type="button" disabled={busyId !== null} onClick={() => void decide(selected, "approve")}><Check size={14} aria-hidden="true" />{busyId === selected.id ? "Saving…" : "Remember this"}</button></div></footer> : <footer className="reference-note-actions"><p><ShieldCheck size={14} aria-hidden="true" />{selected.status === "approved" ? "Available to help with future conversations." : "This note is not used in future conversations."}</p></footer>}
          </article>
        </div> : null}
      </div>

      <section className="reference-memory-settings" aria-labelledby="memory-search-title"><div className="reference-memory-settings-heading"><span className="reference-source-mark"><Search size={18} aria-hidden="true" /></span><div><h2 id="memory-search-title">How memory is found</h2><p>{index ? index.semantic ? "By meaning and context" : "By matching keywords" : "Search settings are unavailable right now."}</p></div><span className="ui-chip">{index ? index.semantic ? "Semantic search" : "Keyword search" : "Unavailable"}</span></div>{index ? <details><summary>Search preferences and consent<ChevronDown size={14} aria-hidden="true" /></summary><div className="reference-consent"><p>{indexDetail}</p><p>Consent is separate from your source library and can be withdrawn. Withdrawing deletes stored memory vectors. Only the memory text being embedded leaves this machine.</p><button className={`ui-btn is-sm${index.consent ? " is-danger" : ""}`} type="button" disabled={indexBusy} onClick={() => void toggleSemantic(!index.consent)}>{indexBusy ? "Working…" : index.consent ? "Withdraw consent and purge vectors" : "Embed my memories for semantic search"}</button></div></details> : <button className="ui-btn is-sm" type="button" disabled={loading || indexBusy} onClick={() => void load()}>Try again</button>}</section>
    </div>
  );
}
