"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import { createMemoryProposal, decideMemoryProposal, listMemoryProposals } from "@/lib/api";
import type { MemoryProposal } from "@/lib/types";

type MemoryFilter = "pending" | "approved" | "rejected" | "all";

export function MemoryCenter() {
  const [proposals, setProposals] = useState<MemoryProposal[]>([]);
  const [filter, setFilter] = useState<MemoryFilter>("pending");
  const [loading, setLoading] = useState(true);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [draftKind, setDraftKind] = useState<"user" | "project" | "skill">("project");
  const [savingDraft, setSavingDraft] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setProposals(await listMemoryProposals());
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Could not load memory proposals.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => void load(), [load]);

  const visible = useMemo(
    () => filter === "all" ? proposals : proposals.filter((proposal) => proposal.status === filter),
    [filter, proposals],
  );

  async function decide(proposal: MemoryProposal, decision: "approve" | "reject") {
    setBusyId(proposal.id);
    setError(null);
    try {
      await decideMemoryProposal(proposal.id, decision);
      setProposals((current) => current.map((item) => item.id === proposal.id ? { ...item, status: decision === "approve" ? "approved" : "rejected" } : item));
    } catch (actionError) {
      setError(actionError instanceof Error ? actionError.message : "Could not update this memory proposal.");
    } finally {
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
      setProposals((current) => [proposal, ...current]);
      setDraft("");
      setFilter("pending");
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "Could not create the memory proposal.");
    } finally {
      setSavingDraft(false);
    }
  }

  return (
    <div className="workspacePage memoryPage">
      <header className="pageHeader">
        <div>
          <span className="eyebrow">You stay in control</span>
          <h1>Memory</h1>
          <p>Nothing becomes durable knowledge until you review it. Source runs remain attached for provenance.</p>
        </div>
        <button className="secondaryButton" type="button" onClick={() => void load()} disabled={loading}>Refresh proposals</button>
      </header>

      <section className="memoryComposer" aria-labelledby="memory-composer-title">
        <div>
          <span className="eyebrow">Save context deliberately</span>
          <h2 id="memory-composer-title">Propose a memory</h2>
          <p>Write the durable fact or preference—not the entire conversation. It stays pending until you approve it below.</p>
        </div>
        <textarea value={draft} onChange={(event) => setDraft(event.target.value)} maxLength={20000} placeholder="For example: Use Chicago as the default OCI Generative AI region for this project." />
        <div className="memoryComposerActions">
          <label>Kind<select value={draftKind} onChange={(event) => setDraftKind(event.target.value as "user" | "project" | "skill")}><option value="project">Project fact</option><option value="user">Personal preference</option><option value="skill">Reusable working rule</option></select></label>
          <span>{draft.trim().length.toLocaleString()} / 20,000</span>
          <button className="primaryButton" type="button" disabled={draft.trim().length < 3 || savingDraft} onClick={() => void proposeMemory()}>{savingDraft ? "Proposing…" : "Create proposal"}</button>
        </div>
      </section>

      <div className="memoryPrinciple">
        <span className="principleMark">◎</span>
        <div><strong>Proposal-first learning</strong><p>Metis can notice patterns and suggest useful context, but only approved memories are retrieved in future conversations.</p></div>
      </div>

      <div className="filterTabs" role="tablist" aria-label="Memory status filter">
        {(["pending", "approved", "rejected", "all"] as MemoryFilter[]).map((item) => (
          <button key={item} role="tab" type="button" aria-selected={filter === item} onClick={() => setFilter(item)}>
            {item[0]!.toUpperCase() + item.slice(1)}
            <span>{item === "all" ? proposals.length : proposals.filter((proposal) => proposal.status === item).length}</span>
          </button>
        ))}
      </div>

      {error ? <div className="notice errorNotice" role="alert"><strong>Memory unavailable</strong><span>{error}</span></div> : null}

      <div className="memoryList" aria-live="polite">
        {loading && !proposals.length ? Array.from({ length: 3 }).map((_, index) => <div className="skeletonRow" key={index} />) : null}
        {!loading && !visible.length ? (
          <div className="emptyPanel">
            <span className="emptyGlyph">◎</span>
            <h2>No {filter === "all" ? "memory" : filter} proposals</h2>
            <p>Create one above, or submit an explicit correction from a chat response.</p>
          </div>
        ) : null}
        {visible.map((proposal) => (
          <article className="memoryCard" key={proposal.id}>
            <div className="memoryCardHeader">
              <span className="memoryKind">{proposal.kind}</span>
              <span className={`statusPill status-${proposal.status}`}>{proposal.status}</span>
            </div>
            <blockquote>{proposal.content}</blockquote>
            {proposal.rationale ? <p className="memoryRationale">{proposal.rationale}</p> : null}
            <dl className="memoryMeta">
              <div><dt>Confidence</dt><dd>{proposal.confidence == null ? "Not scored" : `${Math.round(proposal.confidence * 100)}%`}</dd></div>
              <div><dt>Source</dt><dd className="mono">{proposal.source_run_id ? proposal.source_run_id.slice(0, 12) : "System"}</dd></div>
              <div><dt>Proposed</dt><dd>{proposal.created_at ? new Date(proposal.created_at).toLocaleDateString() : "Recently"}</dd></div>
            </dl>
            {proposal.status === "pending" ? (
              <footer className="cardActions">
                <button className="dangerButton" type="button" disabled={busyId === proposal.id} onClick={() => void decide(proposal, "reject")}>Reject</button>
                <button className="primaryButton" type="button" disabled={busyId === proposal.id} onClick={() => void decide(proposal, "approve")}>Remember this</button>
              </footer>
            ) : null}
          </article>
        ))}
      </div>
    </div>
  );
}
