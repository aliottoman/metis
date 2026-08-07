"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import {
  decideAnswer,
  getAnswerConflicts,
  getAnswerEntities,
  getAnswers,
} from "@/lib/api";
import type { AnswerAtom, AnswerEntity } from "@/lib/types";

type Tab = "pending" | "active" | "superseded";

const TAB_LABEL: Record<Tab, string> = {
  pending: "To review",
  active: "In the bank",
  superseded: "Retired",
};

function parseList(raw: string): string[] {
  try {
    const value = JSON.parse(raw || "[]");
    return Array.isArray(value) ? value.map(String) : [];
  } catch {
    return [];
  }
}

export function AnswerBank() {
  const [tab, setTab] = useState<Tab>("pending");
  const [atoms, setAtoms] = useState<AnswerAtom[]>([]);
  const [entities, setEntities] = useState<AnswerEntity[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  // Conflicts are fetched per atom, only while it is being reviewed: the
  // question "what would this replace?" is only worth asking about the one
  // you are deciding on.
  const [conflicts, setConflicts] = useState<Record<string, AnswerAtom[]>>({});
  const [replacing, setReplacing] = useState<Record<string, Set<string>>>({});
  const [entityFilter, setEntityFilter] = useState<string | null>(null);

  const refresh = useCallback(async (which: Tab) => {
    setLoading(true);
    setError(null);
    try {
      const [rows, tags] = await Promise.all([getAnswers(which), getAnswerEntities()]);
      setAtoms(rows);
      setEntities(tags);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Could not read the bank.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh(tab);
  }, [refresh, tab]);

  // Pending atoms are the only ones whose conflicts matter, and they are few.
  useEffect(() => {
    if (tab !== "pending") return;
    let live = true;
    void Promise.all(
      atoms.map(async (atom) => [atom.id, await getAnswerConflicts(atom.id).catch(() => [])] as const),
    ).then((pairs) => {
      if (!live) return;
      const next: Record<string, AnswerAtom[]> = {};
      for (const [id, found] of pairs) next[id] = found as AnswerAtom[];
      setConflicts(next);
    });
    return () => { live = false; };
  }, [atoms, tab]);

  async function decide(atom: AnswerAtom, status: "active" | "rejected" | "superseded") {
    setBusy(atom.id);
    try {
      const supersedes = status === "active" ? [...(replacing[atom.id] ?? [])] : [];
      await decideAnswer(atom.id, status, supersedes);
      await refresh(tab);
    } catch (decideError) {
      setError(decideError instanceof Error ? decideError.message : "Could not apply that.");
    } finally {
      setBusy(null);
    }
  }

  function toggleReplace(atomId: string, targetId: string) {
    setReplacing((current) => {
      const next = new Set(current[atomId] ?? []);
      if (next.has(targetId)) next.delete(targetId);
      else next.add(targetId);
      return { ...current, [atomId]: next };
    });
  }

  const visible = useMemo(
    () =>
      entityFilter
        ? atoms.filter((atom) => parseList(atom.entities_json).some(
            (entity) => entity.toLowerCase() === entityFilter,
          ))
        : atoms,
    [atoms, entityFilter],
  );

  return (
    <div className="workspacePage answerPage">
      <header className="pageHeader">
        <div>
          <span className="eyebrow">Answer bank</span>
          <h1>What you have already worked out</h1>
          <p>
            Answers you have given once, kept with the evidence that made them
            defensible, so the next person who asks gets your wording rather than a
            fresh guess.
          </p>
        </div>
        <button className="secondaryButton" type="button" onClick={() => void refresh(tab)} disabled={loading}>
          {loading ? "Reading…" : "Refresh"}
        </button>
      </header>

      {error ? <div className="composerError" role="alert"><span>!</span><p>{error}</p></div> : null}

      <div className="answerTabs" role="tablist">
        {(Object.keys(TAB_LABEL) as Tab[]).map((name) => (
          <button
            key={name}
            role="tab"
            type="button"
            aria-selected={tab === name}
            className={tab === name ? "selected" : ""}
            onClick={() => { setTab(name); setEntityFilter(null); }}
          >
            {TAB_LABEL[name]}
          </button>
        ))}
      </div>

      {/* What the bank knows about. Entities are the third recall arm, so
          showing them is also showing why a question finds what it finds. */}
      {entities.length ? (
        <div className="answerEntities" aria-label="What the bank knows about">
          {entities.slice(0, 18).map((tag) => (
            <button
              key={tag.entity}
              type="button"
              className={entityFilter === tag.entity ? "selected" : ""}
              onClick={() => setEntityFilter(entityFilter === tag.entity ? null : tag.entity)}
            >
              {tag.entity}<b>{tag.atoms}</b>
            </button>
          ))}
        </div>
      ) : null}

      {!loading && !visible.length ? (
        <section className="settingsSection">
          <p className="sectionLede">
            {tab === "pending"
              ? "Nothing waiting. Answers are offered here after a run that cited its sources — deliberately rarely, so this stays worth opening."
              : tab === "active"
                ? "The bank is empty. Answer something well with sources behind it, and it will be offered here."
                : "Nothing has been retired yet. When a newer answer replaces one, the old one lands here with a pointer to its replacement."}
          </p>
        </section>
      ) : null}

      {visible.map((atom) => {
        const paraphrases = parseList(atom.paraphrases_json);
        const citations = parseList(atom.citations_json);
        const tags = parseList(atom.entities_json);
        const clashes = conflicts[atom.id] ?? [];
        const chosen = replacing[atom.id] ?? new Set<string>();
        return (
          <article className="answerCard" key={atom.id}>
            <div className="answerHead">
              <h2>{atom.question}</h2>
              {atom.status !== "pending" ? (
                <span className={`answerStatus is-${atom.status}`}>{atom.status}</span>
              ) : null}
            </div>
            <p className="answerBody">{atom.answer}</p>

            {paraphrases.length ? (
              <p className="answerMeta">
                <span>Also asked as</span> {paraphrases.join(" · ")}
              </p>
            ) : null}
            {tags.length ? (
              <p className="answerMeta"><span>About</span> {tags.join(" · ")}</p>
            ) : null}
            {citations.length ? (
              <p className="answerMeta"><span>Grounded in</span> {citations.join(" · ")}</p>
            ) : null}

            {/* Supersession is decided here, not at retrieval: two active
                answers that disagree is a bank nobody can trust. */}
            {tab === "pending" && clashes.length ? (
              <div className="answerConflicts">
                <strong>This may replace what you already keep:</strong>
                {clashes.map((clash) => (
                  <label key={clash.id}>
                    <input
                      type="checkbox"
                      checked={chosen.has(clash.id)}
                      onChange={() => toggleReplace(atom.id, clash.id)}
                    />
                    <span>{clash.question}</span>
                  </label>
                ))}
                <small>
                  Ticked answers are retired when you keep this one, and keep a
                  pointer to it.
                </small>
              </div>
            ) : null}

            <div className="answerActions">
              {atom.status === "pending" ? (
                <>
                  <button className="primaryButton" type="button" disabled={busy === atom.id} onClick={() => void decide(atom, "active")}>
                    {chosen.size ? `Keep, retire ${chosen.size}` : "Keep"}
                  </button>
                  <button className="secondaryButton" type="button" disabled={busy === atom.id} onClick={() => void decide(atom, "rejected")}>
                    Discard
                  </button>
                </>
              ) : atom.status === "active" ? (
                <button className="secondaryButton" type="button" disabled={busy === atom.id} onClick={() => void decide(atom, "superseded")}>
                  Retire
                </button>
              ) : (
                <button className="secondaryButton" type="button" disabled={busy === atom.id} onClick={() => void decide(atom, "active")}>
                  Bring back
                </button>
              )}
            </div>
          </article>
        );
      })}
    </div>
  );
}
