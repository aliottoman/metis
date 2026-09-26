"use client";

// The verbatim record. Words light up as the audio plays; clicking one seeks
// to it; a paragraph can be corrected without losing what was heard.

import { useEffect, useRef, useState } from "react";
import { Search } from "lucide-react";

import { clock } from "@/lib/audio";
import type { MeetingTurn } from "@/lib/types";

export function MeetingTranscript({ turns, names, position, sourceTurnId, onSeek, onCorrect }: {
  turns: MeetingTurn[];
  names: Map<string, string>;
  position: number;
  sourceTurnId?: string | null;
  onSeek: (seconds: number | null) => void;
  onCorrect: (turn: MeetingTurn, text: string) => Promise<boolean>;
}) {
  const [query, setQuery] = useState("");
  const [editing, setEditing] = useState<{ id: string; draft: string } | null>(null);
  const [saving, setSaving] = useState(false);
  const sourceRef = useRef<HTMLLIElement>(null);
  useEffect(() => {
    if (!sourceTurnId) return;
    setQuery("");
    const frame = requestAnimationFrame(() => {
      sourceRef.current?.scrollIntoView({ block: "center", behavior: "auto" });
      sourceRef.current?.focus({ preventScroll: true });
    });
    return () => cancelAnimationFrame(frame);
  }, [sourceTurnId]);
  const needle = query.trim().toLowerCase();
  const visible = turns.filter((turn) => !needle || `${names.get(turn.speaker_id) || turn.speaker_id} ${turn.text}`.toLowerCase().includes(needle));

  const save = async (turn: MeetingTurn) => {
    const text = editing?.draft.trim() ?? "";
    if (!text || saving) return;
    if (text === turn.text) { setEditing(null); return; }
    setSaving(true);
    try {
      if (await onCorrect(turn, text)) setEditing(null);
    } finally {
      setSaving(false);
    }
  };

  return (
    <section className="meeting-transcript" aria-label="Transcript">
      <div className="meeting-transcript-head">
        <div><h3>The conversation, in their words</h3><small>Select a timestamp or word to listen from that moment.</small></div>
        <label className="conversation-search"><Search size={16} aria-hidden="true" /><input type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Find a moment…" aria-label="Search transcript" /></label>
      </div>
      {needle ? <small className="meeting-search-count" role="status">{visible.length} of {turns.length} moments match</small> : null}
      <ol className="meeting-turns">
        {visible.map((turn) => {
          const speaker = names.get(turn.speaker_id) || turn.speaker_id;
          return (
            <li key={turn.id} className={`tone-${turn.ordinal % 4}`} ref={turn.id === sourceTurnId ? sourceRef : undefined} tabIndex={turn.id === sourceTurnId ? -1 : undefined} aria-current={turn.id === sourceTurnId ? "location" : undefined}>
              <header>
                <strong>{speaker}</strong>
                <button type="button" className="meeting-time" onClick={() => onSeek(turn.start_seconds)}>{clock(turn.start_seconds)}</button>
                {turn.corrected_at ? <span className="ui-chip">corrected</span> : null}
                <button type="button" className="ui-btn is-quiet is-sm" disabled={saving} onClick={() => setEditing({ id: turn.id, draft: turn.text })}>Correct</button>
              </header>
              {editing?.id === turn.id ? (
                <div className="meeting-correct">
                  <textarea autoFocus disabled={saving} value={editing.draft} onChange={(event) => setEditing({ id: turn.id, draft: event.target.value })} onKeyDown={(event) => { if (event.nativeEvent.isComposing) return; if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) { event.preventDefault(); void save(turn); } if (event.key === "Escape") setEditing(null); }} aria-label="Corrected text" />
                  <span><button type="button" className="ui-btn is-quiet is-sm" disabled={saving} onClick={() => setEditing(null)}>Cancel</button><button type="button" className="ui-btn is-primary is-sm" disabled={saving || !editing.draft.trim()} onClick={() => void save(turn)}>{saving ? "Saving…" : "Save"}</button></span>
                </div>
              ) : (
                <p>
                  {turn.words.length ? turn.words.map((word, index) => (
                    <button type="button" key={index} className={word.start !== null && word.end !== null && position >= word.start && position <= word.end ? "is-spoken" : ""} onClick={() => onSeek(word.start)}>{word.text} </button>
                  )) : turn.text}
                </p>
              )}
              {turn.corrected_at ? <details className="meeting-original"><summary>What was heard</summary><p>{turn.original_text}</p></details> : null}
            </li>
          );
        })}
      </ol>
      {!visible.length ? <div className="conversation-empty is-compact"><h3>{turns.length ? "No matching moments" : "The transcript is on its way"}</h3><p>{turns.length ? "Try a shorter phrase or a speaker's name." : "The speaker transcript will appear here when processing finishes."}</p>{turns.length ? <button type="button" className="ui-btn is-sm" onClick={() => setQuery("")}>Clear search</button> : null}</div> : null}
    </section>
  );
}
