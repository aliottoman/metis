"use client";

// The verbatim record. Words light up as the audio plays; clicking one seeks
// to it; a paragraph can be corrected without losing what was heard.

import { useState } from "react";

import { clock } from "@/lib/audio";
import type { MeetingTurn } from "@/lib/types";

export function MeetingTranscript({ turns, names, position, onSeek, onCorrect }: {
  turns: MeetingTurn[];
  names: Map<string, string>;
  position: number;
  onSeek: (seconds: number | null) => void;
  onCorrect: (turn: MeetingTurn, text: string) => Promise<void>;
}) {
  const [query, setQuery] = useState("");
  const [editing, setEditing] = useState<{ id: string; draft: string } | null>(null);
  const needle = query.trim().toLowerCase();
  const visible = turns.filter((turn) => !needle || `${names.get(turn.speaker_id) || turn.speaker_id} ${turn.text}`.toLowerCase().includes(needle));

  const save = async (turn: MeetingTurn) => {
    const text = editing?.draft.trim() ?? "";
    if (text && text !== turn.text) await onCorrect(turn, text);
    setEditing(null);
  };

  return (
    <section className="meeting-transcript" aria-label="Transcript">
      <div className="meeting-transcript-head">
        <small>Click a word to play from it. Correct a paragraph from its menu.</small>
        <input type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search the transcript" aria-label="Search transcript" />
      </div>
      <ol className="meeting-turns">
        {visible.map((turn) => {
          const speaker = names.get(turn.speaker_id) || turn.speaker_id;
          return (
            <li key={turn.id} className={`tone-${turn.ordinal % 4}`}>
              <header>
                <strong>{speaker}</strong>
                <button type="button" className="meeting-time" onClick={() => onSeek(turn.start_seconds)}>{clock(turn.start_seconds)}</button>
                {turn.corrected_at ? <span className="ui-chip">corrected</span> : null}
                <button type="button" className="ui-btn is-quiet is-sm" onClick={() => setEditing({ id: turn.id, draft: turn.text })}>Correct</button>
              </header>
              {editing?.id === turn.id ? (
                <div className="meeting-correct">
                  <textarea autoFocus value={editing.draft} onChange={(event) => setEditing({ id: turn.id, draft: event.target.value })} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void save(turn); } if (event.key === "Escape") setEditing(null); }} aria-label="Corrected text" />
                  <span><button type="button" className="ui-btn is-quiet is-sm" onClick={() => setEditing(null)}>Cancel</button><button type="button" className="ui-btn is-primary is-sm" onClick={() => void save(turn)}>Save</button></span>
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
      {!visible.length ? <p className="records-empty">{turns.length ? "No moment matches that." : "No transcript yet."}</p> : null}
    </section>
  );
}
