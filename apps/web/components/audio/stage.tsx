"use client";

// The stage every spoken surface stands on: the orb as the only expressive
// element, one mono state label under it, a line of caption, and the
// controls. Voice, interviews and the agent demo all use this.

import { useEffect, useState, type ReactNode } from "react";
import { Mic, MicOff, Volume2 } from "lucide-react";

import { ElevenLabsOrb, type ElevenLabsOrbState } from "@/components/elevenlabs-orb";
import { useToast } from "@/components/ui/toast";
import { downloadText, transcriptText } from "@/lib/audio";

export const ORB_COLORS: [string, string] = ["#72528a", "#ff7759"];

function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    const sync = () => setReduced(media.matches);
    sync();
    media.addEventListener("change", sync);
    return () => media.removeEventListener("change", sync);
  }, []);
  return reduced;
}

export function Stage({ orb, live, label, hint, caption, meters, children, getInputVolume, getOutputVolume }: {
  orb: ElevenLabsOrbState;
  live: boolean;
  /** The state, in a few mono words. */
  label: string;
  /** One sentence under the label. */
  hint?: ReactNode;
  /** What is being said right now, streamed. */
  caption?: ReactNode;
  /** Small figures beside the label: elapsed, a countdown. */
  meters?: ReactNode;
  children?: ReactNode;
  getInputVolume?: () => number;
  getOutputVolume?: () => number;
}) {
  const reduced = useReducedMotion();
  return (
    <section className={`stage${live ? " is-live" : ""}`} data-orb={orb ?? "idle"}>
      <div className="stage-orb" aria-hidden="true">
        {reduced || !orb ? <span className="stage-orb-still" /> : <ElevenLabsOrb className="stage-orb-canvas" colors={ORB_COLORS} agentState={orb} getInputVolume={getInputVolume} getOutputVolume={getOutputVolume} />}
      </div>
      <div className="stage-state" role="status" aria-live="polite" aria-atomic="true">
        <span className="stage-label">{live ? <i className="ui-dot is-live" aria-hidden="true" /> : null}{label}</span>
        {meters ? <span className="stage-meters">{meters}</span> : null}
        {hint ? <p className="stage-hint">{hint}</p> : null}
      </div>
      {caption ? <div className="stage-caption">{caption}</div> : null}
      {children ? <div className="stage-controls">{children}</div> : null}
    </section>
  );
}

export function MicButton({ muted, onToggle, disabled }: { muted: boolean; onToggle: () => void; disabled?: boolean }) {
  return (
    <button type="button" className={`ui-btn${muted ? " is-danger" : ""}`} aria-pressed={muted} disabled={disabled} onClick={onToggle}>
      {muted ? <MicOff size={14} aria-hidden="true" /> : <Mic size={14} aria-hidden="true" />}
      {muted ? "Unmute" : "Mute"}
    </button>
  );
}

export function VolumeControl({ value, onChange }: { value: number; onChange: (volume: number) => void }) {
  const percent = Math.round(value * 100);
  return (
    <label className="stage-volume" title="Output volume">
      <Volume2 size={14} aria-hidden="true" />
      <input type="range" min={0} max={100} step={5} value={percent} aria-label={`Output volume ${percent} percent`} onChange={(event) => onChange(Number(event.target.value) / 100)} />
      <output>{percent}%</output>
    </label>
  );
}

export interface TranscriptTurn {
  id: string;
  speaker: string;
  text: string;
  /** Marks the other party's lines, which read as the column. */
  agent?: boolean;
  /** Anything to show under the line: citations, a receipt. */
  extra?: ReactNode;
}

/** A spoken conversation written down, with copy and export. */
export function Transcript({ turns, filename, title = "Transcript", open = true }: { turns: TranscriptTurn[]; filename: string; title?: string; open?: boolean }) {
  const toast = useToast();
  if (!turns.length) return null;
  const text = transcriptText(turns);
  return (
    <details className="transcript" open={open}>
      <summary>
        <span>{title} <small>{turns.length} {turns.length === 1 ? "turn" : "turns"}</small></span>
        <span className="transcript-actions" onClick={(event) => event.preventDefault()}>
          <button type="button" className="ui-btn is-quiet is-sm" onClick={() => void navigator.clipboard.writeText(text).then(() => toast("Transcript copied")).catch(() => toast("Clipboard unavailable; use Export", "error"))}>Copy</button>
          <button type="button" className="ui-btn is-quiet is-sm" onClick={() => { downloadText(filename, text); toast("Transcript exported"); }}>Export</button>
        </span>
      </summary>
      <ol className="transcript-turns">
        {turns.map((turn) => (
          <li key={turn.id} className={turn.agent ? "is-agent" : "is-user"}>
            <span>{turn.speaker}</span>
            <p>{turn.text}</p>
            {turn.extra}
          </li>
        ))}
      </ol>
    </details>
  );
}
