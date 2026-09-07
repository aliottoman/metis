"use client";

// The room while it is open: the stage, the pinned question, the live
// caption, and the one way out.

import { useState } from "react";
import { Pause, Play, Square } from "lucide-react";

import { MicButton, Stage, Transcript, VolumeControl } from "@/components/audio/stage";
import type { useInterviewSession, InterviewPhase } from "@/hooks/use-interview-session";
import { elapsedLabel, safeFilePart } from "@/lib/audio";

type Interview = ReturnType<typeof useInterviewSession>;

const LABEL: Record<InterviewPhase, string> = {
  setup: "Fill in the role to begin",
  ready: "Ready when you are",
  connecting: "Connecting",
  listening: "Listening",
  agent_speaking: "Chiron is speaking",
  evaluating: "Scoring the interview",
  debriefing: "Your verdict is ready",
  complete: "Interview complete",
  failed: "Connection failed",
  ended_early: "Ended early",
};

export function InterviewLive({ interview, jobTitle, companyName, round, focus }: { interview: Interview; jobTitle: string; companyName: string; round: string; focus: string[] }) {
  const [confirmingEnd, setConfirmingEnd] = useState(false);
  const [typed, setTyped] = useState("");
  const latest = interview.turns.at(-1) ?? null;
  const orb = interview.phase === "listening" ? "listening" : interview.phase === "agent_speaking" ? "talking" : interview.phase === "evaluating" || interview.phase === "connecting" ? "thinking" : null;
  const hint = interview.audioPaused ? "Audio is paused locally. The room stays open."
    : interview.phase === "debriefing" ? "Stay for the spoken debrief, or view the score now."
      : interview.phase === "evaluating" ? "Chiron is scoring the round."
        : interview.phase === "connecting" ? "Setting up the room."
          : "Answer out loud. Chiron will not coach you mid-round.";

  return (
    <section className="interview-live" aria-label="Live interview">
      <header className="interview-live-head">
        <div>
          <span className="ui-eyebrow">{round} round</span>
          <strong>{jobTitle} · {companyName}</strong>
          {focus.length ? <div className="interview-brief-focus">{focus.map((area) => <span key={area} className="ui-chip">{area}</span>)}</div> : null}
        </div>
        <span className="interview-progress">{interview.questionNumber > 0 ? `Question ${interview.questionNumber} of ${interview.questionLimit}` : "Opening"}</span>
      </header>

      <Stage
        orb={orb}
        live={interview.live}
        label={LABEL[interview.phase]}
        hint={hint}
        meters={<span>{elapsedLabel(interview.elapsed)}</span>}
        caption={latest ? <p className={latest.role === "agent" ? "voice-saying" : "voice-heard"}><small>{latest.role === "agent" ? "Chiron" : "You"}</small>{latest.text}</p> : null}
        getInputVolume={interview.getInputVolume}
        getOutputVolume={interview.getOutputVolume}
      >
        <button type="button" className="ui-btn" aria-pressed={interview.audioPaused} onClick={interview.toggleAudioPaused}>{interview.audioPaused ? <Play size={14} aria-hidden="true" /> : <Pause size={14} aria-hidden="true" />}{interview.audioPaused ? "Resume audio" : "Pause audio"}</button>
        <MicButton muted={interview.isMuted} onToggle={() => interview.setMuted(!interview.isMuted)} disabled={interview.audioPaused} />
        <VolumeControl value={interview.outputVolume} onChange={interview.setOutputVolume} />
        {interview.phase === "debriefing" ? (
          <button type="button" className="ui-btn is-primary" onClick={interview.viewResults}>View results</button>
        ) : confirmingEnd ? (
          <span className="interview-end" role="group" aria-label="How to finish" onKeyDown={(event) => { if (event.key === "Escape") setConfirmingEnd(false); }}>
            <small>{interview.finishRequested ? "Chiron has your finish request. Confirm it in the conversation, or leave without a score." : interview.candidateAnswerCount >= 3 ? "Enough answers for a provisional score. Finish through Chiron, or leave without one." : "A score needs three answers. Leave now without one, or keep going."}</small>
            {interview.canFinishAndScore ? <button type="button" className="ui-btn is-primary is-sm" onClick={() => { setConfirmingEnd(false); interview.requestFinishAndScore(); }}>Finish &amp; score</button> : null}
            <button type="button" className="ui-btn is-danger is-sm" onClick={() => { setConfirmingEnd(false); interview.endEarly(); }}>End without score</button>
            <button type="button" className="ui-btn is-sm" autoFocus onClick={() => setConfirmingEnd(false)}>Keep going</button>
          </span>
        ) : (
          <button type="button" className="ui-btn is-danger" onClick={() => setConfirmingEnd(true)}><Square size={12} aria-hidden="true" /> End interview</button>
        )}
      </Stage>

      {interview.finishRequested ? <p className="interview-note" role="status">Finish requested. Chiron will ask for one spoken confirmation, then score what you completed.</p> : null}

      <section className="interview-question" aria-label="Current question">
        <span className="ui-eyebrow">Current question</span>
        <p>{interview.latestQuestion || "Chiron's opening question stays pinned here once the round begins."}</p>
      </section>

      <form className="interview-typed" onSubmit={(event) => { event.preventDefault(); const text = typed.trim(); if (text) { interview.sendText(text); setTyped(""); } }}>
        <input value={typed} placeholder="Can't speak right now? Type your reply." aria-label="Type a reply to Chiron" onChange={(event) => setTyped(event.target.value)} />
        <button type="submit" className="ui-btn is-sm" disabled={!typed.trim()}>Send</button>
      </form>

      <Transcript open={false} turns={interview.turns.map((turn) => ({ id: String(turn.ordinal), speaker: turn.role === "agent" ? "Chiron" : "You", text: turn.text, agent: turn.role === "agent" }))} filename={`${safeFilePart(jobTitle)}-interview-transcript.txt`} />
    </section>
  );
}
