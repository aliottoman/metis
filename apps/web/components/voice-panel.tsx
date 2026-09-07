"use client";

// Voice mode, inside the chat workspace. The orb is the state; the words
// appear under it as they are spoken; the transcript below keeps the same
// words with their sources and any receipt, because an ear cannot scroll
// back.

import { useCallback, useEffect, useRef, useState } from "react";
import { ConversationProvider } from "@elevenlabs/react";
import { useRouter } from "next/navigation";
import { Square } from "lucide-react";

import { MicButton, Stage, Transcript, VolumeControl, type TranscriptTurn } from "@/components/audio/stage";
import { Notice } from "@/components/ui/notice";
import { useVoiceSession, type VoiceHandoff, type VoiceState } from "@/hooks/use-voice-session";
import { getSpeechPreference, getVoiceAvailability, prewarmVoice, setSpeechPreference, undoVoiceWrite } from "@/lib/api";
import { elapsedLabel } from "@/lib/audio";
import type { SpeechPreference, VoiceAvailability, VoiceWriteReceipt } from "@/lib/types";

/** Read once, and only ever set to true; the disclosure is not a nag. */
const DISCLOSURE_KEY = "metis.voice.disclosed";
const OUTPUT_VOLUME_KEY = "metis.voice.output-volume";
const NAVIGATION_PATHS = new Set(["/", "/today", "/meetings", "/interviews", "/customers", "/assets", "/knowledge", "/answers", "/memory", "/sizing", "/tools", "/settings"]);

const END_REASON: Record<string, string> = {
  idle_timeout: "Stopped after two quiet minutes.",
  background: "Stopped when this tab moved to the background.",
  connection_lost: "The live connection ended before you stopped it.",
};
const LABEL: Record<VoiceState, string> = {
  idle: "Ready when you are",
  connecting: "Connecting",
  listening: "Listening",
  thinking: "Looking through your records",
  speaking: "Speaking",
  reconnecting: "Reconnecting",
  failed: "Connection failed",
};
const MODEL_NAMES: Record<string, string> = {
  "deepseek-v4-flash:cloud": "DeepSeek V4 Flash",
  "gpt-oss:20b-cloud": "GPT-OSS 20B",
  "gpt-oss:120b-cloud": "GPT-OSS 120B",
  "glm-5.2:cloud": "GLM 5.2",
};

/** What voice just added, quoted, with the one control that takes it back.
 *  A spoken "undo that" is refused on purpose: deleting a record on a
 *  misheard word is what the refusal list exists for. */
function Receipt({ receipt }: { receipt: VoiceWriteReceipt }) {
  const [state, setState] = useState<"idle" | "undoing" | "undone" | "failed">(receipt.undone_at ? "undone" : "idle");
  const undo = async () => {
    setState("undoing");
    try {
      await undoVoiceWrite(receipt.id, receipt.undo_token);
      setState("undone");
    } catch {
      setState("failed");
    }
  };
  return (
    <div className={`receipt is-${state}`} role="status">
      <span className="ui-chip">{receipt.record_type}</span>
      <p>{receipt.summary}</p>
      {state === "undone" ? <small>Undone. The receipt stays in your audit trail.</small>
        : state === "failed" ? <small>Could not undo that.</small>
          : <button type="button" className="ui-btn is-quiet is-sm" onClick={() => void undo()} disabled={state === "undoing" || !receipt.undo_token}>{state === "undoing" ? "Undoing…" : "Undo"}</button>}
    </div>
  );
}

export function VoicePanel({ onHandoff, onExitToChat }: { onHandoff: (handoff: VoiceHandoff) => void; onExitToChat: () => void }) {
  return (
    <ConversationProvider>
      <VoicePanelBody onHandoff={onHandoff} onExitToChat={onExitToChat} />
    </ConversationProvider>
  );
}

function VoicePanelBody({ onHandoff, onExitToChat }: { onHandoff: (handoff: VoiceHandoff) => void; onExitToChat: () => void }) {
  const router = useRouter();
  const voice = useVoiceSession({ onHandoff });
  const [availability, setAvailability] = useState<VoiceAvailability | null>(null);
  const [speech, setSpeech] = useState<SpeechPreference | null>(null);
  const [modelSaving, setModelSaving] = useState(false);
  const [disclosed, setDisclosed] = useState(true);
  const navigatedTurn = useRef<string | null>(null);
  const { setOutputVolume } = voice;

  useEffect(() => {
    const already = window.localStorage.getItem(DISCLOSURE_KEY) === "true";
    setDisclosed(already);
    void getVoiceAvailability().then((available) => {
      setAvailability(available);
      if (available.available && already) void prewarmVoice().catch(() => undefined);
    }).catch(() => setAvailability(null));
    void getSpeechPreference().then(setSpeech).catch(() => setSpeech(null));
    const stored = Number(window.localStorage.getItem(OUTPUT_VOLUME_KEY));
    if (Number.isFinite(stored) && stored > 0 && stored <= 1) setOutputVolume(stored);
  }, [setOutputVolume]);

  // A spoken "open Customers" navigates once the acknowledgment has had room to land.
  useEffect(() => {
    const turn = voice.turns.at(-1);
    const path = turn?.rendition?.navigation_path;
    if (!turn || turn.rendition?.intent !== "navigation" || !path || !NAVIGATION_PATHS.has(path) || navigatedTurn.current === turn.id) return;
    navigatedTurn.current = turn.id;
    const timer = window.setTimeout(() => (path === "/" ? onExitToChat() : router.push(path)), 1_400);
    return () => window.clearTimeout(timer);
  }, [onExitToChat, router, voice.turns]);

  const accept = useCallback(() => {
    window.localStorage.setItem(DISCLOSURE_KEY, "true");
    setDisclosed(true);
    void prewarmVoice().catch(() => undefined);
  }, []);

  const live = voice.state !== "idle" && voice.state !== "failed";
  const busy = voice.state === "connecting" || voice.state === "reconnecting";
  const orb = voice.state === "speaking" ? "talking" : voice.state === "thinking" ? "thinking" : live ? "listening" : null;

  const chooseModel = async (model: string) => {
    if (!speech || live || modelSaving || model === speech.voice_model) return;
    setModelSaving(true);
    try {
      setSpeech(await setSpeechPreference(speech.stt_provider, speech.spoken_confirmation, model));
    } finally {
      setModelSaving(false);
    }
  };

  if (availability && !availability.available) {
    return (
      <section className="voice">
        <Notice kind="info" title="Voice isn't set up yet">
          <p>Complete these one-time items, then restart Metis:</p>
          <ol>{(availability.missing.length ? availability.missing : [availability.reason]).map((reason) => <li key={reason}>{reason}</li>)}</ol>
          {availability.custom_llm_url ? <p>Point the ElevenLabs agent&rsquo;s Custom LLM at <code>{availability.custom_llm_url}</code> and name the model <code>{availability.public_model_alias}</code>.</p> : null}
          <a className="ui-btn is-sm" href="https://elevenlabs.io/app/agents" target="_blank" rel="noreferrer">Open ElevenLabs Agents</a>
        </Notice>
      </section>
    );
  }

  if (!disclosed) {
    return (
      <section className="voice">
        <Stage orb={null} live={false} label="Before you start talking" hint="Your microphone audio goes to ElevenLabs, which turns it into text and speaks the reply. The question and the records that answer it go to the model you selected. Your indexes stay on this machine.">
          <p className="voice-disclosure">Voice can read your records and, when you ask, add a note, fact, action, contact or win, each with an Undo here. It can open a page you name. It cannot build, approve, delete or run anything.</p>
          <button type="button" className="ui-btn is-primary" onClick={accept}>Understood</button>
        </Stage>
      </section>
    );
  }

  const transcript: TranscriptTurn[] = voice.turns.flatMap((turn) => {
    const said: TranscriptTurn = { id: `${turn.id}-you`, speaker: "You", text: turn.transcript };
    if (!turn.rendition) return [said];
    const rendition = turn.rendition;
    return [said, {
      id: turn.id, speaker: "Metis", text: rendition.written, agent: true,
      extra: (
        <>
          {rendition.citations.length ? (
            <ol className="voice-sources">
              {rendition.citations.map((citation, index) => (
                <li key={index}>{citation.provider === "web" && /^https?:\/\//.test(citation.reference) ? <a href={citation.reference} target="_blank" rel="noreferrer">{citation.label}</a> : citation.label}</li>
              ))}
            </ol>
          ) : null}
          {rendition.write ? <Receipt receipt={rendition.write} /> : null}
          {rendition.intent === "refuse_build" ? <small>Added to your composer. Switch to Chat when you are ready.</small> : null}
        </>
      ),
    }];
  });

  return (
    <section className="voice">
      <Stage
        orb={orb}
        live={live}
        label={LABEL[voice.state]}
        hint={live ? "Speak naturally. You can interrupt at any time." : voice.endReason && END_REASON[voice.endReason] ? END_REASON[voice.endReason] : "Auto-stops after two quiet minutes."}
        meters={live ? <><span>{elapsedLabel(voice.elapsed)}</span><span className={voice.idleSecondsRemaining <= 30 ? "is-ending" : ""}>quiet {elapsedLabel(voice.idleSecondsRemaining)}</span></> : null}
        caption={voice.spoken.length ? <p className="voice-saying">{voice.spoken.join(" ")}</p> : voice.liveTranscript ? <p className="voice-heard">“{voice.liveTranscript}”</p> : null}
        getInputVolume={voice.getInputVolume}
        getOutputVolume={voice.getOutputVolume}
      >
        {live ? (
          <>
            <MicButton muted={voice.isMuted} onToggle={() => voice.setMuted(!voice.isMuted)} />
            <VolumeControl value={voice.outputVolume} onChange={(volume) => { voice.setOutputVolume(volume); window.localStorage.setItem(OUTPUT_VOLUME_KEY, String(volume)); }} />
            <button type="button" className="ui-btn is-danger" onClick={voice.stop}><Square size={12} aria-hidden="true" /> End</button>
          </>
        ) : (
          <>
            <button type="button" className="ui-btn is-primary is-lg" onClick={() => void voice.start()} disabled={busy}>{voice.state === "failed" ? "Try again" : busy ? "Connecting…" : "Start talking"}</button>
            {speech?.voice_models.length ? (
              <label className="ui-field voice-model">
                <span>Reasoning model</span>
                <select value={speech.voice_model} disabled={modelSaving} onChange={(event) => void chooseModel(event.target.value)}>
                  {speech.voice_models.map((model) => <option key={model} value={model}>{MODEL_NAMES[model] ?? model}</option>)}
                </select>
              </label>
            ) : null}
          </>
        )}
      </Stage>

      {voice.error ? <Notice kind="error" onDismiss={voice.dismissError}>{voice.error}</Notice> : null}

      {live && !voice.turns.length ? (
        <p className="voice-try">Try “What needs my attention today?”, “Add a follow-up action for Batelco”, or “What did we decide in the last meeting?”</p>
      ) : null}

      <Transcript turns={transcript} filename="metis-voice-transcript.txt" />
    </section>
  );
}
