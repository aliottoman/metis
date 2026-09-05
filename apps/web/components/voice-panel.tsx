"use client";

// Voice mode, inside the chat workspace rather than on a page of its own.
// Everything spoken is written here too — the same words, with their sources
// and any refusal beside them — because a spoken answer is heard once and an
// ear cannot scroll back. The state label is deliberately specific: silence while Metis
// retrieves must read as "looking through your records", not as a connection
// that quietly died.

import { useCallback, useEffect, useRef, useState } from "react";
import { ConversationProvider } from "@elevenlabs/react";
import { useRouter } from "next/navigation";

import { SelectMenu } from "@/components/select-menu";
import {
  getSpeechPreference,
  getVoiceAvailability,
  prewarmVoice,
  setSpeechPreference,
  undoVoiceWrite,
} from "@/lib/api";
import type { SpeechPreference, VoiceAvailability, VoiceWriteReceipt } from "@/lib/types";
import { useVoiceSession, type VoiceHandoff, type VoiceState } from "@/hooks/use-voice-session";

/** Read once, and only ever set to true — the disclosure is not a nag. */
const DISCLOSURE_KEY = "metis.voice.disclosed";
const OUTPUT_VOLUME_KEY = "metis.voice.output-volume";
const VOICE_NAVIGATION_PATHS = new Set([
  "/", "/today", "/meetings", "/interviews", "/customers", "/assets",
  "/knowledge", "/answers", "/memory", "/sizing", "/tools", "/settings",
]);

const END_REASON_COPY = {
  idle_timeout: "Stopped automatically after two quiet minutes.",
  background: "Stopped when this tab moved to the background.",
  connection_lost: "The live connection ended before you stopped it.",
  stopped: "",
} as const;

const VOICE_MODEL_NAMES: Record<string, string> = {
  "deepseek-v4-flash:cloud": "DeepSeek V4 Flash",
  "gpt-oss:20b-cloud": "GPT-OSS 20B",
  "gpt-oss:120b-cloud": "GPT-OSS 120B",
  "glm-5.2:cloud": "GLM 5.2",
};

function voiceModelOption(model: string) {
  return {
    value: model,
    label: VOICE_MODEL_NAMES[model] ?? model,
    hint: "Ollama Cloud",
    group: "Hosted through Ollama",
  };
}

const STATE_LABEL: Record<VoiceState, string> = {
  idle: "Not listening",
  connecting: "Connecting",
  listening: "Listening",
  thinking: "Looking through your records",
  speaking: "Speaking",
  reconnecting: "Reconnecting",
  failed: "Connection failed",
};

function elapsedLabel(seconds: number): string {
  const whole = Math.max(0, Math.floor(seconds));
  return `${String(Math.floor(whole / 60)).padStart(2, "0")}:${String(whole % 60).padStart(2, "0")}`;
}

const VOICE_SIGNAL_BANDS = Array.from({ length: 13 }, (_, index) => index + 1);

/**
 * A deliberately non-literal voice indicator. The numbered frame, centerline,
 * and restrained signal bands borrow from the editorial / instrument-panel
 * language used by Meetings and Interviews instead of presenting the agent as
 * a glossy character. State remains available as text and an accessible label;
 * motion is only a secondary cue supplied by CSS.
 */
function VoiceSignal({
  state,
  compact = false,
}: {
  state: VoiceState;
  compact?: boolean;
}) {
  const active = state !== "idle" && state !== "failed";

  return (
    <div
      className={`voiceSignal ${compact ? "isCompact" : "isHero"} is-${state}`}
      data-state={state}
      role="img"
      aria-label={`Metis voice is ${STATE_LABEL[state].toLowerCase()}`}
    >
      <div className="voiceSignalTopline" aria-hidden="true">
        <span>METIS / VOICE</span>
        <span>{active ? "LIVE SIGNAL" : "STANDBY"}</span>
      </div>
      <div className="voiceSignalAperture" aria-hidden="true">
        <span className="voiceSignalAxis isHorizontal" />
        <span className="voiceSignalAxis isVertical" />
        <span className="voiceSignalSweep" />
        <div className="voiceSignalBands">
          {VOICE_SIGNAL_BANDS.map((band) => (
            <i key={band} className={`voiceSignalBand isBand${band}`} />
          ))}
        </div>
      </div>
      <div className="voiceSignalBaseline" aria-hidden="true">
        <span>01</span>
        <span>{STATE_LABEL[state]}</span>
        <span>13</span>
      </div>
    </div>
  );
}

function voiceTranscript(turns: ReturnType<typeof useVoiceSession>["turns"]): string {
  return turns
    .map((turn) => [
      `You: ${turn.transcript}`,
      turn.rendition ? `Metis: ${turn.rendition.written}` : "",
    ].filter(Boolean).join("\n"))
    .join("\n\n");
}

function exportText(filename: string, contents: string): void {
  const url = URL.createObjectURL(new Blob([contents], { type: "text/plain;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

/** What voice just added, and the one control that takes it back.
 *
 * The card is the safety mechanism, which is why it quotes the committed text
 * rather than describing it: "Added note to Batelco" tells you something
 * happened, "Added note to Batelco: 'Workshop moved to Thursday'" lets you
 * notice it is wrong. Undo lives here and only here — a spoken "undo that" is
 * refused, because deleting a customer record on a misheard word is precisely
 * what the refusal list exists for. */
function VoiceReceiptCard({ receipt }: { receipt: VoiceWriteReceipt }) {
  const [state, setState] = useState<"idle" | "undoing" | "undone" | "failed">(
    receipt.undone_at ? "undone" : "idle",
  );
  const [error, setError] = useState<string | null>(null);

  async function undo() {
    setState("undoing");
    setError(null);
    try {
      await undoVoiceWrite(receipt.id, receipt.undo_token);
      setState("undone");
    } catch (undoError) {
      setState("failed");
      setError(undoError instanceof Error ? undoError.message : "Could not undo that.");
    }
  }

  return (
    <div className={`voiceReceipt is-${state}`} role="status">
      <span className="voiceReceiptKind">{receipt.record_type}</span>
      <p>{receipt.summary}</p>
      {state === "undone" ? (
        <span className="mutedMeta">Undone. The receipt stays in your audit trail.</span>
      ) : (
        <button
          className="textButton"
          type="button"
          onClick={() => void undo()}
          disabled={state === "undoing" || !receipt.undo_token}
        >
          {state === "undoing" ? "Undoing…" : "Undo"}
        </button>
      )}
      {error ? <span className="mutedMeta" role="alert">{error}</span> : null}
    </div>
  );
}

export function VoicePanel({
  onHandoff,
  onExitToChat,
}: {
  onHandoff: (handoff: VoiceHandoff) => void;
  onExitToChat: () => void;
}) {
  return (
    <ConversationProvider>
      <VoicePanelBody onHandoff={onHandoff} onExitToChat={onExitToChat} />
    </ConversationProvider>
  );
}

function VoicePanelBody({
  onHandoff,
  onExitToChat,
}: {
  onHandoff: (handoff: VoiceHandoff) => void;
  onExitToChat: () => void;
}) {
  const router = useRouter();
  const [availability, setAvailability] = useState<VoiceAvailability | null>(null);
  const [speech, setSpeech] = useState<SpeechPreference | null>(null);
  const [modelSaving, setModelSaving] = useState(false);
  const [modelError, setModelError] = useState<string | null>(null);
  const [disclosed, setDisclosed] = useState(true);
  const [transcriptNotice, setTranscriptNotice] = useState("");
  const navigatedTurnRef = useRef<string | null>(null);
  const voice = useVoiceSession({ onHandoff });

  useEffect(() => {
    const alreadyDisclosed = window.localStorage.getItem(DISCLOSURE_KEY) === "true";
    void getVoiceAvailability()
      .then((available) => {
        setAvailability(available);
        if (available.available && alreadyDisclosed) {
          void prewarmVoice().catch(() => undefined);
        }
      })
      .catch(() => setAvailability(null));
    void getSpeechPreference().then(setSpeech).catch(() => setSpeech(null));
    setDisclosed(alreadyDisclosed);
  }, []);

  useEffect(() => {
    const turn = voice.turns.at(-1);
    const rendition = turn?.rendition;
    if (
      !turn
      || !rendition
      || rendition.intent !== "navigation"
      || !rendition.navigation_path
      || !VOICE_NAVIGATION_PATHS.has(rendition.navigation_path)
      || navigatedTurnRef.current === turn.id
    ) {
      return;
    }
    navigatedTurnRef.current = turn.id;
    // The live event can arrive a fraction before ElevenLabs begins speaking
    // the acknowledgment. Give that short sentence room to land before this
    // page unmounts and closes the audio session.
    const timer = window.setTimeout(
      () => {
        if (rendition.navigation_path === "/") onExitToChat();
        else router.push(rendition.navigation_path!);
      },
      1_400,
    );
    return () => window.clearTimeout(timer);
  }, [onExitToChat, router, voice.turns]);

  useEffect(() => {
    const stored = window.localStorage.getItem(OUTPUT_VOLUME_KEY);
    if (stored === null) return;
    const volume = Number(stored);
    if (Number.isFinite(volume) && volume >= 0 && volume <= 1) {
      voice.setOutputVolume(volume);
    }
    // The session hook keeps this setter stable; preferences load once.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const acceptDisclosure = useCallback(() => {
    window.localStorage.setItem(DISCLOSURE_KEY, "true");
    setDisclosed(true);
    void prewarmVoice().catch(() => undefined);
  }, []);

  const live = voice.state !== "idle" && voice.state !== "failed";
  const busy = voice.state === "connecting" || voice.state === "reconnecting";

  async function chooseModel(model: string) {
    if (!speech || live || modelSaving || model === speech.voice_model) return;
    setModelSaving(true);
    setModelError(null);
    try {
      setSpeech(
        await setSpeechPreference(
          speech.stt_provider,
          speech.spoken_confirmation,
          model,
        ),
      );
    } catch (saveError) {
      setModelError(
        saveError instanceof Error ? saveError.message : "That voice model could not be selected.",
      );
    } finally {
      setModelSaving(false);
    }
  }

  function changeOutputVolume(volume: number) {
    voice.setOutputVolume(volume);
    window.localStorage.setItem(OUTPUT_VOLUME_KEY, String(volume));
  }

  async function copyTranscript() {
    const transcript = voiceTranscript(voice.turns);
    try {
      await navigator.clipboard.writeText(transcript);
      setTranscriptNotice("Transcript copied");
    } catch {
      setTranscriptNotice("Clipboard unavailable — use Export instead");
    }
  }

  if (availability && !availability.available) {
    return (
      <section className="voicePanel voiceStage voiceUnavailable">
        <span className="voiceStageEyebrow">Voice workspace</span>
        <VoiceSignal state="idle" compact />
        <h3>Voice isn&apos;t set up yet</h3>
        <p>Complete these one-time setup items, then restart Metis:</p>
        <ol className="voiceSetupList">
          {(availability.missing?.length ? availability.missing : [availability.reason]).map(
            (reason) => <li key={reason}>{reason}</li>,
          )}
        </ol>
        <a
          className="textButton voiceSetupLink"
          href="https://elevenlabs.io/app/agents"
          target="_blank"
          rel="noreferrer"
        >
          Open ElevenLabs Agents
        </a>
        {availability.custom_llm_url ? (
          <p className="mutedMeta">
            When you create the ElevenLabs agent, point its Custom LLM at{" "}
            <code>{availability.custom_llm_url}</code> and name the model{" "}
            <code>{availability.public_model_alias}</code>.
          </p>
        ) : null}
      </section>
    );
  }

  if (!disclosed) {
    return (
      <section className="voicePanel voiceStage voiceDisclosure">
        <span className="voiceStageEyebrow">Voice workspace</span>
        <VoiceSignal state="idle" compact />
        <h3>Before you start talking</h3>
        <p>
          Your microphone audio goes to ElevenLabs, which turns it into text and speaks
          the reply. The question and whatever records answer it go to the model you
          selected. Your vectors and indexes stay on this machine.
        </p>
        <p>
          Voice can read your records, and can add a note, fact, action, contact or win
          when you explicitly ask it to — each one shows a card here with an Undo. It
          can also open a Metis workspace page when you name it. It cannot build,
          approve, delete, overwrite, or run anything.
        </p>
        <button className="voiceStartButton" type="button" onClick={acceptDisclosure}>
          Understood
        </button>
      </section>
    );
  }

  return (
    <section className={`voicePanel voiceStage is-${voice.state}`}>
      <header className="voiceStageHeader">
        <div>
          <span className="voiceStageEyebrow">Voice workspace</span>
          <span className={`voiceState is-${voice.state}`} role="status" aria-live="polite" aria-atomic="true">
            <i aria-hidden="true" />
            {STATE_LABEL[voice.state]}
          </span>
          <small className="voiceSafetyLine">Background-safe · auto-stops after two quiet minutes</small>
        </div>
        <div className="voiceModelControl">
          <span>Reasoning model · Ollama Cloud</span>
          <SelectMenu
            className="voiceModelSelect"
            hideLabel
            label="Voice reasoning model"
            value={speech?.voice_model ?? ""}
            onChange={(model) => void chooseModel(model)}
            disabled={live || modelSaving || !speech?.voice_models.length}
            options={(speech?.voice_models ?? []).map(voiceModelOption)}
          />
          {modelSaving ? <small>Saving…</small> : null}
        </div>
      </header>

      <div className="voiceSignalStage">
        <VoiceSignal state={voice.state} />
        <div className="voiceStageCopy">
          <strong>
            {voice.state === "idle"
              ? "Ready when you are"
              : voice.state === "failed"
                ? "Let’s reconnect"
                : STATE_LABEL[voice.state]}
          </strong>
          <p>
            {voice.liveTranscript
              ? `“${voice.liveTranscript}”`
              : live
                ? "Speak naturally. You can interrupt Metis at any time."
                : "Start when you’re ready. The signal field reflects session state; ending the conversation always stays explicit."}
          </p>
          {live ? (
            <div className="voiceSessionMeters" aria-label="Live session timing">
              <span className="voiceElapsed">Live {elapsedLabel(voice.elapsed)}</span>
              <span className={voice.idleSecondsRemaining <= 30 ? "isEndingSoon" : ""}>
                Quiet timeout {elapsedLabel(voice.idleSecondsRemaining)}
              </span>
            </div>
          ) : null}
        </div>

        <div className={`voicePanelActions ${live ? "isLive" : ""}`}>
          {live ? (
            <>
              <button
                className="voiceActionButton"
                type="button"
                onClick={() => voice.setMuted(!voice.isMuted)}
                aria-pressed={voice.isMuted}
              >
                <span aria-hidden="true">{voice.isMuted ? "◌" : "◉"}</span>
                {voice.isMuted ? "Unmute" : "Mute"}
              </button>
              <label className="voiceVolumeControl">
                <span>Output</span>
                <input
                  type="range"
                  min="0"
                  max="100"
                  step="5"
                  value={Math.round(voice.outputVolume * 100)}
                  aria-label={`Voice output volume ${Math.round(voice.outputVolume * 100)} percent`}
                  onChange={(event) => changeOutputVolume(Number(event.target.value) / 100)}
                />
                <output>{Math.round(voice.outputVolume * 100)}%</output>
              </label>
              <button className="voiceActionButton isStop" type="button" onClick={voice.stop}>
                <span aria-hidden="true">■</span> End conversation
              </button>
            </>
          ) : (
            <button className="voiceStartButton" type="button" onClick={() => void voice.start()} disabled={busy}>
              <span aria-hidden="true">●</span>
              {voice.state === "failed" ? "Try again" : busy ? "Connecting…" : "Start conversation"}
            </button>
          )}
        </div>
      </div>

      {!live && voice.endReason && END_REASON_COPY[voice.endReason] ? (
        <p className="voiceEndNotice" role="status">{END_REASON_COPY[voice.endReason]}</p>
      ) : null}

      {voice.error ? (
        <div className="composerError" role="alert">
          <span>!</span>
          <p>{voice.error}</p>
          <button
            className="errorDismiss"
            type="button"
            aria-label="Dismiss"
            onClick={voice.dismissError}
          >
            ×
          </button>
        </div>
      ) : null}

      {modelError ? <p className="voiceModelError" role="alert">{modelError}</p> : null}

      {voice.turns.length ? (
        <section className="voiceHistory" aria-label="Voice transcript">
          <header>
            <div><span className="voiceStageEyebrow">Transcript</span><strong>{voice.turns.length} turn{voice.turns.length === 1 ? "" : "s"}</strong></div>
            <div className="voiceHistoryActions">
              <button type="button" onClick={() => void copyTranscript()}>Copy</button>
              <button type="button" onClick={() => { exportText("metis-voice-transcript.txt", voiceTranscript(voice.turns)); setTranscriptNotice("Transcript exported"); }}>Export .txt</button>
            </div>
            <span className="visuallyHidden" role="status" aria-live="polite">{transcriptNotice}</span>
          </header>
          <ol className="voiceTurns hasTurns">
            {voice.turns.map((turn) => (
              <li key={turn.id} className={`voiceTurn is-${turn.rendition?.intent ?? "read"}`}>
                <p className="voiceSaid"><span>You</span>{turn.transcript}</p>
                {turn.rendition ? (
                  <div className="voiceAnswer">
                    {/* The same words that were spoken, with the sources beside them. */}
                    <span className="voiceAnswerLabel">Metis</span>
                    <p>{turn.rendition.written}</p>
                    {turn.rendition.citations.length ? (
                      <ul className="voiceCitations">
                        {turn.rendition.citations.map((citation, index) => (
                          <li key={`${turn.id}-${index}`}>
                            <span>{index + 1}</span>
                            {citation.provider === "web" && /^https?:\/\//.test(citation.reference) ? (
                              <a href={citation.reference} target="_blank" rel="noreferrer">{citation.label}</a>
                            ) : citation.label}
                          </li>
                        ))}
                      </ul>
                    ) : null}
                    {turn.rendition.write ? (
                      <VoiceReceiptCard receipt={turn.rendition.write} />
                    ) : null}
                    {turn.rendition.intent === "refuse_build" ? (
                      <p className="voiceHandoffNote">
                        Added to your composer — switch to Chat when you&apos;re ready.
                      </p>
                    ) : null}
                  </div>
                ) : null}
              </li>
            ))}
          </ol>
        </section>
      ) : null}

      {live && !voice.turns.length ? (
        <div className="voicePromptDeck" aria-label="Things to try">
          <span>Try asking</span>
          <p>“What needs my attention today?”</p>
          <p>“Add a follow-up action for Batelco.”</p>
          <p>“What did we decide in the last meeting?”</p>
        </div>
      ) : null}
    </section>
  );
}
