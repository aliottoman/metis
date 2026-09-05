"use client";

// Voice mode, inside the chat workspace rather than on a page of its own.
// Everything spoken is written here too — the same words, with their sources
// and any refusal beside them — because a spoken answer is heard once and an
// ear cannot scroll back. The state label is deliberately specific: silence while Metis
// retrieves must read as "looking through your records", not as a connection
// that quietly died.

import { useCallback, useEffect, useState } from "react";
import { ConversationProvider } from "@elevenlabs/react";

import { ElevenLabsOrb, type ElevenLabsOrbState } from "@/components/elevenlabs-orb";
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
const METIS_ORB_COLORS: [string, string] = ["#72528a", "#ff7759"];

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

function orbState(state: VoiceState): ElevenLabsOrbState {
  if (state === "listening") return "listening";
  if (state === "speaking") return "talking";
  if (state === "thinking" || state === "connecting" || state === "reconnecting") return "thinking";
  return null;
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

export function VoicePanel({ onHandoff }: { onHandoff: (handoff: VoiceHandoff) => void }) {
  return (
    <ConversationProvider>
      <VoicePanelBody onHandoff={onHandoff} />
    </ConversationProvider>
  );
}

function VoicePanelBody({ onHandoff }: { onHandoff: (handoff: VoiceHandoff) => void }) {
  const [availability, setAvailability] = useState<VoiceAvailability | null>(null);
  const [speech, setSpeech] = useState<SpeechPreference | null>(null);
  const [modelSaving, setModelSaving] = useState(false);
  const [modelError, setModelError] = useState<string | null>(null);
  const [disclosed, setDisclosed] = useState(true);
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

  if (availability && !availability.available) {
    return (
      <section className="voicePanel voiceStage voiceUnavailable">
        <span className="voiceStageEyebrow">Voice workspace</span>
        <div className="voiceEmptyOrb" aria-hidden="true"><i /></div>
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
        <div className="voiceEmptyOrb" aria-hidden="true"><i /></div>
        <h3>Before you start talking</h3>
        <p>
          Your microphone audio goes to ElevenLabs, which turns it into text and speaks
          the reply. The question and whatever records answer it go to the model you
          selected. Your vectors and indexes stay on this machine.
        </p>
        <p>
          Voice can read your records, and can add a note, fact, action, contact or win
          when you explicitly ask it to — each one shows a card here with an Undo. It
          cannot build, approve, delete, overwrite, or run anything.
        </p>
        <button className="voiceStartButton" type="button" onClick={acceptDisclosure}>
          Understood
        </button>
      </section>
    );
  }

  return (
    <section className={`voicePanel voiceStage is-${voice.state}`} aria-live="polite">
      <header className="voiceStageHeader">
        <div>
          <span className="voiceStageEyebrow">Voice workspace</span>
          <span className={`voiceState is-${voice.state}`}>
            <i aria-hidden="true" />
            {STATE_LABEL[voice.state]}
          </span>
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

      <div className="voiceOrbStage">
        <button
          className="voiceOrb"
          type="button"
          onClick={live ? voice.stop : () => void voice.start()}
          disabled={busy}
          aria-label={live ? "End voice conversation" : "Start voice conversation"}
        >
          <ElevenLabsOrb
            className="voiceOrbRenderer"
            colors={METIS_ORB_COLORS}
            agentState={orbState(voice.state)}
            getInputVolume={voice.getInputVolume}
            getOutputVolume={voice.getOutputVolume}
          />
        </button>
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
                : "Tap the orb to begin a live conversation."}
          </p>
          {live ? <span className="voiceElapsed">{elapsedLabel(voice.elapsed)}</span> : null}
        </div>
      </div>

      <div className="voicePanelActions">
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
            <button className="voiceActionButton isStop" type="button" onClick={voice.stop}>
              <span aria-hidden="true">■</span> End
            </button>
          </>
        ) : (
          <button className="voiceStartButton" type="button" onClick={() => void voice.start()} disabled={busy}>
            <span aria-hidden="true">●</span>
            {voice.state === "failed" ? "Try again" : busy ? "Connecting…" : "Start conversation"}
          </button>
        )}
      </div>

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

      <ol className={`voiceTurns ${voice.turns.length ? "hasTurns" : ""}`}>
        {voice.turns.map((turn) => (
          <li key={turn.id} className={`voiceTurn is-${turn.rendition?.intent ?? "read"}`}>
            <p className="voiceSaid">{turn.transcript}</p>
            {turn.rendition ? (
              <div className="voiceAnswer">
                {/* The same words that were spoken, with the sources beside them. */}
                <p>{turn.rendition.written}</p>
                {turn.rendition.citations.length ? (
                  <ul className="voiceCitations">
                    {turn.rendition.citations.map((citation, index) => (
                      <li key={`${turn.id}-${index}`}>
                        <span>{index + 1}</span>
                        {citation.label}
                      </li>
                    ))}
                  </ul>
                ) : null}
                {turn.rendition.write ? (
                  <VoiceReceiptCard receipt={turn.rendition.write} />
                ) : null}
                {turn.rendition.intent === "refuse_build" ? (
                  <p className="voiceHandoffNote">
                    Dropped into your composer — switch over when you&apos;re ready.
                  </p>
                ) : null}
              </div>
            ) : null}
          </li>
        ))}
      </ol>

      {live && !voice.turns.length ? (
        <p className="voiceStagePrompt">Try “What needs my attention today?” or ask about anything in your knowledge base.</p>
      ) : null}
    </section>
  );
}
