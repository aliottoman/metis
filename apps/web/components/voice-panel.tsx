"use client";

// Voice mode, inside the chat workspace rather than on a page of its own.
// Everything spoken has a written twin here — the full answer, its sources,
// and any refusal — because a spoken answer is heard once and an ear cannot
// scroll back. The state label is deliberately specific: silence while Metis
// retrieves must read as "looking through your records", not as a connection
// that quietly died.

import { useCallback, useEffect, useState } from "react";
import { ConversationProvider } from "@elevenlabs/react";

import { getVoiceAvailability } from "@/lib/api";
import type { VoiceAvailability } from "@/lib/types";
import { useVoiceSession, type VoiceHandoff, type VoiceState } from "@/hooks/use-voice-session";

/** Read once, and only ever set to true — the disclosure is not a nag. */
const DISCLOSURE_KEY = "metis.voice.disclosed";

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

export function VoicePanel({ onHandoff }: { onHandoff: (handoff: VoiceHandoff) => void }) {
  return (
    <ConversationProvider>
      <VoicePanelBody onHandoff={onHandoff} />
    </ConversationProvider>
  );
}

function VoicePanelBody({ onHandoff }: { onHandoff: (handoff: VoiceHandoff) => void }) {
  const [availability, setAvailability] = useState<VoiceAvailability | null>(null);
  const [disclosed, setDisclosed] = useState(true);
  const voice = useVoiceSession({ onHandoff });

  useEffect(() => {
    void getVoiceAvailability().then(setAvailability).catch(() => setAvailability(null));
    setDisclosed(window.localStorage.getItem(DISCLOSURE_KEY) === "true");
  }, []);

  const acceptDisclosure = useCallback(() => {
    window.localStorage.setItem(DISCLOSURE_KEY, "true");
    setDisclosed(true);
  }, []);

  const live = voice.state !== "idle" && voice.state !== "failed";

  if (availability && !availability.available) {
    return (
      <section className="voicePanel voiceUnavailable">
        <h3>Voice isn&apos;t set up yet</h3>
        <p>{availability.reason}</p>
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
      <section className="voicePanel voiceDisclosure">
        <h3>Before you start talking</h3>
        <p>
          Your microphone audio goes to ElevenLabs, which turns it into text and speaks
          the reply. The question and whatever records answer it go to the model you
          selected. Your vectors and indexes stay on this machine, and voice can read
          your records but cannot build, approve, delete, or run anything.
        </p>
        <button className="primaryButton" type="button" onClick={acceptDisclosure}>
          Understood
        </button>
      </section>
    );
  }

  return (
    <section className="voicePanel" aria-live="polite">
      <header className="voicePanelHead">
        <span className={`voiceState is-${voice.state}`}>
          <i aria-hidden="true" />
          {STATE_LABEL[voice.state]}
        </span>
        {live ? <span className="voiceElapsed">{elapsedLabel(voice.elapsed)}</span> : null}
        <div className="voicePanelActions">
          {live ? (
            <>
              <button className="secondaryButton" type="button" onClick={voice.interrupt}>
                Interrupt
              </button>
              <button
                className="secondaryButton"
                type="button"
                onClick={() => voice.setMuted(!voice.isMuted)}
                aria-pressed={voice.isMuted}
              >
                {voice.isMuted ? "Unmute" : "Mute"}
              </button>
              <button className="primaryButton" type="button" onClick={voice.stop}>
                Stop
              </button>
            </>
          ) : (
            <button className="primaryButton" type="button" onClick={() => void voice.start()}>
              {voice.state === "failed" ? "Try again" : "Start talking"}
            </button>
          )}
        </div>
      </header>

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

      <ol className="voiceTurns">
        {voice.turns.map((turn) => (
          <li key={turn.id} className={`voiceTurn is-${turn.rendition?.intent ?? "read"}`}>
            <p className="voiceSaid">{turn.transcript}</p>
            {turn.rendition ? (
              <div className="voiceAnswer">
                {/* The written twin. What was spoken was shorter; this is all of it. */}
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
        <p className="mutedMeta">Ask about an account, the queue, or anything in your knowledge base.</p>
      ) : null}
    </section>
  );
}
