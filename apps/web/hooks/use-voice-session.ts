"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useConversation } from "@elevenlabs/react";

import {
  bindVoiceSession,
  endVoiceSession,
  renewVoiceSession,
  startVoiceSession,
  voiceEventsUrl,
} from "@/lib/api";
import type { VoiceRendition, VoiceSession } from "@/lib/types";

/**
 * One spoken conversation, from the browser's side.
 *
 * Two channels, and the split is the whole design. WebRTC carries audio to
 * and from ElevenLabs; a loopback SSE stream carries everything else —
 * written answers, citations, refusal handoffs — straight from Metis, never
 * through the tunnel.
 *
 * Teardown is deliberately over-covered: explicit stop, unmount, navigation,
 * `pagehide`, and a failed connection all end the session, and the lease
 * renewal stops with them. None of that is trusted to be enough. The server
 * closes the tunnel when a lease stops being renewed, which is the only path
 * that survives a browser crash — everything here just makes the common cases
 * fast and honest.
 */

export type VoiceState =
  | "idle"
  | "connecting"
  | "listening"
  | "thinking"
  | "speaking"
  | "reconnecting"
  | "failed";

export type VoiceEndReason =
  | "stopped"
  | "idle_timeout"
  | "background"
  | "connection_lost";

export const DEFAULT_VOICE_IDLE_TIMEOUT_SECONDS = 120;

export interface VoiceTurnRecord {
  id: string;
  transcript: string;
  rendition: VoiceRendition | null;
}

export interface VoiceHandoff {
  transcript: string;
  turnId: string;
}

export function useVoiceSession(options: {
  /** Called with the verbatim utterance when voice defers work to the composer. */
  onHandoff?: (handoff: VoiceHandoff) => void;
} = {}) {
  const [state, setState] = useState<VoiceState>("idle");
  const [session, setSession] = useState<VoiceSession | null>(null);
  const [turns, setTurns] = useState<VoiceTurnRecord[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [leaseSeconds, setLeaseSeconds] = useState(0);
  const [liveTranscript, setLiveTranscript] = useState("");
  // The answer as it is being said, one sentence at a time; cleared when the
  // finished turn arrives with the same words and their sources.
  const [spoken, setSpoken] = useState<string[]>([]);
  const [idleTimeoutSeconds, setIdleTimeoutSeconds] = useState(
    DEFAULT_VOICE_IDLE_TIMEOUT_SECONDS,
  );
  const [idleSecondsRemaining, setIdleSecondsRemaining] = useState(0);
  const [endReason, setEndReason] = useState<VoiceEndReason | null>(null);
  const [outputVolume, setOutputVolumeState] = useState(1);

  const sessionRef = useRef<string | null>(null);
  const mountedRef = useRef(true);
  const generationRef = useRef(0);
  const startPendingRef = useRef(false);
  const leaseTimerRef = useRef<number | null>(null);
  const tickRef = useRef<number | null>(null);
  const connectTimerRef = useRef<number | null>(null);
  const streamRef = useRef<EventSource | null>(null);
  const startedAtRef = useRef<number>(0);
  const activityAtRef = useRef<number>(0);
  const idleTimeoutRef = useRef(DEFAULT_VOICE_IDLE_TIMEOUT_SECONDS);
  const outputVolumeRef = useRef(1);
  const connectedOnceRef = useRef(false);
  const teardownRef = useRef<(
    nextState?: VoiceState,
    reason?: VoiceEndReason | null,
  ) => void>(() => undefined);
  const onHandoffRef = useRef(options.onHandoff);
  onHandoffRef.current = options.onHandoff;

  const markActivity = useCallback(() => {
    activityAtRef.current = Date.now();
    if (mountedRef.current) {
      setIdleSecondsRemaining(idleTimeoutRef.current);
    }
  }, []);

  const conversation = useConversation({
    onConnect: ({ conversationId }: { conversationId: string }) => {
      const id = sessionRef.current;
      if (!id) {
        // Stop may have won the race with the SDK finishing its connection.
        // End it again now that the SDK definitely has something to close.
        try {
          conversationRef.current.endSession();
        } catch {
          // There is no live SDK session after all.
        }
        return;
      }
      void bindVoiceSession(id, conversationId)
        .then((bound) => {
          if (sessionRef.current !== id) return;
          if (connectTimerRef.current !== null) {
            window.clearTimeout(connectTimerRef.current);
            connectTimerRef.current = null;
          }
          setSession(bound);
          connectedOnceRef.current = true;
          markActivity();
          setState("listening");
        })
        .catch(() => {
          if (sessionRef.current !== id) return;
          setError("Metis could not secure the live voice connection. Try again.");
          teardownRef.current("failed", "connection_lost");
        });
    },
    onDisconnect: () => {
      if (!sessionRef.current) return;
      setError("The live voice connection ended. Try reconnecting.");
      teardownRef.current("failed", "connection_lost");
    },
    onError: (message: string) => {
      if (!sessionRef.current && !startPendingRef.current) return;
      setError(message || "The voice connection failed.");
      teardownRef.current("failed", "connection_lost");
    },
    onModeChange: ({ mode }: { mode: "speaking" | "listening" }) => {
      if (!sessionRef.current) return;
      if (mode === "speaking") setLiveTranscript("");
      setState((current) =>
        current === "failed" || current === "idle" ? current : mode === "speaking" ? "speaking" : "listening",
      );
    },
    onMessage: ({ message, role }: { message: string; role: "user" | "agent" }) => {
      if (!sessionRef.current) return;
      markActivity();
      if (role === "user") {
        setLiveTranscript(message);
        setState("thinking");
      }
    },
    onStatusChange: ({ status }: { status: string }) => {
      if (!sessionRef.current) return;
      if (status === "connecting") {
        setState(connectedOnceRef.current ? "reconnecting" : "connecting");
      }
    },
  });
  // The SDK hook may return a fresh facade while its connection state changes.
  // Teardown must always use the latest facade, but it must not itself change
  // identity when that facade does — otherwise React runs the previous effect
  // cleanup during connect and immediately ends the session we just opened.
  const conversationRef = useRef(conversation);
  conversationRef.current = conversation;

  /** Everything that must stop, in the order it must stop in. */
  const teardown = useCallback(
    (
      nextState: VoiceState = "idle",
      reason: VoiceEndReason | null = null,
    ) => {
      // Invalidate an in-flight Start before touching any resources. If its
      // server request later resolves, that generation closes the newly-opened
      // server session without ever opening SSE or WebRTC.
      generationRef.current += 1;
      startPendingRef.current = false;
      if (leaseTimerRef.current !== null) window.clearInterval(leaseTimerRef.current);
      if (tickRef.current !== null) window.clearInterval(tickRef.current);
      if (connectTimerRef.current !== null) window.clearTimeout(connectTimerRef.current);
      leaseTimerRef.current = null;
      tickRef.current = null;
      connectTimerRef.current = null;
      streamRef.current?.close();
      streamRef.current = null;
      const id = sessionRef.current;
      sessionRef.current = null;
      activityAtRef.current = 0;
      connectedOnceRef.current = false;
      try {
        conversationRef.current.endSession();
      } catch {
        // Already gone. The server lease is what actually closes the tunnel.
      }
      if (id) void endVoiceSession(id).catch(() => undefined);
      if (mountedRef.current) {
        setState(nextState);
        setLeaseSeconds(0);
        setIdleSecondsRemaining(0);
        setLiveTranscript("");
        setSpoken([]);
        setEndReason(reason);
      }
    },
    [],
  );

  const stop = useCallback(() => teardown("idle", "stopped"), [teardown]);
  teardownRef.current = teardown;

  const setOutputVolume = useCallback((volume: number) => {
    const next = Number.isFinite(volume) ? Math.min(1, Math.max(0, volume)) : 1;
    outputVolumeRef.current = next;
    if (mountedRef.current) setOutputVolumeState(next);
    try {
      conversationRef.current.setVolume({ volume: next });
    } catch {
      // Remember it locally; Start applies it once the audio output exists.
    }
  }, []);

  const start = useCallback(async () => {
    if (sessionRef.current || startPendingRef.current) return;
    const generation = generationRef.current + 1;
    generationRef.current = generation;
    startPendingRef.current = true;
    const isCurrent = () =>
      mountedRef.current && generationRef.current === generation;
    setError(null);
    setEndReason(null);
    setTurns([]);
    setSpoken([]);
    setState("connecting");
    try {
      const opened = await startVoiceSession();
      if (!isCurrent()) {
        // The user stopped, navigated, or hid the tab while the POST was in
        // flight. The host did create a lease, so close it immediately.
        void endVoiceSession(opened.session.id).catch(() => undefined);
        return;
      }
      sessionRef.current = opened.session.id;
      setSession(opened.session);
      startedAtRef.current = Date.now();
      setElapsed(0);
      const timeout = Math.max(
        30,
        opened.session.idle_timeout_seconds
          ?? DEFAULT_VOICE_IDLE_TIMEOUT_SECONDS,
      );
      idleTimeoutRef.current = timeout;
      setIdleTimeoutSeconds(timeout);
      markActivity();

      // The loopback channel, opened before the audio one: a refusal can
      // arrive on the very first turn, and it must have somewhere to land.
      const stream = new EventSource(voiceEventsUrl(opened.session.id));
      streamRef.current = stream;
      stream.onmessage = (event) => {
        if (!isCurrent() || sessionRef.current !== opened.session.id) return;
        try {
          handleVoiceEvent(JSON.parse(event.data), setTurns, setSpoken, onHandoffRef.current);
        } catch {
          // A malformed frame costs itself, not the conversation.
        }
      };
      for (const type of ["voice.spoken", "voice.turn", "voice.build_deferred", "voice.refused"]) {
        stream.addEventListener(type, (event) => {
          if (!isCurrent() || sessionRef.current !== opened.session.id) return;
          try {
            handleVoiceEvent(
              JSON.parse((event as MessageEvent).data),
              setTurns,
              setSpoken,
              onHandoffRef.current,
            );
          } catch {
            // A malformed frame costs itself, not the conversation.
          }
        });
      }

      // Renewed at a third of the lease, so one missed tick is survivable and
      // two are not — which is the point of a short lease.
      const renewEvery = Math.max(5_000, (opened.lease_seconds * 1_000) / 3);
      setLeaseSeconds(opened.lease_seconds);
      leaseTimerRef.current = window.setInterval(() => {
        const id = sessionRef.current;
        if (!id) return;
        void renewVoiceSession(id)
          .then((renewed) => {
            if (!isCurrent() || sessionRef.current !== opened.session.id) return;
            setSession(renewed);
            if (renewed.state !== "live") {
              if (renewed.reason.includes("no voice activity")) {
                setError("Voice stopped after two minutes without activity.");
                teardown("idle", "idle_timeout");
              } else {
                teardown("idle", "connection_lost");
              }
            }
          })
          .catch(() => {
            if (!isCurrent() || sessionRef.current !== opened.session.id) return;
            setError("Metis stopped renewing this session.");
            teardown("failed", "connection_lost");
          });
      }, renewEvery);

      tickRef.current = window.setInterval(() => {
        if (!isCurrent() || sessionRef.current !== opened.session.id) return;
        const now = Date.now();
        setElapsed((now - startedAtRef.current) / 1000);
        const remaining = Math.max(
          0,
          Math.ceil(
            idleTimeoutRef.current - (now - activityAtRef.current) / 1_000,
          ),
        );
        setIdleSecondsRemaining(remaining);
        if (remaining === 0) {
          setError("Voice stopped after two minutes without activity.");
          teardownRef.current("idle", "idle_timeout");
        }
      }, 1_000);

      connectTimerRef.current = window.setTimeout(() => {
        if (!isCurrent() || connectedOnceRef.current) return;
        setError("Voice could not connect within 25 seconds. Try again.");
        teardownRef.current("failed", "connection_lost");
      }, 25_000);

      await conversationRef.current.startSession({
        conversationToken: opened.conversation_token,
        connectionType: "webrtc",
      });
      if (!isCurrent() || sessionRef.current !== opened.session.id) {
        try {
          conversationRef.current.endSession();
        } catch {
          // It was already closed by teardown.
        }
        void endVoiceSession(opened.session.id).catch(() => undefined);
        return;
      }
      try {
        conversationRef.current.setVolume({ volume: outputVolumeRef.current });
      } catch {
        // Some audio outputs are attached on the first provider frame. The
        // control remains available and can apply the remembered value then.
      }
    } catch (startError) {
      if (!isCurrent()) return;
      setError(
        startError instanceof Error ? startError.message : "Voice could not start.",
      );
      teardown("failed", "connection_lost");
    } finally {
      if (generationRef.current === generation) startPendingRef.current = false;
    }
  }, [markActivity, teardown]);

  // Unmount, navigation, page hide and a backgrounded tab all end it. Browser
  // audio in a hidden tab is too easy to forget and too hard to notice.
  useEffect(() => {
    mountedRef.current = true;
    const leave = () => teardown("idle", null);
    const visibilityChanged = () => {
      if (
        document.visibilityState !== "hidden"
        || (!sessionRef.current && !startPendingRef.current)
      ) {
        return;
      }
      setError("Voice stopped when this tab moved to the background.");
      teardown("idle", "background");
    };
    window.addEventListener("pagehide", leave);
    document.addEventListener("visibilitychange", visibilityChanged);
    return () => {
      window.removeEventListener("pagehide", leave);
      document.removeEventListener("visibilitychange", visibilityChanged);
      mountedRef.current = false;
      leave();
    };
  }, [teardown]);

  return {
    state,
    session,
    turns,
    error,
    elapsed,
    leaseSeconds,
    liveTranscript,
    spoken,
    idleTimeoutSeconds,
    idleSecondsRemaining,
    endReason,
    outputVolume,
    isMuted: conversation.isMuted,
    setMuted: conversation.setMuted,
    setOutputVolume,
    getInputVolume: () => {
      try {
        return conversationRef.current.getInputVolume();
      } catch {
        return 0;
      }
    },
    getOutputVolume: () => {
      try {
        return conversationRef.current.getOutputVolume();
      } catch {
        return 0;
      }
    },
    start,
    stop,
    dismissError: () => setError(null),
  };
}

function handleVoiceEvent(
  event: Record<string, unknown>,
  setTurns: (update: (current: VoiceTurnRecord[]) => VoiceTurnRecord[]) => void,
  setSpoken: (update: (current: string[]) => string[]) => void,
  onHandoff: ((handoff: VoiceHandoff) => void) | undefined,
): void {
  const type = String(event.type ?? "");
  if (type === "voice.spoken") {
    const text = String(event.text ?? "").trim();
    if (text) setSpoken((current) => [...current, text]);
    return;
  }
  if (type === "voice.turn") {
    const rendition = event.rendition as VoiceRendition | undefined;
    if (!rendition) return;
    setSpoken(() => []);
    setTurns((current) => [
      ...current,
      { id: rendition.turn_id, transcript: rendition.transcript, rendition },
    ]);
    return;
  }
  if (type === "voice.build_deferred" || type === "voice.refused") {
    if (event.hand_off === false) return;
    onHandoff?.({
      transcript: String(event.transcript ?? ""),
      turnId: String(event.turn_id ?? ""),
    });
  }
}
