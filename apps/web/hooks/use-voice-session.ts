"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useConversation } from "@elevenlabs/react";

import {
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

  const sessionRef = useRef<string | null>(null);
  const leaseTimerRef = useRef<number | null>(null);
  const tickRef = useRef<number | null>(null);
  const streamRef = useRef<EventSource | null>(null);
  const startedAtRef = useRef<number>(0);
  const onHandoffRef = useRef(options.onHandoff);
  onHandoffRef.current = options.onHandoff;

  const conversation = useConversation({
    onConnect: () => setState("listening"),
    onDisconnect: () => setState((current) => (current === "failed" ? current : "idle")),
    onError: (message: string) => {
      setError(message || "The voice connection failed.");
      setState("failed");
    },
    onModeChange: ({ mode }: { mode: "speaking" | "listening" }) =>
      setState((current) =>
        current === "failed" || current === "idle" ? current : mode === "speaking" ? "speaking" : "listening",
      ),
    onStatusChange: ({ status }: { status: string }) => {
      if (status === "connecting") setState("connecting");
      if (status === "disconnected") setState((current) => (current === "failed" ? current : "idle"));
    },
  });

  /** Everything that must stop, in the order it must stop in. */
  const teardown = useCallback(
    (nextState: VoiceState = "idle") => {
      if (leaseTimerRef.current !== null) window.clearInterval(leaseTimerRef.current);
      if (tickRef.current !== null) window.clearInterval(tickRef.current);
      leaseTimerRef.current = null;
      tickRef.current = null;
      streamRef.current?.close();
      streamRef.current = null;
      const id = sessionRef.current;
      sessionRef.current = null;
      try {
        conversation.endSession();
      } catch {
        // Already gone. The server lease is what actually closes the tunnel.
      }
      if (id) void endVoiceSession(id).catch(() => undefined);
      setState(nextState);
      setLeaseSeconds(0);
    },
    [conversation],
  );

  const stop = useCallback(() => teardown("idle"), [teardown]);

  const start = useCallback(async () => {
    if (sessionRef.current) return;
    setError(null);
    setTurns([]);
    setState("connecting");
    try {
      const opened = await startVoiceSession();
      sessionRef.current = opened.session.id;
      setSession(opened.session);
      startedAtRef.current = Date.now();
      setElapsed(0);

      // The loopback channel, opened before the audio one: a refusal can
      // arrive on the very first turn, and it must have somewhere to land.
      const stream = new EventSource(voiceEventsUrl(opened.session.id));
      streamRef.current = stream;
      stream.onmessage = (event) => {
        try {
          handleVoiceEvent(JSON.parse(event.data), setTurns, onHandoffRef.current);
        } catch {
          // A malformed frame costs itself, not the conversation.
        }
      };
      stream.addEventListener("voice.turn", (event) =>
        handleVoiceEvent(JSON.parse((event as MessageEvent).data), setTurns, onHandoffRef.current),
      );
      stream.addEventListener("voice.build_deferred", (event) =>
        handleVoiceEvent(JSON.parse((event as MessageEvent).data), setTurns, onHandoffRef.current),
      );
      stream.addEventListener("voice.refused", (event) =>
        handleVoiceEvent(JSON.parse((event as MessageEvent).data), setTurns, onHandoffRef.current),
      );

      // Renewed at a third of the lease, so one missed tick is survivable and
      // two are not — which is the point of a short lease.
      const renewEvery = Math.max(5_000, (opened.lease_seconds * 1_000) / 3);
      setLeaseSeconds(opened.lease_seconds);
      leaseTimerRef.current = window.setInterval(() => {
        const id = sessionRef.current;
        if (!id) return;
        void renewVoiceSession(id)
          .then((renewed) => {
            setSession(renewed);
            if (renewed.state !== "live") teardown("idle");
          })
          .catch(() => {
            setError("Metis stopped renewing this session.");
            teardown("failed");
          });
      }, renewEvery);

      tickRef.current = window.setInterval(
        () => setElapsed((Date.now() - startedAtRef.current) / 1000),
        1_000,
      );

      await conversation.startSession({
        conversationToken: opened.conversation_token,
        connectionType: "webrtc",
      });
    } catch (startError) {
      setError(
        startError instanceof Error ? startError.message : "Voice could not start.",
      );
      teardown("failed");
    }
  }, [conversation, teardown]);

  /** Cut the agent off mid-sentence. */
  const interrupt = useCallback(() => {
    try {
      conversation.sendUserActivity();
    } catch {
      // Nothing to interrupt.
    }
  }, [conversation]);

  // Unmount, navigation and page hide all end it. Each is a case where the
  // user has plainly stopped talking, and none of them is trusted alone.
  useEffect(() => {
    const leave = () => teardown("idle");
    window.addEventListener("pagehide", leave);
    return () => {
      window.removeEventListener("pagehide", leave);
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
    isMuted: conversation.isMuted,
    setMuted: conversation.setMuted,
    start,
    stop,
    interrupt,
    dismissError: () => setError(null),
  };
}

function handleVoiceEvent(
  event: Record<string, unknown>,
  setTurns: (update: (current: VoiceTurnRecord[]) => VoiceTurnRecord[]) => void,
  onHandoff: ((handoff: VoiceHandoff) => void) | undefined,
): void {
  const type = String(event.type ?? "");
  if (type === "voice.turn") {
    const rendition = event.rendition as VoiceRendition | undefined;
    if (!rendition) return;
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
