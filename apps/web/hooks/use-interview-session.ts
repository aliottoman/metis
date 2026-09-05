"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useConversation } from "@elevenlabs/react";

import {
  analyzeInterviewDelivery,
  appendInterviewTurns,
  attachInterviewConversation,
  endInterviewSession,
  getInterviewSession,
  startInterviewSession,
  submitInterviewEvaluation,
} from "@/lib/api";
import {
  INTERVIEW_QUESTION_LIMIT,
  announcesEvaluation,
  parseInterviewEvaluation,
  questionNumberFrom,
} from "@/lib/interviews";
import type {
  InterviewContext,
  InterviewDelivery,
  InterviewDeliveryStage,
  InterviewScorecard,
  InterviewSession,
} from "@/lib/types";

/**
 * One five-question interview, from the browser's side.
 *
 * Deliberately simpler than voice mode's hook: the agent runs on ElevenLabs'
 * hosted model, so there is no lease to renew and no loopback stream to hold
 * open. What this hook owns instead is the product state machine — ten
 * explicit phases, never inferred from the SDK's connection status alone —
 * and the one channel back into Metis: the blocking
 * `submit_interview_evaluation` client tool, whose round trip through the API
 * is what turns five criterion scores into the overall score Chiron speaks.
 *
 * Teardown is over-covered on purpose: explicit stop, unmount, navigation,
 * `pagehide`, and a failed connection all end the session, and the server
 * treats a repeated end as a no-op rather than a conflict.
 */

export type InterviewPhase =
  | "setup"
  | "ready"
  | "connecting"
  | "listening"
  | "agent_speaking"
  | "evaluating"
  | "debriefing"
  | "complete"
  | "failed"
  | "ended_early";

export interface LiveInterviewTurn {
  ordinal: number;
  role: "user" | "agent";
  text: string;
}

/** Phases in which the interview conversation is actually running. */
const LIVE_PHASES: ReadonlySet<InterviewPhase> = new Set([
  "connecting",
  "listening",
  "agent_speaking",
  "evaluating",
  "debriefing",
]);

/** Give the hosted agent time to confirm and score, then make retry available. */
const FINISH_REQUEST_TIMEOUT_MS = 60_000;

export function useInterviewSession(options: { context: InterviewContext | null }) {
  const [phase, setPhaseState] = useState<InterviewPhase>("setup");
  const [session, setSession] = useState<InterviewSession | null>(null);
  const [scorecard, setScorecard] = useState<InterviewScorecard | null>(null);
  const [turns, setTurns] = useState<LiveInterviewTurn[]>([]);
  const [questionNumber, setQuestionNumber] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [delivery, setDelivery] = useState<InterviewDelivery | null>(null);
  const [deliveryStage, setDeliveryStage] = useState<InterviewDeliveryStage>("");
  const [deliveryBusy, setDeliveryBusy] = useState(false);
  const [latestQuestion, setLatestQuestion] = useState("");
  const [candidateAnswerCount, setCandidateAnswerCount] = useState(0);
  const [finishRequested, setFinishRequested] = useState(false);
  const [outputVolume, setOutputVolumeState] = useState(1);
  const [audioPaused, setAudioPausedState] = useState(false);

  const phaseRef = useRef<InterviewPhase>("setup");
  const sessionRef = useRef<string | null>(null);
  // Bumped whenever delivery state is cleared for a new round, so a slow
  // in-flight analysis can never write a previous session's numbers into
  // the next one.
  const deliveryEpochRef = useRef(0);
  const scorecardRef = useRef<InterviewScorecard | null>(null);
  const endRequestedRef = useRef(false);
  const ordinalRef = useRef(0);
  const lastTypedRef = useRef<string | null>(null);
  const startedAtRef = useRef(0);
  const tickRef = useRef<number | null>(null);
  const contextRef = useRef(options.context);
  contextRef.current = options.context;
  const teardownRef = useRef<(next: InterviewPhase) => void>(() => undefined);
  const outputVolumeRef = useRef(1);
  const audioPausedRef = useRef(false);
  const mutedBeforePauseRef = useRef(false);
  const activeQuestionNumberRef = useRef<number | null>(null);
  const answeredQuestionNumbersRef = useRef<Set<number>>(new Set());
  const finishRequestedRef = useRef(false);
  const finishRequestTimerRef = useRef<number | null>(null);
  const generationRef = useRef(0);
  const startPendingRef = useRef(false);
  const mountedRef = useRef(true);

  const movePhase = useCallback((next: InterviewPhase) => {
    phaseRef.current = next;
    setPhaseState(next);
  }, []);

  const clearFinishRequest = useCallback(() => {
    if (finishRequestTimerRef.current !== null) {
      window.clearTimeout(finishRequestTimerRef.current);
      finishRequestTimerRef.current = null;
    }
    finishRequestedRef.current = false;
    setFinishRequested(false);
  }, []);

  /** Count one answer per numbered question, never raw transcript messages. */
  const markCurrentQuestionAnswered = useCallback(() => {
    const question = activeQuestionNumberRef.current;
    if (
      question === null ||
      question < 1 ||
      question > INTERVIEW_QUESTION_LIMIT ||
      answeredQuestionNumbersRef.current.has(question)
    ) {
      return;
    }
    answeredQuestionNumbersRef.current.add(question);
    setCandidateAnswerCount(answeredQuestionNumbersRef.current.size);
  }, []);

  // Before a session exists, the phase simply mirrors whether the form is
  // complete: setup while something is missing, ready once it can start.
  useEffect(() => {
    if (phaseRef.current !== "setup" && phaseRef.current !== "ready") return;
    movePhase(options.context ? "ready" : "setup");
  }, [options.context, movePhase]);

  /** Record one transcript line locally and ship it; re-sends are absorbed. */
  const record = useCallback((role: "user" | "agent", text: string) => {
    const ordinal = ordinalRef.current;
    ordinalRef.current += 1;
    setTurns((current) => [...current, { ordinal, role, text }]);
    const id = sessionRef.current;
    if (id) {
      void appendInterviewTurns(id, [{ ordinal, role, text }]).catch(() => undefined);
    }
  }, []);

  const conversation = useConversation({
    volume: audioPaused ? 0 : outputVolume,
    onConnect: ({ conversationId }: { conversationId: string }) => {
      const id = sessionRef.current;
      if (!id) return;
      void attachInterviewConversation(id, conversationId).catch(() => undefined);
      if (phaseRef.current === "connecting") movePhase("listening");
    },
    onDisconnect: () => {
      if (!sessionRef.current) return;
      if (scorecardRef.current) {
        teardownRef.current("complete");
        return;
      }
      if (endRequestedRef.current) {
        teardownRef.current("ended_early");
        return;
      }
      setError(
        "The connection ended before the evaluation arrived. Your transcript is kept below.",
      );
      teardownRef.current("failed");
    },
    onError: (message: string, context?: unknown) => {
      // A failed client-tool call is reported here too. That one is the
      // agent's to retry — it received the error as the tool result — and
      // must not tear down a live interview.
      const toolFailure =
        typeof context === "object" && context !== null && "clientToolName" in context;
      if (toolFailure) {
        setError(message || "The evaluation could not be saved.");
        return;
      }
      if (scorecardRef.current) {
        setError(message || "The spoken debrief ended early. Your scorecard is ready.");
        teardownRef.current("complete");
        return;
      }
      setError(message || "The interview connection failed.");
      if (sessionRef.current) teardownRef.current("failed");
    },
    onModeChange: ({ mode }: { mode: "speaking" | "listening" }) => {
      const current = phaseRef.current;
      if (current !== "listening" && current !== "agent_speaking") return;
      movePhase(mode === "speaking" ? "agent_speaking" : "listening");
    },
    onMessage: ({ message, role }: { message: string; role: "user" | "agent" }) => {
      if (!sessionRef.current) return;
      if (role === "user") {
        // A typed answer was already recorded when it was sent; the platform
        // may echo it back as a user transcript, and once is enough.
        if (lastTypedRef.current && message === lastTypedRef.current) {
          lastTypedRef.current = null;
          return;
        }
        record("user", message);
        markCurrentQuestionAnswered();
        return;
      }
      record("agent", message);
      const announced = questionNumberFrom(message);
      if (announced !== null) {
        activeQuestionNumberRef.current = announced;
        setQuestionNumber((current) => Math.max(current, announced));
        setLatestQuestion(message);
      }
      if (
        announcesEvaluation(message) &&
        !scorecardRef.current &&
        LIVE_PHASES.has(phaseRef.current)
      ) {
        movePhase("evaluating");
      }
    },
  });
  // The SDK hook may return a fresh facade while its connection state
  // changes; teardown must always use the latest one without itself changing
  // identity, or React would end the session it just opened.
  const conversationRef = useRef(conversation);
  conversationRef.current = conversation;

  /** Set the speaker level without losing the chosen level while paused. */
  const setOutputVolume = useCallback((volume: number) => {
    const next = Math.min(1, Math.max(0, volume));
    outputVolumeRef.current = next;
    setOutputVolumeState(next);
    if (audioPausedRef.current) return;
    try {
      conversationRef.current.setVolume({ volume: next });
    } catch {
      // The preference is kept and passed into the next connection.
    }
  }, []);

  /**
   * A local audio pause: keep the interview room alive, but silence its
   * output and stop sending microphone audio until the candidate resumes.
   */
  const toggleAudioPaused = useCallback(() => {
    const next = !audioPausedRef.current;
    audioPausedRef.current = next;
    setAudioPausedState(next);
    try {
      if (next) {
        mutedBeforePauseRef.current = conversationRef.current.isMuted;
        conversationRef.current.setMuted(true);
        conversationRef.current.setVolume({ volume: 0 });
      } else {
        conversationRef.current.setMuted(mutedBeforePauseRef.current);
        conversationRef.current.setVolume({ volume: outputVolumeRef.current });
      }
    } catch {
      // A disconnect racing the control is harmless; teardown settles it.
    }
  }, []);

  /** Preserve the user's mute preference when it changes outside a pause. */
  const setMuted = useCallback((muted: boolean) => {
    if (audioPausedRef.current) {
      mutedBeforePauseRef.current = muted;
      return;
    }
    conversationRef.current.setMuted(muted);
  }, []);

  /** Everything that must stop, and the server told once, idempotently. */
  const teardown = useCallback(
    (next: InterviewPhase) => {
      generationRef.current += 1;
      startPendingRef.current = false;
      if (tickRef.current !== null) window.clearInterval(tickRef.current);
      tickRef.current = null;
      const id = sessionRef.current;
      sessionRef.current = null;
      clearFinishRequest();
      const restoreAudioAfterEnd = audioPausedRef.current;
      try {
        conversationRef.current.endSession();
      } catch {
        // Already gone; the session row settles through the end call below.
      }
      if (restoreAudioAfterEnd) {
        audioPausedRef.current = false;
        setAudioPausedState(false);
        try {
          // Restore preferences only after the live room has been torn down,
          // so neither the microphone nor speaker can blip on the way out.
          conversationRef.current.setMuted(mutedBeforePauseRef.current);
          conversationRef.current.setVolume({ volume: outputVolumeRef.current });
        } catch {
          // The provider may already have released its media devices.
        }
      }
      if (id) {
        const reason = scorecardRef.current
          ? "completed"
          : next === "failed"
            ? "failed"
            : "ended_early";
        void endInterviewSession(id, reason).catch(() => undefined);
      }
      movePhase(next);
    },
    [clearFinishRequest, movePhase],
  );
  teardownRef.current = teardown;

  /** The blocking client tool. Its return value is what Chiron speaks. */
  const handleEvaluation = useCallback(
    async (parameters: Record<string, unknown>): Promise<string> => {
      const id = sessionRef.current;
      if (!id) throw new Error("No interview session is open.");
      if (LIVE_PHASES.has(phaseRef.current)) movePhase("evaluating");
      // Throws on a malformed payload; the SDK hands the message back to the
      // agent as an errored tool result, and the interview stays alive.
      const evaluation = parseInterviewEvaluation(parameters);
      const stored = await submitInterviewEvaluation(id, evaluation);
      scorecardRef.current = stored;
      setScorecard(stored);
      clearFinishRequest();
      setError(null);
      // The tool result is the data Chiron uses for its spoken verdict. Keep
      // the room visible until the agent hangs up, or the user explicitly
      // chooses to leave the spoken debrief and view the scorecard now.
      movePhase("debriefing");
      return JSON.stringify({
        overall_score: stored.overall_score,
        recommendation: stored.recommendation,
        provisional: stored.provisional,
      });
    },
    [clearFinishRequest, movePhase],
  );
  const handleEvaluationRef = useRef(handleEvaluation);
  handleEvaluationRef.current = handleEvaluation;

  const start = useCallback(async () => {
    if (sessionRef.current || startPendingRef.current) return;
    const context = contextRef.current;
    if (!context) return;
    const generation = generationRef.current + 1;
    generationRef.current = generation;
    startPendingRef.current = true;
    const isCurrent = () =>
      mountedRef.current && generationRef.current === generation;
    setError(null);
    setTurns([]);
    setScorecard(null);
    scorecardRef.current = null;
    setDelivery(null);
    setDeliveryStage("");
    deliveryEpochRef.current += 1;
    endRequestedRef.current = false;
    lastTypedRef.current = null;
    ordinalRef.current = 0;
    activeQuestionNumberRef.current = null;
    answeredQuestionNumbersRef.current = new Set();
    setCandidateAnswerCount(0);
    setQuestionNumber(0);
    setLatestQuestion("");
    clearFinishRequest();
    setElapsed(0);
    movePhase("connecting");

    // The microphone is asked for first, on its own, because the SDK's error
    // for a denied microphone is generic. Denial must land as a recoverable
    // state with instructions, not as a dead orb.
    try {
      const probe = await navigator.mediaDevices.getUserMedia({ audio: true });
      probe.getTracks().forEach((track) => track.stop());
      if (!isCurrent()) return;
    } catch {
      if (!isCurrent()) return;
      setError(
        "Microphone access was denied. Allow the microphone for this site in the browser's address bar, then try again.",
      );
      startPendingRef.current = false;
      movePhase("failed");
      return;
    }

    try {
      const opened = await startInterviewSession(context);
      if (!isCurrent()) {
        void endInterviewSession(opened.session.id, "ended_early").catch(() => undefined);
        return;
      }
      sessionRef.current = opened.session.id;
      setSession(opened.session);
      startedAtRef.current = Date.now();
      tickRef.current = window.setInterval(
        () => setElapsed((Date.now() - startedAtRef.current) / 1000),
        1_000,
      );
      await conversationRef.current.startSession({
        conversationToken: opened.conversation_token,
        connectionType: "webrtc",
        dynamicVariables: opened.dynamic_variables,
        clientTools: {
          submit_interview_evaluation: (parameters: Record<string, unknown>) =>
            handleEvaluationRef.current(parameters),
        },
      });
      if (!isCurrent() || sessionRef.current !== opened.session.id) {
        try {
          conversationRef.current.endSession();
        } catch {
          // Teardown already won the race.
        }
        void endInterviewSession(opened.session.id, "ended_early").catch(() => undefined);
      }
    } catch (startError) {
      if (!isCurrent()) return;
      setError(
        startError instanceof Error ? startError.message : "The interview could not start.",
      );
      teardown("failed");
    } finally {
      if (generationRef.current === generation) startPendingRef.current = false;
    }
  }, [clearFinishRequest, movePhase, teardown]);

  /** The explicit stop, after the workbench has confirmed it once. */
  const endEarly = useCallback(() => {
    endRequestedRef.current = true;
    teardown(scorecardRef.current ? "complete" : "ended_early");
  }, [teardown]);

  /** Leave a completed spoken debrief and reveal the stored scorecard. */
  const viewResults = useCallback(() => {
    if (!scorecardRef.current) return;
    teardown("complete");
  }, [teardown]);

  /** The typed fallback. Recorded here; a platform echo is deduplicated. */
  const sendText = useCallback(
    (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || !sessionRef.current) return;
      lastTypedRef.current = trimmed;
      record("user", trimmed);
      try {
        conversationRef.current.sendUserMessage(trimmed);
        markCurrentQuestionAnswered();
      } catch {
        setError("That message could not be sent. Say it instead.");
      }
    },
    [markCurrentQuestionAnswered, record],
  );
  const canFinishAndScore =
    candidateAnswerCount >= 3 && scorecard === null && !finishRequested;

  /**
   * Ask Chiron to take its early-stop edge. The hosted workflow owns the
   * confirmation and evaluation; the browser never fabricates a score.
   */
  const requestFinishAndScore = useCallback(() => {
    if (!sessionRef.current || !canFinishAndScore) return;
    const message =
      "I want to stop the interview now and have you evaluate what I completed. Please confirm that with me once, then finish and score the round.";
    lastTypedRef.current = message;
    record("user", message);
    setError(null);
    clearFinishRequest();
    finishRequestedRef.current = true;
    setFinishRequested(true);
    try {
      conversationRef.current.sendUserMessage(message);
      finishRequestTimerRef.current = window.setTimeout(() => {
        finishRequestTimerRef.current = null;
        if (
          !finishRequestedRef.current ||
          !sessionRef.current ||
          scorecardRef.current ||
          !LIVE_PHASES.has(phaseRef.current)
        ) {
          return;
        }
        finishRequestedRef.current = false;
        setFinishRequested(false);
        setError(
          "Chiron did not return a score. Open End interview to request scoring again, or keep going.",
        );
      }, FINISH_REQUEST_TIMEOUT_MS);
    } catch {
      clearFinishRequest();
      setError("The finish request could not be sent. Tell Chiron you want to stop instead.");
    }
  }, [canFinishAndScore, clearFinishRequest, record]);

  // The recording is analyzed in the background once the session ends — the
  // audio only exists after the call. Poll the session until the delivery
  // metrics settle, then stop; a page left open does not poll forever.
  useEffect(() => {
    if (phase !== "complete" && phase !== "ended_early") return;
    const id = session?.id;
    if (!id) return;
    if (deliveryStage === "ready" || deliveryStage === "unavailable" || deliveryStage === "failed") {
      return;
    }
    let cancelled = false;
    let attempts = 0;
    const poll = window.setInterval(() => {
      attempts += 1;
      if (attempts > 60) {
        // The backend went quiet — an unscheduled analysis or a lost end
        // call. Settle to 'failed' locally (never on the server) so the page
        // offers the retry button instead of claiming to measure forever.
        window.clearInterval(poll);
        setDeliveryStage((current) =>
          current === "" || current === "pending" ? "failed" : current,
        );
        return;
      }
      void getInterviewSession(id)
        .then((fetched) => {
          if (cancelled) return;
          setDeliveryStage(fetched.delivery_stage);
          setDelivery(fetched.delivery);
        })
        .catch(() => {
          // A missed poll is not an error surface; the next tick retries.
        });
    }, 5_000);
    return () => {
      cancelled = true;
      window.clearInterval(poll);
    };
  }, [phase, session?.id, deliveryStage]);

  /** Re-run the post-call analysis after a failure. Delivery is derived data,
   * so unlike the scorecard it is safe to compute again. */
  const retryDelivery = useCallback(async () => {
    const id = session?.id;
    if (!id || deliveryBusy) return;
    const epoch = deliveryEpochRef.current;
    setDeliveryBusy(true);
    try {
      const fetched = await analyzeInterviewDelivery(id);
      // A new round may have started while the analysis ran; its delivery
      // state is not ours to touch.
      if (deliveryEpochRef.current !== epoch) return;
      setDeliveryStage(fetched.delivery_stage);
      setDelivery(fetched.delivery);
    } catch {
      if (deliveryEpochRef.current !== epoch) return;
      setError("The delivery analysis could not run. Try it again in a moment.");
    } finally {
      setDeliveryBusy(false);
    }
  }, [session?.id, deliveryBusy]);

  /** Back to the form — after a scorecard, a failure, or an early end. */
  const reset = useCallback(() => {
    if (sessionRef.current) teardown("ended_early");
    setSession(null);
    setScorecard(null);
    scorecardRef.current = null;
    setTurns([]);
    activeQuestionNumberRef.current = null;
    answeredQuestionNumbersRef.current = new Set();
    setCandidateAnswerCount(0);
    setQuestionNumber(0);
    setLatestQuestion("");
    clearFinishRequest();
    setElapsed(0);
    setError(null);
    setDelivery(null);
    setDeliveryStage("");
    deliveryEpochRef.current += 1;
    movePhase(contextRef.current ? "ready" : "setup");
  }, [clearFinishRequest, movePhase, teardown]);

  // Unmount, navigation and page hide all end a live session. None of them
  // is trusted alone, and the server absorbs the overlap.
  useEffect(() => {
    mountedRef.current = true;
    const leave = () => {
      if (!sessionRef.current && !startPendingRef.current) return;
      teardownRef.current(scorecardRef.current ? "complete" : "ended_early");
    };
    const visibilityChanged = () => {
      if (
        document.visibilityState !== "hidden"
        || (!sessionRef.current && !startPendingRef.current)
      ) {
        return;
      }
      setError("The interview ended when this tab moved to the background.");
      leave();
    };
    window.addEventListener("pagehide", leave);
    document.addEventListener("visibilitychange", visibilityChanged);
    return () => {
      window.removeEventListener("pagehide", leave);
      document.removeEventListener("visibilitychange", visibilityChanged);
      leave();
      mountedRef.current = false;
    };
  }, []);

  return {
    phase,
    session,
    scorecard,
    turns,
    questionNumber,
    questionLimit: INTERVIEW_QUESTION_LIMIT,
    elapsed,
    error,
    delivery,
    deliveryStage,
    deliveryBusy,
    retryDelivery,
    latestQuestion,
    candidateAnswerCount,
    canFinishAndScore,
    finishRequested,
    live: LIVE_PHASES.has(phase),
    isMuted: conversation.isMuted,
    isSpeaking: conversation.isSpeaking,
    setMuted,
    outputVolume,
    setOutputVolume,
    audioPaused,
    toggleAudioPaused,
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
    endEarly,
    requestFinishAndScore,
    viewResults,
    sendText,
    reset,
    dismissError: () => setError(null),
  };
}
