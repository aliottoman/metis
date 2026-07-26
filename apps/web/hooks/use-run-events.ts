"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { runEventsUrl } from "@/lib/api";
import { normalizeRunEvent, parseSseBuffer } from "@/lib/sse";
import type { RunEventV1 } from "@/lib/types";

type ConnectionState = "idle" | "connecting" | "live" | "reconnecting" | "closed" | "error";

const TERMINAL_TYPES = new Set([
  "run.completed",
  "run.failed",
  "run.cancelled",
  "completed",
  "failed",
  "cancelled",
]);

export function useRunEvents(runId: string | null, onEvent?: (event: RunEventV1) => void) {
  const [events, setEvents] = useState<RunEventV1[]>([]);
  const [connection, setConnection] = useState<ConnectionState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [generation, setGeneration] = useState(0);
  const callbackRef = useRef(onEvent);
  const seenRef = useRef(new Set<string>());
  const sequenceRef = useRef(0);
  const previousRunRef = useRef<string | null>(null);

  useEffect(() => {
    callbackRef.current = onEvent;
  }, [onEvent]);

  const reconnect = useCallback(() => setGeneration((value) => value + 1), []);

  useEffect(() => {
    setError(null);
    if (previousRunRef.current !== runId) {
      setEvents([]);
      seenRef.current = new Set();
      sequenceRef.current = 0;
      previousRunRef.current = runId;
    }
    if (!runId) {
      setConnection("idle");
      return;
    }

    const controller = new AbortController();
    let stopped = false;

    async function connect() {
      let retryDelay = 700;
      setConnection("connecting");

      while (!stopped && !controller.signal.aborted) {
        try {
          const response = await fetch(runEventsUrl(runId!, sequenceRef.current), {
            headers: { accept: "text/event-stream" },
            cache: "no-store",
            signal: controller.signal,
          });
          if (!response.ok || !response.body) throw new Error(`Event stream unavailable (${response.status})`);
          setConnection("live");
          setError(null);
          retryDelay = 700;

          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          let buffer = "";
          let terminal = false;

          while (!stopped) {
            const { done, value } = await reader.read();
            buffer += decoder.decode(value, { stream: !done });
            const parsed = parseSseBuffer(buffer);
            buffer = parsed.rest;

            for (const frame of parsed.frames) {
              if (!frame.data) continue;
              let raw: unknown = frame.data;
              try {
                raw = JSON.parse(frame.data) as unknown;
              } catch {
                // Plain-text status frames are valid and normalized below.
              }
              const event = normalizeRunEvent(raw, frame, runId!, sequenceRef.current + 1);
              const key = event.id || String(event.sequence);
              if (seenRef.current.has(key)) continue;
              seenRef.current.add(key);
              sequenceRef.current = Math.max(sequenceRef.current, event.sequence);
              setEvents((current) => [...current.slice(-299), event]);
              callbackRef.current?.(event);
              if (TERMINAL_TYPES.has(event.type)) terminal = true;
            }

            if (done || terminal) break;
          }

          if (terminal) {
            setConnection("closed");
            return;
          }
        } catch (streamError) {
          if (controller.signal.aborted || stopped) return;
          setError(streamError instanceof Error ? streamError.message : "The event stream disconnected.");
        }

        setConnection("reconnecting");
        await new Promise((resolve) => window.setTimeout(resolve, retryDelay));
        retryDelay = Math.min(retryDelay * 1.8, 5000);
      }
    }

    void connect();
    return () => {
      stopped = true;
      controller.abort();
    };
  }, [runId, generation]);

  return { events, connection, error, reconnect };
}
