"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { getSpeechPreference, transcribeAudio } from "@/lib/api";
import type { SpeechPreference } from "@/lib/types";

export type DictationState = "idle" | "recording" | "transcribing" | "unsupported";

/**
 * The mic, from the composer's point of view.
 *
 * Two things this deliberately does not do. It does not keep the stream open
 * between takes — every track is stopped on the way out, so the browser's
 * recording indicator goes off the moment the user is done rather than
 * whenever the page happens to unmount. And it does not stream partial
 * results: the whole clip goes up once, so the transcript arrives as one
 * clean sentence instead of a phrase that keeps rewriting itself under the
 * cursor while you read it.
 *
 * `onText` receives the transcript. The caller decides where it lands, which
 * is what lets dictation append to a half-typed message rather than replace it.
 *
 * `provider` is who is about to hear it. The server picks the transcriber and
 * the clip goes to the same place either way; this exists so the button can
 * name the service honestly instead of hard-coding the one that used to be
 * the only option.
 */
export function useDictation(onText: (text: string) => void) {
  const [state, setState] = useState<DictationState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [provider, setProvider] = useState<SpeechPreference["stt_provider"]>("cohere");
  // How loud it is right now, 0–1, for the level ring on the button. Kept in
  // a ref-fed state rather than recomputed from the stream by the consumer.
  const [level, setLevel] = useState(0);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const frameRef = useRef<number | null>(null);
  const discardRef = useRef(false);
  const transcriptionAbortRef = useRef<AbortController | null>(null);
  const operationRef = useRef(0);
  const mountedRef = useRef(true);
  const onTextRef = useRef(onText);
  onTextRef.current = onText;

  // A getUserMedia-less browser (or an insecure origin) should show a mic that
  // explains itself, not one that throws on click.
  useEffect(() => {
    const supported =
      typeof window !== "undefined" &&
      typeof window.MediaRecorder !== "undefined" &&
      Boolean(navigator.mediaDevices?.getUserMedia);
    if (!supported) setState("unsupported");
  }, []);

  // Read once, on mount. Changing the transcriber in Settings is rare and the
  // choice is the server's anyway — this only decides which name the button
  // shows, so a label that is one navigation stale is not worth polling for.
  useEffect(() => {
    let current = true;
    void getSpeechPreference()
      .then((preference) => { if (current) setProvider(preference.stt_provider); })
      .catch(() => undefined);
    return () => { current = false; };
  }, []);

  const teardown = useCallback(() => {
    if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
    frameRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    void audioContextRef.current?.close().catch(() => undefined);
    audioContextRef.current = null;
    if (mountedRef.current) setLevel(0);
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      operationRef.current += 1;
      discardRef.current = true;
      transcriptionAbortRef.current?.abort();
      transcriptionAbortRef.current = null;
      if (recorderRef.current?.state === "recording") recorderRef.current.stop();
      teardown();
    };
  }, [teardown]);

  const stop = useCallback(() => {
    recorderRef.current?.state === "recording" && recorderRef.current.stop();
  }, []);

  const start = useCallback(async () => {
    if (state === "recording" || state === "transcribing" || state === "unsupported") return;
    const operation = ++operationRef.current;
    discardRef.current = false;
    setError(null);
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      // Denial and absence are the same to us, and both are the user's to fix.
      setError("Microphone access was declined. Allow it in your browser settings to dictate.");
      return;
    }
    if (!mountedRef.current || operationRef.current !== operation) {
      stream.getTracks().forEach((track) => track.stop());
      return;
    }
    streamRef.current = stream;

    // The level ring. Analysed locally and never sent anywhere; it exists so
    // the button proves it is hearing you before you commit to a sentence.
    try {
      const context = new AudioContext();
      audioContextRef.current = context;
      const analyser = context.createAnalyser();
      analyser.fftSize = 512;
      context.createMediaStreamSource(stream).connect(analyser);
      const samples = new Uint8Array(analyser.frequencyBinCount);
      const tick = () => {
        analyser.getByteTimeDomainData(samples);
        let peak = 0;
        for (const sample of samples) peak = Math.max(peak, Math.abs(sample - 128) / 128);
        setLevel(peak);
        frameRef.current = requestAnimationFrame(tick);
      };
      tick();
    } catch {
      // No meter is a cosmetic loss; recording continues without it.
    }

    // Safari records mp4, Chrome and Firefox webm. Whichever the browser
    // picks is what the transcriber is told it is receiving, so the container
    // type is read back off the recorder rather than assumed here. What
    // happens next differs by service — one of them needs the clip decoded
    // first — and that decision belongs to the host, not to this button.
    const recorder = new MediaRecorder(stream);
    recorderRef.current = recorder;
    chunksRef.current = [];
    recorder.ondataavailable = (event) => {
      if (event.data.size) chunksRef.current.push(event.data);
    };
    recorder.onstop = () => {
      teardown();
      const type = recorder.mimeType || "audio/webm";
      const clip = new Blob(chunksRef.current, { type });
      chunksRef.current = [];
      recorderRef.current = null;
      if (
        discardRef.current
        || !mountedRef.current
        || operationRef.current !== operation
      ) {
        if (mountedRef.current) setState("idle");
        return;
      }
      if (clip.size < 1024) {
        // A tap rather than a hold. Nothing was said, so nothing is sent.
        setState("idle");
        return;
      }
      setState("transcribing");
      const extension = type.includes("mp4") ? "m4a" : type.includes("ogg") ? "ogg" : "webm";
      const controller = new AbortController();
      transcriptionAbortRef.current = controller;
      void transcribeAudio(clip, `dictation.${extension}`, controller.signal)
        .then((text) => {
          if (
            mountedRef.current
            && operationRef.current === operation
            && text.trim()
          ) {
            onTextRef.current(text.trim());
          }
        })
        .catch((transcribeError: unknown) => {
          if (transcribeError instanceof DOMException && transcribeError.name === "AbortError") return;
          if (!mountedRef.current || operationRef.current !== operation) return;
          setError(
            transcribeError instanceof Error
              ? transcribeError.message
              : "That recording could not be transcribed.",
          );
        })
        .finally(() => {
          if (transcriptionAbortRef.current === controller) {
            transcriptionAbortRef.current = null;
          }
          if (mountedRef.current && operationRef.current === operation) setState("idle");
        });
    };
    recorder.start();
    setState("recording");
  }, [state, teardown]);

  const cancel = useCallback(() => {
    operationRef.current += 1;
    discardRef.current = true;
    transcriptionAbortRef.current?.abort();
    transcriptionAbortRef.current = null;
    chunksRef.current = [];
    if (recorderRef.current?.state === "recording") recorderRef.current.stop();
    // `MediaRecorder.stop()` dispatches `onstop` asynchronously. Release the
    // actual microphone tracks now so a caller can safely hand audio ownership
    // to live Voice immediately after cancel returns.
    teardown();
    if (mountedRef.current) setState("idle");
  }, [teardown]);

  const toggle = useCallback(() => {
    if (state === "recording") stop();
    else void start();
  }, [start, state, stop]);

  return {
    state,
    level,
    error,
    provider,
    toggle,
    cancel,
    dismissError: () => setError(null),
  };
}
