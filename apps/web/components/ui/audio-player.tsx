"use client";

// One player for every recording: play, ten seconds either way, a scrubber
// with the time, speed. A page that needs to follow along (a transcript that
// highlights words) reads the time and seeks through the handle.

import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { Pause, Play } from "lucide-react";

const SPEEDS = [1, 1.25, 1.5, 2];
const SKIP_SECONDS = 10;

export interface AudioPlayerHandle {
  seek(seconds: number): void;
}

function clock(seconds: number): string {
  if (!Number.isFinite(seconds)) return "–:––";
  const whole = Math.max(0, Math.floor(seconds));
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

export const AudioPlayer = forwardRef<AudioPlayerHandle, { src: string; label: string; onTime?: (seconds: number) => void; onError?: () => void }>(
  function AudioPlayer({ src, label, onTime, onError }, handle) {
    const audio = useRef<HTMLAudioElement | null>(null);
    const pendingSeek = useRef<number | null>(null);
    const [playing, setPlaying] = useState(false);
    const [time, setTime] = useState(0);
    const [duration, setDuration] = useState(NaN);
    const [speed, setSpeed] = useState(1);

    useEffect(() => {
      setPlaying(false);
      setTime(0);
      setDuration(NaN);
      pendingSeek.current = null;
    }, [src]);

    useEffect(() => {
      if (audio.current) audio.current.playbackRate = speed;
    }, [speed, src]);

    const seek = (seconds: number) => {
      const element = audio.current;
      // A transcript can ask to seek before metadata has arrived. An invalid
      // currentTime throws and should never break the rest of the workbench.
      if (!element || !Number.isFinite(seconds)) return false;
      if (!Number.isFinite(element.duration) || element.duration <= 0) {
        pendingSeek.current = seconds;
        return true;
      }
      const next = Math.max(0, Math.min(element.duration, seconds));
      element.currentTime = next;
      pendingSeek.current = null;
      setTime(next);
      onTime?.(next);
      return true;
    };
    const play = () => {
      const element = audio.current;
      if (!element) return;
      void element.play().catch((error: unknown) => {
        // Replacing the recording cancels its pending play request normally.
        if (audio.current !== element || (error instanceof DOMException && error.name === "AbortError")) return;
        setPlaying(false);
        onError?.();
      });
    };
    useImperativeHandle(handle, () => ({
      seek(seconds) {
        if (seek(seconds)) play();
      },
    }));

    const seekable = Number.isFinite(duration) && duration > 0;

    return (
      <div className="ui-audio" role="group" aria-label={label}>
        <audio
          key={src}
          ref={audio}
          src={src}
          preload="metadata"
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onEnded={() => setPlaying(false)}
          onTimeUpdate={(event) => { setTime(event.currentTarget.currentTime); onTime?.(event.currentTarget.currentTime); }}
          onLoadedMetadata={(event) => {
            setDuration(event.currentTarget.duration);
            if (pendingSeek.current !== null) seek(pendingSeek.current);
          }}
          onDurationChange={(event) => setDuration(event.currentTarget.duration)}
          onError={() => { setPlaying(false); onError?.(); }}
        />
        <button type="button" className="ui-audio-play" aria-label={playing ? `Pause ${label}` : `Play ${label}`} onClick={() => (playing ? audio.current?.pause() : play())}>
          {playing ? <Pause size={16} aria-hidden="true" /> : <Play size={16} aria-hidden="true" />}
        </button>
        <button type="button" className="ui-btn is-quiet is-sm" disabled={!seekable || time <= 0} onClick={() => seek(time - SKIP_SECONDS)} aria-label={`Back ${SKIP_SECONDS} seconds`}>−{SKIP_SECONDS}</button>
        <input type="range" min={0} max={seekable ? duration : 0} step={0.5} value={seekable ? Math.min(time, duration) : 0} disabled={!seekable} aria-label={`${label} position`} aria-valuetext={`${clock(time)} of ${clock(duration)}`} onChange={(event) => seek(Number(event.target.value))} />
        <button type="button" className="ui-btn is-quiet is-sm" disabled={!seekable || time >= duration} onClick={() => seek(time + SKIP_SECONDS)} aria-label={`Forward ${SKIP_SECONDS} seconds`}>+{SKIP_SECONDS}</button>
        <span className="ui-audio-time">{clock(time)} / {clock(duration)}</span>
        <button type="button" className="ui-btn is-quiet is-sm" onClick={() => setSpeed(SPEEDS[(SPEEDS.indexOf(speed) + 1) % SPEEDS.length])} aria-label={`Playback speed ${speed} times. Change speed`}>{speed}×</button>
      </div>
    );
  },
);
