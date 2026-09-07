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
  const whole = Math.floor(seconds);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

export const AudioPlayer = forwardRef<AudioPlayerHandle, { src: string; label: string; onTime?: (seconds: number) => void; onError?: () => void }>(
  function AudioPlayer({ src, label, onTime, onError }, handle) {
    const audio = useRef<HTMLAudioElement | null>(null);
    const [playing, setPlaying] = useState(false);
    const [time, setTime] = useState(0);
    const [duration, setDuration] = useState(NaN);
    const [speed, setSpeed] = useState(1);

    useEffect(() => {
      if (audio.current) audio.current.playbackRate = speed;
    }, [speed]);

    const seek = (seconds: number) => {
      const element = audio.current;
      if (!element) return;
      const next = Math.max(0, Math.min(Number.isFinite(element.duration) ? element.duration : seconds, seconds));
      element.currentTime = next;
      setTime(next);
    };
    useImperativeHandle(handle, () => ({
      seek(seconds) {
        seek(seconds);
        void audio.current?.play().catch(() => undefined);
      },
    }));

    return (
      <div className="ui-audio" aria-label={label}>
        <audio
          ref={audio}
          src={src}
          preload="metadata"
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onEnded={() => setPlaying(false)}
          onTimeUpdate={(event) => { setTime(event.currentTarget.currentTime); onTime?.(event.currentTarget.currentTime); }}
          onLoadedMetadata={(event) => setDuration(event.currentTarget.duration)}
          onError={onError}
        />
        <button type="button" className="ui-audio-play" aria-label={playing ? "Pause" : "Play"} onClick={() => (playing ? audio.current?.pause() : void audio.current?.play().catch(() => onError?.()))}>
          {playing ? <Pause size={16} /> : <Play size={16} />}
        </button>
        <button type="button" className="ui-btn is-quiet is-sm" onClick={() => seek(time - SKIP_SECONDS)} aria-label={`Back ${SKIP_SECONDS} seconds`}>−{SKIP_SECONDS}</button>
        <input type="range" min={0} max={Number.isFinite(duration) ? duration : 0} step={0.5} value={time} aria-label="Position" onChange={(event) => seek(Number(event.target.value))} />
        <button type="button" className="ui-btn is-quiet is-sm" onClick={() => seek(time + SKIP_SECONDS)} aria-label={`Forward ${SKIP_SECONDS} seconds`}>+{SKIP_SECONDS}</button>
        <span className="ui-audio-time">{clock(time)} / {clock(duration)}</span>
        <button type="button" className="ui-btn is-quiet is-sm" onClick={() => setSpeed(SPEEDS[(SPEEDS.indexOf(speed) + 1) % SPEEDS.length])} aria-label="Playback speed">{speed}×</button>
      </div>
    );
  },
);
