"use client";

// One player for every recording: play, a scrubber with the time, speed.
// Volume is the system's.

import { useEffect, useRef, useState } from "react";
import { Pause, Play } from "lucide-react";

const SPEEDS = [1, 1.25, 1.5, 2];

function clock(seconds: number): string {
  if (!Number.isFinite(seconds)) return "–:––";
  const whole = Math.floor(seconds);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

export function AudioPlayer({ src, label }: { src: string; label: string }) {
  const audio = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(NaN);
  const [speed, setSpeed] = useState(1);

  useEffect(() => {
    const element = audio.current;
    if (!element) return;
    element.playbackRate = speed;
  }, [speed]);

  return (
    <div className="ui-audio" aria-label={label}>
      <audio
        ref={audio}
        src={src}
        preload="metadata"
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
        onTimeUpdate={(event) => setTime(event.currentTarget.currentTime)}
        onLoadedMetadata={(event) => setDuration(event.currentTarget.duration)}
      />
      <button
        type="button"
        className="ui-audio-play"
        aria-label={playing ? "Pause" : "Play"}
        onClick={() => (playing ? audio.current?.pause() : void audio.current?.play())}
      >
        {playing ? <Pause size={16} /> : <Play size={16} />}
      </button>
      <input
        type="range"
        min={0}
        max={Number.isFinite(duration) ? duration : 0}
        step={0.5}
        value={time}
        aria-label="Position"
        onChange={(event) => {
          const next = Number(event.target.value);
          if (audio.current) audio.current.currentTime = next;
          setTime(next);
        }}
      />
      <span className="ui-audio-time">
        {clock(time)} / {clock(duration)}
      </span>
      <button
        type="button"
        className="ui-btn is-sm"
        onClick={() => setSpeed(SPEEDS[(SPEEDS.indexOf(speed) + 1) % SPEEDS.length])}
        aria-label="Playback speed"
      >
        {speed}×
      </button>
    </div>
  );
}
