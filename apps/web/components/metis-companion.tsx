"use client";

import { useEffect, useRef } from "react";

export type CompanionMood = "idle" | "listening" | "thinking" | "done" | "trouble";

/**
 * The companion — a small luminous creature that lives beside the conversation.
 *
 * An iridescent bead: an oil film turns inside a lit body, so the surface
 * shifts hue as it moves rather than showing the gradient spokes a conic fill
 * gives you.
 *
 * It is alive without a face, for three reasons. It breathes — the silhouette
 * itself morphs rather than scaling. It has weight — it bobs above a contact
 * shadow that tightens as it rises, which is most of the effect. And it
 * attends to you: the highlight tracks the pointer, so it reads as lit by the
 * room instead of by a loop.
 *
 * Every mood is CSS on [data-mood]. The pointer is the only JavaScript, and
 * the bead is complete without it.
 */
export function MetisCompanion({
  mood = "idle",
  size,
  className = "",
}: {
  mood?: CompanionMood;
  size?: number;
  className?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  // Below about 28px a morphing silhouette reads as a wobble, not a breath.
  const still = size !== undefined && size < 28;

  useEffect(() => {
    const node = ref.current;
    if (!node || still) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const onMove = (event: PointerEvent) => {
      const box = node.getBoundingClientRect();
      const dx = (event.clientX - (box.left + box.width / 2)) / 320;
      const dy = (event.clientY - (box.top + box.height / 2)) / 320;
      node.style.setProperty("--mx", String(Math.max(-1, Math.min(1, dx))));
      node.style.setProperty("--my", String(Math.max(-1, Math.min(1, dy))));
    };
    window.addEventListener("pointermove", onMove, { passive: true });
    return () => window.removeEventListener("pointermove", onMove);
  }, [still]);

  return (
    <div
      ref={ref}
      className={`companion ${still ? "isSmall" : ""} ${className}`.trim()}
      data-mood={mood}
      style={size ? { width: size, height: size } : undefined}
      aria-hidden="true"
    >
      <div className="pearlBody">
        <span className="pearlFilm" />
        <span className="pearlSpec" />
      </div>
      <span className="pearlShadow" />
    </div>
  );
}
