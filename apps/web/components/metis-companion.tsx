export type CompanionMood = "idle" | "listening" | "thinking" | "done" | "trouble";

/**
 * The companion — a small luminous creature that lives beside the conversation.
 *
 * Liquid-glass material: a dark body with a colour field drifting inside it, a
 * bright rim, and a travelling specular highlight. No Voronoi seams — they read
 * as cracks — and no face.
 *
 * It is alive without one. The character comes from three things a creature
 * does and a shape does not: it breathes (the silhouette itself morphs, rather
 * than scaling), it has weight (it bobs above a shadow that tightens as it
 * rises), and it attends to you — the bright core inside drifts toward whatever
 * is happening, concentrating when it works and going still when something
 * breaks.
 *
 * Every mood is CSS on [data-mood]; nothing animates from JavaScript.
 */
export function MetisCompanion({
  mood = "idle",
  className = "",
}: {
  mood?: CompanionMood;
  className?: string;
}) {
  return (
    <div className={`companion ${className}`.trim()} data-mood={mood} aria-hidden="true">
      <div className="companionBody">
        {/* The colour field — masses drifting and merging behind the glass. */}
        <div className="companionField">
          <span className="fieldCell cellCoral" />
          <span className="fieldCell cellCobalt" />
          <span className="fieldCell cellGreen" />
          <span className="fieldCell cellLilac" />
        </div>
        {/* The core: where its attention is. */}
        <span className="companionCore" />
        <div className="companionGlass" />
        <div className="companionSpecular" />
      </div>
      <span className="companionShadow" />
    </div>
  );
}
