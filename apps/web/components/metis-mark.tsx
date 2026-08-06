import type { SVGProps } from "react";

type MetisMarkProps = SVGProps<SVGSVGElement> & {
  animated?: boolean;
};

export function MetisWordmark({ className = "" }: { className?: string }) {
  return <span className={`metisWordmark ${className}`.trim()}>Metis</span>;
}

/**
 * The Metis mark — the companion's bead, reduced to something that survives
 * at 20px.
 *
 * The old mark was four flat shapes in four unrelated hues, drawn for the
 * palette this product used before. It had no relationship to the creature
 * sitting at the top of the front door, so the app carried two identities.
 * This is the same object as the companion: a lit body with an oil film
 * turning inside it. Same construction, fewer layers, no filters — the warp
 * that gives the large bead its liquid edge costs more than it reads at this
 * size, so the film is drawn as three overlapping pools instead.
 */
export function MetisMark({ animated = false, className = "", ...props }: MetisMarkProps) {
  return (
    <svg
      viewBox="0 0 40 40"
      focusable="false"
      aria-hidden="true"
      className={`metisMark ${animated ? "metisMarkAnimated" : ""} ${className}`.trim()}
      {...props}
    >
      <defs>
        {/* The body: lit from the upper left, falling to lavender in shade. */}
        <radialGradient id="metisBeadBody" cx="33%" cy="25%" r="82%">
          <stop offset="0%" stopColor="#fffdfb" />
          <stop offset="38%" stopColor="#ffdccb" />
          <stop offset="70%" stopColor="#e3bbea" />
          <stop offset="100%" stopColor="#a67fc6" />
        </radialGradient>
        {/* The film: the same four hues the companion's oil film runs. */}
        <radialGradient id="metisBeadCoral" cx="72%" cy="30%" r="46%">
          <stop offset="0%" stopColor="#ff7759" stopOpacity="0.8" />
          <stop offset="100%" stopColor="#ff7759" stopOpacity="0" />
        </radialGradient>
        <radialGradient id="metisBeadTeal" cx="30%" cy="74%" r="48%">
          <stop offset="0%" stopColor="#4fd3e4" stopOpacity="0.85" />
          <stop offset="100%" stopColor="#4fd3e4" stopOpacity="0" />
        </radialGradient>
        <radialGradient id="metisBeadGold" cx="24%" cy="33%" r="36%">
          <stop offset="0%" stopColor="#ffd25a" stopOpacity="0.8" />
          <stop offset="100%" stopColor="#ffd25a" stopOpacity="0" />
        </radialGradient>
        {/* The highlight, and the shaded rim that gives the bead its round. */}
        <radialGradient id="metisBeadSpec" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#ffffff" stopOpacity="0.96" />
          <stop offset="100%" stopColor="#ffffff" stopOpacity="0" />
        </radialGradient>
        <radialGradient id="metisBeadRim" cx="38%" cy="32%" r="72%">
          <stop offset="72%" stopColor="#5f3b7a" stopOpacity="0" />
          <stop offset="100%" stopColor="#5f3b7a" stopOpacity="0.34" />
        </radialGradient>
      </defs>

      <circle cx="20" cy="20" r="18.4" fill="url(#metisBeadBody)" />
      <g className="metisBeadFilm">
        <circle cx="20" cy="20" r="18.4" fill="url(#metisBeadCoral)" />
        <circle cx="20" cy="20" r="18.4" fill="url(#metisBeadTeal)" />
        <circle cx="20" cy="20" r="18.4" fill="url(#metisBeadGold)" />
      </g>
      <circle cx="20" cy="20" r="18.4" fill="url(#metisBeadRim)" />
      <ellipse className="metisBeadSpec" cx="14.2" cy="12" rx="6.4" ry="4.3" fill="url(#metisBeadSpec)" />
    </svg>
  );
}
