import type { SVGProps } from "react";

type MetisMarkProps = SVGProps<SVGSVGElement> & {
  animated?: boolean;
};

export function MetisWordmark({ className = "" }: { className?: string }) {
  return <span className={`metisWordmark ${className}`.trim()}>Metis</span>;
}

/**
 * The compact four-shape Metis mark used across navigation, welcome, and chat.
 * Animation is CSS-only so it stays light, offline, and reduced-motion aware.
 */
export function MetisMark({ animated = false, className = "", ...props }: MetisMarkProps) {
  return (
    <svg
      viewBox="0 0 40 40"
      focusable="false"
      aria-hidden="true"
      className={`${animated ? "metisMarkAnimated" : ""} ${className}`.trim()}
      {...props}
    >
      <rect className="metisShape metisShapeCobalt" x="2" y="2" width="16.6" height="16.6" rx="5.4" fill="#6175e8" />
      <circle className="metisShape metisShapeLavender" cx="30" cy="10.3" r="8.3" fill="#c28fe0" />
      <rect className="metisShape metisShapeCoral" x="2" y="21.4" width="16.6" height="16.6" rx="5.4" fill="#ff7b61" />
      <circle className="metisShape metisShapeInk" cx="30" cy="30" r="3.7" fill="#4b9b72" />
    </svg>
  );
}

export function MetisCompanion() {
  return (
    <div className="metisCompanion" aria-hidden="true">
      <span className="companionHalo" />
      <span className="companionOrbit companionOrbitCobalt"><i /></span>
      <span className="companionOrbit companionOrbitCoral"><i /></span>
      <MetisMark animated />
      <span className="companionShadow" />
    </div>
  );
}
