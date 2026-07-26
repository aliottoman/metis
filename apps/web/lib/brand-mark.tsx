import type { ReactElement } from "react";

// The Metis brand cluster — a cobalt rounded square, a lavender circle, a coral
// rounded square, and a small ink dot on paper. Shared by the generated favicon
// and apple-touch icon. Explicit pixel geometry keeps it Satori-compatible.
export function brandCluster(px: number): ReactElement {
  const pad = Math.round(px * 0.16);
  const cluster = px - pad * 2;
  const gap = Math.max(2, Math.round(cluster * 0.085));
  const cell = Math.round((cluster - gap) / 2);
  const dot = Math.round(cell * 0.46);
  const round = Math.round(cell * 0.32);
  return (
    <div
      style={{
        width: px,
        height: px,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "#f2f4f3",
      }}
    >
      <div
        style={{
          width: cell * 2 + gap,
          height: cell * 2 + gap,
          display: "flex",
          flexWrap: "wrap",
          gap,
        }}
      >
        <div style={{ width: cell, height: cell, background: "#5669df", borderRadius: round }} />
        <div style={{ width: cell, height: cell, background: "#b894d8", borderRadius: cell }} />
        <div style={{ width: cell, height: cell, background: "#ff735c", borderRadius: round }} />
        <div
          style={{
            width: cell,
            height: cell,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <div style={{ width: dot, height: dot, background: "#1b1c22", borderRadius: dot }} />
        </div>
      </div>
    </div>
  );
}
