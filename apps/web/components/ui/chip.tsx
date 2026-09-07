// A small labelled pill: a scope, a tag, a count. Never a button.

import type { ReactNode } from "react";

export function Chip({
  children,
  count,
  tone = "default",
}: {
  children: ReactNode;
  count?: number;
  tone?: "default" | "accent" | "outline";
}) {
  return (
    <span className={`ui-chip${tone !== "default" ? ` is-${tone}` : ""}`}>
      {children}
      {count !== undefined ? <small>{count}</small> : null}
    </span>
  );
}
