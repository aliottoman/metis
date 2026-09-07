// A label over a control, with room for the one problem it might have.

import type { ReactNode } from "react";

export function Field({
  label,
  children,
  problem,
  className,
}: {
  label: string;
  children: ReactNode;
  problem?: string | null;
  className?: string;
}) {
  return (
    <label className={`ui-field${className ? ` ${className}` : ""}`}>
      <span>{label}</span>
      {children}
      {problem ? <small role="alert">{problem}</small> : null}
    </label>
  );
}
