// What happened, and what to do about it. Carries its own action so an error
// is never a dead end.

import type { ReactNode } from "react";

import { Button } from "@/components/ui/button";

export function Notice({
  kind = "info",
  title,
  children,
  action,
  onAction,
  onDismiss,
}: {
  kind?: "info" | "error" | "success";
  title?: string;
  children?: ReactNode;
  /** The verb on the button, e.g. "Retry". */
  action?: string;
  onAction?: () => void;
  onDismiss?: () => void;
}) {
  return (
    <div className={`ui-notice is-${kind}`} role={kind === "error" ? "alert" : "status"}>
      <div>
        {title ? <strong>{title}</strong> : null}
        {children}
      </div>
      {action && onAction ? (
        <Button size="sm" onClick={onAction}>
          {action}
        </Button>
      ) : null}
      {onDismiss ? (
        <Button size="sm" emphasis="quiet" aria-label="Dismiss" onClick={onDismiss}>
          ×
        </Button>
      ) : null}
    </div>
  );
}
