import type { ApprovalRequest, RunEventV1 } from "@/lib/types";

/**
 * The actionable approval carried by an event, or null. Only the persisted
 * `approval.required` event carries an id — status events such as
 * `run.awaiting_approval` and `approval.applied` do not, so they are ignored.
 */
export function approvalFrom(event: RunEventV1): ApprovalRequest | null {
  if (event.type !== "approval.required") return null;
  const nested = event.payload.approval;
  const payload = nested && typeof nested === "object" ? nested as Record<string, unknown> : event.payload;
  const id = payload.id ?? payload.approval_id;
  if (typeof id !== "string" || !id) return null;
  return {
    id: String(id),
    run_id: event.run_id,
    title: String(payload.title ?? payload.action ?? "Approve this action?"),
    summary: String(payload.summary ?? payload.description ?? "Metis needs your permission before it can continue."),
    risk_level: payload.risk_level ? String(payload.risk_level) as ApprovalRequest["risk_level"] : undefined,
    permissions: Array.isArray(payload.permissions) ? payload.permissions.map(String) : [],
    action_digest: payload.action_digest ? String(payload.action_digest) : payload.input_digest ? String(payload.input_digest) : undefined,
    status: payload.status ? String(payload.status) as ApprovalRequest["status"] : "pending",
    blocked_reason: payload.blocked_reason ? String(payload.blocked_reason) : undefined,
  };
}

/**
 * The single approval still awaiting a decision — newest first — so the thread
 * can surface it inline without the whole timeline. Decided or non-pending
 * requests are skipped, matching what the drawer would show as resolved.
 */
export function latestPendingApproval(
  events: RunEventV1[],
  decided: ReadonlySet<string>,
): ApprovalRequest | null {
  // A finished run has nothing waiting on you, and an approval already applied
  // is resolved even if this browser was not the one that decided it. Neither
  // can be read from the request event alone — its status is frozen at
  // "pending" — so reconcile against the run's later events first. Without this,
  // reopening an old thread resurfaces a decision you already made.
  const resolved = new Set<string>();
  let runEnded = false;
  for (const event of events) {
    if (event.type === "approval.applied" || event.type === "approval.decided") {
      const payload = event.payload as Record<string, unknown>;
      const nested = payload.approval && typeof payload.approval === "object"
        ? payload.approval as Record<string, unknown>
        : payload;
      const id = nested.id ?? nested.approval_id;
      if (typeof id === "string" && id) resolved.add(id);
    } else if (event.type === "run.completed" || event.type === "run.failed" || event.type === "run.cancelled") {
      runEnded = true;
    }
  }
  if (runEnded) return null;
  const ordered = [...events].sort((a, b) => b.sequence - a.sequence);
  for (const event of ordered) {
    const approval = approvalFrom(event);
    if (approval && approval.status === "pending" && !decided.has(approval.id) && !resolved.has(approval.id)) {
      return approval;
    }
  }
  return null;
}
