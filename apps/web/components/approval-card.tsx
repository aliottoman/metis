"use client";

import type { ApprovalRequest } from "@/lib/types";

interface ApprovalCardProps {
  approval: ApprovalRequest;
  decided: boolean;
  decisionBusy?: string | null;
  onDecision: (approvalId: string, decision: "approve" | "reject") => Promise<void>;
  approveLabel?: string;
  /** "inline" renders in the reading column; the default suits the drawer. */
  variant?: "inline" | "drawer";
}

/**
 * One approval, its buttons, and its three resolved states (recorded, blocked,
 * open). Shared so the drawer timeline and the inline dock in the thread stay
 * identical — the only difference is where it sits, carried by `variant`.
 */
export function ApprovalCard({
  approval,
  decided,
  decisionBusy,
  onDecision,
  approveLabel = "Approve once",
  variant = "drawer",
}: ApprovalCardProps) {
  return (
    <section className={`approvalCard ${variant === "inline" ? "isInline" : ""}`.trim()}>
      <div className="approvalTitle"><span>{approval.risk_level ?? "R3"}</span><strong>{approval.title}</strong></div>
      <p>{approval.summary}</p>
      {approval.permissions?.length ? <div className="permissionList">{approval.permissions.map((permission) => <span key={permission}>{permission}</span>)}</div> : null}
      {approval.action_digest ? <code title={approval.action_digest}>Action {approval.action_digest.slice(0, 12)}</code> : null}
      {decided ? (
        <div className="decisionRecorded">✓ Decision recorded</div>
      ) : approval.blocked_reason ? (
        // No Approve button at all, rather than a disabled one: a greyed button
        // reads as "try again", and this never becomes approvable without a
        // follow-up that fixes it.
        <div className="approvalBlocked">
          <p><strong>Cannot be applied.</strong> {approval.blocked_reason}</p>
          <div className="approvalActions">
            <button type="button" className="dangerButton" disabled={decisionBusy === approval.id} onClick={() => void onDecision(approval.id, "reject")}>Reject</button>
          </div>
        </div>
      ) : (
        <div className="approvalActions">
          <button type="button" className="dangerButton" disabled={decisionBusy === approval.id} onClick={() => void onDecision(approval.id, "reject")}>Reject</button>
          <button type="button" className="primaryButton" disabled={decisionBusy === approval.id} onClick={() => void onDecision(approval.id, "approve")}>{approveLabel}</button>
        </div>
      )}
    </section>
  );
}
