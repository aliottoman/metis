"use client";

import type { CustomerActionSuggestion } from "@/lib/types";

interface ApplyCardProps {
  suggestion: CustomerActionSuggestion;
  state: "idle" | "applying" | "applied";
  onApply: (proposalId: string) => Promise<void>;
  onDismiss: (proposalId: string) => void;
}

/**
 * The one-click apply card for a filed note's analysis. It reuses the approval
 * card's shell so it sits in the thread like any other gated action. The point
 * is the gate: the analysis has run and proposed facts, but nothing reaches the
 * account profile until Apply is tapped here — the same review the workbench
 * asks for, brought to where you filed the note.
 */
export function ApplyCard({ suggestion, state, onApply, onDismiss }: ApplyCardProps) {
  return (
    <section className="approvalCard isInline">
      <div className="approvalTitle">
        <span>Apply</span>
        <strong>Analysis found {suggestion.summary}</strong>
      </div>
      <p>Apply them to the account profile? Nothing is written until you do.</p>
      {state === "applied" ? (
        <div className="decisionRecorded">✓ Applied to the profile</div>
      ) : (
        <div className="approvalActions">
          <button
            type="button"
            className="secondaryButton"
            disabled={state === "applying"}
            onClick={() => onDismiss(suggestion.proposal_id)}
          >
            Dismiss
          </button>
          <button
            type="button"
            className="primaryButton"
            disabled={state === "applying"}
            onClick={() => void onApply(suggestion.proposal_id)}
          >
            {state === "applying" ? "Applying…" : "Apply to profile"}
          </button>
        </div>
      )}
    </section>
  );
}
