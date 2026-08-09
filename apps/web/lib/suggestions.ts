import type { CustomerActionSuggestion, RunEventV1 } from "@/lib/types";

/**
 * The apply suggestion carried by a `customer.action_suggested` event, or null.
 * Mirrors `approvalFrom`: the run streams these events, and the thread renders
 * the newest one still awaiting a decision as an inline card.
 */
export function suggestionFrom(event: RunEventV1): CustomerActionSuggestion | null {
  if (event.type !== "customer.action_suggested") return null;
  const payload = event.payload as Record<string, unknown>;
  const proposalId = typeof payload.proposal_id === "string" ? payload.proposal_id : "";
  if (!proposalId) return null;
  const count = (key: string) => (typeof payload[key] === "number" ? (payload[key] as number) : 0);
  return {
    run_id: event.run_id,
    kind: "apply_extraction",
    proposal_id: proposalId,
    account_id: typeof payload.account_id === "string" ? payload.account_id : "",
    facts: count("facts"),
    actions: count("actions"),
    people: count("people"),
    summary: typeof payload.summary === "string" ? payload.summary : "new items",
  };
}

/**
 * The newest apply suggestion the user has not yet dismissed. Applied ones are
 * left visible (the card shows "✓ Applied"); only a dismissal removes it.
 */
export function latestActionSuggestion(
  events: RunEventV1[],
  dismissed: ReadonlySet<string>,
): CustomerActionSuggestion | null {
  const ordered = [...events].sort((a, b) => b.sequence - a.sequence);
  for (const event of ordered) {
    const suggestion = suggestionFrom(event);
    if (suggestion && !dismissed.has(suggestion.proposal_id)) return suggestion;
  }
  return null;
}
