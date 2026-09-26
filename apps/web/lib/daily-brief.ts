import type { AttentionFeed, AttentionItem } from "./types.ts";

/** A briefing remains useful against an older API while the service restarts. */
export function dailySignals(feed: AttentionFeed) {
  const now = Date.parse(feed.generated_at);
  const neglected = feed.neglected ?? feed.items.filter((item) => {
    if (item.kind !== "customer_action") return false;
    const updated = item.updated_at ? Date.parse(item.updated_at) : NaN;
    return item.overdue || (!item.due_at && Number.isFinite(updated) && now - updated >= 7 * 86_400_000);
  });
  const opportunities = feed.opportunities ?? feed.items.filter((item) => item.kind === "customer_opportunity");
  return { neglected, opportunities, focus: feed.top.slice(0, 3) };
}

export function nextStepFor(item: AttentionItem): string {
  if (item.next_step) return item.next_step;
  switch (item.kind) {
    case "customer_action": return "Review the commitment and its source, finish the work, then mark it complete.";
    case "customer_opportunity": return "Qualify the recorded need with the account and agree a concrete follow-up.";
    case "run_approval": return "Open the paused run, inspect the proposed change, then approve or reject it.";
    case "customer_note": return "Read the captured note, analyze it when ready, and review the proposed updates.";
    case "memory": case "answer_atom": return "Check the proposed knowledge against its source before keeping it.";
    case "asset_trust": return "Inspect the launch recipe and approve it only when the commands match your intent.";
    case "tool_proposal": return "Inspect the tool's purpose, permissions, and proposed behavior before deciding.";
    case "stale_source": return "Open the source, check its indexing status and permissions, then retry if needed.";
  }
}

export function preparedStepsFor(item: AttentionItem): string {
  if (item.prepared_prompt) return item.prepared_prompt;
  return [
    item.title,
    item.detail,
    "",
    `Next step: ${nextStepFor(item)}`,
    `Source: ${item.source_href || item.href}`,
  ].filter((line) => line !== undefined).join("\n");
}

export function reasonFor(item: AttentionItem): string {
  if (item.why_now) return item.why_now;
  if (item.overdue) return "The recorded due date has passed.";
  if (item.kind === "run_approval") return "Work is paused until you make this decision.";
  if (item.kind === "customer_action") return "An open commitment is waiting for a next step.";
  if (item.kind === "customer_opportunity") return "A recently recorded need may be worth following up.";
  return "This review will make existing work useful again.";
}
