// Small readers shared by the customer pages: dates, money, kinds.

import type { CustomerAccount, CustomerAction } from "@/lib/types";

export const WIN_SERVICES = ["Generative AI Services", "Generative AI Agents", "DAC", "Model-Import", "On-demand", "ODA"];
export const FACT_KINDS = ["requirement", "decision", "use_case", "risk", "question", "constraint", "model", "dac_note", "other"];
// The kinds an engineer reads before a design conversation.
export const TECHNICAL_KINDS = ["requirement", "constraint", "model", "dac_note", "risk"];
export const SOURCE_KINDS = ["note", "meeting", "chat", "notion", "attachment"] as const;

export type AccountFilter = "all" | "wins" | "open" | "overdue" | "waiting";
export const ACCOUNT_FILTERS: Array<[AccountFilter, string]> = [["all", "All"], ["wins", "Wins"], ["open", "Open actions"], ["overdue", "Needs a nudge"], ["waiting", "Waiting"]];

export function matchesFilter(account: CustomerAccount, filter: AccountFilter): boolean {
  if (filter === "wins") return account.wins > 0;
  if (filter === "open") return account.open_actions > 0;
  if (filter === "waiting") return account.pending_notes > 0;
  // A nudge: work outstanding that nobody has touched in a month.
  if (filter === "overdue") {
    if (!account.open_actions) return false;
    return Date.now() - new Date(account.last_interaction_at || account.updated_at).getTime() > 30 * 86_400_000;
  }
  return true;
}

export function when(value?: string | null): string {
  return value ? new Intl.DateTimeFormat(undefined, { dateStyle: "medium" }).format(new Date(value)) : "Not dated";
}

// A win date is a calendar day stored as midnight UTC; read it back in UTC so
// the day chosen is the day shown everywhere west of Greenwich.
export function winDay(value?: string | null): string {
  return value ? new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeZone: "UTC" }).format(new Date(value)) : "Not dated";
}

export function dateInputValue(value?: string | null): string {
  if (!value) return "";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? "" : parsed.toISOString().slice(0, 10);
}

export function instantFromDateInput(value: string): string | null {
  return value ? `${value}T00:00:00Z` : null;
}

export function usd(value?: number | null, compact = false): string {
  if (value === null || value === undefined) return "—";
  const big = compact && value >= 1_000_000;
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", notation: big ? "compact" : "standard", maximumFractionDigits: big ? 2 : 0 }).format(value);
}

export function parseUsd(value: string): number | null {
  const number = Number(value.replace(/[^0-9.]/g, ""));
  return value.trim() && Number.isFinite(number) ? number : null;
}

export function isOverdue(action: CustomerAction): boolean {
  return action.status === "open" && Boolean(action.due_at) && new Date(action.due_at as string).getTime() < Date.now();
}

export function actionMeta(action: CustomerAction): string {
  const owner = action.owner || "Unassigned";
  if (!action.due_at) return action.status === "cancelled" ? `${owner} · cancelled` : owner;
  return `${owner} · ${isOverdue(action) ? "overdue since" : "due"} ${when(action.due_at)}`;
}
