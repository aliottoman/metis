import assert from "node:assert/strict";
import test from "node:test";

import { dailySignals, nextStepFor, preparedStepsFor, reasonFor } from "../lib/daily-brief.ts";
import type { AttentionFeed, AttentionItem } from "../lib/types.ts";

const item = (overrides: Partial<AttentionItem> = {}): AttentionItem => ({
  key: "customer_action:a1", kind: "customer_action", kind_label: "Commitment",
  title: "Share the sizing review", detail: "Acme", href: "/customers?account=c1&tab=actions&action=a1",
  account_id: "c1", due_at: null, created_at: "2026-09-01T09:00:00Z", overdue: false, priority: 80, deferred_until: null,
  ...overrides,
});
const feed = (items: AttentionItem[], extra: Partial<AttentionFeed> = {}): AttentionFeed => ({
  generated_at: "2026-09-10T09:00:00Z", items, top: items, deferred_items: [], total: items.length, deferred: 0, counts: {}, ...extra,
});

test("a briefing never invents a third action when only two records exist", () => {
  const items = [item(), item({ key: "customer_action:a2" })];
  assert.equal(dailySignals(feed(items)).focus.length, 2);
});

test("fallback neglect uses the server clock and excludes future commitments and housekeeping", () => {
  const stale = item({ updated_at: "2026-09-01T09:00:00Z" });
  const future = item({ key: "future", due_at: "2026-10-01T09:00:00Z" });
  const memory = item({ key: "memory:m", kind: "memory" });
  const unknownUpdate = item({ key: "unknown", created_at: "2020-01-01T00:00:00Z" });
  assert.deepEqual(dailySignals(feed([stale, future, memory, unknownUpdate])).neglected, [stale]);
});

test("server signals override the compatibility fallback, including an empty result", () => {
  assert.deepEqual(dailySignals(feed([item()], { neglected: [], opportunities: [] })).neglected, []);
});

test("a recent recorded update keeps an old undated commitment out of neglected work", () => {
  assert.deepEqual(dailySignals(feed([item({ updated_at: "2026-09-09T09:00:00Z" })])).neglected, []);
});

test("prepared steps preserve the exact supplied source and never submit work", () => {
  const source = "/customers?account=c1&tab=sources&source=s1";
  const action = item({ source_href: source });
  assert.match(preparedStepsFor(action), /Share the sizing review/);
  assert.ok(preparedStepsFor(action).endsWith(`Source: ${source}`));
  assert.match(nextStepFor(action), /mark it complete/);
  assert.equal(reasonFor(item({ overdue: true })), "The recorded due date has passed.");
});

test("server-prepared steps and evidence-based reasons are shown verbatim", () => {
  const action = item({ prepared_prompt: "Review the customer's exact note.", next_step: "Confirm the revised due date.", why_now: "No recorded update in 12 days." });
  assert.equal(preparedStepsFor(action), "Review the customer's exact note.");
  assert.equal(nextStepFor(action), "Confirm the revised due date.");
  assert.equal(reasonFor(action), "No recorded update in 12 days.");
});
