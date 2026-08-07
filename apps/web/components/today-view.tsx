"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { batchAttention, deferAttention, getAttention, undeferAttention } from "@/lib/api";
import type { AttentionFeed, AttentionItem } from "@/lib/types";

/** Order the groups appear in. Matches the queue's own weighting, so the page
 *  and the ranking can never tell different stories about what matters. */
const GROUP_ORDER: AttentionItem["kind"][] = [
  "run_approval",
  "customer_action",
  "customer_note",
  "tool_proposal",
  "memory",
  "asset_trust",
  "stale_source",
];

const GROUP_TITLE: Record<AttentionItem["kind"], string> = {
  run_approval: "Runs waiting on a decision",
  customer_action: "Commitments to customers",
  customer_note: "Notes captured but not analyzed",
  tool_proposal: "Tools awaiting review",
  memory: "Memory proposals",
  asset_trust: "Assets awaiting trust",
  stale_source: "Knowledge sources",
};

/** Kinds whose decision is genuinely one click. A batch control over
 *  anything that needs reading first would turn review into a rubber stamp,
 *  so the rest stay openable-only. */
const BATCHABLE: ReadonlySet<AttentionItem["kind"]> = new Set([
  "memory",
  "customer_action",
]);

function relative(value: string | null): string {
  if (!value) return "";
  const then = new Date(value).getTime();
  if (Number.isNaN(then)) return "";
  const days = Math.round((Date.now() - then) / 86_400_000);
  if (days <= 0) return "today";
  if (days === 1) return "yesterday";
  if (days < 30) return `${days} days ago`;
  return `${Math.round(days / 30)} months ago`;
}

function dueLabel(item: AttentionItem): string {
  if (!item.due_at) return "";
  const days = Math.round((new Date(item.due_at).getTime() - Date.now()) / 86_400_000);
  if (item.overdue) return `overdue by ${Math.abs(days)}d`;
  if (days === 0) return "due today";
  return `due in ${days}d`;
}

export function TodayView() {
  const [feed, setFeed] = useState<AttentionFeed | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busyKey, setBusyKey] = useState<string | null>(null);
  const [showDeferred, setShowDeferred] = useState(false);
  const [selected, setSelected] = useState<ReadonlySet<string>>(new Set());
  const [batchNote, setBatchNote] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setFeed(await getAttention(3));
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Could not read the queue.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  async function defer(item: AttentionItem, days: number) {
    setBusyKey(item.key);
    try {
      setFeed(await deferAttention(item.key, item.kind, days));
    } catch (deferError) {
      setError(deferError instanceof Error ? deferError.message : "Could not defer that.");
    } finally {
      setBusyKey(null);
    }
  }

  function toggle(key: string) {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  async function runBatch(decision: "approve" | "reject" | "defer") {
    const keys = [...selected];
    if (!keys.length) return;
    setBusyKey("__batch__");
    setBatchNote(null);
    try {
      const result = await batchAttention(keys, decision);
      setFeed(result.feed);
      setSelected(new Set());
      setBatchNote(
        result.skipped.length
          ? `${result.applied.length} applied · ${result.skipped.length} skipped (no one-click decision for those)`
          : `${result.applied.length} applied`,
      );
    } catch (batchError) {
      setError(batchError instanceof Error ? batchError.message : "Could not apply that.");
    } finally {
      setBusyKey(null);
    }
  }

  async function restore(key: string) {
    setBusyKey(key);
    try {
      setFeed(await undeferAttention(key));
    } catch (restoreError) {
      setError(restoreError instanceof Error ? restoreError.message : "Could not restore that.");
    } finally {
      setBusyKey(null);
    }
  }

  const total = feed?.total ?? 0;
  const headline =
    total === 0
      ? "Nothing needs you"
      : total === 1
        ? "1 thing needs you"
        : `${total} things need you`;

  const grouped = GROUP_ORDER.map((kind) => ({
    kind,
    items: (feed?.items ?? []).filter((item) => item.kind === kind),
  })).filter((group) => group.items.length > 0);

  return (
    <div className="workspacePage todayPage">
      <header className="pageHeader">
        <div>
          <span className="eyebrow">Today</span>
          <h1>{loading && !feed ? "Checking what's waiting" : headline}</h1>
          <p>
            Everything waiting on you, from every workbench, ranked by what it costs
            to leave it until tomorrow.
          </p>
        </div>
        <button className="secondaryButton" type="button" onClick={() => void refresh()} disabled={loading}>
          {loading ? "Checking…" : "Refresh"}
        </button>
      </header>

      {error ? <div className="composerError" role="alert"><span>!</span><p>{error}</p></div> : null}

      {/* The batch bar exists only while something is selected, so the page
          reads as a queue at rest and as a worksheet the moment you start
          clearing it. Approve means "yes" for a memory and "done" for a
          commitment — the verb each kind actually needs. */}
      {selected.size ? (
        <div className="todayBatchBar" role="region" aria-label="Selected items">
          <strong>{selected.size} selected</strong>
          <div>
            <button
              className="primaryButton"
              type="button"
              disabled={busyKey === "__batch__"}
              onClick={() => void runBatch("approve")}
            >
              Approve / complete
            </button>
            <button
              className="secondaryButton"
              type="button"
              disabled={busyKey === "__batch__"}
              onClick={() => void runBatch("reject")}
            >
              Reject
            </button>
            <button
              className="secondaryButton"
              type="button"
              disabled={busyKey === "__batch__"}
              onClick={() => void runBatch("defer")}
            >
              Later
            </button>
            <button className="textButton" type="button" onClick={() => setSelected(new Set())}>
              Clear
            </button>
          </div>
        </div>
      ) : null}
      {batchNote ? <p className="mutedMeta" role="status">{batchNote}</p> : null}

      {/* The headline three. If the page says three things need you, these are
          them — ranked by consequence, so a promise due today outranks a
          backlog that has waited months without anything breaking. */}
      {feed?.top.length ? (
        <section className="todayTop" aria-label="What needs you first">
          {feed.top.map((item, index) => (
            <article key={item.key} className={`todayTopCard ${item.overdue ? "isOverdue" : ""}`}>
              <span className="todayRank">{index + 1}</span>
              <div>
                <span className="todayKind">{item.kind_label}{dueLabel(item) ? ` · ${dueLabel(item)}` : ""}</span>
                <strong>{item.title}</strong>
                {item.detail ? <small>{item.detail}</small> : null}
              </div>
              <div className="todayTopActions">
                <Link className="primaryButton" href={item.href || "/"}>Open</Link>
                <button
                  className="textButton"
                  type="button"
                  disabled={busyKey === item.key}
                  onClick={() => void defer(item, 7)}
                >
                  Later
                </button>
              </div>
            </article>
          ))}
        </section>
      ) : null}

      {total === 0 && !loading ? (
        <section className="settingsSection">
          <p className="sectionLede">
            Nothing is waiting for a decision. New approvals, customer actions, notes,
            and memory proposals will collect here as they appear.
          </p>
        </section>
      ) : null}

      {grouped.map((group) => (
        <section className="settingsSection" key={group.kind}>
          <div className="sectionTitle">
            <div><h2>{GROUP_TITLE[group.kind]}</h2></div>
            {BATCHABLE.has(group.kind) ? (
              <button
                className="textButton"
                type="button"
                onClick={() => {
                  const keys = group.items.map((item) => item.key);
                  const all = keys.every((key) => selected.has(key));
                  setSelected((current) => {
                    const next = new Set(current);
                    keys.forEach((key) => (all ? next.delete(key) : next.add(key)));
                    return next;
                  });
                }}
              >
                {group.items.every((item) => selected.has(item.key)) ? "Clear" : "Select all"}
              </button>
            ) : null}
            <span className="sectionBadge">{group.items.length}</span>
          </div>
          <ul className="todayList">
            {group.items.map((item) => (
              <li key={item.key} className={item.overdue ? "isOverdue" : ""}>
                {BATCHABLE.has(item.kind) ? (
                  <input
                    type="checkbox"
                    className="todayCheck"
                    checked={selected.has(item.key)}
                    onChange={() => toggle(item.key)}
                    aria-label={`Select: ${item.title}`}
                  />
                ) : (
                  <span className="todayCheck todayCheckSpacer" aria-hidden="true" />
                )}
                <div className="todayItemBody">
                  <strong>{item.title}</strong>
                  <small>
                    {[item.detail, dueLabel(item), relative(item.created_at)]
                      .filter(Boolean)
                      .join(" · ")}
                  </small>
                </div>
                <div className="todayItemActions">
                  <Link className="secondaryButton" href={item.href || "/"}>Open</Link>
                  <button
                    className="textButton"
                    type="button"
                    disabled={busyKey === item.key}
                    onClick={() => void defer(item, 7)}
                    title="Ask me again in a week"
                  >
                    Later
                  </button>
                </div>
              </li>
            ))}
          </ul>
        </section>
      ))}

      {feed?.deferred ? (
        <section className="settingsSection compactSection">
          <div className="sectionTitle">
            <div>
              <h2>Deferred</h2>
              <p>Snoozed, not dismissed — each returns on its own date.</p>
            </div>
            <button className="textButton" type="button" onClick={() => setShowDeferred((value) => !value)}>
              {showDeferred ? "Hide" : `Show ${feed.deferred}`}
            </button>
          </div>
          {showDeferred ? (
            <ul className="todayList">
              {feed.deferred_items.map((item) => (
                  <li key={item.key}>
                    <div className="todayItemBody">
                      <strong>{item.title}</strong>
                      <small>returns {new Date(item.deferred_until as string).toLocaleDateString()}</small>
                    </div>
                    <button
                      className="textButton"
                      type="button"
                      disabled={busyKey === item.key}
                      onClick={() => void restore(item.key)}
                    >
                      Bring back
                    </button>
                  </li>
              ))}
            </ul>
          ) : null}
        </section>
      ) : null}
    </div>
  );
}
