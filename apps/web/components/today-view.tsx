"use client";

// The Today page — the front door. It answers "what needs me now" as a single
// screen you don't scroll: a compact head, a rail of piles you can read at a
// glance, and a deck of cards you roll through sideways. Each card is one pile
// (Start here, a kind of work, deferred, or the day's recap); its items scroll
// inside the card only when there are many, so the page itself never grows.

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import {
  batchAttention,
  deferAttention,
  getAttention,
  getMorningBrief,
  undeferAttention,
} from "@/lib/api";
import type { AttentionFeed, AttentionItem, AttentionKind, MorningBrief } from "@/lib/types";

/** Order the kinds appear in — matches the queue's own consequence weighting,
 *  so the deck and the ranking never tell different stories about what matters. */
const GROUP_ORDER: AttentionKind[] = [
  "run_approval",
  "customer_action",
  "customer_note",
  "tool_proposal",
  "memory",
  "asset_trust",
  "stale_source",
];

/** Full titles for a card heading. */
const GROUP_TITLE: Record<AttentionKind, string> = {
  run_approval: "Runs waiting",
  customer_action: "Commitments",
  customer_note: "Notes to analyze",
  tool_proposal: "Tools to review",
  memory: "Memory proposals",
  asset_trust: "Assets to trust",
  stale_source: "Knowledge sources",
};

/** Kinds whose decision is genuinely one click, so a batch control is honest. */
const BATCHABLE: ReadonlySet<AttentionKind> = new Set(["memory", "customer_action"]);

/** A human "3 days ago" from a timestamp. */
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

/** A short due phrase for an item that carries a deadline. */
function dueLabel(item: AttentionItem): string {
  if (!item.due_at) return "";
  const days = Math.round((new Date(item.due_at).getTime() - Date.now()) / 86_400_000);
  if (item.overdue) return `overdue by ${Math.abs(days)}d`;
  if (days === 0) return "due today";
  return `due in ${days}d`;
}

/** Today as "Sat · August 8" for the head. */
function todayLabel(): string {
  return new Date().toLocaleDateString(undefined, {
    weekday: "short",
    month: "long",
    day: "numeric",
  });
}

/** Join truthy class names. */
function cx(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(" ");
}

/** What a single deck card is about. */
interface Slide {
  id: string;
  label: string;
  count: number;
  variant: "focus" | "group" | "deferred" | "recap";
  kind?: AttentionKind;
  items: AttentionItem[];
  lede?: string;
  batchable?: boolean;
  changed?: string[];
}

interface CardHandlers {
  busyKey: string | null;
  selected: ReadonlySet<string>;
  onDefer: (item: AttentionItem) => void;
  onToggle: (key: string) => void;
}

/** One item row. `showKind` names the kind on an eyebrow — on inside the mixed
 *  "Start here" card, off inside a single-kind card whose heading already says it. */
function AttentionCard({
  item,
  showKind,
  handlers,
}: {
  item: AttentionItem;
  showKind: boolean;
  handlers: CardHandlers;
}) {
  const batchable = BATCHABLE.has(item.kind);
  const due = dueLabel(item);
  const eyebrow = showKind ? `${item.kind_label}${due ? ` · ${due}` : ""}` : null;
  const metaText = showKind
    ? item.detail
    : [item.detail, due, relative(item.created_at)].filter(Boolean).join(" · ");
  const selected = handlers.selected.has(item.key);

  return (
    <li
      className={cx("todayRow", batchable && "hasCheck", selected && "isSelected", item.overdue && "isOverdue")}
      data-kind={item.kind}
    >
      {batchable ? (
        <input
          type="checkbox"
          className="todayCheck"
          checked={selected}
          onChange={() => handlers.onToggle(item.key)}
          aria-label={`Select: ${item.title}`}
        />
      ) : null}
      <div className="todayRowBody">
        {eyebrow ? <span className="todayRowKind">{eyebrow}</span> : null}
        <span className="todayRowTitle">{item.title}</span>
        {metaText ? <span className="todayRowMeta">{metaText}</span> : null}
      </div>
      <div className="todayRowActions">
        <Link className="secondaryButton" href={item.href || "/"}>
          Open
        </Link>
        <button
          className="textButton"
          type="button"
          disabled={handlers.busyKey === item.key}
          onClick={() => handlers.onDefer(item)}
          title="Ask me again in a week"
        >
          Later
        </button>
      </div>
    </li>
  );
}

export function TodayView() {
  const [feed, setFeed] = useState<AttentionFeed | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busyKey, setBusyKey] = useState<string | null>(null);
  const [selected, setSelected] = useState<ReadonlySet<string>>(new Set());
  const [batchNote, setBatchNote] = useState<string | null>(null);
  const [brief, setBrief] = useState<MorningBrief | null>(null);

  // Load the queue.
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
    // The brief is a second, slower read: the deck must render immediately even
    // when a model is cold or unreachable.
    void getMorningBrief().then(setBrief).catch(() => setBrief(null));
  }, [refresh]);

  // Snooze one item for a week.
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

  // Add or drop one item from the current selection.
  function toggle(key: string) {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  // Apply one decision to everything selected.
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
          ? `${result.applied.length} applied · ${result.skipped.length} skipped`
          : `${result.applied.length} applied`,
      );
    } catch (batchError) {
      setError(batchError instanceof Error ? batchError.message : "Could not apply that.");
    } finally {
      setBusyKey(null);
    }
  }

  // Pull one deferred item back into the queue.
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

  // Select or clear every item on one card.
  function toggleGroup(items: AttentionItem[]) {
    const keys = items.map((item) => item.key);
    const all = keys.every((key) => selected.has(key));
    setSelected((current) => {
      const next = new Set(current);
      keys.forEach((key) => (all ? next.delete(key) : next.add(key)));
      return next;
    });
  }

  const total = feed?.total ?? 0;

  // Top items lead in "Start here" and are pulled out of their kind cards, so
  // the deck never shows the same thing twice.
  const topItems = feed?.top ?? [];
  const topKeys = new Set(topItems.map((item) => item.key));

  // Build the deck: Start here, then each kind that has anything, then deferred,
  // then the day's recap.
  const slides: Slide[] = [];
  if (topItems.length) {
    slides.push({
      id: "focus",
      label: "Start here",
      count: topItems.length,
      variant: "focus",
      items: topItems,
      lede: brief?.recommendation,
    });
  }
  for (const kind of GROUP_ORDER) {
    const items = (feed?.items ?? []).filter((item) => item.kind === kind && !topKeys.has(item.key));
    if (items.length) {
      slides.push({
        id: kind,
        label: GROUP_TITLE[kind],
        count: items.length,
        variant: "group",
        kind,
        items,
        batchable: BATCHABLE.has(kind),
      });
    }
  }
  if (feed?.deferred) {
    slides.push({
      id: "deferred",
      label: "Deferred",
      count: feed.deferred,
      variant: "deferred",
      items: feed.deferred_items,
    });
  }
  if (brief && brief.changed.length) {
    slides.push({
      id: "recap",
      label: "Since yesterday",
      count: brief.changed.length,
      variant: "recap",
      items: [],
      changed: brief.changed,
    });
  }

  // A pile-per-kind count for the rail, so the whole day reads at a glance even
  // before you roll.
  const railCounts = new Map<string, number>();
  slides.forEach((slide) => railCounts.set(slide.id, slide.count));

  // Jump to a pile from the rail. Every pile is on screen at once now, so this
  // just brings the named card into view instead of rolling a one-at-a-time deck.
  const scrollToPile = useCallback((id: string) => {
    document
      .getElementById(`pile-${id}`)
      ?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, []);

  const handlers: CardHandlers = {
    busyKey,
    selected,
    onDefer: (item) => void defer(item, 7),
    onToggle: toggle,
  };

  const headline =
    total === 0
      ? "You're all clear"
      : total === 1
        ? "1 thing needs you"
        : `${total} things need you`;

  return (
    <div className="workspacePage todayPage todayGridPage">
      <header className="todayTopbar">
        <div className="todayTopbarCopy">
          <span className="eyebrow">{todayLabel()}</span>
          <h1>{loading && !feed ? "Checking what's waiting" : headline}</h1>
        </div>
        <button className="secondaryButton todayRefresh" type="button" onClick={() => void refresh()} disabled={loading}>
          {loading ? "Checking…" : "Refresh"}
        </button>
      </header>

      {error ? (
        <div className="composerError todayError" role="alert">
          <span>!</span>
          <p>{error}</p>
        </div>
      ) : null}

      {total === 0 && !loading && !slides.length ? (
        <section className="todayEmpty">
          <div className="todayEmptyMark" aria-hidden="true" />
          <h2>Nothing is waiting for a decision.</h2>
          <p>New approvals, commitments, notes, and proposals collect here as they appear. Enjoy the quiet.</p>
        </section>
      ) : null}

      {slides.length ? (
        <>
          {/* The rail: the whole day at a glance; each chip jumps to its pile. */}
          <nav className="todayRail" aria-label="Piles">
            {slides.map((slide) => (
              <button
                key={slide.id}
                type="button"
                className="todayRailChip"
                data-kind={slide.kind}
                data-variant={slide.variant}
                onClick={() => scrollToPile(slide.id)}
              >
                <span className="todayRailCount">{railCounts.get(slide.id)}</span>
                <span className="todayRailLabel">{slide.label}</span>
              </button>
            ))}
          </nav>

          {/* Every pile at once — a dense grid, Start here featured across the top. */}
          <div className="todayGridWrap">
            <div className="todayGrid">
              {slides.map((slide) => (
                <section
                  key={slide.id}
                  id={`pile-${slide.id}`}
                  className={cx("todaySlide", `is-${slide.variant}`)}
                  data-kind={slide.kind}
                >
                  <div className="todaySlideHead">
                    <div className="todaySlideTitle">
                      {slide.kind ? <span className="todaySlideDot" aria-hidden="true" /> : null}
                      <h2>{slide.label}</h2>
                      <span className="todaySlideCount">{slide.count}</span>
                    </div>
                    {slide.batchable ? (
                      <button className="textButton" type="button" onClick={() => toggleGroup(slide.items)}>
                        {slide.items.every((item) => selected.has(item.key)) ? "Clear" : "Select all"}
                      </button>
                    ) : null}
                  </div>
                  {slide.lede ? <p className="todaySlideLede">{slide.lede}</p> : null}

                  {slide.variant === "recap" ? (
                    <ul className="todayRecapList">
                      {(slide.changed ?? []).map((line) => (
                        <li key={line}>{line}</li>
                      ))}
                    </ul>
                  ) : slide.variant === "deferred" ? (
                    <ul className="todayRows">
                      {slide.items.map((item) => (
                        <li key={item.key} className="todayRow isDeferred" data-kind={item.kind}>
                          <div className="todayRowBody">
                            <span className="todayRowTitle">{item.title}</span>
                            <span className="todayRowMeta">
                              returns {new Date(item.deferred_until as string).toLocaleDateString()}
                            </span>
                          </div>
                          <div className="todayRowActions">
                            <button
                              className="textButton"
                              type="button"
                              disabled={busyKey === item.key}
                              onClick={() => void restore(item.key)}
                            >
                              Bring back
                            </button>
                          </div>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <ul className="todayRows">
                      {slide.items.map((item) => (
                        <AttentionCard
                          key={item.key}
                          item={item}
                          showKind={slide.variant === "focus"}
                          handlers={handlers}
                        />
                      ))}
                    </ul>
                  )}
                </section>
              ))}
            </div>
          </div>
        </>
      ) : null}

      {selected.size ? (
        <div className="todayBatchBar" role="region" aria-label="Selected items">
          <strong>{selected.size} selected</strong>
          <div>
            <button className="primaryButton" type="button" disabled={busyKey === "__batch__"} onClick={() => void runBatch("approve")}>
              Approve / complete
            </button>
            <button className="secondaryButton" type="button" disabled={busyKey === "__batch__"} onClick={() => void runBatch("reject")}>
              Reject
            </button>
            <button className="secondaryButton" type="button" disabled={busyKey === "__batch__"} onClick={() => void runBatch("defer")}>
              Later
            </button>
            <button className="textButton" type="button" onClick={() => setSelected(new Set())}>
              Clear
            </button>
          </div>
        </div>
      ) : null}
      {batchNote ? (
        <p className="mutedMeta todayBatchNote" role="status">
          {batchNote}
        </p>
      ) : null}
    </div>
  );
}
