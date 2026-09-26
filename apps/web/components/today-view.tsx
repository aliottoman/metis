"use client";

import { Sun } from "lucide-react";

// The front door: one ranked list of what needs you, walkable from the
// keyboard. Start here leads, then each kind of work, then what you put off,
// then what changed since yesterday. The brief reads it aloud on request.

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState, type KeyboardEvent } from "react";

import { AudioPlayer } from "@/components/ui/audio-player";
import { Notice } from "@/components/ui/notice";
import { PageHeader } from "@/components/ui/page-header";
import { Skeleton } from "@/components/ui/skeleton";
import { DailyBriefing } from "@/components/today/briefing";
import { batchAttention, deferAttention, getAttention, getMorningBrief, getMorningBriefAudio, undeferAttention } from "@/lib/api";
import type { AttentionFeed, AttentionItem, AttentionKind, MorningBrief } from "@/lib/types";

// The order kinds appear in: the queue's own consequence weighting.
const KIND_ORDER: AttentionKind[] = ["run_approval", "customer_action", "customer_opportunity", "customer_note", "tool_proposal", "answer_atom", "memory", "asset_trust", "stale_source"];
const KIND_TITLE: Record<AttentionKind, string> = {
  run_approval: "Runs waiting",
  customer_action: "Commitments",
  customer_opportunity: "Opportunity signals",
  customer_note: "Notes to review",
  tool_proposal: "Tools to review",
  answer_atom: "Answers to keep",
  memory: "Memory proposals",
  asset_trust: "Assets to trust",
  stale_source: "Knowledge sources",
};
// Kinds whose decision is genuinely one click, so a batch control is honest.
const BATCHABLE = new Set<AttentionKind>(["memory", "customer_action"]);
const DEFER_DAYS = 7;

interface Section {
  id: string;
  label: string;
  items: AttentionItem[];
  batchable: boolean;
  deferred?: boolean;
  changed?: string[];
}

function relative(value: string | null): string {
  if (!value) return "";
  const days = Math.round((Date.now() - new Date(value).getTime()) / 86_400_000);
  if (Number.isNaN(days)) return "";
  if (days <= 0) return "today";
  if (days === 1) return "yesterday";
  return days < 30 ? `${days} days ago` : `${Math.round(days / 30)} months ago`;
}

function due(item: AttentionItem): string {
  if (!item.due_at) return "";
  const days = Math.round((new Date(item.due_at).getTime() - Date.now()) / 86_400_000);
  if (item.overdue) return `overdue by ${Math.abs(days)}d`;
  return days === 0 ? "due today" : `due in ${days}d`;
}

function sections(feed: AttentionFeed | null, brief: MorningBrief | null): Section[] {
  if (!feed) return [];
  const top = new Set(feed.top.map((item) => item.key));
  const out: Section[] = [];
  for (const kind of KIND_ORDER) {
    const items = feed.items.filter((item) => item.kind === kind && !top.has(item.key));
    if (items.length) out.push({ id: kind, label: KIND_TITLE[kind], items, batchable: BATCHABLE.has(kind) });
  }
  if (feed.deferred) out.push({ id: "deferred", label: "Deferred", items: feed.deferred_items, batchable: false, deferred: true });
  if (brief?.changed.length) out.push({ id: "recap", label: "Since yesterday", items: [], batchable: false, changed: brief.changed });
  return out;
}

export function TodayView() {
  const router = useRouter();
  const [feed, setFeed] = useState<AttentionFeed | null>(null);
  const [brief, setBrief] = useState<MorningBrief | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busyKey, setBusyKey] = useState<string | null>(null);
  const [selected, setSelected] = useState<ReadonlySet<string>>(new Set());
  const [batchNote, setBatchNote] = useState<string | null>(null);
  const [clip, setClip] = useState<{ url: string } | "loading" | null>(null);
  const [cursor, setCursor] = useState(0);
  const list = useRef<HTMLDivElement>(null);
  const clipAbort = useRef<AbortController | null>(null);
  const clipUrl = useRef<string | null>(null);
  const updateBusy = useRef(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const current = await getAttention(3);
      setFeed(current);
      // Older services generate prose on this read. Only ask a service that
      // advertises the record-derived briefing contract; refresh is read-only.
      if (current.neglected !== undefined) {
        setBrief(await getMorningBrief(24, false).catch(() => null));
      }
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Could not read the queue.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void refresh(); }, [refresh]);

  // The rendered clip lives for the page; leaving it drops the request and the URL.
  useEffect(() => () => {
    clipAbort.current?.abort();
    if (clipUrl.current) URL.revokeObjectURL(clipUrl.current);
  }, []);

  // A saved rendition describes one queue snapshot. After a decision, the
  // next Listen should describe the updated work instead of replaying it.
  useEffect(() => {
    clipAbort.current?.abort();
    clipAbort.current = null;
    if (clipUrl.current) URL.revokeObjectURL(clipUrl.current);
    clipUrl.current = null;
    setClip(null);
  }, [feed]);

  const listen = async () => {
    if (clipAbort.current && !clipAbort.current.signal.aborted) return;
    const controller = new AbortController();
    clipAbort.current = controller;
    setClip("loading");
    setError(null);
    try {
      const blob = await getMorningBriefAudio(24, controller.signal);
      if (controller.signal.aborted) return;
      if (clipUrl.current) URL.revokeObjectURL(clipUrl.current);
      clipUrl.current = URL.createObjectURL(blob);
      setClip({ url: clipUrl.current });
    } catch (playError) {
      if (playError instanceof DOMException && playError.name === "AbortError") return;
      setClip(null);
      setError(playError instanceof Error ? playError.message : "The brief could not be read aloud.");
    } finally {
      if (clipAbort.current === controller) clipAbort.current = null;
    }
  };

  // One shape for every queue call: mark the row busy, swap in the new feed.
  const update = async (key: string, call: () => Promise<AttentionFeed>, fallback: string) => {
    if (updateBusy.current) return;
    updateBusy.current = true;
    setBusyKey(key);
    setError(null);
    try {
      setFeed(await call());
    } catch (updateError) {
      setError(updateError instanceof Error ? updateError.message : fallback);
    } finally {
      updateBusy.current = false;
      setBusyKey(null);
    }
  };
  const defer = (item: AttentionItem) => update(item.key, () => deferAttention(item.key, item.kind, DEFER_DAYS), "Could not defer that.");
  const restore = (item: AttentionItem) => update(item.key, () => undeferAttention(item.key), "Could not restore that.");
  const complete = (item: AttentionItem) => update(item.key, async () => {
    const result = await batchAttention([item.key], "approve");
    setBatchNote(result.applied.includes(item.key) ? `Completed: ${item.title}` : "This commitment has changed. The queue has been refreshed.");
    return result.feed;
  }, "Could not complete that commitment.");
  const toggle = (key: string) => setSelected((current) => {
    const next = new Set(current);
    if (!next.delete(key)) next.add(key);
    return next;
  });
  const runBatch = async (decision: "approve" | "reject" | "defer") => {
    const keys = [...selected];
    if (!keys.length || updateBusy.current) return;
    updateBusy.current = true;
    setBusyKey("batch");
    setBatchNote(null);
    setError(null);
    try {
      const result = await batchAttention(keys, decision);
      setFeed(result.feed);
      setSelected(new Set());
      setBatchNote(`${result.applied.length} applied${result.skipped.length ? ` · ${result.skipped.length} skipped` : ""}`);
    } catch (batchError) {
      setError(batchError instanceof Error ? batchError.message : "Could not apply that.");
    } finally {
      updateBusy.current = false;
      setBusyKey(null);
    }
  };

  const groups = sections(feed, brief);
  const rows = groups.flatMap((section) => section.items.map((item) => ({ item, section })));
  const total = feed?.total ?? 0;
  const headline = !feed ? loading ? "Checking what's waiting" : "Today is unavailable" : total === 0 ? "You're all clear" : `${total} ${total === 1 ? "thing needs" : "things need"} you`;

  // Refreshes and decisions can remove or reorder rows. Keep selection and the
  // keyboard cursor attached to the queue that is actually on screen.
  useEffect(() => {
    const currentGroups = sections(feed, null);
    const selectable = new Set(currentGroups.filter((section) => section.batchable).flatMap((section) => section.items.map((item) => item.key)));
    setSelected((current) => new Set([...current].filter((key) => selectable.has(key))));
    const count = currentGroups.reduce((sum, section) => sum + section.items.length, 0);
    setCursor((current) => Math.max(0, Math.min(current, count - 1)));
  }, [feed]);

  // Arrow keys walk the list; Enter opens; L defers; X selects; R brings back.
  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (!rows.length || event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey || event.repeat) return;
    // Let links, buttons and checkboxes keep their native keyboard behavior.
    if (event.target instanceof HTMLElement && event.target.closest("a, button, input, textarea, select, [contenteditable='true']")) return;
    const row = rows[cursor];
    const move = (delta: number) => {
      const next = Math.max(0, Math.min(rows.length - 1, cursor + delta));
      setCursor(next);
      list.current?.querySelector<HTMLElement>(`[data-row="${next}"]`)?.scrollIntoView({ block: "nearest" });
    };
    switch (event.key) {
      case "ArrowDown": case "j": event.preventDefault(); move(1); break;
      case "ArrowUp": case "k": event.preventDefault(); move(-1); break;
      case "Enter": if (row && !row.section.deferred) { event.preventDefault(); router.push(row.item.href || "/"); } break;
      case "l": if (row && !row.section.deferred && !busyKey) { event.preventDefault(); void defer(row.item); } break;
      case "r": if (row?.section.deferred && !busyKey) { event.preventDefault(); void restore(row.item); } break;
      case "x": case " ": if (row?.section.batchable && !busyKey) { event.preventDefault(); toggle(row.item.key); } break;
      default: return;
    }
  };

  let index = -1;
  return (
    <div className="today">
      <PageHeader
        icon={<Sun />}
        eyebrow={new Date().toLocaleDateString(undefined, { weekday: "long", month: "long", day: "numeric" })}
        title="Your daily briefing"
        lede={headline}
        actions={
          <>
            {clip && clip !== "loading" ? (
              <AudioPlayer src={clip.url} label="Daily brief" />
            ) : (
              <button type="button" className="ui-btn" onClick={() => void listen()} disabled={clip === "loading" || !feed}>{clip === "loading" ? "Preparing…" : "Listen to the brief"}</button>
            )}
            <button type="button" className="ui-btn is-quiet" onClick={() => void refresh()} disabled={loading || busyKey !== null}>{loading ? "Refreshing…" : "Refresh"}</button>
          </>
        }
      />

      {error ? <Notice kind="error" action={!feed ? "Try again" : undefined} onAction={() => void refresh()} onDismiss={feed ? () => setError(null) : undefined}>{error}</Notice> : null}

      {feed ? <DailyBriefing feed={feed} busyKey={busyKey} onDefer={(item) => void defer(item)} onComplete={(item) => void complete(item)} onError={setError} /> : null}
      {feed && groups.length ? <div className="daily-section-heading"><div><span className="ui-eyebrow">Keep everything moving</span><h2>Rest of your queue</h2></div></div> : null}

      {groups.length > 1 ? (
        <nav className="today-nav" aria-label="Sections">
          {groups.map((section) => (
            <a key={section.id} href={`#today-${section.id}`} className="ui-chip" onClick={(event) => { event.preventDefault(); document.getElementById(`today-${section.id}`)?.scrollIntoView({ behavior: "smooth", block: "start" }); }}>
              {section.label} <small>{section.changed ? section.changed.length : section.items.length}</small>
            </a>
          ))}
        </nav>
      ) : null}

      {loading && !feed ? <Skeleton rows={6} height={44} /> : null}

      {!loading && feed && feed.total === 0 && !groups.length ? (
        <section className="today-empty">
          <h2>Nothing is waiting for a decision.</h2>
          <p>Approvals, commitments, notes and proposals collect here as they appear.</p>
        </section>
      ) : null}

      <div className="today-list" ref={list} role="region" tabIndex={rows.length ? 0 : -1} onKeyDown={onKeyDown} aria-label="What needs you" aria-keyshortcuts="ArrowUp ArrowDown Enter l x r" aria-busy={busyKey !== null}>
        {groups.map((section) => (
          <section key={section.id} id={`today-${section.id}`} className="today-section">
            <header>
              <h2>{section.label}</h2>
              <span>{section.changed ? section.changed.length : section.items.length}</span>
              {section.batchable ? (
                <button type="button" className="ui-btn is-quiet is-sm" disabled={busyKey !== null} onClick={() => {
                  const keys = section.items.map((item) => item.key);
                  const all = keys.every((key) => selected.has(key));
                  setSelected((current) => { const next = new Set(current); keys.forEach((key) => (all ? next.delete(key) : next.add(key))); return next; });
                }}>
                  {section.items.every((item) => selected.has(item.key)) ? "Clear" : "Select all"}
                </button>
              ) : null}
            </header>
            {section.changed ? (
              <ul className="today-recap">{section.changed.map((line) => <li key={line}>{line}</li>)}</ul>
            ) : (
              <ul className="today-rows">
                {section.items.map((item) => {
                  index += 1;
                  const at = index;
                  const meta = section.id === "focus"
                    ? [item.kind_label, due(item), item.detail].filter(Boolean).join(" · ")
                    : [item.detail, due(item), relative(item.created_at)].filter(Boolean).join(" · ");
                  return (
                    <li key={item.key} data-row={at} className={`today-row${cursor === at ? " is-cursor" : ""}${item.overdue ? " is-overdue" : ""}${selected.has(item.key) ? " is-selected" : ""}`} onPointerEnter={() => setCursor(at)}>
                      {section.batchable ? <input type="checkbox" checked={selected.has(item.key)} disabled={busyKey !== null} onChange={() => toggle(item.key)} aria-label={`Select ${item.title}`} /> : <span className={`ui-dot ${item.overdue ? "is-needs-review" : section.deferred ? "is-stopped" : "is-waiting"}`} aria-hidden="true" />}
                      <div className="today-row-body">
                        <span className="today-row-title">{item.title}</span>
                        <small>{section.deferred && item.deferred_until ? `returns ${new Date(item.deferred_until).toLocaleDateString()}` : meta}</small>
                      </div>
                      <div className="today-row-actions">
                        {section.deferred ? (
                          <button type="button" className="ui-btn is-quiet is-sm" disabled={busyKey !== null} onClick={() => void restore(item)}>{busyKey === item.key ? "Restoring…" : "Bring back"}</button>
                        ) : (
                          <>
                            <a className="ui-btn is-sm" href={item.href || "/"}>Open</a>
                            <button type="button" className="ui-btn is-quiet is-sm" disabled={busyKey !== null} title={`Ask again in ${DEFER_DAYS} days`} onClick={() => void defer(item)}>{busyKey === item.key ? "Deferring…" : "Later"}</button>
                          </>
                        )}
                      </div>
                    </li>
                  );
                })}
              </ul>
            )}
          </section>
        ))}
      </div>

      {selected.size ? (
        <div className="today-batch" role="region" aria-label="Selected items">
          <strong>{selected.size} selected</strong>
          <button type="button" className="ui-btn is-primary is-sm" disabled={busyKey !== null} onClick={() => void runBatch("approve")}>{busyKey === "batch" ? "Applying…" : "Approve / complete"}</button>
          <button type="button" className="ui-btn is-sm" disabled={busyKey !== null} onClick={() => void runBatch("reject")}>Reject</button>
          <button type="button" className="ui-btn is-sm" disabled={busyKey !== null} onClick={() => void runBatch("defer")}>Later</button>
          <button type="button" className="ui-btn is-quiet is-sm" disabled={busyKey !== null} onClick={() => setSelected(new Set())}>Clear</button>
        </div>
      ) : null}
      {batchNote ? <p className="today-note" role="status">{batchNote}</p> : null}
    </div>
  );
}
