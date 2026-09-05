"use client";

// The Today page — the front door. It answers "what needs me now" as a single
// screen you don't scroll: a compact head, a rail of piles you can read at a
// glance, and a deck of cards you roll through sideways. Each card is one pile
// (Start here, a kind of work, deferred, or the day's recap); its items scroll
// inside the card only when there are many, so the page itself never grows.

import Link from "next/link";
import type { CSSProperties } from "react";
import { useCallback, useEffect, useRef, useState } from "react";

import {
  batchAttention,
  deferAttention,
  getAttention,
  getMorningBrief,
  getMorningBriefAudio,
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

/** Where the Listen button is in its own little life cycle. */
type ListenState = "idle" | "loading" | "playing" | "paused";

const BRIEF_SPEEDS = [1, 1.25, 1.5, 2] as const;

/** A player clock that stays useful for clips longer than an hour. */
function playerClock(seconds: number): string {
  const whole = Math.max(0, Math.floor(Number.isFinite(seconds) ? seconds : 0));
  const minutes = Math.floor(whole / 60);
  const hours = Math.floor(minutes / 60);
  const tail = `${String(minutes % 60).padStart(hours ? 2 : 1, "0")}:${String(whole % 60).padStart(2, "0")}`;
  return hours ? `${hours}:${tail}` : tail;
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
  const [briefRefreshing, setBriefRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busyKey, setBusyKey] = useState<string | null>(null);
  const [selected, setSelected] = useState<ReadonlySet<string>>(new Set());
  const [batchNote, setBatchNote] = useState<string | null>(null);
  const [brief, setBrief] = useState<MorningBrief | null>(null);
  const [listen, setListen] = useState<ListenState>("idle");
  const [listenError, setListenError] = useState<string | null>(null);
  const [briefPosition, setBriefPosition] = useState(0);
  const [briefDuration, setBriefDuration] = useState(0);
  const [briefSpeed, setBriefSpeed] = useState(1);
  const [briefVolume, setBriefVolume] = useState(0.9);
  const [briefPreferencesReady, setBriefPreferencesReady] = useState(false);
  const [activePileId, setActivePileId] = useState<string | null>(null);
  const playerRef = useRef<HTMLAudioElement | null>(null);
  // The epoch prevents stale UI work and the controller stops the actual
  // request when Stop is pressed or this view is left.
  const listenRequestRef = useRef(0);
  const listenAbortRef = useRef<AbortController | null>(null);
  // The rendered clip, held for the session. Fetched once; replaying it after
  // that is a local decision rather than a second request, and the server has
  // already cached the rendering behind it either way.
  const clipUrlRef = useRef<string | null>(null);

  useEffect(
    () => () => {
      listenRequestRef.current += 1;
      listenAbortRef.current?.abort();
      listenAbortRef.current = null;
      const player = playerRef.current;
      if (player) {
        player.pause();
        player.removeAttribute("src");
        player.load();
      }
      if (clipUrlRef.current) URL.revokeObjectURL(clipUrlRef.current);
      clipUrlRef.current = null;
    },
    [],
  );

  useEffect(() => {
    const speedValue = window.localStorage.getItem("metis.brief.speed");
    const volumeValue = window.localStorage.getItem("metis.brief.volume");
    const storedSpeed = speedValue === null ? Number.NaN : Number(speedValue);
    const storedVolume = volumeValue === null ? Number.NaN : Number(volumeValue);
    if (speedValue !== null && BRIEF_SPEEDS.includes(storedSpeed as (typeof BRIEF_SPEEDS)[number])) {
      setBriefSpeed(storedSpeed);
    }
    if (volumeValue !== null && Number.isFinite(storedVolume) && storedVolume >= 0 && storedVolume <= 1) {
      setBriefVolume(storedVolume);
    }
    setBriefPreferencesReady(true);
  }, []);

  useEffect(() => {
    if (!briefPreferencesReady) return;
    const player = playerRef.current;
    if (!player) return;
    player.playbackRate = briefSpeed;
    player.volume = briefVolume;
    window.localStorage.setItem("metis.brief.speed", String(briefSpeed));
    window.localStorage.setItem("metis.brief.volume", String(briefVolume));
  }, [briefSpeed, briefVolume, briefPreferencesReady]);

  /** Play, pause, resume, or replay — whichever the button currently means. */
  async function toggleListen() {
    const player = playerRef.current;
    if (!player) return;
    if (listen === "playing") {
      player.pause();
      return;
    }
    if (listen === "paused" || clipUrlRef.current) {
      // Ended clips land back on "idle" with the URL still loaded, so this is
      // both "resume" and "replay" depending on where the playhead is.
      void player.play().catch(() => setListenError("That recording could not be played."));
      return;
    }
    const requestId = ++listenRequestRef.current;
    listenAbortRef.current?.abort();
    const controller = new AbortController();
    listenAbortRef.current = controller;
    setListen("loading");
    setListenError(null);
    try {
      const clip = await getMorningBriefAudio(24, controller.signal);
      if (requestId !== listenRequestRef.current) return;
      clipUrlRef.current = URL.createObjectURL(clip);
      player.src = clipUrlRef.current;
      player.playbackRate = briefSpeed;
      player.volume = briefVolume;
      await player.play();
    } catch (playError) {
      if (requestId !== listenRequestRef.current) return;
      if (playError instanceof DOMException && playError.name === "AbortError") return;
      setListen("idle");
      setListenError(
        playError instanceof Error ? playError.message : "The brief could not be read aloud.",
      );
    } finally {
      if (listenAbortRef.current === controller) listenAbortRef.current = null;
    }
  }

  /** Stop means stop: silence now, reset the playhead, and cancel preparation. */
  function stopBrief() {
    listenRequestRef.current += 1;
    listenAbortRef.current?.abort();
    listenAbortRef.current = null;
    const player = playerRef.current;
    if (player) {
      player.pause();
      player.currentTime = 0;
    }
    setBriefPosition(0);
    setListen("idle");
    setListenError(null);
  }

  function moveBrief(by: number) {
    const player = playerRef.current;
    if (!player) return;
    player.currentTime = Math.max(0, Math.min(player.duration || 0, player.currentTime + by));
    setBriefPosition(player.currentTime);
  }

  function seekBrief(value: number) {
    const player = playerRef.current;
    if (!player) return;
    player.currentTime = value;
    setBriefPosition(value);
  }

  // Load the queue.
  const refresh = useCallback(async (regenerateBrief = false) => {
    setLoading(true);
    if (regenerateBrief) setBriefRefreshing(true);
    setError(null);
    if (regenerateBrief) {
      listenRequestRef.current += 1;
      listenAbortRef.current?.abort();
      listenAbortRef.current = null;
      playerRef.current?.pause();
      playerRef.current?.removeAttribute("src");
      playerRef.current?.load();
      if (clipUrlRef.current) URL.revokeObjectURL(clipUrlRef.current);
      clipUrlRef.current = null;
      setBriefPosition(0);
      setBriefDuration(0);
      setListen("idle");
    }
    const briefRequest = getMorningBrief(24, regenerateBrief)
      .then(setBrief)
      .catch(() => undefined);
    try {
      setFeed(await getAttention(3));
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Could not read the queue.");
    } finally {
      setLoading(false);
    }
    await briefRequest;
    if (regenerateBrief) setBriefRefreshing(false);
  }, []);

  useEffect(() => {
    // The brief begins in parallel, while the decision queue stays the first
    // thing the page waits for.
    void refresh(false);
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

  const slideIds = slides.map((slide) => slide.id).join("|");
  useEffect(() => {
    const ids = slideIds ? slideIds.split("|") : [];
    if (!ids.length) {
      setActivePileId(null);
      return;
    }
    setActivePileId((current) => current && ids.includes(current) ? current : ids[0]!);
    const root = document.querySelector<HTMLElement>(".appMain");
    const elements = ids
      .map((id) => document.getElementById(`pile-${id}`))
      .filter((element): element is HTMLElement => Boolean(element));
    if (!elements.length || typeof IntersectionObserver === "undefined") return;
    const visible = new Map<string, number>();
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          const id = (entry.target as HTMLElement).id.replace(/^pile-/, "");
          if (entry.isIntersecting) visible.set(id, entry.intersectionRatio);
          else visible.delete(id);
        }
        const next = [...visible.entries()].sort((a, b) => b[1] - a[1])[0]?.[0];
        if (next) setActivePileId(next);
      },
      { root, rootMargin: "-18% 0px -52%", threshold: [0.05, 0.25, 0.5, 0.75] },
    );
    elements.forEach((element) => observer.observe(element));
    return () => observer.disconnect();
  }, [slideIds]);

  // Jump to a pile from the rail. Every pile is on screen at once now, so this
  // just brings the named card into view instead of rolling a one-at-a-time deck.
  const scrollToPile = useCallback((id: string) => {
    setActivePileId(id);
    document
      .getElementById(`pile-${id}`)
      ?.scrollIntoView({
        behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth",
        block: "nearest",
      });
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
      <section className="todayStudioHero" aria-labelledby="today-heading">
        <header className="todayTopbar">
          <div className="todayTopbarCopy">
            <span className="eyebrow">{todayLabel()}</span>
            <h1 id="today-heading">{loading && !feed ? "Checking what's waiting" : headline}</h1>
          </div>
          <div className="todayTopbarActions">
            <button className="secondaryButton todayRefresh" type="button" onClick={() => void refresh(true)} disabled={loading || briefRefreshing}>
              {loading || briefRefreshing ? "Refreshing…" : "Refresh brief"}
            </button>
          </div>
        </header>

        <section className={`dailyBriefPlayer is-${listen}`} aria-label="Daily brief player">
          <div className="dailyBriefIdentity">
            <span className="dailyBriefMark" aria-hidden="true"><i /><i /><i /></span>
            <div>
              <span className="eyebrow">Daily brief · ElevenLabs</span>
              <strong>{brief?.narrative || brief?.recommendation || "Your priority scan, read aloud when you want it."}</strong>
              {brief?.narrative && brief.recommendation ? (
                <span className="dailyBriefRecommendation">First move · {brief.recommendation}</span>
              ) : null}
            </div>
          </div>

          <div className="dailyBriefTransport">
          <button
            className="dailyBriefPlay"
            type="button"
            onClick={() => void toggleListen()}
            disabled={listen === "loading"}
            aria-label={listen === "playing" ? "Pause daily brief" : "Play daily brief"}
          >
            <span aria-hidden="true">{listen === "loading" ? "…" : listen === "playing" ? "Ⅱ" : "▶"}</span>
          </button>
          <button className="dailyBriefSkip is-back" type="button" onClick={() => moveBrief(-15)} disabled={!clipUrlRef.current} aria-label="Go back 15 seconds">−15</button>
          <div className="dailyBriefTimeline">
            <input
              type="range"
              min={0}
              max={Math.max(briefDuration, 1)}
              step={0.1}
              value={Math.min(briefPosition, Math.max(briefDuration, 1))}
              onChange={(event) => seekBrief(Number(event.target.value))}
              disabled={!briefDuration}
              aria-label="Daily brief position"
              style={{ "--brief-progress": `${briefDuration ? (briefPosition / briefDuration) * 100 : 0}%` } as CSSProperties}
            />
            <span>{playerClock(briefPosition)} <i>/</i> {briefDuration ? playerClock(briefDuration) : "—:—"}</span>
          </div>
          <button className="dailyBriefSkip is-forward" type="button" onClick={() => moveBrief(15)} disabled={!clipUrlRef.current} aria-label="Go forward 15 seconds">+15</button>
          <button
            className="dailyBriefSpeed"
            type="button"
            onClick={() => {
              const index = BRIEF_SPEEDS.indexOf(briefSpeed as (typeof BRIEF_SPEEDS)[number]);
              setBriefSpeed(BRIEF_SPEEDS[(index + 1) % BRIEF_SPEEDS.length]);
            }}
            aria-label={`Playback speed ${briefSpeed} times. Change speed`}
          >
            {briefSpeed}×
          </button>
          <label className="dailyBriefVolume" title="Brief volume">
            <span aria-hidden="true">◖</span>
            <input type="range" min={0} max={1} step={0.05} value={briefVolume} onChange={(event) => setBriefVolume(Number(event.target.value))} aria-label="Daily brief volume" />
          </label>
          <button
            className="dailyBriefStop"
            type="button"
            onClick={stopBrief}
            disabled={listen === "idle" && briefPosition === 0}
          >
            <span aria-hidden="true">■</span> Stop
          </button>
          </div>

          <audio
            ref={playerRef}
            hidden
            onLoadedMetadata={(event) => setBriefDuration(event.currentTarget.duration || 0)}
            onDurationChange={(event) => setBriefDuration(event.currentTarget.duration || 0)}
            onTimeUpdate={(event) => setBriefPosition(event.currentTarget.currentTime)}
            onPlay={() => setListen("playing")}
            onPause={() => setListen((state) => (state === "playing" ? "paused" : state))}
            onEnded={(event) => { event.currentTarget.currentTime = 0; setListen("idle"); setBriefPosition(0); }}
            onError={() => {
              setListen("idle");
              setListenError("That recording could not be played.");
            }}
          />
        </section>

        {slides.length ? (
          <nav className="todayRail" aria-label="Piles">
            {slides.map((slide) => (
              <button
                key={slide.id}
                type="button"
                className={cx("todayRailChip", activePileId === slide.id && "isActive")}
                data-kind={slide.kind}
                data-variant={slide.variant}
                aria-current={activePileId === slide.id ? "location" : undefined}
                onClick={() => scrollToPile(slide.id)}
              >
                <span className="todayRailCount">{railCounts.get(slide.id)}</span>
                <span className="todayRailLabel">{slide.label}</span>
              </button>
            ))}
          </nav>
        ) : null}
      </section>

      {listenError ? (
        <div className="composerError todayError" role="alert">
          <span>!</span>
          <p>{listenError}</p>
          <button className="errorDismiss" type="button" aria-label="Dismiss" onClick={() => setListenError(null)}>×</button>
        </div>
      ) : null}

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
