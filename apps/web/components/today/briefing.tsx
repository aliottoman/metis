"use client";

import Link from "next/link";
import { AlarmClock, ArrowUpRight, Check, ClipboardCopy, Sprout } from "lucide-react";

import { useToast } from "@/components/ui/toast";
import { dailySignals, nextStepFor, preparedStepsFor, reasonFor } from "@/lib/daily-brief";
import type { AttentionFeed, AttentionItem } from "@/lib/types";

import "./briefing.css";

export function DailyBriefing({ feed, busyKey, onDefer, onComplete, onError }: {
  feed: AttentionFeed;
  busyKey: string | null;
  onDefer: (item: AttentionItem) => void;
  onComplete: (item: AttentionItem) => void;
  onError: (message: string) => void;
}) {
  const toast = useToast();
  const { focus, neglected, opportunities } = dailySignals(feed);
  const copySteps = async (item: AttentionItem) => {
    try {
      await navigator.clipboard.writeText(preparedStepsFor(item));
      toast("Next steps copied. Ready to adapt and use.");
    } catch {
      onError("The next steps could not be copied. Open the prepared checklist to select its text.");
    }
  };

  return (
    <div className="daily-briefing">
      <dl className="daily-pulse" aria-label="Daily briefing at a glance">
        <div><dt>Open commitments</dt><dd>{feed.counts.customer_action ?? 0}</dd></div>
        <div className={neglected.length ? "needs-attention" : ""}><dt>Need a follow-up</dt><dd>{neglected.length}</dd></div>
        <div><dt>Opportunity signals</dt><dd>{opportunities.length}</dd></div>
        <div><dt>Awaiting approval</dt><dd>{feed.counts.run_approval ?? 0}</dd></div>
      </dl>

      {focus.length ? (
        <section className="daily-focus" aria-labelledby="daily-focus-title">
          <header className="daily-section-heading">
            <div><span className="ui-eyebrow">Make today count</span><h2 id="daily-focus-title">{focus.length === 3 ? "Your three highest-value next actions" : "Your next actions"}</h2></div>
            <p>Prioritized from recorded deadlines, paused work, and customer needs.</p>
          </header>
          <div className="daily-focus-grid">
            {focus.map((item, index) => (
              <article key={item.key} className={`daily-action${item.overdue ? " is-overdue" : ""}`}>
                <header><span className="daily-rank">0{index + 1}</span><span>{item.account_name || item.kind_label}</span>{item.overdue ? <span className="daily-urgency">Overdue</span> : null}</header>
                <h3>{item.title}</h3>
                <p className="daily-reason">{reasonFor(item)}</p>
                <div className="daily-next-step"><strong>Next step</strong><p>{nextStepFor(item)}</p></div>
                <details className="daily-prepared">
                  <summary>View prepared checklist</summary>
                  <pre>{preparedStepsFor(item)}</pre>
                  <button type="button" className="ui-btn is-quiet is-sm" onClick={() => void copySteps(item)}><ClipboardCopy size={14} aria-hidden="true" /> Copy next steps</button>
                </details>
                <footer>
                  <Link className="ui-btn is-primary is-sm" href={item.href || "/"}>{item.kind === "run_approval" ? "Review approval" : item.kind === "customer_opportunity" ? "Explore opportunity" : "Open next step"}<ArrowUpRight size={14} aria-hidden="true" /></Link>
                  <Link className="ui-btn is-quiet is-sm" href={item.source_href || item.href || "/"}>View source</Link>
                  <div className="daily-action-decisions">
                    {item.kind === "customer_action" ? <button type="button" className="ui-btn is-quiet is-sm" disabled={busyKey !== null} onClick={() => onComplete(item)}><Check size={14} aria-hidden="true" />{busyKey === item.key ? "Updating…" : "Mark complete"}</button> : null}
                    <button type="button" className="ui-btn is-quiet is-sm" disabled={busyKey !== null} onClick={() => onDefer(item)} title="Return to this in seven days">{busyKey === item.key ? "Updating…" : "Later"}</button>
                  </div>
                </footer>
              </article>
            ))}
          </div>
        </section>
      ) : null}

      <div className="daily-signals">
        <section className="daily-signal-panel" aria-labelledby="daily-neglected-title">
          <header><AlarmClock size={18} aria-hidden="true" /><h2 id="daily-neglected-title">Commitments needing attention</h2><span>{neglected.length}</span></header>
          <p>Overdue, or without a due date and no recorded update for at least seven days.</p>
          {neglected.length ? <ul>{neglected.slice(0, 4).map((item) => <li key={item.key}><Link href={item.href}><strong>{item.title}</strong><small>{item.account_name ? `${item.account_name} · ` : ""}{reasonFor(item)}</small></Link><button type="button" className="ui-btn is-quiet is-sm" disabled={busyKey !== null} onClick={() => onComplete(item)} aria-label={`Mark complete: ${item.title}`}><Check size={15} aria-hidden="true" /></button></li>)}</ul> : <div className="daily-signal-empty"><Check size={16} aria-hidden="true" /><span>No neglected commitments in your current queue.</span></div>}
          {neglected.length > 4 ? <Link className="ui-btn is-quiet is-sm" href="/customers">Review all {neglected.length} commitments <ArrowUpRight size={14} aria-hidden="true" /></Link> : null}
        </section>
        <section className="daily-signal-panel" aria-labelledby="daily-opportunities-title">
          <header><Sprout size={18} aria-hidden="true" /><h2 id="daily-opportunities-title">Emerging opportunities</h2><span>{opportunities.length}</span></header>
          <p>Recently reviewed customer needs to qualify. These signals are grounded in saved records.</p>
          {opportunities.length ? <ul>{opportunities.slice(0, 4).map((item) => <li key={item.key}><Link href={item.source_href || item.href}><strong>{item.title}</strong><small>{item.account_name ? `${item.account_name} · ` : ""}Review the evidence and agree a follow-up.</small></Link><button type="button" className="ui-btn is-quiet is-sm" onClick={() => void copySteps(item)} aria-label={`Copy next steps for ${item.title}`}><ClipboardCopy size={15} aria-hidden="true" /></button></li>)}</ul> : <div className="daily-signal-empty"><Sprout size={16} aria-hidden="true" /><span>No new signals in reviewed records. Capture and review your latest customer conversations.</span></div>}
          {opportunities.length > 4 ? <Link className="ui-btn is-quiet is-sm" href="/customers">Explore all {opportunities.length} signals <ArrowUpRight size={14} aria-hidden="true" /></Link> : null}
        </section>
      </div>
    </div>
  );
}
