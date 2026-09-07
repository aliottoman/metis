"use client";

// The estimate on a win without a figure. A settled win shows nothing; the
// number is not up for second-guessing.

import { useState } from "react";

import { acceptWinValuation, dismissWinValuation, estimateWinValuation } from "@/lib/api";
import { usd } from "@/lib/customers";
import type { Customers } from "@/hooks/use-customers";
import type { CustomerWin } from "@/lib/types";

export function WinValuation({ win, c }: { win: CustomerWin; c: Customers }) {
  const [estimating, setEstimating] = useState(false);
  if (win.yearly_arr !== null) return null;
  const valuation = win.valuation;
  const busy = c.busy === win.id;

  const estimate = async () => {
    setEstimating(true);
    await c.run(`estimate-${win.id}`, async () => {
      const result = await estimateWinValuation(win.id);
      c.toast(result.lines.length ? `Estimated ${usd(result.estimated_yearly_arr)} from the account notes. Review it before it counts.` : "The notes did not describe anything billable.");
    });
    setEstimating(false);
  };
  const accept = (corrected?: number) => c.run(win.id, () => acceptWinValuation(win.id, corrected ?? null), `${win.title} now counts toward ARR won.`);

  if (estimating) return <div className="win-estimate is-empty">Reading this account&rsquo;s notes…</div>;
  if (!valuation || valuation.status === "dismissed" || !valuation.lines.length) {
    return (
      <div className="win-estimate is-empty">
        <span>{!valuation ? "No value recorded." : valuation.status === "dismissed" ? "Estimate dismissed." : `The notes don't describe anything billable${valuation.unpriced.length ? ` (${valuation.unpriced.join(", ")} has no rate)` : ""}.`}</span>
        <button type="button" className="ui-btn is-quiet is-sm" onClick={() => void estimate()}>{valuation ? "Estimate again" : "Estimate from notes"}</button>
      </div>
    );
  }
  return (
    <div className="win-estimate">
      <header>
        <div><span className="ui-eyebrow">Estimated · not counted yet</span><strong>{usd(valuation.estimated_yearly_arr)}<em>/year</em></strong></div>
        <span className="ui-chip">{valuation.confidence} confidence</span>
      </header>
      {valuation.explanation ? <p>{valuation.explanation}</p> : null}
      <ul>
        {valuation.lines.map((line) => (
          <li key={`${line.sku}-${line.name}`}><span><b>{line.name}</b><small>{line.basis}{line.why ? ` — ${line.why}` : ""}</small></span><i>{usd(line.yearly_amount)}</i></li>
        ))}
      </ul>
      {valuation.unpriced.length ? <small className="win-warn">No rate for {valuation.unpriced.join(", ")}; excluded from the total.</small> : null}
      {!valuation.rates_verified ? <small className="win-warn">Priced with unverified list rates. <button type="button" className="ui-btn is-quiet is-sm" onClick={() => c.setDialog({ kind: "rates" })}>Review the rate card</button></small> : null}
      <footer>
        <button type="button" className="ui-btn is-primary is-sm" disabled={busy} onClick={() => void accept()}>{busy ? "Saving…" : "Accept as ARR"}</button>
        <button type="button" className="ui-btn is-sm" disabled={busy} onClick={() => {
          const answer = window.prompt(`Yearly ARR for “${win.title}” (USD)`, String(Math.round(valuation.estimated_yearly_arr ?? 0)));
          if (answer === null) return;
          const corrected = Number(answer.replace(/[^0-9.]/g, ""));
          if (Number.isFinite(corrected)) void accept(corrected);
        }}>Edit &amp; accept</button>
        <button type="button" className="ui-btn is-quiet is-sm" disabled={busy} onClick={() => void estimate()}>Re-run</button>
        <button type="button" className="ui-btn is-quiet is-sm" disabled={busy} onClick={() => void c.run(win.id, () => dismissWinValuation(win.id))}>Dismiss</button>
        {valuation.model_used ? <small>via {valuation.model_used}</small> : null}
      </footer>
    </div>
  );
}
