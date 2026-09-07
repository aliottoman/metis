"use client";

// Everything at once: the ARR headline, the counts, what is open across
// accounts, and the latest wins.

import { Stat } from "@/components/ui/stat";
import { updateCustomerAction } from "@/lib/api";
import { actionMeta, isOverdue, usd, winDay } from "@/lib/customers";
import type { Customers } from "@/hooks/use-customers";

export function Dashboard({ c }: { c: Customers }) {
  const d = c.dashboard;
  if (!d) return null;
  const services = Object.entries(d.wins_by_service).sort((a, b) => b[1] - a[1]).slice(0, 6);
  const pendingEstimates = d.recent_wins.filter((win) => win.yearly_arr === null && win.valuation?.status === "proposed").length;

  return (
    <div className="dash">
      <div className="dash-stats">
        <Stat value={usd(d.total_yearly_arr, true)} label="Yearly ARR won" />
        <Stat value={d.total_wins} label={d.dac_wins ? `Wins · ${d.dac_wins} DAC` : "Wins"} />
        <Stat value={d.active_accounts} label="Accounts" />
        <Stat value={d.open_actions} label="Open actions" />
        <Stat value={d.overdue_actions} label="Overdue" attention={d.overdue_actions > 0} />
        <Stat value={d.waiting_notes} label="Notes waiting" attention={d.waiting_notes > 0} />
      </div>
      {services.length ? (
        <div className="dash-services">{services.map(([service, count]) => <span key={service} className="ui-chip">{service} <small>{count}</small></span>)}</div>
      ) : null}

      <section className="dash-section">
        <header><h2>Needs you</h2><span>{d.priority_actions.length ? `${d.priority_actions.length} open across your accounts` : "Every captured action is closed"}</span></header>
        {d.priority_actions.map((action) => (
          <article key={action.id} className={`dash-action${isOverdue(action) ? " is-overdue" : ""}`}>
            <button type="button" className="dash-account" onClick={() => c.select(action.account_id, "actions")}>{action.account_name || "Account"}</button>
            <div><strong>{action.description}</strong><small>{actionMeta(action)}</small></div>
            <button type="button" className="ui-btn is-sm" disabled={c.busy === action.id} onClick={() => void c.run(action.id, () => updateCustomerAction(action.id, "done"))}>{c.busy === action.id ? "…" : "Done"}</button>
          </article>
        ))}
      </section>

      <section className="dash-section">
        <header>
          <h2>Recent wins</h2>
          <span>{pendingEstimates ? `${pendingEstimates} estimate${pendingEstimates === 1 ? "" : "s"} awaiting review` : d.recent_wins.length ? "Newest first" : "No wins recorded yet"}</span>
          <button type="button" className="ui-btn is-quiet is-sm" disabled={!c.rateCard} onClick={() => c.setDialog({ kind: "rates" })}>Rate card</button>
        </header>
        {d.recent_wins.map((win) => (
          <article key={win.id} className="dash-win">
            <time>{winDay(win.won_at || win.created_at)}</time>
            <button type="button" className="dash-account" onClick={() => c.select(win.account_id, "wins")}>{win.account_name}</button>
            <span>{win.title}</span>
            {win.yearly_arr !== null ? <b>{usd(win.yearly_arr)}</b>
              : win.valuation?.status === "proposed" && win.valuation.estimated_yearly_arr !== null ? <b className="is-estimate" title="Estimated from the notes, not yet reviewed">~{usd(win.valuation.estimated_yearly_arr)}</b>
                : <b className="is-blank">—</b>}
          </article>
        ))}
      </section>
    </div>
  );
}
