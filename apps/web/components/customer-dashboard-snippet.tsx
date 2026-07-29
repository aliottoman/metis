"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { getCustomerDashboard } from "@/lib/api";
import type { CustomerDashboard } from "@/lib/types";

export function CustomerDashboardSnippet() {
  const [dashboard, setDashboard] = useState<CustomerDashboard | null>(null);

  useEffect(() => {
    let mounted = true;
    void getCustomerDashboard()
      .then((value) => mounted && setDashboard(value))
      .catch(() => undefined);
    return () => { mounted = false; };
  }, []);

  return (
    <section className="customerSnippet" aria-label="Customer workbench summary">
      <header>
        <div>
          <span className="eyebrow">Customer intelligence</span>
          <strong>Today</strong>
        </div>
        <Link href="/customers">Open workbench <span aria-hidden="true">↗</span></Link>
      </header>
      <div className="customerSnippetMetrics">
        <span><strong>{dashboard?.active_accounts ?? 0}</strong><small>Active accounts</small></span>
        <span><strong>{dashboard?.open_actions ?? 0}</strong><small>Open actions</small></span>
        <span className={(dashboard?.overdue_actions ?? 0) > 0 ? "attention" : ""}><strong>{dashboard?.overdue_actions ?? 0}</strong><small>Overdue</small></span>
        <span><strong>{dashboard?.waiting_notes ?? 0}</strong><small>Notes waiting</small></span>
      </div>
      {dashboard?.recent_accounts.length ? (
        <div className="customerSnippetRecent">
          <small>Recent</small>
          {dashboard.recent_accounts.slice(0, 3).map((account) => (
            <Link key={account.id} href={`/customers?account=${encodeURIComponent(account.id)}`}>
              <span>{account.name}</span>
              <b>{account.open_actions} open</b>
            </Link>
          ))}
        </div>
      ) : (
        <p>Add your first account, then capture meeting notes and turn them into reviewed updates.</p>
      )}
    </section>
  );
}

