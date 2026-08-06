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

  // Four figures on one line, each reading as a phrase. The earlier version
  // set them as four 32px numerals over their captions, which on a workspace
  // where three of the four are usually 0 gave the front door a row of giant
  // zeroes competing with the wordmark.
  return (
    <section className="customerSnippet" aria-label="Customer workbench summary">
      <header>
        <span className="eyebrow">Customer intelligence</span>
        <Link href="/customers">Open workbench <span aria-hidden="true">↗</span></Link>
      </header>
      <div className="customerSnippetMetrics">
        <span><b>{dashboard?.active_accounts ?? 0}</b> accounts</span>
        <span><b>{dashboard?.open_actions ?? 0}</b> open</span>
        <span className={(dashboard?.overdue_actions ?? 0) > 0 ? "attention" : ""}>
          <b>{dashboard?.overdue_actions ?? 0}</b> overdue
        </span>
        <span><b>{dashboard?.waiting_notes ?? 0}</b> waiting</span>
      </div>
      {dashboard?.recent_accounts.length ? (
        <div className="customerSnippetRecent">
          {dashboard.recent_accounts.slice(0, 3).map((account) => (
            <Link key={account.id} href={`/customers?account=${encodeURIComponent(account.id)}`}>
              {account.name}
              {account.open_actions ? <b>{account.open_actions}</b> : null}
            </Link>
          ))}
        </div>
      ) : (
        <p>Add your first account, then capture meeting notes and turn them into reviewed updates.</p>
      )}
    </section>
  );
}

