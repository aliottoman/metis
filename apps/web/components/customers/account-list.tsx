"use client";

import { useMemo, useState } from "react";
import { ArrowUpRight, Building2, CheckCheck, Clock3, Plus, Search, Trophy } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import { SelectMenu } from "@/components/select-menu";
import { createCustomer } from "@/lib/api";
import { ACCOUNT_FILTERS, matchesFilter, when } from "@/lib/customers";
import type { Customers } from "@/hooks/use-customers";

export function AccountList({ c, adding, setAdding }: { c: Customers; adding: boolean; setAdding: (value: boolean) => void }) {
  const [name, setName] = useState("");
  const [sort, setSort] = useState("recent");
  const accounts = useMemo(() => [...c.visibleAccounts].sort((a, b) => sort === "name" ? a.name.localeCompare(b.name) : sort === "actions" ? b.open_actions - a.open_actions || a.name.localeCompare(b.name) : (b.last_interaction_at || b.updated_at).localeCompare(a.last_interaction_at || a.updated_at)), [c.visibleAccounts, sort]);
  const filtered = Boolean(c.query.trim()) || c.filter !== "all";
  const clear = () => { c.setQuery(""); c.setFilter("all"); };
  const add = () => {
    if (!name.trim()) return;
    void c.run("add-account", async () => {
      const account = await createCustomer({ name: name.trim() });
      setName("");
      setAdding(false);
      c.select(account.id);
    }, "Account created.");
  };

  return <section className="customer-catalog" aria-label="Account catalog">
    {adding ? <form className="customer-create" onSubmit={(event) => { event.preventDefault(); add(); }}>
      <span className="customer-create-mark"><Building2 size={21} aria-hidden="true" /></span>
      <label className="ui-field"><span>Give this relationship a home</span><input autoFocus value={name} disabled={Boolean(c.busy)} onChange={(event) => setName(event.target.value)} placeholder="Account or organization name" aria-label="New account name" /></label>
      <button type="button" className="ui-btn is-quiet" disabled={Boolean(c.busy)} onClick={() => setAdding(false)}>Close</button>
      <button type="submit" className="ui-btn is-primary" disabled={Boolean(c.busy) || !name.trim()}>{c.busy === "add-account" ? "Creating…" : "Create account"}</button>
    </form> : null}
    {c.dashboard ? <div className="customer-catalog-pulse" aria-label="Portfolio at a glance"><div><Building2 size={17} aria-hidden="true" /><strong>{c.dashboard.active_accounts}</strong><span>active accounts</span></div><div><CheckCheck size={17} aria-hidden="true" /><strong>{c.dashboard.open_actions}</strong><span>open actions</span></div><div><Clock3 size={17} aria-hidden="true" /><strong>{c.dashboard.waiting_notes}</strong><span>sources waiting</span></div><div><Trophy size={17} aria-hidden="true" /><strong>{c.dashboard.total_wins}</strong><span>wins recorded</span></div></div> : null}
    <div className="workspace-toolbar"><label className="workspace-search"><Search size={17} aria-hidden="true" /><input type="search" value={c.query} onChange={(event) => c.setQuery(event.target.value)} placeholder="Find an account by name, industry, or region…" aria-label="Search accounts" /></label><SelectMenu className="customer-sort" label="Sort accounts" hideLabel value={sort} onChange={setSort} options={[{ value: "recent", label: "Recently active" }, { value: "name", label: "Name, A–Z" }, { value: "actions", label: "Most open actions" }]} /></div>
    <div className="customer-catalog-filters" role="group" aria-label="Filter accounts">{ACCOUNT_FILTERS.map(([value, label]) => <button key={value} type="button" className={`ui-chip${c.filter === value ? " is-accent" : ""}`} aria-pressed={c.filter === value} onClick={() => c.setFilter(value)}>{value === "all" ? "All accounts" : value === "waiting" ? "Sources waiting" : label}<small>{c.loaded ? c.accounts.filter((account) => matchesFilter(account, value)).length : "—"}</small></button>)}</div>
    <div className="customer-results"><span role="status">{c.loaded ? `${accounts.length} ${accounts.length === 1 ? "account" : "accounts"}${filtered ? ` of ${c.accounts.length}` : " in your workspace"}` : c.error ? "Accounts unavailable" : "Loading your accounts…"}</span>{filtered ? <button type="button" className="ui-btn is-quiet is-sm" onClick={clear}>Clear filters</button> : <span>Open an account to pick up where you left off</span>}</div>
    {!c.loaded && !c.error ? <Skeleton rows={4} height={100} /> : accounts.length ? <div className="customer-card-grid">{accounts.map((account) => <button key={account.id} type="button" className="customer-card" onClick={() => c.select(account.id)}>
      <span className="customer-card-heading"><span className="customer-card-mark" aria-hidden="true">{account.name.trim().split(/\s+/).slice(0, 2).map((word) => word[0]).join("").toUpperCase()}</span><span className={`customer-account-state is-${account.status}`}>{account.status}</span><ArrowUpRight size={17} className="customer-card-arrow" aria-hidden="true" /></span>
      <span className="customer-card-title">{account.name}</span><span className="customer-card-description">{[account.industry, account.region].filter(Boolean).join(" · ") || "Account details ready to add"}</span>
      <span className="customer-card-metrics"><span><CheckCheck size={14} aria-hidden="true" />{account.open_actions} open</span><span><Clock3 size={14} aria-hidden="true" />{account.pending_notes} waiting</span>{account.wins > 0 ? <span><Trophy size={14} aria-hidden="true" />{account.wins} {account.wins === 1 ? "win" : "wins"}</span> : null}</span>
      <span className="customer-card-footer">{account.last_interaction_at ? `Last interaction ${when(account.last_interaction_at)}` : `Added ${when(account.created_at)}`}</span>
    </button>)}</div> : c.loaded ? <div className="workspace-empty"><Building2 aria-hidden="true" /><h2>{c.accounts.length ? "No matching accounts" : "Your next relationship starts here"}</h2><p>{c.accounts.length ? "Try another name or filter to find the account you need." : "Add a customer to keep notes, commitments, people, and wins in one place."}</p><button type="button" className="ui-btn is-primary" onClick={c.accounts.length ? clear : () => setAdding(true)}>{c.accounts.length ? "Show all accounts" : <><Plus size={15} aria-hidden="true" />Create your first account</>}</button></div> : null}
  </section>;
}
