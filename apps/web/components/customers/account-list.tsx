"use client";

// The master column: search, a filter, the accounts, and a way to add one.
// "Everything" at the top is the dashboard across accounts.

import { useState, type KeyboardEvent } from "react";

import { createCustomer } from "@/lib/api";
import { ACCOUNT_FILTERS } from "@/lib/customers";
import type { Customers } from "@/hooks/use-customers";

export function AccountList({ c }: { c: Customers }) {
  const [name, setName] = useState("");
  const total = c.dashboard ? c.dashboard.open_actions + c.dashboard.waiting_notes : 0;

  const add = () => {
    const trimmed = name.trim();
    if (!trimmed) return;
    void c.run("add-account", async () => {
      const account = await createCustomer({ name: trimmed });
      setName("");
      c.select(account.id);
    });
  };

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
    const buttons = [...event.currentTarget.querySelectorAll<HTMLButtonElement>("button[data-account]")];
    const at = buttons.findIndex((button) => button === document.activeElement);
    const next = buttons[Math.max(0, Math.min(buttons.length - 1, at + (event.key === "ArrowDown" ? 1 : -1)))];
    if (next) { event.preventDefault(); next.focus(); }
  };

  return (
    <aside className="accounts" aria-label="Accounts">
      <input type="search" className="accounts-search" value={c.query} onChange={(event) => c.setQuery(event.target.value)} placeholder={`Search ${c.accounts.length} accounts`} aria-label="Search accounts" />
      <div className="accounts-filters" role="group" aria-label="Filter accounts">
        {ACCOUNT_FILTERS.map(([value, label]) => (
          <button key={value} type="button" className={`ui-chip${c.filter === value ? " is-accent" : ""}`} aria-pressed={c.filter === value} onClick={() => c.setFilter(value)}>{label}</button>
        ))}
      </div>
      <div className="accounts-list" onKeyDown={onKeyDown}>
        <button type="button" data-account="" className={`accounts-row is-home${c.selectedId === null ? " is-selected" : ""}`} aria-current={c.selectedId === null ? "page" : undefined} onClick={() => c.select(null)}>
          <span><strong>Everything</strong><small>All accounts, wins and open work</small></span>
          {total ? <b className={c.dashboard?.overdue_actions ? "is-alert" : ""}>{total}</b> : null}
        </button>
        {c.visibleAccounts.map((account) => (
          <button key={account.id} type="button" data-account={account.id} className={`accounts-row${c.selectedId === account.id ? " is-selected" : ""}`} aria-current={c.selectedId === account.id ? "page" : undefined} onClick={() => c.select(account.id)}>
            <span><strong>{account.name}</strong><small>{[account.industry, account.region].filter(Boolean).join(" · ") || "Customer account"}</small></span>
            {account.wins > 0 ? <i title={`${account.wins} ${account.wins === 1 ? "win" : "wins"}`}>🏆{account.wins}</i> : null}
            {account.open_actions > 0 ? <b>{account.open_actions}</b> : null}
          </button>
        ))}
        {c.loaded && !c.accounts.length ? <p className="accounts-empty">No accounts yet. Add one below.</p> : null}
        {c.accounts.length && !c.visibleAccounts.length ? (
          <p className="accounts-empty">Nothing matches. <button type="button" className="ui-btn is-quiet is-sm" onClick={() => { c.setQuery(""); c.setFilter("all"); }}>Clear</button></p>
        ) : null}
      </div>
      <form className="accounts-add" onSubmit={(event) => { event.preventDefault(); add(); }}>
        <input value={name} onChange={(event) => setName(event.target.value)} placeholder="New account" aria-label="New account name" />
        <button type="submit" className="ui-btn is-sm" disabled={c.busy === "add-account" || !name.trim()}>Add</button>
      </form>
    </aside>
  );
}
