"use client";

// The Customers page: accounts on the left, the open account (or everything
// at once) on the right. State lives in useCustomers.

import { useState } from "react";

import { CustomerSearch } from "@/components/customer-search";
import { AccountDetail } from "@/components/customers/account-detail";
import { AccountList } from "@/components/customers/account-list";
import { Dashboard } from "@/components/customers/dashboard";
import { CaptureDialog, RateCardDialog, ReviewDialog, WinDialog } from "@/components/customers/dialogs";
import { Notice } from "@/components/ui/notice";
import { PageHeader } from "@/components/ui/page-header";
import { Skeleton } from "@/components/ui/skeleton";
import { useCustomers, type Tab } from "@/hooks/use-customers";
import type { CustomerSearchHit } from "@/lib/types";

const TAB_FOR_HIT: Record<CustomerSearchHit["kind"], Tab> = { account: "overview", note: "notes", fact: "facts", action: "actions", win: "wins", source: "sources" };

export function CustomerWorkbench() {
  const c = useCustomers();
  const [noteComposerOpen, setNoteComposerOpen] = useState(false);
  const account = c.detail?.account;
  const dialog = c.dialog;

  return (
    <div className="customers">
      <PageHeader
        eyebrow="Customer intelligence"
        title="Customers"
        lede="Notes become reviewed, account-scoped facts, actions and ready-to-paste updates."
        actions={
          <>
            <button type="button" className="ui-btn is-quiet" onClick={() => c.setDialog({ kind: "search" })}>Search everything <kbd>⌘⇧K</kbd></button>
            <button type="button" className="ui-btn" disabled={!account} onClick={() => { c.setTab("notes"); setNoteComposerOpen(true); }}>Add note</button>
            <button type="button" className="ui-btn" disabled={!account} onClick={() => c.setDialog({ kind: "win" })}>Record win</button>
            <button type="button" className="ui-btn is-primary" disabled={!account} onClick={() => c.setDialog({ kind: "capture" })}>Capture note</button>
          </>
        }
      />
      {c.error ? <Notice kind="error" onDismiss={() => c.setError(null)}>{c.error}</Notice> : null}

      <div className="customers-body">
        <AccountList c={c} />
        <main className="customers-detail">
          {c.selectedId === null ? (c.dashboard ? <Dashboard c={c} /> : <Skeleton rows={5} height={40} />)
            : c.detail && c.detail.account.id === c.selectedId ? <AccountDetail key={c.selectedId} c={c} detail={c.detail} noteComposerOpen={noteComposerOpen} setNoteComposerOpen={setNoteComposerOpen} />
              : <Skeleton rows={6} height={32} />}
        </main>
      </div>

      {dialog?.kind === "search" ? (
        <CustomerSearch onDismiss={() => c.setDialog(null)} onOpen={(hit) => { c.setDialog(null); c.setQuery(""); c.setFilter("all"); c.select(hit.account_id, TAB_FOR_HIT[hit.kind]); }} />
      ) : null}
      {dialog?.kind === "capture" && account ? <CaptureDialog c={c} accountId={account.id} /> : null}
      {dialog?.kind === "win" && account ? <WinDialog key={dialog.win?.id ?? "new"} c={c} accountId={account.id} accountName={account.name} win={dialog.win} /> : null}
      {dialog?.kind === "rates" ? <RateCardDialog c={c} /> : null}
      {c.proposal ? <ReviewDialog key={c.proposal.id} c={c} proposal={c.proposal} /> : null}
    </div>
  );
}
