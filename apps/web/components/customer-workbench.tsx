"use client";

// Discover accounts, then open a focused workspace without losing the catalog.

import { useEffect, useState } from "react";
import { ArrowLeft, Building2, ChartNoAxesCombined, LayoutGrid, NotebookPen, Plus, Search, UsersRound } from "lucide-react";

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
  const [addingAccount, setAddingAccount] = useState(false);
  const [catalogView, setCatalogView] = useState<"accounts" | "portfolio">("accounts");
  const account = c.detail?.account.id === c.selectedId ? c.detail.account : undefined;
  const dialog = c.dialog;
  useEffect(() => { setNoteComposerOpen(false); }, [c.selectedId]);

  return (
    <div className={`customers${c.selectedId ? " is-account-workspace" : " is-customer-catalog"}`}>
      {c.selectedId ? <div className="customer-breadcrumb"><button type="button" className="ui-btn is-quiet is-sm" onClick={() => { c.select(null); setNoteComposerOpen(false); }}><ArrowLeft size={15} aria-hidden="true" />Back to customers</button><span>Account workspace</span></div> : null}
      <PageHeader
        icon={c.selectedId ? <Building2 /> : <UsersRound />}
        eyebrow={c.selectedId ? "Customer account" : "Relationships, in context"}
        title={c.selectedId ? account?.name || "Opening account…" : "Customers"}
        lede={c.selectedId ? account ? [account.industry, account.region, account.aliases.length ? `Also known as ${account.aliases.join(", ")}` : ""].filter(Boolean).join(" · ") || "Keep the people, commitments, and context for this relationship together." : "Loading notes, actions, and account context." : "Keep every relationship moving, with the right context close at hand."}
        actions={
          <>
            <button type="button" className="ui-btn is-quiet" onClick={() => c.setDialog({ kind: "search" })}><Search size={15} aria-hidden="true" />Search everything</button>
            {c.selectedId ? <><button type="button" className="ui-btn" disabled={!account} onClick={() => c.setDialog({ kind: "capture" })}>Capture source</button><button type="button" className="ui-btn is-primary" disabled={!account} onClick={() => { c.setTab("notes"); setNoteComposerOpen(true); }}><NotebookPen size={15} aria-hidden="true" />Add note</button></> : <button type="button" className="ui-btn is-primary" onClick={() => { setCatalogView("accounts"); setAddingAccount(true); }}><Plus size={15} aria-hidden="true" />New account</button>}
          </>
        }
      />
      {c.error ? <Notice kind="error" action="Try again" onAction={() => { c.setError(null); void c.refreshAll().catch((error) => c.setError(error instanceof Error ? error.message : "Customer data could not be loaded.")); }} onDismiss={c.loaded ? () => c.setError(null) : undefined}>{c.error}</Notice> : null}

      {c.selectedId === null ? <nav className="customer-catalog-nav" aria-label="Customer views"><button type="button" aria-current={catalogView === "accounts" ? "page" : undefined} onClick={() => setCatalogView("accounts")}><LayoutGrid size={16} aria-hidden="true" />Accounts <small>{c.loaded ? c.accounts.length : "—"}</small></button><button type="button" aria-current={catalogView === "portfolio" ? "page" : undefined} onClick={() => setCatalogView("portfolio")}><ChartNoAxesCombined size={16} aria-hidden="true" />Portfolio overview</button></nav> : null}
      <div className="customers-detail">
          {c.selectedId === null ? <><div hidden={catalogView !== "accounts"}><AccountList c={c} adding={addingAccount} setAdding={setAddingAccount} /></div>{catalogView === "portfolio" ? c.dashboard ? <Dashboard c={c} /> : c.error ? <div className="workspace-empty"><h2>Your portfolio is unavailable</h2><p>Retry the connection to load your customer workspace.</p></div> : <Skeleton rows={5} height={40} /> : null}</>
            : c.detail && c.detail.account.id === c.selectedId ? <AccountDetail key={c.selectedId} c={c} detail={c.detail} noteComposerOpen={noteComposerOpen} setNoteComposerOpen={setNoteComposerOpen} />
              : c.detailError ? <Notice kind="error" title="This account could not be opened" action="Try again" onAction={() => { void c.refreshAll().catch((error) => c.setError(error instanceof Error ? error.message : "Customer data could not be loaded.")); }}>{c.detailError}</Notice>
                : <Skeleton rows={6} height={32} />}
      </div>

      {dialog?.kind === "search" ? (
        <CustomerSearch onDismiss={() => c.setDialog(null)} onOpen={(hit) => { c.setDialog(null); c.select(hit.account_id, TAB_FOR_HIT[hit.kind], hit.kind === "note" || hit.kind === "source" || hit.kind === "action" || hit.kind === "fact" ? { kind: hit.kind, id: hit.id } : undefined); }} />
      ) : null}
      {dialog?.kind === "capture" && account ? <CaptureDialog key={account.id} c={c} accountId={account.id} /> : null}
      {dialog?.kind === "win" && account ? <WinDialog key={`${account.id}:${dialog.win?.id ?? "new"}`} c={c} accountId={account.id} accountName={account.name} win={dialog.win} /> : null}
      {dialog?.kind === "rates" ? <RateCardDialog c={c} /> : null}
      {c.proposal ? <ReviewDialog key={c.proposal.id} c={c} proposal={c.proposal} /> : null}
    </div>
  );
}
