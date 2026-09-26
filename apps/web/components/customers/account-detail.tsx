"use client";

// One account: its name and profile, the tabs, and the section for the tab.

import { useEffect, useState, type KeyboardEvent } from "react";
import { Pencil, Trophy } from "lucide-react";

import { Actions, Facts, Outputs, Overview, People, Sources, Timeline, Wins } from "@/components/customers/account-sections";
import { Notes } from "@/components/customers/notes";
import { Status } from "@/components/ui/status";
import { deleteCustomer, updateCustomer } from "@/lib/api";
import { TABS, type Customers } from "@/hooks/use-customers";
import type { CustomerAccount, CustomerAccountDetail } from "@/lib/types";

type Profile = { name: string; aliases: string; industry: string; region: string; status: CustomerAccount["status"] };

export function AccountDetail({ c, detail, noteComposerOpen, setNoteComposerOpen }: { c: Customers; detail: CustomerAccountDetail; noteComposerOpen: boolean; setNoteComposerOpen: (open: boolean) => void }) {
  const [profile, setProfile] = useState<Profile | null>(null);
  const account = detail.account;

  // A deep link lands on the exact note or action and flashes it.
  useEffect(() => {
    const target = c.requestedSource ? `source-${c.requestedSource}` : c.requestedAction ? `action-${c.requestedAction}` : c.requestedFact ? `fact-${c.requestedFact}` : null;
    const node = target ? document.getElementById(target) : null;
    if (!node || node.closest("[hidden]")) return;
    const sourceContent = node.querySelector("details");
    if (sourceContent && c.requestedSource) sourceContent.open = true;
    node.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "center" });
    node.classList.add("isDeepLinked");
    const timer = window.setTimeout(() => node.classList.remove("isDeepLinked"), 2400);
    return () => window.clearTimeout(timer);
  }, [c.requestedAction, c.requestedSource, c.requestedFact, c.tab, detail]);

  const counts: Partial<Record<string, number>> = {
    notes: detail.notes.length, wins: detail.wins.length, facts: detail.facts.length, people: detail.people.length,
    sources: detail.sources.length, actions: detail.actions.filter((item) => item.status === "open").length,
  };

  const remove = () => {
    if (!window.confirm(`Delete “${account.name}”? This permanently removes its notes, facts, actions and outputs.`)) return;
    void c.run("delete-account", async () => {
      await deleteCustomer(account.id);
      c.select(null);
    }, `Deleted ${account.name}.`);
  };

  return (
    <div className="account customer-workspace">
      {profile ? (
        <form className="account-profile" onSubmit={(event) => { event.preventDefault(); void c.run("profile", () => updateCustomer(account.id, { name: profile.name.trim(), aliases: profile.aliases.split(",").map((item) => item.trim()).filter(Boolean), industry: profile.industry.trim(), region: profile.region.trim(), status: profile.status }), "Account details saved.").then((ok) => ok && setProfile(null)); }}>
          <label className="ui-field"><span>Name</span><input value={profile.name} onChange={(event) => setProfile({ ...profile, name: event.target.value })} /></label>
          <label className="ui-field"><span>Status</span><select value={profile.status} onChange={(event) => setProfile({ ...profile, status: event.target.value as Profile["status"] })}><option value="active">Active</option><option value="paused">Paused</option><option value="archived">Archived</option></select></label>
          <label className="ui-field"><span>Industry</span><input value={profile.industry} onChange={(event) => setProfile({ ...profile, industry: event.target.value })} placeholder="Government" /></label>
          <label className="ui-field"><span>Region</span><input value={profile.region} onChange={(event) => setProfile({ ...profile, region: event.target.value })} placeholder="UAE" /></label>
          <label className="ui-field is-wide"><span>Also known as, comma separated</span><input value={profile.aliases} onChange={(event) => setProfile({ ...profile, aliases: event.target.value })} placeholder="OHI UNHCR, UNHCR Oman" /></label>
          <div className="account-profile-actions">
            <button type="button" className="ui-btn is-quiet is-sm is-danger-text customer-delete-account" disabled={Boolean(c.busy)} onClick={remove}>Delete account</button>
            <button type="button" className="ui-btn is-quiet is-sm" disabled={Boolean(c.busy)} onClick={() => setProfile(null)}>Cancel</button>
            <button type="submit" className="ui-btn is-primary is-sm" disabled={!profile.name.trim() || Boolean(c.busy)}>{c.busy === "profile" ? "Saving…" : "Save details"}</button>
          </div>
        </form>
      ) : (
        <div className="customer-workspace-meta">
          <span className={`customer-account-state is-${account.status}`}>{account.status} account</span>
          <Status state={c.modelReady ? "ready" : "waiting"} label={c.modelReady ? "Ready to analyze sources" : c.sessionState === "loading" ? "Preparing analysis" : "Capture now · analyze when ready"} />
          <span className="customer-workspace-meta-spacer" />
          <button type="button" className="ui-btn is-quiet is-sm" onClick={() => setProfile({ name: account.name, aliases: account.aliases.join(", "), industry: account.industry, region: account.region, status: account.status })}><Pencil size={13} aria-hidden="true" />Account details</button>
          <button type="button" className="ui-btn is-sm" onClick={() => c.setDialog({ kind: "win" })}><Trophy size={14} aria-hidden="true" />Record win</button>
        </div>
      )}

      <div className="customer-workspace-tabs" role="tablist" aria-label="Account sections" onKeyDown={(event: KeyboardEvent<HTMLDivElement>) => {
        if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
        const buttons = [...event.currentTarget.querySelectorAll<HTMLButtonElement>('[role="tab"]')];
        const current = buttons.indexOf(document.activeElement as HTMLButtonElement);
        const next = event.key === "Home" ? 0 : event.key === "End" ? buttons.length - 1 : (current + (event.key === "ArrowRight" ? 1 : -1) + buttons.length) % buttons.length;
        event.preventDefault(); buttons[next]?.focus(); c.setTab(TABS[next][0]);
      }}>
        {TABS.map(([value, label]) => (
          <button key={value} type="button" role="tab" id={`customer-tab-${value}`} aria-controls={`customer-panel-${value}`} aria-selected={c.tab === value} tabIndex={c.tab === value ? 0 : -1} onClick={() => c.setTab(value)}>
            {label}{counts[value] ? <small>{counts[value]}</small> : null}
          </button>
        ))}
      </div>

      <div className="customer-workspace-panels">
        <section id="customer-panel-overview" role="tabpanel" aria-labelledby="customer-tab-overview" hidden={c.tab !== "overview"}><Overview c={c} detail={detail} /></section>
        <section id="customer-panel-notes" role="tabpanel" aria-labelledby="customer-tab-notes" hidden={c.tab !== "notes"}><Notes c={c} detail={detail} composerOpen={noteComposerOpen} setComposerOpen={setNoteComposerOpen} /></section>
        <section id="customer-panel-actions" role="tabpanel" aria-labelledby="customer-tab-actions" hidden={c.tab !== "actions"}><Actions c={c} detail={detail} /></section>
        <section id="customer-panel-wins" role="tabpanel" aria-labelledby="customer-tab-wins" hidden={c.tab !== "wins"}><Wins c={c} detail={detail} /></section>
        <section id="customer-panel-facts" role="tabpanel" aria-labelledby="customer-tab-facts" hidden={c.tab !== "facts"}><Facts c={c} detail={detail} /></section>
        <section id="customer-panel-people" role="tabpanel" aria-labelledby="customer-tab-people" hidden={c.tab !== "people"}><People c={c} detail={detail} /></section>
        <section id="customer-panel-sources" role="tabpanel" aria-labelledby="customer-tab-sources" hidden={c.tab !== "sources"}><Sources c={c} detail={detail} /></section>
        <section id="customer-panel-timeline" role="tabpanel" aria-labelledby="customer-tab-timeline" hidden={c.tab !== "timeline"}><Timeline detail={detail} /></section>
        <section id="customer-panel-outputs" role="tabpanel" aria-labelledby="customer-tab-outputs" hidden={c.tab !== "outputs"}><Outputs c={c} detail={detail} /></section>
      </div>
    </div>
  );
}
