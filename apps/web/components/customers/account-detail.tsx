"use client";

// One account: its name and profile, the tabs, and the section for the tab.

import { useEffect, useState } from "react";

import { Actions, Facts, Notes, Outputs, Overview, People, Sources, Timeline, Wins } from "@/components/customers/account-sections";
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
    const target = c.requestedSource ? `source-${c.requestedSource}` : c.requestedAction ? `action-${c.requestedAction}` : null;
    const node = target ? document.getElementById(target) : null;
    if (!node) return;
    node.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "center" });
    node.classList.add("isDeepLinked");
    const timer = window.setTimeout(() => node.classList.remove("isDeepLinked"), 2400);
    return () => window.clearTimeout(timer);
  }, [c.requestedAction, c.requestedSource, c.tab, detail]);

  const counts: Partial<Record<string, number>> = {
    notes: detail.notes.length, wins: detail.wins.length, facts: detail.facts.length, people: detail.people.length,
    sources: detail.sources.length, actions: detail.actions.filter((item) => item.status === "open").length,
  };

  const remove = () => {
    if (!window.confirm(`Delete “${account.name}”? This permanently removes its notes, facts, actions and outputs.`)) return;
    void c.run("delete-account", () => deleteCustomer(account.id), `Deleted ${account.name}.`).then((ok) => ok && c.select(null));
  };

  return (
    <div className="account">
      {profile ? (
        <form className="account-profile" onSubmit={(event) => { event.preventDefault(); void c.run("profile", () => updateCustomer(account.id, { name: profile.name.trim(), aliases: profile.aliases.split(",").map((item) => item.trim()).filter(Boolean), industry: profile.industry.trim(), region: profile.region.trim(), status: profile.status }), "Account details saved.").then((ok) => ok && setProfile(null)); }}>
          <label className="ui-field"><span>Name</span><input value={profile.name} onChange={(event) => setProfile({ ...profile, name: event.target.value })} /></label>
          <label className="ui-field"><span>Status</span><select value={profile.status} onChange={(event) => setProfile({ ...profile, status: event.target.value as Profile["status"] })}><option value="active">Active</option><option value="paused">Paused</option><option value="archived">Archived</option></select></label>
          <label className="ui-field"><span>Industry</span><input value={profile.industry} onChange={(event) => setProfile({ ...profile, industry: event.target.value })} placeholder="Government" /></label>
          <label className="ui-field"><span>Region</span><input value={profile.region} onChange={(event) => setProfile({ ...profile, region: event.target.value })} placeholder="UAE" /></label>
          <label className="ui-field is-wide"><span>Also known as, comma separated</span><input value={profile.aliases} onChange={(event) => setProfile({ ...profile, aliases: event.target.value })} placeholder="OHI UNHCR, UNHCR Oman" /></label>
          <div className="account-profile-actions">
            <button type="button" className="ui-btn is-quiet is-sm" onClick={() => setProfile(null)}>Cancel</button>
            <button type="submit" className="ui-btn is-primary is-sm" disabled={!profile.name.trim() || c.busy === "profile"}>{c.busy === "profile" ? "Saving…" : "Save details"}</button>
          </div>
        </form>
      ) : (
        <header className="account-head">
          <div>
            <h2>{account.name}</h2>
            <p>{[account.status !== "active" ? account.status : "", account.industry, account.region, account.aliases.length ? `aka ${account.aliases.join(", ")}` : ""].filter(Boolean).join(" · ") || "No profile details yet"}</p>
          </div>
          <Status state={c.modelReady ? "ready" : c.sessionState === "loading" ? "waiting" : "stopped"} label={c.modelLabel ? `${c.modelLabel} ready` : "Model off · capture still works"} />
          <button type="button" className="ui-btn is-quiet is-sm" onClick={() => setProfile({ name: account.name, aliases: account.aliases.join(", "), industry: account.industry, region: account.region, status: account.status })}>Edit details</button>
          <button type="button" className="ui-btn is-quiet is-sm is-danger-text" disabled={c.busy === "delete-account"} onClick={remove}>{c.busy === "delete-account" ? "Deleting…" : "Delete"}</button>
        </header>
      )}

      <nav className="account-tabs" aria-label="Account sections">
        {TABS.map(([value, label]) => (
          <button key={value} type="button" className={c.tab === value ? "is-active" : ""} aria-current={c.tab === value ? "page" : undefined} onClick={() => c.setTab(value)}>
            {label}{counts[value] ? <small>{counts[value]}</small> : null}
          </button>
        ))}
      </nav>

      {c.tab === "overview" ? <Overview c={c} detail={detail} /> : null}
      {c.tab === "notes" ? <Notes c={c} detail={detail} composerOpen={noteComposerOpen} setComposerOpen={setNoteComposerOpen} /> : null}
      {c.tab === "actions" ? <Actions c={c} detail={detail} /> : null}
      {c.tab === "wins" ? <Wins c={c} detail={detail} /> : null}
      {c.tab === "facts" ? <Facts c={c} detail={detail} /> : null}
      {c.tab === "people" ? <People c={c} detail={detail} /> : null}
      {c.tab === "sources" ? <Sources c={c} detail={detail} /> : null}
      {c.tab === "timeline" ? <Timeline detail={detail} /> : null}
      {c.tab === "outputs" ? <Outputs c={c} detail={detail} /> : null}
    </div>
  );
}
