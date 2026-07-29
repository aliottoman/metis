"use client";

import { useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";

import {
  analyzeCustomerSource,
  captureCustomerSource,
  createCustomer,
  createCustomerOutput,
  createCustomerWin,
  deleteCustomer,
  deleteCustomerWin,
  getCustomer,
  getCustomerDashboard,
  getCustomerSettings,
  getLocalModelSession,
  listCustomers,
  saveCustomerProposal,
  saveCustomerSettings,
  updateCustomerAction,
} from "@/lib/api";
import type {
  CustomerAccount,
  CustomerAccountDetail,
  CustomerDashboard,
  CustomerExtraction,
  CustomerOutput,
  CustomerProposal,
  CustomerSettings,
  LocalModelSession,
} from "@/lib/types";

type CustomerTab = "overview" | "timeline" | "actions" | "wins" | "technical" | "people" | "sources" | "outputs";

const TABS: Array<[CustomerTab, string]> = [
  ["overview", "Overview"],
  ["timeline", "Timeline"],
  ["actions", "Actions"],
  ["wins", "Wins"],
  ["technical", "Technical"],
  ["people", "People"],
  ["sources", "Sources"],
  ["outputs", "Outputs"],
];

const WIN_SERVICES = [
  "Generative AI Services",
  "Generative AI Agents",
  "DAC",
  "Model-Import",
  "On-demand",
  "ODA",
];

function when(value?: string | null): string {
  if (!value) return "Not dated";
  return new Intl.DateTimeFormat(undefined, { dateStyle: "medium" }).format(new Date(value));
}

// A win date is a calendar date the user picked, stored as midnight UTC. Formatting
// it in the viewer's zone would show the previous day everywhere west of UTC, so
// it is read back in UTC — the day chosen is the day shown.
function winDay(value?: string | null): string {
  if (!value) return "Not dated";
  return new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeZone: "UTC" }).format(new Date(value));
}

function arr(value?: number | null): string {
  if (value === null || value === undefined) return "—";
  return new Intl.NumberFormat("en-US", {
    style: "currency", currency: "USD", maximumFractionDigits: 0,
  }).format(value);
}

export function CustomerWorkbench() {
  const params = useSearchParams();
  const requestedAccount = params.get("account");
  const [accounts, setAccounts] = useState<CustomerAccount[]>([]);
  const [dashboard, setDashboard] = useState<CustomerDashboard | null>(null);
  const [detail, setDetail] = useState<CustomerAccountDetail | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(requestedAccount);
  const [tab, setTab] = useState<CustomerTab>("overview");
  const [session, setSession] = useState<LocalModelSession | null>(null);
  const [settings, setSettings] = useState<CustomerSettings>({ tracker_url: "", activity_template: "", updated_at: null });
  const [newAccountName, setNewAccountName] = useState("");
  const [creating, setCreating] = useState(false);
  const [captureOpen, setCaptureOpen] = useState(false);
  const [noteTitle, setNoteTitle] = useState("");
  const [noteContent, setNoteContent] = useState("");
  const [noteKind, setNoteKind] = useState<"note" | "meeting" | "chat" | "notion" | "attachment">("meeting");
  const [winOpen, setWinOpen] = useState(false);
  const [winTitle, setWinTitle] = useState("");
  const [winBrief, setWinBrief] = useState("");
  const [winServices, setWinServices] = useState<string[]>([]);
  const [winDacShape, setWinDacShape] = useState("");
  const [winArr, setWinArr] = useState("");
  const [winDate, setWinDate] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [proposal, setProposal] = useState<CustomerProposal | null>(null);
  const [review, setReview] = useState<CustomerExtraction | null>(null);
  const [output, setOutput] = useState<CustomerOutput | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refreshIndex = useCallback(async () => {
    const [nextAccounts, nextDashboard, nextSession, nextSettings] = await Promise.all([
      listCustomers(),
      getCustomerDashboard(),
      getLocalModelSession().catch(() => null),
      getCustomerSettings(),
    ]);
    setAccounts(nextAccounts);
    setDashboard(nextDashboard);
    if (nextSession) setSession(nextSession);
    setSettings(nextSettings);
    setSelectedId((current) => current || requestedAccount || nextAccounts[0]?.id || null);
  }, [requestedAccount]);

  const refreshDetail = useCallback(async (id: string) => {
    setDetail(await getCustomer(id));
  }, []);

  useEffect(() => {
    void refreshIndex().catch((loadError) => setError(loadError instanceof Error ? loadError.message : "Customer data could not be loaded."));
  }, [refreshIndex]);

  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      return;
    }
    void refreshDetail(selectedId).catch((loadError) => setError(loadError instanceof Error ? loadError.message : "That account could not be opened."));
  }, [refreshDetail, selectedId]);

  useEffect(() => {
    const listener = (event: Event) => {
      const next = (event as CustomEvent<LocalModelSession>).detail;
      if (next) setSession(next);
    };
    window.addEventListener("metis:model-session", listener);
    return () => window.removeEventListener("metis:model-session", listener);
  }, []);

  const technicalFacts = useMemo(
    () => detail?.facts.filter((item) => ["requirement", "constraint", "model", "dac_note", "risk"].includes(item.kind)) ?? [],
    [detail],
  );

  async function addAccount() {
    if (!newAccountName.trim() || creating) return;
    setCreating(true);
    setError(null);
    try {
      const account = await createCustomer({ name: newAccountName.trim() });
      setNewAccountName("");
      setSelectedId(account.id);
      await refreshIndex();
    } catch (createError) {
      setError(createError instanceof Error ? createError.message : "The account could not be added.");
    } finally {
      setCreating(false);
    }
  }

  async function removeAccount() {
    if (!detail || busy || !window.confirm(`Delete “${detail.account.name}”? This permanently removes its notes, facts, actions, and outputs.`)) return;
    const accountId = detail.account.id;
    setBusy("delete-account");
    setError(null);
    try {
      await deleteCustomer(accountId);
      const remaining = accounts.filter((account) => account.id !== accountId);
      setSelectedId(remaining[0]?.id ?? null);
      setDetail(null);
      setNotice(`Deleted ${detail.account.name}.`);
      await refreshIndex();
    } catch (deleteError) {
      setError(deleteError instanceof Error ? deleteError.message : "The account could not be deleted.");
    } finally {
      setBusy(null);
    }
  }

  async function capture() {
    if (!selectedId || !noteTitle.trim() || !noteContent.trim() || busy) return;
    setBusy("capture");
    setError(null);
    setNotice(null);
    try {
      const source = await captureCustomerSource({
        account_id: selectedId,
        title: noteTitle.trim(),
        content: noteContent.trim(),
        source_kind: noteKind,
      });
      setNoteTitle("");
      setNoteContent("");
      setCaptureOpen(false);
      setTab("sources");
      setNotice(source.status === "duplicate"
        ? "This exact note was already captured, so no duplicate record was created."
        : session?.state === "ready"
          ? "Saved locally. Choose Analyze when you are ready to review the extracted update."
          : "Saved locally as Waiting for analysis. Launch the model when you want to analyze it.");
      await Promise.all([refreshDetail(selectedId), refreshIndex()]);
    } catch (captureError) {
      setError(captureError instanceof Error ? captureError.message : "The note could not be saved.");
    } finally {
      setBusy(null);
    }
  }

  async function analyze(sourceId: string) {
    if (busy) return;
    setBusy(sourceId);
    setError(null);
    try {
      const next = await analyzeCustomerSource(sourceId);
      setProposal(next);
      setReview(structuredClone(next.extraction));
    } catch (analysisError) {
      setError(analysisError instanceof Error ? analysisError.message : "The note could not be analyzed.");
    } finally {
      setBusy(null);
    }
  }

  async function saveReview() {
    if (!proposal || !review || busy) return;
    setBusy("review");
    setError(null);
    try {
      await saveCustomerProposal(proposal.id, review);
      setProposal(null);
      setReview(null);
      setNotice("Update saved. Every fact and action remains linked to the source note.");
      if (selectedId) await Promise.all([refreshDetail(selectedId), refreshIndex()]);
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "The reviewed update could not be saved.");
    } finally {
      setBusy(null);
    }
  }

  async function recordWin() {
    if (!selectedId || !winTitle.trim() || busy) return;
    setBusy("win");
    setError(null);
    try {
      const parsedArr = winArr.trim() ? Number(winArr.replace(/[^0-9.]/g, "")) : null;
      await createCustomerWin(selectedId, {
        title: winTitle.trim(),
        brief: winBrief.trim(),
        services: winServices,
        dac_shape: winDacShape.trim(),
        yearly_arr: parsedArr !== null && Number.isFinite(parsedArr) ? parsedArr : null,
        won_at: winDate ? `${winDate}T00:00:00Z` : null,
      });
      setWinOpen(false);
      setWinTitle("");
      setWinBrief("");
      setWinServices([]);
      setWinDacShape("");
      setWinArr("");
      setWinDate("");
      setTab("wins");
      setNotice("Win recorded. The tracker above reflects it immediately.");
      await Promise.all([refreshDetail(selectedId), refreshIndex()]);
    } catch (winError) {
      setError(winError instanceof Error ? winError.message : "The win could not be recorded.");
    } finally {
      setBusy(null);
    }
  }

  async function removeWin(winId: string) {
    if (!selectedId || busy || !window.confirm("Remove this win from the tracker?")) return;
    setBusy(winId);
    setError(null);
    try {
      await deleteCustomerWin(winId);
      await Promise.all([refreshDetail(selectedId), refreshIndex()]);
    } catch (winError) {
      setError(winError instanceof Error ? winError.message : "The win could not be removed.");
    } finally {
      setBusy(null);
    }
  }

  async function toggleAction(actionId: string, current: "open" | "done" | "cancelled") {
    if (!selectedId) return;
    setBusy(actionId);
    try {
      await updateCustomerAction(actionId, current === "open" ? "done" : "open");
      await Promise.all([refreshDetail(selectedId), refreshIndex()]);
    } catch (actionError) {
      setError(actionError instanceof Error ? actionError.message : "The action could not be updated.");
    } finally {
      setBusy(null);
    }
  }

  async function generateOutput() {
    if (!selectedId || busy) return;
    setBusy("output");
    setError(null);
    try {
      setOutput(await createCustomerOutput(selectedId, detail?.interactions[0]?.id));
    } catch (outputError) {
      setError(outputError instanceof Error ? outputError.message : "The Markdown update could not be created.");
    } finally {
      setBusy(null);
    }
  }

  async function saveTrackerUrl() {
    setBusy("settings");
    try {
      setSettings(await saveCustomerSettings(settings));
      setNotice("Activity tracker link saved locally.");
    } catch (settingsError) {
      setError(settingsError instanceof Error ? settingsError.message : "The tracker link could not be saved.");
    } finally {
      setBusy(null);
    }
  }

  function copyAndOpen() {
    if (!output) return;
    void navigator.clipboard.writeText(output.content);
    const target = output.tracker_url || settings.tracker_url;
    if (target) window.open(target, "_blank", "noopener,noreferrer");
    setNotice(target ? "Markdown copied. Paste it into the tracker tab." : "Markdown copied. Add the tracker URL to open it at the same time.");
  }

  return (
    <div className="customerWorkbench">
      <header className="customerWorkbenchHeader">
        <div>
          <span className="eyebrow">Customer intelligence</span>
          <h1>Customer Workbench</h1>
          <p>Turn notes into reviewed, account-scoped facts, actions, and ready-to-paste updates.</p>
        </div>
        <button className="primaryButton" type="button" onClick={() => setCaptureOpen(true)} disabled={!selectedId}>＋ Capture note</button>
      </header>

      <section className="customerToday">
        {[
          [dashboard?.active_accounts ?? 0, "Active accounts"],
          [dashboard?.open_actions ?? 0, "Open actions"],
          [dashboard?.overdue_actions ?? 0, "Overdue"],
          [dashboard?.waiting_notes ?? 0, "Waiting for analysis"],
        ].map(([value, label]) => <article key={label}><strong>{value}</strong><span>{label}</span></article>)}
      </section>

      <section className="customerWinsTracker">
        <header>
          <div><span className="eyebrow">Win tracker</span><strong>Customer wins</strong></div>
          <button className="secondaryButton" type="button" onClick={() => setWinOpen(true)} disabled={!selectedId}>🏆 Record win</button>
        </header>
        <div className="customerWinsTiles">
          <article><strong>{dashboard?.total_wins ?? 0}</strong><span>Total wins</span></article>
          <article><strong>{dashboard?.dac_wins ?? 0}</strong><span>DAC wins</span></article>
          <article><strong>{arr(dashboard?.total_yearly_arr ?? 0)}</strong><span>Yearly ARR won</span></article>
          <article className="customerWinsServices">
            <span>By service</span>
            <div>
              {Object.entries(dashboard?.wins_by_service ?? {}).sort((a, b) => b[1] - a[1]).map(([service, count]) => (
                <b key={service}>{service} · {count}</b>
              ))}
              {!Object.keys(dashboard?.wins_by_service ?? {}).length ? <em>No wins recorded yet.</em> : null}
            </div>
          </article>
        </div>
        {dashboard?.recent_wins.length ? (
          <ul className="customerWinsRecent">
            {dashboard.recent_wins.map((win) => (
              <li key={win.id}>
                <time>{winDay(win.won_at || win.created_at)}</time>
                <button type="button" onClick={() => { setSelectedId(win.account_id); setTab("wins"); }}>{win.account_name}</button>
                <span>{win.title}</span>
                <b>{win.yearly_arr !== null ? arr(win.yearly_arr) : ""}</b>
              </li>
            ))}
          </ul>
        ) : null}
      </section>

      {error ? <div className="customerNotice error" role="alert">{error}<button type="button" onClick={() => setError(null)}>×</button></div> : null}
      {notice ? <div className="customerNotice" role="status">{notice}<button type="button" onClick={() => setNotice(null)}>×</button></div> : null}

      <div className="customerWorkbenchBody">
        <aside className="customerAccounts">
          <header><strong>Accounts</strong><span>{accounts.length}</span></header>
          <div className="customerAddAccount">
            <input value={newAccountName} onChange={(event) => setNewAccountName(event.target.value)} placeholder="New account name" onKeyDown={(event) => { if (event.key === "Enter") void addAccount(); }} />
            <button type="button" onClick={() => void addAccount()} disabled={creating || !newAccountName.trim()}>＋</button>
          </div>
          <nav>
            {accounts.map((account) => (
              <button key={account.id} className={selectedId === account.id ? "active" : ""} type="button" onClick={() => { setSelectedId(account.id); setTab("overview"); }}>
                <span><strong>{account.name}</strong><small>{account.industry || account.region || "Customer account"}</small></span>
                {account.wins > 0 ? <i className="customerWinBadge" title={`${account.wins} win${account.wins === 1 ? "" : "s"}`}>🏆{account.wins}</i> : null}
                <b>{account.open_actions}</b>
              </button>
            ))}
            {!accounts.length ? <p>No accounts yet. Add one above to start.</p> : null}
          </nav>
        </aside>

        <main className="customerAccountPanel">
          {detail ? (
            <>
              <header className="customerAccountHeader">
                <div><span className="eyebrow">Account</span><h2>{detail.account.name}</h2><p>{[detail.account.industry, detail.account.region].filter(Boolean).join(" · ") || "No profile details yet"}</p></div>
                <div className="customerAccountActions"><span className={`customerModelState state-${session?.state ?? "off"}`}><i />{session?.state === "ready" ? `${session.selected_model} ready` : "Model off · capture still works"}</span><button className="dangerButton" type="button" onClick={() => void removeAccount()} disabled={busy === "delete-account"}>{busy === "delete-account" ? "Deleting…" : "Delete customer"}</button></div>
              </header>
              <nav className="customerTabs" aria-label="Account sections">
                {TABS.map(([value, label]) => <button type="button" key={value} className={tab === value ? "active" : ""} onClick={() => setTab(value)}>{label}</button>)}
              </nav>

              {tab === "overview" ? (
                <section className="customerOverview">
                  <article><span>Last interaction</span><strong>{when(detail.account.last_interaction_at)}</strong></article>
                  <article><span>Open actions</span><strong>{detail.account.open_actions}</strong></article>
                  <article><span>Saved facts</span><strong>{detail.facts.length}</strong></article>
                  <article><span>People</span><strong>{detail.people.length}</strong></article>
                  <div className="customerSectionCard wide"><header><strong>Latest understanding</strong></header>{detail.facts.slice(0, 6).map((fact) => <p key={fact.id}><b>{fact.kind.replace("_", " ")}</b>{fact.content}</p>)}{!detail.facts.length ? <em>No reviewed facts yet.</em> : null}</div>
                  <div className="customerSectionCard wide"><header><strong>Next actions</strong></header>{detail.actions.filter((item) => item.status === "open").slice(0, 5).map((action) => <label key={action.id}><input type="checkbox" checked={false} onChange={() => void toggleAction(action.id, action.status)} /><span>{action.description}<small>{action.owner || "Unassigned"}{action.due_at ? ` · ${when(action.due_at)}` : ""}</small></span></label>)}{!detail.actions.some((item) => item.status === "open") ? <em>No open actions.</em> : null}</div>
                </section>
              ) : null}

              {tab === "timeline" ? <section className="customerTimeline">{detail.interactions.map((item) => <article key={item.id}><time>{when(item.occurred_at)}</time><div><strong>{item.title}</strong><p>{item.summary}</p></div></article>)}{!detail.interactions.length ? <div className="customerEmpty">No saved interactions yet.</div> : null}</section> : null}

              {tab === "actions" ? <section className="customerActionList">{detail.actions.map((action) => <label key={action.id} className={action.status !== "open" ? "complete" : ""}><input type="checkbox" checked={action.status === "done"} disabled={busy === action.id} onChange={() => void toggleAction(action.id, action.status)} /><span><strong>{action.description}</strong><small>{action.owner || "Unassigned"}{action.due_at ? ` · due ${when(action.due_at)}` : ""}</small></span></label>)}{!detail.actions.length ? <div className="customerEmpty">No actions captured yet.</div> : null}</section> : null}

              {tab === "wins" ? (
                <section className="customerWinList">
                  <div className="customerWinListHead">
                    <p>{detail.wins.length ? `${detail.wins.length} win${detail.wins.length === 1 ? "" : "s"} recorded for ${detail.account.name}.` : "No wins recorded for this account yet."}</p>
                    <button className="primaryButton" type="button" onClick={() => setWinOpen(true)}>🏆 Record win</button>
                  </div>
                  {detail.wins.map((win) => (
                    <article key={win.id}>
                      <header>
                        <div>
                          <strong>{win.title}</strong>
                          <small>{winDay(win.won_at || win.created_at)}{win.yearly_arr !== null ? ` · ${arr(win.yearly_arr)} yearly ARR` : ""}</small>
                        </div>
                        <button type="button" className="dangerButton" disabled={busy === win.id} onClick={() => void removeWin(win.id)}>{busy === win.id ? "Removing…" : "Remove"}</button>
                      </header>
                      {win.services.length ? <div className="customerWinChips">{win.services.map((service) => <b key={service}>{service}</b>)}</div> : null}
                      {win.dac_shape ? <small className="customerWinShape">DAC: {win.dac_shape}</small> : null}
                      {win.brief ? <p>{win.brief}</p> : null}
                    </article>
                  ))}
                </section>
              ) : null}

              {tab === "technical" ? <section className="customerFactGrid">{technicalFacts.map((fact) => <article key={fact.id}><span>{fact.kind.replace("_", " ")}</span><p>{fact.content}</p><small>{Math.round(fact.confidence * 100)}% confidence · {fact.evidence.quote || "source linked"}</small></article>)}{!technicalFacts.length ? <div className="customerEmpty">No technical requirements, risks, or model notes yet.</div> : null}</section> : null}

              {tab === "people" ? <section className="customerPeopleGrid">{detail.people.map((person) => <article key={person.name}><i>{person.name.slice(0, 1).toUpperCase()}</i><div><strong>{person.name}</strong><span>{person.role || "Role not captured"}</span><small>{person.organization}</small></div></article>)}{!detail.people.length ? <div className="customerEmpty">No people captured yet.</div> : null}</section> : null}

              {tab === "sources" ? <section className="customerSourceList">{detail.sources.map((source) => <article key={source.id}><header><div><span>{source.source_kind}</span><strong>{source.title}</strong></div><b className={`source-${source.status}`}>{source.status === "waiting" ? "Waiting for analysis" : source.status}</b></header><p>{source.content}</p><footer><small>{when(source.occurred_at || source.created_at)}</small>{source.status === "waiting" ? <button className="secondaryButton" type="button" disabled={session?.state !== "ready" || busy === source.id} onClick={() => void analyze(source.id)}>{session?.state === "ready" ? busy === source.id ? "Analyzing…" : "Analyze note" : "Launch model to analyze"}</button> : null}</footer></article>)}{!detail.sources.length ? <div className="customerEmpty">Capture a note to start the source trail.</div> : null}</section> : null}

              {tab === "outputs" ? <section className="customerOutputs"><div className="customerOutputSetup"><label><span>Company activity tracker URL</span><input value={settings.tracker_url} onChange={(event) => setSettings((current) => ({ ...current, tracker_url: event.target.value }))} placeholder="https://company.example/activity" /></label><button className="secondaryButton" type="button" onClick={() => void saveTrackerUrl()} disabled={busy === "settings"}>Save link</button></div><button className="primaryButton" type="button" onClick={() => void generateOutput()} disabled={busy === "output" || !detail.interactions.length}>{busy === "output" ? "Building…" : "Generate activity tracker Markdown"}</button>{output ? <article className="customerOutputPreview"><header><strong>Activity tracker update</strong><button type="button" className="primaryButton" onClick={copyAndOpen}>Copy Markdown &amp; open tracker</button></header><pre>{output.content}</pre></article> : null}</section> : null}
            </>
          ) : <div className="customerEmpty large"><strong>Select or add an account</strong><span>The workbench keeps every note, fact, action, and output scoped to that customer.</span></div>}
        </main>
      </div>

      {captureOpen ? <div className="customerModalBackdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) setCaptureOpen(false); }}><section className="customerModal" role="dialog" aria-modal="true" aria-label="Capture customer note"><header><div><span className="eyebrow">Raw source</span><strong>Capture a customer note</strong></div><button type="button" onClick={() => setCaptureOpen(false)}>×</button></header><label><span>Type</span><select value={noteKind} onChange={(event) => setNoteKind(event.target.value as typeof noteKind)}><option value="meeting">Meeting</option><option value="note">Note</option><option value="chat">Chat</option><option value="notion">Notion</option><option value="attachment">Attachment</option></select></label><label><span>Title</span><input value={noteTitle} onChange={(event) => setNoteTitle(event.target.value)} placeholder="Discovery call · July 28" /></label><label><span>Markdown notes</span><textarea value={noteContent} onChange={(event) => setNoteContent(event.target.value)} placeholder="Paste the original notes here. Metis saves them first; analysis is a separate action." /></label><p>{session?.state === "ready" ? "The model is ready, but analysis still waits for your explicit click." : "The model is off. This note will be saved as Waiting for analysis without launching anything."}</p><footer><button className="secondaryButton" type="button" onClick={() => setCaptureOpen(false)}>Cancel</button><button className="primaryButton" type="button" disabled={!noteTitle.trim() || !noteContent.trim() || busy === "capture"} onClick={() => void capture()}>{busy === "capture" ? "Saving…" : "Save raw note"}</button></footer></section></div> : null}

      {winOpen ? (
        <div className="customerModalBackdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) setWinOpen(false); }}>
          <section className="customerModal" role="dialog" aria-modal="true" aria-label="Record customer win">
            <header>
              <div><span className="eyebrow">Win tracker</span><strong>Record a win{detail ? ` · ${detail.account.name}` : ""}</strong></div>
              <button type="button" onClick={() => setWinOpen(false)}>×</button>
            </header>
            <label><span>Title</span><input value={winTitle} onChange={(event) => setWinTitle(event.target.value)} placeholder="Cohere Command A DAC deployed" /></label>
            <label><span>Services</span>
              <div className="customerWinServicePicker">
                {WIN_SERVICES.map((service) => (
                  <button
                    key={service}
                    type="button"
                    className={winServices.includes(service) ? "active" : ""}
                    onClick={() => setWinServices((current) => current.includes(service) ? current.filter((item) => item !== service) : [...current, service])}
                  >
                    {service}
                  </button>
                ))}
              </div>
            </label>
            <div className="customerWinFormRow">
              <label><span>Yearly ARR (USD)</span><input inputMode="numeric" value={winArr} onChange={(event) => setWinArr(event.target.value)} placeholder="110000" /></label>
              <label><span>Win date</span><input type="date" value={winDate} onChange={(event) => setWinDate(event.target.value)} /></label>
            </div>
            <label><span>DAC shape (marks it a DAC win)</span><input value={winDacShape} onChange={(event) => setWinDacShape(event.target.value)} placeholder="Model Import DAC (2xA100-40G)" /></label>
            <label><span>Brief</span><textarea value={winBrief} onChange={(event) => setWinBrief(event.target.value)} placeholder="What was deployed, for which use case, and what it unlocks." /></label>
            <footer>
              <button className="secondaryButton" type="button" onClick={() => setWinOpen(false)}>Cancel</button>
              <button className="primaryButton" type="button" disabled={!winTitle.trim() || !selectedId || busy === "win"} onClick={() => void recordWin()}>{busy === "win" ? "Recording…" : "Record win"}</button>
            </footer>
          </section>
        </div>
      ) : null}

      {proposal && review ? <div className="customerModalBackdrop"><section className="customerReviewModal" role="dialog" aria-modal="true" aria-label="Review extracted customer update"><header><div><span className="eyebrow">One review · one save</span><strong>Review customer update</strong><p>Nothing below becomes account knowledge until you save it.</p></div><button type="button" onClick={() => { setProposal(null); setReview(null); }}>×</button></header><div className="customerReviewBody"><label><span>Summary</span><textarea value={review.summary} onChange={(event) => setReview((current) => current ? { ...current, summary: event.target.value } : current)} /></label><section><header><strong>Facts</strong><span>{review.facts.length}</span></header>{review.facts.map((fact, index) => <article key={`${fact.kind}-${index}`}><select value={fact.kind} onChange={(event) => setReview((current) => current ? { ...current, facts: current.facts.map((item, itemIndex) => itemIndex === index ? { ...item, kind: event.target.value } : item) } : current)}>{["requirement", "decision", "use_case", "risk", "question", "constraint", "model", "dac_note", "other"].map((kind) => <option key={kind} value={kind}>{kind.replace("_", " ")}</option>)}</select><textarea value={fact.content} onChange={(event) => setReview((current) => current ? { ...current, facts: current.facts.map((item, itemIndex) => itemIndex === index ? { ...item, content: event.target.value } : item) } : current)} /><small>“{fact.evidence.quote || "No evidence quote"}”</small><button type="button" onClick={() => setReview((current) => current ? { ...current, facts: current.facts.filter((_, itemIndex) => itemIndex !== index) } : current)}>Remove</button></article>)}</section><section><header><strong>Actions</strong><span>{review.actions.length}</span></header>{review.actions.map((action, index) => <article key={index}><textarea value={action.description} onChange={(event) => setReview((current) => current ? { ...current, actions: current.actions.map((item, itemIndex) => itemIndex === index ? { ...item, description: event.target.value } : item) } : current)} /><input value={action.owner} placeholder="Owner" onChange={(event) => setReview((current) => current ? { ...current, actions: current.actions.map((item, itemIndex) => itemIndex === index ? { ...item, owner: event.target.value } : item) } : current)} /><small>“{action.evidence.quote || "No evidence quote"}”</small><button type="button" onClick={() => setReview((current) => current ? { ...current, actions: current.actions.filter((_, itemIndex) => itemIndex !== index) } : current)}>Remove</button></article>)}</section><section><header><strong>People</strong><span>{review.people.length}</span></header>{review.people.map((person, index) => <article key={index}><input value={person.name} onChange={(event) => setReview((current) => current ? { ...current, people: current.people.map((item, itemIndex) => itemIndex === index ? { ...item, name: event.target.value } : item) } : current)} /><input value={person.role} placeholder="Role" onChange={(event) => setReview((current) => current ? { ...current, people: current.people.map((item, itemIndex) => itemIndex === index ? { ...item, role: event.target.value } : item) } : current)} /><small>“{person.evidence.quote || "No evidence quote"}”</small><button type="button" onClick={() => setReview((current) => current ? { ...current, people: current.people.filter((_, itemIndex) => itemIndex !== index) } : current)}>Remove</button></article>)}</section></div><footer><span>Extracted locally with {proposal.model}</span><button className="primaryButton" type="button" onClick={() => void saveReview()} disabled={busy === "review"}>{busy === "review" ? "Saving…" : "Save update"}</button></footer></section></div> : null}
    </div>
  );
}
