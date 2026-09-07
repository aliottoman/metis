"use client";

// The four dialogs: capture a raw note, record or edit a win, the rate card,
// and the review of an extracted update. One Modal shell under all of them.

import { useEffect, useState, type ReactNode } from "react";
import { X } from "lucide-react";

import { captureCustomerSource, createCustomerWin, estimateWinValuation, saveSkuRates, updateCustomerWin } from "@/lib/api";
import { dateInputValue, FACT_KINDS, parseUsd, WIN_SERVICES } from "@/lib/customers";
import type { Customers } from "@/hooks/use-customers";
import type { CustomerExtraction, CustomerProposal, CustomerWin } from "@/lib/types";

export function Modal({ title, eyebrow, onClose, wide, children, footer }: { title: string; eyebrow?: string; onClose: () => void; wide?: boolean; children: ReactNode; footer?: ReactNode }) {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => { if (event.key === "Escape") onClose(); };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
      <section className={`modal${wide ? " is-wide" : ""}`} role="dialog" aria-modal="true" aria-label={title}>
        <header className="modal-head">
          <div>{eyebrow ? <span className="ui-eyebrow">{eyebrow}</span> : null}<strong>{title}</strong></div>
          <button type="button" className="ui-btn is-quiet is-sm" aria-label="Close" onClick={onClose}><X size={16} /></button>
        </header>
        <div className="modal-body">{children}</div>
        {footer ? <footer className="modal-foot">{footer}</footer> : null}
      </section>
    </div>
  );
}

export function CaptureDialog({ c, accountId }: { c: Customers; accountId: string }) {
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [kind, setKind] = useState<"note" | "meeting" | "chat" | "notion" | "attachment">("meeting");
  const close = () => c.setDialog(null);
  const save = () => void c.run("capture", async () => {
    const source = await captureCustomerSource({ account_id: accountId, title: title.trim(), content: content.trim(), source_kind: kind });
    c.toast(source.status === "duplicate" ? "This exact note was already captured." : c.modelReady ? "Saved. Choose Analyze when you want the extracted update." : "Saved as waiting for analysis.");
    c.setTab("sources");
  }).then((ok) => ok && close());
  return (
    <Modal title="Capture a customer note" eyebrow="Raw source" onClose={close} footer={<><button type="button" className="ui-btn is-sm" onClick={close}>Cancel</button><button type="button" className="ui-btn is-primary is-sm" disabled={!title.trim() || !content.trim() || c.busy === "capture"} onClick={save}>{c.busy === "capture" ? "Saving…" : "Save raw note"}</button></>}>
      <label className="ui-field"><span>Type</span><select value={kind} onChange={(event) => setKind(event.target.value as typeof kind)}>{["meeting", "note", "chat", "notion", "attachment"].map((item) => <option key={item} value={item}>{item}</option>)}</select></label>
      <label className="ui-field"><span>Title</span><input value={title} onChange={(event) => setTitle(event.target.value)} placeholder="Discovery call · July 28" /></label>
      <label className="ui-field"><span>Notes</span><textarea rows={8} value={content} onChange={(event) => setContent(event.target.value)} placeholder="Paste the original notes. Metis saves them first; analysis is a separate step." /></label>
      <p className="modal-hint">Writing something down for yourself? Use <b>Add note</b> instead; it skips analysis entirely.</p>
    </Modal>
  );
}

export function WinDialog({ c, accountId, accountName, win }: { c: Customers; accountId: string; accountName: string; win?: CustomerWin }) {
  const [title, setTitle] = useState(win?.title ?? "");
  const [brief, setBrief] = useState(win?.brief ?? "");
  const [services, setServices] = useState<string[]>(win?.services ?? []);
  const [dacShape, setDacShape] = useState(win?.dac_shape ?? "");
  const [arr, setArr] = useState(win?.yearly_arr != null ? String(win.yearly_arr) : "");
  const [date, setDate] = useState(dateInputValue(win?.won_at));
  const close = () => c.setDialog(null);
  const save = () => void c.run("win", async () => {
    const value = parseUsd(arr);
    const payload = { title: title.trim(), brief: brief.trim(), services, dac_shape: dacShape.trim(), yearly_arr: value, won_at: date ? `${date}T00:00:00Z` : null };
    const saved = win ? await updateCustomerWin(win.id, payload) : await createCustomerWin(accountId, payload);
    c.setTab("wins");
    // A new win with no figure gets one estimated after the save has landed,
    // so a slow model call can never cost the win itself.
    if (!win && value === null) void estimateWinValuation(saved.id).then(() => c.refreshAll()).catch(() => undefined);
  }, win ? "Win updated." : "Win recorded.").then((ok) => ok && close());
  return (
    <Modal title={win ? "Edit win" : "Record a win"} eyebrow={accountName} onClose={close} footer={<><button type="button" className="ui-btn is-sm" onClick={close}>Cancel</button><button type="button" className="ui-btn is-primary is-sm" disabled={!title.trim() || c.busy === "win"} onClick={save}>{c.busy === "win" ? "Saving…" : win ? "Save win" : "Record win"}</button></>}>
      <label className="ui-field"><span>Title</span><input value={title} onChange={(event) => setTitle(event.target.value)} placeholder="Cohere Command A DAC deployed" /></label>
      <div className="ui-field"><span>Services</span><div className="modal-chips">{WIN_SERVICES.map((service) => <button key={service} type="button" className={`ui-chip${services.includes(service) ? " is-accent" : ""}`} aria-pressed={services.includes(service)} onClick={() => setServices((current) => current.includes(service) ? current.filter((item) => item !== service) : [...current, service])}>{service}</button>)}</div></div>
      <div className="modal-row">
        <label className="ui-field"><span>Yearly ARR (USD)</span><input inputMode="numeric" value={arr} onChange={(event) => setArr(event.target.value)} placeholder="110000" /></label>
        <label className="ui-field"><span>Win date</span><input type="date" value={date} onChange={(event) => setDate(event.target.value)} /></label>
      </div>
      <label className="ui-field"><span>DAC shape (marks it a DAC win)</span><input value={dacShape} onChange={(event) => setDacShape(event.target.value)} placeholder="Model Import DAC (2xA100-40G)" /></label>
      <label className="ui-field"><span>Brief</span><textarea rows={4} value={brief} onChange={(event) => setBrief(event.target.value)} placeholder="What was deployed, for which use case, and what it unlocks." /></label>
    </Modal>
  );
}

export function RateCardDialog({ c }: { c: Customers }) {
  const [edits, setEdits] = useState<Record<string, string>>({});
  const card = c.rateCard;
  const close = () => c.setDialog(null);
  if (!card) return null;
  const count = Object.keys(edits).length;
  // Editing a rate is what verifies it: the seeded figures are unverified because nobody has looked yet.
  const save = () => void c.run("rates", async () => {
    const updates = Object.entries(edits).map(([key, raw]) => ({ key, value: Number(raw.replace(/[^0-9.]/g, "")), verified: true })).filter((item) => Number.isFinite(item.value));
    c.setRateCard(await saveSkuRates(updates));
    setEdits({});
  }, "Rate card saved. Re-estimate a win to price it at the new rates.");
  return (
    <Modal title="Rate card" eyebrow={`${card.catalog_size} Oracle SKUs · ${card.rates.length} priced`} onClose={close} wide footer={<><small>{card.source_urls[0]}</small><button type="button" className="ui-btn is-sm" onClick={close}>Close</button><button type="button" className="ui-btn is-primary is-sm" disabled={!count || c.busy === "rates"} onClick={save}>{c.busy === "rates" ? "Saving…" : `Save ${count || ""} ${count === 1 ? "rate" : "rates"}`}</button></>}>
      <p className="modal-hint">Oracle publishes every SKU and its billing unit, but no prices. These rates are seeded from public list pricing and stay <b>unverified</b> until you save one.</p>
      <div className="rates">
        {card.rates.map((rate) => (
          <label key={rate.key} className={`rates-row${rate.verified ? " is-verified" : ""}`}>
            <span><b>{rate.label}</b><small>{rate.part_number || "no part number"} · per {rate.unit.toLowerCase().replace(/^(\d+,?\d*\s)?/, "")}</small></span>
            <input inputMode="decimal" value={edits[rate.key] ?? String(rate.value)} onChange={(event) => setEdits((current) => ({ ...current, [rate.key]: event.target.value }))} aria-label={`Rate for ${rate.label}`} />
            <i title={rate.verified ? "Verified by you" : "Seeded, not verified"}>{rate.verified ? "✓" : "?"}</i>
          </label>
        ))}
      </div>
    </Modal>
  );
}

export function ReviewDialog({ c, proposal }: { c: Customers; proposal: CustomerProposal }) {
  const [review, setReview] = useState<CustomerExtraction>(() => structuredClone(proposal.extraction));
  const close = () => c.setProposal(null);
  const edit = <K extends "facts" | "actions" | "people">(list: K, index: number, patch: Partial<CustomerExtraction[K][number]>) =>
    setReview((current) => ({ ...current, [list]: current[list].map((item, at) => (at === index ? { ...item, ...patch } : item)) }));
  const drop = (list: "facts" | "actions" | "people", index: number) =>
    setReview((current) => ({ ...current, [list]: current[list].filter((_, at) => at !== index) }));
  return (
    <Modal title="Review customer update" eyebrow="One review · one save" onClose={close} wide footer={<><small>Extracted with {proposal.model}</small><button type="button" className="ui-btn is-primary is-sm" disabled={c.busy === "review"} onClick={() => void c.saveProposal(review)}>{c.busy === "review" ? "Saving…" : "Save update"}</button></>}>
      <p className="modal-hint">Nothing below becomes account knowledge until you save it.</p>
      <label className="ui-field"><span>Summary</span><textarea rows={3} value={review.summary} onChange={(event) => setReview({ ...review, summary: event.target.value })} /></label>
      <section className="review-group"><h3>Facts <small>{review.facts.length}</small></h3>
        {review.facts.map((fact, index) => (
          <article key={index} className="review-item">
            <select value={fact.kind} onChange={(event) => edit("facts", index, { kind: event.target.value })} aria-label="Fact kind">{FACT_KINDS.map((kind) => <option key={kind} value={kind}>{kind.replace("_", " ")}</option>)}</select>
            <textarea value={fact.content} onChange={(event) => edit("facts", index, { content: event.target.value })} aria-label="Fact" />
            <small>“{fact.evidence.quote || "No evidence quote"}”</small>
            <button type="button" className="ui-btn is-quiet is-sm" onClick={() => drop("facts", index)}>Remove</button>
          </article>
        ))}
      </section>
      <section className="review-group"><h3>Actions <small>{review.actions.length}</small></h3>
        {review.actions.map((action, index) => (
          <article key={index} className="review-item">
            <textarea value={action.description} onChange={(event) => edit("actions", index, { description: event.target.value })} aria-label="Action" />
            <input value={action.owner} placeholder="Owner" onChange={(event) => edit("actions", index, { owner: event.target.value })} aria-label="Owner" />
            <small>“{action.evidence.quote || "No evidence quote"}”</small>
            <button type="button" className="ui-btn is-quiet is-sm" onClick={() => drop("actions", index)}>Remove</button>
          </article>
        ))}
      </section>
      <section className="review-group"><h3>People <small>{review.people.length}</small></h3>
        {review.people.map((person, index) => (
          <article key={index} className="review-item">
            <input value={person.name} onChange={(event) => edit("people", index, { name: event.target.value })} aria-label="Name" />
            <input value={person.role} placeholder="Role" onChange={(event) => edit("people", index, { role: event.target.value })} aria-label="Role" />
            <small>“{person.evidence.quote || "No evidence quote"}”</small>
            <button type="button" className="ui-btn is-quiet is-sm" onClick={() => drop("people", index)}>Remove</button>
          </article>
        ))}
      </section>
    </Modal>
  );
}
