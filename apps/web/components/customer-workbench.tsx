"use client";

import { useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";

import { CustomerSearch } from "@/components/customer-search";
import {
  acceptWinValuation,
  addCustomerPerson,
  analyzeCustomerSource,
  captureCustomerSource,
  createCustomer,
  createCustomerAction,
  createCustomerFact,
  createCustomerNote,
  createCustomerOutput,
  createCustomerWin,
  deleteCustomer,
  deleteCustomerAction,
  deleteCustomerFact,
  deleteCustomerNote,
  deleteCustomerPerson,
  deleteCustomerSource,
  deleteCustomerWin,
  dismissWinValuation,
  editCustomerAction,
  estimateWinValuation,
  getCustomer,
  getCustomerDashboard,
  getCustomerSettings,
  getLocalModelSession,
  getSkuRates,
  listCustomers,
  saveCustomerProposal,
  saveCustomerSettings,
  saveSkuRates,
  updateCustomer,
  updateCustomerAction,
  updateCustomerFact,
  updateCustomerNote,
  updateCustomerPerson,
  updateCustomerSource,
  updateCustomerWin,
} from "@/lib/api";
import type {
  CustomerAccount,
  CustomerAccountDetail,
  CustomerAction,
  CustomerDashboard,
  CustomerExtraction,
  CustomerFact,
  CustomerNote,
  CustomerOutput,
  CustomerPerson,
  CustomerProposal,
  CustomerSearchHit,
  CustomerSettings,
  CustomerSource,
  CustomerWin,
  LocalModelSession,
  SkuRateCard,
} from "@/lib/types";

type CustomerTab =
  | "overview" | "notes" | "timeline" | "actions" | "wins"
  | "facts" | "people" | "sources" | "outputs";

const TABS: Array<[CustomerTab, string]> = [
  ["overview", "Overview"],
  ["notes", "Notes"],
  ["timeline", "Timeline"],
  ["actions", "Actions"],
  ["wins", "Wins"],
  ["facts", "Facts"],
  ["people", "People"],
  ["sources", "Sources"],
  ["outputs", "Outputs"],
];

/** Which tab answers for a search hit. */
const TAB_FOR_HIT: Record<CustomerSearchHit["kind"], CustomerTab> = {
  account: "overview",
  note: "notes",
  fact: "facts",
  action: "actions",
  win: "wins",
  source: "sources",
};

const WIN_SERVICES = [
  "Generative AI Services",
  "Generative AI Agents",
  "DAC",
  "Model-Import",
  "On-demand",
  "ODA",
];

const FACT_KINDS = [
  "requirement", "decision", "use_case", "risk", "question",
  "constraint", "model", "dac_note", "other",
];

// The kinds an engineer reads before a design conversation. Kept as a preset
// filter rather than a separate tab, because a fact the model labelled
// "decision" is no less technical for it.
const TECHNICAL_KINDS = ["requirement", "constraint", "model", "dac_note", "risk"];

const FACT_FILTERS: Array<[string, string]> = [
  ["all", "All"],
  ["technical", "Technical"],
  ["decision", "Decisions"],
  ["risk", "Risks"],
  ["question", "Questions"],
];

const SOURCE_KINDS = ["note", "meeting", "chat", "notion", "attachment"] as const;

type AccountFilter = "all" | "wins" | "open" | "overdue" | "waiting";

const ACCOUNT_FILTERS: Array<[AccountFilter, string]> = [
  ["all", "All"],
  ["wins", "🏆 Wins"],
  ["open", "Open actions"],
  ["overdue", "Needs a nudge"],
  ["waiting", "Waiting"],
];

function matchesFilter(account: CustomerAccount, filter: AccountFilter): boolean {
  if (filter === "wins") return account.wins > 0;
  if (filter === "open") return account.open_actions > 0;
  if (filter === "waiting") return account.pending_notes > 0;
  // "Needs a nudge": an account with work outstanding that nobody has touched
  // in a month. The account list has no per-account due date, so staleness is
  // the honest proxy for it here.
  if (filter === "overdue") {
    if (!account.open_actions) return false;
    const last = account.last_interaction_at || account.updated_at;
    return Date.now() - new Date(last).getTime() > 30 * 24 * 60 * 60 * 1000;
  }
  return true;
}

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

/** A stored instant as the `YYYY-MM-DD` a date input wants, read in UTC for the
 *  same reason `winDay` is: the day chosen must be the day shown back. */
function dateInputValue(value?: string | null): string {
  if (!value) return "";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? "" : parsed.toISOString().slice(0, 10);
}

function instantFromDateInput(value: string): string | null {
  return value ? `${value}T00:00:00Z` : null;
}

function arr(value?: number | null): string {
  if (value === null || value === undefined) return "—";
  return new Intl.NumberFormat("en-US", {
    style: "currency", currency: "USD", maximumFractionDigits: 0,
  }).format(value);
}

// The hero figure has to stay legible at a glance, so past a million it reads
// as "$1.46M" rather than nine digits the eye has to count.
function compactArr(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency", currency: "USD",
    notation: value >= 1_000_000 ? "compact" : "standard",
    maximumFractionDigits: value >= 1_000_000 ? 2 : 0,
  }).format(value);
}

function isOverdue(action: CustomerAction): boolean {
  return action.status === "open" && Boolean(action.due_at) && new Date(action.due_at!).getTime() < Date.now();
}

type NoteDraft = { title: string; body: string; pinned: boolean };
type FactDraft = { kind: string; content: string; status: CustomerFact["status"] };
type ActionDraft = { description: string; owner: string; due: string; status: CustomerAction["status"] };
type PersonDraft = { name: string; role: string; organization: string };
type SourceDraft = { title: string; content: string; source_kind: string };
type ProfileDraft = { name: string; aliases: string; industry: string; region: string; status: CustomerAccount["status"] };

const EMPTY_NOTE: NoteDraft = { title: "", body: "", pinned: false };
const EMPTY_FACT: FactDraft = { kind: "requirement", content: "", status: "active" };
const EMPTY_ACTION: ActionDraft = { description: "", owner: "", due: "", status: "open" };
const EMPTY_PERSON: PersonDraft = { name: "", role: "", organization: "" };

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
  const [editingWinId, setEditingWinId] = useState<string | null>(null);
  const [winTitle, setWinTitle] = useState("");
  const [winBrief, setWinBrief] = useState("");
  const [winServices, setWinServices] = useState<string[]>([]);
  const [winDacShape, setWinDacShape] = useState("");
  const [winArr, setWinArr] = useState("");
  const [winDate, setWinDate] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<AccountFilter>("all");
  const [rateCard, setRateCard] = useState<SkuRateCard | null>(null);
  const [ratesOpen, setRatesOpen] = useState(false);
  const [rateEdits, setRateEdits] = useState<Record<string, string>>({});
  const [estimating, setEstimating] = useState<string | null>(null);
  const [proposal, setProposal] = useState<CustomerProposal | null>(null);
  const [review, setReview] = useState<CustomerExtraction | null>(null);
  const [output, setOutput] = useState<CustomerOutput | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Cross-account surfaces
  const [searchOpen, setSearchOpen] = useState(false);
  const [attentionOpen, setAttentionOpen] = useState(false);

  // Hand-edit surfaces. Each holds the id being edited plus its working copy,
  // so an in-progress edit is never confused with the saved record.
  const [profileDraft, setProfileDraft] = useState<ProfileDraft | null>(null);
  const [newNote, setNewNote] = useState<NoteDraft>(EMPTY_NOTE);
  const [noteComposerOpen, setNoteComposerOpen] = useState(false);
  const [editingNote, setEditingNote] = useState<{ id: string; draft: NoteDraft } | null>(null);
  const [newFact, setNewFact] = useState<FactDraft | null>(null);
  const [editingFact, setEditingFact] = useState<{ id: string; draft: FactDraft } | null>(null);
  const [factFilter, setFactFilter] = useState("all");
  const [newAction, setNewAction] = useState<ActionDraft | null>(null);
  const [editingAction, setEditingAction] = useState<{ id: string; draft: ActionDraft } | null>(null);
  const [newPerson, setNewPerson] = useState<PersonDraft | null>(null);
  const [editingPerson, setEditingPerson] = useState<{ id: string; draft: PersonDraft } | null>(null);
  const [editingSource, setEditingSource] = useState<{ id: string; draft: SourceDraft } | null>(null);

  const refreshIndex = useCallback(async () => {
    const [nextAccounts, nextDashboard, nextSession, nextSettings, nextRates] = await Promise.all([
      listCustomers(),
      getCustomerDashboard(),
      getLocalModelSession().catch(() => null),
      getCustomerSettings(),
      getSkuRates().catch(() => null),
    ]);
    setAccounts(nextAccounts);
    setDashboard(nextDashboard);
    if (nextSession) setSession(nextSession);
    setSettings(nextSettings);
    if (nextRates) setRateCard(nextRates);
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

  // ⌘⇧K, one shift away from the shell's ⌘K conversation search: same reflex,
  // different haystack. The shell explicitly leaves the shifted chord alone.
  useEffect(() => {
    const listener = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.shiftKey && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setSearchOpen(true);
      }
    };
    window.addEventListener("keydown", listener);
    return () => window.removeEventListener("keydown", listener);
  }, []);

  /** Every in-flight edit belongs to the account that was open; switching
   *  accounts must not carry a half-written note onto the next one. */
  function selectAccount(id: string | null, nextTab: CustomerTab = "overview") {
    setSelectedId(id);
    setTab(nextTab);
    setProfileDraft(null);
    setNoteComposerOpen(false);
    setNewNote(EMPTY_NOTE);
    setEditingNote(null);
    setNewFact(null);
    setEditingFact(null);
    setNewAction(null);
    setEditingAction(null);
    setNewPerson(null);
    setEditingPerson(null);
    setEditingSource(null);
    setOutput(null);
  }

  const visibleAccounts = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return accounts.filter((account) => {
      if (!matchesFilter(account, filter)) return false;
      if (!needle) return true;
      // Aliases are searched too — accounts arrive from Notion under names the
      // user does not necessarily type ("OHI UNHCR" vs "UNHCR").
      return [account.name, account.industry, account.region, ...account.aliases]
        .some((field) => field && field.toLowerCase().includes(needle));
    });
  }, [accounts, filter, query]);

  const visibleFacts = useMemo(() => {
    const facts = detail?.facts ?? [];
    if (factFilter === "all") return facts;
    if (factFilter === "technical") return facts.filter((fact) => TECHNICAL_KINDS.includes(fact.kind));
    return facts.filter((fact) => fact.kind === factFilter);
  }, [detail, factFilter]);

  const estimatedPending = useMemo(
    () => (dashboard?.recent_wins ?? []).filter(
      (win) => win.yearly_arr === null && win.valuation?.status === "proposed",
    ).length,
    [dashboard],
  );

  const attention = dashboard?.priority_actions ?? [];

  function fail(problem: unknown, fallback: string) {
    setError(problem instanceof Error ? problem.message : fallback);
  }

  /** Re-read the account and the index together: almost every edit changes a
   *  count the summary band or the account list is showing. */
  async function refreshAll() {
    if (selectedId) await Promise.all([refreshDetail(selectedId), refreshIndex()]);
    else await refreshIndex();
  }

  async function addAccount() {
    if (!newAccountName.trim() || creating) return;
    setCreating(true);
    setError(null);
    try {
      const account = await createCustomer({ name: newAccountName.trim() });
      setNewAccountName("");
      selectAccount(account.id);
      await refreshIndex();
    } catch (createError) {
      fail(createError, "The account could not be added.");
    } finally {
      setCreating(false);
    }
  }

  async function saveProfile() {
    if (!detail || !profileDraft || !profileDraft.name.trim() || busy) return;
    setBusy("profile");
    setError(null);
    try {
      await updateCustomer(detail.account.id, {
        name: profileDraft.name.trim(),
        aliases: profileDraft.aliases.split(",").map((item) => item.trim()).filter(Boolean),
        industry: profileDraft.industry.trim(),
        region: profileDraft.region.trim(),
        status: profileDraft.status,
      });
      setProfileDraft(null);
      setNotice("Account details saved.");
      await refreshAll();
    } catch (saveError) {
      fail(saveError, "The account details could not be saved.");
    } finally {
      setBusy(null);
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
      selectAccount(remaining[0]?.id ?? null);
      setDetail(null);
      setNotice(`Deleted ${detail.account.name}.`);
      await refreshIndex();
    } catch (deleteError) {
      fail(deleteError, "The account could not be deleted.");
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
      await refreshAll();
    } catch (captureError) {
      fail(captureError, "The note could not be saved.");
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
      fail(analysisError, "The note could not be analyzed.");
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
      await refreshAll();
    } catch (saveError) {
      fail(saveError, "The reviewed update could not be saved.");
    } finally {
      setBusy(null);
    }
  }

  // ── Direct notes ─────────────────────────────────────────────────────────

  async function addNote() {
    if (!selectedId || !newNote.body.trim() || busy) return;
    setBusy("new-note");
    setError(null);
    try {
      await createCustomerNote(selectedId, {
        title: newNote.title.trim(),
        body: newNote.body.trim(),
        pinned: newNote.pinned,
      });
      setNewNote(EMPTY_NOTE);
      setNoteComposerOpen(false);
      setTab("notes");
      setNotice(newNote.pinned
        ? "Note saved and pinned — it now travels with this account into scoped conversations."
        : "Note saved to this account.");
      await refreshAll();
    } catch (noteError) {
      fail(noteError, "The note could not be saved.");
    } finally {
      setBusy(null);
    }
  }

  async function saveNote() {
    if (!editingNote || !editingNote.draft.body.trim() || busy) return;
    setBusy(editingNote.id);
    setError(null);
    try {
      await updateCustomerNote(editingNote.id, {
        title: editingNote.draft.title.trim(),
        body: editingNote.draft.body.trim(),
        pinned: editingNote.draft.pinned,
      });
      setEditingNote(null);
      await refreshAll();
    } catch (noteError) {
      fail(noteError, "The note could not be saved.");
    } finally {
      setBusy(null);
    }
  }

  async function togglePin(note: CustomerNote) {
    if (busy) return;
    setBusy(note.id);
    setError(null);
    try {
      await updateCustomerNote(note.id, {
        title: note.title, body: note.body, pinned: !note.pinned,
      });
      await refreshAll();
    } catch (noteError) {
      fail(noteError, "The note could not be updated.");
    } finally {
      setBusy(null);
    }
  }

  async function removeNote(noteId: string) {
    if (busy || !window.confirm("Delete this note?")) return;
    setBusy(noteId);
    setError(null);
    try {
      await deleteCustomerNote(noteId);
      if (editingNote?.id === noteId) setEditingNote(null);
      await refreshAll();
    } catch (noteError) {
      fail(noteError, "The note could not be deleted.");
    } finally {
      setBusy(null);
    }
  }

  // ── Facts, actions, people, sources ──────────────────────────────────────

  async function addFact() {
    if (!selectedId || !newFact?.content.trim() || busy) return;
    setBusy("new-fact");
    setError(null);
    try {
      await createCustomerFact(selectedId, { kind: newFact.kind, content: newFact.content.trim() });
      setNewFact(null);
      await refreshAll();
    } catch (factError) {
      fail(factError, "The fact could not be added.");
    } finally {
      setBusy(null);
    }
  }

  async function saveFact() {
    if (!editingFact || !editingFact.draft.content.trim() || busy) return;
    setBusy(editingFact.id);
    setError(null);
    try {
      await updateCustomerFact(editingFact.id, {
        kind: editingFact.draft.kind,
        content: editingFact.draft.content.trim(),
        status: editingFact.draft.status,
      });
      setEditingFact(null);
      await refreshAll();
    } catch (factError) {
      fail(factError, "The fact could not be saved.");
    } finally {
      setBusy(null);
    }
  }

  async function removeFact(factId: string) {
    if (busy || !window.confirm("Delete this fact?")) return;
    setBusy(factId);
    setError(null);
    try {
      await deleteCustomerFact(factId);
      if (editingFact?.id === factId) setEditingFact(null);
      await refreshAll();
    } catch (factError) {
      fail(factError, "The fact could not be deleted.");
    } finally {
      setBusy(null);
    }
  }

  async function addAction() {
    if (!selectedId || !newAction?.description.trim() || busy) return;
    setBusy("new-action");
    setError(null);
    try {
      await createCustomerAction(selectedId, {
        description: newAction.description.trim(),
        owner: newAction.owner.trim(),
        due_at: instantFromDateInput(newAction.due),
      });
      setNewAction(null);
      await refreshAll();
    } catch (actionError) {
      fail(actionError, "The action could not be added.");
    } finally {
      setBusy(null);
    }
  }

  async function saveAction() {
    if (!editingAction || !editingAction.draft.description.trim() || busy) return;
    setBusy(editingAction.id);
    setError(null);
    try {
      await editCustomerAction(editingAction.id, {
        description: editingAction.draft.description.trim(),
        owner: editingAction.draft.owner.trim(),
        due_at: instantFromDateInput(editingAction.draft.due),
        status: editingAction.draft.status,
      });
      setEditingAction(null);
      await refreshAll();
    } catch (actionError) {
      fail(actionError, "The action could not be saved.");
    } finally {
      setBusy(null);
    }
  }

  async function removeAction(actionId: string) {
    if (busy || !window.confirm("Delete this action?")) return;
    setBusy(actionId);
    setError(null);
    try {
      await deleteCustomerAction(actionId);
      if (editingAction?.id === actionId) setEditingAction(null);
      await refreshAll();
    } catch (actionError) {
      fail(actionError, "The action could not be deleted.");
    } finally {
      setBusy(null);
    }
  }

  async function toggleAction(actionId: string, current: CustomerAction["status"]) {
    setBusy(actionId);
    try {
      await updateCustomerAction(actionId, current === "open" ? "done" : "open");
      await refreshAll();
    } catch (actionError) {
      fail(actionError, "The action could not be updated.");
    } finally {
      setBusy(null);
    }
  }

  async function addPerson() {
    if (!selectedId || !newPerson?.name.trim() || busy) return;
    setBusy("new-person");
    setError(null);
    try {
      await addCustomerPerson(selectedId, {
        name: newPerson.name.trim(),
        role: newPerson.role.trim(),
        organization: newPerson.organization.trim(),
      });
      setNewPerson(null);
      await refreshAll();
    } catch (personError) {
      fail(personError, "The contact could not be added.");
    } finally {
      setBusy(null);
    }
  }

  async function savePerson() {
    if (!editingPerson || !editingPerson.draft.name.trim() || busy) return;
    setBusy(editingPerson.id);
    setError(null);
    try {
      await updateCustomerPerson(editingPerson.id, {
        name: editingPerson.draft.name.trim(),
        role: editingPerson.draft.role.trim(),
        organization: editingPerson.draft.organization.trim(),
      });
      setEditingPerson(null);
      await refreshAll();
    } catch (personError) {
      fail(personError, "The contact could not be saved.");
    } finally {
      setBusy(null);
    }
  }

  async function removePerson(personId: string) {
    if (busy || !window.confirm("Remove this contact?")) return;
    setBusy(personId);
    setError(null);
    try {
      await deleteCustomerPerson(personId);
      if (editingPerson?.id === personId) setEditingPerson(null);
      await refreshAll();
    } catch (personError) {
      fail(personError, "The contact could not be removed.");
    } finally {
      setBusy(null);
    }
  }

  async function saveSource() {
    if (!editingSource || !editingSource.draft.title.trim() || !editingSource.draft.content.trim() || busy) return;
    setBusy(editingSource.id);
    setError(null);
    try {
      await updateCustomerSource(editingSource.id, {
        title: editingSource.draft.title.trim(),
        content: editingSource.draft.content.trim(),
        source_kind: editingSource.draft.source_kind as CustomerSource["source_kind"],
      });
      setEditingSource(null);
      setNotice("Note corrected. Facts and actions already saved from it are unchanged.");
      await refreshAll();
    } catch (sourceError) {
      fail(sourceError, "The note could not be saved.");
    } finally {
      setBusy(null);
    }
  }

  async function removeSource(sourceId: string) {
    // Deleting the source cascades to its timeline entry, but the facts and
    // actions saved from it were reviewed on their own and survive — worth
    // saying, because "delete the note" reads like "undo everything it caused".
    if (busy || !window.confirm("Delete this captured note? It leaves the timeline with it; facts and actions already saved from it remain.")) return;
    setBusy(sourceId);
    setError(null);
    try {
      await deleteCustomerSource(sourceId);
      if (editingSource?.id === sourceId) setEditingSource(null);
      await refreshAll();
    } catch (sourceError) {
      fail(sourceError, "The note could not be deleted.");
    } finally {
      setBusy(null);
    }
  }

  // ── Wins ─────────────────────────────────────────────────────────────────

  function openWinForm(win?: CustomerWin) {
    setEditingWinId(win?.id ?? null);
    setWinTitle(win?.title ?? "");
    setWinBrief(win?.brief ?? "");
    setWinServices(win?.services ?? []);
    setWinDacShape(win?.dac_shape ?? "");
    setWinArr(win?.yearly_arr != null ? String(win.yearly_arr) : "");
    setWinDate(dateInputValue(win?.won_at));
    setWinOpen(true);
  }

  async function submitWin() {
    if (!selectedId || !winTitle.trim() || busy) return;
    setBusy("win");
    setError(null);
    try {
      const parsedArr = winArr.trim() ? Number(winArr.replace(/[^0-9.]/g, "")) : null;
      const value = parsedArr !== null && Number.isFinite(parsedArr) ? parsedArr : null;
      const payload = {
        title: winTitle.trim(),
        brief: winBrief.trim(),
        services: winServices,
        dac_shape: winDacShape.trim(),
        yearly_arr: value,
        won_at: winDate ? `${winDate}T00:00:00Z` : null,
      };
      const win = editingWinId
        ? await updateCustomerWin(editingWinId, payload)
        : await createCustomerWin(selectedId, payload);
      const wasEditing = Boolean(editingWinId);
      setWinOpen(false);
      setEditingWinId(null);
      setWinTitle("");
      setWinBrief("");
      setWinServices([]);
      setWinDacShape("");
      setWinArr("");
      setWinDate("");
      setTab("wins");
      setNotice(wasEditing
        ? "Win updated."
        : value === null
          ? "Win recorded with no value. Estimating one from this account's notes…"
          : "Win recorded. The tracker above reflects it immediately.");
      await refreshAll();
      // A new win without a figure gets one estimated automatically — after the
      // save has already landed, so a slow or failed model call can never cost
      // the user the win they just recorded.
      if (!wasEditing && value === null) void estimate(win.id, { silent: true });
    } catch (winError) {
      fail(winError, "The win could not be saved.");
    } finally {
      setBusy(null);
    }
  }

  async function estimate(winId: string, options?: { silent?: boolean }) {
    setEstimating(winId);
    if (!options?.silent) setError(null);
    try {
      const valuation = await estimateWinValuation(winId);
      await refreshAll();
      setNotice(
        valuation.lines.length
          ? `Estimated ${arr(valuation.estimated_yearly_arr)} from the account notes. Review it before it counts.`
          : "The notes did not describe anything billable, so no estimate was produced.",
      );
    } catch (estimateError) {
      fail(estimateError, "The estimate could not be produced.");
    } finally {
      setEstimating(null);
    }
  }

  async function acceptEstimate(win: CustomerWin, corrected?: number | null) {
    setBusy(win.id);
    setError(null);
    try {
      await acceptWinValuation(win.id, corrected ?? null);
      setNotice(`${win.title} now counts toward ARR won.`);
      await refreshAll();
    } catch (acceptError) {
      fail(acceptError, "The estimate could not be accepted.");
    } finally {
      setBusy(null);
    }
  }

  async function dismissEstimate(winId: string) {
    setBusy(winId);
    try {
      await dismissWinValuation(winId);
      await refreshAll();
    } catch (dismissError) {
      fail(dismissError, "The estimate could not be dismissed.");
    } finally {
      setBusy(null);
    }
  }

  async function saveRates() {
    const updates = Object.entries(rateEdits)
      .map(([key, raw]) => ({ key, value: Number(raw.replace(/[^0-9.]/g, "")) }))
      .filter((item) => Number.isFinite(item.value));
    if (!updates.length) return;
    setBusy("rates");
    try {
      // Editing a rate is what verifies it: the seeded figures are unverified
      // precisely because nobody has looked at them yet.
      setRateCard(await saveSkuRates(updates.map((item) => ({ ...item, verified: true }))));
      setRateEdits({});
      setNotice("Rate card saved. Re-estimate a win to price it at the new rates.");
    } catch (rateError) {
      fail(rateError, "The rate card could not be saved.");
    } finally {
      setBusy(null);
    }
  }

  async function removeWin(winId: string) {
    if (busy || !window.confirm("Remove this win from the tracker?")) return;
    setBusy(winId);
    setError(null);
    try {
      await deleteCustomerWin(winId);
      await refreshAll();
    } catch (winError) {
      fail(winError, "The win could not be removed.");
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
      fail(outputError, "The Markdown update could not be created.");
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
      fail(settingsError, "The tracker link could not be saved.");
    } finally {
      setBusy(null);
    }
  }

  /** The estimate panel on a win card.
   *
   *  A win with a confirmed figure shows nothing: the number is settled and the
   *  card should not invite second-guessing it. Everything else is a prompt to
   *  either produce an estimate or decide about the one that exists.
   */
  function renderValuation(win: CustomerWin) {
    if (win.yearly_arr !== null) return null;
    const valuation = win.valuation;
    const running = estimating === win.id;

    if (running) {
      return <div className="customerWinEstimate isRunning"><i className="livePulse" />Reading this account&rsquo;s notes…</div>;
    }
    if (!valuation || valuation.status === "dismissed") {
      return (
        <div className="customerWinEstimate isEmpty">
          <span>{valuation ? "Estimate dismissed." : "No value recorded."}</span>
          <button type="button" onClick={() => void estimate(win.id)}>
            {valuation ? "Estimate again" : "Estimate from notes"}
          </button>
        </div>
      );
    }
    if (!valuation.lines.length) {
      return (
        <div className="customerWinEstimate isEmpty">
          <span>The notes don&rsquo;t describe anything billable{valuation.unpriced.length ? ` (${valuation.unpriced.join(", ")} has no rate)` : ""}.</span>
          <button type="button" onClick={() => void estimate(win.id)}>Try again</button>
        </div>
      );
    }

    return (
      <div className={`customerWinEstimate conf-${valuation.confidence}`}>
        <header>
          <div>
            <span className="eyebrow">Estimated · not counted yet</span>
            <strong>{arr(valuation.estimated_yearly_arr)}<em>/year</em></strong>
          </div>
          <span className="customerEstimateConfidence">{valuation.confidence} confidence</span>
        </header>
        {valuation.explanation ? <p>{valuation.explanation}</p> : null}
        <ul>
          {valuation.lines.map((line) => (
            <li key={`${line.sku}-${line.name}`}>
              <span>
                <b>{line.name}</b>
                <small>{line.basis}{line.why ? ` — ${line.why}` : ""}</small>
              </span>
              <i>{arr(line.yearly_amount)}</i>
            </li>
          ))}
        </ul>
        {valuation.unpriced.length ? (
          <small className="customerEstimateWarn">No rate for {valuation.unpriced.join(", ")} — excluded from the total.</small>
        ) : null}
        {!valuation.rates_verified ? (
          <small className="customerEstimateWarn">
            Priced with unverified list rates.
            <button type="button" onClick={() => setRatesOpen(true)}>Review the rate card</button>
          </small>
        ) : null}
        <footer>
          <button className="primaryButton" type="button" disabled={busy === win.id} onClick={() => void acceptEstimate(win)}>
            {busy === win.id ? "Saving…" : "Accept as ARR"}
          </button>
          <button className="secondaryButton" type="button" disabled={busy === win.id} onClick={() => {
            const answer = window.prompt(`Yearly ARR for “${win.title}” (USD)`, String(Math.round(valuation.estimated_yearly_arr ?? 0)));
            if (answer === null) return;
            const corrected = Number(answer.replace(/[^0-9.]/g, ""));
            if (Number.isFinite(corrected)) void acceptEstimate(win, corrected);
          }}>Edit &amp; accept</button>
          <button className="ghostButton" type="button" disabled={busy === win.id} onClick={() => void estimate(win.id)}>Re-run</button>
          <button className="ghostButton" type="button" disabled={busy === win.id} onClick={() => void dismissEstimate(win.id)}>Dismiss</button>
          {valuation.model_used ? <small>via {valuation.model_used}</small> : null}
        </footer>
      </div>
    );
  }

  function copyAndOpen() {
    if (!output) return;
    void navigator.clipboard.writeText(output.content);
    const target = output.tracker_url || settings.tracker_url;
    if (target) window.open(target, "_blank", "noopener,noreferrer");
    setNotice(target ? "Markdown copied. Paste it into the tracker tab." : "Markdown copied. Add the tracker URL to open it at the same time.");
  }

  function renderPerson(person: CustomerPerson) {
    if (editingPerson?.id === person.id) {
      return (
        <article key={person.id} className="isEditing">
          <div className="customerInlineForm">
            <input
              value={editingPerson.draft.name}
              onChange={(event) => setEditingPerson({ ...editingPerson, draft: { ...editingPerson.draft, name: event.target.value } })}
              placeholder="Name"
              aria-label="Contact name"
            />
            <input
              value={editingPerson.draft.role}
              onChange={(event) => setEditingPerson({ ...editingPerson, draft: { ...editingPerson.draft, role: event.target.value } })}
              placeholder="Role"
              aria-label="Contact role"
            />
            <input
              value={editingPerson.draft.organization}
              onChange={(event) => setEditingPerson({ ...editingPerson, draft: { ...editingPerson.draft, organization: event.target.value } })}
              placeholder="Organization"
              aria-label="Contact organization"
            />
            <div className="customerInlineActions">
              <button className="ghostButton" type="button" onClick={() => setEditingPerson(null)}>Cancel</button>
              <button className="primaryButton" type="button" disabled={!editingPerson.draft.name.trim() || busy === person.id} onClick={() => void savePerson()}>
                {busy === person.id ? "Saving…" : "Save"}
              </button>
            </div>
          </div>
        </article>
      );
    }
    return (
      <article key={person.id}>
        <i>{person.name.slice(0, 1).toUpperCase()}</i>
        <div>
          <strong>{person.name}</strong>
          <span>{person.role || "Role not captured"}</span>
          <small>{person.organization}</small>
        </div>
        <div className="customerRowActions">
          <button type="button" onClick={() => setEditingPerson({ id: person.id, draft: { name: person.name, role: person.role, organization: person.organization } })}>Edit</button>
          <button type="button" className="isDanger" disabled={busy === person.id} onClick={() => void removePerson(person.id)}>Remove</button>
        </div>
      </article>
    );
  }

  return (
    <div className="customerWorkbench">
      <header className="customerWorkbenchHeader">
        <div>
          <span className="eyebrow">Customer intelligence</span>
          <h1>Customer Workbench</h1>
          <p>Turn notes into reviewed, account-scoped facts, actions, and ready-to-paste updates.</p>
        </div>
        <div className="customerWorkbenchActions">
          <button className="secondaryButton customerSearchTrigger" type="button" onClick={() => setSearchOpen(true)}>
            <span aria-hidden="true">⌕</span> Search everything <kbd>⌘⇧K</kbd>
          </button>
          <button className="secondaryButton" type="button" onClick={() => { setTab("notes"); setNoteComposerOpen(true); }} disabled={!selectedId}>✎ Add note</button>
          <button className="primaryButton" type="button" onClick={() => setCaptureOpen(true)} disabled={!selectedId}>＋ Capture note</button>
        </div>
      </header>

      {/* One band, one hierarchy. ARR won is the headline the page exists to
          report; wins qualify it; the operational counts sit beside it as a
          rail rather than competing for the same visual weight. */}
      <section className="customerSummary">
        <div className="customerSummaryHero">
          <span className="eyebrow">Yearly ARR won</span>
          <strong>{compactArr(dashboard?.total_yearly_arr ?? 0)}</strong>
          <p>
            <b>{dashboard?.total_wins ?? 0}</b> {dashboard?.total_wins === 1 ? "win" : "wins"}
            {dashboard?.dac_wins ? <> · <b>{dashboard.dac_wins}</b> DAC</> : null}
            {estimatedPending > 0 ? (
              <> · <em>{estimatedPending} awaiting review</em></>
            ) : null}
          </p>
          <div className="customerSummaryChips">
            {Object.entries(dashboard?.wins_by_service ?? {}).sort((a, b) => b[1] - a[1]).slice(0, 6).map(([service, count]) => (
              <b key={service}>{service}<i>{count}</i></b>
            ))}
            {!Object.keys(dashboard?.wins_by_service ?? {}).length ? <em>No wins recorded yet.</em> : null}
          </div>
        </div>

        {/* The counts are the entry points to the work, so each one is the
            control that opens what it is counting. */}
        <div className="customerSummaryRail">
          <article><strong>{dashboard?.active_accounts ?? 0}</strong><span>Accounts</span></article>
          <button
            type="button"
            className={`customerRailButton ${attentionOpen ? "isOpen" : ""}`}
            aria-expanded={attentionOpen}
            aria-label={`${dashboard?.open_actions ?? 0} open actions — show the queue`}
            onClick={() => setAttentionOpen((value) => !value)}
          >
            <strong>{dashboard?.open_actions ?? 0}</strong><span>Open actions</span>
          </button>
          <button
            type="button"
            className={`customerRailButton ${dashboard?.overdue_actions ? "tone-alert" : ""} ${attentionOpen ? "isOpen" : ""}`}
            aria-expanded={attentionOpen}
            aria-label={`${dashboard?.overdue_actions ?? 0} overdue actions — show the queue`}
            onClick={() => setAttentionOpen((value) => !value)}
          >
            <strong>{dashboard?.overdue_actions ?? 0}</strong><span>Overdue</span>
          </button>
          <button
            type="button"
            className={`customerRailButton ${dashboard?.waiting_notes ? "tone-waiting" : ""} ${filter === "waiting" ? "isOpen" : ""}`}
            aria-pressed={filter === "waiting"}
            aria-label={`${dashboard?.waiting_notes ?? 0} notes waiting for analysis — filter the account list`}
            onClick={() => setFilter((current) => current === "waiting" ? "all" : "waiting")}
          >
            <strong>{dashboard?.waiting_notes ?? 0}</strong><span>Waiting</span>
          </button>
          <button className="secondaryButton" type="button" onClick={() => openWinForm()} disabled={!selectedId}>🏆 Record win</button>
          <button
            className="ghostButton"
            type="button"
            disabled={!rateCard}
            title={rateCard ? "Review the Oracle rates estimates are priced with" : "Rates are unavailable while the API is unreachable"}
            onClick={() => setRatesOpen(true)}
          >
            Rate card
          </button>
        </div>

        {dashboard?.recent_wins.length ? (
          <ul className="customerWinsRecent">
            {dashboard.recent_wins.map((win) => (
              <li key={win.id}>
                <time>{winDay(win.won_at || win.created_at)}</time>
                <button type="button" onClick={() => selectAccount(win.account_id, "wins")}>{win.account_name}</button>
                <span>{win.title}</span>
                {win.yearly_arr !== null ? (
                  <b>{arr(win.yearly_arr)}</b>
                ) : win.valuation?.status === "proposed" && win.valuation.estimated_yearly_arr !== null ? (
                  <b className="isEstimate" title="Estimated from this account's notes — not yet reviewed">
                    ~{arr(win.valuation.estimated_yearly_arr)}
                  </b>
                ) : <b className="isBlank">—</b>}
              </li>
            ))}
          </ul>
        ) : null}
      </section>

      {/* The queue the dashboard was already computing and nobody could see:
          open work across every account, overdue first. */}
      {attentionOpen ? (
        <section className="customerAttention" aria-label="Open actions across accounts">
          <header>
            <div>
              <span className="eyebrow">Needs you</span>
              <strong>{attention.length ? `${attention.length} open action${attention.length === 1 ? "" : "s"} across your accounts` : "Nothing open across your accounts"}</strong>
            </div>
            <button className="ghostButton" type="button" onClick={() => setAttentionOpen(false)}>Hide</button>
          </header>
          {attention.map((action) => (
            <article key={action.id} className={isOverdue(action) ? "isOverdue" : ""}>
              <button type="button" className="customerAttentionAccount" onClick={() => { setAttentionOpen(false); selectAccount(action.account_id, "actions"); }}>
                {action.account_name || "Account"}
              </button>
              <div>
                <strong>{action.description}</strong>
                <small>
                  {action.owner || "Unassigned"}
                  {action.due_at ? ` · ${isOverdue(action) ? "overdue since" : "due"} ${when(action.due_at)}` : " · no date"}
                </small>
              </div>
              <button className="secondaryButton" type="button" disabled={busy === action.id} onClick={() => void toggleAction(action.id, action.status)}>
                {busy === action.id ? "…" : "Done"}
              </button>
            </article>
          ))}
          {!attention.length ? <p className="customerEmpty">Every captured action is closed.</p> : null}
        </section>
      ) : null}

      {error ? <div className="customerNotice error" role="alert">{error}<button type="button" onClick={() => setError(null)}>×</button></div> : null}
      {notice ? <div className="customerNotice" role="status">{notice}<button type="button" onClick={() => setNotice(null)}>×</button></div> : null}

      <div className="customerWorkbenchBody">
        <aside className="customerAccounts">
          <div className="customerAccountsTop">
            <header>
              <strong>Accounts</strong>
              <span>{visibleAccounts.length === accounts.length ? accounts.length : `${visibleAccounts.length} of ${accounts.length}`}</span>
            </header>
            <div className="customerAccountSearch">
              <input
                type="search"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Search accounts"
                aria-label="Search accounts"
              />
              {query ? <button type="button" onClick={() => setQuery("")} aria-label="Clear search">×</button> : null}
            </div>
            <div className="customerAccountFilters" role="group" aria-label="Filter accounts">
              {ACCOUNT_FILTERS.map(([value, label]) => (
                <button
                  key={value}
                  type="button"
                  className={filter === value ? "active" : ""}
                  aria-pressed={filter === value}
                  onClick={() => setFilter(value)}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          <nav>
            {visibleAccounts.map((account) => (
              <button key={account.id} className={selectedId === account.id ? "active" : ""} type="button" onClick={() => selectAccount(account.id)}>
                <span><strong>{account.name}</strong><small>{account.industry || account.region || "Customer account"}</small></span>
                {account.wins > 0 ? <i className="customerWinBadge" title={`${account.wins} win${account.wins === 1 ? "" : "s"}`}>🏆{account.wins}</i> : null}
                <b>{account.open_actions}</b>
              </button>
            ))}
            {!accounts.length ? <p>No accounts yet. Add one below to start.</p> : null}
            {accounts.length && !visibleAccounts.length ? (
              <p>
                Nothing matches {query.trim() ? <>“{query.trim()}”</> : "that filter"}.
                <button type="button" className="customerAccountsReset" onClick={() => { setQuery(""); setFilter("all"); }}>Clear</button>
              </p>
            ) : null}
          </nav>

          <div className="customerAddAccount">
            <input value={newAccountName} onChange={(event) => setNewAccountName(event.target.value)} placeholder="New account name" onKeyDown={(event) => { if (event.key === "Enter") void addAccount(); }} />
            <button type="button" onClick={() => void addAccount()} disabled={creating || !newAccountName.trim()}>＋</button>
          </div>
        </aside>

        <main className="customerAccountPanel">
          {detail ? (
            <>
              {profileDraft ? (
                <section className="customerProfileForm" aria-label="Edit account details">
                  <div className="customerProfileGrid">
                    <label><span>Name</span><input value={profileDraft.name} onChange={(event) => setProfileDraft({ ...profileDraft, name: event.target.value })} /></label>
                    <label><span>Status</span>
                      <select value={profileDraft.status} onChange={(event) => setProfileDraft({ ...profileDraft, status: event.target.value as CustomerAccount["status"] })}>
                        <option value="active">Active</option>
                        <option value="paused">Paused</option>
                        <option value="archived">Archived</option>
                      </select>
                    </label>
                    <label><span>Industry</span><input value={profileDraft.industry} onChange={(event) => setProfileDraft({ ...profileDraft, industry: event.target.value })} placeholder="Government" /></label>
                    <label><span>Region</span><input value={profileDraft.region} onChange={(event) => setProfileDraft({ ...profileDraft, region: event.target.value })} placeholder="UAE" /></label>
                    <label className="wide"><span>Also known as (comma separated)</span><input value={profileDraft.aliases} onChange={(event) => setProfileDraft({ ...profileDraft, aliases: event.target.value })} placeholder="OHI UNHCR, UNHCR Oman" /></label>
                  </div>
                  <footer>
                    <small>Aliases are searched alongside the name, so an account found under either spelling is the same account.</small>
                    <button className="ghostButton" type="button" onClick={() => setProfileDraft(null)}>Cancel</button>
                    <button className="primaryButton" type="button" disabled={!profileDraft.name.trim() || busy === "profile"} onClick={() => void saveProfile()}>
                      {busy === "profile" ? "Saving…" : "Save details"}
                    </button>
                  </footer>
                </section>
              ) : (
                <header className="customerAccountHeader">
                  <div>
                    <span className="eyebrow">Account{detail.account.status !== "active" ? ` · ${detail.account.status}` : ""}</span>
                    <h2>{detail.account.name}</h2>
                    <p>{[detail.account.industry, detail.account.region, detail.account.aliases.length ? `aka ${detail.account.aliases.join(", ")}` : ""].filter(Boolean).join(" · ") || "No profile details yet"}</p>
                  </div>
                  <div className="customerAccountActions">
                    <span className={`customerModelState state-${session?.state ?? "off"}`}><i />{session?.state === "ready" ? `${session.selected_model} ready` : "Model off · capture still works"}</span>
                    <button className="secondaryButton" type="button" onClick={() => setProfileDraft({
                      name: detail.account.name,
                      aliases: detail.account.aliases.join(", "),
                      industry: detail.account.industry,
                      region: detail.account.region,
                      status: detail.account.status,
                    })}>Edit details</button>
                    <button className="dangerButton" type="button" onClick={() => void removeAccount()} disabled={busy === "delete-account"}>{busy === "delete-account" ? "Deleting…" : "Delete customer"}</button>
                  </div>
                </header>
              )}

              <nav className="customerTabs" aria-label="Account sections">
                {TABS.map(([value, label]) => {
                  const count = value === "notes" ? detail.notes.length
                    : value === "wins" ? detail.wins.length
                    : value === "facts" ? detail.facts.length
                    : value === "people" ? detail.people.length
                    : value === "sources" ? detail.sources.length
                    : value === "actions" ? detail.actions.filter((item) => item.status === "open").length
                    : 0;
                  return (
                    <button type="button" key={value} className={tab === value ? "active" : ""} onClick={() => setTab(value)}>
                      {label}{count ? <i>{count}</i> : null}
                    </button>
                  );
                })}
              </nav>

              {tab === "overview" ? (
                <section className="customerOverview">
                  <article><span>Last interaction</span><strong>{when(detail.account.last_interaction_at)}</strong></article>
                  <article><span>Open actions</span><strong>{detail.account.open_actions}</strong></article>
                  <article><span>Saved facts</span><strong>{detail.facts.length}</strong></article>
                  <article><span>Notes</span><strong>{detail.notes.length}</strong></article>
                  <div className="customerSectionCard wide">
                    <header><strong>Latest understanding</strong></header>
                    {detail.facts.slice(0, 6).map((fact) => <p key={fact.id}><b>{fact.kind.replace("_", " ")}</b>{fact.content}</p>)}
                    {!detail.facts.length ? <em>No reviewed facts yet.</em> : null}
                  </div>
                  <div className="customerSectionCard wide">
                    <header><strong>Next actions</strong></header>
                    {detail.actions.filter((item) => item.status === "open").slice(0, 5).map((action) => (
                      <label key={action.id}>
                        <input type="checkbox" checked={false} disabled={busy === action.id} onChange={() => void toggleAction(action.id, action.status)} />
                        <span>{action.description}<small>{action.owner || "Unassigned"}{action.due_at ? ` · ${when(action.due_at)}` : ""}</small></span>
                      </label>
                    ))}
                    {!detail.actions.some((item) => item.status === "open") ? <em>No open actions.</em> : null}
                  </div>
                  {detail.notes.some((note) => note.pinned) ? (
                    <div className="customerSectionCard wide">
                      <header><strong>Pinned notes</strong></header>
                      {detail.notes.filter((note) => note.pinned).map((note) => (
                        <p key={note.id}><b>{note.title || "note"}</b>{note.body.slice(0, 220)}{note.body.length > 220 ? "…" : ""}</p>
                      ))}
                    </div>
                  ) : null}
                </section>
              ) : null}

              {tab === "notes" ? (
                <section className="customerNoteList">
                  <div className="customerListHead">
                    <p>Notes are yours: written directly, never analyzed, and never queued for review. Pin one to hand it to every conversation scoped to this account.</p>
                    <button className="primaryButton" type="button" onClick={() => setNoteComposerOpen((value) => !value)}>
                      {noteComposerOpen ? "Close" : "✎ New note"}
                    </button>
                  </div>

                  {noteComposerOpen ? (
                    <article className="customerNoteComposer">
                      <input
                        value={newNote.title}
                        onChange={(event) => setNewNote({ ...newNote, title: event.target.value })}
                        placeholder="Title (optional)"
                        aria-label="Note title"
                      />
                      <textarea
                        value={newNote.body}
                        onChange={(event) => setNewNote({ ...newNote, body: event.target.value })}
                        placeholder="What should this account always carry with it?"
                        aria-label="Note body"
                      />
                      <footer>
                        <label className="customerPinToggle">
                          <input type="checkbox" checked={newNote.pinned} onChange={(event) => setNewNote({ ...newNote, pinned: event.target.checked })} />
                          <span>Pin to account context</span>
                        </label>
                        <button className="ghostButton" type="button" onClick={() => { setNoteComposerOpen(false); setNewNote(EMPTY_NOTE); }}>Cancel</button>
                        <button className="primaryButton" type="button" disabled={!newNote.body.trim() || busy === "new-note"} onClick={() => void addNote()}>
                          {busy === "new-note" ? "Saving…" : "Save note"}
                        </button>
                      </footer>
                    </article>
                  ) : null}

                  {detail.notes.map((note) => editingNote?.id === note.id ? (
                    <article key={note.id} className="customerNoteComposer">
                      <input
                        value={editingNote.draft.title}
                        onChange={(event) => setEditingNote({ ...editingNote, draft: { ...editingNote.draft, title: event.target.value } })}
                        placeholder="Title (optional)"
                        aria-label="Note title"
                      />
                      <textarea
                        value={editingNote.draft.body}
                        onChange={(event) => setEditingNote({ ...editingNote, draft: { ...editingNote.draft, body: event.target.value } })}
                        aria-label="Note body"
                      />
                      <footer>
                        <label className="customerPinToggle">
                          <input type="checkbox" checked={editingNote.draft.pinned} onChange={(event) => setEditingNote({ ...editingNote, draft: { ...editingNote.draft, pinned: event.target.checked } })} />
                          <span>Pin to account context</span>
                        </label>
                        <button className="ghostButton" type="button" onClick={() => setEditingNote(null)}>Cancel</button>
                        <button className="primaryButton" type="button" disabled={!editingNote.draft.body.trim() || busy === note.id} onClick={() => void saveNote()}>
                          {busy === note.id ? "Saving…" : "Save"}
                        </button>
                      </footer>
                    </article>
                  ) : (
                    <article key={note.id} className={note.pinned ? "isPinned" : ""}>
                      <header>
                        <div>
                          <strong>{note.title || "Note"}</strong>
                          <small>
                            {when(note.updated_at)}
                            {note.origin === "chat" ? " · saved from a conversation" : ""}
                            {note.pinned ? " · pinned" : ""}
                          </small>
                        </div>
                        <div className="customerRowActions">
                          <button type="button" disabled={busy === note.id} onClick={() => void togglePin(note)}>{note.pinned ? "Unpin" : "Pin"}</button>
                          <button type="button" onClick={() => setEditingNote({ id: note.id, draft: { title: note.title, body: note.body, pinned: note.pinned } })}>Edit</button>
                          <button type="button" className="isDanger" disabled={busy === note.id} onClick={() => void removeNote(note.id)}>Delete</button>
                        </div>
                      </header>
                      <p>{note.body}</p>
                    </article>
                  ))}
                  {!detail.notes.length && !noteComposerOpen ? <div className="customerEmpty">No notes on this account yet.</div> : null}
                </section>
              ) : null}

              {tab === "timeline" ? (
                <section className="customerTimeline">
                  {detail.interactions.map((item) => (
                    <article key={item.id}><time>{when(item.occurred_at)}</time><div><strong>{item.title}</strong><p>{item.summary}</p></div></article>
                  ))}
                  {!detail.interactions.length ? <div className="customerEmpty">No saved interactions yet.</div> : null}
                </section>
              ) : null}

              {tab === "actions" ? (
                <section className="customerActionList">
                  <div className="customerListHead">
                    <p>{detail.actions.length ? `${detail.actions.filter((item) => item.status === "open").length} open of ${detail.actions.length}.` : "No actions captured yet."}</p>
                    <button className="primaryButton" type="button" onClick={() => setNewAction(newAction ? null : EMPTY_ACTION)}>{newAction ? "Close" : "＋ Add action"}</button>
                  </div>

                  {newAction ? (
                    <div className="customerInlineForm">
                      <input value={newAction.description} onChange={(event) => setNewAction({ ...newAction, description: event.target.value })} placeholder="What needs doing" aria-label="Action description" />
                      <input value={newAction.owner} onChange={(event) => setNewAction({ ...newAction, owner: event.target.value })} placeholder="Owner" aria-label="Action owner" />
                      <input type="date" value={newAction.due} onChange={(event) => setNewAction({ ...newAction, due: event.target.value })} aria-label="Due date" />
                      <div className="customerInlineActions">
                        <button className="ghostButton" type="button" onClick={() => setNewAction(null)}>Cancel</button>
                        <button className="primaryButton" type="button" disabled={!newAction.description.trim() || busy === "new-action"} onClick={() => void addAction()}>
                          {busy === "new-action" ? "Adding…" : "Add"}
                        </button>
                      </div>
                    </div>
                  ) : null}

                  {detail.actions.map((action) => editingAction?.id === action.id ? (
                    <div className="customerInlineForm" key={action.id}>
                      <input value={editingAction.draft.description} onChange={(event) => setEditingAction({ ...editingAction, draft: { ...editingAction.draft, description: event.target.value } })} aria-label="Action description" />
                      <input value={editingAction.draft.owner} onChange={(event) => setEditingAction({ ...editingAction, draft: { ...editingAction.draft, owner: event.target.value } })} placeholder="Owner" aria-label="Action owner" />
                      <input type="date" value={editingAction.draft.due} onChange={(event) => setEditingAction({ ...editingAction, draft: { ...editingAction.draft, due: event.target.value } })} aria-label="Due date" />
                      <select value={editingAction.draft.status} onChange={(event) => setEditingAction({ ...editingAction, draft: { ...editingAction.draft, status: event.target.value as CustomerAction["status"] } })} aria-label="Action status">
                        <option value="open">Open</option>
                        <option value="done">Done</option>
                        <option value="cancelled">Cancelled</option>
                      </select>
                      <div className="customerInlineActions">
                        <button className="ghostButton" type="button" onClick={() => setEditingAction(null)}>Cancel</button>
                        <button className="primaryButton" type="button" disabled={!editingAction.draft.description.trim() || busy === action.id} onClick={() => void saveAction()}>
                          {busy === action.id ? "Saving…" : "Save"}
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div className={`customerActionRow ${action.status !== "open" ? "complete" : ""} ${isOverdue(action) ? "isOverdue" : ""}`} key={action.id}>
                      <label>
                        <input type="checkbox" checked={action.status === "done"} disabled={busy === action.id} onChange={() => void toggleAction(action.id, action.status)} />
                        <span>
                          <strong>{action.description}</strong>
                          <small>
                            {action.owner || "Unassigned"}
                            {action.due_at ? ` · ${isOverdue(action) ? "overdue since" : "due"} ${when(action.due_at)}` : ""}
                            {action.status === "cancelled" ? " · cancelled" : ""}
                          </small>
                        </span>
                      </label>
                      <div className="customerRowActions">
                        <button type="button" onClick={() => setEditingAction({ id: action.id, draft: { description: action.description, owner: action.owner, due: dateInputValue(action.due_at), status: action.status } })}>Edit</button>
                        <button type="button" className="isDanger" disabled={busy === action.id} onClick={() => void removeAction(action.id)}>Delete</button>
                      </div>
                    </div>
                  ))}
                  {!detail.actions.length && !newAction ? <div className="customerEmpty">No actions captured yet.</div> : null}
                </section>
              ) : null}

              {tab === "wins" ? (
                <section className="customerWinList">
                  <div className="customerWinListHead">
                    <p>{detail.wins.length ? `${detail.wins.length} win${detail.wins.length === 1 ? "" : "s"} recorded for ${detail.account.name}.` : "No wins recorded for this account yet."}</p>
                    <button className="primaryButton" type="button" onClick={() => openWinForm()}>🏆 Record win</button>
                  </div>
                  {detail.wins.map((win) => (
                    <article key={win.id}>
                      <header>
                        <div>
                          <strong>{win.title}</strong>
                          <small>{winDay(win.won_at || win.created_at)}{win.yearly_arr !== null ? ` · ${arr(win.yearly_arr)} yearly ARR` : ""}</small>
                        </div>
                        <div className="customerRowActions">
                          <button type="button" onClick={() => openWinForm(win)}>Edit</button>
                          <button type="button" className="isDanger" disabled={busy === win.id} onClick={() => void removeWin(win.id)}>{busy === win.id ? "Removing…" : "Remove"}</button>
                        </div>
                      </header>
                      {win.services.length ? <div className="customerWinChips">{win.services.map((service) => <b key={service}>{service}</b>)}</div> : null}
                      {win.dac_shape ? <small className="customerWinShape">DAC: {win.dac_shape}</small> : null}
                      {win.brief ? <p>{win.brief}</p> : null}
                      {renderValuation(win)}
                    </article>
                  ))}
                </section>
              ) : null}

              {tab === "facts" ? (
                <section className="customerFactSection">
                  <div className="customerListHead">
                    <div className="customerAccountFilters" role="group" aria-label="Filter facts">
                      {FACT_FILTERS.map(([value, label]) => (
                        <button key={value} type="button" className={factFilter === value ? "active" : ""} aria-pressed={factFilter === value} onClick={() => setFactFilter(value)}>{label}</button>
                      ))}
                    </div>
                    <button className="primaryButton" type="button" onClick={() => setNewFact(newFact ? null : EMPTY_FACT)}>{newFact ? "Close" : "＋ Add fact"}</button>
                  </div>

                  {newFact ? (
                    <div className="customerInlineForm">
                      <select value={newFact.kind} onChange={(event) => setNewFact({ ...newFact, kind: event.target.value })} aria-label="Fact kind">
                        {FACT_KINDS.map((kind) => <option key={kind} value={kind}>{kind.replace("_", " ")}</option>)}
                      </select>
                      <textarea value={newFact.content} onChange={(event) => setNewFact({ ...newFact, content: event.target.value })} placeholder="What is true about this account?" aria-label="Fact content" />
                      <div className="customerInlineActions">
                        <button className="ghostButton" type="button" onClick={() => setNewFact(null)}>Cancel</button>
                        <button className="primaryButton" type="button" disabled={!newFact.content.trim() || busy === "new-fact"} onClick={() => void addFact()}>
                          {busy === "new-fact" ? "Adding…" : "Add"}
                        </button>
                      </div>
                    </div>
                  ) : null}

                  <div className="customerFactGrid">
                    {visibleFacts.map((fact) => editingFact?.id === fact.id ? (
                      <div className="customerInlineForm" key={fact.id}>
                        <select value={editingFact.draft.kind} onChange={(event) => setEditingFact({ ...editingFact, draft: { ...editingFact.draft, kind: event.target.value } })} aria-label="Fact kind">
                          {FACT_KINDS.map((kind) => <option key={kind} value={kind}>{kind.replace("_", " ")}</option>)}
                        </select>
                        <textarea value={editingFact.draft.content} onChange={(event) => setEditingFact({ ...editingFact, draft: { ...editingFact.draft, content: event.target.value } })} aria-label="Fact content" />
                        <select value={editingFact.draft.status} onChange={(event) => setEditingFact({ ...editingFact, draft: { ...editingFact.draft, status: event.target.value as CustomerFact["status"] } })} aria-label="Fact status">
                          <option value="active">Active</option>
                          <option value="disputed">Disputed</option>
                          <option value="superseded">Superseded</option>
                        </select>
                        <div className="customerInlineActions">
                          <button className="ghostButton" type="button" onClick={() => setEditingFact(null)}>Cancel</button>
                          <button className="primaryButton" type="button" disabled={!editingFact.draft.content.trim() || busy === fact.id} onClick={() => void saveFact()}>
                            {busy === fact.id ? "Saving…" : "Save"}
                          </button>
                        </div>
                      </div>
                    ) : (
                      <article key={fact.id} className={fact.status !== "active" ? `is-${fact.status}` : ""}>
                        <span>{fact.kind.replace("_", " ")}{fact.status !== "active" ? ` · ${fact.status}` : ""}</span>
                        <p>{fact.content}</p>
                        <small>{Math.round(fact.confidence * 100)}% confidence · {fact.evidence.quote || (fact.interaction_id ? "source linked" : "written by you")}</small>
                        <div className="customerRowActions">
                          <button type="button" onClick={() => setEditingFact({ id: fact.id, draft: { kind: fact.kind, content: fact.content, status: fact.status } })}>Edit</button>
                          <button type="button" className="isDanger" disabled={busy === fact.id} onClick={() => void removeFact(fact.id)}>Delete</button>
                        </div>
                      </article>
                    ))}
                    {!visibleFacts.length ? (
                      <div className="customerEmpty">
                        {detail.facts.length ? "No facts of that kind yet." : "No facts saved yet — add one, or analyze a captured note."}
                      </div>
                    ) : null}
                  </div>
                </section>
              ) : null}

              {tab === "people" ? (
                <section className="customerPeopleSection">
                  <div className="customerListHead">
                    <p>{detail.people.length ? `${detail.people.length} contact${detail.people.length === 1 ? "" : "s"}.` : "No people captured yet."}</p>
                    <button className="primaryButton" type="button" onClick={() => setNewPerson(newPerson ? null : EMPTY_PERSON)}>{newPerson ? "Close" : "＋ Add contact"}</button>
                  </div>
                  {newPerson ? (
                    <div className="customerInlineForm">
                      <input value={newPerson.name} onChange={(event) => setNewPerson({ ...newPerson, name: event.target.value })} placeholder="Name" aria-label="Contact name" />
                      <input value={newPerson.role} onChange={(event) => setNewPerson({ ...newPerson, role: event.target.value })} placeholder="Role" aria-label="Contact role" />
                      <input value={newPerson.organization} onChange={(event) => setNewPerson({ ...newPerson, organization: event.target.value })} placeholder="Organization" aria-label="Contact organization" />
                      <div className="customerInlineActions">
                        <button className="ghostButton" type="button" onClick={() => setNewPerson(null)}>Cancel</button>
                        <button className="primaryButton" type="button" disabled={!newPerson.name.trim() || busy === "new-person"} onClick={() => void addPerson()}>
                          {busy === "new-person" ? "Adding…" : "Add"}
                        </button>
                      </div>
                    </div>
                  ) : null}
                  <div className="customerPeopleGrid">
                    {detail.people.map(renderPerson)}
                    {!detail.people.length && !newPerson ? <div className="customerEmpty">No people captured yet.</div> : null}
                  </div>
                </section>
              ) : null}

              {tab === "sources" ? (
                <section className="customerSourceList">
                  {detail.sources.map((source) => editingSource?.id === source.id ? (
                    <article key={source.id}>
                      <div className="customerInlineForm">
                        <input value={editingSource.draft.title} onChange={(event) => setEditingSource({ ...editingSource, draft: { ...editingSource.draft, title: event.target.value } })} aria-label="Note title" />
                        <select value={editingSource.draft.source_kind} onChange={(event) => setEditingSource({ ...editingSource, draft: { ...editingSource.draft, source_kind: event.target.value } })} aria-label="Note type">
                          {SOURCE_KINDS.map((kind) => <option key={kind} value={kind}>{kind}</option>)}
                        </select>
                        <textarea value={editingSource.draft.content} onChange={(event) => setEditingSource({ ...editingSource, draft: { ...editingSource.draft, content: event.target.value } })} aria-label="Note content" />
                        <div className="customerInlineActions">
                          <button className="ghostButton" type="button" onClick={() => setEditingSource(null)}>Cancel</button>
                          <button className="primaryButton" type="button" disabled={busy === source.id} onClick={() => void saveSource()}>
                            {busy === source.id ? "Saving…" : "Save note"}
                          </button>
                        </div>
                      </div>
                    </article>
                  ) : (
                    <article key={source.id}>
                      <header>
                        <div><span>{source.source_kind}</span><strong>{source.title}</strong></div>
                        <b className={`source-${source.status}`}>{source.status === "waiting" ? "Waiting for analysis" : source.status}</b>
                      </header>
                      <p>{source.content}</p>
                      <footer>
                        <small>{when(source.occurred_at || source.created_at)}</small>
                        <div className="customerRowActions">
                          <button type="button" onClick={() => setEditingSource({ id: source.id, draft: { title: source.title, content: source.content, source_kind: source.source_kind } })}>Edit</button>
                          <button type="button" className="isDanger" disabled={busy === source.id} onClick={() => void removeSource(source.id)}>Delete</button>
                          {source.status === "waiting" ? (
                            <button className="secondaryButton" type="button" disabled={session?.state !== "ready" || busy === source.id} onClick={() => void analyze(source.id)}>
                              {session?.state === "ready" ? busy === source.id ? "Analyzing…" : "Analyze note" : "Launch model to analyze"}
                            </button>
                          ) : null}
                        </div>
                      </footer>
                    </article>
                  ))}
                  {!detail.sources.length ? <div className="customerEmpty">Capture a note to start the source trail.</div> : null}
                </section>
              ) : null}

              {tab === "outputs" ? (
                <section className="customerOutputs">
                  <div className="customerOutputSetup">
                    <label><span>Company activity tracker URL</span><input value={settings.tracker_url} onChange={(event) => setSettings((current) => ({ ...current, tracker_url: event.target.value }))} placeholder="https://company.example/activity" /></label>
                    <button className="secondaryButton" type="button" onClick={() => void saveTrackerUrl()} disabled={busy === "settings"}>Save link</button>
                  </div>
                  <button className="primaryButton" type="button" onClick={() => void generateOutput()} disabled={busy === "output" || !detail.interactions.length}>{busy === "output" ? "Building…" : "Generate activity tracker Markdown"}</button>
                  {output ? (
                    <article className="customerOutputPreview">
                      <header><strong>Activity tracker update</strong><button type="button" className="primaryButton" onClick={copyAndOpen}>Copy Markdown &amp; open tracker</button></header>
                      <pre>{output.content}</pre>
                    </article>
                  ) : null}
                </section>
              ) : null}
            </>
          ) : <div className="customerEmpty large"><strong>Select or add an account</strong><span>The workbench keeps every note, fact, action, and output scoped to that customer.</span></div>}
        </main>
      </div>

      {searchOpen ? (
        <CustomerSearch
          onDismiss={() => setSearchOpen(false)}
          onOpen={(hit) => {
            setSearchOpen(false);
            setQuery("");
            setFilter("all");
            selectAccount(hit.account_id, TAB_FOR_HIT[hit.kind]);
          }}
        />
      ) : null}

      {captureOpen ? <div className="customerModalBackdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) setCaptureOpen(false); }}><section className="customerModal" role="dialog" aria-modal="true" aria-label="Capture customer note"><header><div><span className="eyebrow">Raw source</span><strong>Capture a customer note</strong></div><button type="button" onClick={() => setCaptureOpen(false)}>×</button></header><label><span>Type</span><select value={noteKind} onChange={(event) => setNoteKind(event.target.value as typeof noteKind)}><option value="meeting">Meeting</option><option value="note">Note</option><option value="chat">Chat</option><option value="notion">Notion</option><option value="attachment">Attachment</option></select></label><label><span>Title</span><input value={noteTitle} onChange={(event) => setNoteTitle(event.target.value)} placeholder="Discovery call · July 28" /></label><label><span>Markdown notes</span><textarea value={noteContent} onChange={(event) => setNoteContent(event.target.value)} placeholder="Paste the original notes here. Metis saves them first; analysis is a separate action." /></label><p>{session?.state === "ready" ? "The model is ready, but analysis still waits for your explicit click." : "The model is off. This note will be saved as Waiting for analysis without launching anything."} Writing something down for yourself? Use <b>Add note</b> instead — it skips analysis entirely.</p><footer><button className="secondaryButton" type="button" onClick={() => setCaptureOpen(false)}>Cancel</button><button className="primaryButton" type="button" disabled={!noteTitle.trim() || !noteContent.trim() || busy === "capture"} onClick={() => void capture()}>{busy === "capture" ? "Saving…" : "Save raw note"}</button></footer></section></div> : null}

      {winOpen ? (
        <div className="customerModalBackdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) setWinOpen(false); }}>
          <section className="customerModal" role="dialog" aria-modal="true" aria-label={editingWinId ? "Edit customer win" : "Record customer win"}>
            <header>
              <div><span className="eyebrow">Win tracker</span><strong>{editingWinId ? "Edit win" : "Record a win"}{detail ? ` · ${detail.account.name}` : ""}</strong></div>
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
              <button className="primaryButton" type="button" disabled={!winTitle.trim() || !selectedId || busy === "win"} onClick={() => void submitWin()}>
                {busy === "win" ? "Saving…" : editingWinId ? "Save win" : "Record win"}
              </button>
            </footer>
          </section>
        </div>
      ) : null}

      {ratesOpen && rateCard ? (
        <div className="customerModalBackdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) setRatesOpen(false); }}>
          <section className="customerModal wide" role="dialog" aria-modal="true" aria-label="Oracle SKU rate card">
            <header>
              <div>
                <span className="eyebrow">{rateCard.catalog_size} Oracle SKUs · {rateCard.rates.length} priced</span>
                <strong>Rate card</strong>
              </div>
              <button type="button" onClick={() => setRatesOpen(false)}>×</button>
            </header>
            <p className="customerRatesNote">
              Oracle&rsquo;s service descriptions publish every SKU and the unit it bills in, but no prices.
              These rates are seeded from public list pricing and are <b>unverified</b> until you confirm them.
              Saving a rate marks it verified.
            </p>
            <div className="customerRateList">
              {rateCard.rates.map((rate) => (
                <label key={rate.key} className={rate.verified ? "isVerified" : ""}>
                  <span>
                    <b>{rate.label}</b>
                    <small>{rate.part_number || "no part number"} · per {rate.unit.toLowerCase().replace(/^(\d+,?\d*\s)?/, "")}</small>
                  </span>
                  <input
                    inputMode="decimal"
                    value={rateEdits[rate.key] ?? String(rate.value)}
                    onChange={(event) => setRateEdits((current) => ({ ...current, [rate.key]: event.target.value }))}
                    aria-label={`Rate for ${rate.label}`}
                  />
                  <i title={rate.verified ? "Verified by you" : "Seeded, not verified"}>{rate.verified ? "✓" : "?"}</i>
                </label>
              ))}
            </div>
            <footer>
              <small>{rateCard.source_urls[0]}</small>
              <button className="secondaryButton" type="button" onClick={() => { setRateEdits({}); setRatesOpen(false); }}>Close</button>
              <button className="primaryButton" type="button" disabled={!Object.keys(rateEdits).length || busy === "rates"} onClick={() => void saveRates()}>
                {busy === "rates" ? "Saving…" : `Save ${Object.keys(rateEdits).length || ""} rate${Object.keys(rateEdits).length === 1 ? "" : "s"}`}
              </button>
            </footer>
          </section>
        </div>
      ) : null}

      {proposal && review ? <div className="customerModalBackdrop"><section className="customerReviewModal" role="dialog" aria-modal="true" aria-label="Review extracted customer update"><header><div><span className="eyebrow">One review · one save</span><strong>Review customer update</strong><p>Nothing below becomes account knowledge until you save it.</p></div><button type="button" onClick={() => { setProposal(null); setReview(null); }}>×</button></header><div className="customerReviewBody"><label><span>Summary</span><textarea value={review.summary} onChange={(event) => setReview((current) => current ? { ...current, summary: event.target.value } : current)} /></label><section><header><strong>Facts</strong><span>{review.facts.length}</span></header>{review.facts.map((fact, index) => <article key={`${fact.kind}-${index}`}><select value={fact.kind} onChange={(event) => setReview((current) => current ? { ...current, facts: current.facts.map((item, itemIndex) => itemIndex === index ? { ...item, kind: event.target.value } : item) } : current)}>{FACT_KINDS.map((kind) => <option key={kind} value={kind}>{kind.replace("_", " ")}</option>)}</select><textarea value={fact.content} onChange={(event) => setReview((current) => current ? { ...current, facts: current.facts.map((item, itemIndex) => itemIndex === index ? { ...item, content: event.target.value } : item) } : current)} /><small>“{fact.evidence.quote || "No evidence quote"}”</small><button type="button" onClick={() => setReview((current) => current ? { ...current, facts: current.facts.filter((_, itemIndex) => itemIndex !== index) } : current)}>Remove</button></article>)}</section><section><header><strong>Actions</strong><span>{review.actions.length}</span></header>{review.actions.map((action, index) => <article key={index}><textarea value={action.description} onChange={(event) => setReview((current) => current ? { ...current, actions: current.actions.map((item, itemIndex) => itemIndex === index ? { ...item, description: event.target.value } : item) } : current)} /><input value={action.owner} placeholder="Owner" onChange={(event) => setReview((current) => current ? { ...current, actions: current.actions.map((item, itemIndex) => itemIndex === index ? { ...item, owner: event.target.value } : item) } : current)} /><small>“{action.evidence.quote || "No evidence quote"}”</small><button type="button" onClick={() => setReview((current) => current ? { ...current, actions: current.actions.filter((_, itemIndex) => itemIndex !== index) } : current)}>Remove</button></article>)}</section><section><header><strong>People</strong><span>{review.people.length}</span></header>{review.people.map((person, index) => <article key={index}><input value={person.name} onChange={(event) => setReview((current) => current ? { ...current, people: current.people.map((item, itemIndex) => itemIndex === index ? { ...item, name: event.target.value } : item) } : current)} /><input value={person.role} placeholder="Role" onChange={(event) => setReview((current) => current ? { ...current, people: current.people.map((item, itemIndex) => itemIndex === index ? { ...item, role: event.target.value } : item) } : current)} /><small>“{person.evidence.quote || "No evidence quote"}”</small><button type="button" onClick={() => setReview((current) => current ? { ...current, people: current.people.filter((_, itemIndex) => itemIndex !== index) } : current)}>Remove</button></article>)}</section></div><footer><span>Extracted locally with {proposal.model}</span><button className="primaryButton" type="button" onClick={() => void saveReview()} disabled={busy === "review"}>{busy === "review" ? "Saving…" : "Save update"}</button></footer></section></div> : null}
    </div>
  );
}
