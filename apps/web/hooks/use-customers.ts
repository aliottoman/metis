"use client";

// The customer workbench's state: the account index and dashboard, the open
// account, and one `run` that every edit goes through (busy flag, error,
// toast, and a re-read of the index and the account, since almost every edit
// changes a count somewhere).

import { useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { useToast } from "@/components/ui/toast";
import { analyzeCustomerSource, getCustomer, getCustomerDashboard, getCustomerSettings, getCustomerSourceProposal, getLocalModelSession, getModelPreference, getSkuRates, listCustomers, saveCustomerProposal } from "@/lib/api";
import { ACCOUNT_FILTERS, matchesFilter, type AccountFilter } from "@/lib/customers";
import { activeModelLabel, isCloudActive } from "@/lib/model";
import type { CustomerAccount, CustomerAccountDetail, CustomerDashboard, CustomerExtraction, CustomerProposal, CustomerSettings, CustomerWin, LocalModelSession, ModelPreference, SkuRateCard } from "@/lib/types";

export type Tab = "overview" | "notes" | "actions" | "wins" | "facts" | "people" | "sources" | "timeline" | "outputs";
export const TABS: Array<[Tab, string]> = [["overview", "Overview"], ["notes", "Notes"], ["actions", "Actions"], ["wins", "Wins"], ["facts", "Facts"], ["people", "People"], ["sources", "Sources"], ["timeline", "Timeline"], ["outputs", "Outputs"]];

export type Dialog = { kind: "capture" } | { kind: "win"; win?: CustomerWin } | { kind: "rates" } | { kind: "search" } | null;

function isTab(value: string | null): value is Tab {
  return TABS.some(([tab]) => tab === value);
}

export function useCustomers() {
  const params = useSearchParams();
  const router = useRouter();
  const toast = useToast();
  const requestedAccount = params.get("account") || null;
  const requestedTab = params.get("tab");
  const requestedQuery = params.get("q") ?? "";
  const requestedFilter = params.get("filter");

  const [accounts, setAccounts] = useState<CustomerAccount[]>([]);
  const [dashboard, setDashboard] = useState<CustomerDashboard | null>(null);
  const [settings, setSettings] = useState<CustomerSettings>({ tracker_url: "", activity_template: "", updated_at: null });
  const [rateCard, setRateCard] = useState<SkuRateCard | null>(null);
  const [session, setSession] = useState<LocalModelSession | null>(null);
  const [modelPreference, setModelPreference] = useState<ModelPreference | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(requestedAccount);
  const [detail, setDetail] = useState<CustomerAccountDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(Boolean(requestedAccount));
  const [detailError, setDetailError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>(isTab(requestedTab) ? requestedTab : "overview");
  const [query, setQuery] = useState(requestedQuery);
  const [filter, setFilter] = useState<AccountFilter>(ACCOUNT_FILTERS.some(([value]) => value === requestedFilter) ? requestedFilter as AccountFilter : "all");
  const [dialog, setDialog] = useState<Dialog>(null);
  const [proposal, setProposal] = useState<CustomerProposal | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const busyRef = useRef<string | null>(null);
  const selectedIdRef = useRef<string | null>(requestedAccount);
  const detailRequest = useRef(0);
  const indexRequest = useRef(0);

  const refreshIndex = useCallback(async () => {
    const request = ++indexRequest.current;
    const [nextAccounts, nextDashboard, nextSettings, nextSession, nextRates, nextPreference] = await Promise.all([
      listCustomers(), getCustomerDashboard(), getCustomerSettings(),
      getLocalModelSession().catch(() => null), getSkuRates().catch(() => null), getModelPreference().catch(() => null),
    ]);
    if (request !== indexRequest.current) return;
    setAccounts(nextAccounts);
    setDashboard(nextDashboard);
    setSettings(nextSettings);
    if (nextSession) setSession(nextSession);
    if (nextRates) setRateCard(nextRates);
    if (nextPreference) setModelPreference(nextPreference);
    setLoaded(true);
  }, []);
  const refreshDetail = useCallback(async (id: string) => {
    const request = ++detailRequest.current;
    if (selectedIdRef.current === id) {
      setDetailLoading(true);
      setDetailError(null);
    }
    try {
      const next = await getCustomer(id);
      if (request === detailRequest.current && selectedIdRef.current === id) setDetail(next);
    } catch (loadError) {
      if (request === detailRequest.current && selectedIdRef.current === id) setDetailError(loadError instanceof Error ? loadError.message : "That account could not be opened.");
      throw loadError;
    } finally {
      if (request === detailRequest.current && selectedIdRef.current === id) setDetailLoading(false);
    }
  }, []);
  const refreshAll = useCallback(async () => {
    const id = selectedIdRef.current;
    await Promise.all([refreshIndex(), id ? refreshDetail(id) : Promise.resolve()]);
  }, [refreshDetail, refreshIndex]);

  useEffect(() => {
    void refreshIndex().catch((loadError) => setError(loadError instanceof Error ? loadError.message : "Customer data could not be loaded."));
  }, [refreshIndex]);

  // A deep link carries the account and the tab; follow both.
  useEffect(() => {
    selectedIdRef.current = requestedAccount;
    setSelectedId(requestedAccount);
    setTab(isTab(requestedTab) ? requestedTab : "overview");
  }, [requestedAccount, requestedTab]);

  useEffect(() => {
    setQuery(requestedQuery);
    setFilter(ACCOUNT_FILTERS.some(([value]) => value === requestedFilter) ? requestedFilter as AccountFilter : "all");
  }, [requestedQuery, requestedFilter]);

  useEffect(() => {
    selectedIdRef.current = selectedId;
    setDetail(null);
    setDetailError(null);
    if (!selectedId) { setDetailLoading(false); return; }
    void refreshDetail(selectedId).catch(() => undefined);
  }, [refreshDetail, selectedId]);

  useEffect(() => {
    const listener = (event: Event) => {
      const next = (event as CustomEvent<LocalModelSession>).detail;
      if (next) setSession(next);
    };
    window.addEventListener("metis:model-session", listener);
    return () => window.removeEventListener("metis:model-session", listener);
  }, []);

  // ⌘⇧K: one shift away from the shell's ⌘K, a different haystack.
  useEffect(() => {
    const listener = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.shiftKey && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setDialog({ kind: "search" });
      }
    };
    window.addEventListener("keydown", listener);
    return () => window.removeEventListener("keydown", listener);
  }, []);

  /** Open an account (or the dashboard with null) on a tab; the URL follows. */
  const select = useCallback((id: string | null, nextTab: Tab = "overview", record?: { kind: "note" | "source" | "action" | "fact"; id: string }) => {
    const changingAccount = selectedIdRef.current !== id;
    selectedIdRef.current = id;
    setSelectedId(id);
    setTab(nextTab);
    const next = new URLSearchParams();
    if (query) next.set("q", query);
    if (filter !== "all") next.set("filter", filter);
    if (id) { next.set("account", id); next.set("tab", nextTab); }
    if (id && record) next.set(record.kind, record.id);
    const href = `/customers${next.size ? `?${next}` : ""}`;
    if (changingAccount) router.push(href, { scroll: false });
    else router.replace(href, { scroll: false });
  }, [router, query, filter]);
  const changeTab = useCallback((nextTab: Tab) => select(selectedIdRef.current, nextTab), [select]);
  const changeQuery = (value: string) => { setQuery(value); const next = new URLSearchParams(window.location.search); if (value) next.set("q", value); else next.delete("q"); window.history.replaceState(null, "", `/customers${next.size ? `?${next}` : ""}`); };
  const changeFilter = (value: AccountFilter) => { setFilter(value); const next = new URLSearchParams(window.location.search); if (value !== "all") next.set("filter", value); else next.delete("filter"); window.history.replaceState(null, "", `/customers${next.size ? `?${next}` : ""}`); };

  /** Every edit: busy, try, toast, re-read. Returns whether it succeeded. */
  const run = useCallback(async (key: string, work: () => Promise<unknown>, notice?: string): Promise<boolean> => {
    if (busyRef.current) return false;
    busyRef.current = key;
    setBusy(key);
    setError(null);
    try {
      await work();
      if (notice) toast(notice);
      // The write succeeded even if its follow-up read failed. Returning false
      // here would leave a new-note draft ready to create a duplicate on retry.
      await refreshAll().catch((problem) => setError(`Saved, but the latest account data could not be loaded. ${problem instanceof Error ? problem.message : "Please refresh to see the change."}`));
      return true;
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : "That change could not be saved.");
      return false;
    } finally {
      busyRef.current = null;
      setBusy(null);
    }
  }, [refreshAll, toast]);

  /** Analyze a captured note, or open the pending update one already has. */
  const openProposal = useCallback((sourceId: string, analyze: boolean) => run(sourceId, async () => {
    const next = analyze ? await analyzeCustomerSource(sourceId) : await getCustomerSourceProposal(sourceId);
    if (next) setProposal(next);
    else toast("That note has no pending update to review.");
  }), [run, toast]);
  const saveProposal = useCallback((review: CustomerExtraction) => {
    if (!proposal) return Promise.resolve(false);
    return run("review", () => saveCustomerProposal(proposal.id, review), "Update saved. Every fact and action stays linked to its note.").then((ok) => { if (ok) setProposal(null); return ok; });
  }, [proposal, run]);

  const visibleAccounts = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return accounts.filter((account) => matchesFilter(account, filter)
      && (!needle || [account.name, account.industry, account.region, ...account.aliases].some((field) => field && field.toLowerCase().includes(needle))));
  }, [accounts, filter, query]);

  // What runs capture and analysis is the model the chat header shows; a
  // cloud pin counts as ready with nothing resident on-device.
  const modelLabel = activeModelLabel(modelPreference, session);
  const modelReady = isCloudActive(modelPreference) || session?.state === "ready";

  return {
    accounts, visibleAccounts, dashboard, settings, setSettings, rateCard, setRateCard, loaded,
    selectedId, detail, detailLoading, detailError, tab, setTab: changeTab, select, query, setQuery: changeQuery, filter, setFilter: changeFilter,
    dialog, setDialog, proposal, setProposal, openProposal, saveProposal,
    busy, error, setError, run, refreshAll, toast, modelLabel, modelReady, sessionState: session?.state ?? "off",
    requestedSource: params.get("source"), requestedAction: params.get("action"),
    requestedFact: params.get("fact"),
    requestedNote: params.get("note"),
  };
}

export type Customers = ReturnType<typeof useCustomers>;
