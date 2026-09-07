"use client";

// The customer workbench's state: the account index and dashboard, the open
// account, and one `run` that every edit goes through (busy flag, error,
// toast, and a re-read of the index and the account, since almost every edit
// changes a count somewhere).

import { useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { useToast } from "@/components/ui/toast";
import { analyzeCustomerSource, getCustomer, getCustomerDashboard, getCustomerSettings, getCustomerSourceProposal, getLocalModelSession, getModelPreference, getSkuRates, listCustomers, saveCustomerProposal } from "@/lib/api";
import { matchesFilter, type AccountFilter } from "@/lib/customers";
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

  const [accounts, setAccounts] = useState<CustomerAccount[]>([]);
  const [dashboard, setDashboard] = useState<CustomerDashboard | null>(null);
  const [settings, setSettings] = useState<CustomerSettings>({ tracker_url: "", activity_template: "", updated_at: null });
  const [rateCard, setRateCard] = useState<SkuRateCard | null>(null);
  const [session, setSession] = useState<LocalModelSession | null>(null);
  const [modelPreference, setModelPreference] = useState<ModelPreference | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(requestedAccount);
  const [detail, setDetail] = useState<CustomerAccountDetail | null>(null);
  const [tab, setTab] = useState<Tab>(isTab(requestedTab) ? requestedTab : "overview");
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<AccountFilter>("all");
  const [dialog, setDialog] = useState<Dialog>(null);
  const [proposal, setProposal] = useState<CustomerProposal | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const busyRef = useRef<string | null>(null);

  const refreshIndex = useCallback(async () => {
    const [nextAccounts, nextDashboard, nextSettings, nextSession, nextRates, nextPreference] = await Promise.all([
      listCustomers(), getCustomerDashboard(), getCustomerSettings(),
      getLocalModelSession().catch(() => null), getSkuRates().catch(() => null), getModelPreference().catch(() => null),
    ]);
    setAccounts(nextAccounts);
    setDashboard(nextDashboard);
    setSettings(nextSettings);
    if (nextSession) setSession(nextSession);
    if (nextRates) setRateCard(nextRates);
    if (nextPreference) setModelPreference(nextPreference);
    setLoaded(true);
  }, []);
  const refreshDetail = useCallback(async (id: string) => setDetail(await getCustomer(id)), []);
  const refreshAll = useCallback(async () => {
    await Promise.all([refreshIndex(), selectedId ? refreshDetail(selectedId) : Promise.resolve()]);
  }, [refreshDetail, refreshIndex, selectedId]);

  useEffect(() => {
    void refreshIndex().catch((loadError) => setError(loadError instanceof Error ? loadError.message : "Customer data could not be loaded."));
  }, [refreshIndex]);

  // A deep link carries the account and the tab; follow both.
  useEffect(() => {
    if (requestedAccount) setSelectedId(requestedAccount);
    if (isTab(requestedTab)) setTab(requestedTab);
  }, [requestedAccount, requestedTab]);

  useEffect(() => {
    if (!selectedId) { setDetail(null); return; }
    let mounted = true;
    void getCustomer(selectedId).then((next) => mounted && setDetail(next)).catch((loadError) => mounted && setError(loadError instanceof Error ? loadError.message : "That account could not be opened."));
    return () => { mounted = false; };
  }, [selectedId]);

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
  const select = useCallback((id: string | null, nextTab: Tab = "overview") => {
    setSelectedId(id);
    setTab(nextTab);
    router.replace(id ? `/customers?account=${encodeURIComponent(id)}&tab=${nextTab}` : "/customers");
  }, [router]);

  /** Every edit: busy, try, toast, re-read. Returns whether it succeeded. */
  const run = useCallback(async (key: string, work: () => Promise<unknown>, notice?: string): Promise<boolean> => {
    if (busyRef.current) return false;
    busyRef.current = key;
    setBusy(key);
    setError(null);
    try {
      await work();
      if (notice) toast(notice);
      await refreshAll();
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
    selectedId, detail, tab, setTab, select, query, setQuery, filter, setFilter,
    dialog, setDialog, proposal, setProposal, openProposal, saveProposal,
    busy, error, setError, run, refreshAll, toast, modelLabel, modelReady, sessionState: session?.state ?? "off",
    requestedSource: params.get("source"), requestedAction: params.get("action"),
  };
}

export type Customers = ReturnType<typeof useCustomers>;
