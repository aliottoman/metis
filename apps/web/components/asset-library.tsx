"use client";

// The Assets page: the catalog as a list with running assets first, a
// settings drawer for one asset, and a full-width player for a running one.

import { useCallback, useEffect, useMemo, useState } from "react";

import { AssetDrawer } from "@/components/assets/asset-drawer";
import { AssetList } from "@/components/assets/asset-list";
import { AssetPlayer } from "@/components/assets/asset-player";
import { Notice } from "@/components/ui/notice";
import { PageHeader } from "@/components/ui/page-header";
import { Skeleton } from "@/components/ui/skeleton";
import { useToast } from "@/components/ui/toast";
import { approveAsset, generateAssetRecipe, listAssets, revokeAssetApproval, scanAssets, startAsset, stopAsset } from "@/lib/api";
import { isActive, isFailed, mergeAsset, sortAssets, type AssetAction } from "@/lib/assets";
import type { AssetV1 } from "@/lib/types";
import { usePoll } from "@/hooks/use-poll";

const CATALOG_POLL_MS = 3_000;

type Filter = "all" | "active" | "ready" | "review" | "setup";
const FILTERS: Array<[Filter, string]> = [["all", "All"], ["active", "Active"], ["ready", "Ready"], ["review", "Needs review"], ["setup", "Setup needed"]];

function matches(asset: AssetV1, filter: Filter): boolean {
  switch (filter) {
    case "active": return isActive(asset);
    case "ready": return asset.launchApproved && !isActive(asset);
    case "review": return asset.launchConfigured && !asset.launchApproved;
    case "setup": return !asset.launchConfigured;
    default: return true;
  }
}

const ACTION_FAILED: Record<AssetAction, string> = {
  start: "This asset could not be started.",
  stop: "This asset could not be stopped.",
  approve: "The launch trust decision could not be saved.",
  revoke: "The launch trust decision could not be saved.",
  recipe: "A launch recipe could not be generated.",
};

export function AssetLibrary() {
  const toast = useToast();
  const [assets, setAssets] = useState<AssetV1[]>([]);
  const [loading, setLoading] = useState(true);
  const [scanning, setScanning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("all");
  const [filter, setFilter] = useState<Filter>("all");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [playerId, setPlayerId] = useState<string | null>(null);
  const [busy, setBusy] = useState<{ id: string; action: AssetAction } | null>(null);

  const load = useCallback(async (quiet = false) => {
    if (!quiet) setLoading(true);
    try {
      setAssets(await listAssets());
      setError(null);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "The local asset catalog could not be loaded.");
    } finally {
      if (!quiet) setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);
  // Runtime state overlays the saved catalog, and it only moves while an
  // asset is running or on its way there; that is the only time to look.
  const moving = useMemo(() => assets.some(isActive), [assets]);
  usePoll(() => load(true), CATALOG_POLL_MS, moving);

  // ?player= opens the preview; a standalone tab gets the library behind it so Back works.
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get("player") && params.get("standalone") === "1") {
      const here = `${window.location.pathname}${window.location.search}`;
      window.history.replaceState(window.history.state, "", "/assets");
      window.history.pushState(window.history.state, "", here);
    }
    const sync = () => setPlayerId(new URLSearchParams(window.location.search).get("player"));
    sync();
    window.addEventListener("popstate", sync);
    return () => window.removeEventListener("popstate", sync);
  }, []);

  const selected = assets.find((asset) => asset.id === selectedId) ?? null;
  const player = assets.find((asset) => asset.id === playerId) ?? null;
  const categories = useMemo(() => Array.from(new Set(assets.map((asset) => asset.category).filter(Boolean))).sort(), [assets]);
  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return sortAssets(assets.filter((asset) =>
      (category === "all" || asset.category === category)
      && matches(asset, filter)
      && (!needle || [asset.name, asset.summary, asset.category, asset.framework, asset.entrypoint, ...asset.tags].some((value) => value.toLowerCase().includes(needle)))));
  }, [assets, category, filter, query]);
  const counts = useMemo(() => Object.fromEntries(FILTERS.map(([key]) => [key, assets.filter((asset) => matches(asset, key)).length])) as Record<Filter, number>, [assets]);
  const filtered = Boolean(query) || category !== "all" || filter !== "all";

  const closePlayer = () => {
    const params = new URLSearchParams(window.location.search);
    if (params.get("player") && params.get("standalone") === "1") window.history.back();
    else setPlayerId(null);
  };
  const open = (asset: AssetV1) => (isActive(asset) ? setPlayerId(asset.id) : setSelectedId(asset.id));

  const scan = async () => {
    setScanning(true);
    setError(null);
    try {
      const found = await scanAssets();
      setAssets(found);
      toast(`Scan complete: ${found.length} ${found.length === 1 ? "project" : "projects"} in the catalog.`);
    } catch (scanError) {
      setError(scanError instanceof Error ? scanError.message : "Metis could not scan the projects folder.");
    } finally {
      setScanning(false);
    }
  };

  const act = async (asset: AssetV1, action: AssetAction) => {
    if (busy) return;
    setBusy({ id: asset.id, action });
    setError(null);
    try {
      const call = { start: () => startAsset(asset.id, {}), stop: () => stopAsset(asset.id), approve: () => approveAsset(asset.id), revoke: () => revokeAssetApproval(asset.id), recipe: () => generateAssetRecipe(asset.id) }[action];
      const updated = await call();
      setAssets((current) => mergeAsset(current, updated));
      if (action === "start") { setPlayerId(updated.id); setSelectedId(null); }
      if (action === "stop" && playerId === updated.id) setPlayerId(null);
      void load(true);
    } catch (actionError) {
      setError(actionError instanceof Error ? actionError.message : ACTION_FAILED[action]);
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className={`assets${selected ? " has-drawer" : ""}`}>
      {player ? (
        <AssetPlayer asset={player} busy={busy?.id === player.id} onBack={closePlayer} onSettings={() => setSelectedId(player.id)} onStop={() => void act(player, "stop")} />
      ) : (
        <>
          <PageHeader
            eyebrow="Local projects"
            title="Assets"
            lede={`${assets.length} in the catalog · ${counts.active} active${assets.some(isFailed) ? ` · ${assets.filter(isFailed).length} failed` : ""}`}
            actions={<button type="button" className="ui-btn" onClick={() => void scan()} disabled={scanning}>{scanning ? "Scanning…" : "Scan for updates"}</button>}
          />
          <div className="assets-toolbar">
            <input type="search" className="assets-search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search assets" aria-label="Search assets" />
            <div className="assets-filters" role="group" aria-label="Launch state">
              {FILTERS.map(([key, label]) => (
                <button key={key} type="button" className={`ui-chip${filter === key ? " is-accent" : ""}`} aria-pressed={filter === key} onClick={() => setFilter(key)}>{label} <small>{counts[key]}</small></button>
              ))}
            </div>
            {categories.length > 1 ? (
              <select className="assets-category" value={category} onChange={(event) => setCategory(event.target.value)} aria-label="Category">
                <option value="all">All categories</option>
                {categories.map((item) => <option key={item} value={item}>{item}</option>)}
              </select>
            ) : null}
            {filtered ? <button type="button" className="ui-btn is-quiet is-sm" onClick={() => { setQuery(""); setCategory("all"); setFilter("all"); }}>Clear</button> : null}
          </div>
          {error ? <Notice kind="error" onDismiss={() => setError(null)}>{error}</Notice> : null}
          {loading ? <Skeleton rows={6} height={56} /> : visible.length ? (
            <AssetList assets={visible} busyId={busy?.action === "stop" ? busy.id : null} onOpen={open} onSettings={(asset) => setSelectedId(asset.id)} onStop={(asset) => void act(asset, "stop")} />
          ) : (
            <section className="assets-empty">
              <h2>{assets.length ? "No assets match" : "No projects discovered yet"}</h2>
              <p>{assets.length ? "Try another search, category or state." : "Scan the projects folder. Launchable projects also need a reviewed .metis/asset.json recipe."}</p>
              {assets.length ? <button type="button" className="ui-btn is-sm" onClick={() => { setQuery(""); setCategory("all"); setFilter("all"); }}>Clear filters</button> : <button type="button" className="ui-btn is-primary is-sm" onClick={() => void scan()} disabled={scanning}>{scanning ? "Scanning…" : "Scan for updates"}</button>}
            </section>
          )}
        </>
      )}

      {selected ? (
        <>
          <button type="button" className="assets-scrim" aria-label="Close asset details" onClick={() => setSelectedId(null)} />
          <AssetDrawer key={selected.id} asset={selected} busy={busy?.id === selected.id ? busy.action : null} onAction={(action) => void act(selected, action)} onUpdated={(updated) => setAssets((current) => mergeAsset(current, updated))} onClose={() => setSelectedId(null)} />
        </>
      ) : null}
    </div>
  );
}
