"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Grid2X2, List, Search, RefreshCw, Pin, Package } from "lucide-react";
import { AssetList } from "@/components/assets/asset-list";
import { Notice } from "@/components/ui/notice";
import { PageHeader } from "@/components/ui/page-header";
import { Skeleton } from "@/components/ui/skeleton";
import { useToast } from "@/components/ui/toast";
import { useAssetCatalog } from "@/hooks/use-asset-catalog";
import { useAssetPreferences } from "@/hooks/use-asset-preferences";
import { assetHref, matchesAsset, sortAssets, type AssetFilter } from "@/lib/assets";
import type { AssetV1 } from "@/lib/types";

const FILTERS: Array<[AssetFilter, string]> = [["all", "All assets"], ["pinned", "Pinned"], ["active", "Running"], ["ready", "Ready"], ["failed", "Failed"], ["review", "Needs review"], ["setup", "Setup needed"]];

export function AssetLibrary() {
  const router = useRouter();
  const params = useSearchParams();
  const catalog = useAssetCatalog();
  const { assets, loading, loaded, loadError, error, scanning, busy } = catalog;
  const { pinned, togglePin } = useAssetPreferences();
  const toast = useToast();
  const query = params.get("q") ?? "";
  const category = params.get("category") ?? "all";
  const rawFilter = params.get("state");
  const filter: AssetFilter = FILTERS.some(([key]) => key === rawFilter) ? rawFilter as AssetFilter : "all";
  const [layout, setLayout] = useState<"grid" | "list">("grid");
  useEffect(() => { try { if (localStorage.getItem("metis.assets.layout") === "list") setLayout("list"); } catch { /* defaults */ } }, []);
  const changeLayout = (value: "grid" | "list") => { setLayout(value); try { localStorage.setItem("metis.assets.layout", value); } catch { /* current visit */ } };
  const updateFilter = (key: string, value: string) => {
    const next = new URLSearchParams(params.toString());
    if (!value || value === "all") next.delete(key); else next.set(key, value);
    window.history.replaceState(null, "", `/assets${next.size ? `?${next}` : ""}`);
  };
  // Old bookmarked players now resolve to the same durable workspace as catalog links.
  const legacyPlayer = params.get("player");
  useEffect(() => { if (legacyPlayer) router.replace(assetHref(legacyPlayer)); }, [legacyPlayer, router]);
  useEffect(() => { try { sessionStorage.setItem("metis.assets.return", `/assets${params.size ? `?${params}` : ""}`); } catch { /* direct entry has a safe fallback */ } }, [params]);

  const categories = useMemo(() => Array.from(new Set(assets.map((asset) => asset.category).filter(Boolean))).sort(), [assets]);
  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return sortAssets(assets.filter((asset) => (category === "all" || asset.category === category)
      && matchesAsset(asset, filter, pinned)
      && (!needle || [asset.name, asset.summary, asset.category, asset.framework, asset.entrypoint, ...asset.tags].some((value) => value.toLowerCase().includes(needle)))));
  }, [assets, category, filter, query, pinned]);
  const counts = Object.fromEntries(FILTERS.map(([key]) => [key, assets.filter((asset) => matchesAsset(asset, key, pinned)).length]));
  const favorites = sortAssets(assets.filter((asset) => pinned.includes(asset.id)));
  const filtered = Boolean(query) || category !== "all" || filter !== "all";
  const scan = async () => { const count = await catalog.scan(); if (count !== null) toast(`Catalog updated · ${count} ${count === 1 ? "asset" : "assets"}`); };
  const start = (asset: AssetV1) => { void catalog.act(asset, "start"); router.push(assetHref(asset.id)); };
  const listProps = { busy, disabled: scanning, pinned, onPin: togglePin, onStart: (asset: AssetV1) => void start(asset) };

  return (
    <div className="assets asset-library">
      <PageHeader icon={<Package />} eyebrow="Your launchpad" title="Assets" lede="Your tools, demos, and projects. Ready when you are."
        actions={<button type="button" className="ui-btn" onClick={() => void scan()} disabled={scanning || loading || busy !== null}><RefreshCw size={14} className={scanning ? "asset-spin" : ""} aria-hidden="true" />{scanning ? "Updating…" : "Update catalog"}</button>} />
      {error || loadError ? <Notice kind="error" action={loadError ? "Try again" : undefined} onAction={() => void catalog.load()} onDismiss={error ? catalog.dismissError : undefined}>{error || loadError}</Notice> : null}
      {!filtered && favorites.length > 0 ? <section className="asset-pinned" aria-labelledby="asset-pinned-title">
        <div className="asset-section-heading"><h2 id="asset-pinned-title"><Pin size={14} aria-hidden="true" /> Pinned for quick access</h2><span>{favorites.length}</span></div>
        <AssetList assets={favorites} layout="compact" {...listProps} />
      </section> : null}
      <section className="asset-catalog" aria-label="Asset library">
        <div className="asset-catalog-tools">
          <label className="asset-search-field"><Search size={17} aria-hidden="true" /><input type="search" value={query} onChange={(event) => updateFilter("q", event.target.value)} placeholder="Find an asset by name, purpose, or technology…" aria-label="Search assets" /></label>
          {categories.length > 1 ? <select className="assets-category" value={category} onChange={(event) => updateFilter("category", event.target.value)} aria-label="Category"><option value="all">All categories</option>{categories.map((item) => <option key={item} value={item}>{item}</option>)}</select> : null}
          <div className="asset-layout-switch" role="group" aria-label="Catalog layout">
            <button className="ui-btn is-quiet is-sm" type="button" aria-label="Grid view" aria-pressed={layout === "grid"} onClick={() => changeLayout("grid")}><Grid2X2 size={16} /></button>
            <button className="ui-btn is-quiet is-sm" type="button" aria-label="List view" aria-pressed={layout === "list"} onClick={() => changeLayout("list")}><List size={16} /></button>
          </div>
        </div>
        <div className="assets-filters" role="group" aria-label="Launch state">{FILTERS.filter(([key]) => key !== "failed" || counts.failed > 0).map(([key, label]) => <button key={key} type="button" className={`ui-chip${filter === key ? " is-accent" : ""}`} aria-pressed={filter === key} onClick={() => updateFilter("state", key)}>{label}<small>{loaded ? counts[key] : "—"}</small></button>)}</div>
        <div className="asset-results-label"><span role="status" aria-live="polite">{!loaded ? loading ? "Loading your catalog…" : "Catalog unavailable" : `${visible.length} ${visible.length === 1 ? "asset" : "assets"}${filtered ? ` of ${assets.length}` : " in your library"}`}</span>{filtered ? <button type="button" className="ui-btn is-quiet is-sm" onClick={() => router.replace("/assets", { scroll: false })}>Clear filters</button> : <span>Pin the ones you use most</span>}</div>
        {loading && !loaded ? <Skeleton rows={6} height={72} /> : visible.length ? <AssetList assets={visible} layout={layout} {...listProps} /> : loaded ? <section className="assets-empty"><Package size={32} aria-hidden="true" /><h2>{assets.length ? filter === "pinned" ? "Keep your favorites close" : "No matching assets" : "Your launchpad starts here"}</h2><p>{assets.length ? filter === "pinned" ? "Use the pin on any asset to keep it within reach." : "Try a different search, category, or launch state." : "Update the catalog to discover projects in your configured projects folder."}</p>{assets.length ? <button type="button" className="ui-btn" onClick={() => router.replace("/assets", { scroll: false })}>Show all assets</button> : <button type="button" className="ui-btn is-primary" onClick={() => void scan()} disabled={scanning}>{scanning ? "Updating…" : "Discover assets"}</button>}</section> : null}
      </section>
    </div>
  );
}
