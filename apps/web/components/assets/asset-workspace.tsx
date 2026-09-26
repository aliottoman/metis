"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useState, type KeyboardEvent } from "react";
import { ArrowLeft, Maximize2, Minimize2, Pin, Play, Square, Monitor, Terminal, Settings2 } from "lucide-react";
import { useAssetCatalog } from "@/hooks/use-asset-catalog";
import { useAssetPreferences } from "@/hooks/use-asset-preferences";
import { assetHref, assetView, isActive, normalizedStatus, statusOf, type AssetView } from "@/lib/assets";
import { AssetDrawer } from "@/components/assets/asset-drawer";
import { AssetPlayer } from "@/components/assets/asset-player";
import { AssetLogs } from "@/components/assets/asset-logs";
import { Notice } from "@/components/ui/notice";
import { Skeleton } from "@/components/ui/skeleton";
import { Status } from "@/components/ui/status";

const VIEWS = [{ id: "preview", label: "App", icon: Monitor }, { id: "logs", label: "Logs", icon: Terminal }, { id: "settings", label: "Settings", icon: Settings2 }] as const;

export function AssetWorkspace({ assetId }: { assetId: string }) {
  const catalog = useAssetCatalog();
  const router = useRouter();
  const params = useSearchParams();
  const view = assetView(params.get("view"));
  const focused = params.get("focus") === "1";
  const { pinned, togglePin } = useAssetPreferences();
  const [backHref, setBackHref] = useState("/assets");
  const asset = catalog.assets.find((item) => item.id === assetId);
  const busy = catalog.busy?.id === assetId ? catalog.busy.action : null;
  useEffect(() => {
    try { const saved = sessionStorage.getItem("metis.assets.return"); if (saved && /^\/assets(?:\?|$)/.test(saved) && !saved.includes("player=")) setBackHref(saved); } catch { /* direct link */ }
  }, []);
  useEffect(() => { if (asset) document.title = `${asset.name} · Assets · Metis`; }, [asset, view, focused]);
  const selectView = (next: AssetView) => router.replace(`${assetHref(assetId, next)}${focused ? `${next === "preview" ? "?" : "&"}focus=1` : ""}`, { scroll: false });
  const keyTabs = (event: KeyboardEvent<HTMLDivElement>) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const current = VIEWS.findIndex((item) => item.id === view);
    const next = event.key === "Home" ? 0 : event.key === "End" ? VIEWS.length - 1 : (current + (event.key === "ArrowRight" ? 1 : -1) + VIEWS.length) % VIEWS.length;
    selectView(VIEWS[next].id);
    event.currentTarget.querySelector<HTMLButtonElement>(`#asset-tab-${VIEWS[next].id}`)?.focus();
  };
  if (!asset) return <div className="assets asset-missing"><Link href={backHref} className="ui-btn is-quiet"><ArrowLeft size={14} />All assets</Link>{catalog.loading && !catalog.loaded ? <Skeleton rows={5} height={64} /> : catalog.loadError ? <Notice kind="error" action="Try again" onAction={() => void catalog.load()}>{catalog.loadError}</Notice> : <section className="assets-empty"><h1>Asset not found</h1><p>This project may have moved. Update the catalog to look for it again.</p><button type="button" className="ui-btn" disabled={catalog.scanning} onClick={() => void catalog.scan()}>{catalog.scanning ? "Updating…" : "Update catalog"}</button>{catalog.error ? <Notice kind="error">{catalog.error}</Notice> : null}</section>}</div>;
  const active = isActive(asset) || busy === "start";
  const status = statusOf(asset);
  const stopping = busy === "stop" || normalizedStatus(asset) === "stopping";
  const toggleFocus = () => router.replace(`${assetHref(assetId, view)}${!focused ? `${view === "preview" ? "?" : "&"}focus=1` : ""}`, { scroll: false });

  return <section className="asset-workspace" aria-labelledby="asset-workspace-title">
    <header className="asset-workspace-header"><Link href={backHref} className="ui-btn is-quiet is-sm asset-back"><ArrowLeft size={15} /><span>Assets</span></Link><div className="asset-workspace-name"><span className="ui-eyebrow">{asset.category} · {asset.framework === "Unknown" ? "Local project" : asset.framework}</span><h1 id="asset-workspace-title">{asset.name}</h1></div><div className="asset-workspace-actions"><button type="button" className="ui-btn is-quiet is-sm asset-pin" aria-label={`${pinned.includes(asset.id) ? "Unpin" : "Pin"} ${asset.name}`} aria-pressed={pinned.includes(asset.id)} onClick={() => togglePin(asset.id)}><Pin size={15} /></button><button type="button" className="ui-btn is-quiet is-sm" onClick={toggleFocus} aria-label={focused ? "Exit focus mode" : "Focus app"} title={focused ? "Exit focus mode" : "Hide navigation for more space"}>{focused ? <Minimize2 size={16} /> : <Maximize2 size={16} />}</button>{active ? <button type="button" className="ui-btn is-sm is-danger" disabled={catalog.scanning || stopping || Boolean(catalog.busy && (catalog.busy.id !== asset.id || busy !== "start"))} onClick={() => void catalog.act(asset, "stop")}><Square size={12} />{stopping ? "Stopping…" : "Stop app"}</button> : asset.launchConfigured && asset.launchApproved ? <button type="button" className="ui-btn is-primary is-sm" disabled={catalog.busy !== null || catalog.scanning} onClick={() => void catalog.act(asset, "start")}><Play size={13} />Start app</button> : null}</div></header>
    <div className="asset-workspace-subbar"><div className="asset-workspace-tabs" role="tablist" aria-label="Asset workspace" onKeyDown={keyTabs}>{VIEWS.map(({ id, label, icon: Icon }) => <button key={id} id={`asset-tab-${id}`} role="tab" type="button" aria-selected={view === id} aria-controls={`asset-panel-${id}`} tabIndex={view === id ? 0 : -1} onClick={() => selectView(id)}><Icon size={14} />{label}</button>)}</div><Status state={busy === "start" ? "waiting" : status.state} label={busy === "start" ? "Starting" : status.label} /></div>
    {catalog.error || catalog.loadError ? <div className="asset-workspace-notice"><Notice kind="error" action={catalog.loadError ? "Refresh status" : "View logs"} onAction={() => catalog.loadError ? void catalog.load(true) : selectView("logs")} onDismiss={catalog.error ? catalog.dismissError : undefined}>{catalog.error || catalog.loadError}</Notice></div> : null}
    <div className="asset-workspace-panels">
      <div role="tabpanel" id="asset-panel-preview" aria-labelledby="asset-tab-preview" hidden={view !== "preview"} tabIndex={0}><AssetPlayer asset={asset} starting={busy === "start"} disabled={catalog.busy !== null || catalog.scanning} onStart={() => void catalog.act(asset, "start")} onSettings={() => selectView("settings")} onLogs={() => selectView("logs")} /></div>
      <div role="tabpanel" id="asset-panel-logs" aria-labelledby="asset-tab-logs" hidden={view !== "logs"} tabIndex={0}><AssetLogs key={asset.id} asset={asset} visible={view === "logs"} /></div>
      <div role="tabpanel" id="asset-panel-settings" aria-labelledby="asset-tab-settings" hidden={view !== "settings"} tabIndex={0}><div className="asset-settings-intro"><h2>Launch & configuration</h2><p>{asset.summary}</p></div><AssetDrawer key={asset.id} asset={asset} embedded busy={catalog.busy?.action ?? null} onAction={(action) => void catalog.act(asset, action)} onUpdated={catalog.update} onClose={() => selectView("preview")} /></div>
    </div>
  </section>;
}
