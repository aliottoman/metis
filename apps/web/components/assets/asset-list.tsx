"use client";

import Link from "next/link";
import { ArrowUpRight, Pin, Play, Settings2 } from "lucide-react";
import { Status } from "@/components/ui/status";
import { assetHref, isActive, isFailed, statusOf, type AssetAction } from "@/lib/assets";
import type { AssetV1 } from "@/lib/types";

export function assetInitials(name: string): string {
  const words = name.replace(/[^\p{L}\p{N}\s]/gu, " ").trim().split(/\s+/).filter(Boolean);
  return (words.length > 1 ? `${words[0]![0]}${words[1]![0]}` : (words[0] ?? "?").slice(0, 2)).toUpperCase();
}

export function AssetList({ assets, layout, busy, disabled = false, pinned, onPin, onStart }: {
  assets: AssetV1[];
  layout: "grid" | "list" | "compact";
  busy: { id: string; action: AssetAction } | null;
  disabled?: boolean;
  pinned: string[];
  onPin: (id: string) => void;
  onStart: (asset: AssetV1) => void;
}) {
  return <div className={`asset-collection is-${layout}`} role="list" aria-label={layout === "compact" ? "Pinned assets" : "Asset catalog"}>
    {assets.map((asset) => {
      const active = isActive(asset);
      const status = statusOf(asset);
      const trusted = asset.launchConfigured && asset.launchApproved;
      const isPinned = pinned.includes(asset.id);
      const working = busy?.id === asset.id;
      return <article key={asset.id} className={`asset-card${active ? " is-active" : ""}`} role="listitem">
        <div className="asset-card-top"><span className="asset-mark" aria-hidden="true">{assetInitials(asset.name)}</span><span className="asset-card-category">{asset.category}</span><button type="button" className={`asset-pin ui-btn is-quiet is-sm${isPinned ? " is-pinned" : ""}`} aria-label={`${isPinned ? "Unpin" : "Pin"} ${asset.name}`} aria-pressed={isPinned} onClick={() => onPin(asset.id)}><Pin size={15} aria-hidden="true" /></button></div>
        <Link className="asset-card-main" href={assetHref(asset.id)} aria-label={`Open ${asset.name}`}><h3>{asset.name}</h3><p>{asset.summary || "Open this local project to see its launch settings and preview."}</p></Link>
        <div className="asset-card-meta"><Status state={working && busy.action === "start" ? "waiting" : status.state} label={working && busy.action === "start" ? "Starting" : status.label} /><span>{asset.framework !== "Unknown" ? asset.framework : "Local project"}</span></div>
        <div className="asset-card-foot"><Link href={assetHref(asset.id, "settings")} className="ui-btn is-quiet is-sm" aria-label={`Settings for ${asset.name}`}><Settings2 size={14} aria-hidden="true" /><span>Settings</span></Link>{trusted && !active ? <button type="button" className="ui-btn is-sm" disabled={disabled || busy !== null} onClick={() => onStart(asset)} aria-label={`${isFailed(asset) ? "Retry" : "Start"} ${asset.name}`}><Play size={13} aria-hidden="true" />{working ? "Starting…" : isFailed(asset) ? "Retry launch" : "Start & open"}</button> : <Link className={`ui-btn is-sm${active ? " is-primary" : ""}`} href={assetHref(asset.id, active ? "preview" : "settings")}>{active ? "Open app" : asset.launchConfigured ? "Review setup" : "Set up"}<ArrowUpRight size={14} aria-hidden="true" /></Link>}</div>
      </article>;
    })}
  </div>;
}
