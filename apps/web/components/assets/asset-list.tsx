"use client";

// The catalog as rows, running assets first. Arrow keys walk it; Enter opens.

import { useState, type KeyboardEvent } from "react";

import { Status } from "@/components/ui/status";
import { isActive, isRunning, statusOf, tagsOf } from "@/lib/assets";
import type { AssetV1 } from "@/lib/types";

/** Two letters from the name, skipping emoji and punctuation. */
function initials(name: string): string {
  const words = name.replace(/[^\p{L}\p{N}\s]/gu, " ").trim().split(/\s+/).filter(Boolean);
  return (words.length > 1 ? `${words[0]![0]}${words[1]![0]}` : (words[0] ?? "?").slice(0, 2)).toUpperCase();
}

export function AssetList({ assets, busyId, onOpen, onSettings, onStop }: {
  assets: AssetV1[];
  busyId: string | null;
  onOpen: (asset: AssetV1) => void;
  onSettings: (asset: AssetV1) => void;
  onStop: (asset: AssetV1) => void;
}) {
  const [cursor, setCursor] = useState(0);

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (!assets.length) return;
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      const next = Math.max(0, Math.min(assets.length - 1, cursor + (event.key === "ArrowDown" ? 1 : -1)));
      setCursor(next);
      event.currentTarget.querySelector<HTMLElement>(`[data-row="${next}"]`)?.scrollIntoView({ block: "nearest" });
    } else if (event.key === "Enter" && assets[cursor]) {
      onOpen(assets[cursor]);
    }
  };

  return (
    <div className="asset-rows" role="list" tabIndex={0} onKeyDown={onKeyDown} aria-label="Asset catalog">
      {assets.map((asset, index) => {
        const active = isActive(asset);
        const status = statusOf(asset);
        return (
          <div key={asset.id} role="listitem" data-row={index} className={`asset-row${cursor === index ? " is-cursor" : ""}${active ? " is-active" : ""}`} onPointerEnter={() => setCursor(index)}>
            <button type="button" className="asset-row-main" onClick={() => onOpen(asset)} aria-label={active ? `Open ${asset.name}` : `Configure ${asset.name}`}>
              <span className="asset-mark" aria-hidden="true">{initials(asset.name)}</span>
              <span className="asset-row-name">
                <strong>{asset.name}</strong>
                <small>{asset.summary || asset.entrypoint || "Project"}</small>
              </span>
              <span className="asset-row-tags">{tagsOf(asset, 3).map((tag) => <span key={tag} className="ui-chip">{tag}</span>)}</span>
              <Status state={status.state} label={status.label} />
            </button>
            <span className="asset-row-actions">
              {active ? (
                <button type="button" className="ui-btn is-quiet is-sm" disabled={busyId === asset.id || !isRunning(asset)} onClick={() => onStop(asset)}>{busyId === asset.id ? "Stopping…" : "Stop"}</button>
              ) : null}
              <button type="button" className="ui-btn is-quiet is-sm" onClick={() => onSettings(asset)}>Settings</button>
            </span>
          </div>
        );
      })}
    </div>
  );
}
