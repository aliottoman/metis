"use client";

// A running asset, full width, with its status and the way back.

import { ArrowLeft } from "lucide-react";

import { Status } from "@/components/ui/status";
import { isActive, isRunning, launchUrl, normalizedStatus, statusOf } from "@/lib/assets";
import type { AssetV1 } from "@/lib/types";

export function AssetPlayer({ asset, busy, onBack, onSettings, onStop }: {
  asset: AssetV1;
  busy: boolean;
  onBack: () => void;
  onSettings: () => void;
  onStop: () => void;
}) {
  const url = launchUrl(asset);
  const status = statusOf(asset);
  const stopping = normalizedStatus(asset) === "stopping";
  const live = isRunning(asset) && url;
  return (
    <section className="asset-player" aria-labelledby="asset-player-title">
      <header className="asset-player-bar">
        <button type="button" className="ui-btn is-quiet is-sm" onClick={onBack}><ArrowLeft size={14} aria-hidden="true" /> Assets</button>
        <h1 id="asset-player-title">{asset.name}</h1>
        <Status state={status.state} label={status.label} />
        <span className="asset-player-actions">
          <button type="button" className="ui-btn is-sm" onClick={onSettings}>Settings</button>
          {isActive(asset) ? <button type="button" className="ui-btn is-danger is-sm" disabled={busy || stopping} onClick={onStop}>{busy || stopping ? "Stopping…" : "Stop"}</button> : null}
          {live ? <a className="ui-btn is-quiet is-sm" href={`/assets?player=${encodeURIComponent(asset.id)}&standalone=1`} target="_blank" rel="noopener noreferrer">Open separately ↗</a> : null}
        </span>
      </header>
      {live ? (
        <iframe className="asset-player-frame" src={url} title={`${asset.name} live preview`} sandbox="allow-downloads allow-forms allow-modals allow-popups allow-popups-to-escape-sandbox allow-same-origin allow-scripts" allow="autoplay; camera; clipboard-read; clipboard-write; fullscreen; microphone" allowFullScreen />
      ) : (
        <div className="asset-player-wait" role="status" aria-live="polite">
          <h2>{stopping ? `Stopping ${asset.name}` : isRunning(asset) ? `Connecting to ${asset.name}` : isActive(asset) ? `Starting ${asset.name}` : `${asset.name} is not running`}</h2>
          <p>{stopping ? "Metis is closing the process and releasing its port." : isActive(asset) ? "The preview appears here as soon as the local app is ready." : "Open Settings to read the logs or start it again."}</p>
        </div>
      )}
    </section>
  );
}
