"use client";

import { useEffect, useState } from "react";
import { ArrowUpRight, Check, Circle, LoaderCircle, Package, Play, RotateCw, TriangleAlert } from "lucide-react";
import { isActive, isFailed, isRunning, launchUrl, normalizedStatus } from "@/lib/assets";
import type { AssetV1 } from "@/lib/types";

export function AssetPlayer({ asset, starting, disabled, onStart, onSettings, onLogs }: {
  asset: AssetV1; starting: boolean; disabled: boolean;
  onStart: () => void; onSettings: () => void; onLogs: () => void;
}) {
  const url = launchUrl(asset);
  const live = Boolean(isRunning(asset) && url);
  const active = isActive(asset) || starting;
  const failed = isFailed(asset) && !starting;
  const stopping = normalizedStatus(asset) === "stopping";
  const [revision, setRevision] = useState(0);
  const [loaded, setLoaded] = useState(false);
  const [slow, setSlow] = useState(false);
  useEffect(() => {
    setLoaded(false); setSlow(false);
    if (!live) return;
    const timer = window.setTimeout(() => setSlow(true), 12_000);
    return () => window.clearTimeout(timer);
  }, [asset.id, url, revision, live]);

  if (live && url) return <div className="asset-preview">
    <div className="asset-preview-toolbar"><span className="asset-preview-address" title={url}><span className="asset-live-dot" />{url}</span><button type="button" className="ui-btn is-quiet is-sm" onClick={() => setRevision((value) => value + 1)} aria-label="Reload preview"><RotateCw size={14} aria-hidden="true" />Reload</button><a className="ui-btn is-quiet is-sm" href={url} target="_blank" rel="noopener noreferrer">Open in browser<ArrowUpRight size={14} aria-hidden="true" /></a></div>
    {!loaded && slow ? <div className="asset-preview-hint" role="status">The preview is taking longer than expected. Try reloading, check Logs, or open it in your browser.</div> : null}
    <iframe key={`${asset.id}-${url}-${revision}`} className="asset-player-frame" src={url} title={`${asset.name} live preview`} onLoad={() => setLoaded(true)} onError={() => { setLoaded(false); setSlow(true); }} sandbox="allow-downloads allow-forms allow-modals allow-popups allow-popups-to-escape-sandbox allow-same-origin allow-scripts" allow="autoplay; camera; clipboard-read; clipboard-write; fullscreen; microphone" allowFullScreen />
    <div className="asset-preview-caption"><span>Running on your Mac · leaving this page keeps it running.</span><span>Blank preview? <a href={url} target="_blank" rel="noopener noreferrer">Open in browser ↗</a></span></div>
  </div>;

  const ready = asset.launchConfigured && asset.launchApproved;
  const steps = [{ label: "Launch recipe", done: asset.launchConfigured }, { label: "Recipe trusted", done: asset.launchApproved }, { label: "App running", done: isRunning(asset) }];
  return <div className="asset-launch-stage" aria-live="polite">
    <div className={`asset-launch-symbol${failed ? " is-failed" : ""}`}>{failed ? <TriangleAlert size={30} /> : active ? <LoaderCircle size={30} className="asset-spin" /> : <Package size={30} />}</div>
    <span className="ui-eyebrow">{failed ? "Let’s get it running" : active ? "Local runtime" : "Your asset workspace"}</span>
    <h2>{failed ? "Launch needs attention" : stopping ? "Stopping the app…" : active ? "Getting your app ready…" : !asset.launchConfigured ? "Set up this asset" : !asset.launchApproved ? "Review once. Launch when you need it." : "Ready to launch"}</h2>
    <p>{failed ? "The process stopped before it was ready. Its logs can help explain what happened." : stopping ? "Metis is closing the process and releasing its port." : active ? asset.buildCommand.length ? "Building the project and starting its local server. Follow the output in Logs." : "Starting the local server. The preview will appear here when it is ready." : !asset.launchConfigured ? "Add a launch recipe in Settings so Metis knows how to run this project." : !asset.launchApproved ? "Review the exact command in Settings before running this project on your Mac." : asset.summary || "Start the app to use it here. Your settings and logs stay one tab away."}</p>
    <ol className="asset-launch-steps">{steps.map((step) => <li key={step.label} className={step.done ? "is-done" : ""}>{step.done ? <Check size={14} /> : <Circle size={12} />}{step.label}</li>)}</ol>
    <div className="asset-stage-actions">{!active ? ready ? <button type="button" className="ui-btn is-primary" disabled={disabled} onClick={onStart}><Play size={15} />{failed ? "Retry launch" : "Start app"}</button> : <button type="button" className="ui-btn is-primary" disabled={disabled} onClick={onSettings}>{asset.launchConfigured ? "Review launch recipe" : "Open settings"}</button> : null}<button type="button" className="ui-btn" onClick={onLogs}>View logs</button></div>
  </div>;
}
