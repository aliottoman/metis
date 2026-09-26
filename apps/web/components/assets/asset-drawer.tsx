"use client";

// One asset's settings: the launch recipe and its trust, the .env, start or
// stop, and the process output. Mount it with key={asset.id} so every draft
// and log belongs to the asset shown.

import { useCallback, useEffect, useRef, useState } from "react";
import { X } from "lucide-react";

import { Notice } from "@/components/ui/notice";
import { Status } from "@/components/ui/status";
import { getAssetLogs, saveAssetEnv } from "@/lib/api";
import { commandLabel, isActive, isFailed, normalizedStatus, statusOf, tagsOf, type AssetAction } from "@/lib/assets";
import type { AssetV1 } from "@/lib/types";
import { usePoll } from "@/hooks/use-poll";
import { useDialogFocus } from "@/hooks/use-dialog-focus";

const LOG_POLL_MS = 3_000;

export function AssetDrawer({ asset, busy, error, onDismissError, onAction, onUpdated, onClose, embedded = false }: {
  asset: AssetV1;
  busy: AssetAction | null;
  error?: string | null;
  onDismissError?: () => void;
  onAction: (action: AssetAction) => void;
  onUpdated: (asset: AssetV1) => void;
  onClose: () => void;
  embedded?: boolean;
}) {
  const active = isActive(asset);
  const status = statusOf(asset);
  const [env, setEnv] = useState<Record<string, string>>({});
  const [revealed, setRevealed] = useState<Record<string, boolean>>({});
  const [envBusy, setEnvBusy] = useState(false);
  const [envNote, setEnvNote] = useState<{ kind: "error" | "success"; text: string } | null>(null);
  const [logs, setLogs] = useState<string | null>(null);
  const [logsError, setLogsError] = useState<string | null>(null);
  const [logsOpen, setLogsOpen] = useState(isFailed(asset));
  const drawer = useRef<HTMLElement>(null);
  useDialogFocus(drawer, !embedded, onClose);

  // Logs load once opened and keep refreshing while the process is alive.
  const loadLogs = useCallback(() => getAssetLogs(asset.id)
    .then((result) => { setLogs(result.logs); setLogsError(null); })
    .catch((error) => setLogsError(error instanceof Error ? error.message : "Runtime output is not available yet.")), [asset.id]);
  useEffect(() => { if (!embedded && logsOpen) void loadLogs(); }, [embedded, loadLogs, logsOpen]);
  usePoll(loadLogs, LOG_POLL_MS, !embedded && logsOpen && active);

  // Only typed values are sent: a blank field means "keep what is on disk".
  const pending = Object.fromEntries(Object.entries(env).filter(([, value]) => value.length > 0));
  const saveEnv = async () => {
    if (!Object.keys(pending).length || envBusy) return;
    setEnvBusy(true);
    setEnvNote(null);
    try {
      onUpdated(await saveAssetEnv(asset.id, pending));
      setEnv({});
      setRevealed({});
      const count = Object.keys(pending).length;
      setEnvNote({ kind: "success", text: `Saved ${count} ${count === 1 ? "variable" : "variables"} to .env.` });
    } catch (error) {
      setEnvNote({ kind: "error", text: error instanceof Error ? error.message : "The project .env file could not be updated." });
    } finally {
      setEnvBusy(false);
    }
  };

  const approve = () => {
    const build = asset.buildCommand.length ? `\n\nBuilds before launch (also on your Mac):\n${asset.buildCommand.map(commandLabel).join("\n")}` : "";
    if (window.confirm(`Trust this exact launch recipe for ${asset.name}?\n\n${commandLabel(asset.launchCommand)}${build}\n\nThis starts project code directly on your Mac with your user account's filesystem and network access. Any recipe change will require approval again.`)) onAction("approve");
  };

  return (
    <aside ref={drawer} className={embedded ? "asset-settings" : "asset-drawer"} role={embedded ? "region" : "dialog"} aria-modal={embedded ? undefined : true} aria-label={embedded ? "Asset settings" : undefined} aria-labelledby={embedded ? undefined : "asset-drawer-title"}>
      {!embedded ? <header className="asset-drawer-head">
        <div>
          <span className="ui-eyebrow">{[asset.category, asset.framework].filter(Boolean).join(" · ")}</span>
          <h2 id="asset-drawer-title">{asset.name}</h2>
          {asset.summary ? <p>{asset.summary}</p> : null}
          <div className="asset-drawer-tags">{tagsOf(asset).map((tag) => <span key={tag} className="ui-chip">{tag}</span>)}</div>
        </div>
        <button type="button" className="ui-btn is-quiet is-sm" aria-label="Close asset details" onClick={onClose}><X size={16} /></button>
      </header> : null}

      <div className="asset-drawer-body">
        {error ? <Notice kind="error" onDismiss={onDismissError}>{error}</Notice> : null}
        {!embedded ? <div className="asset-runtime">
          <Status state={status.state} label={status.label} />
          <code>{asset.entrypoint || "Entrypoint not reported"}</code>
          {asset.launchApproved ? (
            active ? (
              <button type="button" className="ui-btn is-danger is-sm" disabled={busy != null || normalizedStatus(asset) === "stopping"} onClick={() => onAction("stop")}>{busy === "stop" || normalizedStatus(asset) === "stopping" ? "Stopping…" : "Stop"}</button>
            ) : (
              <button type="button" className="ui-btn is-primary is-sm" disabled={busy != null} onClick={() => onAction("start")}>{busy === "start" ? "Starting…" : "Start"}</button>
            )
          ) : null}
        </div> : null}

        {!asset.launchConfigured ? (
          <Notice kind="info" title="A reviewed launch recipe is required" action={busy === "recipe" ? "Drafting…" : "Generate recipe"} onAction={() => busy == null && onAction("recipe")}>
            Metis found this project but will not guess how to run it. Your selected model can draft <code>.metis/asset.json</code> from the project&rsquo;s own files; the command still needs your review before anything runs.
          </Notice>
        ) : null}

        {asset.launchConfigured && !asset.launchApproved ? (
          <section className="asset-recipe">
            <strong>Review this host launch recipe</strong>
            <p>It runs directly as your macOS user with access to your files and network. Approval is remembered for this exact recipe only.</p>
            <pre>{commandLabel(asset.launchCommand)}</pre>
            {asset.buildCommand.length ? <><small>Builds before launch:</small><pre>{asset.buildCommand.map(commandLabel).join("\n")}</pre></> : null}
            <button type="button" className="ui-btn is-primary is-sm" disabled={busy != null} onClick={approve}>{busy === "approve" ? "Saving trust…" : "Trust this exact recipe"}</button>
          </section>
        ) : null}

        {asset.launchApproved ? (
          <div className="asset-trusted">
            <span><strong>Exact recipe trusted.</strong> A manifest change requires review again.</span>
            <button type="button" className="ui-btn is-quiet is-sm" disabled={busy != null || active} title={active ? "Stop the asset before revoking trust" : "Require review before the next launch"} onClick={() => onAction("revoke")}>{busy === "revoke" ? "Revoking…" : "Revoke"}</button>
          </div>
        ) : null}

        {embedded && asset.launchApproved ? <details className="asset-section">
          <summary><span>Launch recipe</span><small>Trusted command</small></summary>
          <div className="asset-recipe"><strong>Launch command</strong><pre>{commandLabel(asset.launchCommand)}</pre>{asset.buildCommand.length ? <><strong>Builds before launch</strong><pre>{asset.buildCommand.map(commandLabel).join("\n")}</pre></> : null}<small>Entrypoint: {asset.entrypoint || "Not reported"}</small></div>
        </details> : null}

        <details className="asset-section" open={!active}>
          <summary><span>.env</span><small>{asset.envFilePresent ? `${asset.envFile.length} ${asset.envFile.length === 1 ? "variable" : "variables"}` : "No file"}</small></summary>
          {asset.envFilePresent ? (
            <div className="asset-env">
              <p>Metis reports only whether a value is set and never reads one back. Typing a value replaces it in the file.</p>
              {asset.envFile.map(({ key, isSet, sensitive }) => (
                <label key={key} className="ui-field asset-env-field">
                  <span>{key} · {isSet ? "set" : "empty"}{sensitive ? " · sensitive" : ""}</span>
                  <span className="asset-env-control">
                    <input type={sensitive && !revealed[key] ? "password" : "text"} disabled={envBusy} value={env[key] ?? ""} onChange={(event) => { setEnvNote(null); setEnv((current) => ({ ...current, [key]: event.target.value })); }} autoComplete="off" spellCheck={false} placeholder={isSet ? "Value set — type to replace" : `Enter ${key}`} />
                    {sensitive ? <button type="button" className="ui-btn is-quiet is-sm" aria-pressed={Boolean(revealed[key])} onClick={() => setRevealed((current) => ({ ...current, [key]: !current[key] }))}>{revealed[key] ? "Hide" : "Show"}</button> : null}
                  </span>
                </label>
              ))}
              {envNote ? <Notice kind={envNote.kind}>{envNote.text}</Notice> : null}
              <div className="asset-env-actions">
                <button type="button" className="ui-btn is-primary is-sm" disabled={envBusy || !Object.keys(pending).length} onClick={() => void saveEnv()}>{envBusy ? "Saving…" : "Save to .env"}</button>
                {active ? <small>Saved values take effect on the next start.</small> : null}
              </div>
            </div>
          ) : (
            <p className="asset-env">This project has no <code>.env</code> file. Add one in the project folder and it appears here.</p>
          )}
        </details>

        {!embedded ? <details className="asset-section" open={logsOpen} onToggle={(event) => setLogsOpen(event.currentTarget.open)}>
          <summary><span>Logs</span><small>{active ? "live" : "process output"}</small></summary>
          {logsError ? <Notice kind="error">{logsError}</Notice> : null}
          <pre className="asset-logs" aria-live="polite">{logs ?? (logsError ? "Runtime output unavailable." : "Loading runtime output…")}</pre>
        </details> : null}
      </div>
    </aside>
  );
}
