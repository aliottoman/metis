"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import {
  getLocalModelSession,
  launchLocalModel,
  stopLocalModel,
} from "@/lib/api";
import type { LocalModelSession } from "@/lib/types";

const IDLE_OPTIONS: Array<[LocalModelSession["idle_timeout_seconds"], string]> = [
  [60, "1 min"],
  [300, "5 min"],
  [900, "15 min"],
  [1800, "30 min"],
  [86400, "Until stopped"],
];

const CONTEXT_OPTIONS: Array<[LocalModelSession["context_window"], string]> = [
  [32768, "32K · recommended"],
  [65536, "64K · more memory"],
  [131072, "128K · heavy"],
];

function stateLabel(session: LocalModelSession | null, now: number): string {
  if (!session) return "Checking model";
  if (session.state === "loading") return "Loading model";
  if (session.state === "busy") return "Model busy";
  if (session.state === "ready") {
    if (session.idle_timeout_seconds >= 86400) return "Ready · until stopped";
    if (session.expires_at) {
      const seconds = Math.max(
        0, Math.ceil((new Date(session.expires_at).getTime() - now) / 1000),
      );
      return `Ready · ${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
    }
    return session.selected_model || "Model ready";
  }
  if (session.state === "error") return "Ollama unavailable";
  return "Launch model";
}

export function ModelSessionControl() {
  const [session, setSession] = useState<LocalModelSession | null>(null);
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [model, setModel] = useState("");
  const [idle, setIdle] = useState<LocalModelSession["idle_timeout_seconds"]>(300);
  const [context, setContext] = useState<LocalModelSession["context_window"]>(32768);
  const [now, setNow] = useState(() => Date.now());

  const apply = useCallback((next: LocalModelSession) => {
    setSession(next);
    setModel((current) => current || next.selected_model || next.models[0]?.id || "");
    setIdle(next.idle_timeout_seconds);
    setContext(next.context_window);
    window.dispatchEvent(new CustomEvent("metis:model-session", { detail: next }));
  }, []);

  const refresh = useCallback(async () => {
    try {
      apply(await getLocalModelSession());
      setError(null);
    } catch (refreshError) {
      setError(refreshError instanceof Error ? refreshError.message : "Model status unavailable.");
    }
  }, [apply]);

  useEffect(() => {
    void refresh();
    const interval = window.setInterval(() => void refresh(), open ? 3000 : 10000);
    return () => window.clearInterval(interval);
  }, [open, refresh]);

  useEffect(() => {
    if (session?.state !== "ready" || !session.expires_at) return;
    const interval = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(interval);
  }, [session?.expires_at, session?.state]);

  async function launch() {
    if (!model || busy) return;
    setBusy(true);
    setError(null);
    setSession((current) => current ? { ...current, state: "loading", error: null } : current);
    try {
      apply(await launchLocalModel(model, idle, context));
      setOpen(false);
    } catch (launchError) {
      setError(launchError instanceof Error ? launchError.message : "The model could not be launched.");
    } finally {
      setBusy(false);
    }
  }

  async function stop() {
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      apply(await stopLocalModel(true));
    } catch (stopError) {
      setError(stopError instanceof Error ? stopError.message : "The model could not be stopped.");
    } finally {
      setBusy(false);
    }
  }

  const warning = useMemo(() => {
    if (context === 131072) return "128K can create sustained memory pressure and heat. Use it only when a task truly needs it.";
    if (context === 65536) return "64K uses noticeably more unified memory than the recommended 32K setting.";
    if (idle >= 1800) return "A longer warm period uses more battery. Five minutes is the laptop-safe default.";
    return "32K and a 5-minute idle window are recommended for everyday use.";
  }, [context, idle]);

  return (
    <div className="modelSessionDock">
      <button
        type="button"
        className={`modelSessionTrigger state-${session?.state ?? "off"}`}
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
      >
        <i aria-hidden="true" />
        <span>{stateLabel(session, now)}</span>
        <b aria-hidden="true">⌄</b>
      </button>
      {open ? (
        <section className="modelSessionPanel" aria-label="Local model session">
          <header>
            <div>
              <span className="eyebrow">On-device session</span>
              <strong>{session?.state === "ready" ? "Ready for local work" : "Choose what to launch"}</strong>
            </div>
            <button type="button" onClick={() => setOpen(false)} aria-label="Close model controls">×</button>
          </header>
          <label>
            <span>Installed model</span>
            <select value={model} onChange={(event) => setModel(event.target.value)} disabled={busy}>
              {!session?.models.length ? <option value="">No installed models found</option> : null}
              {session?.models.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.name}{item.parameter_size ? ` · ${item.parameter_size}` : ""}
                </option>
              ))}
            </select>
          </label>
          <div className="modelSessionRow">
            <label>
              <span>Unload after idle</span>
              <select value={idle} onChange={(event) => setIdle(Number(event.target.value) as LocalModelSession["idle_timeout_seconds"])} disabled={busy}>
                {IDLE_OPTIONS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
              </select>
            </label>
            <label>
              <span>Context</span>
              <select value={context} onChange={(event) => setContext(Number(event.target.value) as LocalModelSession["context_window"])} disabled={busy}>
                {CONTEXT_OPTIONS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
              </select>
            </label>
          </div>
          <p className="modelSessionAdvice">{warning}</p>
          {session?.state === "error" || error ? (
            <p className="modelSessionError" role="alert">{error || session?.error}</p>
          ) : null}
          <footer>
            {session?.state === "ready" || session?.state === "busy" ? (
              <button className="secondaryButton" type="button" onClick={() => void stop()} disabled={busy || session.state === "busy"}>
                Stop model
              </button>
            ) : <span />}
            <button className="primaryButton" type="button" onClick={() => void launch()} disabled={busy || !model}>
              {busy ? "Working…" : session?.state === "ready" && model === session.selected_model ? "Relaunch with settings" : `Launch ${model || "model"}`}
            </button>
          </footer>
        </section>
      ) : null}
    </div>
  );
}
