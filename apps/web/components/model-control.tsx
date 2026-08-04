"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  getLocalModelSession,
  launchLocalModel,
  stopLocalModel,
} from "@/lib/api";
import type {
  LocalModelSession,
  ModelPreference,
  ProjectMode,
  ProjectWorkspace,
} from "@/lib/types";

const IDLE_OPTIONS: Array<[LocalModelSession["idle_timeout_seconds"], string]> = [
  [60, "1 min"],
  [300, "5 min"],
  [900, "15 min"],
  [1800, "30 min"],
  [86400, "Until stopped"],
];

const CONTEXT_OPTIONS: Array<[LocalModelSession["context_window"], string]> = [
  [8192, "8K · leanest"],
  [16384, "16K · light"],
  [32768, "32K · recommended"],
  [65536, "64K · long documents"],
  [131072, "128K · heavy"],
];

function gigabytes(bytes: number): string {
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
}

/** "qwen3-coder:30b" → "qwen3-coder", so the trigger stays short. */
function shortModel(id: string | null | undefined): string {
  if (!id) return "model";
  return id.split(":")[0] ?? id;
}

function localStateLabel(session: LocalModelSession | null, now: number): string {
  if (!session) return "checking";
  if (session.state === "loading") return "loading…";
  if (session.state === "busy") return "busy";
  if (session.state === "ready") {
    if (session.idle_timeout_seconds >= 86400) return shortModel(session.selected_model);
    if (session.expires_at) {
      const seconds = Math.max(
        0,
        Math.ceil((new Date(session.expires_at).getTime() - now) / 1000),
      );
      return `${shortModel(session.selected_model)} · ${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
    }
    return shortModel(session.selected_model);
  }
  if (session.state === "error") return "Ollama unavailable";
  return "not launched";
}

type ModelControlProps = {
  preference: ModelPreference | null;
  onChooseProvider: (provider: "local" | "oci") => void;
  providerSaving: boolean;
  project: ProjectWorkspace | null;
  projectMode: ProjectMode;
  onChooseProjectMode: (mode: ProjectMode) => void;
  projectBusy: boolean;
  disabled?: boolean;
};

/**
 * One control, top-right of the chat pane, for what runs the next message.
 *
 * Plain chat picks a provider — on-device or Grok in the cloud. A project has
 * its own two modes instead (Grok maps once then North runs turns, or Grok
 * leads every step), so when a project is scoped this shows those rather than a
 * provider toggle the project routing would ignore. The on-device launch
 * controls live under a divider, since both plain-local chat and the Grok→Local
 * project mode run answers on the same local weights.
 */
export function ModelControl({
  preference,
  onChooseProvider,
  providerSaving,
  project,
  projectMode,
  onChooseProjectMode,
  projectBusy,
  disabled = false,
}: ModelControlProps) {
  const [session, setSession] = useState<LocalModelSession | null>(null);
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [model, setModel] = useState("");
  const [idle, setIdle] = useState<LocalModelSession["idle_timeout_seconds"]>(300);
  const [context, setContext] = useState<LocalModelSession["context_window"]>(32768);
  const [now, setNow] = useState(() => Date.now());
  // The status poll reports what is running now; a pending idle/context choice
  // is the intent for the next launch and must survive a poll.
  const formTouched = useRef(false);
  const sessionRef = useRef<LocalModelSession | null>(null);
  const rootRef = useRef<HTMLDivElement>(null);

  const provider = preference?.provider ?? "local";
  const ociAvailable = preference?.oci_available === true;

  const apply = useCallback((next: LocalModelSession) => {
    setSession(next);
    sessionRef.current = next;
    setModel((current) => current || next.selected_model || next.models[0]?.id || "");
    if (!formTouched.current) {
      setIdle(next.idle_timeout_seconds);
      setContext(next.context_window);
    }
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

  // Fetch once on mount regardless of visibility, so the panel always has the
  // installed-model list. Only the RECURRING poll is gated on visibility: an
  // unseen poll is wasted battery, and the host reads a live poll as "a window
  // is open", so a backgrounded app should stop claiming to be one and let the
  // weights be released. The one mount fetch is fine — the component only exists
  // while the chat page is open.
  useEffect(() => {
    let interval = 0;
    const stop = () => {
      if (interval) window.clearInterval(interval);
      interval = 0;
    };
    const startPolling = () => {
      stop();
      interval = window.setInterval(() => void refresh(), open ? 3000 : 10000);
    };
    const sync = () => (document.visibilityState === "visible" ? startPolling() : stop());
    void refresh();
    sync();
    document.addEventListener("visibilitychange", sync);
    return () => {
      document.removeEventListener("visibilitychange", sync);
      stop();
    };
  }, [open, refresh]);

  // Opening shows what is actually running; every edit after belongs to the user.
  useEffect(() => {
    if (!open) return;
    formTouched.current = false;
    const live = sessionRef.current;
    if (live) {
      setIdle(live.idle_timeout_seconds);
      setContext(live.context_window);
    }
  }, [open]);

  // Close on an outside click or Escape, like the pickers do.
  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  useEffect(() => {
    if (session?.state !== "ready" || !session.expires_at) return;
    const interval = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(interval);
  }, [session?.expires_at, session?.state]);

  async function launch() {
    if (!model || busy) return;
    setBusy(true);
    setError(null);
    setSession((current) => (current ? { ...current, state: "loading", error: null } : current));
    try {
      formTouched.current = false;
      apply(await launchLocalModel(model, idle, context));
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

  const weightBytes = useMemo(
    () => session?.models.find((item) => item.id === model)?.size_bytes ?? 0,
    [model, session?.models],
  );

  const advice = useMemo(() => {
    const weights = weightBytes ? `${gigabytes(weightBytes)} of weights stay resident while this model runs. ` : "";
    if (context >= 131072) return `${weights}128K only costs more once a conversation grows into it, but a full one adds several GB and sustained heat.`;
    if (context >= 65536) return `${weights}64K costs nothing extra until a conversation is long; it is worth it for whole documents.`;
    if (context <= 16384) return `${weights}A smaller context caps how much a long conversation can add, at the price of a shorter memory.`;
    if (idle >= 86400) return `${weights}Until stopped keeps it resident until you press Stop — the largest sustained cost of any setting here.`;
    if (idle >= 1800) return `${weights}A longer warm window trades memory and battery for a faster next answer.`;
    return `${weights}A shorter idle window is the one setting that reliably gives the memory back.`;
  }, [context, idle, weightBytes]);

  const memoryLine = useMemo(() => {
    if (!session?.resident_bytes) return null;
    const total = session.total_memory_bytes ? ` of ${gigabytes(session.total_memory_bytes)}` : "";
    return `Ollama is holding ${gigabytes(session.resident_bytes)}${total}`;
  }, [session?.resident_bytes, session?.total_memory_bytes]);

  // The trigger's dot state: for a live local session it mirrors the session;
  // cloud/Grok shows a solid dot; a project shows its own steady green.
  const localState = session?.state ?? "off";
  const dotState = project
    ? projectMode === "grok_continuous" && !ociAvailable
      ? "off"
      : "ready"
    : provider === "oci"
      ? ociAvailable ? "ready" : "off"
      : localState;

  const triggerLabel = project
    ? projectMode === "grok_continuous"
      ? "Keep Grok"
      : "Grok → Local"
    : provider === "oci"
      ? "Cloud · Grok"
      : `Local · ${localStateLabel(session, now)}`;

  const showLocalSession = !project || projectMode === "grok_bootstrap_local";

  return (
    <div className="modelControl" ref={rootRef}>
      <button
        type="button"
        className={`modelControlTrigger state-${dotState}`}
        aria-expanded={open}
        aria-haspopup="dialog"
        onClick={() => setOpen((value) => !value)}
        title={project ? "Project reasoning mode" : "Model for this conversation"}
      >
        <i aria-hidden="true" />
        <span>{triggerLabel}</span>
        <b aria-hidden="true">⌄</b>
      </button>

      {open ? (
        <section className="modelControlPanel" aria-label="Model for this conversation">
          {project ? (
            <>
              <div className="modelControlEyebrow">
                <span className="eyebrow">Whole-project mode</span>
                <p>Grok creates the first local map. These choose who leads each bounded, approval-gated step after that.</p>
              </div>
              <div className="modelControlChoice" role="radiogroup" aria-label="Project reasoning mode">
                <button
                  type="button"
                  role="radio"
                  aria-checked={projectMode === "grok_bootstrap_local"}
                  className={projectMode === "grok_bootstrap_local" ? "selected" : ""}
                  disabled={disabled || projectBusy}
                  onClick={() => onChooseProjectMode("grok_bootstrap_local")}
                >
                  <strong>Grok → Local</strong>
                  <small>Grok maps once; North handles project turns on-device.</small>
                </button>
                <button
                  type="button"
                  role="radio"
                  aria-checked={projectMode === "grok_continuous"}
                  className={projectMode === "grok_continuous" ? "selected" : ""}
                  disabled={disabled || projectBusy || !ociAvailable}
                  title={ociAvailable ? undefined : "Configure OCI in Settings first"}
                  onClick={() => onChooseProjectMode("grok_continuous")}
                >
                  <strong>Keep Grok</strong>
                  <small>{ociAvailable ? "Grok leads every bounded project step — largest context." : "Needs OCI Responses configured."}</small>
                </button>
              </div>
            </>
          ) : (
            <div className="modelControlChoice" role="radiogroup" aria-label="Reasoning model">
              <button
                type="button"
                role="radio"
                aria-checked={provider !== "oci"}
                className={provider !== "oci" ? "selected" : ""}
                disabled={disabled || providerSaving}
                onClick={() => onChooseProvider("local")}
              >
                <strong>Local</strong>
                <small>On-device Ollama models. Nothing leaves this machine.</small>
              </button>
              <button
                type="button"
                role="radio"
                aria-checked={provider === "oci"}
                className={provider === "oci" ? "selected" : ""}
                disabled={disabled || providerSaving || !ociAvailable}
                title={ociAvailable ? undefined : "Configure OCI in Settings first"}
                onClick={() => onChooseProvider("oci")}
              >
                <strong>Cloud · Grok</strong>
                <small>{ociAvailable ? "Grok 4.3 through OCI, for the largest context." : "Needs OCI configured in Settings."}</small>
              </button>
            </div>
          )}

          {showLocalSession ? (
            <div className="modelControlSession">
              <div className="modelControlSessionHead">
                <span className="eyebrow">On-device model</span>
                {session?.state === "ready" || session?.state === "busy" ? (
                  <button type="button" className="modelControlStop" onClick={() => void stop()} disabled={busy || session.state === "busy"}>
                    Stop
                  </button>
                ) : null}
              </div>
              <label>
                <select value={model} onChange={(event) => setModel(event.target.value)} disabled={busy}>
                  {!session?.models.length ? <option value="">No installed models found</option> : null}
                  {session?.models.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.name}{item.parameter_size ? ` · ${item.parameter_size}` : ""}
                      {item.size_bytes ? ` · ${gigabytes(item.size_bytes)}` : ""}
                    </option>
                  ))}
                </select>
              </label>
              <div className="modelControlRow">
                <label>
                  <span>Unload after idle</span>
                  <select
                    value={idle}
                    onChange={(event) => {
                      formTouched.current = true;
                      setIdle(Number(event.target.value) as LocalModelSession["idle_timeout_seconds"]);
                    }}
                    disabled={busy}
                  >
                    {IDLE_OPTIONS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
                  </select>
                </label>
                <label>
                  <span>Context</span>
                  <select
                    value={context}
                    onChange={(event) => {
                      formTouched.current = true;
                      setContext(Number(event.target.value) as LocalModelSession["context_window"]);
                    }}
                    disabled={busy}
                  >
                    {CONTEXT_OPTIONS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
                  </select>
                </label>
              </div>
              {memoryLine ? <p className="modelControlMemory">{memoryLine}</p> : null}
              <p className="modelControlAdvice">{advice}</p>
              <button className="modelControlLaunch" type="button" onClick={() => void launch()} disabled={busy || !model}>
                {busy ? "Working…" : session?.state === "ready" && model === session.selected_model ? "Relaunch with settings" : `Launch ${shortModel(model)}`}
              </button>
            </div>
          ) : null}

          {(session?.state === "error" && showLocalSession) || error ? (
            <p className="modelControlError" role="alert">{error || session?.error}</p>
          ) : null}
        </section>
      ) : null}
    </div>
  );
}
