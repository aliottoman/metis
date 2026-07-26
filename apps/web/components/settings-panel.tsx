"use client";

import { useCallback, useEffect, useState } from "react";

import { API_BASE, getHealth, getModelPreference, setModelPreference } from "@/lib/api";
import type { HealthSnapshot, ModelPreference } from "@/lib/types";

const configuredModels = [
  { id: "qwen3.6:35b-mlx", label: "Qwen3.6 35B", role: "Planning · review · vision", context: "32K working context" },
  { id: "north-mini-code-1.0:mlx-nvfp4", label: "North Mini Code", role: "Default code generation", context: "Fast profile" },
  { id: "north-mini-code-1.0:mlx-mxfp8", label: "North Mini Code Max", role: "Quality and repair", context: "Quality profile" },
];

export function SettingsPanel() {
  const [health, setHealth] = useState<HealthSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [preference, setPreference] = useState<ModelPreference | null>(null);
  const [pinnedChoice, setPinnedChoice] = useState<string>(configuredModels[0].id);
  const [savingPreference, setSavingPreference] = useState(false);
  const [preferenceError, setPreferenceError] = useState<string | null>(null);
  const [ociTools, setOciTools] = useState<Array<"x_search" | "code_interpreter">>(["code_interpreter"]);

  const loadPreference = useCallback(async () => {
    try {
      const current = await getModelPreference();
      setPreference(current);
      if (current.model) setPinnedChoice(current.model);
      setOciTools(current.oci_tools);
    } catch {
      // Non-fatal — the picker just keeps its default; health above already
      // reports whether the API is reachable at all.
    }
  }, []);

  async function choosePerTask() {
    setSavingPreference(true);
    setPreferenceError(null);
    try {
      setPreference(await setModelPreference("split", null, preference?.provider ?? "local", ociTools));
    } catch (saveError) {
      setPreferenceError(saveError instanceof Error ? saveError.message : "Could not update model routing.");
    } finally {
      setSavingPreference(false);
    }
  }

  async function choosePinned(model: string) {
    setPinnedChoice(model);
    setSavingPreference(true);
    setPreferenceError(null);
    try {
      setPreference(await setModelPreference("pinned", model, preference?.provider ?? "local", ociTools));
    } catch (saveError) {
      setPreferenceError(saveError instanceof Error ? saveError.message : "Could not update model routing.");
    } finally {
      setSavingPreference(false);
    }
  }

  const isPinned = preference?.mode === "pinned";

  async function chooseProvider(provider: "local" | "oci") {
    setSavingPreference(true);
    setPreferenceError(null);
    try {
      setPreference(await setModelPreference(
        preference?.mode ?? "split",
        preference?.model ?? null,
        provider,
        ociTools,
      ));
    } catch (saveError) {
      setPreferenceError(saveError instanceof Error ? saveError.message : "Could not update the model provider.");
    } finally {
      setSavingPreference(false);
    }
  }

  async function toggleOciTool(tool: "x_search" | "code_interpreter") {
    const selected = ociTools.includes(tool)
      ? ociTools.filter((item) => item !== tool)
      : [...ociTools, tool];
    setOciTools(selected);
    setSavingPreference(true);
    setPreferenceError(null);
    try {
      setPreference(await setModelPreference(
        preference?.mode ?? "split",
        preference?.model ?? null,
        preference?.provider ?? "local",
        selected,
      ));
    } catch (saveError) {
      setOciTools(ociTools);
      setPreferenceError(saveError instanceof Error ? saveError.message : "Could not update OCI tools.");
    } finally {
      setSavingPreference(false);
    }
  }

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setHealth(await getHealth());
    } catch (healthError) {
      setHealth(null);
      setError(healthError instanceof Error ? healthError.message : "The local service did not respond.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
    void loadPreference();
    const timer = window.setInterval(() => void refresh(), 15_000);
    return () => window.clearInterval(timer);
  }, [refresh, loadPreference]);

  return (
    <div className="workspacePage settingsPage">
      <header className="pageHeader">
        <div>
          <span className="eyebrow">Runtime</span>
          <h1>Settings & health</h1>
          <p>Choose where Metis reasons while keeping permissions, tools, and memory under local control.</p>
        </div>
        <button className="secondaryButton" type="button" onClick={() => void refresh()} disabled={loading}>{loading ? "Checking…" : "Check now"}</button>
      </header>

      <section className={`healthHero ${health?.status === "ok" ? "healthy" : "unhealthy"}`}>
        <span className="healthOrb"><i /></span>
        <div>
          <span className="eyebrow">System status</span>
          <h2>{health?.status === "ok" ? "Metis is ready" : error ? "Metis is waiting for its local service" : "Checking Metis"}</h2>
          <p>{error ?? `API ${health?.version ? `version ${health.version} ` : ""}is reachable at ${API_BASE}.`}</p>
        </div>
      </section>

      <section className="settingsSection">
        <div className="sectionTitle"><div><h2>Reasoning provider</h2><p>The selected provider is pinned into every new run for reliable replay.</p></div><span className="sectionBadge">{preference?.provider === "oci" ? "Cloud" : "Local"}</span></div>
        <div className="providerChoiceGrid">
          <button type="button" className={`providerChoice ${preference?.provider !== "oci" ? "selected" : ""}`} onClick={() => void chooseProvider("local")} disabled={savingPreference}>
            <span>On device</span><strong>Local Ollama</strong><small>Private, offline reasoning with Qwen and North.</small>
          </button>
          <button type="button" className={`providerChoice ${preference?.provider === "oci" ? "selected" : ""}`} onClick={() => void chooseProvider("oci")} disabled={savingPreference || !preference?.oci_available}>
            <span>OCI Responses</span><strong>Grok 4.3</strong><small>{preference?.oci_available ? "Large-context cloud reasoning with governed native tools." : "Configure the OCI Responses project OCID to enable."}</small>
          </button>
        </div>
        <div className="nativeToolChoices" aria-label="OCI native tools">
          <label><input type="checkbox" checked={ociTools.includes("code_interpreter")} onChange={() => void toggleOciTool("code_interpreter")} disabled={savingPreference || !preference?.oci_available} /><span><strong>Code Interpreter</strong><small>Temporary OCI-managed Python container</small></span></label>
          <label><input type="checkbox" checked={ociTools.includes("x_search")} onChange={() => void toggleOciTool("x_search")} disabled={savingPreference || !preference?.oci_available} /><span><strong>X Search</strong><small>Native X search—not general web search</small></span></label>
        </div>
        <p className="sectionLede">Metis tools remain available through the governed local planner. Any model call made while one of those tools executes is forced through local Ollama, even in a Grok-authored run. OCI service-side memory remains off.</p>
      </section>

      <section className="settingsSection">
        <div className="sectionTitle"><div><h2>Model routing</h2><p>One heavyweight model runs at a time to respect unified memory.</p></div><span className="sectionBadge">Ollama</span></div>
        <div className="modelList">
          {configuredModels.map((configured, index) => {
            const runtime = health?.models?.find((model) => model.id === configured.id || model.label === configured.id);
            const state = runtime?.status ?? (health?.ollama?.status === "ok" ? "unknown" : "offline");
            return (
              <article key={configured.id}>
                <span className="modelOrdinal">0{index + 1}</span>
                <div className="modelDescription"><strong>{configured.label}</strong><span>{configured.id}</span><p>{configured.role}</p></div>
                <div className="modelContext"><strong>{runtime?.context_window ? `${Math.round(runtime.context_window / 1024)}K context` : configured.context}</strong><span>{runtime?.loaded ? "Loaded now" : "Loaded on demand"}</span></div>
                <span className={`healthStatus health-${state}`}><i />{state}</span>
              </article>
            );
          })}
        </div>
        <p className="sectionLede">
          By default Metis picks a different model per task (planning vs. code generation), and
          every switch costs a reload in unified memory. Pin one model to stop the back-and-forth.
        </p>
        <div className="cardActions">
          <button
            type="button"
            className={isPinned ? "secondaryButton" : "primaryButton"}
            onClick={() => void choosePerTask()}
            disabled={savingPreference}
          >
            Per-task (default)
          </button>
          <select
            value={pinnedChoice}
            onChange={(event) => setPinnedChoice(event.target.value)}
            disabled={savingPreference}
          >
            {configuredModels.map((model) => (
              <option key={model.id} value={model.id}>{model.label}</option>
            ))}
          </select>
          <button
            type="button"
            className={isPinned ? "primaryButton" : "secondaryButton"}
            onClick={() => void choosePinned(pinnedChoice)}
            disabled={savingPreference}
          >
            {savingPreference ? "Saving…" : "Always use this model"}
          </button>
        </div>
        {isPinned && preference?.model ? (
          <span className="mutedMeta">
            Pinned to {configuredModels.find((model) => model.id === preference.model)?.label ?? preference.model} for every request — new conversations only.
          </span>
        ) : null}
        {preferenceError ? <span className="mutedMeta" role="alert">{preferenceError}</span> : null}
      </section>

      <div className="settingsGrid">
        <section className="settingsSection compactSection">
          <div className="sectionTitle"><div><h2>Privacy boundary</h2><p>Cloud use is explicit and run-pinned.</p></div></div>
          <ul className="checkList">
            <li><span>✓</span><div><strong>Loopback only</strong><small>API bound to this device</small></div></li>
            <li><span>✓</span><div><strong>Local tool broker</strong><small>Executing tools use on-device models</small></div></li>
            <li><span>✓</span><div><strong>Approval required</strong><small>Persistent learning is proposal-first</small></div></li>
          </ul>
        </section>
        <section className="settingsSection compactSection">
          <div className="sectionTitle"><div><h2>Services</h2><p>Readiness reported by the control plane.</p></div></div>
          <dl className="serviceList">
            <div><dt>API</dt><dd><span className={health ? "serviceUp" : "serviceDown"} />{health ? "Connected" : "Unavailable"}</dd></div>
            <div><dt>Ollama</dt><dd><span className={health?.ollama?.status === "ok" ? "serviceUp" : "serviceNeutral"} />{health?.ollama?.status ?? "Unknown"}</dd></div>
            <div><dt>Database</dt><dd><span className={health?.database === "ok" ? "serviceUp" : "serviceNeutral"} />{health?.database ?? "Unknown"}</dd></div>
            <div><dt>Podman sandbox</dt><dd><span className={health?.sandbox === "ok" ? "serviceUp" : "serviceNeutral"} />{health?.sandbox ?? "On demand"}</dd></div>
          </dl>
        </section>
      </div>
    </div>
  );
}
