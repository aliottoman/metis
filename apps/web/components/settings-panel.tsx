"use client";

import { useCallback, useEffect, useState } from "react";

import { API_BASE, getHealth, getModelPreference, setModelPreference } from "@/lib/api";
import type { HealthSnapshot, ModelPreference } from "@/lib/types";

type Provider = ModelPreference["provider"];

export function SettingsPanel() {
  const [health, setHealth] = useState<HealthSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [preference, setPreference] = useState<ModelPreference | null>(null);
  const [pinnedChoice, setPinnedChoice] = useState<string>("");
  const [savingPreference, setSavingPreference] = useState(false);
  const [preferenceError, setPreferenceError] = useState<string | null>(null);
  const [ociTools, setOciTools] = useState<Array<"x_search" | "code_interpreter">>(["code_interpreter"]);

  // The installed local lineup, straight from the runtime — a hardcoded list
  // here went stale the first time the lineup changed, and stayed stale.
  const localModels = health?.models ?? [];

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

  async function save(
    mode: "split" | "pinned",
    model: string | null,
    provider: Provider,
    tools: Array<"x_search" | "code_interpreter">,
    failure: string,
  ) {
    setSavingPreference(true);
    setPreferenceError(null);
    try {
      setPreference(await setModelPreference(mode, model, provider, tools));
    } catch (saveError) {
      setPreferenceError(saveError instanceof Error ? saveError.message : failure);
    } finally {
      setSavingPreference(false);
    }
  }

  const chooseProvider = (provider: Provider) =>
    save(preference?.mode ?? "split", preference?.model ?? null, provider, ociTools, "Could not update the model provider.");

  const choosePerTask = () =>
    save("split", null, preference?.provider ?? "local", ociTools, "Could not update model routing.");

  const choosePinned = (model: string) => {
    setPinnedChoice(model);
    return save("pinned", model, preference?.provider ?? "local", ociTools, "Could not update model routing.");
  };

  async function toggleOciTool(tool: "x_search" | "code_interpreter") {
    const selected = ociTools.includes(tool)
      ? ociTools.filter((item) => item !== tool)
      : [...ociTools, tool];
    setOciTools(selected);
    await save(preference?.mode ?? "split", preference?.model ?? null, preference?.provider ?? "local", selected, "Could not update OCI tools.");
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

  const isPinned = preference?.mode === "pinned";
  const provider = preference?.provider ?? "local";
  const providerBadge =
    provider === "oci" ? "Cloud · Grok" : provider === "cohere" ? "Cloud · Command A+" : "Local";

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
        <div className="sectionTitle"><div><h2>Reasoning provider</h2><p>The selected provider is pinned into every new run for reliable replay. Notes, sizing, and analysis follow this choice too.</p></div><span className="sectionBadge">{providerBadge}</span></div>
        <div className="providerChoiceGrid">
          <button type="button" className={`providerChoice ${provider === "local" ? "selected" : ""}`} onClick={() => void chooseProvider("local")} disabled={savingPreference}>
            <span>On device</span><strong>Local Ollama</strong><small>Private, offline reasoning with the models installed below.</small>
          </button>
          <button type="button" className={`providerChoice ${provider === "oci" ? "selected" : ""}`} onClick={() => void chooseProvider("oci")} disabled={savingPreference || !preference?.oci_available}>
            <span>OCI Responses</span><strong>Grok</strong><small>{preference?.oci_available ? "Large-context cloud reasoning with governed native tools." : "Configure the OCI Responses project OCID to enable."}</small>
          </button>
          <button type="button" className={`providerChoice ${provider === "cohere" ? "selected" : ""}`} onClick={() => void chooseProvider("cohere")} disabled={savingPreference || !preference?.cohere_available}>
            <span>Cohere</span><strong>Command A+</strong><small>{preference?.cohere_available ? "Strong tool use and structured output; also powers dictation." : "Configure Cohere on OCI Generative AI to enable."}</small>
          </button>
        </div>
        <div className="nativeToolChoices" aria-label="OCI native tools">
          <label><input type="checkbox" checked={ociTools.includes("code_interpreter")} onChange={() => void toggleOciTool("code_interpreter")} disabled={savingPreference || !preference?.oci_available} /><span><strong>Code Interpreter</strong><small>Temporary OCI-managed Python container (Grok runs only)</small></span></label>
          <label><input type="checkbox" checked={ociTools.includes("x_search")} onChange={() => void toggleOciTool("x_search")} disabled={savingPreference || !preference?.oci_available} /><span><strong>X Search</strong><small>Native X search — the Web scope in chat is the general one</small></span></label>
        </div>
        <p className="sectionLede">Metis tools remain available through the governed planner. Model calls made while a tool executes follow the same provider as the run, under each tool&apos;s per-run call budget and pinned prompts. Cloud service-side memory remains off.</p>
      </section>

      <section className="settingsSection">
        <div className="sectionTitle"><div><h2>Local models</h2><p>One heavyweight model runs at a time to respect unified memory.</p></div><span className="sectionBadge">Ollama</span></div>
        {localModels.length ? (
          <div className="modelList">
            {localModels.map((model, index) => (
              <article key={model.id}>
                <span className="modelOrdinal">{String(index + 1).padStart(2, "0")}</span>
                <div className="modelDescription"><strong>{model.label || model.id}</strong><span>{model.id}</span>{model.role ? <p>{model.role}</p> : null}</div>
                <div className="modelContext"><strong>{model.context_window ? `${Math.round(model.context_window / 1024)}K context` : "Context on load"}</strong><span>{model.loaded ? "Loaded now" : "Loaded on demand"}</span></div>
                <span className={`healthStatus health-${model.status}`}><i />{model.status}</span>
              </article>
            ))}
          </div>
        ) : (
          <p className="sectionLede">{health?.ollama?.status === "ok" ? "No local models reported yet — launch one from the chat header." : "Ollama is not reachable, so the local lineup is unknown."}</p>
        )}
        <p className="sectionLede">
          In the local lane Metis picks a different model per task (planning vs. code generation) by
          default, and every switch costs a reload in unified memory. Pin one model to stop the
          back-and-forth. Cloud lanes are single-model, so pinning applies locally.
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
            disabled={savingPreference || !localModels.length}
          >
            {!localModels.length ? <option value="">No local models</option> : null}
            {localModels.map((model) => (
              <option key={model.id} value={model.id}>{model.label || model.id}</option>
            ))}
          </select>
          <button
            type="button"
            className={isPinned ? "primaryButton" : "secondaryButton"}
            onClick={() => void choosePinned(pinnedChoice)}
            disabled={savingPreference || !pinnedChoice}
          >
            {savingPreference ? "Saving…" : "Always use this model"}
          </button>
        </div>
        {isPinned && preference?.model ? (
          <span className="mutedMeta">
            Pinned to {localModels.find((model) => model.id === preference.model)?.label ?? preference.model} for every request — new conversations only.
          </span>
        ) : null}
        {preferenceError ? <span className="mutedMeta" role="alert">{preferenceError}</span> : null}
      </section>

      <div className="settingsGrid">
        <section className="settingsSection compactSection">
          <div className="sectionTitle"><div><h2>Privacy boundary</h2><p>Cloud use is explicit and run-pinned.</p></div></div>
          <ul className="checkList">
            <li><span>✓</span><div><strong>Loopback only</strong><small>API bound to this device</small></div></li>
            <li><span>✓</span><div><strong>Governed tool broker</strong><small>Tool model calls follow your provider, budgeted per run</small></div></li>
            <li><span>✓</span><div><strong>Approval required</strong><small>Persistent learning is proposal-first</small></div></li>
            <li><span>✓</span><div><strong>Web is per-message</strong><small>Search runs only when a message selects the Web scope</small></div></li>
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
