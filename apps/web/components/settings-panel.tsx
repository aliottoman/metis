"use client";

import { SlidersHorizontal } from "lucide-react";

import { useCallback, useEffect, useState } from "react";

import {
  API_BASE,
  getHealth,
  getModelPreference,
  getSpeechPreference,
  getVoiceAvailability,
  setModelPreference,
  setSpeechPreference,
} from "@/lib/api";
import { clinePassReady } from "@/lib/model-route";
import { ThemeControl } from "@/components/ui/theme-control";
import type { HealthSnapshot, ModelPreference, SpeechPreference, VoiceAvailability } from "@/lib/types";
import { MetisCompanion } from "@/components/metis-companion";
import { Notice } from "@/components/ui/notice";
import { PageHeader } from "@/components/ui/page-header";
import { Status } from "@/components/ui/status";
import { usePoll } from "@/hooks/use-poll";
import { SelectMenu } from "@/components/select-menu";
import {
  readCompanionEnergy,
  writeCompanionEnergy,
  type CompanionEnergy,
} from "@/lib/companion-preference";

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
  const [companionEnergy, setCompanionEnergy] = useState<CompanionEnergy>("expressive");
  const [speech, setSpeech] = useState<SpeechPreference | null>(null);
  const [savingSpeech, setSavingSpeech] = useState(false);
  const [speechError, setSpeechError] = useState<string | null>(null);
  const [voiceAvailability, setVoiceAvailability] = useState<VoiceAvailability | null>(null);

  // The installed local lineup, straight from the runtime — a hardcoded list
  // here went stale the first time the lineup changed, and stayed stale.
  const localModels = health?.models ?? [];

  const loadPreference = useCallback(async () => {
    setPreferenceError(null);
    try {
      const current = await getModelPreference();
      setPreference(current);
      if (current.model) setPinnedChoice(current.model);
      setOciTools(current.oci_tools);
    } catch (loadError) {
      setPreferenceError(loadError instanceof Error ? loadError.message : "Model preferences could not be loaded.");
    }
  }, []);

  async function save(
    mode: "split" | "pinned",
    model: string | null,
    provider: Provider,
    tools: Array<"x_search" | "code_interpreter">,
    failure: string,
  ) {
    if (savingPreference || !preference) return;
    setSavingPreference(true);
    setPreferenceError(null);
    try {
      const saved = await setModelPreference(mode, model, provider, tools);
      setPreference(saved);
      setOciTools(saved.oci_tools);
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

  const loadSpeech = useCallback(async () => {
    setSpeechError(null);
    try {
      setSpeech(await getSpeechPreference());
    } catch (loadError) {
      setSpeechError(loadError instanceof Error ? loadError.message : "Voice preferences could not be loaded.");
    }
  }, []);

  async function chooseTranscriber(provider: SpeechPreference["stt_provider"]) {
    setSavingSpeech(true);
    setSpeechError(null);
    try {
      setSpeech(await setSpeechPreference(provider, speech?.spoken_confirmation ?? false));
    } catch (saveError) {
      setSpeechError(saveError instanceof Error ? saveError.message : "Could not change the transcriber.");
    } finally {
      setSavingSpeech(false);
    }
  }

  async function saveVoiceSettings(options: {
    model?: string;
    spokenConfirmation?: boolean;
  }) {
    if (!speech) return;
    setSavingSpeech(true);
    setSpeechError(null);
    try {
      setSpeech(
        await setSpeechPreference(
          speech.stt_provider,
          options.spokenConfirmation ?? speech.spoken_confirmation,
          options.model,
        ),
      );
    } catch (saveError) {
      setSpeechError(saveError instanceof Error ? saveError.message : "Could not change voice settings.");
    } finally {
      setSavingSpeech(false);
    }
  }

  async function toggleOciTool(tool: "x_search" | "code_interpreter") {
    const selected = ociTools.includes(tool)
      ? ociTools.filter((item) => item !== tool)
      : [...ociTools, tool];
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
    setCompanionEnergy(readCompanionEnergy());
    void refresh();
    void loadPreference();
    void loadSpeech();
    void getVoiceAvailability().then(setVoiceAvailability).catch(() => setVoiceAvailability(null));
  }, [refresh, loadPreference, loadSpeech]);
  // Health moves on its own; re-read it while the page is on screen.
  usePoll(refresh, 15_000);

  useEffect(() => {
    if (!pinnedChoice && localModels.length) setPinnedChoice(localModels[0]!.id);
  }, [localModels, pinnedChoice]);

  const isPinned = preference?.mode === "pinned";
  const provider = preference?.provider ?? "local";
  const clineModels = preference?.cline_models ?? [];
  const clineReady = clinePassReady(
    preference?.cline_available === true,
    clineModels,
  );
  const providerBadge =
    provider === "oci"
      ? "Cloud · Grok"
      : provider === "cohere"
        ? "Cloud · Command A+"
        : provider === "cline"
          ? "Cloud · ClinePass"
          : "Local";
  const transcriber = speech?.stt_provider ?? "cohere";
  const projectCoding = health?.project_coding_engine;
  const systemStatusTitle =
    health?.status === "ok"
      ? "Metis is ready"
      : error
        ? "Metis is waiting for its local service"
        : health?.status === "degraded"
          ? "Metis needs attention"
          : "Checking Metis";
  const systemStatusMessage = error
    ?? projectCoding?.reason
    ?? (!health
      ? "Checking required local services…"
      : health.status === "degraded"
        ? "A required local service is unavailable."
        : `API ${health.version ? `version ${health.version} ` : ""}is reachable at ${API_BASE}.`);
  const projectCodingStatus = !projectCoding
    ? "Unknown"
    : projectCoding.ready
      ? "Vertical-slice coding ready"
      : "Setup needed";

  return (
    <div className="workspacePage settingsPage">
      <PageHeader
        icon={<SlidersHorizontal />}
        eyebrow="Make Metis yours"
        title="Settings & health"
        lede="Personalize your workspace, choose your models, and keep an eye on connected services."
        actions={<button className="ui-btn" type="button" onClick={() => { void refresh(); void loadPreference(); void loadSpeech(); }} disabled={loading || savingPreference || savingSpeech}>{loading ? "Checking…" : "Check now"}</button>}
      />

      <nav className="workspace-jump-nav" aria-label="Settings sections">{[["appearance", "Appearance"], ["reasoning", "Models & reasoning"], ["voice", "Voice & audio"], ["services", "Services & privacy"]].map(([id, label]) => <a key={id} href={`#settings-${id}`}>{label}</a>)}</nav>
      <section className={`healthHero ${health?.status === "ok" ? "healthy" : "unhealthy"}`}>
        <span className="healthOrb"><i /></span>
        <div>
          <span className="eyebrow">System status</span>
          <h2>{systemStatusTitle}</h2>
          <p>{systemStatusMessage}</p>
        </div>
      </section>

      <section className="settingsSection" id="settings-appearance">
        <div className="sectionTitle"><div><h2>Appearance</h2><p>Follow your Mac, or pin light or dark. Remembered on this device.</p></div></div>
        <ThemeControl />
      </section>

      <section className="settingsSection">
        <div className="sectionTitle">
          <div><h2>Companion expression</h2><p>Choose how visibly the companion reacts. Mood color still communicates listening, working, done, and trouble in every mode.</p></div>
          <span className="sectionBadge">{companionEnergy}</span>
        </div>
        <div className="companion-energy" role="radiogroup" aria-label="Companion expression" onKeyDown={(event) => {
          const choices: CompanionEnergy[] = ["calm", "expressive", "playful"];
          const current = choices.indexOf(companionEnergy);
          const next = event.key === "ArrowRight" || event.key === "ArrowDown" ? (current + 1) % choices.length : event.key === "ArrowLeft" || event.key === "ArrowUp" ? (current + choices.length - 1) % choices.length : -1;
          if (next < 0) return;
          event.preventDefault();
          const choice = choices[next]!;
          setCompanionEnergy(choice);
          writeCompanionEnergy(choice);
          event.currentTarget.querySelectorAll<HTMLButtonElement>("[role=radio]")[next]?.focus();
        }}>
          {([
            ["calm", "Calm", "Slow breathing and restrained color."],
            ["expressive", "Expressive", "Fluid motion and clear mood shifts."],
            ["playful", "Playful", "More bounce, glow, and orbiting sparks."],
          ] as const).map(([value, label, description]) => (
            <button
              type="button"
              role="radio"
              aria-checked={companionEnergy === value}
              tabIndex={companionEnergy === value ? 0 : -1}
              className={companionEnergy === value ? "selected" : ""}
              key={value}
              onClick={() => { setCompanionEnergy(value); writeCompanionEnergy(value); }}
            >
              <span className="companion-energy-preview" data-preview-energy={value}>
                <MetisCompanion mood={value === "calm" ? "idle" : "thinking"} size={46} energy={value} />
              </span>
              <span><strong>{label}</strong><small>{description}</small></span>
              <i aria-hidden="true" />
            </button>
          ))}
        </div>
      </section>

      <section className="settingsSection" id="settings-reasoning">
        <div className="sectionTitle"><div><h2>Reasoning provider</h2><p>The selected provider is pinned into every new run for reliable replay. Notes, sizing, and analysis follow this choice too.</p></div><span className="sectionBadge">{preference ? providerBadge : preferenceError ? "Unavailable" : "Loading…"}</span></div>
        <div className="providerChoiceGrid" role="group" aria-label="Reasoning provider">
          <button type="button" aria-pressed={preference !== null && provider === "local"} className={`providerChoice ${preference && provider === "local" ? "selected" : ""}`} onClick={() => void chooseProvider("local")} disabled={savingPreference || !preference}>
            <span>On device</span><strong>Local Ollama</strong><small>Private, offline reasoning with the models installed below.</small>
          </button>
          <button type="button" aria-pressed={preference !== null && provider === "oci"} className={`providerChoice ${provider === "oci" ? "selected" : ""}`} onClick={() => void chooseProvider("oci")} disabled={savingPreference || !preference?.oci_available}>
            <span>OCI Responses</span><strong>Grok</strong><small>{preference?.oci_available ? "Large-context cloud reasoning with governed native tools." : "Configure the OCI Responses project OCID to enable."}</small>
          </button>
          <button type="button" aria-pressed={preference !== null && provider === "cohere"} className={`providerChoice ${provider === "cohere" ? "selected" : ""}`} onClick={() => void chooseProvider("cohere")} disabled={savingPreference || !preference?.cohere_available}>
            <span>Cohere</span><strong>Command A+</strong><small>{preference?.cohere_available ? "Strong tool use and structured output; also powers dictation." : "Configure Cohere on OCI Generative AI to enable."}</small>
          </button>
          <button type="button" aria-pressed={preference !== null && provider === "cline"} className={`providerChoice ${provider === "cline" ? "selected" : ""}`} onClick={() => void chooseProvider("cline")} disabled={savingPreference || !clineReady}>
            <span>ClinePass</span><strong>Planner + coder roles</strong><small>{clineReady ? `${clineModels.length} verified subscription model${clineModels.length === 1 ? "" : "s"}; choose each role and backup from the chat model control.` : "Configure WAQIL_CLINE_API_KEY to enable the verified ClinePass catalog."}</small>
          </button>
        </div>
        <div className="nativeToolChoices" aria-label="OCI native tools">
          <label><input type="checkbox" checked={ociTools.includes("code_interpreter")} onChange={() => void toggleOciTool("code_interpreter")} disabled={savingPreference || !preference?.oci_available} /><span><strong>Code Interpreter</strong><small>Temporary OCI-managed Python container (Grok runs only)</small></span></label>
          <label><input type="checkbox" checked={ociTools.includes("x_search")} onChange={() => void toggleOciTool("x_search")} disabled={savingPreference || !preference?.oci_available} /><span><strong>X Search</strong><small>Native X search — the Web scope in chat is the general one</small></span></label>
        </div>
        <p className="sectionLede">Metis tools remain available through the governed planner. Model calls made while a tool executes follow the same provider as the run, under each tool&apos;s per-run call budget and pinned prompts. Cloud service-side memory remains off.</p>
        {preferenceError ? <Notice kind="error" action={!preference ? "Try again" : undefined} onAction={() => void loadPreference()} onDismiss={preference ? () => setPreferenceError(null) : undefined}>{preferenceError}</Notice> : null}
      </section>

      <section className="settingsSection" id="settings-voice">
        <div className="sectionTitle">
          <div><h2>Voice &amp; audio</h2><p>Who transcribes dictation, what reasons during a live conversation, and how carefully Metis confirms a spoken write.</p></div>
          <Status state={voiceAvailability?.available ? "ready" : "stopped"} label={voiceAvailability?.available ? "Live voice ready" : "Live setup needed"} />
        </div>
        <div className="settings-audio">
          <div className="settings-audio-block">
            <h3>Composer dictation</h3>
            <p>The transcript stays a draft until you choose Send. Metis keeps no clip after transcription.</p>
            <div className="providerChoiceGrid">
              <button type="button" aria-pressed={transcriber === "cohere"} className={`providerChoice ${transcriber === "cohere" ? "selected" : ""}`} onClick={() => void chooseTranscriber("cohere")} disabled={savingSpeech || !speech?.cohere_available}>
                <span>Cohere</span><strong>Transcribe</strong><small>{speech?.cohere_available ? "Audio is converted to WAV locally before upload." : "Add WAQIL_COHERE_API_KEY to enable."}</small>
              </button>
              <button type="button" aria-pressed={transcriber === "elevenlabs"} className={`providerChoice ${transcriber === "elevenlabs" ? "selected" : ""}`} onClick={() => void chooseTranscriber("elevenlabs")} disabled={savingSpeech || !speech?.elevenlabs_available}>
                <span>ElevenLabs</span><strong>Scribe</strong><small>{speech?.elevenlabs_available ? "Uses the browser recording without re-encoding it." : "Add WAQIL_ELEVENLABS_API_KEY to enable."}</small>
              </button>
            </div>
          </div>
          <div className="settings-audio-block">
            <h3>Live voice</h3>
            <p>ElevenLabs runs the audio room; Metis controls retrieval and writes. Sessions end after two quiet minutes or when the tab leaves the foreground, and every spoken write gets a receipt with Undo.</p>
            {!voiceAvailability?.available && voiceAvailability?.missing.length ? <ol className="settings-audio-missing">{voiceAvailability.missing.map((reason) => <li key={reason}>{reason}</li>)}</ol> : null}
            <label className="ui-field">
              <span>Reasoning model</span>
              <select value={speech?.voice_model ?? ""} onChange={(event) => void saveVoiceSettings({ model: event.target.value })} disabled={savingSpeech || !speech?.voice_models.length}>
                {!speech?.voice_models.length ? <option value="">No voice models available</option> : null}
                {(speech?.voice_models ?? []).map((model) => <option key={model} value={model}>{model}</option>)}
              </select>
            </label>
            <label className="records-check">
              <input type="checkbox" checked={speech?.spoken_confirmation ?? false} onChange={(event) => void saveVoiceSettings({ spokenConfirmation: event.target.checked })} disabled={savingSpeech || !speech} />
              <span>Read back before writing: Metis asks for a spoken confirmation before it adds a record.</span>
            </label>
          </div>
        </div>
        {speechError ? <Notice kind="error" action={!speech ? "Try again" : undefined} onAction={() => void loadSpeech()} onDismiss={speech ? () => setSpeechError(null) : undefined}>{speechError}</Notice> : null}
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
            aria-pressed={preference !== null && !isPinned}
            onClick={() => void choosePerTask()}
            disabled={savingPreference || !preference}
          >
            Per-task (default)
          </button>
          <SelectMenu
            className="settingsModelSelect"
            hideLabel
            label="Pinned local model"
            value={pinnedChoice}
            onChange={setPinnedChoice}
            disabled={savingPreference || !preference || !localModels.length}
            options={localModels.length
              ? localModels.map((model) => ({ value: model.id, label: model.label || model.id }))
              : [{ value: "", label: "No local models", disabled: true }]}
          />
          <button
            type="button"
            className={isPinned ? "primaryButton" : "secondaryButton"}
            onClick={() => void choosePinned(pinnedChoice)}
            disabled={savingPreference || !preference || !pinnedChoice}
          >
            {savingPreference ? "Saving…" : "Always use this model"}
          </button>
        </div>
        {isPinned && preference?.model ? (
          <span className="mutedMeta">
            Pinned to {localModels.find((model) => model.id === preference.model)?.label ?? preference.model} for every request — new conversations only.
          </span>
        ) : null}
      </section>

      <div className="settingsGrid" id="settings-services">
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
            <div><dt>Project building</dt><dd><span className={projectCoding?.ready ? "serviceUp" : projectCoding ? "serviceDown" : "serviceNeutral"} />{projectCodingStatus}</dd></div>
          </dl>
        </section>
      </div>
    </div>
  );
}
