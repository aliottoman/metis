"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  getLocalModelSession,
  launchLocalModel,
  setModelPreference,
  stopLocalModel,
} from "@/lib/api";
import type {
  LocalModelSession,
  ModelPreference,
  ModelRole,
  ProjectMode,
  ProjectWorkspace,
  RoleChainEntry,
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

// The three routing roles a preference may ladder, with the words a user
// should see. Order matters: coder first, because its ladder steers builds.
// The Cline gateway serves both seats from one key. Measured on the same
// planning question: Opus answered correctly in 17 output tokens where the
// open-weight models spent 186 to 1,129 — so the orchestrator seat leads with
// it, and the coders are the ones the subscription covers.
const CLINE_MODELS = [
  "anthropic/claude-opus-4.5",
  "cline-pass/deepseek-v4-pro",
  "cline-pass/glm-5.2",
  "cline-pass/kimi-k2.7-code",
  "x-ai/grok-4.3",
];

const ROLE_ROWS: Array<[ModelRole, string, string]> = [
  ["coder", "Coder", "Writes project builds — the ladder that matters most"],
  ["planner", "Planner", "Routes requests and plans work"],
  ["quality", "Reviewer", "Quality and review passes"],
];

/** One ladder rung as a select value: "local:<model>", "cohere:", "oci:", "". */
function encodeRung(entry: RoleChainEntry | undefined): string {
  if (!entry) return "";
  if (entry.provider === "local") return entry.model ? `local:${entry.model}` : "";
  // Cline serves both seats from one key, so a rung names the model as well as
  // the lane — that is the whole point of putting a strong model in the
  // orchestrator seat and a cheap one in the coder's.
  if (entry.provider === "cline") return `cline:${entry.model ?? ""}`;
  return `${entry.provider}:`;
}

function decodeRung(value: string): RoleChainEntry | null {
  if (!value) return null;
  if (value === "cohere:") return { provider: "cohere", model: null };
  if (value === "oci:") return { provider: "oci", model: null };
  if (value.startsWith("local:")) return { provider: "local", model: value.slice(6) };
  if (value.startsWith("cline:")) return { provider: "cline", model: value.slice(6) || null };
  return null;
}

/** One rung as the few words a collapsed row can afford. */
function rungLabel(entry: RoleChainEntry | undefined): string {
  if (!entry) return "";
  if (entry.provider === "cohere") return "Command A+";
  if (entry.provider === "oci") return "Grok";
  if (entry.provider === "cline") return shortModel(entry.model?.split("/").pop() ?? "Cline");
  return shortModel(entry.model);
}

/**
 * A whole ladder as one sentence: "deepseek-v4-pro → qwen3-coder".
 *
 * This is the collapsed row's entire content, so it has to carry the fact that
 * matters — what runs first, and what catches it — without the three selects
 * that made the panel taller than the window.
 */
function chainSummary(ladder: RoleChainEntry[] | undefined): string {
  const rungs = (ladder ?? []).map(rungLabel).filter(Boolean);
  if (!rungs.length) return "Current selection";
  return rungs.join(" → ");
}

function gigabytes(bytes: number): string {
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
}

/** "qwen3-coder:30b" → "qwen3-coder", so the trigger stays short. */
function shortModel(id: string | null | undefined): string {
  if (!id) return "model";
  return id.split(":")[0] ?? id;
}

/** Mirrors the API's is_cloud_model: both hosted spellings Ollama uses. */
function isCloudModel(id: string | null | undefined): boolean {
  if (!id) return false;
  return id.endsWith("-cloud") || id.endsWith(":cloud");
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
  onChooseProvider: (provider: "local" | "oci" | "cohere" | "cline") => void;
  onPreferenceChange?: (preference: ModelPreference) => void;
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
 * Plain chat picks a provider — the Ollama lane, Grok, or Cohere. A project
 * has its own three modes instead (the Ollama lane runs each step, or Grok or
 * Command A+ leads every step), so when a project is scoped this shows those
 * rather than a provider toggle the project routing would ignore. The
 * on-device launch controls live under a divider, since both plain-local chat
 * and the Ollama-lane project mode run answers through the same daemon.
 */
export function ModelControl({
  preference,
  onChooseProvider,
  onPreferenceChange,
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
  const modelTouched = useRef(false);
  const sessionRef = useRef<LocalModelSession | null>(null);
  const rootRef = useRef<HTMLDivElement>(null);
  // Ladder edits under composition: preference refreshes must not clobber a
  // half-built chain, so the store only follows the server while untouched.
  const [chains, setChains] = useState<ModelPreference["role_chains"]>({});
  const [chainsDirty, setChainsDirty] = useState(false);
  // At most one ladder is open at a time. Three roles × three selects was the
  // block that pushed the panel past the bottom of the window; a ladder is
  // read far more often than it is edited, so reading is the default state.
  const [openRole, setOpenRole] = useState<ModelRole | null>(null);

  const provider = preference?.provider ?? "local";
  const ociAvailable = preference?.oci_available === true;
  const cohereAvailable = preference?.cohere_available === true;
  const clineAvailable = preference?.cline_available === true;

  useEffect(() => {
    if (!chainsDirty) setChains(preference?.role_chains ?? {});
  }, [preference, chainsDirty]);

  function setRung(role: ModelRole, slot: number, value: string) {
    setChainsDirty(true);
    setChains((current) => {
      const ladder = [...(current[role] ?? [])];
      const entry = decodeRung(value);
      if (entry) {
        while (ladder.length < slot) ladder.push({ provider: "local", model: null });
        ladder[slot] = entry;
      } else {
        ladder.splice(slot);
      }
      const next = { ...current };
      const kept = ladder.filter((item) => item.provider !== "local" || item.model);
      if (kept.length) next[role] = kept;
      else delete next[role];
      return next;
    });
  }

  async function saveChains() {
    if (!preference) return;
    setBusy(true);
    setError(null);
    try {
      const saved = await setModelPreference(
        preference.mode,
        preference.model,
        preference.provider,
        preference.oci_tools,
        chains,
      );
      setChainsDirty(false);
      onPreferenceChange?.(saved);
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "The ladder could not be saved.");
    } finally {
      setBusy(false);
    }
  }

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
    setOpenRole(null);
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

  // When the saved preference is a hosted model, the picker opens on it rather
  // than on whatever local model last ran — until the user picks for themselves.
  useEffect(() => {
    const pinned = preference?.mode === "pinned" ? preference.model : null;
    if (!modelTouched.current && pinned && isCloudModel(pinned)) setModel(pinned);
  }, [preference?.mode, preference?.model]);

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

  /** A hosted model has no weights to launch — picking it is a preference
      save, and the API refuses hosted models measured to ignore tool calls. */
  async function pinHosted() {
    if (!model || busy) return;
    setBusy(true);
    setError(null);
    try {
      const saved = await setModelPreference("pinned", model, "local", preference?.oci_tools ?? []);
      onPreferenceChange?.(saved);
    } catch (pinError) {
      setError(pinError instanceof Error ? pinError.message : "This hosted model could not be selected.");
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

  // A pinned hosted model runs on Ollama Cloud: always available, nothing
  // resident locally, so the trigger reports it instead of the local session.
  const hostedPinned =
    !project && provider !== "oci" && preference?.mode === "pinned" && isCloudModel(preference?.model);
  const cloudSelected = isCloudModel(model);

  // Ollama serves local weights and hosted models through the same daemon and
  // the same inventory call, but they are not the same choice: one spends this
  // machine's memory, the other spends a subscription and leaves the room. So
  // they are separate routes here rather than two entries in one model list.
  const cloudModels = useMemo(
    () => session?.models.filter((item) => isCloudModel(item.id)) ?? [],
    [session?.models],
  );
  const localModels = useMemo(
    () => session?.models.filter((item) => !isCloudModel(item.id)) ?? [],
    [session?.models],
  );
  const route: "local" | "ollama_cloud" | "oci" | "cohere" =
    provider === "oci" ? "oci" : provider === "cohere" ? "cohere" : hostedPinned ? "ollama_cloud" : "local";

  /** Move between the four routes, pinning a sensible model for each. */
  async function chooseRoute(next: "local" | "ollama_cloud" | "oci" | "cohere") {
    if (next === route || busy || providerSaving) return;
    if (next === "oci" || next === "cohere") {
      onChooseProvider(next);
      return;
    }
    // Leaving a cloud provider goes through the parent, which owns that save.
    if (provider === "oci" || provider === "cohere") onChooseProvider("local");
    const target = next === "ollama_cloud"
      ? (isCloudModel(model) ? model : cloudModels[0]?.id)
      : (session?.selected_model && !isCloudModel(session.selected_model)
          ? session.selected_model
          : localModels[0]?.id);
    if (!target) {
      setError(next === "ollama_cloud"
        ? "This Ollama has no cloud models available. Sign in to Ollama Cloud, then refresh."
        : "No local models are installed.");
      return;
    }
    modelTouched.current = true;
    setModel(target);
    setBusy(true);
    setError(null);
    try {
      onPreferenceChange?.(
        await setModelPreference("pinned", target, "local", preference?.oci_tools ?? []),
      );
    } catch (routeError) {
      setError(routeError instanceof Error ? routeError.message : "That route could not be selected.");
    } finally {
      setBusy(false);
    }
  }

  // The trigger's dot state: for a live local session it mirrors the session;
  // cloud/Grok shows a solid dot; a project shows its own steady green.
  const localState = session?.state ?? "off";
  // A continuous project mode is only as live as the key behind it.
  const projectModeReady =
    projectMode === "grok_continuous"
      ? ociAvailable
      : projectMode === "cohere_continuous"
        ? cohereAvailable
        : true;
  const dotState = project
    ? projectModeReady ? "ready" : "off"
    : provider === "oci"
      ? ociAvailable ? "ready" : "off"
      : provider === "cohere"
        ? cohereAvailable ? "ready" : "off"
        : hostedPinned
          ? "ready"
          : localState;

  // The local-lane project mode runs steps on whatever the Ollama lane
  // resolves: the pinned model when one is pinned there, else the session's.
  const ollamaLaneModel =
    provider === "local" && preference?.mode === "pinned" && preference.model
      ? preference.model
      : session?.selected_model ?? null;

  const triggerLabel = project
    ? projectMode === "grok_continuous"
      ? "Grok"
      : projectMode === "cohere_continuous"
        ? "Command A+"
        : ollamaLaneModel
          ? `Ollama · ${shortModel(ollamaLaneModel)}`
          : `Local · ${localStateLabel(session, now)}`
    : provider === "oci"
      ? "Cloud · Grok"
      : provider === "cohere"
        ? "Cloud · Command A+"
        : hostedPinned
          ? `Hosted · ${shortModel(preference?.model)}`
          : `Local · ${localStateLabel(session, now)}`;

  const showLocalSession = !project || projectMode === "grok_bootstrap_local";

  /**
   * The routes as data, so each one is a single compact row instead of a card.
   *
   * The prose that used to sit under every option now appears once, for the
   * route that is actually selected: four paragraphs to describe four choices
   * is what made this panel a form. An unavailable route still says so — as a
   * short state word in the row, and in full in its title — because "why is
   * this greyed out" is the one question the panel must always answer.
   */
  const routes = useMemo(() => {
    if (project) {
      return [
        {
          key: "grok_bootstrap_local",
          label: "Ollama lane",
          state: ollamaLaneModel ? shortModel(ollamaLaneModel) : "local",
          note: `A cloud model maps the repo once; ${
            ollamaLaneModel ? shortModel(ollamaLaneModel) : "your Ollama model"
          } runs each project step.`,
          disabled: false,
          title: undefined as string | undefined,
          selected: projectMode === "grok_bootstrap_local",
          choose: () => onChooseProjectMode("grok_bootstrap_local"),
        },
        {
          key: "grok_continuous",
          label: "Grok",
          state: ociAvailable ? "ready" : "off",
          note: ociAvailable
            ? "Grok leads every bounded project step — largest context."
            : "Off — needs the Grok lane enabled and OCI Responses configured.",
          disabled: !ociAvailable,
          title: ociAvailable
            ? undefined
            : "Enable the Grok lane (WAQIL_GROK_LANE_ENABLED) and configure OCI Responses first",
          selected: projectMode === "grok_continuous",
          choose: () => onChooseProjectMode("grok_continuous"),
        },
        {
          key: "cohere_continuous",
          label: "Command A+",
          state: cohereAvailable ? "ready" : "off",
          note: cohereAvailable
            ? "Cohere Command A+ leads every bounded step."
            : "Needs a Cohere API key configured.",
          disabled: !cohereAvailable,
          title: cohereAvailable ? undefined : "Add WAQIL_COHERE_API_KEY first",
          selected: projectMode === "cohere_continuous",
          choose: () => onChooseProjectMode("cohere_continuous"),
        },
      ];
    }
    return [
      {
        key: "local",
        label: "Local",
        state: localStateLabel(session, now),
        note: "On-device weights. Nothing leaves this machine.",
        disabled: false,
        title: undefined as string | undefined,
        selected: route === "local",
        choose: () => void chooseRoute("local"),
      },
      {
        key: "ollama_cloud",
        label: "Ollama Cloud",
        state: cloudModels.length ? `${cloudModels.length} hosted` : "off",
        note: cloudModels.length
          ? `${cloudModels.length} hosted model${cloudModels.length === 1 ? "" : "s"} on your subscription — no local memory used.`
          : "No hosted models are available to this Ollama.",
        disabled: !cloudModels.length,
        title: cloudModels.length ? undefined : "Sign in to Ollama Cloud, then refresh",
        selected: route === "ollama_cloud",
        choose: () => void chooseRoute("ollama_cloud"),
      },
      {
        key: "oci",
        label: "Cloud · Grok",
        state: ociAvailable ? "ready" : "off",
        note: ociAvailable
          ? "Grok 4.3 through OCI, for the largest context."
          : "Needs OCI configured in Settings.",
        disabled: !ociAvailable,
        title: ociAvailable ? undefined : "Configure OCI in Settings first",
        selected: route === "oci",
        choose: () => void chooseRoute("oci"),
      },
      {
        key: "cohere",
        label: "Cloud · Command A+",
        state: cohereAvailable ? "ready" : "off",
        note: cohereAvailable
          ? "Cohere Command A+ through your Cohere key."
          : "Needs a Cohere API key configured.",
        disabled: !cohereAvailable,
        title: cohereAvailable ? undefined : "Add WAQIL_COHERE_API_KEY first",
        selected: route === "cohere",
        choose: () => void chooseRoute("cohere"),
      },
    ];
    // `chooseRoute` is a stable declaration in this render scope; the routes
    // depend on what the panel is showing, not on its identity.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    project,
    projectMode,
    onChooseProjectMode,
    ollamaLaneModel,
    ociAvailable,
    cohereAvailable,
    route,
    session,
    now,
    cloudModels.length,
  ]);

  const selectedNote = routes.find((item) => item.selected)?.note ?? "";

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
          <div className="modelControlBody">
          {project ? (
            <div className="modelControlEyebrow">
              <span className="eyebrow">Whole-project mode</span>
            </div>
          ) : null}
          <div
            className="modelControlRoutes"
            role="radiogroup"
            aria-label={project ? "Project reasoning mode" : "Reasoning model"}
          >
            {routes.map((item) => (
              <button
                key={item.key}
                type="button"
                role="radio"
                aria-checked={item.selected}
                className={`modelControlRoute${item.selected ? " selected" : ""}`}
                disabled={
                  disabled
                  || item.disabled
                  || (project ? projectBusy : providerSaving || busy)
                }
                title={item.title}
                onClick={item.choose}
              >
                <strong>{item.label}</strong>
                <em>{item.state}</em>
              </button>
            ))}
          </div>
          {selectedNote ? <p className="modelControlAdvice">{selectedNote}</p> : null}

          {/* Hosted models have no weights to place, so this route carries a
              model list and nothing else — no idle window, no context size,
              no launch. Showing those controls greyed out beside a hosted
              model was what made "launch gpt-oss:120b-cloud" look like a
              thing you could do. */}
          {!project && route === "ollama_cloud" ? (
            <div className="modelControlSession">
              <div className="modelControlSessionHead">
                <span className="eyebrow">Hosted model</span>
              </div>
              <label>
                <select
                  value={cloudSelected ? model : cloudModels[0]?.id ?? ""}
                  onChange={(event) => {
                    modelTouched.current = true;
                    setModel(event.target.value);
                  }}
                  disabled={busy}
                >
                  {!cloudModels.length ? <option value="">No hosted models found</option> : null}
                  {cloudModels.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.name}{item.parameter_size ? ` · ${item.parameter_size}` : ""}
                    </option>
                  ))}
                </select>
              </label>
              <p className="modelControlHosted">
                Runs on Ollama Cloud under your subscription. Nothing is held in
                this machine&rsquo;s memory, and there is nothing to launch or unload.
              </p>
              <button
                className="modelControlLaunch"
                type="button"
                onClick={() => void pinHosted()}
                disabled={busy || !model || !cloudSelected || preference?.model === model}
              >
                {busy
                  ? "Working…"
                  : preference?.model === model
                    ? `${shortModel(model)} is selected`
                    : `Use ${shortModel(model)}`}
              </button>
            </div>
          ) : null}

          {showLocalSession && (project || route === "local") ? (
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
                <select
                  value={cloudSelected ? (localModels[0]?.id ?? "") : model}
                  onChange={(event) => {
                    modelTouched.current = true;
                    setModel(event.target.value);
                  }}
                  disabled={busy}
                >
                  {!localModels.length ? <option value="">No installed models found</option> : null}
                  {localModels.map((item) => (
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
              <button className="modelControlLaunch" type="button" onClick={() => void launch()} disabled={busy || !model || cloudSelected}>
                {busy ? "Working…" : session?.state === "ready" && model === session.selected_model ? "Relaunch with settings" : `Launch ${shortModel(model)}`}
              </button>
            </div>
          ) : null}

          {/* Each role reads as a sentence and opens only when you ask it to.
              Collapsed, a ladder is "Coder · deepseek-v4-pro → qwen3-coder" —
              the two facts worth knowing at a glance — and the three selects
              that state them appear for one role at a time. */}
          <div className="modelControlSession">
            <div className="modelControlSessionHead">
              <span className="eyebrow">Roles &amp; backups</span>
            </div>
            {ROLE_ROWS.map(([role, label, hint]) => {
              const expanded = openRole === role;
              return (
                <div key={role} className={`modelControlRole${expanded ? " open" : ""}`}>
                  <button
                    type="button"
                    className="modelControlRoleHead"
                    aria-expanded={expanded}
                    title={hint}
                    onClick={() => setOpenRole(expanded ? null : role)}
                  >
                    <span className="modelControlRoleName">{label}</span>
                    <span className="modelControlRoleChain">{chainSummary(chains[role])}</span>
                    <b aria-hidden="true">⌄</b>
                  </button>
                  {expanded ? (
                    <div className="modelControlRoleRungs">
                      {[0, 1, 2].map((slot) => (
                        <label key={slot}>
                          <span>{slot === 0 ? "Primary" : `Backup ${slot}`}</span>
                          <select
                            aria-label={`${label} ${slot === 0 ? "primary" : `backup ${slot}`}`}
                            value={encodeRung(chains[role]?.[slot])}
                            onChange={(event) => setRung(role, slot, event.target.value)}
                            disabled={busy || providerSaving}
                          >
                            <option value="">{slot === 0 ? "Current selection" : "— none —"}</option>
                            {(session?.models ?? []).map((item) => (
                              <option key={item.id} value={`local:${item.id}`}>{item.name}</option>
                            ))}
                            {cohereAvailable ? <option value="cohere:">Command A+ (Cohere)</option> : null}
                            {ociAvailable ? <option value="oci:">Grok (OCI)</option> : null}
                            {clineAvailable
                              ? CLINE_MODELS.map((item) => (
                                  <option key={item} value={`cline:${item}`}>
                                    {item.split("/").pop()} (Cline)
                                  </option>
                                ))
                              : null}
                          </select>
                        </label>
                      ))}
                    </div>
                  ) : null}
                </div>
              );
            })}
            <p className="modelControlAdvice">
              First rung is the primary; a lane that stops answering falls to the
              next rung mid-run. The coder ladder steers project builds.
            </p>
          </div>
          </div>

          {/* Pinned outside the scrolling body: an unsaved ladder was exactly
              the control that used to fall off the bottom of the window. */}
          {chainsDirty ? (
            <div className="modelControlFoot">
              <span>Ladder not saved</span>
              <button
                type="button"
                className="modelControlLaunch"
                onClick={() => void saveChains()}
                disabled={busy || providerSaving}
              >
                {busy ? "Saving…" : "Save"}
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
