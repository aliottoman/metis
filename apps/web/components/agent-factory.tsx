"use client";

// The agent factory: draft → review → deploy → test. The review screen is the
// approval surface — what it shows is exactly what deploys, and Deploy asks
// once, naming what it is about to create in the owner's ElevenLabs account.

import { useCallback, useEffect, useState } from "react";

import {
  agentDemoUrl,
  deployAgent,
  draftAgent,
  getAgent,
  getAgentAvailability,
  listAgents,
  removeAgent,
  runAgentTests,
  updateAgentSpec,
} from "@/lib/api";
import {
  EMPTY_BRIEF,
  briefToRequest,
  embedSnippet,
  splitLines,
  statusLabel,
  testsSummary,
  validateBrief,
  type AgentBriefDraft,
} from "@/lib/agents";
import type {
  AgentFactoryAvailability,
  AgentKnowledge,
  AgentSpec,
  AgentTest,
  VoiceAgent,
} from "@/lib/types";

// Named platform models the review screen offers; empty is the platform default.
const LLM_OPTIONS = ["", "gemini-2.5-flash", "gpt-4.1-mini", "claude-sonnet-4-5"];

export function AgentFactory() {
  const [availability, setAvailability] = useState<AgentFactoryAvailability | null>(null);
  const [agents, setAgents] = useState<VoiceAgent[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const selected = agents.find((agent) => agent.id === selectedId) ?? null;

  const replace = useCallback((agent: VoiceAgent) => {
    setAgents((current) =>
      current.some((item) => item.id === agent.id)
        ? current.map((item) => (item.id === agent.id ? agent : item))
        : [agent, ...current],
    );
  }, []);

  useEffect(() => {
    void getAgentAvailability().then(setAvailability).catch(() => setAvailability(null));
    void listAgents().then(setAgents).catch((loadError: Error) => setError(loadError.message));
  }, []);

  // A test run is a background job on the platform; read it back until it settles.
  useEffect(() => {
    if (!selected || selected.tests_stage !== "running") return;
    const timer = window.setInterval(() => {
      void getAgent(selected.id).then(replace).catch(() => undefined);
    }, 4000);
    return () => window.clearInterval(timer);
  }, [selected, replace]);

  return (
    <div className="workspacePage agentsPage">
      <header className="pageHeader">
        <div>
          <span className="eyebrow">ElevenLabs</span>
          <h1>Agents</h1>
          <p>
            A prospect brief in, a deployed voice agent out: prompt, knowledge,
            simulation tests, and a demo page. Nothing reaches ElevenLabs until you
            have read it and said so.
          </p>
        </div>
      </header>

      {error ? (
        <div className="composerError" role="alert">
          <span>!</span>
          <p>{error}</p>
          <button className="errorDismiss" type="button" aria-label="Dismiss" onClick={() => setError(null)}>
            ×
          </button>
        </div>
      ) : null}

      <div className="agentLayout">
        <aside className="agentList" aria-label="Agents">
          <button
            type="button"
            className={`agentListItem isNew ${selectedId === null ? "selected" : ""}`}
            onClick={() => setSelectedId(null)}
          >
            + New agent
          </button>
          {agents.map((agent) => (
            <button
              key={agent.id}
              type="button"
              className={`agentListItem ${agent.id === selectedId ? "selected" : ""}`}
              onClick={() => setSelectedId(agent.id)}
            >
              <strong>{agent.spec.name}</strong>
              <span>
                <i className={`agentStatusDot is-${agent.status}`} aria-hidden="true" />
                {agent.company} · {statusLabel(agent.status)}
              </span>
            </button>
          ))}
        </aside>

        {selected ? (
          <AgentReview
            key={selected.id}
            agent={selected}
            onChange={replace}
            onRemoved={(agentId) => {
              setAgents((current) => current.filter((item) => item.id !== agentId));
              setSelectedId(null);
            }}
            onError={setError}
          />
        ) : (
          <BriefForm
            availability={availability}
            onDrafted={(agent) => {
              replace(agent);
              setSelectedId(agent.id);
            }}
            onError={setError}
          />
        )}
      </div>
    </div>
  );
}

// -- the brief ---------------------------------------------------------------

function BriefForm({
  availability,
  onDrafted,
  onError,
}: {
  availability: AgentFactoryAvailability | null;
  onDrafted: (agent: VoiceAgent) => void;
  onError: (message: string) => void;
}) {
  const [draft, setDraft] = useState<AgentBriefDraft>(EMPTY_BRIEF);
  const [busy, setBusy] = useState(false);
  const problems = validateBrief(draft);
  const request = briefToRequest(draft);
  const blocked = availability !== null && !availability.available;

  const update = (field: keyof AgentBriefDraft, value: string) =>
    setDraft((current) => ({ ...current, [field]: value }));

  async function submit() {
    if (!request || busy) return;
    setBusy(true);
    try {
      onDrafted(await draftAgent(request));
      setDraft(EMPTY_BRIEF);
    } catch (draftError) {
      onError(draftError instanceof Error ? draftError.message : "The draft failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="agentPanel" aria-label="New agent brief">
      {blocked ? (
        <div className="notice" role="status">
          <strong>The factory isn&apos;t configured yet</strong>
          <ul>
            {availability?.missing.map((item) => <li key={item}>{item}</li>)}
          </ul>
        </div>
      ) : null}
      <div className="agentFormGrid">
        <label className="agentField">
          <span>Company</span>
          <input type="text" value={draft.company} maxLength={200} placeholder="Batelco"
            onChange={(event) => update("company", event.target.value)} />
        </label>
        <label className="agentField">
          <span>Pages to read (one per line)</span>
          <textarea value={draft.source_urls} rows={3} placeholder="https://www.batelco.com/help/billing"
            onChange={(event) => update("source_urls", event.target.value)} />
          {problems.source_urls ? <small className="agentFieldProblem">{problems.source_urls}</small> : null}
        </label>
      </div>
      <label className="agentField">
        <span>What the agent is for</span>
        <textarea value={draft.brief} rows={4} maxLength={6000}
          placeholder="Answer billing questions for postpaid customers, book a callback when it can't, and never quote a price it wasn't given."
          onChange={(event) => update("brief", event.target.value)} />
      </label>
      <label className="agentField">
        <span>Notes and pasted material (optional)</span>
        <textarea value={draft.notes} rows={6} maxLength={40000}
          placeholder="Policies, product facts, an FAQ, a call transcript…"
          onChange={(event) => update("notes", event.target.value)} />
      </label>
      <div className="agentActions">
        <button type="button" className="agentActionButton isPrimary" disabled={!request || busy || blocked} onClick={() => void submit()}>
          {busy ? "Reading the material and drafting…" : "Draft the agent"}
        </button>
        <span className="mutedMeta">Drafting reads the pages and asks your selected model. It creates nothing in ElevenLabs.</span>
      </div>
    </section>
  );
}

// -- review, deploy, test ----------------------------------------------------

function AgentReview({
  agent,
  onChange,
  onRemoved,
  onError,
}: {
  agent: VoiceAgent;
  onChange: (agent: VoiceAgent) => void;
  onRemoved: (agentId: string) => void;
  onError: (message: string) => void;
}) {
  const [spec, setSpec] = useState<AgentSpec>(agent.spec);
  const [dirty, setDirty] = useState(false);
  const [busy, setBusy] = useState<"" | "save" | "deploy" | "tests" | "remove">("");
  const [confirming, setConfirming] = useState<"" | "deploy" | "remove">("");
  const [copied, setCopied] = useState(false);

  const edit = (patch: Partial<AgentSpec>) => {
    setSpec((current) => ({ ...current, ...patch }));
    setDirty(true);
  };

  async function run<T>(what: typeof busy, work: () => Promise<T>): Promise<T | null> {
    setBusy(what);
    try {
      return await work();
    } catch (workError) {
      onError(workError instanceof Error ? workError.message : "That didn't work.");
      return null;
    } finally {
      setBusy("");
    }
  }

  async function save(): Promise<VoiceAgent | null> {
    const saved = await run("save", () => updateAgentSpec(agent.id, spec));
    if (saved) {
      onChange(saved);
      setDirty(false);
    }
    return saved;
  }

  async function deploy() {
    setConfirming("");
    // Unsaved edits deploy too: what is on screen is what goes.
    if (dirty && !(await save())) return;
    const deployed = await run("deploy", () => deployAgent(agent.id));
    if (deployed) onChange(deployed);
  }

  async function remove() {
    setConfirming("");
    const done = await run("remove", () => removeAgent(agent.id).then(() => true));
    if (done) onRemoved(agent.id);
  }

  async function copySnippet() {
    try {
      await navigator.clipboard.writeText(embedSnippet(agent.elevenlabs_agent_id));
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      onError("The clipboard is not available here; the snippet is shown below.");
    }
  }

  const deployed = agent.status === "deployed" && Boolean(agent.elevenlabs_agent_id);

  return (
    <section className="agentPanel" aria-label={`Review ${spec.name}`}>
      <header className="agentReviewHead">
        <div>
          <span className="eyebrow">{agent.company}</span>
          <h2>{spec.name}</h2>
          <p className="mutedMeta">
            <i className={`agentStatusDot is-${agent.status}`} aria-hidden="true" />
            {statusLabel(agent.status)}
            {agent.elevenlabs_agent_id ? ` · ${agent.elevenlabs_agent_id}` : ""}
            {dirty ? " · unsaved edits" : ""}
          </p>
        </div>
      </header>
      {agent.error ? <p className="agentFieldProblem" role="alert">{agent.error}</p> : null}

      <div className="agentFormGrid">
        <label className="agentField">
          <span>Name</span>
          <input type="text" value={spec.name} maxLength={80} onChange={(event) => edit({ name: event.target.value })} />
        </label>
        <label className="agentField">
          <span>Voice id</span>
          <input type="text" value={spec.voice_id} maxLength={80} onChange={(event) => edit({ voice_id: event.target.value })} />
        </label>
        <label className="agentField">
          <span>Model</span>
          <select value={spec.llm} onChange={(event) => edit({ llm: event.target.value })}>
            {LLM_OPTIONS.map((option) => (
              <option key={option} value={option}>{option || "Platform default"}</option>
            ))}
          </select>
        </label>
        <label className="agentField">
          <span>Language</span>
          <input type="text" value={spec.language} maxLength={8} onChange={(event) => edit({ language: event.target.value })} />
        </label>
      </div>
      <label className="agentField">
        <span>First message</span>
        <input type="text" value={spec.first_message} maxLength={400} onChange={(event) => edit({ first_message: event.target.value })} />
      </label>
      <label className="agentField">
        <span>System prompt</span>
        <textarea value={spec.system_prompt} rows={14} maxLength={12000} onChange={(event) => edit({ system_prompt: event.target.value })} />
      </label>

      <KnowledgeEditor items={spec.knowledge} onChange={(knowledge) => edit({ knowledge })} />
      <TestsEditor items={spec.tests} onChange={(tests) => edit({ tests })} />

      <div className="agentFormGrid">
        <label className="agentField">
          <span>Demo headline</span>
          <input type="text" value={spec.demo_headline} maxLength={160} onChange={(event) => edit({ demo_headline: event.target.value })} />
        </label>
        <label className="agentField">
          <span>Things to try (one per line)</span>
          <textarea value={spec.demo_prompts.join("\n")} rows={3}
            onChange={(event) => edit({ demo_prompts: splitLines(event.target.value).slice(0, 4) })} />
        </label>
      </div>

      <div className="agentActions">
        <button type="button" className="agentActionButton" disabled={!dirty || busy !== ""} onClick={() => void save()}>
          {busy === "save" ? "Saving…" : "Save"}
        </button>
        {confirming === "deploy" ? (
          <span className="agentConfirm" role="group" aria-label="Confirm the deploy">
            <span>
              {deployed ? `Update agent ${agent.elevenlabs_agent_id}` : "Create an agent"}, {spec.knowledge.length} knowledge{" "}
              {spec.knowledge.length === 1 ? "document" : "documents"} and {spec.tests.length}{" "}
              {spec.tests.length === 1 ? "test" : "tests"} in your ElevenLabs account, then run the tests?
            </span>
            <button type="button" className="agentActionButton isPrimary" onClick={() => void deploy()}>Deploy</button>
            <button type="button" className="agentActionButton" onClick={() => setConfirming("")}>Not now</button>
          </span>
        ) : (
          <button type="button" className="agentActionButton isPrimary" disabled={busy !== ""} onClick={() => setConfirming("deploy")}>
            {busy === "deploy" ? "Deploying…" : deployed ? "Redeploy" : "Deploy to ElevenLabs"}
          </button>
        )}
        {deployed ? (
          <>
            <a className="agentActionButton" href={agentDemoUrl(agent.id)} target="_blank" rel="noreferrer">
              Open demo page
            </a>
            <button type="button" className="agentActionButton" onClick={() => void copySnippet()}>
              {copied ? "Copied" : "Copy embed snippet"}
            </button>
            <button type="button" className="agentActionButton" disabled={busy !== "" || agent.tests_stage === "running"}
              onClick={() => void run("tests", () => runAgentTests(agent.id)).then((next) => next && onChange(next))}>
              Run tests
            </button>
          </>
        ) : null}
        {confirming === "remove" ? (
          <span className="agentConfirm" role="group" aria-label="Confirm removal">
            <span>Remove this agent{deployed ? " and everything it created in ElevenLabs" : ""}?</span>
            <button type="button" className="agentActionButton isStop" onClick={() => void remove()}>Remove</button>
            <button type="button" className="agentActionButton" onClick={() => setConfirming("")}>Keep</button>
          </span>
        ) : (
          <button type="button" className="agentActionButton isStop" disabled={busy !== ""} onClick={() => setConfirming("remove")}>
            {busy === "remove" ? "Removing…" : "Remove"}
          </button>
        )}
      </div>

      {deployed ? (
        <div className="agentResults">
          <div>
            <span className="eyebrow">Simulation tests</span>
            <p className="mutedMeta">{testsSummary(agent)}</p>
            <ul className="agentTestResults">
              {agent.tests.map((result) => (
                <li key={result.name}>
                  <i className={`agentStatusDot is-${result.status}`} aria-hidden="true" />
                  <strong>{result.name}</strong>
                  {result.rationale ? <p>{result.rationale}</p> : null}
                </li>
              ))}
            </ul>
          </div>
          <div>
            <span className="eyebrow">Embed anywhere</span>
            <pre className="agentSnippet">{embedSnippet(agent.elevenlabs_agent_id)}</pre>
          </div>
        </div>
      ) : null}
    </section>
  );
}

// -- the two list editors ----------------------------------------------------

function KnowledgeEditor({
  items,
  onChange,
}: {
  items: AgentKnowledge[];
  onChange: (items: AgentKnowledge[]) => void;
}) {
  const setItem = (index: number, patch: Partial<AgentKnowledge>) =>
    onChange(items.map((item, at) => (at === index ? { ...item, ...patch } : item)));
  const add = (kind: AgentKnowledge["kind"]) =>
    onChange([...items, { kind, name: kind === "url" ? "Page" : "Facts", url: "", text: "" }].slice(0, 8));
  return (
    <div className="agentListEditor">
      <div className="agentListEditorHead">
        <span className="eyebrow">Knowledge</span>
        <button type="button" className="agentActionButton" onClick={() => add("url")}>+ Page</button>
        <button type="button" className="agentActionButton" onClick={() => add("text")}>+ Text</button>
      </div>
      {items.map((item, index) => (
        <div className="agentListItemEditor" key={`${item.kind}-${index}`}>
          <div className="agentFormGrid">
            <label className="agentField">
              <span>{item.kind === "url" ? "Page" : "Document"} name</span>
              <input type="text" value={item.name} maxLength={80} onChange={(event) => setItem(index, { name: event.target.value })} />
            </label>
            {item.kind === "url" ? (
              <label className="agentField">
                <span>Address</span>
                <input type="url" value={item.url} maxLength={400} onChange={(event) => setItem(index, { url: event.target.value })} />
              </label>
            ) : null}
          </div>
          {item.kind === "text" ? (
            <label className="agentField">
              <span>Text</span>
              <textarea value={item.text} rows={4} maxLength={8000} onChange={(event) => setItem(index, { text: event.target.value })} />
            </label>
          ) : null}
          <button type="button" className="agentActionButton isStop" onClick={() => onChange(items.filter((_, at) => at !== index))}>
            Remove
          </button>
        </div>
      ))}
    </div>
  );
}

function TestsEditor({
  items,
  onChange,
}: {
  items: AgentTest[];
  onChange: (items: AgentTest[]) => void;
}) {
  const setItem = (index: number, patch: Partial<AgentTest>) =>
    onChange(items.map((item, at) => (at === index ? { ...item, ...patch } : item)));
  return (
    <div className="agentListEditor">
      <div className="agentListEditorHead">
        <span className="eyebrow">Simulation tests</span>
        <button type="button" className="agentActionButton"
          onClick={() => onChange([...items, { name: "New test", scenario: "", success_conditions: [], max_turns: 6 }].slice(0, 6))}>
          + Test
        </button>
      </div>
      {items.map((item, index) => (
        <div className="agentListItemEditor" key={index}>
          <div className="agentFormGrid">
            <label className="agentField">
              <span>Name</span>
              <input type="text" value={item.name} maxLength={80} onChange={(event) => setItem(index, { name: event.target.value })} />
            </label>
            <label className="agentField">
              <span>Max turns</span>
              <input type="number" min={2} max={12} value={item.max_turns}
                onChange={(event) => setItem(index, { max_turns: Math.min(12, Math.max(2, Number(event.target.value) || 6)) })} />
            </label>
          </div>
          <label className="agentField">
            <span>Caller and scenario</span>
            <textarea value={item.scenario} rows={3} maxLength={2000} onChange={(event) => setItem(index, { scenario: event.target.value })} />
          </label>
          <label className="agentField">
            <span>Passes when (one condition per line)</span>
            <textarea value={item.success_conditions.join("\n")} rows={3}
              onChange={(event) => setItem(index, { success_conditions: splitLines(event.target.value).slice(0, 5) })} />
          </label>
          <button type="button" className="agentActionButton isStop" onClick={() => onChange(items.filter((_, at) => at !== index))}>
            Remove
          </button>
        </div>
      ))}
    </div>
  );
}
