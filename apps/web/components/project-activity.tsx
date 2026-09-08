"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import { runEventSummary, runEventTitle, runEventTone } from "@/components/run-timeline";
import type { RunEventV1 } from "@/lib/types";

const HIDDEN_EVENT_TYPES = new Set([
  "message.delta",
  "message.reasoning",
  "assistant.message",
  "message.completed",
]);

function actorFor(event: RunEventV1): string | null {
  const role = typeof event.payload.role === "string" ? event.payload.role : "";
  if (role === "planner") return "Planner";
  if (role === "coder") return "Coder";
  if (role === "quality" || role === "reviewer") return "Reviewer";
  if (event.type.startsWith("project.direction")) return "Orchestrator";
  if (event.type === "project.agent_step" || event.type.includes("code_authored")) return "Coder";
  if (event.type === "project.phase" || event.type.includes("build_planned") || event.type.includes("plan_revised")) return "Planner";
  if (event.type.startsWith("project.check") || event.type.includes("verification") || event.type.includes("policy")) return "Verifier";
  if (event.type.includes("approval") || event.type.includes("interrupted")) return "Orchestrator";
  return null;
}

function modelFor(event: RunEventV1): string | null {
  const model = event.payload.model ?? event.payload.provider;
  return typeof model === "string" && model ? model.split("/").pop() ?? model : null;
}

type ActivityStep = {
  id: string;
  title: string;
  summary: string;
  tone: string;
  meta: string | null;
  timestamp?: string;
};

function compactSummary(value: string): string {
  const singleLine = value.replace(/\s+/g, " ").trim();
  return singleLine.length > 180 ? `${singleLine.slice(0, 177).trimEnd()}…` : singleLine;
}

function stepsFrom(events: readonly RunEventV1[]): ActivityStep[] {
  const ordered = [...events]
    .filter((event) => !HIDDEN_EVENT_TYPES.has(event.type) && !event.type.includes("delta"))
    .sort((a, b) => a.sequence - b.sequence);
  const steps: ActivityStep[] = [];
  for (const event of ordered) {
    const title = runEventTitle(event.type, event.payload);
    const summary = compactSummary(runEventSummary(event));
    const meta = [actorFor(event), modelFor(event)].filter(Boolean).join(" · ") || null;
    const prior = steps[steps.length - 1];
    if (prior && prior.title === title && prior.summary === summary && prior.meta === meta) continue;
    steps.push({
      id: event.id || `${event.run_id}-${event.sequence}`,
      title,
      summary,
      tone: runEventTone(event),
      meta,
      timestamp: event.timestamp,
    });
  }
  return steps.slice(-18);
}

export function ProjectActivity({
  events,
  reasoning,
  live,
  attention = false,
  stageLabel,
  projectName,
}: {
  events: RunEventV1[];
  reasoning?: string;
  live: boolean;
  attention?: boolean;
  stageLabel?: string | null;
  projectName?: string | null;
}) {
  const [manual, setManual] = useState<boolean | null>(null);
  const open = manual ?? (live || attention);
  const bodyRef = useRef<HTMLDivElement>(null);
  const steps = useMemo(() => stepsFrom(events), [events]);
  const latest = steps[steps.length - 1];
  const activity = stageLabel || latest?.title || "Understanding the task";
  const context = latest?.summary || (projectName ? `Preparing the next safe step in ${projectName}.` : "Choosing the next safe step.");

  useEffect(() => {
    if (!open) return;
    const body = bodyRef.current;
    if (body) body.scrollTop = body.scrollHeight;
  }, [attention, live, open, reasoning, steps.length]);

  return (
    <section className={`projectActivity ${live ? "isLive" : attention ? "isAttention" : "isSettled"} ${open ? "isOpen" : ""}`}>
      <button type="button" onClick={() => setManual(!open)} aria-expanded={open}>
        <span className="projectActivityPulse" aria-hidden="true"><i /><i /><i /></span>
        <span className="projectActivityHeading">
          <strong>{attention ? "Metis is waiting for you" : live ? `Metis is working on ${activity.replace(/…$/, "").replace(/^Working\s*/i, "")}` : "See how Metis worked"}</strong>
          <small>{projectName ? `${projectName} · ${context}` : context}</small>
        </span>
        <span className="projectActivityCount">{steps.length || 1}</span>
        <span className="projectActivityChevron" aria-hidden="true">⌄</span>
      </button>

      {open ? (
        <div className="projectActivityBody" ref={bodyRef}>
          <ol>
            {!steps.length ? (
              <li className="tone-model isCurrent">
                <span className="projectActivityNode" />
                <div><strong>{activity}</strong><p>{context}</p></div>
              </li>
            ) : steps.map((step, index) => (
              <li className={`tone-${step.tone} ${live && index === steps.length - 1 ? "isCurrent" : ""}`} key={step.id}>
                <span className="projectActivityNode" />
                <div>
                  <span className="projectActivityStepHead">
                    <strong>{step.title}</strong>
                    {step.timestamp ? <time>{new Date(step.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</time> : null}
                  </span>
                  <p>{step.summary}</p>
                  {step.meta ? <small>{step.meta}</small> : null}
                </div>
              </li>
            ))}
          </ol>
          {reasoning?.trim() ? (
            <details className="projectReasoning" open={live || undefined}>
              <summary><span>Model reasoning</span><small>Available from this model</small></summary>
              <p>{reasoning}</p>
            </details>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}
