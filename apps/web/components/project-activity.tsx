"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import { runEventSummary, runEventTitle, runEventTone } from "@/components/run-timeline";
import type { RunEventV1 } from "@/lib/types";

// The conversation shows milestones. The task panel retains the complete
// event log for anyone who needs the model, policy, and retrieval details.
const VISIBLE_EVENT_TYPES = new Set([
  "stage.entered",
  "context.knowledge_error",
  "project.check_result",
  "project.build_checked",
  "project.staged_verified",
  "project.vertical_slice_checked",
  "project.coding_round",
  "tool.started",
  "tool.completed",
  "tool.evaluated",
  "artifact.created",
  "run.model_fallback",
  "run.model_exhausted",
  "run.failed",
  "run.cancelled",
]);

function milestoneFor(event: RunEventV1): { title: string; summary: string } | null {
  if (event.type === "tool.started" || event.type === "tool.completed" || event.type === "tool.evaluated") {
    const name = ["display_name", "tool_name", "tool", "name", "slug"]
      .map((key) => event.payload[key])
      .find((value): value is string => typeof value === "string" && Boolean(value.trim()));
    const summary = [event.payload.summary, event.payload.message]
      .find((value): value is string => typeof value === "string" && Boolean(value.trim()))?.trim() ?? "";
    // Tool plumbing without a useful label belongs in the full task log.
    if (!name && !summary) return null;
    const label = name?.split(/[.:/]/).at(-1)?.replace(/[_-]+/g, " ").trim();
    const action = event.type === "tool.started" ? "Using" : event.type === "tool.completed" ? "Finished" : "Checked";
    return { title: label ? `${action} ${label}` : `${action} a tool`, summary };
  }
  if (event.type !== "stage.entered") {
    return { title: runEventTitle(event.type, event.payload), summary: runEventSummary(event) };
  }
  const stage = event.payload.stage;
  if (["ingesting", "retrieving", "embedding", "reranking"].includes(String(stage))) {
    return { title: "Finding context", summary: "Reading the request and relevant information." };
  }
  if (["synthesizing", "revising", "reviewing"].includes(String(stage))) {
    return { title: "Preparing the answer", summary: "Writing and checking the response." };
  }
  return { title: runEventTitle(event.type, event.payload), summary: runEventSummary(event) };
}

type ActivityStep = {
  id: string;
  title: string;
  summary: string;
  tone: string;
};

function compactSummary(value: string): string {
  const singleLine = value.replace(/\s+/g, " ").trim();
  return singleLine.length > 180 ? `${singleLine.slice(0, 177).trimEnd()}…` : singleLine;
}

function stepsFrom(events: readonly RunEventV1[]): ActivityStep[] {
  const ordered = [...events]
    .filter((event) => VISIBLE_EVENT_TYPES.has(event.type))
    .sort((a, b) => a.sequence - b.sequence);
  const steps: ActivityStep[] = [];
  for (const event of ordered) {
    const milestone = milestoneFor(event);
    if (!milestone) continue;
    const title = milestone.title;
    const summary = compactSummary(milestone.summary);
    const prior = steps[steps.length - 1];
    if (prior && prior.title === title && prior.summary === summary) continue;
    steps.push({
      id: event.id || `${event.run_id}-${event.sequence}`,
      title,
      summary,
      tone: runEventTone(event),
    });
  }
  return steps.slice(-8);
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
  const [open, setOpen] = useState(false);
  const bodyRef = useRef<HTMLDivElement>(null);
  const steps = useMemo(() => stepsFrom(events), [events]);
  const latest = steps[steps.length - 1];
  const activity = latest?.title || stageLabel || "Thinking";
  const context = latest?.summary || (projectName ? `Working in ${projectName}.` : "Working on your request.");

  useEffect(() => {
    if (!open) return;
    const body = bodyRef.current;
    if (body) body.scrollTop = body.scrollHeight;
  }, [attention, live, open, reasoning, steps.length]);

  return (
    <section className={`projectActivity ${live ? "isLive" : attention ? "isAttention" : "isSettled"} ${open ? "isOpen" : ""}`}>
      <button type="button" onClick={() => setOpen((wasOpen) => !wasOpen)} aria-expanded={open} aria-label={`${attention ? "Waiting for your input" : live ? "Metis is working" : "Metis activity"}. ${open ? "Hide" : "Show"} activity details`}>
        <span className="projectActivityPulse" aria-hidden="true" />
        <span className="projectActivityHeading">
          <strong>{attention ? "Waiting for your input" : live ? "Metis is working" : "Activity"}</strong>
          {live ? <small>{activity.replace(/…$/, "")}</small> : null}
        </span>
        <span className="projectActivityAction">{open ? "Hide" : "Details"}</span>
        <span className="projectActivityChevron" aria-hidden="true">⌄</span>
      </button>

      {open ? (
        <div className="projectActivityBody" ref={bodyRef}>
          <ol role="list">
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
                  </span>
                  {step.summary ? <p>{step.summary}</p> : null}
                </div>
              </li>
            ))}
          </ol>
          {reasoning?.trim() ? (
            <details className="projectReasoning">
              <summary><span>Model reasoning</span><small>Available from this model</small></summary>
              <p>{reasoning}</p>
            </details>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}
