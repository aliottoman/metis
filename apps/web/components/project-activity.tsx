"use client";

import { useId, useMemo, useState } from "react";
import { ChevronDown, CircleAlert, CircleCheck, LoaderCircle } from "lucide-react";

import { runEventSummary, runEventTitle, runEventTone } from "@/components/run-timeline";
import type { RunEventV1 } from "@/lib/types";

// Keep the conversation brief. The task overview retains the event-by-event
// record, including model and retrieval details.
const VISIBLE = new Set([
  "stage.entered", "context.knowledge_error", "project.check_result",
  "project.build_checked", "project.staged_verified", "project.vertical_slice_checked",
  "project.coding_round", "tool.started", "tool.completed", "tool.evaluated",
  "artifact.created", "run.model_fallback", "run.model_exhausted",
  "run.failed", "run.cancelled",
]);

type ActivityGroup = {
  key: string;
  title: string;
  summary: string;
  tone: string;
  count: number;
  sequence: number;
};

function compact(value: string): string {
  const line = value.replace(/\s+/g, " ").trim();
  return line.length > 180 ? `${line.slice(0, 177).trimEnd()}…` : line;
}

function usefulSummary(event: RunEventV1): string {
  const summary = runEventSummary(event);
  return summary === "Recorded by the control plane" ? "" : compact(summary);
}

function activityFor(event: RunEventV1): Omit<ActivityGroup, "count" | "sequence"> | null {
  const payload = event.payload;
  const tone = runEventTone(event);
  if (event.type === "stage.entered") {
    const stage = String(payload.stage ?? "");
    if (stage === "ingesting") return { key: "reading", title: "Reading the request", summary: "Understanding what you need.", tone };
    if (["retrieving", "embedding", "reranking"].includes(stage)) return { key: "context", title: "Finding context", summary: "Gathering relevant information.", tone };
    if (stage === "planning") return { key: "planning", title: "Planning", summary: "Choosing how to handle the request.", tone };
    if (["synthesizing", "authoring", "designing", "drafting", "rendering"].includes(stage)) return { key: "writing", title: "Writing the response", summary: "Preparing the answer.", tone };
    if (["reviewing", "revising"].includes(stage)) return { key: "review", title: "Checking the response", summary: "Reviewing the answer before sending it.", tone };
    if (stage === "project_check") return { key: "checks", title: "Checking the work", summary: "Running project checks.", tone };
    if (["project_reasoning", "project_tool"].includes(stage)) return { key: "project", title: "Working in the project", summary: "Making progress on the requested change.", tone };
  }
  if (event.type.startsWith("tool.")) {
    const name = ["display_name", "tool_name", "tool", "name", "slug"]
      .map((key) => payload[key])
      .find((value): value is string => typeof value === "string" && Boolean(value.trim()));
    const summary = [payload.summary, payload.message]
      .find((value): value is string => typeof value === "string" && Boolean(value.trim()))?.trim();
    if (!name && !summary) return null;
    const label = name?.split(/[.:/]/).at(-1)?.replace(/[_-]+/g, " ").trim();
    const verb = event.type === "tool.started" ? "Using" : event.type === "tool.completed" ? "Used" : "Checked";
    return { key: "tools", title: "Using tools", summary: compact([label ? `${verb} ${label}` : "Tool activity", summary].filter(Boolean).join(" · ")), tone };
  }
  if ((event.type.startsWith("project.") && event.type.includes("check")) || ["project.staged_verified", "project.vertical_slice_checked"].includes(event.type)) {
    return { key: "checks", title: "Checking the work", summary: usefulSummary(event), tone };
  }
  if (event.type === "project.coding_round") return { key: "project", title: "Working in the project", summary: usefulSummary(event), tone };
  if (event.type === "artifact.created") return { key: "outputs", title: "Preparing output", summary: usefulSummary(event), tone };
  if (event.type === "run.model_fallback") return { key: "model", title: "Switching models", summary: usefulSummary(event), tone };
  if (["run.failed", "run.model_exhausted", "context.knowledge_error"].includes(event.type)) {
    return { key: "problem", title: "Needs attention", summary: usefulSummary(event), tone: "danger" };
  }
  if (event.type === "run.cancelled") return { key: "stopped", title: "Stopped", summary: usefulSummary(event), tone };
  return { key: event.type, title: runEventTitle(event.type, payload), summary: usefulSummary(event), tone };
}

function groupsFrom(events: readonly RunEventV1[]): ActivityGroup[] {
  const groups = new Map<string, ActivityGroup>();
  for (const event of [...events].sort((a, b) => a.sequence - b.sequence)) {
    if (!VISIBLE.has(event.type)) continue;
    const activity = activityFor(event);
    if (!activity) continue;
    const previous = groups.get(activity.key);
    groups.set(activity.key, {
      ...activity,
      summary: activity.summary || previous?.summary || "",
      count: (previous?.count ?? 0) + 1,
      sequence: event.sequence,
    });
  }
  return [...groups.values()].sort((a, b) => a.sequence - b.sequence);
}

export function ProjectActivity({
  events, reasoning, live, attention = false, stageLabel, projectName,
}: {
  events: RunEventV1[];
  reasoning?: string;
  live: boolean;
  attention?: boolean;
  stageLabel?: string | null;
  projectName?: string | null;
}) {
  const [open, setOpen] = useState(false);
  const detailId = useId();
  const groups = useMemo(() => groupsFrom(events), [events]);
  const latest = groups.at(-1);
  const failed = !live && (latest?.key === "problem" || latest?.tone === "danger");
  const status = attention ? "Waiting for your input" : failed ? "Needs attention" : live ? (stageLabel?.trim() || latest?.title || "Working") : "View activity";
  const secondary = live ? latest?.title && latest.title !== status ? latest.title : projectName ? `Working in ${projectName}` : "Metis is working" : groups.length ? `${groups.length} ${groups.length === 1 ? "part" : "parts"}` : "";
  const recent = groups.slice(-4);
  const earlier = groups.slice(0, -4);

  return (
    <section className={`projectActivity ${live ? "isLive" : attention || failed ? "isAttention" : "isSettled"} ${open ? "isOpen" : ""}`}>
      <button type="button" onClick={() => setOpen((wasOpen) => !wasOpen)} aria-expanded={open} aria-controls={open ? detailId : undefined} aria-label={`${status}. ${open ? "Hide" : "Show"} activity details`}>
        <span className="projectActivityState" aria-hidden="true">{live ? <LoaderCircle size={14} /> : attention || failed ? <CircleAlert size={14} /> : <CircleCheck size={14} />}</span>
        <span className="projectActivityHeading"><strong aria-live={live ? "polite" : "off"}>{status}</strong>{secondary ? <small>{secondary}</small> : null}</span>
        <ChevronDown className="projectActivityChevron" size={14} aria-hidden="true" />
      </button>
      {open ? (
        <div className="projectActivityBody" id={detailId}>
          <ol aria-label="Activity summary">
            {!groups.length ? <li><span className="projectActivityNode" aria-hidden="true" /><div><strong>{status}</strong><p>{projectName ? `Working in ${projectName}.` : "Working on your request."}</p></div></li> : null}
            {recent.map((group) => <li className={`tone-${group.tone} ${live && group.key === latest?.key ? "isCurrent" : ""}`} key={group.key}>
              <span className="projectActivityNode" aria-hidden="true" />
              <div><span className="projectActivityStepHead"><strong>{group.title}</strong>{group.count > 1 ? <small>{group.count} updates</small> : null}</span>{group.summary ? <p>{group.summary}</p> : null}</div>
            </li>)}
          </ol>
          {earlier.length ? <details className="projectActivityEarlier"><summary>{earlier.length} earlier {earlier.length === 1 ? "part" : "parts"}<ChevronDown size={13} aria-hidden="true" /></summary><ul>{earlier.map((group) => <li key={group.key}><strong>{group.title}</strong>{group.summary ? <p>{group.summary}</p> : null}</li>)}</ul></details> : null}
          {reasoning?.trim() ? <details className="projectReasoning"><summary>Model reasoning<ChevronDown size={13} aria-hidden="true" /></summary><p>{reasoning}</p></details> : null}
        </div>
      ) : null}
    </section>
  );
}
