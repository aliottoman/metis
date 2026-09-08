"use client";

import { useMemo, useState } from "react";

import { ApprovalCard } from "@/components/approval-card";
import { approvalFrom } from "@/lib/approvals";
import {
  BACKEND_REASON_LABELS,
  humanizeToken,
  OPERATION_LABELS,
} from "@/lib/run-event-labels";
import {
  codingRoundSummary,
  codingStepSummary,
  directContractSummary,
  isFailedCodingStep,
} from "@/lib/coding-diagnostics";
import type { RunEventV1 } from "@/lib/types";

interface RunTimelineProps {
  events: RunEventV1[];
  connection: string;
  streamError?: string | null;
  onDecision: (
    approvalId: string,
    decision: "approve" | "reject",
  ) => Promise<void>;
  decidedApprovals: ReadonlySet<string>;
  decisionBusy?: string | null;
  approveLabel?: string;
}

function getText(
  payload: Record<string, unknown>,
  ...keys: string[]
): string | undefined {
  for (const key of keys) {
    if (typeof payload[key] === "string" && payload[key])
      return String(payload[key]);
  }
  return undefined;
}

function numText(value: unknown, fallback = "?"): string {
  if (typeof value === "number" && Number.isFinite(value)) return String(value);
  if (typeof value === "string" && value) return value;
  return fallback;
}

// The coarse step the control plane says it is on. The payload's `label`
// (the summary line) says what it is doing right now; this names the step.
const STAGE_TITLES: Record<string, string> = {
  ingesting: "Reading the request",
  retrieving: "Searching",
  planning: "Planning",
  synthesizing: "Writing the answer",
  revising: "Revising",
  reviewing: "Reviewing",
  rendering: "Rendering",
  authoring: "Writing the document",
  designing: "Designing",
  drafting: "Drafting a tool",
  building: "Building a tool",
  running: "Running a tool",
  project_ask: "Question for you",
  project_reasoning: "Working in the project",
  project_check: "Checking the build",
  project_tool: "Using a project tool",
};

function stageTitle(payload?: Record<string, unknown>): string {
  const stage = payload ? getText(payload, "stage") : undefined;
  if (!stage) return "Working";
  return STAGE_TITLES[stage] ?? stage.replaceAll("_", " ").replace(/^\w/, (character) => character.toUpperCase());
}

export function runEventTitle(type: string, payload?: Record<string, unknown>): string {
  if (type === "stage.entered") return stageTitle(payload);
  const exact: Record<string, string> = {
    "run.created": "Run started",
    "run.started": "Run started",
    "run.resumed": "Run resumed",
    "run.recovered": "Run recovered",
    "input.ingested": "Request read",
    "input.truncated": "Context trimmed to budget",
    "context.retrieved": "Context retrieved",
    "context.knowledge_error": "Knowledge search unavailable",
    "project.direct_contract": "Cline coding session",
    "project.check_requested": "Verification check",
    "project.coding_event": "Coding step",
    "project.coding_round": "Coding round",
    "project.build_checked": "Verification",
    "memory.retrieved": "Context retrieved",
    "plan.created": "Plan ready",
    "model.response": "Model finished",
    "model.started": "Model working",
    "model.completed": "Model finished",
    "answer.grounding_reviewed": "Grounding checked",
    "architecture.spec_created": "Architecture drafted",
    "diagram.code_created": "Diagram code ready",
    "tool.proposed": "New tool proposed",
    "tool.proposal_created": "New tool proposed",
    "tool.execution_reused": "Existing tool reused",
    "tool.evaluated": "Tool evaluated",
    "tool.started": "Tool started",
    "tool.completed": "Tool finished",
    "run.broker_call": "Brokered model call",
    "tool.definition_drafted": "Tool definition drafted",
    "tool.definition_refused": "Tool creation refused",
    "tool.definition_decided": "Definition decision",
    "tool.build_decided": "Build decision",
    "tool.code_authored": "Tool code authored",
    "tool.code_reviewed": "Tool code reviewed",
    "tool.code_review_skipped": "Code review skipped",
    "tool.output": "Tool output",
    "evaluation.completed": "Evaluation complete",
    "project.check_result": "Verification check",
    "project.verification_decided": "Verification decision",
    "project.build_planned": "Build plan",
    "project.plan_revised": "Build plan corrected",
    "project.focused": "Narrowed to one file",
    "project.phase": "Phase",
    "project.agent_step": "Project action",
    "project.vertical_slice_checked": "Vertical slice verified",
    "project.staged_verified": "Build verification",
    "run.model_fallback": "Model fallback",
    "run.model_exhausted": "Model ladder exhausted",
    "approval.required": "Approval needed",
    "approval.applied": "Approval recorded",
    "run.awaiting_approval": "Waiting for approval",
    "run.interrupted": "Waiting for approval",
    "artifact.created": "Artifact created",
    "message.created": "Reply delivered",
    "run.completed": "Run complete",
    "run.failed": "Run failed",
    "run.cancelled": "Run cancelled",
  };
  return (
    exact[type] ??
    type
      .replace(/[._-]/g, " ")
      .replace(/\b\w/g, (character) => character.toUpperCase())
  );
}

export function runEventTone(event: RunEventV1): string {
  const { type, payload } = event;
  if (type === "context.knowledge_error") return "attention";
  // Payload-dependent tones for the Tool Factory events.
  if (type === "tool.evaluated")
    return payload.passed ? "success" : "attention";
  if (type === "project.check_result") return payload.ok ? "success" : "danger";
  if (type === "project.verification_decided")
    return payload.approved ? "success" : "attention";
  if (type === "project.staged_verified")
    return Number(payload.errors ?? 0) > 0 ? "danger" : "success";
  if (type === "project.vertical_slice_checked")
    return Number(payload.errors ?? 0) > 0 ? "danger" : "success";
  if (type === "run.model_exhausted") return "danger";
  if (type === "project.coding_event")
    return isFailedCodingStep(payload) ? "danger" : "muted";
  if (type === "project.direct_contract") {
    const unresolved = Array.isArray(payload.unresolved)
      ? payload.unresolved
      : [];
    return unresolved.length ? "attention" : "success";
  }
  if (type === "project.check_requested")
    return payload.ok ? "success" : "attention";
  if (type === "project.coding_round") {
    const changed = Array.isArray(payload.changed_paths)
      ? payload.changed_paths
      : [];
    return changed.length ? "success" : "attention";
  }
  if (type === "tool.output")
    return payload.contract_ok ? "success" : "attention";
  if (type === "tool.code_reviewed")
    return payload.safe === false ? "danger" : "success";
  const exact: Record<string, string> = {
    "run.broker_call": "model",
    "tool.definition_drafted": "neutral",
    "tool.definition_refused": "attention",
    "tool.code_authored": "model",
    "tool.code_review_skipped": "attention",
  };
  if (exact[type]) return exact[type];
  if (
    type.includes("fail") ||
    type.includes("error") ||
    type.includes("reject")
  )
    return "danger";
  if (
    type.includes("approval") ||
    type.includes("interrupt") ||
    type.includes("proposal")
  )
    return "attention";
  if (
    type.includes("complete") ||
    type.includes("artifact") ||
    type.includes("approved")
  )
    return "success";
  if (type.includes("model")) return "model";
  return "neutral";
}

export function runEventSummary(event: RunEventV1): string {
  const payload = event.payload;
  if (event.type === "stage.entered") {
    return getText(payload, "label") ?? "Working…";
  }
  if (event.type === "answer.grounding_reviewed") {
    if (payload.revision)
      return "Retrieved sources went uncited — sending one revision to ground the answer.";
    if (payload.has_attachments)
      return "Answered from the attached document — kept as written.";
    if (payload.strong_retrieval)
      return "Answer is grounded in the retrieved sources.";
    return "No strongly-relevant sources to ground against.";
  }
  if (event.type === "context.knowledge_error") {
    return (
      getText(payload, "summary") ??
      "Knowledge search is unavailable. Continuing with the attached files and local context."
    );
  }
  if (event.type === "run.broker_call") {
    const role = getText(payload, "role") ?? "model";
    const template = getText(payload, "template") ?? "template";
    const model = getText(payload, "model") ?? "model";
    return `${role} · ${template} · call ${numText(payload.call_index)}/${numText(payload.budget)} · ${model}`;
  }
  if (event.type === "tool.definition_drafted") {
    const definition =
      payload.definition && typeof payload.definition === "object"
        ? (payload.definition as Record<string, unknown>)
        : {};
    return (
      getText(definition, "name") ??
      getText(payload, "slug", "name") ??
      "A new tool definition was drafted."
    );
  }
  if (event.type === "tool.definition_refused") {
    return getText(payload, "reason") ?? "Tool creation was refused.";
  }
  if (
    event.type === "tool.definition_decided" ||
    event.type === "tool.build_decided"
  ) {
    return `${getText(payload, "slug") ?? "definition"}: ${getText(payload, "status") ?? "decided"}`;
  }
  if (event.type === "tool.evaluated") {
    return `${payload.passed ? "passed" : "failed"} · score ${numText(payload.score, "n/a")}`;
  }
  if (event.type === "tool.output") {
    return `${getText(payload, "slug") ?? "tool"} · ${getText(payload, "authored_by") ?? "unknown"}`;
  }
  if (event.type === "tool.code_authored") {
    return `${getText(payload, "slug") ?? "tool"} · ${numText(payload.chars, "?")} chars of run() code`;
  }
  if (event.type === "tool.code_reviewed") {
    if (payload.safe === false) return "reviewer flagged the code as unsafe";
    return `reviewed by ${getText(payload, "reviewer") ?? "model"}${payload.improved ? " · improved" : ""}`;
  }
  if (event.type === "tool.code_review_skipped") {
    return (
      getText(payload, "reason") ??
      "code review unavailable — AST gate still applied"
    );
  }
  if (event.type === "project.check_result") {
    const name = getText(payload, "name") ?? "check";
    const failure = getText(payload, "error");
    if (failure) return `${name} · ${failure}`;
    if (payload.timed_out) return `${name} · timed out`;
    const verdict = payload.ok
      ? "passed"
      : `failed (exit ${numText(payload.exit_code, "?")})`;
    return `${name} · ${verdict} · ${numText(payload.duration_seconds, "?")}s`;
  }
  if (event.type === "project.verification_decided") {
    return payload.approved
      ? "Verification checks approved for this project."
      : "Verification checks were declined.";
  }
  if (event.type === "project.build_planned") {
    const files = Array.isArray(payload.files) ? payload.files : [];
    const intent = getText(payload, "intent");
    const after = numText(payload.after_steps, "0");
    const read = `after reading ${after} step${after === "1" ? "" : "s"}`;
    if (!files.length) return `${intent || "no files"} · ${read}`;
    return `${files.length} file(s) · ${files.join(", ")} · ${read}`;
  }
  if (event.type === "project.plan_revised") {
    const files = Array.isArray(payload.files) ? payload.files : [];
    const reason = getText(payload, "reason");
    const now = files.length ? files.join(", ") : "no new files";
    return reason ? `${reason} → ${now}` : `now ${now}`;
  }
  if (event.type === "project.focused") {
    const path = getText(payload, "path") ?? "one file";
    return `writing ${path} first — the turn was reading without converging`;
  }
  if (event.type === "project.phase") {
    return getText(payload, "phase") === "building"
      ? "exploration answered — building now"
      : "exploring the project";
  }
  if (event.type === "project.agent_step") {
    const tool = getText(payload, "tool");
    const status = getText(payload, "status") ?? "working";
    const step = numText(payload.step, "?");
    return tool
      ? `${tool.replaceAll("_", " ")} · step ${step}`
      : `${status.replaceAll("_", " ")} · step ${step}`;
  }
  if (event.type === "project.build_checked") {
    const errors = Number(payload.errors ?? 0);
    const checks = Number(payload.ran ?? 0);
    return errors
      ? `${errors} blocking problem${errors === 1 ? "" : "s"} — sent back to the coding session`
      : `${checks} check${checks === 1 ? "" : "s"} passed`;
  }
  if (event.type === "project.vertical_slice_checked") {
    const name = getText(payload, "name") ?? "Current slice";
    const errors = Number(payload.errors ?? 0);
    const checks = Number(payload.ran ?? 0);
    const files = Array.isArray(payload.files) ? payload.files.length : 0;
    return errors
      ? `${name} · ${errors} blocker${errors === 1 ? "" : "s"} · repair stays in this slice`
      : `${name} · ${files} file${files === 1 ? "" : "s"} · ${checks} check${checks === 1 ? "" : "s"} passed`;
  }
  if (event.type === "project.staged_verified") {
    const errors = Number(payload.errors ?? 0);
    const warnings = Number(payload.warnings ?? 0);
    const checks = Number(payload.ran ?? 0);
    const findings = Array.isArray(payload.findings) ? payload.findings : [];
    const first =
      findings[0] && typeof findings[0] === "object"
        ? (findings[0] as Record<string, unknown>)
        : null;
    if (errors > 0) {
      const path = first ? getText(first, "path") : undefined;
      const detail = first ? getText(first, "error") : undefined;
      const lead = `${errors} blocking problem${errors === 1 ? "" : "s"}`;
      return path && detail ? `${lead} · ${path}: ${detail}` : lead;
    }
    return `${checks} check${checks === 1 ? "" : "s"} passed${warnings ? ` · ${warnings} warning${warnings === 1 ? "" : "s"}` : ""}`;
  }
  if (event.type === "project.direct_contract") {
    return directContractSummary(payload);
  }
  if (event.type === "project.check_requested") {
    const check = getText(payload, "check") ?? "check";
    const errors = numText(payload.errors, "0");
    const warnings = numText(payload.warnings, "0");
    const used = numText(payload.checks_used, "?");
    const budget = numText(payload.checks_budget, "?");
    const verdict = payload.ok
      ? "no blocking problems"
      : `${errors} blocking problem${errors === "1" ? "" : "s"}`;
    return `${check} · ${verdict}${warnings === "0" ? "" : ` · ${warnings} warning(s)`} · check ${used} of ${budget}`;
  }
  if (event.type === "project.coding_event") {
    return codingStepSummary(payload);
  }
  if (event.type === "project.coding_round") {
    return codingRoundSummary(payload);
  }
  if (event.type === "run.model_fallback") {
    const from = getText(payload, "from") ?? "the primary";
    const to = getText(payload, "to") ?? "a backup";
    const reason = humanizeToken(
      getText(payload, "reason") ?? "unavailable",
      BACKEND_REASON_LABELS,
    );
    return `${from} stopped answering — ${reason} — continuing on ${to}`;
  }
  if (event.type === "run.model_exhausted") {
    const model = getText(payload, "model") ?? "the final model";
    const operation = humanizeToken(
      getText(payload, "operation") ?? "the requested operation",
      OPERATION_LABELS,
    );
    const reason = humanizeToken(
      getText(payload, "reason") ?? "unavailable",
      BACKEND_REASON_LABELS,
    );
    return `${model} could not finish ${operation} — ${reason} — no backup model remains`;
  }
  return (
    getText(
      payload,
      "summary",
      "message",
      "status",
      "tool_name",
      "tool",
      "node",
      "error",
    ) ??
    (event.type.includes("delta")
      ? "Streaming response"
      : "Recorded by the control plane")
  );
}

// Bookkeeping the control plane records for itself: shown only on request.
const ROUTINE = new Set(["run.created", "run.started", "input.ingested", "message.created", "message.delta", "message.reasoning", "assistant.message", "message.completed"]);
const FALLBACK_SUMMARY = "Recorded by the control plane";

function isRoutine(event: RunEventV1): boolean {
  return ROUTINE.has(event.type) || event.type.includes("delta") || runEventSummary(event) === FALLBACK_SUMMARY;
}

function seconds(from?: string, to?: string): number | null {
  if (!from || !to) return null;
  const delta = (new Date(to).getTime() - new Date(from).getTime()) / 1000;
  return Number.isFinite(delta) && delta >= 0 ? delta : null;
}

function duration(value: number | null): string {
  if (value === null) return "";
  if (value < 1) return "<1s";
  if (value < 60) return `${value.toFixed(value < 10 ? 1 : 0)}s`;
  return `${Math.floor(value / 60)}m ${Math.round(value % 60)}s`;
}

export function RunTimeline({
  events,
  connection,
  streamError,
  onDecision,
  decidedApprovals,
  decisionBusy,
  approveLabel = "Approve once",
}: RunTimelineProps) {
  const [showRoutine, setShowRoutine] = useState(false);
  const ordered = useMemo(() => [...events].sort((a, b) => a.sequence - b.sequence), [events]);
  const shown = useMemo(() => (showRoutine ? ordered : ordered.filter((event) => !isRoutine(event))), [ordered, showRoutine]);
  const routineCount = ordered.length - shown.length;
  const elapsed = duration(seconds(ordered[0]?.timestamp, ordered[ordered.length - 1]?.timestamp));
  const live = connection === "live" || connection === "connecting" || connection === "reconnecting";

  return (
    <div className="tl">
      <header className="tl-head">
        <strong>Activity</strong>
        <span className="tl-meta">
          {elapsed ? <span>{elapsed}</span> : null}
          <span className="ui-status"><span className={`ui-dot ${live ? "is-waiting" : connection === "error" ? "is-needs-review" : "is-stopped"}`} aria-hidden="true" />{connection === "closed" ? "finished" : connection}</span>
        </span>
      </header>
      {streamError ? <p className="tl-warning">Connection interrupted. Reconnecting from the last event…</p> : null}
      <div className="tl-list" aria-live="polite">
        {!ordered.length ? <p className="tl-empty">Steps, approvals and checks appear here as Metis works.</p> : null}
        {shown.map((event, index) => {
          const approval = approvalFrom(event);
          const decided = approval ? decidedApprovals.has(approval.id) || approval.status !== "pending" : false;
          const took = duration(seconds(event.timestamp, shown[index + 1]?.timestamp));
          return (
            <article className={`tl-step tone-${runEventTone(event)}`} key={event.id}>
              <span className="tl-node" aria-hidden="true" />
              <div className="tl-body">
                <div className="tl-title">
                  <strong>{runEventTitle(event.type, event.payload)}</strong>
                  <time>{index === 0 && event.timestamp ? new Date(event.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : took}</time>
                </div>
                <p>{runEventSummary(event)}</p>
                {approval ? <ApprovalCard approval={approval} decided={decided} decisionBusy={decisionBusy} onDecision={onDecision} approveLabel={approveLabel} /> : null}
              </div>
            </article>
          );
        })}
        {routineCount || showRoutine ? (
          <button type="button" className="ui-btn is-quiet is-sm tl-routine" aria-expanded={showRoutine} onClick={() => setShowRoutine((value) => !value)}>
            {showRoutine ? "Hide routine events" : `Show ${routineCount} routine events`}
          </button>
        ) : null}
      </div>
    </div>
  );
}
