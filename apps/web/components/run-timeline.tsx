"use client";

import { useMemo } from "react";

import type { ApprovalRequest, RunEventV1 } from "@/lib/types";

interface RunTimelineProps {
  events: RunEventV1[];
  connection: string;
  streamError?: string | null;
  onDecision: (approvalId: string, decision: "approve" | "reject") => Promise<void>;
  decidedApprovals: ReadonlySet<string>;
  decisionBusy?: string | null;
  approveLabel?: string;
}

function getText(payload: Record<string, unknown>, ...keys: string[]): string | undefined {
  for (const key of keys) {
    if (typeof payload[key] === "string" && payload[key]) return String(payload[key]);
  }
  return undefined;
}

function numText(value: unknown, fallback = "?"): string {
  if (typeof value === "number" && Number.isFinite(value)) return String(value);
  if (typeof value === "string" && value) return value;
  return fallback;
}

function titleFor(type: string): string {
  const exact: Record<string, string> = {
    "run.created": "Run started",
    "run.started": "Run started",
    "run.resumed": "Run resumed",
    "run.recovered": "Run recovered",
    "stage.entered": "Working",
    "input.ingested": "Request read",
    "input.truncated": "Context trimmed to budget",
    "context.retrieved": "Context retrieved",
    "context.knowledge_error": "Knowledge search unavailable",
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
  return exact[type] ?? type.replace(/[._-]/g, " ").replace(/\b\w/g, (character) => character.toUpperCase());
}

function eventTone(event: RunEventV1): string {
  const { type, payload } = event;
  if (type === "context.knowledge_error") return "attention";
  // Payload-dependent tones for the Tool Factory events.
  if (type === "tool.evaluated") return payload.passed ? "success" : "attention";
  if (type === "project.check_result") return payload.ok ? "success" : "danger";
  if (type === "project.verification_decided") return payload.approved ? "success" : "attention";
  if (type === "tool.output") return payload.contract_ok ? "success" : "attention";
  if (type === "tool.code_reviewed") return payload.safe === false ? "danger" : "success";
  const exact: Record<string, string> = {
    "run.broker_call": "model",
    "tool.definition_drafted": "neutral",
    "tool.definition_refused": "attention",
    "tool.code_authored": "model",
    "tool.code_review_skipped": "attention",
  };
  if (exact[type]) return exact[type];
  if (type.includes("fail") || type.includes("error") || type.includes("reject")) return "danger";
  if (type.includes("approval") || type.includes("interrupt") || type.includes("proposal")) return "attention";
  if (type.includes("complete") || type.includes("artifact") || type.includes("approved")) return "success";
  if (type.includes("model")) return "model";
  return "neutral";
}

function eventSummary(event: RunEventV1): string {
  const payload = event.payload;
  if (event.type === "stage.entered") {
    return getText(payload, "label") ?? "Working…";
  }
  if (event.type === "answer.grounding_reviewed") {
    if (payload.revision) return "Retrieved sources went uncited — sending one revision to ground the answer.";
    if (payload.has_attachments) return "Answered from the attached document — kept as written.";
    if (payload.strong_retrieval) return "Answer is grounded in the retrieved sources.";
    return "No strongly-relevant sources to ground against.";
  }
  if (event.type === "context.knowledge_error") {
    return getText(payload, "summary")
      ?? "Knowledge search is unavailable. Continuing with the attached files and local context.";
  }
  if (event.type === "run.broker_call") {
    const role = getText(payload, "role") ?? "model";
    const template = getText(payload, "template") ?? "template";
    const model = getText(payload, "model") ?? "model";
    return `${role} · ${template} · call ${numText(payload.call_index)}/${numText(payload.budget)} · ${model}`;
  }
  if (event.type === "tool.definition_drafted") {
    const definition = payload.definition && typeof payload.definition === "object"
      ? (payload.definition as Record<string, unknown>)
      : {};
    return getText(definition, "name") ?? getText(payload, "slug", "name") ?? "A new tool definition was drafted.";
  }
  if (event.type === "tool.definition_refused") {
    return getText(payload, "reason") ?? "Tool creation was refused.";
  }
  if (event.type === "tool.definition_decided" || event.type === "tool.build_decided") {
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
    return getText(payload, "reason") ?? "code review unavailable — AST gate still applied";
  }
  if (event.type === "project.check_result") {
    const name = getText(payload, "name") ?? "check";
    const failure = getText(payload, "error");
    if (failure) return `${name} · ${failure}`;
    if (payload.timed_out) return `${name} · timed out`;
    const verdict = payload.ok ? "passed" : `failed (exit ${numText(payload.exit_code, "?")})`;
    return `${name} · ${verdict} · ${numText(payload.duration_seconds, "?")}s`;
  }
  if (event.type === "project.verification_decided") {
    return payload.approved
      ? "Verification checks approved for this project."
      : "Verification checks were declined.";
  }
  return getText(payload, "summary", "message", "status", "tool_name", "tool", "node", "error") ??
    (event.type.includes("delta") ? "Streaming response" : "Recorded by the control plane");
}

function approvalFrom(event: RunEventV1): ApprovalRequest | null {
  // Only the persisted request event is actionable. Status events such as
  // run.awaiting_approval and approval.applied do not carry an approval ID.
  if (event.type !== "approval.required") return null;
  const nested = event.payload.approval;
  const payload = nested && typeof nested === "object" ? nested as Record<string, unknown> : event.payload;
  const id = payload.id ?? payload.approval_id;
  if (typeof id !== "string" || !id) return null;
  return {
    id: String(id),
    run_id: event.run_id,
    title: String(payload.title ?? payload.action ?? "Approve this action?"),
    summary: String(payload.summary ?? payload.description ?? "Metis needs your permission before it can continue."),
    risk_level: payload.risk_level ? String(payload.risk_level) as ApprovalRequest["risk_level"] : undefined,
    permissions: Array.isArray(payload.permissions) ? payload.permissions.map(String) : [],
    action_digest: payload.action_digest ? String(payload.action_digest) : payload.input_digest ? String(payload.input_digest) : undefined,
    status: payload.status ? String(payload.status) as ApprovalRequest["status"] : "pending",
  };
}

export function RunTimeline({ events, connection, streamError, onDecision, decidedApprovals, decisionBusy, approveLabel = "Approve once" }: RunTimelineProps) {
  const ordered = useMemo(() => [...events].sort((a, b) => a.sequence - b.sequence), [events]);

  return (
    <div className="timelinePanel">
      <header className="timelineHeader">
        <div><span className="eyebrow">Live run</span><h2>Activity</h2></div>
        <span className={`streamState stream-${connection}`}><i />{connection}</span>
      </header>
      {streamError ? <div className="streamWarning">Connection interrupted. Reconnecting from the last event…</div> : null}
      <div className="timelineList" aria-live="polite">
        {!ordered.length ? (
          <div className="timelineEmpty"><span>↗</span><p>Run steps, approvals, and validation results appear here.</p></div>
        ) : null}
        {ordered.map((event) => {
          const approval = approvalFrom(event);
          const decided = approval ? decidedApprovals.has(approval.id) || approval.status !== "pending" : false;
          return (
            <article className={`timelineEvent tone-${eventTone(event)}`} key={event.id}>
              <span className="timelineNode" />
              <div className="timelineEventBody">
                <div className="timelineEventTitle">
                  <strong>{titleFor(event.type)}</strong>
                  <time>{event.timestamp ? new Date(event.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : `#${event.sequence}`}</time>
                </div>
                <p>{eventSummary(event)}</p>
                {approval ? (
                  <section className="approvalCard">
                    <div className="approvalTitle"><span>{approval.risk_level ?? "R3"}</span><strong>{approval.title}</strong></div>
                    <p>{approval.summary}</p>
                    {approval.permissions?.length ? <div className="permissionList">{approval.permissions.map((permission) => <span key={permission}>{permission}</span>)}</div> : null}
                    {approval.action_digest ? <code title={approval.action_digest}>Action {approval.action_digest.slice(0, 12)}</code> : null}
                    {decided ? (
                      <div className="decisionRecorded">✓ Decision recorded</div>
                    ) : (
                      <div className="approvalActions">
                        <button type="button" className="dangerButton" disabled={decisionBusy === approval.id} onClick={() => void onDecision(approval.id, "reject")}>Reject</button>
                        <button type="button" className="primaryButton" disabled={decisionBusy === approval.id} onClick={() => void onDecision(approval.id, "approve")}>{approveLabel}</button>
                      </div>
                    )}
                  </section>
                ) : null}
              </div>
            </article>
          );
        })}
      </div>
      <footer className="timelineFooter">Only operational summaries are shown. Private model reasoning is never exposed.</footer>
    </div>
  );
}
