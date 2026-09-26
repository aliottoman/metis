"use client";

import { ArrowDownToLine, Check, CheckCheck, ChevronDown, CircleAlert, ClipboardList, CornerDownLeft, FileText, LoaderCircle, MessageSquare, Square } from "lucide-react";

import { ApprovalCard } from "@/components/approval-card";
import { ArtifactViewer } from "@/components/artifact-viewer";
import { ElicitationCard } from "@/components/elicitation-card";
import { RunTimeline, runEventSummary, runEventTitle } from "@/components/run-timeline";
import { Notice } from "@/components/ui/notice";
import type { Chat } from "@/hooks/use-chat";
import { EXECUTION_STATE_LABELS, type TaskExecution } from "@/lib/task-execution";

function currentActivity(execution: TaskExecution, stageLabel: string | null): string {
  if (execution.state === "completed") return "Response ready";
  if (execution.state === "cancelled") return "Work stopped";
  if (execution.state === "failed") return execution.issues.find((issue) => issue.id === "run")?.detail ?? "Review what happened before continuing.";
  if (execution.state === "waiting") return "Your input is needed";
  if (execution.state === "reconnecting") return "Restoring the connection";
  if (execution.state === "idle") return execution.latest ? "Most recent activity" : "Ready";
  return stageLabel || (execution.latest ? runEventTitle(execution.latest.type, execution.latest.payload) : "Working");
}

export function TaskExecutionBar({ execution, stageLabel, open, onOpen }: {
  execution: TaskExecution;
  stageLabel: string | null;
  open: boolean;
  onOpen: () => void;
}) {
  const live = execution.state === "working" || execution.state === "reconnecting";
  return (
    <button type="button" className={`task-bar is-${execution.state}`} onClick={onOpen} aria-expanded={open} aria-controls="chat-activity">
      <span className="task-bar-icon" aria-hidden="true">{live ? <LoaderCircle size={14} /> : execution.state === "completed" ? <CheckCheck size={14} /> : execution.state === "waiting" || execution.state === "failed" ? <CircleAlert size={14} /> : <ClipboardList size={14} />}</span>
      <span className="task-bar-copy"><strong>{currentActivity(execution, stageLabel)}</strong><span>{EXECUTION_STATE_LABELS[execution.state]}</span></span>
      <span className="task-bar-link">Details <ChevronDown size={14} aria-hidden="true" /></span>
    </button>
  );
}

export function TaskExecutionPanel({ chat, execution, stopping, steering, onSteer, onStop, onShowResponse }: {
  chat: Chat;
  execution: TaskExecution;
  stopping: boolean;
  steering: boolean;
  onSteer: () => void;
  onStop: () => void;
  onShowResponse: () => void;
}) {
  const assistantIndex = chat.messages.findIndex((message) => message.role === "assistant" && (message.run_id === chat.activeRunId || message.id === `assistant-${chat.activeRunId}`));
  const request = [...(assistantIndex >= 0 ? chat.messages.slice(0, assistantIndex) : chat.messages)].reverse().find((message) => message.role === "user");
  const finalAnswer = chat.messages.find((message) => message.role === "assistant" && message.run_id === chat.activeRunId && !message.streaming && message.content && !message.failed);
  const hasPlan = execution.plan.summary || execution.plan.steps.length || execution.plan.files.length;
  const active = chat.runActive && !execution.terminal;
  const decisionCount = Number(Boolean(chat.pendingApproval)) + Number(Boolean(chat.pendingElicitation));
  let latestMeaningful: (typeof execution.progress)[number] | null = null;
  for (let index = execution.progress.length - 1; index >= 0; index -= 1) {
    const event = execution.progress[index];
    if (runEventSummary(event) !== "Recorded by the control plane") {
      latestMeaningful = event;
      break;
    }
  }

  return (
    <section className="task-execution" aria-label="Task execution">
      <header className="task-header">
        <h2>Activity</h2>
        <p className="task-request" title={request?.content}>{request?.content || "Your latest task activity appears here."}</p>
        <div className={`task-state is-${execution.state}`} role="status">
          <span className="task-state-dot" aria-hidden="true" />
          <strong>{EXECUTION_STATE_LABELS[execution.state]}</strong>
          {chat.selectedProject ? <span>{chat.selectedProject.name}</span> : null}
        </div>
        <p className="task-current">{currentActivity(execution, chat.stageLabel)}</p>
        <div className="task-controls">
          <button type="button" className="ui-btn is-sm" onClick={onSteer}><MessageSquare size={14} aria-hidden="true" />{active ? "Add direction" : "Continue task"}</button>
          {finalAnswer ? <button type="button" className="ui-btn is-sm is-quiet" onClick={onShowResponse}>View response</button> : null}
          {active ? <button type="button" className="ui-btn is-sm is-danger" disabled={stopping} onClick={onStop}>{stopping ? <LoaderCircle size={13} aria-hidden="true" /> : <Square size={12} aria-hidden="true" />}{stopping ? "Stopping…" : "Stop task"}</button> : null}
        </div>
        {chat.error ? <Notice kind="error" onDismiss={() => chat.setError(null)}>{chat.error}</Notice> : null}
        {steering ? <p className="task-steering" role="status">{active ? "Write your direction in the composer. It sends after this run; stop the task first to change course now." : "Write a follow-up in the composer to continue from this conversation."}</p> : null}
        {chat.queued ? <div className="task-queued"><CornerDownLeft size={14} aria-hidden="true" /><span><strong>Next message is queued</strong><small>{chat.queued.content || `${chat.queued.attachments.length} attached files`}</small></span><button type="button" className="ui-btn is-quiet is-sm" onClick={chat.unqueue}>Edit</button></div> : null}
      </header>

      {decisionCount > 0 ? <section className="task-section task-decisions" aria-labelledby="task-decisions-title">
        <div className="task-section-title"><h3 id="task-decisions-title">Your decision</h3><span className="ui-chip is-accent">{decisionCount} waiting</span></div>
        {chat.pendingApproval ? <ApprovalCard approval={chat.pendingApproval} decided={chat.decidedApprovals.has(chat.pendingApproval.id)} decisionBusy={chat.decisionBusy} onDecision={chat.decide} approveLabel={chat.approveLabel} /> : null}
        {chat.pendingElicitation ? <ElicitationCard key={chat.pendingElicitation.id} elicitation={chat.pendingElicitation} answered={chat.answeredElicitations.has(chat.pendingElicitation.id)} answerBusy={chat.answerBusy} onAnswer={chat.answer} /> : null}
      </section> : null}

      <section className="task-section" aria-labelledby="task-progress-title">
        <div className="task-section-title"><h3 id="task-progress-title">Latest</h3></div>
        <div className="task-evidence">
          {execution.checksReported ? <span><CheckCheck size={14} aria-hidden="true" /><strong>{execution.checksPassed}</strong> checks passed</span> : null}
          {execution.changedFiles.length ? <span><FileText size={14} aria-hidden="true" /><strong>{execution.changedFiles.length}</strong> files changed</span> : null}
          {chat.artifacts.length ? <span><ArrowDownToLine size={14} aria-hidden="true" /><strong>{chat.artifacts.length}</strong> outputs ready</span> : null}
        </div>
        {latestMeaningful ? <div className="task-latest"><strong>{runEventTitle(latestMeaningful.type, latestMeaningful.payload)}</strong><p>{runEventSummary(latestMeaningful)}</p>{latestMeaningful.timestamp ? <time dateTime={latestMeaningful.timestamp}>Updated {new Date(latestMeaningful.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</time> : null}</div> : <p className="task-empty">Progress appears here as work begins.</p>}
      </section>

      {hasPlan ? <details className="task-section task-fold">
        <summary><ClipboardList size={15} aria-hidden="true" /><strong>Plan</strong>{execution.plan.steps.length ? <small>{execution.plan.steps.length} {execution.plan.steps.length === 1 ? "step" : "steps"}</small> : null}<ChevronDown className="task-fold-chevron" size={15} aria-hidden="true" /></summary>
        <div className="task-fold-body">
        {execution.plan.summary ? <p className="task-plan-summary">{execution.plan.summary}</p> : null}
        {execution.plan.steps.length ? <ol className="task-plan">{execution.plan.steps.map((step, index) => <li key={step.id} className={step.checked ? "is-checked" : ""}><span className="task-plan-number" aria-label={step.checked ? "Checked" : `Step ${index + 1}`}>{step.checked ? <Check size={13} aria-hidden="true" /> : index + 1}</span><div><strong>{step.title}</strong>{step.description ? <p>{step.description}</p> : null}{step.checked ? <small>Verification passed</small> : null}</div></li>)}</ol> : null}
        {execution.plan.revision ? <p className="task-plan-revision"><strong>Plan updated</strong>{execution.plan.revision}</p> : null}
        {execution.plan.files.length ? <details className="task-details"><summary><ChevronDown size={13} aria-hidden="true" />{execution.plan.files.length} planned files</summary><ul className="task-files">{execution.plan.files.map((file) => <li key={file.path}><FileText size={13} aria-hidden="true" /><code>{file.path}</code><span>{file.state === "checked" ? "Checked" : file.state === "changed" ? "Changed" : "Planned"}</span></li>)}</ul></details> : null}
        {execution.plan.scenarios.length ? <details className="task-details"><summary><ChevronDown size={13} aria-hidden="true" />Success criteria</summary><ul className="task-criteria">{execution.plan.scenarios.map((scenario, index) => <li key={`${index}-${scenario}`}>{scenario}</li>)}</ul></details> : null}
        {execution.plan.assumptions.length ? <details className="task-details"><summary><ChevronDown size={13} aria-hidden="true" />Assumptions</summary><ul className="task-criteria">{execution.plan.assumptions.map((assumption, index) => <li key={`${index}-${assumption}`}>{assumption}</li>)}</ul></details> : null}
        </div>
      </details> : null}

      {execution.issues.length ? <section className="task-section" aria-labelledby="task-blockers-title">
        <div className="task-section-title"><h3 id="task-blockers-title"><CircleAlert size={15} aria-hidden="true" />Needs attention</h3><span>{execution.issues.length}</span></div>
        <ul className="task-blockers">{execution.issues.map((issue) => <li key={issue.id}><strong>{issue.title}</strong><p>{issue.detail}</p></li>)}</ul>
      </section> : null}

      {chat.artifacts.length || execution.changedFiles.length ? <details className="task-section task-fold task-outputs">
        <summary><ArrowDownToLine size={15} aria-hidden="true" /><strong>Outputs</strong><small>{chat.artifacts.length ? `${chat.artifacts.length} ready` : `${execution.changedFiles.length} files changed`}</small><ChevronDown className="task-fold-chevron" size={15} aria-hidden="true" /></summary>
        <div className="task-fold-body">
          {chat.artifacts.length ? <ArtifactViewer artifacts={chat.artifacts} /> : null}
          {execution.changedFiles.length ? <details className="task-details"><summary><ChevronDown size={13} aria-hidden="true" />Changed project files</summary><ul className="task-files">{execution.changedFiles.map((path) => <li key={path}><FileText size={13} aria-hidden="true" /><code>{path}</code></li>)}</ul></details> : null}
        </div>
      </details> : null}

      {execution.decisions.length ? <details className="task-section task-fold"><summary><Check size={15} aria-hidden="true" /><strong>Decisions</strong><small>{execution.decisions.length}</small><ChevronDown className="task-fold-chevron" size={15} aria-hidden="true" /></summary><div className="task-fold-body"><ul className="task-decision-history">{execution.decisions.map((event) => <li key={event.id}><Check size={13} aria-hidden="true" /><div><strong>{runEventTitle(event.type, event.payload)}</strong><p>{runEventSummary(event)}</p></div></li>)}</ul></div></details> : null}

      <details className="task-full-activity"><summary>Full activity log <span>{chat.events.length} events</span><ChevronDown size={14} aria-hidden="true" /></summary><RunTimeline events={chat.events} connection={chat.connection} streamError={chat.streamError} onDecision={chat.decide} decidedApprovals={chat.decidedApprovals} decisionBusy={chat.decisionBusy} approveLabel={chat.approveLabel} showDecisions={false} /></details>
    </section>
  );
}
