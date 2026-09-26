import type { RunEventV1 } from "./types.ts";

type Bucket = "summary" | "checks" | "approvals" | "questions" | "outputs" | "changes";
// Separate budgets keep a noisy verification loop from evicting the plan,
// pending decisions, or downloadable outputs. The full stream remains capped
// independently; these are the milestones needed to explain a long task.
const LIMITS: Record<Bucket, number> = { summary: 12, checks: 40, approvals: 48, questions: 32, outputs: 48, changes: 24 };
const TERMINAL = new Set(["run.completed", "run.failed", "run.cancelled", "completed", "failed", "cancelled", "run.resumed", "run.recovered"]);

function identity(event: RunEventV1): { bucket: Bucket; key: string; first?: boolean } | null {
  const p = event.payload;
  const nested = p.approval && typeof p.approval === "object" ? p.approval as Record<string, unknown> : p;
  if (event.type === "run.created" || event.type === "run.started") return { bucket: "summary", key: "started", first: true };
  if (TERMINAL.has(event.type)) return { bucket: "summary", key: "lifecycle" };
  if (["plan.created", "project.build_planned", "project.plan_revised", "project.direct_contract", "project.phase", "project.direction", "run.model_exhausted", "run.model_fallback", "model.started"].includes(event.type)) return { bucket: "summary", key: event.type };
  if (["project.staged_verified", "project.build_checked", "project.vertical_slice_checked", "project.check_result", "project.check_requested"].includes(event.type)) return { bucket: "checks", key: `${event.type}:${String(p.name ?? p.check ?? "")}` };
  if (["approval.required", "approval.applied", "approval.decided", "project.verification_decided"].includes(event.type)) return { bucket: "approvals", key: `${event.type}:${String(nested.approval_id ?? nested.id ?? event.id)}` };
  if (["elicitation.requested", "elicitation.answered"].includes(event.type)) return { bucket: "questions", key: `${event.type}:${String(p.elicitation_id ?? p.id ?? event.id)}` };
  if (event.type === "artifact.created") return { bucket: "outputs", key: String(p.id ?? p.artifact_id ?? event.id) };
  if (event.type === "project.coding_round" && Array.isArray(p.changed_paths) && p.changed_paths.length) return { bucket: "changes", key: event.id };
  return null;
}

/** Called for each decoded frame, before React batches a large SSE replay.
 * Returns the same array for routine deltas to avoid extra renders. */
export function retainExecutionMilestone(previous: RunEventV1[], event: RunEventV1): RunEventV1[] {
  const slot = identity(event);
  if (!slot) return previous;
  const current = previous.filter((item) => item.run_id === event.run_id);
  const match = current.find((item) => {
    const candidate = identity(item);
    return candidate?.bucket === slot.bucket && candidate.key === slot.key;
  });
  if (match && (slot.first || match.sequence >= event.sequence)) return current.length === previous.length ? previous : current;
  const next = [...current.filter((item) => item !== match), event].sort((a, b) => a.sequence - b.sequence);
  const inBucket = next.filter((item) => identity(item)?.bucket === slot.bucket);
  const evicted = new Set(inBucket.slice(0, Math.max(0, inBucket.length - LIMITS[slot.bucket])));
  return next.filter((item) => !evicted.has(item));
}

/** Rejoin retained milestones and recent live updates without duplicate rows. */
export function mergeExecutionEvents(milestones: readonly RunEventV1[], recent: readonly RunEventV1[], runId: string | null): RunEventV1[] {
  if (!runId) return [];
  return Array.from(new Map([...milestones, ...recent].filter((event) => event.run_id === runId).map((event) => [event.id || `${event.run_id}-${event.sequence}`, event])).values()).sort((a, b) => a.sequence - b.sequence);
}
