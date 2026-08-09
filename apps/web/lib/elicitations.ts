import type { ElicitationRequest, RunEventV1 } from "@/lib/types";

/**
 * The clarifying question carried by an event, or null. Only the
 * `elicitation.requested` event carries the request; status events such as
 * `run.awaiting_input` and `elicitation.answered` do not, so they are ignored.
 */
export function elicitationFrom(event: RunEventV1): ElicitationRequest | null {
  if (event.type !== "elicitation.requested") return null;
  const payload = event.payload as Record<string, unknown>;
  const id = payload.id;
  const question = payload.question;
  if (typeof id !== "string" || !id) return null;
  if (typeof question !== "string" || !question) return null;
  return {
    id,
    run_id: event.run_id,
    question,
    options: Array.isArray(payload.options) ? payload.options.map(String) : [],
    // The request forces free text on when it offered no options, but honor an
    // explicit false the same way the card does.
    allow_text: payload.allow_text !== false,
  };
}

/**
 * The single question still awaiting an answer — newest first — surfaced inline
 * in the thread the same way a pending approval is. An answered question, or a
 * finished run, has nothing waiting on you; both are read from the run's later
 * events, not the request alone (whose payload is frozen at request time), so
 * reopening a resolved thread never resurfaces the card.
 */
export function latestPendingElicitation(
  events: RunEventV1[],
  answered: ReadonlySet<string>,
): ElicitationRequest | null {
  const resolved = new Set<string>();
  let runEnded = false;
  for (const event of events) {
    if (event.type === "elicitation.answered") {
      const payload = event.payload as Record<string, unknown>;
      const id = payload.elicitation_id ?? payload.id;
      if (typeof id === "string" && id) resolved.add(id);
    } else if (
      event.type === "run.completed"
      || event.type === "run.failed"
      || event.type === "run.cancelled"
    ) {
      runEnded = true;
    }
  }
  if (runEnded) return null;
  const ordered = [...events].sort((a, b) => b.sequence - a.sequence);
  for (const event of ordered) {
    const elicitation = elicitationFrom(event);
    if (
      elicitation
      && !answered.has(elicitation.id)
      && !resolved.has(elicitation.id)
    ) {
      return elicitation;
    }
  }
  return null;
}
