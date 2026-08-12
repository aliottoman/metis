/**
 * Plain-language rendering for coding steps and their token cost.
 *
 * A live session recorded nine failed editor calls whose timeline entries said
 * only "editor · failed". That is indistinguishable from a refused edit, and
 * it is why three expensive runs could not explain themselves. These helpers
 * exist so the reason and the running cost are both visible in the activity
 * view, in the same non-technical voice as the rest of it.
 *
 * Redaction and bounding already happened in the sidecar; nothing here widens
 * what is shown, it only stops the useful part being thrown away.
 */

const FAILURE_STATUS = /fail|error|denied|reject|abort/i;

function text(
  source: Record<string, unknown>,
  key: string,
): string | undefined {
  const value = source[key];
  return typeof value === "string" && value.trim() ? value : undefined;
}

function count(value: unknown): string | undefined {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toLocaleString("en-US")
    : undefined;
}

function object(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

export function isFailedCodingStep(payload: Record<string, unknown>): boolean {
  return FAILURE_STATUS.test(text(payload, "status") ?? "");
}

/** One sentence describing a coding step, cost included when reported. */
export function codingStepSummary(payload: Record<string, unknown>): string {
  const tool = text(payload, "tool");
  const label = tool ? tool.replaceAll("_", " ") : undefined;
  const iteration = count(payload.iteration);
  const step = iteration && iteration !== "0" ? ` · step ${iteration}` : "";

  if (isFailedCodingStep(payload)) {
    const failureClass = text(payload, "failure_class");
    const why = text(payload, "message") ?? "no reason was reported";
    const what = label
      ? `${label} could not finish`
      : "a step could not finish";
    return `${what}${failureClass ? ` (${failureClass})` : ""} — ${why}${step}`;
  }

  const usage = object(payload.usage);
  if (usage) {
    const delta = object(payload.usage_delta);
    // "unknown" rather than 0: a provider that reported nothing has not told
    // us the step was free.
    const total = count(usage.totalTokens) ?? "unknown";
    const added = delta ? (count(delta.totalTokens) ?? "unknown") : "unknown";
    return `${total} tokens so far · ${added} added since the last step${step}`;
  }

  const detail = text(payload, "message");
  const status = (text(payload, "status") ?? "working").replaceAll("_", " ");
  const lead = label ?? status;
  return detail ? `${lead} — ${detail}${step}` : `${lead}${step}`;
}

/** One sentence describing a whole coding round and what it produced. */
export function codingRoundSummary(payload: Record<string, unknown>): string {
  const changed = Array.isArray(payload.changed_paths)
    ? payload.changed_paths.map(String)
    : [];
  const iterations = count(payload.iterations) ?? "?";
  const usage = object(payload.usage);
  const total = usage ? (count(usage.totalTokens) ?? "unknown") : "unknown";
  const wrote = changed.length
    ? `wrote ${changed.join(", ")}`
    : "wrote nothing";
  const ended =
    text(payload, "controlled_stop_reason") === "max_iterations"
      ? " · stopped at its step limit"
      : "";
  return `${wrote} · ${iterations} step${iterations === "1" ? "" : "s"} · ${total} tokens${ended}`;
}

/**
 * What one Cline coding session is allowed to do, shown before it starts.
 *
 * A person approving the result afterwards should have been able to see the
 * envelope beforehand: where it may write, what it may not touch, and whether
 * anything in their own wording could not be resolved.
 */
export function directContractSummary(
  payload: Record<string, unknown>,
): string {
  const roots = Array.isArray(payload.writable_roots)
    ? payload.writable_roots.map(String)
    : ["."];
  const protectedFiles = Array.isArray(payload.protected_files)
    ? payload.protected_files.map(String)
    : [];
  const unresolved = Array.isArray(payload.unresolved)
    ? payload.unresolved.map(String)
    : [];
  const where =
    roots.length === 1 && roots[0] === "."
      ? "can edit anything in this project"
      : `can edit under ${roots.join(", ")}`;
  const kept = protectedFiles.length
    ? `keeping ${protectedFiles.join(", ")} unchanged`
    : "with no files singled out to keep unchanged";
  if (unresolved.length) {
    // Nothing has run yet: this is the reason it stopped, not a warning
    // attached to work already done.
    return `waiting — could not work out which files you meant by ${unresolved
      .map((item) => `“${item}”`)
      .join(", ")}`;
  }
  return `one session, ${where}, ${kept}`;
}
