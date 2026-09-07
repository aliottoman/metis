// One dot and five words. Every status in the app maps onto these.

export type StatusState = "live" | "waiting" | "ready" | "needs-review" | "stopped";

const WORDS: Record<StatusState, string> = {
  live: "live",
  waiting: "waiting",
  ready: "ready",
  "needs-review": "needs review",
  stopped: "stopped",
};

export function StatusDot({ state }: { state: StatusState }) {
  return <i className={`ui-dot is-${state}`} aria-hidden="true" />;
}

/** The dot with its word, or with a label of your own beside the dot. */
export function Status({ state, label }: { state: StatusState; label?: string }) {
  return (
    <span className="ui-status">
      <StatusDot state={state} />
      {label ?? WORDS[state]}
    </span>
  );
}
