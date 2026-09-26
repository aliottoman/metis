/** A slow refresh must finish before another one starts, including after a
 * tab is hidden and reopened. Keep the runner for the lifetime of the view. */
export function createPollRunner(tick: () => void | Promise<unknown>): () => Promise<void> {
  let running = false;
  return async () => {
    if (running) return;
    running = true;
    try {
      await tick();
    } catch {
      // The view owns its error message. A rejected background refresh must
      // not become an unhandled rejection or prevent the next retry.
    } finally {
      running = false;
    }
  };
}
