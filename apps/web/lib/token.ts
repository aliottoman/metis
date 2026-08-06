/**
 * A unique-enough token for client-side navigation (`/?new=<token>`).
 *
 * `crypto.randomUUID` is gated on secure contexts, and this app is served
 * over plain HTTP on loopback — which every current engine treats as secure,
 * but "every current engine" is exactly the assumption that broke silently
 * once before. These tokens only need to differ from the previous click, so
 * a time-and-randomness fallback is fully adequate and can never throw.
 */
export function freshToken(): string {
  try {
    if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
      return crypto.randomUUID();
    }
  } catch {
    // fall through to the arithmetic path
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}
