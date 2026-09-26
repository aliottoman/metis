// A deliberate upward scroll pauses following, even if the reader is still
// close to the end. Layout changes never count as a request to resume.
export const THREAD_BOTTOM_PX = 96;

export function nextThreadFollowing(
  following: boolean,
  previousTop: number,
  top: number,
  distanceFromBottom: number,
  source: "scroll" | "resize",
): boolean {
  if (source === "resize") return following;
  if (top < previousTop - 1) return false;
  if (distanceFromBottom <= THREAD_BOTTOM_PX) return true;
  return following;
}
