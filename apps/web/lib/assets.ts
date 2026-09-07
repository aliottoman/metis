// How an asset's runtime state reads, shared by the list, the drawer and
// the player.

import type { StatusState } from "@/components/ui/status";
import type { AssetV1 } from "@/lib/types";

export type AssetAction = "start" | "stop" | "approve" | "revoke" | "recipe";

export function normalizedStatus(asset: AssetV1): string {
  return asset.status.trim().toLowerCase();
}

export function isRunning(asset: AssetV1): boolean {
  return normalizedStatus(asset) === "running";
}

/** Starting, running or stopping: a process exists. */
export function isActive(asset: AssetV1): boolean {
  return ["starting", "running", "stopping"].includes(normalizedStatus(asset));
}

export function isFailed(asset: AssetV1): boolean {
  return ["error", "failed", "crashed"].includes(normalizedStatus(asset));
}

/** The dot and the word beside it. */
export function statusOf(asset: AssetV1): { state: StatusState; label: string } {
  const status = normalizedStatus(asset);
  if (status === "running") return { state: "live", label: "Running" };
  if (status === "starting" || status === "stopping") return { state: "waiting", label: status === "starting" ? "Starting" : "Stopping" };
  if (isFailed(asset)) return { state: "needs-review", label: "Failed" };
  if (!asset.launchConfigured) return { state: "stopped", label: "Setup needed" };
  if (!asset.launchApproved) return { state: "needs-review", label: "Trust required" };
  return { state: "ready", label: "Ready" };
}

/** Running first, then ready, then the ones that need something, by name. */
export function rank(asset: AssetV1): number {
  if (isActive(asset)) return 0;
  if (isFailed(asset)) return 1;
  if (asset.launchApproved) return 2;
  if (asset.launchConfigured) return 3;
  return 4;
}

export function sortAssets(assets: AssetV1[]): AssetV1[] {
  return [...assets].sort((a, b) => rank(a) - rank(b) || a.name.localeCompare(b.name));
}

export function tagsOf(asset: AssetV1, limit = 5): string[] {
  return Array.from(new Set([asset.category, asset.framework, ...asset.tags].map((value) => value.trim()).filter(Boolean))).slice(0, limit);
}

export function launchUrl(asset: AssetV1): string | null {
  return asset.url && /^(https?:\/\/|\/)/i.test(asset.url) ? asset.url : null;
}

export function commandLabel(parts: string[]): string {
  return parts.map((part) => JSON.stringify(part)).join(" ");
}

export function mergeAsset(items: AssetV1[], updated: AssetV1): AssetV1[] {
  return items.some((asset) => asset.id === updated.id) ? items.map((asset) => (asset.id === updated.id ? updated : asset)) : [updated, ...items];
}
