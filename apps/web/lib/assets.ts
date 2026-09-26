// How an asset's runtime state reads, shared by the list, the drawer and
// the player.

import type { StatusState } from "@/components/ui/status";
import type { AssetV1 } from "@/lib/types";

export type AssetAction = "start" | "stop" | "approve" | "revoke" | "recipe";
export type AssetFilter = "all" | "active" | "ready" | "failed" | "review" | "setup" | "pinned";
export type AssetView = "preview" | "logs" | "settings";

export function canBeginAssetAction(pending: { id: string; action: AssetAction } | null, id: string, action: AssetAction): boolean {
  return !pending || (pending.id === id && pending.action === "start" && action === "stop");
}

export function assetHref(id: string, view: AssetView = "preview"): string {
  return `/assets/${encodeURIComponent(id)}${view === "preview" ? "" : `?view=${view}`}`;
}

export function assetView(value: string | null): AssetView {
  return value === "logs" || value === "settings" ? value : "preview";
}

export function matchesAsset(asset: AssetV1, filter: AssetFilter, pinned: string[] = []): boolean {
  switch (filter) {
    case "active": return isActive(asset);
    case "ready": return asset.launchConfigured && asset.launchApproved && !isActive(asset) && !isFailed(asset);
    case "failed": return isFailed(asset);
    case "review": return asset.launchConfigured && !asset.launchApproved;
    case "setup": return !asset.launchConfigured;
    case "pinned": return pinned.includes(asset.id);
    default: return true;
  }
}

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
  if (!asset.url) return null;
  const value = asset.url.trim();
  // Relative app paths are allowed, but protocol-relative URLs are not.
  if (value.startsWith("/") && !value.startsWith("//") && !value.includes("\\")) return value;
  try {
    const url = new URL(value);
    return ["http:", "https:"].includes(url.protocol) && !url.username && !url.password ? url.href : null;
  } catch { return null; }
}

export function commandLabel(parts: string[]): string {
  return parts.map((part) => JSON.stringify(part)).join(" ");
}

export function mergeAsset(items: AssetV1[], updated: AssetV1): AssetV1[] {
  return items.some((asset) => asset.id === updated.id) ? items.map((asset) => (asset.id === updated.id ? updated : asset)) : [updated, ...items];
}

export interface AssetCatalogSnapshot {
  assets: AssetV1[];
  loaded: boolean;
  loading: boolean;
  error: string | null;
  loadError: string | null;
  scanning: boolean;
  busy: { id: string; action: AssetAction } | null;
}

const EMPTY_CATALOG: AssetCatalogSnapshot = {
  assets: [], loaded: false, loading: true, error: null, loadError: null, scanning: false, busy: null,
};

/** Own work in flight across route transitions; injected transport lets tests
 * exercise launch/stop races without starting a project. */
export function createAssetCatalogStore(transport: {
  list: () => Promise<AssetV1[]>;
  scan: () => Promise<AssetV1[]>;
  perform: (assetId: string, action: AssetAction) => Promise<AssetV1>;
}) {
  let snapshot: AssetCatalogSnapshot = { ...EMPTY_CATALOG };
  const listeners = new Set<() => void>();
  let revision = 0;
  let requestSequence = 0;
  let actionSequence = 0;
  let currentRead: { revision: number; promise: Promise<void> } | null = null;
  const publish = (patch: Partial<AssetCatalogSnapshot>) => {
    snapshot = { ...snapshot, ...patch };
    listeners.forEach((notify) => notify());
  };
  const update = (asset: AssetV1) => {
    revision += 1;
    publish({ assets: mergeAsset(snapshot.assets, asset) });
  };
  const load = (quiet = false): Promise<void> => {
    if (snapshot.scanning) return Promise.resolve();
    if (!quiet && !snapshot.loading) publish({ loading: true });
    if (currentRead?.revision === revision) return currentRead.promise;
    const version = revision;
    const id = ++requestSequence;
    const promise = (async () => {
      // Install the shared request before invoking a transport that can throw
      // synchronously; otherwise a failed read could stay cached forever.
      await Promise.resolve();
      try {
        const next = await transport.list();
        if (id !== requestSequence || version !== revision) return;
        publish({ assets: next, loaded: true, loadError: null });
      } catch (problem) {
        if (id === requestSequence && version === revision) publish({ loadError: problem instanceof Error ? problem.message : "The asset catalog could not be loaded." });
      } finally {
        if (id === requestSequence) {
          currentRead = null;
          publish({ loading: false });
        }
      }
    })();
    currentRead = { revision: version, promise };
    return promise;
  };
  const act = async (asset: AssetV1, action: AssetAction): Promise<AssetV1 | null> => {
    if (!canBeginAssetAction(snapshot.busy, asset.id, action) || snapshot.scanning) return null;
    const sequence = ++actionSequence;
    revision += 1;
    publish({ busy: { id: asset.id, action }, error: null });
    try {
      const result = await transport.perform(asset.id, action);
      // Stop owns a superseded launch: its late result/error cannot reopen the
      // app or clear a more recent action's pending state.
      if (sequence !== actionSequence) return null;
      update(result);
      return result;
    } catch (problem) {
      if (sequence === actionSequence) publish({ error: `${asset.name}: ${problem instanceof Error ? problem.message : `Could not ${action} this asset. Check its logs and try again.`}` });
      return null;
    } finally {
      if (sequence === actionSequence) {
        revision += 1;
        publish({ busy: null });
        void load(true);
      }
    }
  };
  const scan = async (): Promise<number | null> => {
    if (snapshot.scanning || snapshot.busy) return null;
    revision += 1;
    publish({ scanning: true, error: null });
    try {
      const found = await transport.scan();
      revision += 1;
      publish({ assets: found, loaded: true, loading: false, loadError: null });
      return found.length;
    } catch (problem) {
      publish({ error: problem instanceof Error ? problem.message : "Metis could not scan the projects folder." });
      return null;
    } finally {
      publish({ scanning: false });
    }
  };
  return {
    getSnapshot: () => snapshot,
    getServerSnapshot: () => EMPTY_CATALOG,
    subscribe: (notify: () => void) => { listeners.add(notify); return () => { listeners.delete(notify); }; },
    load, update, act, scan,
    dismissError: () => publish({ error: null }),
  };
}

export const ASSET_PIN_KEY = "metis.assets.pinned";

export function assetPins(raw: string): string[] {
  try {
    const value: unknown = JSON.parse(raw);
    return Array.isArray(value) ? [...new Set(value.filter((id): id is string => typeof id === "string" && id.length > 0))] : [];
  } catch { return []; }
}

/** Storage can be readable but unwritable (for example at quota). Keep the
 * current visit useful and reconcile when another window changes the value. */
export function createAssetPinStore(storage: () => Pick<Storage, "getItem" | "setItem">) {
  let current = "[]";
  let localOnly = false;
  const read = () => {
    if (localOnly) return current;
    try { current = storage().getItem(ASSET_PIN_KEY) ?? "[]"; } catch { /* retain the current visit */ }
    return current;
  };
  return {
    read,
    sync: () => { localOnly = false; read(); },
    toggle: (id: string) => {
      const pinned = assetPins(read());
      current = JSON.stringify(pinned.includes(id) ? pinned.filter((item) => item !== id) : [...pinned, id]);
      try { storage().setItem(ASSET_PIN_KEY, current); localOnly = false; } catch { localOnly = true; }
    },
  };
}
