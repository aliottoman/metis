"use client";

import { useEffect, useSyncExternalStore } from "react";
import { approveAsset, generateAssetRecipe, listAssets, revokeAssetApproval, scanAssets, startAsset, stopAsset } from "@/lib/api";
import { createAssetCatalogStore, isActive } from "@/lib/assets";
import { usePoll } from "@/hooks/use-poll";

// Keep launches and their results alive when Start & open changes the route.
// A new workspace subscribes to the same pending action immediately.
const catalog = createAssetCatalogStore({
  list: listAssets,
  scan: scanAssets,
  perform: (id, action) => ({
    start: () => startAsset(id, {}),
    stop: () => stopAsset(id),
    approve: () => approveAsset(id),
    revoke: () => revokeAssetApproval(id),
    recipe: () => generateAssetRecipe(id),
  })[action](),
});

export function useAssetCatalog() {
  const state = useSyncExternalStore(catalog.subscribe, catalog.getSnapshot, catalog.getServerSnapshot);
  useEffect(() => { void catalog.load(); }, []);
  // Multiple mounted consumers share a coalesced request. A launch continues
  // on the host when its view closes; only visible views need to poll it.
  usePoll(() => catalog.load(true), state.assets.some(isActive) || state.busy ? 3_000 : 12_000);
  return { ...state, load: catalog.load, scan: catalog.scan, act: catalog.act, update: catalog.update, dismissError: catalog.dismissError };
}
