"use client";

import { useCallback, useSyncExternalStore } from "react";
import { ASSET_PIN_KEY, assetPins, createAssetPinStore } from "@/lib/assets";

const CHANGED = "metis:asset-preferences";
const preferences = createAssetPinStore(() => window.localStorage);
function subscribe(notify: () => void) {
  const sync = (event: StorageEvent) => { if (event.key === ASSET_PIN_KEY || event.key === null) { preferences.sync(); notify(); } };
  window.addEventListener("storage", sync);
  window.addEventListener(CHANGED, notify);
  return () => { window.removeEventListener("storage", sync); window.removeEventListener(CHANGED, notify); };
}
export function useAssetPreferences() {
  const raw = useSyncExternalStore(subscribe, preferences.read, () => "[]");
  const pinned = assetPins(raw);
  const togglePin = useCallback((id: string) => {
    preferences.toggle(id);
    window.dispatchEvent(new Event(CHANGED));
  }, []);
  return { pinned, togglePin };
}
