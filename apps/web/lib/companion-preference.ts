"use client";

export type CompanionEnergy = "calm" | "expressive" | "playful";

const STORAGE_KEY = "metis.companionEnergy";
export const COMPANION_ENERGY_CHANGED_EVENT = "metis:companion-energy-changed";

export function readCompanionEnergy(): CompanionEnergy {
  if (typeof window === "undefined") return "expressive";
  const value = window.localStorage.getItem(STORAGE_KEY);
  return value === "calm" || value === "playful" ? value : "expressive";
}

export function writeCompanionEnergy(value: CompanionEnergy): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(STORAGE_KEY, value);
  window.dispatchEvent(new CustomEvent(COMPANION_ENERGY_CHANGED_EVENT, { detail: value }));
}
