"use client";

// Three choices, one pressed: system, light, dark.

import { Monitor, Moon, Sun } from "lucide-react";
import { useEffect, useState } from "react";

import { applyTheme, THEME_CHANGED_EVENT, THEME_KEY, type ThemeChoice } from "@/lib/theme";

const CHOICES: Array<{ value: ThemeChoice; label: string; icon: typeof Sun }> = [
  { value: "system", label: "System", icon: Monitor },
  { value: "light", label: "Light", icon: Sun },
  { value: "dark", label: "Dark", icon: Moon },
];

export function ThemeControl() {
  const [choice, setChoice] = useState<ThemeChoice>("system");
  useEffect(() => {
    const sync = () => {
      const active = document.documentElement.getAttribute("data-theme");
      setChoice(active === "light" || active === "dark" ? active : "system");
    };
    const onStorage = (event: StorageEvent) => {
      if (event.key === THEME_KEY || event.key === null) sync();
    };
    sync();
    window.addEventListener(THEME_CHANGED_EVENT, sync);
    window.addEventListener("storage", onStorage);
    return () => {
      window.removeEventListener(THEME_CHANGED_EVENT, sync);
      window.removeEventListener("storage", onStorage);
    };
  }, []);
  return (
    <div role="group" aria-label="Appearance" className="theme-control">
      {CHOICES.map((item) => {
        const Icon = item.icon;
        const pressed = choice === item.value;
        return (
          <button
            key={item.value}
            type="button"
            className={`ui-btn${pressed ? " is-primary" : ""}`}
            aria-pressed={pressed}
            onClick={() => {
              setChoice(item.value);
              applyTheme(item.value);
            }}
          >
            <Icon size={15} aria-hidden="true" />
            {item.label}
          </button>
        );
      })}
    </div>
  );
}
