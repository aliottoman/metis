"use client";

// Three choices, one pressed: system, light, dark.

import { Monitor, Moon, Sun } from "lucide-react";
import { useEffect, useState } from "react";

import { applyTheme, readTheme, type ThemeChoice } from "@/lib/theme";

const CHOICES: Array<{ value: ThemeChoice; label: string; icon: typeof Sun }> = [
  { value: "system", label: "System", icon: Monitor },
  { value: "light", label: "Light", icon: Sun },
  { value: "dark", label: "Dark", icon: Moon },
];

export function ThemeControl() {
  const [choice, setChoice] = useState<ThemeChoice>("system");
  useEffect(() => setChoice(readTheme()), []);
  return (
    <div role="group" aria-label="Appearance" style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
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
