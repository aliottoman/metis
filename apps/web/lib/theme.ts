// The appearance choice: follow the system, or pin light or dark. Stamped on
// the root element as data-theme; "system" removes the stamp so the tokens'
// prefers-color-scheme rule decides. The inline script in the layout applies
// the saved choice before first paint, so a dark page never flashes light.

export type ThemeChoice = "system" | "light" | "dark";

export const THEME_KEY = "metis.theme";
export const THEME_CHANGED_EVENT = "metis:theme-changed";

export function readTheme(): ThemeChoice {
  try {
    const raw = window.localStorage.getItem(THEME_KEY);
    return raw === "light" || raw === "dark" ? raw : "system";
  } catch {
    return "system";
  }
}

export function applyTheme(choice: ThemeChoice): void {
  const root = document.documentElement;
  if (choice === "system") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", choice);
  try {
    if (choice === "system") window.localStorage.removeItem(THEME_KEY);
    else window.localStorage.setItem(THEME_KEY, choice);
  } catch {
    // The stamp is applied even when it cannot be remembered.
  }
  window.dispatchEvent(new Event(THEME_CHANGED_EVENT));
}

/** Runs before paint. Kept tiny and dependency-free because it is inlined. */
export const THEME_BOOT_SCRIPT =
  "(function(){try{var t=localStorage.getItem('metis.theme');if(t==='light'||t==='dark'){document.documentElement.setAttribute('data-theme',t);}}catch(e){}})();";
