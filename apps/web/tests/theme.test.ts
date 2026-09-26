import assert from "node:assert/strict";
import test from "node:test";
import { runInNewContext } from "node:vm";

import {
  applyTheme,
  readTheme,
  syncThemeFromStorage,
  THEME_BOOT_SCRIPT,
  THEME_CHANGED_EVENT,
  THEME_KEY,
} from "../lib/theme.ts";

function browserFixture({ saved, stamped, blocked = false }: { saved?: string; stamped?: string; blocked?: boolean } = {}) {
  const stored = new Map<string, string>();
  const attributes = new Map<string, string>();
  const writes: string[] = [];
  const events: Array<{ type: string; theme: string | null }> = [];
  let reads = 0;
  if (saved !== undefined) stored.set(THEME_KEY, saved);
  if (stamped !== undefined) attributes.set("data-theme", stamped);
  const documentElement = {
    getAttribute: (key: string) => attributes.get(key) ?? null,
    setAttribute: (key: string, value: string) => { attributes.set(key, value); },
    removeAttribute: (key: string) => { attributes.delete(key); },
  };
  const localStorage = {
    getItem: (key: string) => { reads += 1; return stored.get(key) ?? null; },
    setItem: (key: string, value: string) => { writes.push(`set:${key}`); stored.set(key, value); },
    removeItem: (key: string) => { writes.push(`remove:${key}`); stored.delete(key); },
  };
  const document = { documentElement };
  const window = {
    get localStorage() {
      if (blocked) throw new Error("Browser storage is unavailable");
      return localStorage;
    },
    dispatchEvent: (event: Event) => {
      events.push({ type: event.type, theme: documentElement.getAttribute("data-theme") });
      return true;
    },
  };
  return { window, document, localStorage, stored, writes, events, readCount: () => reads };
}

function withBrowser(options: Parameters<typeof browserFixture>[0], check: (browser: ReturnType<typeof browserFixture>) => void) {
  const browser = browserFixture(options);
  const originalWindow = Object.getOwnPropertyDescriptor(globalThis, "window");
  const originalDocument = Object.getOwnPropertyDescriptor(globalThis, "document");
  Object.defineProperty(globalThis, "window", { configurable: true, value: browser.window });
  Object.defineProperty(globalThis, "document", { configurable: true, value: browser.document });
  try {
    check(browser);
  } finally {
    if (originalWindow) Object.defineProperty(globalThis, "window", originalWindow);
    else Reflect.deleteProperty(globalThis, "window");
    if (originalDocument) Object.defineProperty(globalThis, "document", originalDocument);
    else Reflect.deleteProperty(globalThis, "document");
  }
}

test("explicit light and dark choices apply immediately, persist, and notify every control", () => {
  for (const choice of ["light", "dark"] as const) {
    withBrowser({}, (browser) => {
      applyTheme(choice);
      assert.equal(browser.document.documentElement.getAttribute("data-theme"), choice);
      assert.equal(browser.stored.get(THEME_KEY), choice);
      assert.equal(readTheme(), choice);
      assert.deepEqual(browser.events, [{ type: THEME_CHANGED_EVENT, theme: choice }]);
    });
  }
});

test("system mode releases the explicit theme and preserves unrelated saved preferences", () => {
  withBrowser({ saved: "dark", stamped: "dark" }, (browser) => {
    browser.stored.set("metis.companion", "calm");
    applyTheme("system");
    assert.equal(browser.document.documentElement.getAttribute("data-theme"), null);
    assert.equal(browser.stored.has(THEME_KEY), false);
    assert.equal(browser.stored.get("metis.companion"), "calm");
    assert.equal(readTheme(), "system");
    assert.deepEqual(browser.events, [{ type: THEME_CHANGED_EVENT, theme: null }]);
  });
});

test("missing or invalid saved choices follow the system", () => {
  for (const saved of [undefined, "system", "sepia", ""]) {
    withBrowser({ saved }, () => assert.equal(readTheme(), "system"));
  }
});

test("blocked storage still allows an explicit theme and a return to system for this visit", () => {
  withBrowser({ blocked: true }, (browser) => {
    assert.equal(readTheme(), "system");
    applyTheme("dark");
    assert.equal(readTheme(), "dark");
    applyTheme("light");
    assert.equal(readTheme(), "light");
    applyTheme("system");
    assert.equal(readTheme(), "system");
    assert.deepEqual(browser.writes, []);
    assert.deepEqual(browser.events.map((event) => event.theme), ["dark", "light", null]);
  });
});

test("another window's theme choice updates this window without writing it back", () => {
  withBrowser({ saved: "dark", stamped: "light" }, (browser) => {
    syncThemeFromStorage({ key: THEME_KEY });
    assert.equal(browser.document.documentElement.getAttribute("data-theme"), "dark");
    assert.deepEqual(browser.writes, []);
    assert.deepEqual(browser.events, [{ type: THEME_CHANGED_EVENT, theme: "dark" }]);
  });
});

test("removing the saved theme or clearing storage restores system mode across windows", () => {
  for (const key of [THEME_KEY, null]) {
    withBrowser({ stamped: "dark" }, (browser) => {
      syncThemeFromStorage({ key });
      assert.equal(browser.document.documentElement.getAttribute("data-theme"), null);
      assert.deepEqual(browser.writes, []);
      assert.deepEqual(browser.events, [{ type: THEME_CHANGED_EVENT, theme: null }]);
    });
  }
});

test("unrelated storage events cannot change the theme or notify appearance controls", () => {
  withBrowser({ saved: "dark", stamped: "light" }, (browser) => {
    syncThemeFromStorage({ key: "metis.companion" });
    assert.equal(browser.document.documentElement.getAttribute("data-theme"), "light");
    assert.equal(browser.readCount(), 0);
    assert.deepEqual(browser.writes, []);
    assert.deepEqual(browser.events, []);
  });
});

test("the boot script applies a saved theme synchronously before client code is available", () => {
  for (const saved of ["light", "dark"]) {
    const browser = browserFixture({ saved });
    // This environment deliberately has no React, window, events, or timers.
    runInNewContext(THEME_BOOT_SCRIPT, { localStorage: browser.localStorage, document: browser.document });
    assert.equal(browser.document.documentElement.getAttribute("data-theme"), saved);
    assert.deepEqual(browser.writes, []);
  }
});

test("the boot script leaves system mode unstamped and tolerates inaccessible storage", () => {
  for (const saved of [undefined, "system", "invalid"]) {
    const browser = browserFixture({ saved });
    runInNewContext(THEME_BOOT_SCRIPT, { localStorage: browser.localStorage, document: browser.document });
    assert.equal(browser.document.documentElement.getAttribute("data-theme"), null);
  }
  const browser = browserFixture();
  const context = {
    document: browser.document,
    get localStorage() { throw new Error("Browser storage is unavailable"); },
  };
  assert.doesNotThrow(() => runInNewContext(THEME_BOOT_SCRIPT, context));
  assert.equal(browser.document.documentElement.getAttribute("data-theme"), null);
});
