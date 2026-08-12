import assert from "node:assert/strict";
import test from "node:test";

import {
  getProjectProtections,
  invalidProtectionPatterns,
  parseProtectionPatterns,
  setProjectProtections,
} from "../lib/api.ts";

function jsonResponse(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "content-type": "application/json" },
  });
}

test("patterns are parsed one per line, trimmed and de-duplicated", () => {
  const parsed = parseProtectionPatterns(
    "extractor.py\n  excel_writer.py  \n\nextractor.py\nassets/*.xlsx,config.py\n",
  );

  assert.deepEqual(parsed, [
    "extractor.py",
    "excel_writer.py",
    "assets/*.xlsx",
    "config.py",
  ]);
});

test("an empty editor saves nothing rather than an empty pattern", () => {
  assert.deepEqual(parseProtectionPatterns("   \n\n  "), []);
});

test("patterns the API will refuse are named before the user saves", () => {
  const invalid = invalidProtectionPatterns([
    "extractor.py",
    "/etc/passwd",
    "../outside.py",
    "assets/*.xlsx",
  ]);

  assert.deepEqual(invalid, ["/etc/passwd", "../outside.py"]);
});

test("reading protections exposes stored patterns, resolved paths and misses", async () => {
  const original = globalThis.fetch;
  globalThis.fetch = (async () =>
    jsonResponse({
      protected_files: ["assets/*.xlsx", "billing_*.py"],
      resolved: ["assets/DHL_Template.xlsx"],
      unmatched: ["billing_*.py"],
    })) as typeof fetch;
  try {
    const result = await getProjectProtections("asset_1");

    assert.deepEqual(result.protected_files, ["assets/*.xlsx", "billing_*.py"]);
    // The preview is what makes a pattern safe to rely on.
    assert.deepEqual(result.resolved, ["assets/DHL_Template.xlsx"]);
    // And a pattern matching nothing is shown rather than silently kept.
    assert.deepEqual(result.unmatched, ["billing_*.py"]);
  } finally {
    globalThis.fetch = original;
  }
});

test("saving sends the explicit list and nothing else", async () => {
  const original = globalThis.fetch;
  const seen: { url: string; method?: string; body?: string }[] = [];
  globalThis.fetch = (async (input: unknown, init?: RequestInit) => {
    seen.push({
      url: String(input),
      ...(init?.method === undefined ? {} : { method: init.method }),
      ...(typeof init?.body === "string" ? { body: init.body } : {}),
    });
    return jsonResponse({
      protected_files: ["extractor.py"],
      resolved: ["extractor.py"],
      unmatched: [],
    });
  }) as typeof fetch;
  try {
    const result = await setProjectProtections("asset_1", ["extractor.py"]);

    assert.equal(seen.length, 1);
    assert.equal(seen[0]?.method, "PUT");
    assert.match(seen[0]?.url ?? "", /\/projects\/asset_1\/protections$/);
    assert.deepEqual(JSON.parse(seen[0]?.body ?? "{}"), {
      protected_files: ["extractor.py"],
    });
    assert.deepEqual(result.resolved, ["extractor.py"]);
  } finally {
    globalThis.fetch = original;
  }
});

test("nothing is sent unless save is called", async () => {
  const original = globalThis.fetch;
  let calls = 0;
  globalThis.fetch = (async () => {
    calls += 1;
    return jsonResponse({ protected_files: [], resolved: [], unmatched: [] });
  }) as typeof fetch;
  try {
    // Editing is local: parsing and validating never touch the API, so a
    // half-typed pattern is never persisted.
    parseProtectionPatterns("extract");
    invalidProtectionPatterns(["extract"]);

    assert.equal(calls, 0);
  } finally {
    globalThis.fetch = original;
  }
});
