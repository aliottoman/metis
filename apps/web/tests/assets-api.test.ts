import assert from "node:assert/strict";
import test from "node:test";

import {
  approveAsset,
  getAssetLogs,
  listAssets,
  saveAssetEnv,
  scanAssets,
  startAsset,
  stopAsset,
  revokeAssetApproval,
} from "../lib/api.ts";

function jsonResponse(value: unknown): Response {
  return new Response(JSON.stringify(value), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

test("normalizes the asset catalog contract and legacy field casing", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => jsonResponse({
    assets: [{
      asset_id: "demo-one",
      name: "Demo One",
      description: "A compact project demo.",
      category: "Demos",
      tags: ["react", "local"],
      framework: "Next.js",
      entrypoint: "npm run dev",
      status: "ready",
      launch_configured: true,
      launch_approved: true,
      launch_command: ["pnpm", "dev"],
      env_keys: ["DAC_OCID"],
      env_file: [
        { key: "DAC_OCID", is_set: true, sensitive: false },
        { key: "DAC_API_KEY", is_set: false, sensitive: true },
      ],
      env_file_present: true,
      launch_url: null,
    }],
  });

  try {
    const assets = await listAssets();
    assert.deepEqual(assets[0], {
      id: "demo-one",
      name: "Demo One",
      summary: "A compact project demo.",
      category: "Demos",
      tags: ["react", "local"],
      framework: "Next.js",
      entrypoint: "npm run dev",
      status: "ready",
      launchConfigured: true,
      launchApproved: true,
      launchCommand: ["pnpm", "dev"],
      envKeys: ["DAC_OCID"],
      envFile: [
        { key: "DAC_OCID", isSet: true, sensitive: false },
        { key: "DAC_API_KEY", isSet: false, sensitive: true },
      ],
      envFilePresent: true,
      url: null,
    });
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("writes runtime values to the project .env through the v1 route", async () => {
  const originalFetch = globalThis.fetch;
  const calls: Array<{ url: string; method: string; body?: unknown }> = [];
  globalThis.fetch = async (input, init) => {
    calls.push({
      url: String(input),
      method: init?.method ?? "GET",
      body: init?.body ? JSON.parse(String(init.body)) as unknown : undefined,
    });
    return jsonResponse({
      id: "demo/one",
      name: "Demo One",
      summary: "A demo.",
      category: "Demos",
      tags: [],
      framework: "Vite",
      entrypoint: "npm run dev",
      status: "ready",
      launchConfigured: true,
      launchApproved: true,
      launchCommand: ["npm", "run", "dev"],
      envKeys: [],
      env_file: [{ key: "API_TOKEN", is_set: true, sensitive: true }],
      env_file_present: true,
      url: null,
    });
  };

  try {
    const saved = await saveAssetEnv("demo/one", { API_TOKEN: "written-to-disk" });
    assert.match(calls[0]!.url, /\/assets\/demo%2Fone\/env$/);
    assert.equal(calls[0]!.method, "PUT");
    assert.deepEqual(calls[0]!.body, { values: { API_TOKEN: "written-to-disk" } });
    // The response carries presence only — never the value that was just sent.
    assert.deepEqual(saved.envFile, [{ key: "API_TOKEN", isSet: true, sensitive: true }]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("scans, starts, stops, and loads asset logs through the v1 routes", async () => {
  const originalFetch = globalThis.fetch;
  const calls: Array<{ url: string; method: string; body?: unknown }> = [];
  globalThis.fetch = async (input, init) => {
    const url = String(input);
    calls.push({
      url,
      method: init?.method ?? "GET",
      body: init?.body ? JSON.parse(String(init.body)) as unknown : undefined,
    });
    if (url.endsWith("/scan")) return jsonResponse([]);
    if (url.endsWith("/logs")) return jsonResponse({ assetId: "demo/one", logs: ["ready", "listening"] });
    return jsonResponse({
      id: "demo/one",
      name: "Demo One",
      summary: "A demo.",
      category: "Demos",
      tags: [],
      framework: "Vite",
      entrypoint: "npm run dev",
      status: url.endsWith("/stop") ? "stopped" : "running",
      launchConfigured: true,
      launchApproved: true,
      launchCommand: ["npm", "run", "dev"],
      envKeys: ["API_TOKEN"],
      url: url.endsWith("/stop") ? null : "http://127.0.0.1:4173",
    });
  };

  try {
    await scanAssets();
    await approveAsset("demo/one");
    const started = await startAsset("demo/one", { API_TOKEN: "session-only" });
    const stopped = await stopAsset("demo/one");
    await revokeAssetApproval("demo/one");
    const logResult = await getAssetLogs("demo/one");

    assert.match(calls[0]!.url, /\/api\/v1\/assets\/scan$/);
    assert.equal(calls[0]!.method, "POST");
    assert.match(calls[1]!.url, /\/assets\/demo%2Fone\/approval$/);
    assert.equal(calls[1]!.method, "POST");
    assert.match(calls[2]!.url, /\/assets\/demo%2Fone\/start$/);
    assert.deepEqual(calls[2]!.body, { env: { API_TOKEN: "session-only" } });
    assert.match(calls[3]!.url, /\/assets\/demo%2Fone\/stop$/);
    assert.match(calls[4]!.url, /\/assets\/demo%2Fone\/approval$/);
    assert.equal(calls[4]!.method, "DELETE");
    assert.match(calls[5]!.url, /\/assets\/demo%2Fone\/logs$/);
    assert.equal(started.status, "running");
    assert.equal(stopped.status, "stopped");
    assert.equal(logResult.logs, "ready\nlistening");
  } finally {
    globalThis.fetch = originalFetch;
  }
});
