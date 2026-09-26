import assert from "node:assert/strict";
import test from "node:test";

import { createAssetCatalogStore, createAssetPinStore, assetPins, assetHref, assetView, canBeginAssetAction, launchUrl } from "../lib/assets.ts";
import type { AssetV1 } from "../lib/types.ts";

const asset = (status = "ready"): AssetV1 => ({
  id: "demo", name: "Demo", summary: "", category: "Apps", tags: [], framework: "Vite", entrypoint: "dev",
  status, launchConfigured: true, launchApproved: true, launchCommand: ["npm", "run", "dev"], buildCommand: [], envKeys: [], envFile: [], envFilePresent: false, url: status === "running" ? "http://127.0.0.1:4173" : null,
});
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

test("launch state and failures survive the catalog subscriber being replaced by its workspace", async () => {
  const start = deferred<AssetV1>();
  const store = createAssetCatalogStore({ list: async () => [asset()], scan: async () => [], perform: () => start.promise });
  const leaveCatalog = store.subscribe(() => {});
  const launching = store.act(asset(), "start");
  leaveCatalog();
  const seen: string[] = [];
  const leaveWorkspace = store.subscribe(() => { const error = store.getSnapshot().error; if (error) seen.push(error); });
  assert.deepEqual(store.getSnapshot().busy, { id: "demo", action: "start" });
  start.reject(new Error("Build command exited with code 1"));
  assert.equal(await launching, null);
  assert.equal(store.getSnapshot().busy, null);
  assert.ok(seen.some((message) => message.includes("Demo: Build command exited with code 1")));
  assert.notEqual(store.getSnapshot().assets[0]?.status, "starting");
  leaveWorkspace();
});

test("stop during a long build wins even when the old start resolves last", async () => {
  const start = deferred<AssetV1>();
  const stop = deferred<AssetV1>();
  const store = createAssetCatalogStore({ list: async () => [asset("stopped")], scan: async () => [], perform: (_id, action) => action === "start" ? start.promise : stop.promise });
  const starting = store.act(asset(), "start");
  const stopping = store.act(asset("starting"), "stop");
  assert.equal(store.getSnapshot().busy?.action, "stop");
  stop.resolve(asset("stopped"));
  assert.equal((await stopping)?.status, "stopped");
  start.resolve(asset("running"));
  assert.equal(await starting, null);
  assert.equal(store.getSnapshot().assets[0]?.status, "stopped");
  assert.equal(store.getSnapshot().busy, null);
});

test("a superseded start cannot clear a stop still in flight or publish its failure", async () => {
  const start = deferred<AssetV1>();
  const stop = deferred<AssetV1>();
  const store = createAssetCatalogStore({ list: async () => [asset("stopped")], scan: async () => [], perform: (_id, action) => action === "start" ? start.promise : stop.promise });
  const starting = store.act(asset(), "start");
  const stopping = store.act(asset("starting"), "stop");
  start.reject(new Error("Build cancelled"));
  await starting;
  assert.equal(store.getSnapshot().busy?.action, "stop");
  assert.equal(store.getSnapshot().error, null);
  stop.resolve(asset("stopped"));
  await stopping;
});

test("a catalog response from before a launch cannot overwrite its newer result", async () => {
  const oldRead = deferred<AssetV1[]>();
  let calls = 0;
  const store = createAssetCatalogStore({ list: () => ++calls === 1 ? oldRead.promise : Promise.resolve([asset("running")]), scan: async () => [], perform: async () => asset("running") });
  const loading = store.load();
  await store.act(asset(), "start");
  oldRead.resolve([asset("stopped")]);
  await loading;
  assert.equal(store.getSnapshot().assets[0]?.status, "running");
  assert.equal(store.getSnapshot().loaded, true);
});

test("concurrent workspace polling shares one read and stale reads cannot erase saved settings", async () => {
  const read = deferred<AssetV1[]>();
  let calls = 0;
  const store = createAssetCatalogStore({ list: () => { calls += 1; return read.promise; }, scan: async () => [], perform: async () => asset() });
  const first = store.load(true);
  const second = store.load(true);
  assert.equal(first, second);
  await Promise.resolve();
  assert.equal(calls, 1);
  store.update({ ...asset(), envFilePresent: true });
  read.resolve([asset()]);
  await first;
  assert.equal(store.getSnapshot().assets[0]?.envFilePresent, true);
});

test("scanning cannot race a launch, and a completed scan supersedes older loads", async () => {
  const read = deferred<AssetV1[]>();
  const scanning = deferred<AssetV1[]>();
  let actions = 0;
  const store = createAssetCatalogStore({ list: () => read.promise, scan: () => scanning.promise, perform: async () => { actions += 1; return asset(); } });
  const loading = store.load();
  const scan = store.scan();
  assert.equal(await store.act(asset(), "start"), null);
  assert.equal(actions, 0);
  scanning.resolve([{ ...asset(), name: "New catalog" }]);
  assert.equal(await scan, 1);
  read.resolve([{ ...asset(), name: "Old catalog" }]);
  await loading;
  assert.equal(store.getSnapshot().assets[0]?.name, "New catalog");
});

test("durable asset routes encode IDs and unsupported views return to the preview", () => {
  assert.equal(assetHref("demo/one", "logs"), "/assets/demo%2Fone?view=logs");
  assert.equal(assetView("unknown"), "preview");
  assert.equal(canBeginAssetAction({ id: "a", action: "start" }, "a", "stop"), true);
  assert.equal(canBeginAssetAction({ id: "a", action: "start" }, "b", "start"), false);
  assert.equal(launchUrl({ ...asset(), url: "//outside.example/app" }), null);
  assert.equal(launchUrl({ ...asset(), url: "javascript:alert(1)" }), null);
});

test("a synchronously failed catalog transport can be retried", async () => {
  let calls = 0;
  const store = createAssetCatalogStore({ list: () => { if (++calls === 1) throw new Error("Disconnected"); return Promise.resolve([asset()]); }, scan: async () => [], perform: async () => asset() });
  await store.load();
  assert.equal(store.getSnapshot().loadError, "Disconnected");
  await store.load();
  assert.equal(calls, 2);
  assert.equal(store.getSnapshot().loadError, null);
  assert.equal(store.getSnapshot().loaded, true);
});

test("a read begun during a failed launch cannot replace its authoritative failure state", async () => {
  const start = deferred<AssetV1>();
  const oldRead = deferred<AssetV1[]>();
  let reads = 0;
  const store = createAssetCatalogStore({ list: () => ++reads === 1 ? oldRead.promise : Promise.resolve([asset("failed")]), scan: async () => [], perform: () => start.promise });
  const launching = store.act(asset(), "start");
  const loading = store.load(true);
  await Promise.resolve();
  start.reject(new Error("Build failed"));
  await launching;
  oldRead.resolve([asset("starting")]);
  await loading;
  await store.load(true);
  assert.equal(store.getSnapshot().assets[0]?.status, "failed");
  assert.match(store.getSnapshot().error ?? "", /Build failed/);
});

test("pins stay usable when storage is readable but writes are blocked", () => {
  let saved = '["one"]';
  const store = createAssetPinStore(() => ({ getItem: () => saved, setItem: () => { throw new Error("Quota exceeded"); } }));
  store.toggle("two");
  assert.deepEqual(assetPins(store.read()), ["one", "two"]);
  store.toggle("one");
  assert.deepEqual(assetPins(store.read()), ["two"]);
  saved = '["from-another-window"]';
  store.sync();
  assert.deepEqual(assetPins(store.read()), ["from-another-window"]);
});

test("malformed pin preferences never break the catalog or duplicate entries", () => {
  assert.deepEqual(assetPins("bad JSON"), []);
  assert.deepEqual(assetPins('["one",null,"",42,"one","two"]'), ["one", "two"]);
});
