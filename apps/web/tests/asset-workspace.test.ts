import assert from "node:assert/strict";
import test from "node:test";
import { assetHref, assetView, isActive, launchUrl, matchesAsset, sortAssets } from "../lib/assets.ts";
import { normalizeAsset } from "../lib/api.ts";

const ready = normalizeAsset({ id: "demo", name: "Demo", launch_configured: true, launch_approved: true, status: "ready" });
test("dedicated asset URLs preserve opaque IDs and select durable tabs", () => {
  assert.equal(assetHref("a/b?#", "settings"), "/assets/a%2Fb%3F%23?view=settings");
  assert.equal(assetHref("demo"), "/assets/demo");
  assert.equal(assetView("logs"), "logs");
  assert.equal(assetView("invalid"), "preview");
});
test("catalog separates failed launches from ready assets and includes all live process states", () => {
  const failed = { ...ready, status: "failed" };
  assert.equal(matchesAsset(failed, "ready"), false);
  assert.equal(matchesAsset(failed, "failed"), true);
  for (const status of ["starting", "running", "stopping"]) {
    const asset = { ...ready, status };
    assert.equal(isActive(asset), true);
    assert.equal(matchesAsset(asset, "ready"), false);
  }
  assert.equal(matchesAsset(ready, "pinned", ["demo"]), true);
  assert.equal(matchesAsset(ready, "pinned", ["other"]), false);
  assert.equal(sortAssets([ready, failed, { ...ready, id: "live", status: "running" }])[0].id, "live");
});
test("preview URLs accept usable HTTP app locations and reject unsafe or malformed locations", () => {
  assert.equal(launchUrl({ ...ready, url: "http://127.0.0.1:8080/demo" }), "http://127.0.0.1:8080/demo");
  assert.equal(launchUrl({ ...ready, url: "/app" }), "/app");
  for (const url of ["javascript:alert(1)", "//other.test/app", "/\\other.test", "http://user:secret@host.test", "file:///tmp/app", "http://"]) assert.equal(launchUrl({ ...ready, url }), null);
});
