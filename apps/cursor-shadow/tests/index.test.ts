import assert from "node:assert/strict";
import { mkdtemp, mkdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import {
  changedPaths,
  descriptor,
  snapshotWorkspace,
  validateRequest,
} from "../src/index.js";

test("descriptor keeps Cursor outside the production harness", () => {
  const value = descriptor();
  assert.equal(value.productionSelectable, false);
  assert.equal(value.liveByDefault, false);
  assert.equal(value.controls.sandboxEnabled, true);
  assert.equal(value.controls.approvalDisabled, true);
  assert.equal(value.controls.oneSendPerBenchmark, true);
});

test("request validation is bounded and requires exact disjoint scope", () => {
  const request = validateRequest({
    protocolVersion: "1",
    workspace: "/tmp/metis-cursor-shadow-fixture",
    prompt: "Update the UI and its tests.",
    requiredFiles: ["app/index.html", "tests/test_ui.py"],
    protectedFiles: ["app/api.py"],
  });
  assert.equal(request.model, "composer-2.5");
  assert.equal(request.timeoutSeconds, 300);
  assert.equal(request.maxTotalTokens, 80_000);
  assert.throws(
    () => validateRequest({ ...request, requiredFiles: ["app/api.py"] }),
    /disjoint/,
  );
  assert.throws(
    () => validateRequest({ ...request, requiredFiles: ["../escape.py"] }),
    /invalid project-relative path/,
  );
});

test("workspace diff ignores adapter state and reports exact changed paths", async () => {
  const root = await mkdtemp(join(tmpdir(), "metis-cursor-shadow-test-"));
  await mkdir(join(root, "app"));
  await writeFile(join(root, "app", "main.py"), "before\n");
  const before = await snapshotWorkspace(root);
  await writeFile(join(root, "app", "main.py"), "after\n");
  await writeFile(join(root, "README.md"), "docs\n");
  await mkdir(join(root, ".cursor"));
  await writeFile(join(root, ".cursor", "state.json"), "{}\n");
  const after = await snapshotWorkspace(root);

  assert.deepEqual(changedPaths(before, after), ["README.md", "app/main.py"]);
});
