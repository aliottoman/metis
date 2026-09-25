import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { ClineRuntime } from "../src/cline-runtime.js";

test("coding runtime overrides persisted native web search enablement", async () => {
  const dataDirectory = await mkdtemp(join(tmpdir(), "metis-cline-search-"));
  const settingsPath = join(dataDirectory, "settings", "global-settings.json");
  let runtime: ClineRuntime | undefined;
  try {
    await mkdir(join(dataDirectory, "settings"));
    await writeFile(settingsPath, JSON.stringify({ tools: { web_search: { enabled: true } } }));
    runtime = await ClineRuntime.create(dataDirectory, undefined, {
      fetch: async () => Response.json({ object: "list", data: [] }),
    });
    const sdk = await import("@cline/sdk");
    const persisted = JSON.parse(await readFile(settingsPath, "utf8")) as {
      tools?: { web_search?: { enabled?: boolean } };
    };
    assert.equal(persisted.tools?.web_search?.enabled, false);
    assert.equal(sdk.isModelToolEnabledGlobally("web_search"), false);
  } finally {
    await runtime?.shutdown();
    await rm(dataDirectory, { recursive: true, force: true });
  }
});
