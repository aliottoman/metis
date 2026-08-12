import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { FakeRuntime } from "../src/fake-runtime.js";
import { parseRpcRequest } from "../src/protocol.js";
import { SecretRedactor, findSecretLeaks } from "../src/sanitize.js";
import { RpcService, type RpcClient } from "../src/service.js";

test("provider credentials never enter results, events, or sidecar-owned artifacts", async () => {
  const dataDirectory = await mkdtemp(join(tmpdir(), "metis-secret-data-"));
  const workspaceRoot = await mkdtemp(join(tmpdir(), "metis-secret-workspace-"));
  const canary = "sk-metis-secret-canary-123456789";
  try {
    const redactor = new SecretRedactor();
    const runtime = new FakeRuntime(redactor);
    const service = new RpcService(runtime, dataDirectory, 32, redactor);
    const events: unknown[] = [];
    const client: RpcClient = {
      subscriptions: new Set<string>(),
      deliver: (event) => events.push(event),
    };
    await service.dispatch(
      parseRpcRequest({
        version: "1",
        id: "subscribe",
        method: "subscribe",
        params: { sessionId: "secret-session", afterCursor: 0 },
      }),
      client,
    );
    const result = await service.dispatch(
      parseRpcRequest({
        version: "1",
        id: "start",
        method: "startSlice",
        params: {
          sessionId: "secret-session",
          prompt: `Do not echo ${canary}`,
          workspaceRoot,
          provider: { providerId: "fake", modelId: "fake", apiKey: canary },
        },
      }),
      client,
    );
    await new Promise((resolve) => setImmediate(resolve));
    const serialized = JSON.stringify({ result, events });
    assert.equal(serialized.includes(canary), false);
    assert.equal(await findSecretLeaks(dataDirectory, [canary]).then((leaks) => leaks.length), 0);

    const deliberateLeak = join(dataDirectory, "deliberate-leak.txt");
    await writeFile(deliberateLeak, canary);
    assert.deepEqual(await findSecretLeaks(dataDirectory, [canary]), ["deliberate-leak.txt"]);
  } finally {
    await rm(dataDirectory, { recursive: true, force: true });
    await rm(workspaceRoot, { recursive: true, force: true });
  }
});
