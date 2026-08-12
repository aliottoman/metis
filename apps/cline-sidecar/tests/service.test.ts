import assert from "node:assert/strict";
import { access, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { FakeRuntime } from "../src/fake-runtime.js";
import { parseRpcRequest } from "../src/protocol.js";
import { RpcService, type RpcClient } from "../src/service.js";

test("delete clears replay data when the SDK record is already absent", async () => {
  const dataDirectory = await mkdtemp(join(tmpdir(), "metis-service-"));
  const runtime = new FakeRuntime();
  const service = new RpcService(runtime, dataDirectory);
  const client: RpcClient = {
    subscriptions: new Set(),
    deliver: () => undefined,
  };
  const sessionId = "already-removed";
  const eventPath = join(dataDirectory, "metis-events-v1", `${sessionId}.jsonl`);
  try {
    await service.dispatch(
      parseRpcRequest({
        version: "1",
        id: "start",
        method: "startSlice",
        params: {
          sessionId,
          prompt: "Create replay events",
          workspaceRoot: join(dataDirectory, "workspace"),
          provider: { providerId: "fake", modelId: "fake" },
        },
      }),
      client,
    );
    for (let attempt = 0; attempt < 20; attempt += 1) {
      try {
        await access(eventPath);
        break;
      } catch {
        await new Promise((resolve) => setImmediate(resolve));
      }
    }
    await access(eventPath);

    assert.equal((await runtime.deleteSession({ sessionId })).deleted, true);
    const result = (await service.dispatch(
      parseRpcRequest({
        version: "1",
        id: "delete",
        method: "deleteSession",
        params: { sessionId },
      }),
      client,
    )) as { deleted: boolean };
    assert.equal(result.deleted, false);
    await assert.rejects(access(eventPath), { code: "ENOENT" });
  } finally {
    service.disconnect(client);
    await runtime.shutdown();
    await rm(dataDirectory, { recursive: true, force: true });
  }
});
