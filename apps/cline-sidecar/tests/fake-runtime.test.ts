import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { FakeRuntime } from "../src/fake-runtime.js";

test("deterministic fake mode enforces iteration and wall-time limits without inference", async () => {
  const workspaceRoot = await mkdtemp(join(tmpdir(), "metis-fake-workspace-"));
  try {
    const runtime = new FakeRuntime();
    const iterations = await runtime.startSlice({
      sessionId: "iterations",
      prompt: "[fake:iterations=5]",
      workspaceRoot,
      provider: { providerId: "fake", modelId: "fake" },
      limits: { maxIterations: 2 },
    });
  assert.equal(iterations.state, "failed");
    assert.equal(iterations.finishReason, "error");
    assert.equal(iterations.controlledStopReason, "max_iterations");
    assert.equal(iterations.usage.requests, 2);
    assert.equal(iterations.summary, "Agent runtime exceeded maxIterations (2)");

    const timeout = await runtime.startSlice({
      sessionId: "timeout",
      prompt: "[fake:duration=2000]",
      workspaceRoot,
      provider: { providerId: "fake", modelId: "fake" },
      limits: { timeoutMs: 1000 },
    });
    assert.equal(timeout.state, "failed");
    assert.match(timeout.summary ?? "", /wall-time/);
  } finally {
    await rm(workspaceRoot, { recursive: true, force: true });
  }
});

test("deleting the active repair session releases its private ancestry", async () => {
  const workspaceRoot = await mkdtemp(join(tmpdir(), "metis-fake-workspace-"));
  try {
    const runtime = new FakeRuntime();
    await runtime.startSlice({
      sessionId: "build",
      prompt: "Build",
      workspaceRoot,
      provider: { providerId: "fake", modelId: "broad" },
    });
    await runtime.restartWithModel({
      sessionId: "build",
      newSessionId: "repair",
      prompt: "Repair",
      provider: { providerId: "fake", modelId: "repair" },
    });

    const deleted = await runtime.deleteSession({ sessionId: "repair" });
    assert.equal(deleted.deleted, true);
    assert.deepEqual(deleted.deletedSessionIds, ["repair", "build"]);
    assert.equal((await runtime.deleteSession({ sessionId: "build" })).deleted, false);
  } finally {
    await rm(workspaceRoot, { recursive: true, force: true });
  }
});
