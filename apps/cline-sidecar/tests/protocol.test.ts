import assert from "node:assert/strict";
import { test } from "node:test";

import {
  RpcFault,
  parseRpcRequest,
  type ContinueSliceParams,
  type StartSliceParams,
  type SubscriptionParams,
} from "../src/protocol.js";

const validStart = {
  version: "1",
  id: "request-1",
  method: "startSlice",
  params: {
    sessionId: "session-1",
    prompt: "Build the document review slice",
    workspaceRoot: "/tmp/metis-workspace",
    provider: {
      providerId: "cline-pass",
      modelId: "cline-pass/deepseek-v4-pro",
      apiKey: "secret-canary-value",
    },
    limits: { maxIterations: 24, timeoutMs: 60_000, maxTokensPerTurn: 16_384 },
  },
};

test("parses and normalizes the strict v1 start contract", () => {
  const request = parseRpcRequest(validStart);
  assert.equal(request.method, "startSlice");
  const params = request.params as StartSliceParams;
  assert.equal(params.provider.providerId, "cline-pass");
  assert.equal(params.provider.modelId, "cline-pass/deepseek-v4-pro");
  assert.equal(params.limits?.maxIterations, 24);
});

test("rejects unknown fields instead of silently widening authority", () => {
  assert.throws(
    () =>
      parseRpcRequest({
        ...validStart,
        params: { ...validStart.params, enableShell: true },
      }),
    (error: unknown) => error instanceof RpcFault && error.code === "INVALID_REQUEST",
  );
});

test("rejects an unknown RPC method", () => {
  assert.throws(
    () => parseRpcRequest({ version: "1", id: "r", method: "runShell", params: {} }),
    (error: unknown) => error instanceof RpcFault && error.code === "METHOD_NOT_FOUND",
  );
});

test("requires a non-root absolute workspace", () => {
  for (const workspaceRoot of ["relative/path", "/"]) {
    assert.throws(
      () =>
        parseRpcRequest({
          ...validStart,
          params: { ...validStart.params, workspaceRoot },
        }),
      RpcFault,
    );
  }
});

test("rejects provider URLs containing embedded credentials", () => {
  assert.throws(
    () =>
      parseRpcRequest({
        ...validStart,
        params: {
          ...validStart.params,
          provider: {
            providerId: "openai-compatible",
            modelId: "model",
            baseUrl: "https://user:password@example.invalid/v1",
          },
        },
      }),
    RpcFault,
  );
});

test("parses an event replay cursor", () => {
  const request = parseRpcRequest({
    version: "1",
    id: "subscribe-1",
    method: "subscribe",
    params: { sessionId: "session-1", afterCursor: 19 },
  });
  assert.deepEqual(request.params as SubscriptionParams, {
    sessionId: "session-1",
    afterCursor: 19,
  });
});

test("preserves the host-chosen continuation recovery identity", () => {
  const request = parseRpcRequest({
    version: "1",
    id: "continue-1",
    method: "continueSlice",
    params: {
      sessionId: "session-1",
      recoverySessionId: "session-1-round-2",
      prompt: "Repair the exact verifier finding",
      provider: validStart.params.provider,
    },
  });
  assert.equal(
    (request.params as ContinueSliceParams).recoverySessionId,
    "session-1-round-2",
  );
});

test("enforces bounded model-loop limits", () => {
  assert.throws(
    () =>
      parseRpcRequest({
        ...validStart,
        params: { ...validStart.params, limits: { maxIterations: 201 } },
      }),
    RpcFault,
  );
});

test("parses only an empty startup-handshake request", () => {
  const request = parseRpcRequest({
    version: "1",
    id: "info-1",
    method: "getInfo",
    params: {},
  });
  assert.equal(request.method, "getInfo");
  assert.deepEqual(request.params, {});
  assert.throws(
    () =>
      parseRpcRequest({
        version: "1",
        id: "info-2",
        method: "getInfo",
        params: { runtime: "fake" },
      }),
    RpcFault,
  );
});
