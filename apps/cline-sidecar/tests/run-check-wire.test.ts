/**
 * What the model is actually sent, taken off the wire.
 *
 * The previous design could pass every unit test and still fail live, because
 * the thing that broke was the *schema the provider advertised*: a shell tool's
 * `command`/`args`/`cwd`. So this drives the pinned ClineCore through a fake
 * provider and reads the tool definitions out of the real request body.
 */

import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { ClineRuntime } from "../src/cline-runtime.js";
import { HostBridge } from "../src/host-bridge.js";
import { SecretRedactor } from "../src/sanitize.js";
import type { HostCheck, HostCheckResult } from "../src/protocol.js";

interface WireTool {
  type?: string;
  function?: {
    name?: string;
    description?: string;
    parameters?: Record<string, unknown>;
  };
}

function sse(chunks: readonly unknown[]): Response {
  const body = `${chunks
    .map((chunk) => `data: ${typeof chunk === "string" ? chunk : JSON.stringify(chunk)}`)
    .join("\n\n")}\n\n`;
  return new Response(body, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

function toolCallChunk(id: string, name: string, args: unknown) {
  return [
    {
      id: `chatcmpl_${id}`,
      object: "chat.completion.chunk",
      choices: [
        {
          index: 0,
          delta: {
            role: "assistant",
            tool_calls: [
              {
                index: 0,
                id,
                type: "function",
                function: { name, arguments: JSON.stringify(args) },
              },
            ],
          },
          finish_reason: null,
        },
      ],
    },
    {
      id: `chatcmpl_${id}_done`,
      object: "chat.completion.chunk",
      choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }],
      usage: { prompt_tokens: 10, completion_tokens: 4, total_tokens: 14 },
    },
    "[DONE]",
  ];
}

function finalChunk(text: string) {
  return [
    {
      id: "chatcmpl_final",
      object: "chat.completion.chunk",
      choices: [
        { index: 0, delta: { role: "assistant", content: text }, finish_reason: null },
      ],
    },
    {
      id: "chatcmpl_final_done",
      object: "chat.completion.chunk",
      choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
      usage: { prompt_tokens: 12, completion_tokens: 5, total_tokens: 17 },
    },
    "[DONE]",
  ];
}

// A process may host only one Cline storage root, so both tests below share
// this one; each still gets its own workspace and its own session id.
const sharedDataDirectory = await mkdtemp(join(tmpdir(), "metis-runcheck-data-"));

test(
  "the model is shown run_check with the closed enum, and never a shell",
  { timeout: 20_000 },
  async () => {
    const dataDirectory = sharedDataDirectory;
    const workspaceRoot = await mkdtemp(join(tmpdir(), "metis-runcheck-ws-"));
    let wireTools: WireTool[] = [];
    let posts = 0;
    let runtime: ClineRuntime | undefined;

    const fakeFetch = (
      _input: string | URL | Request,
      init?: RequestInit,
    ): Promise<Response> => {
      if ((init?.method ?? "GET").toUpperCase() === "GET") {
        return Promise.resolve(
          Response.json({ object: "list", data: [{ id: "test-model", object: "model" }] }),
        );
      }
      posts += 1;
      const body = JSON.parse(String(init?.body)) as { tools?: WireTool[] };
      if (posts === 1) wireTools = body.tools ?? [];
      // One valid check, then finish.
      if (posts === 1) {
        return Promise.resolve(sse(toolCallChunk("call_1", "run_check", { check: "pytest" })));
      }
      return Promise.resolve(sse(finalChunk("Checked.")));
    };

    try {
      await writeFile(join(workspaceRoot, "seed.txt"), "seed\n");
      runtime = await ClineRuntime.create(dataDirectory, new SecretRedactor(), {
        fetch: fakeFetch,
      });
      const asked: HostCheck[] = [];
      const bridge = new HostBridge((frame) => {
        asked.push(frame.params.check);
        queueMicrotask(() =>
          bridge.settle(frame.id, {
            check: frame.params.check,
            ok: true,
            errors: 0,
            warnings: 0,
            findings: [],
            durationMs: 3,
            truncated: false,
          } satisfies HostCheckResult),
        );
      });
      runtime.attachHost(bridge);

      const result = await runtime.startSlice({
        sessionId: "runcheck-wire",
        prompt: "Run the tests.",
        workspaceRoot,
        provider: {
          providerId: "openai-compatible",
          modelId: "test-model",
          apiKey: "sk-metis-runcheck-canary-0001",
          baseUrl: "https://example.invalid/v1",
        },
        limits: { maxIterations: 4, timeoutMs: 8_000, maxTokensPerTurn: 1024 },
      });

      const names = wireTools
        .flatMap((tool) => (tool.function?.name === undefined ? [] : [tool.function.name]))
        .sort();

      // 1. The shell is absent from the model's tool schema entirely.
      assert.equal(names.includes("run_commands"), false);
      assert.deepEqual(names, ["editor", "read_files", "run_check", "search_codebase"]);

      // 2. run_check is present with exactly the closed enum and nothing else.
      const runCheck = wireTools.find((tool) => tool.function?.name === "run_check");
      assert.ok(runCheck, "run_check was not advertised");
      const parameters = runCheck.function?.parameters as {
        type?: string;
        properties?: Record<string, { type?: string; enum?: string[] }>;
        required?: string[];
        additionalProperties?: boolean;
      };
      assert.equal(parameters.type, "object");
      assert.deepEqual(Object.keys(parameters.properties ?? {}), ["check"]);
      assert.deepEqual(parameters.properties?.check?.enum, [
        "imports",
        "pytest",
        "ruff",
        "acceptance",
        "full",
      ]);
      assert.deepEqual(parameters.required, ["check"]);
      assert.equal(parameters.additionalProperties, false);

      // 3. The call actually reached Metis and the session finished cleanly.
      assert.deepEqual(asked, ["pytest"]);
      assert.equal(result.state, "completed");
      assert.equal(result.finishReason, "completed");
    } finally {
      await runtime?.deleteSession({ sessionId: "runcheck-wire" }).catch(() => undefined);
      await runtime?.shutdown();
      await rm(workspaceRoot, { recursive: true, force: true });
    }
  },
);

test(
  "three valid checks in a row finish the turn instead of tripping the mistake limit",
  { timeout: 20_000 },
  async () => {
    const dataDirectory = sharedDataDirectory;
    const workspaceRoot = await mkdtemp(join(tmpdir(), "metis-runcheck-seq-ws-"));
    let posts = 0;
    let runtime: ClineRuntime | undefined;
    const sequence = ["imports", "pytest", "ruff"] as const;

    const fakeFetch = (
      _input: string | URL | Request,
      init?: RequestInit,
    ): Promise<Response> => {
      if ((init?.method ?? "GET").toUpperCase() === "GET") {
        return Promise.resolve(
          Response.json({ object: "list", data: [{ id: "test-model", object: "model" }] }),
        );
      }
      posts += 1;
      if (posts <= sequence.length) {
        return Promise.resolve(
          sse(toolCallChunk(`call_${posts}`, "run_check", { check: sequence[posts - 1] })),
        );
      }
      return Promise.resolve(sse(finalChunk("All three checks are clean.")));
    };

    try {
      await writeFile(join(workspaceRoot, "seed.txt"), "seed\n");
      runtime = await ClineRuntime.create(dataDirectory, new SecretRedactor(), {
        fetch: fakeFetch,
      });
      const asked: HostCheck[] = [];
      const bridge = new HostBridge((frame) => {
        asked.push(frame.params.check);
        queueMicrotask(() =>
          bridge.settle(frame.id, {
            check: frame.params.check,
            ok: true,
            errors: 0,
            warnings: 0,
            findings: [],
            durationMs: 2,
            truncated: false,
          } satisfies HostCheckResult),
        );
      });
      runtime.attachHost(bridge);

      const result = await runtime.startSlice({
        sessionId: "runcheck-seq",
        prompt: "Verify the workspace three ways.",
        workspaceRoot,
        provider: {
          providerId: "openai-compatible",
          modelId: "test-model",
          apiKey: "sk-metis-runcheck-canary-0002",
          baseUrl: "https://example.invalid/v1",
        },
        limits: { maxIterations: 10, timeoutMs: 10_000, maxTokensPerTurn: 1024 },
      });

      // Every check ran, in order, and the turn ended on its own terms rather
      // than on maxConsecutiveMistakes.
      assert.deepEqual(asked, ["imports", "pytest", "ruff"]);
      assert.equal(result.state, "completed");
      assert.equal(result.finishReason, "completed");
      assert.equal(result.summary, "All three checks are clean.");
      assert.equal(result.toolCallCount, 3);
    } finally {
      await runtime?.deleteSession({ sessionId: "runcheck-seq" }).catch(() => undefined);
      await runtime?.shutdown();
      await rm(workspaceRoot, { recursive: true, force: true });
    }
  },
);
