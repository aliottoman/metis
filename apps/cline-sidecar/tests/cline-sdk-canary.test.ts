import assert from "node:assert/strict";
import { mkdtemp, readFile, realpath, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { ClineRuntime } from "../src/cline-runtime.js";
import { SecretRedactor, findSecretLeaks } from "../src/sanitize.js";

function openAiSseResponse(chunks: readonly unknown[]): Response {
  const stream = `${chunks
    .map((chunk) => `data: ${typeof chunk === "string" ? chunk : JSON.stringify(chunk)}`)
    .join("\n\n")}\n\n`;
  return new Response(stream, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

test(
  "pinned ClineCore executes canonical read_files and editor calls without persisting credentials",
  { timeout: 15_000 },
  async () => {
    const dataDirectory = await mkdtemp(join(tmpdir(), "metis-cline-tool-data-"));
    const workspaceRoot = await mkdtemp(join(tmpdir(), "metis-cline-tool-workspace-"));
    const canonicalWorkspace = await realpath(workspaceRoot);
    const seedPath = join(canonicalWorkspace, "existing.txt");
    const editedPath = join(canonicalWorkspace, "generated-by-cline.txt");
    const canary = "sk-metis-real-sdk-canary-123456789";
    const redactor = new SecretRedactor();
    let postRequests = 0;
    let advertisedToolNames: string[] = [];
    let sawToolResult = false;
    let runtime: ClineRuntime | undefined;
    let unsubscribe: (() => void) | undefined;
    const observedReadStatuses: string[] = [];
    const observedEditorStatuses: string[] = [];

    const fakeToolFetch = (
      _input: string | URL | Request,
      init?: RequestInit,
    ): Promise<Response> => {
      if ((init?.method ?? "GET").toUpperCase() === "GET") {
        return Promise.resolve(
          Response.json({ object: "list", data: [{ id: "test-model", object: "model" }] }),
        );
      }

      postRequests += 1;
      const body = JSON.parse(String(init?.body)) as {
        messages?: Array<{ role?: string }>;
        tools?: Array<{ function?: { name?: string } }>;
      };
      if (postRequests === 1) {
        advertisedToolNames = (body.tools ?? [])
          .flatMap((tool) =>
            tool.function?.name === undefined ? [] : [tool.function.name],
          )
          .sort();
      }

      if (postRequests === 1) {
        return Promise.resolve(
          openAiSseResponse([
            {
              id: "chatcmpl_tool",
              object: "chat.completion.chunk",
              choices: [
                {
                  index: 0,
                  delta: {
                    role: "assistant",
                    tool_calls: [
                      {
                        index: 0,
                        id: "call_read_canary",
                        type: "function",
                        function: {
                          name: "read_files",
                          arguments: JSON.stringify({ files: [{ path: "/src/existing.txt" }] }),
                        },
                      },
                    ],
                  },
                  finish_reason: null,
                },
              ],
            },
            {
              id: "chatcmpl_tool",
              object: "chat.completion.chunk",
              choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }],
              usage: { prompt_tokens: 7, completion_tokens: 3, total_tokens: 10 },
            },
            "[DONE]",
          ]),
        );
      }

      if (postRequests === 2) {
        sawToolResult ||= body.messages?.some((message) => message.role === "tool") ?? false;
        return Promise.resolve(
          openAiSseResponse([
            {
              id: "chatcmpl_recovered_tools",
              object: "chat.completion.chunk",
              choices: [
                {
                  index: 0,
                  delta: {
                    role: "assistant",
                    tool_calls: [
                      {
                        index: 0,
                        id: "call_read_recovered",
                        type: "function",
                        function: {
                          name: "read_files",
                          arguments: JSON.stringify({ files: [{ path: seedPath }] }),
                        },
                      },
                      {
                        index: 1,
                        id: "call_editor_canary",
                        type: "function",
                        function: {
                          name: "editor",
                          arguments: JSON.stringify({
                            path: editedPath,
                            new_text: "created through the pinned ClineCore editor\n",
                          }),
                        },
                      },
                    ],
                  },
                  finish_reason: null,
                },
              ],
            },
            {
              id: "chatcmpl_recovered_tools",
              object: "chat.completion.chunk",
              choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }],
              usage: { prompt_tokens: 11, completion_tokens: 4, total_tokens: 15 },
            },
            "[DONE]",
          ]),
        );
      }

      sawToolResult ||= body.messages?.some((message) => message.role === "tool") ?? false;
      return Promise.resolve(
        openAiSseResponse([
          {
            id: "chatcmpl_done",
            object: "chat.completion.chunk",
            choices: [
              {
                index: 0,
                delta: { role: "assistant", content: "Created the requested file." },
                finish_reason: null,
              },
            ],
          },
          {
            id: "chatcmpl_done",
            object: "chat.completion.chunk",
            choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
            usage: { prompt_tokens: 11, completion_tokens: 4, total_tokens: 15 },
          },
          "[DONE]",
        ]),
      );
    };

    try {
      await writeFile(seedPath, "read through the pinned ClineCore tool\n");
      runtime = await ClineRuntime.create(dataDirectory, redactor, {
        fetch: fakeToolFetch,
      });
      unsubscribe = runtime.subscribe((event) => {
        if (event.type === "tool" && event.tool === "read_files" && event.status !== undefined) {
          observedReadStatuses.push(event.status);
        }
        if (event.type === "tool" && event.tool === "editor" && event.status !== undefined) {
          observedEditorStatuses.push(event.status);
        }
      });
      const provider = {
        providerId: "openai-compatible",
        modelId: "test-model",
        apiKey: canary,
        baseUrl: "https://example.invalid/v1",
      };
      const result = await runtime.startSlice({
        sessionId: "sdk-tool-canary",
        prompt: "Create generated-by-cline.txt with the requested content.",
        workspaceRoot,
        provider,
        limits: { maxIterations: 4, timeoutMs: 5000, maxTokensPerTurn: 1024 },
      });

      assert.equal(result.sessionId, "sdk-tool-canary");
      assert.equal(result.state, "completed");
      assert.equal(result.finishReason, "completed");
      assert.equal(result.iterations, 3);
      assert.equal(result.toolCallCount, 3);
      assert.equal(result.summary, "Created the requested file.");
      assert.equal(postRequests, 3);
      // The SDK's shell is not advertised at all. This session supplied no
      // host bridge, so `run_check` is not registered either -- offering a
      // check Metis could not answer would be worse than not offering one.
      assert.deepEqual(advertisedToolNames, ["editor", "read_files", "search_codebase"]);
      assert.equal(sawToolResult, true);
      assert.deepEqual(observedReadStatuses, ["denied", "failed", "started", "finished"]);
      assert.deepEqual(observedEditorStatuses, ["started", "finished"]);
      assert.equal(
        await readFile(editedPath, "utf8"),
        "created through the pinned ClineCore editor\n",
      );
      assert.deepEqual(await findSecretLeaks(dataDirectory, [canary]), []);
      assert.equal(JSON.stringify(result).includes(canary), false);

      // Simulate a supervised child-process restart. Credentials are supplied
      // again, never recovered from disk. This also proves shutdown resolves
      // after a real tool turn so Python can reap that exact child generation.
      unsubscribe();
      unsubscribe = undefined;
      await runtime.shutdown();
      runtime = await ClineRuntime.create(dataDirectory, redactor, { fetch: fakeToolFetch });
      const restored = await runtime.restoreSlice({ sessionId: "sdk-tool-canary", provider });
      assert.equal(restored.model?.modelId, "test-model");
      const continued = await runtime.continueSlice({
        sessionId: "sdk-tool-canary",
        recoverySessionId: "sdk-tool-canary-recovered",
        prompt: "Continue briefly.",
        provider,
        timeoutMs: 5000,
      });
      assert.equal(continued.sessionId, "sdk-tool-canary-recovered");
      assert.equal(continued.parentSessionId, "sdk-tool-canary");
      assert.deepEqual(await findSecretLeaks(dataDirectory, [canary]), []);
      await runtime.deleteSession({ sessionId: continued.sessionId });
      await runtime.deleteSession({ sessionId: "sdk-tool-canary" });
    } finally {
      unsubscribe?.();
      await runtime?.shutdown();
      await rm(dataDirectory, { recursive: true, force: true });
      await rm(workspaceRoot, { recursive: true, force: true });
    }
  },
);
