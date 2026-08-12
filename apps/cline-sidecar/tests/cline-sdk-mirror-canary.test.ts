import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { ClineRuntime } from "../src/cline-runtime.js";
import { RpcFault } from "../src/protocol.js";
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

// Kept in its own file/process: ClineRuntime pins one Cline data directory
// per process (configureStorage), so it cannot share a process with another
// suite that already configured a different one.
test(
  "a credential already present in the workspace mirror fails the session closed",
  { timeout: 15_000 },
  async () => {
    const dataDirectory = await mkdtemp(join(tmpdir(), "metis-cline-mirror-canary-data-"));
    const workspaceRoot = await mkdtemp(join(tmpdir(), "metis-cline-mirror-canary-workspace-"));
    const canary = "sk-metis-mirror-leak-canary-123456789";
    const redactor = new SecretRedactor();
    let runtime: ClineRuntime | undefined;

    // A trivial immediate completion: the scan under test must run
    // regardless of what the model did or did not do during the turn.
    const immediateStopFetch = (
      _input: string | URL | Request,
      init?: RequestInit,
    ): Promise<Response> => {
      if ((init?.method ?? "GET").toUpperCase() === "GET") {
        return Promise.resolve(
          Response.json({ object: "list", data: [{ id: "test-model", object: "model" }] }),
        );
      }
      return Promise.resolve(
        openAiSseResponse([
          {
            id: "chatcmpl_stop",
            object: "chat.completion.chunk",
            choices: [
              {
                index: 0,
                delta: { role: "assistant", content: "Nothing to do." },
                finish_reason: null,
              },
            ],
          },
          {
            id: "chatcmpl_stop",
            object: "chat.completion.chunk",
            choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
            usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 },
          },
          "[DONE]",
        ]),
      );
    };

    try {
      // Simulates a defect elsewhere already having leaked the literal key
      // into a file the model can read -- not something ClineCore itself
      // should ever produce, but the canary must catch it regardless of
      // where the bytes came from.
      await writeFile(join(workspaceRoot, "leaked-config.py"), `API_KEY = "${canary}"\n`);
      runtime = await ClineRuntime.create(dataDirectory, redactor, {
        fetch: immediateStopFetch,
      });
      const provider = {
        providerId: "openai-compatible",
        modelId: "test-model",
        apiKey: canary,
        baseUrl: "https://example.invalid/v1",
      };

      await assert.rejects(
        runtime.startSlice({
          sessionId: "mirror-leak-canary",
          prompt: "Say there is nothing to do.",
          workspaceRoot,
          provider,
          limits: { maxIterations: 2, timeoutMs: 5000, maxTokensPerTurn: 512 },
        }),
        (error: unknown) =>
          error instanceof RpcFault &&
          error.code === "POLICY_DENIED" &&
          Array.isArray(error.data?.files) &&
          (error.data.files as string[]).some((file) => file.startsWith("workspace:")),
      );

      // Fail-closed means gone, not just flagged: the SDK's own session
      // record must not survive the rejected session either.
      await assert.rejects(
        runtime.getUsage({ sessionId: "mirror-leak-canary" }),
        (error: unknown) => error instanceof RpcFault && error.code === "SESSION_NOT_FOUND",
      );

      // The scan is fail-closed, not fail-loud-then-continue: the leaked file
      // itself is left exactly as found, since Metis never edits the user's
      // real project from this boundary -- only the private session is torn
      // down.
      assert.deepEqual(await findSecretLeaks(workspaceRoot, [canary]), ["leaked-config.py"]);
    } finally {
      await runtime?.shutdown();
      await rm(dataDirectory, { recursive: true, force: true });
      await rm(workspaceRoot, { recursive: true, force: true });
    }
  },
);
