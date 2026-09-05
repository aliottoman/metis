import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PassThrough } from "node:stream";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { FakeRuntime } from "../src/fake-runtime.js";
import { JsonlRpcServer } from "../src/server.js";
import { RpcService } from "../src/service.js";

test("NDJSON transport rejects malformed and oversized frames without executing them", async () => {
  const dataDirectory = await mkdtemp(join(tmpdir(), "metis-server-"));
  const input = new PassThrough();
  const output = new PassThrough();
  const lines: string[] = [];
  output.setEncoding("utf8");
  output.on("data", (chunk: string) => lines.push(...chunk.split("\n").filter(Boolean)));
  try {
    const server = new JsonlRpcServer(
      new RpcService(new FakeRuntime(), dataDirectory),
      input,
      output,
      { maxFrameBytes: 128 },
    );
    server.start();
    input.write("not-json\n");
    input.write(`${"x".repeat(129)}\n`);
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(JSON.parse(lines[0] ?? "{}").error.code, "INVALID_REQUEST");
    assert.equal(JSON.parse(lines[1] ?? "{}").error.code, "FRAME_TOO_LARGE");
    server.close();
  } finally {
    input.destroy();
    output.destroy();
    await rm(dataDirectory, { recursive: true, force: true });
  }
});

test(
  "CLI flushes its shutdown response and exits despite a referenced timer",
  { timeout: 12_000 },
  async () => {
    const dataDirectory = await mkdtemp(join(tmpdir(), "metis-cli-shutdown-"));
    const entrypoint = fileURLToPath(new URL("../src/index.js", import.meta.url));
    const timerFixture = fileURLToPath(
      new URL("./fixtures/referenced-timer.js", import.meta.url),
    );
    const child = spawn(
      process.execPath,
      [
        "--import",
        timerFixture,
        entrypoint,
        "--stdio",
        "--runtime",
        "fake",
        "--data-dir",
        dataDirectory,
      ],
      { stdio: ["pipe", "pipe", "pipe"] },
    );
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk: string) => {
      stderr += chunk;
    });

    let timer: NodeJS.Timeout | undefined;
    try {
      child.stdin.write(
        `${JSON.stringify({ version: "1", id: "shutdown-1", method: "shutdown", params: {} })}\n`,
      );
      type Exit = [code: number | null, signal: NodeJS.Signals | null];
      const exited = once(child, "exit") as Promise<Exit>;
      const boundedExit = new Promise<never>((_resolve, reject) => {
        timer = setTimeout(() => {
          reject(new Error(`sidecar did not exit after shutdown: ${stderr}`));
        // The CLI imports ClineCore before it can read stdin, and concurrent
        // test workers can make that startup take several seconds. Eight
        // seconds is still far below the deliberately leaked 60-second timer,
        // so this continues to prove explicit shutdown without being flaky.
        }, 8000);
      });
      const [code, signal] = await Promise.race([exited, boundedExit]);

      assert.equal(code, 0);
      assert.equal(signal, null);
      const responses = stdout
        .split("\n")
        .filter(Boolean)
        .map((line) => JSON.parse(line) as Record<string, unknown>);
      assert.deepEqual(responses, [
        {
          version: "1",
          id: "shutdown-1",
          result: { state: "shutting_down" },
        },
      ]);
    } finally {
      if (timer !== undefined) clearTimeout(timer);
      if (child.exitCode === null && child.signalCode === null) {
        child.kill("SIGKILL");
        await once(child, "exit").catch(() => undefined);
      }
      await rm(dataDirectory, { recursive: true, force: true });
    }
  },
);
