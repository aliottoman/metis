#!/usr/bin/env node

import { mkdir, realpath } from "node:fs/promises";
import { isAbsolute, parse, resolve } from "node:path";
import { pathToFileURL } from "node:url";

import { ClineRuntime } from "./cline-runtime.js";
import { FakeRuntime } from "./fake-runtime.js";
import { JsonlRpcServer } from "./server.js";
import { SecretRedactor } from "./sanitize.js";
import { RpcService } from "./service.js";

type RuntimeMode = "cline" | "fake";

interface CliOptions {
  runtime: RuntimeMode;
  dataDirectory: string;
  maxReplayEvents: number;
}

function parseInteger(
  value: string | undefined,
  label: string,
  minimum: number,
  maximum: number,
): number {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < minimum || parsed > maximum) {
    throw new Error(
      `${label} must be an integer from ${minimum} through ${maximum}`,
    );
  }
  return parsed;
}

export function parseCliOptions(argv: readonly string[]): CliOptions {
  let stdio = false;
  let runtime: RuntimeMode | undefined;
  let dataDirectory: string | undefined;
  let maxReplayEvents = 256;
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === "--stdio") {
      stdio = true;
      continue;
    }
    if (argument === "--runtime") {
      const value = argv[++index];
      if (value !== "cline" && value !== "fake")
        throw new Error("--runtime must be cline or fake");
      runtime = value;
      continue;
    }
    if (argument === "--data-dir") {
      dataDirectory = argv[++index];
      continue;
    }
    if (argument === "--max-replay-events") {
      maxReplayEvents = parseInteger(
        argv[++index],
        "--max-replay-events",
        2,
        4096,
      );
      continue;
    }
    throw new Error(`Unknown sidecar argument: ${argument ?? "<missing>"}`);
  }
  if (!stdio)
    throw new Error("--stdio is required; no network listener is exposed");
  if (runtime === undefined) throw new Error("--runtime is required");
  if (dataDirectory === undefined || !isAbsolute(dataDirectory)) {
    throw new Error("--data-dir must be an explicit absolute Metis-owned path");
  }
  const normalized = resolve(dataDirectory);
  if (normalized === parse(normalized).root)
    throw new Error("--data-dir must not be a filesystem root");
  return { runtime, dataDirectory: normalized, maxReplayEvents };
}

export async function main(argv = process.argv.slice(2)): Promise<void> {
  const options = parseCliOptions(argv);
  await mkdir(options.dataDirectory, { recursive: true, mode: 0o700 });
  const dataDirectory = await realpath(options.dataDirectory);
  const redactor = new SecretRedactor();
  const runtime =
    options.runtime === "fake"
      ? new FakeRuntime(redactor)
      : await ClineRuntime.create(dataDirectory, redactor);
  const service = new RpcService(
    runtime,
    dataDirectory,
    options.maxReplayEvents,
    redactor,
  );
  const server = new JsonlRpcServer(service, process.stdin, process.stdout, {
    // run_check is the one call that travels sidecar -> host, so the runtime
    // only learns how to make it once stdio exists.
    onHostBridge: (bridge) => {
      const attach = (runtime as { attachHost?: (value: unknown) => void })
        .attachHost;
      if (typeof attach === "function") attach.call(runtime, bridge);
    },
    onShutdown: () => {
      server.close();
      process.stdin.pause();
      // ClineCore 0.0.72 can retain a referenced editor timeout after the
      // operation has completed. JsonlRpcServer has already flushed the
      // shutdown response, so this child can now terminate deterministically.
      process.exit(0);
    },
  });
  const stop = (): void => {
    void runtime.shutdown().finally(() => {
      server.close();
      process.exit(0);
    });
  };
  process.once("SIGINT", stop);
  process.once("SIGTERM", stop);
  server.start();
}

const invokedPath = process.argv[1];
if (
  invokedPath !== undefined &&
  import.meta.url === pathToFileURL(resolve(invokedPath)).href
) {
  void main().catch((error: unknown) => {
    const message =
      error instanceof Error ? error.message : "Unknown sidecar startup error";
    process.stderr.write(`Metis Cline sidecar failed to start: ${message}\n`);
    process.exitCode = 1;
  });
}

export * from "./protocol.js";
