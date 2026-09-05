#!/usr/bin/env node

import { createHash } from "node:crypto";
import { lstat, readFile, readdir, realpath } from "node:fs/promises";
import { isAbsolute, relative, resolve, sep } from "node:path";
import { pathToFileURL } from "node:url";

const PROTOCOL_VERSION = "1" as const;
const DEFAULT_TIMEOUT_SECONDS = 300;
const DEFAULT_MAX_TOTAL_TOKENS = 80_000;
const MAX_PROMPT_CHARS = 40_000;
const IGNORED_ROOTS = new Set([".cursor", ".git", ".metis", "node_modules"]);

export type ShadowRequest = {
  protocolVersion: typeof PROTOCOL_VERSION;
  workspace: string;
  prompt: string;
  requiredFiles: string[];
  protectedFiles: string[];
  model?: string;
  timeoutSeconds?: number;
  maxTotalTokens?: number;
};

type TokenUsage = {
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
  cacheReadTokens?: number;
  cacheWriteTokens?: number;
};

type FileSnapshot = Map<string, string>;

export function descriptor() {
  return {
    protocolVersion: PROTOCOL_VERSION,
    engine: "cursor-sdk-shadow",
    productionSelectable: false,
    liveByDefault: false,
    runtime: "local",
    sdk: "@cursor/sdk",
    controls: {
      disposableWorkspaceRequired: true,
      disposableWorkspaceMarker: ".metis-cursor-shadow",
      sandboxEnabled: true,
      ambientSettingsLoaded: false,
      mcpEnabled: false,
      subagentsEnabled: false,
      webEnabled: false,
      oneSendPerBenchmark: true,
      timeoutSeconds: DEFAULT_TIMEOUT_SECONDS,
      maxTotalTokens: DEFAULT_MAX_TOTAL_TOKENS,
      finalDiffMustMatchRequestedScope: true,
      protectedDiffRejected: true,
      independentMetisVerificationRequired: true,
      approvalDisabled: true,
    },
  };
}

function canonicalPath(value: string): string {
  if (
    !value ||
    value.startsWith("/") ||
    value.includes("\\") ||
    value.split("/").some((part) => !part || part === "." || part === "..")
  ) {
    throw new Error(`invalid project-relative path: ${JSON.stringify(value)}`);
  }
  return value;
}

export function validateRequest(value: unknown): ShadowRequest {
  if (!value || typeof value !== "object") throw new Error("request must be an object");
  const request = value as Partial<ShadowRequest>;
  if (request.protocolVersion !== PROTOCOL_VERSION) {
    throw new Error(`protocolVersion must be ${PROTOCOL_VERSION}`);
  }
  if (typeof request.workspace !== "string" || !isAbsolute(request.workspace)) {
    throw new Error("workspace must be an absolute path");
  }
  if (
    typeof request.prompt !== "string" ||
    !request.prompt.trim() ||
    request.prompt.length > MAX_PROMPT_CHARS
  ) {
    throw new Error(`prompt must contain 1-${MAX_PROMPT_CHARS} characters`);
  }
  const requiredFiles = (request.requiredFiles ?? []).map(canonicalPath);
  const protectedFiles = (request.protectedFiles ?? []).map(canonicalPath);
  if (!requiredFiles.length || new Set(requiredFiles).size !== requiredFiles.length) {
    throw new Error("requiredFiles must be a non-empty unique path list");
  }
  if (new Set(protectedFiles).size !== protectedFiles.length) {
    throw new Error("protectedFiles must be a unique path list");
  }
  if (requiredFiles.some((path) => protectedFiles.includes(path))) {
    throw new Error("requiredFiles and protectedFiles must be disjoint");
  }
  const timeoutSeconds = request.timeoutSeconds ?? DEFAULT_TIMEOUT_SECONDS;
  const maxTotalTokens = request.maxTotalTokens ?? DEFAULT_MAX_TOTAL_TOKENS;
  if (!Number.isInteger(timeoutSeconds) || timeoutSeconds < 10 || timeoutSeconds > 900) {
    throw new Error("timeoutSeconds must be an integer from 10 to 900");
  }
  if (!Number.isInteger(maxTotalTokens) || maxTotalTokens < 1_000 || maxTotalTokens > 500_000) {
    throw new Error("maxTotalTokens must be an integer from 1000 to 500000");
  }
  return {
    protocolVersion: PROTOCOL_VERSION,
    workspace: request.workspace,
    prompt: request.prompt,
    requiredFiles,
    protectedFiles,
    model: request.model?.trim() || "composer-2.5",
    timeoutSeconds,
    maxTotalTokens,
  };
}

async function walk(root: string, current: string, snapshot: FileSnapshot): Promise<void> {
  for (const entry of await readdir(current, { withFileTypes: true })) {
    const absolute = resolve(current, entry.name);
    const projectPath = relative(root, absolute).split(sep).join("/");
    if (!projectPath || IGNORED_ROOTS.has(projectPath.split("/")[0]!)) continue;
    const status = await lstat(absolute);
    if (status.isSymbolicLink()) {
      snapshot.set(projectPath, "symlink");
    } else if (status.isDirectory()) {
      await walk(root, absolute, snapshot);
    } else if (status.isFile()) {
      const digest = createHash("sha256").update(await readFile(absolute)).digest("hex");
      snapshot.set(projectPath, digest);
    }
  }
}

export async function snapshotWorkspace(root: string): Promise<FileSnapshot> {
  const snapshot: FileSnapshot = new Map();
  await walk(root, root, snapshot);
  return snapshot;
}

export function changedPaths(before: FileSnapshot, after: FileSnapshot): string[] {
  return [...new Set([...before.keys(), ...after.keys()])]
    .filter((path) => before.get(path) !== after.get(path))
    .sort();
}

function addUsage(target: TokenUsage, usage: TokenUsage | undefined): TokenUsage {
  if (!usage) return target;
  return {
    inputTokens: (target.inputTokens ?? 0) + (usage.inputTokens ?? 0),
    outputTokens: (target.outputTokens ?? 0) + (usage.outputTokens ?? 0),
    totalTokens: (target.totalTokens ?? 0) + (usage.totalTokens ?? 0),
    cacheReadTokens: (target.cacheReadTokens ?? 0) + (usage.cacheReadTokens ?? 0),
    cacheWriteTokens: (target.cacheWriteTokens ?? 0) + (usage.cacheWriteTokens ?? 0),
  };
}

export async function runShadow(rawRequest: unknown) {
  const request = validateRequest(rawRequest);
  const workspace = await realpath(request.workspace);
  if (workspace === "/" || workspace === resolve(process.env.HOME || "/nonexistent")) {
    throw new Error("workspace must be a disposable project directory, not a broad root");
  }
  let marker = "";
  try {
    marker = await readFile(resolve(workspace, ".metis-cursor-shadow"), "utf8");
  } catch {
    // Hand-written paths are too easy to point at a real checkout. The host
    // creates this marker only inside the disposable benchmark fixture.
  }
  if (marker !== "metis-cursor-shadow-v1\n") {
    throw new Error("workspace is missing the Metis disposable-shadow marker");
  }
  const before = await snapshotWorkspace(workspace);
  const apiKey = process.env.CURSOR_API_KEY;
  if (!apiKey) throw new Error("CURSOR_API_KEY is required for a live Cursor shadow run");
  const { Agent } = await import("@cursor/sdk");

  await using agent = await Agent.create({
    apiKey,
    model: { id: request.model! },
    tools: ["read", "grep", "glob", "ls", "edit", "write", "shell"],
    local: {
      cwd: workspace,
      sandboxOptions: { enabled: true },
      autoReview: true,
      enableAgentRetries: false,
    },
  });

  const startedAt = Date.now();
  const run = await agent.send(request.prompt);
  let usage: TokenUsage = {};
  let cancelledFor = "";
  const timer = setTimeout(() => {
    cancelledFor = "timeout";
    void run.cancel();
  }, request.timeoutSeconds! * 1_000);
  const eventCounts: Record<string, number> = {};
  try {
    for await (const event of run.stream()) {
      eventCounts[event.type] = (eventCounts[event.type] ?? 0) + 1;
      if (event.type === "usage") {
        // Cursor documents usage stream events as per-turn, so summing them is
        // the in-flight cumulative counter used for the cancellation ceiling.
        usage = addUsage(usage, event.usage as TokenUsage);
        if ((usage.totalTokens ?? 0) > request.maxTotalTokens!) {
          cancelledFor = "token_budget";
          await run.cancel();
        }
      }
    }
  } finally {
    clearTimeout(timer);
  }
  const result = await run.wait();
  // The terminal result is already cumulative across the run and therefore
  // replaces (rather than adds to) the streamed running total when available.
  usage = (result.usage as TokenUsage | undefined) ?? usage;
  const after = await snapshotWorkspace(workspace);
  const changed = changedPaths(before, after);
  const required = new Set(request.requiredFiles);
  const protectedSet = new Set(request.protectedFiles);
  const missingRequired = request.requiredFiles.filter((path) => !changed.includes(path));
  const outOfScope = changed.filter((path) => !required.has(path));
  const protectedTouched = changed.filter((path) => protectedSet.has(path));
  const exactRequestedScope = !missingRequired.length && !outOfScope.length;

  let billed: Awaited<ReturnType<typeof agent.getUsage>> | undefined;
  try {
    billed = await agent.getUsage();
  } catch {
    // Cost settlement is useful evidence but must never turn a completed
    // benchmark into a failure. The caller records it as unavailable.
  }
  return {
    ...descriptor(),
    live: true,
    model: request.model,
    status: result.status,
    cancelledFor,
    durationSeconds: Math.round((Date.now() - startedAt) / 100) / 10,
    usage,
    billedUsage: billed ?? null,
    eventCounts,
    changedPaths: changed,
    missingRequiredFiles: missingRequired,
    outOfScopePaths: outOfScope,
    protectedPathsTouched: protectedTouched,
    exactRequestedScope,
    eligibleForMetisVerification: Boolean(
      result.status === "finished" &&
      !cancelledFor &&
      exactRequestedScope &&
      !protectedTouched.length
    ),
    approvalOffered: false,
    resultPreview: String(result.result ?? "").slice(0, 2_000),
  };
}

async function readStdin(): Promise<string> {
  const chunks: Buffer[] = [];
  for await (const chunk of process.stdin) chunks.push(Buffer.from(chunk));
  return Buffer.concat(chunks).toString("utf8");
}

async function main(): Promise<void> {
  if (process.argv.includes("--describe")) {
    process.stdout.write(`${JSON.stringify(descriptor(), null, 2)}\n`);
    return;
  }
  if (!process.argv.includes("--live")) {
    throw new Error("refusing to spend a Cursor request without --live");
  }
  const body = await readStdin();
  const report = await runShadow(JSON.parse(body));
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
}

const invoked = process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href;
if (invoked) {
  main().catch((error: unknown) => {
    const message = error instanceof Error ? error.message : String(error);
    process.stderr.write(`cursor shadow benchmark failed: ${message}\n`);
    process.exitCode = 1;
  });
}
