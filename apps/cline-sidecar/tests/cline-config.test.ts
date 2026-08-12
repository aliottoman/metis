import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import {
  buildClineStartInput,
  buildMetisSystemPrompt,
  compactRecoveryMessages,
  controlledStopReason,
} from "../src/cline-runtime.js";
import { CLINE_SDK_VERSION } from "../src/protocol.js";

test("real ClineCore input wires hard limits and disables expansive capabilities", () => {
  const input = buildClineStartInput({
    sessionId: "session-1",
    prompt: "Implement one vertical slice",
    workspaceRoot: "/tmp/metis-owned-mirror",
    provider: {
      providerId: "cline-pass",
      modelId: "cline-pass/deepseek-v4-pro",
      apiKey: "secret-canary-value",
    },
    systemPrompt: "Metis policy",
    limits: { maxIterations: 17, timeoutMs: 120_000, maxTokensPerTurn: 8192 },
  });

  assert.equal(input.config.maxIterations, 17);
  assert.equal(input.sessionMetadata?.maxIterations, 17);
  assert.equal(input.config.maxTokensPerTurn, 8192);
  assert.equal(input.config.execution?.apiTimeoutMs, 120_000);
  assert.equal(input.config.execution?.maxConsecutiveMistakes, 3);
  assert.deepEqual(input.config.execution?.loopDetection, { softThreshold: 2, hardThreshold: 3 });
  assert.equal(input.config.execution?.reminderAfterIterations, 10);
  assert.match(input.config.execution?.reminderText ?? "", /every remaining planned file/);
  assert.match(input.config.execution?.reminderText ?? "", /independent checks/);
  assert.match(input.config.execution?.reminderText ?? "", /5500 characters/);
  assert.equal(input.config.enableSpawnAgent, false);
  assert.equal(input.config.enableAgentTeams, false);
  assert.equal(input.config.disableMcpSettingsTools, true);
  assert.deepEqual(input.config.skills, []);
  assert.deepEqual(input.localRuntime?.configExtensions, []);
  // The SDK's shell stays disabled and unadvertised. Verification is the
  // registered `run_check` tool, with its own closed schema.
  assert.equal(input.toolPolicies?.run_commands?.enabled, false);
  assert.equal(input.toolPolicies?.run_check?.enabled, true);
  assert.equal(input.toolPolicies?.fetch_web_content?.enabled, false);
  assert.equal(input.toolPolicies?.apply_patch?.enabled, false);
  assert.equal(input.toolPolicies?.read_files?.autoApprove, false);
});

test("normalizes only the pinned max-iteration error contract", () => {
  const exact = {
    finishReason: "error",
    iterations: 24,
    text: "Agent runtime exceeded maxIterations (24)",
  };
  assert.equal(controlledStopReason(exact, 24), "max_iterations");
  assert.equal(
    controlledStopReason({ ...exact, text: "An arbitrary runtime error" }, 24),
    undefined,
  );
  assert.equal(controlledStopReason({ ...exact, iterations: 23 }, 24), undefined);
  assert.equal(
    controlledStopReason({ ...exact, finishReason: "mistake_limit" }, 24),
    undefined,
  );
  assert.equal(
    controlledStopReason({ ...exact, finishReason: "max_iterations" }, 24),
    "max_iterations",
  );
});

test("Metis retains Cline's canonical workspace guidance and appends stricter policy", () => {
  const workspaceRoot = "/private/tmp/metis-owned-mirror/project";
  const prompt = buildMetisSystemPrompt(
    workspaceRoot,
    "openai-compatible",
    "Modify exactly src/app.ts.",
  );

  assert.match(prompt, /Working Directory: \/private\/tmp\/metis-owned-mirror\/project/);
  assert.match(prompt, /Always use absolute paths when referring to files/);
  assert.match(prompt, /Metis task-specific instructions/);
  assert.match(prompt, /Modify exactly src\/app\.ts\./);
  assert.match(prompt, /Metis execution boundary \(highest priority\)/);
  assert.match(prompt, /Use only read_files, search_codebase, and editor/);
  assert.match(prompt, /Batch all independent initial reads/);
  assert.match(prompt, /hard-rejects any individual old_text or new_text longer than 5500 characters/);
  assert.match(prompt, /Never retry a rejected oversized payload unchanged/);
  assert.match(prompt, /Finish every explicitly planned file/);
  assert.match(prompt, /Never use shell commands, network access, MCP, plugins, skills, questions, subagents, or teams/);
  assert.ok(prompt.lastIndexOf("Metis execution boundary") > prompt.indexOf("Modify exactly"));
});

test("the system prompt states the exact per-slice write allowlist", () => {
  const workspaceRoot = "/private/tmp/metis-owned-mirror/project";
  const scoped = buildMetisSystemPrompt(
    workspaceRoot,
    "openai-compatible",
    undefined,
    ["app/planned.py", "app/planned_test.py"],
  );
  assert.match(scoped, /may create or edit only these exact files: app\/planned\.py, app\/planned_test\.py/);
  assert.match(scoped, /an earlier slice already wrote, is read-only for this round/);

  const readOnly = buildMetisSystemPrompt(workspaceRoot, "openai-compatible", undefined, []);
  assert.match(readOnly, /This round is read-only: the editor tool is not available/);

  const unrestricted = buildMetisSystemPrompt(workspaceRoot, "openai-compatible");
  assert.equal(/may create or edit only/.test(unrestricted), false);
  assert.equal(/This round is read-only/.test(unrestricted), false);
});

test("a real ClineCore session denies an editor write outside the wired allowlist", async () => {
  const workspaceRoot = await mkdtemp(join(tmpdir(), "metis-slice-wiring-"));
  try {
    await mkdir(join(workspaceRoot, "app"), { recursive: true });
    const input = buildClineStartInput({
      sessionId: "session-slice",
      prompt: "Implement one vertical slice",
      workspaceRoot,
      provider: { providerId: "cline-pass", modelId: "cline-pass/deepseek-v4-pro" },
      systemPrompt: "Metis policy",
      allowedPaths: ["app/planned.py"],
    });
    const denied = await input.localRuntime?.hooks?.beforeTool?.({
      tool: { name: "editor" },
      input: { path: "app/other.py" },
    } as never);
    assert.equal(denied?.skip, true);
    assert.equal(denied?.stop, true);

    const allowed = await input.localRuntime?.hooks?.beforeTool?.({
      tool: { name: "editor" },
      input: { path: "app/planned.py" },
    } as never);
    assert.equal(allowed?.skip, undefined);
  } finally {
    await rm(workspaceRoot, { recursive: true, force: true });
  }
});

test("model restart compacts verbose tool history while preserving cumulative usage", () => {
  const messages: NonNullable<Parameters<typeof compactRecoveryMessages>[0]> = [
    {
      role: "user",
      content: [{ type: "text", text: "Original exact project task" }],
    },
  ];
  for (let index = 1; index <= 6; index += 1) {
    messages.push({
      role: "assistant",
      content: [
        {
          type: "tool_use",
          id: `call-${index}`,
          name: "editor",
          input: { path: `/tmp/file-${index}`, new_text: "x".repeat(12_000) },
        },
      ],
      metrics: {
        inputTokens: index * 100,
        outputTokens: index * 10,
        cacheReadTokens: index,
        cacheWriteTokens: index * 2,
        cost: index / 100,
      },
      modelInfo: { id: "old-model", provider: "test" },
    });
    messages.push({
      role: "user",
      content: [
        {
          type: "tool_result",
          tool_use_id: `call-${index}`,
          name: "editor",
          content: "rendered diff ".repeat(1_000),
        },
      ],
    });
  }

  const compacted = compactRecoveryMessages(messages);
  assert.equal(compacted.length, 2);
  assert.deepEqual(compacted[0], messages[0]);
  assert.match(JSON.stringify(compacted[1]), /current workspace files are authoritative/);
  assert.equal(JSON.stringify(compacted).includes("rendered diff"), false);
  assert.equal(JSON.stringify(compacted).includes('"tool_use"'), false);
  assert.deepEqual((compacted[1] as { metrics?: unknown }).metrics, {
    inputTokens: 2_100,
    outputTokens: 210,
    cacheReadTokens: 21,
    cacheWriteTokens: 42,
    cost: 0.21000000000000002,
  });
});

test("model restart keeps full history when usage cannot remain exact", () => {
  const messages: NonNullable<Parameters<typeof compactRecoveryMessages>[0]> = [
    { role: "user", content: "original" },
    ...Array.from({ length: 12 }, (_unused, index) => ({
      role: index % 2 === 0 ? ("assistant" as const) : ("user" as const),
      content: `message ${index}`,
    })),
  ];
  assert.equal(compactRecoveryMessages(messages), messages);
});

test("the reported SDK version is pinned to the installed package contract", async () => {
  const packageJson = JSON.parse(
    await readFile(new URL("../../package.json", import.meta.url), "utf8"),
  ) as { dependencies?: Record<string, string> };
  assert.equal(packageJson.dependencies?.["@cline/sdk"], CLINE_SDK_VERSION);
});
