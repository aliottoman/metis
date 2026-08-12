import assert from "node:assert/strict";
import { test } from "node:test";

import { ALL_DEFAULT_TOOL_NAMES } from "@cline/sdk";

import { HOST_CHECKS, type HostCheckResult } from "../src/protocol.js";
import {
  RUN_CHECK_INPUT_SCHEMA,
  RUN_CHECK_TOOL_NAME,
  createRunCheckTool,
  malformedCheckOutput,
  renderRunCheckOutput,
} from "../src/run-check-tool.js";
import {
  ALLOWED_TOOLS,
  FAIL_CLOSED_TOOL_POLICIES,
  createFailClosedHooks,
  decideToolApproval,
  type SafeToolObservation,
} from "../src/security.js";

// The exact command-shaped payload GLM sent, four times, in the live run that
// died on it. Preserved verbatim from the captured session journal.
const GLM_COMMAND_SHAPED_PAYLOAD = {
  command: "python -m pytest tests/ -v",
  cwd: "/private/var/folders/metis-code-workspace/project",
  timeout: 120,
};

test("run_commands is disabled, hidden, and no longer the check wire", () => {
  assert.equal(ALLOWED_TOOLS.has("run_commands" as never), false);
  assert.equal(FAIL_CLOSED_TOOL_POLICIES.run_commands?.enabled, false);
  assert.equal(FAIL_CLOSED_TOOL_POLICIES.run_commands?.autoApprove, false);
  // Approval refuses it too, so policy and capability agree.
  assert.deepEqual(
    decideToolApproval({
      sessionId: "s",
      agentId: "a",
      conversationId: "c",
      iteration: 1,
      toolCallId: "t",
      toolName: "run_commands",
      input: GLM_COMMAND_SHAPED_PAYLOAD,
      policy: {},
    }),
    { approved: false, reason: "Tool run_commands is disabled by Metis" },
  );
  // Still every SDK built-in has an explicit policy.
  assert.deepEqual(
    (ALL_DEFAULT_TOOL_NAMES as readonly string[]).filter(
      (name: string) => FAIL_CLOSED_TOOL_POLICIES[name] === undefined,
    ),
    [],
  );
});

test("run_check is a registered tool with exactly the closed check enum", () => {
  const tool = createRunCheckTool({ runCheck: async () => clean("pytest") });
  assert.equal(tool.name, RUN_CHECK_TOOL_NAME);
  assert.equal(tool.name, "run_check");
  // run_check is NOT a built-in wearing a new label.
  assert.equal(
    (ALL_DEFAULT_TOOL_NAMES as readonly string[]).includes("run_check"),
    false,
  );

  // The schema the model is shown, exactly.
  const properties = RUN_CHECK_INPUT_SCHEMA.properties as Record<
    string,
    { type?: string; enum?: string[]; description?: string }
  >;
  assert.equal(RUN_CHECK_INPUT_SCHEMA.type, "object");
  // No command, argv, cwd, environment, shell or free-text field exists.
  assert.deepEqual(Object.keys(properties), ["check"]);
  assert.equal(properties.check?.type, "string");
  assert.deepEqual(properties.check?.enum, [...HOST_CHECKS]);
  assert.deepEqual(properties.check?.enum, [
    "imports",
    "pytest",
    "ruff",
    "acceptance",
    "full",
  ]);
  assert.deepEqual(RUN_CHECK_INPUT_SCHEMA.required, ["check"]);
  assert.equal(RUN_CHECK_INPUT_SCHEMA.additionalProperties, false);
  assert.deepEqual(tool.inputSchema, RUN_CHECK_INPUT_SCHEMA);
});

test("GLM's captured command payload cannot be a valid run_check call", async () => {
  // 1. It violates the advertised schema: every key is unknown, and the one
  //    required key is missing.
  const properties = RUN_CHECK_INPUT_SCHEMA.properties as Record<string, unknown>;
  for (const key of Object.keys(GLM_COMMAND_SHAPED_PAYLOAD)) {
    assert.equal(
      Object.hasOwn(properties, key),
      false,
      `${key} must not be an accepted field`,
    );
  }
  assert.equal(
    Object.hasOwn(GLM_COMMAND_SHAPED_PAYLOAD, "check"),
    false,
    "the payload has no check field, so it cannot satisfy `required`",
  );

  // 2. And if a provider let it through anyway, execute refuses it without
  //    running anything.
  const asked: string[] = [];
  const observations: SafeToolObservation[] = [];
  const tool = createRunCheckTool({
    runCheck: async (check) => {
      asked.push(check);
      return clean(check);
    },
    observer: (event) => observations.push(event),
  });
  const output = (await tool.execute(
    GLM_COMMAND_SHAPED_PAYLOAD as never,
    {} as never,
  )) as { status: string; detail: string };
  assert.equal(output.status, "unavailable");
  assert.match(output.detail, /Metis runs checks, you do not/);
  assert.match(output.detail, /imports, pytest, ruff, acceptance, full/);
  // Nothing was asked of the host, and nothing was executed here.
  assert.deepEqual(asked, []);
  assert.deepEqual(observations, [
    { tool: "run_check", status: "denied", reasonCode: "command_not_permitted" },
  ]);
});

test("an unknown check name is refused without reaching the host", async () => {
  const asked: string[] = [];
  const tool = createRunCheckTool({
    runCheck: async (check) => {
      asked.push(check);
      return clean(check);
    },
  });
  for (const bad of ["mypy", "PYTEST", "pytest;rm -rf /", "", "full full"]) {
    const output = (await tool.execute({ check: bad } as never, {} as never)) as {
      status: string;
    };
    assert.equal(output.status, "unavailable", `${bad} was not refused`);
  }
  assert.deepEqual(asked, []);
});

test("three valid checks in sequence all succeed and none is a mistake", async () => {
  // The failure mode this replaces: every check was a refusal, and three
  // consecutive refusals tripped the SDK's maxConsecutiveMistakes.
  const asked: string[] = [];
  const observations: SafeToolObservation[] = [];
  const tool = createRunCheckTool({
    runCheck: async (check) => {
      asked.push(check);
      return clean(check);
    },
    observer: (event) => observations.push(event),
  });
  for (const check of ["imports", "pytest", "ruff"] as const) {
    const output = (await tool.execute({ check }, {} as never)) as {
      status: string;
      check: string;
    };
    assert.equal(output.status, "clean");
    assert.equal(output.check, check);
  }
  assert.deepEqual(asked, ["imports", "pytest", "ruff"]);
  // Nothing was ever denied, so nothing counted against the session.
  assert.deepEqual(
    observations.filter((event) => event.status === "denied"),
    [],
  );
  assert.equal(
    observations.filter((event) => event.status === "started").length,
    3,
  );
});

test("blocked findings come back attributed and bounded", async () => {
  const tool = createRunCheckTool({
    runCheck: async () => ({
      check: "imports",
      ok: false,
      errors: 1,
      warnings: 0,
      findings: [
        {
          path: "app/main.py",
          severity: "error",
          detail: "No module named 'python-multipart'",
        },
      ],
      durationMs: 12,
      truncated: false,
    }),
  });
  const output = (await tool.execute({ check: "imports" }, {} as never)) as {
    status: string;
    errors: number;
    findings: Array<{ path: string; detail: string }>;
    detail: string;
  };
  assert.equal(output.status, "blocked");
  assert.equal(output.errors, 1);
  assert.deepEqual(output.findings, [
    {
      path: "app/main.py",
      severity: "error",
      detail: "No module named 'python-multipart'",
    },
  ]);
  assert.match(output.detail, /Fix the root causes/);
});

test("an unavailable check is reported as a fact, not an error to fight", () => {
  const output = renderRunCheckOutput({
    check: "pytest",
    ok: false,
    unavailable: "the workspace mirror is not available",
    errors: 0,
    warnings: 0,
    findings: [],
  });
  assert.equal(output.status, "unavailable");
  assert.match(output.detail, /Continue with the work you can do without it/);
  assert.deepEqual(output.findings, []);
});

test("the diagnostic detail stays bounded", () => {
  const output = malformedCheckOutput("x".repeat(50_000));
  assert.ok(output.detail.length <= 2_001, `detail was ${output.detail.length}`);
});

test("the hook lets run_check through and never counts it as inspection", async () => {
  const observations: SafeToolObservation[] = [];
  const hooks = createFailClosedHooks({
    workspaceRoot: process.cwd(),
    broadScope: true,
    runCheck: async () => clean("full"),
    observer: (event) => observations.push(event),
  });
  // Three checks in a row: allowed every time, no stop, no denial.
  for (let index = 0; index < 3; index += 1) {
    const decision = await hooks.beforeTool?.({
      tool: { name: "run_check" },
      input: { check: "full" },
    } as never);
    assert.equal(decision?.skip, undefined);
    assert.equal(decision?.stop, undefined);
    assert.equal(decision?.policy?.enabled, true);
  }
  assert.deepEqual(observations, []);
});

test("run_check is refused when the session has no host to ask", async () => {
  const hooks = createFailClosedHooks({ workspaceRoot: process.cwd() });
  const decision = await hooks.beforeTool?.({
    tool: { name: "run_check" },
    input: { check: "full" },
  } as never);
  assert.equal(decision?.skip, true);
  assert.equal(decision?.policy?.enabled, false);
});

function clean(check: string): HostCheckResult {
  return {
    check: check as never,
    ok: true,
    errors: 0,
    warnings: 0,
    findings: [],
    durationMs: 5,
    truncated: false,
  };
}
