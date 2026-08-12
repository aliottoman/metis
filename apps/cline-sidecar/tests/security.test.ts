import assert from "node:assert/strict";
import {
  mkdtemp,
  mkdir,
  realpath,
  rm,
  symlink,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { ALL_DEFAULT_TOOL_NAMES } from "@cline/sdk";

import {
  FAIL_CLOSED_TOOL_POLICIES,
  compactSuccessfulEditorResult,
  createFailClosedHooks,
  decideToolApproval,
  evaluateToolCall,
  type SafeToolObservation,
} from "../src/security.js";

test("all SDK built-ins have an explicit policy", () => {
  assert.deepEqual(
    (ALL_DEFAULT_TOOL_NAMES as readonly string[]).filter(
      (name: string) => FAIL_CLOSED_TOOL_POLICIES[name] === undefined,
    ),
    [],
  );
  assert.equal(FAIL_CLOSED_TOOL_POLICIES.read_files?.autoApprove, false);
  // The SDK's shell is disabled and hidden. Verification reaches Metis through
  // the registered `run_check` tool instead; see run-check-tool.test.ts.
  assert.equal(FAIL_CLOSED_TOOL_POLICIES.run_commands?.enabled, false);
  assert.equal(FAIL_CLOSED_TOOL_POLICIES.run_commands?.autoApprove, false);
  assert.equal(FAIL_CLOSED_TOOL_POLICIES.run_check?.enabled, true);
  assert.equal(FAIL_CLOSED_TOOL_POLICIES.run_check?.autoApprove, false);
  assert.equal(FAIL_CLOSED_TOOL_POLICIES.fetch_web_content?.enabled, false);
  assert.equal(FAIL_CLOSED_TOOL_POLICIES.skills?.enabled, false);
});

test("an unknown future tool is denied by both approval and pre-tool guards", async () => {
  const root = await mkdtemp(join(tmpdir(), "metis-tool-guard-"));
  try {
    assert.deepEqual(
      decideToolApproval({
        sessionId: "s",
        agentId: "a",
        conversationId: "c",
        iteration: 1,
        toolCallId: "t",
        toolName: "future_unlisted_tool",
        input: {},
        policy: {},
      }),
      {
        approved: false,
        reason: "Tool future_unlisted_tool is disabled by Metis",
      },
    );
    assert.deepEqual(await evaluateToolCall(root, "future_unlisted_tool", {}), {
      allowed: false,
      reason: "Tool future_unlisted_tool is not in the Metis allowlist",
      reasonCode: "tool_not_allowed",
    });
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("file tools are workspace-scoped and reject protected or symlinked paths", async () => {
  const root = await mkdtemp(join(tmpdir(), "metis-tool-root-"));
  const outside = await mkdtemp(join(tmpdir(), "metis-tool-outside-"));
  try {
    const canonicalRoot = await realpath(root);
    await mkdir(join(root, "src"));
    await writeFile(join(root, "src", "app.ts"), "export {};\n");
    await symlink(outside, join(root, "escape"));
    assert.deepEqual(
      await evaluateToolCall(root, "read_files", {
        files: [{ path: "src/app.ts" }],
      }),
      { allowed: true },
    );
    assert.deepEqual(
      await evaluateToolCall(root, "read_files", {
        files: [
          {
            path: join(canonicalRoot, "src", "app.ts"),
            start_line: 1,
            end_line: 10,
          },
        ],
      }),
      { allowed: true },
    );
    assert.equal(
      (
        await evaluateToolCall(root, "read_files", {
          files: [{ path: "/src/app.ts" }],
        })
      ).reasonCode,
      "outside_workspace",
    );
    assert.equal(
      (await evaluateToolCall(root, "editor", { path: "../outside" }))
        .reasonCode,
      "outside_workspace",
    );
    assert.equal(
      (
        await evaluateToolCall(root, "search_codebase", {
          queries: ["token"],
          path: outside,
        })
      ).reasonCode,
      "invalid_search_scope",
    );
    assert.equal(
      (await evaluateToolCall(root, "read_files", { files: [] })).reasonCode,
      "missing_path",
    );
    assert.equal(
      (await evaluateToolCall(root, "read_files", { path: "src/app.ts\0.env" }))
        .reasonCode,
      "invalid_path",
    );
    assert.equal(
      (await evaluateToolCall(root, "read_files", { path: ".env" })).reasonCode,
      "protected_path",
    );
    assert.equal(
      (await evaluateToolCall(root, "editor", { path: "escape/new.ts" }))
        .reasonCode,
      "symlink_path",
    );
    assert.equal(
      (
        await evaluateToolCall(root, "apply_patch", {
          input: "*** Begin Patch\n*** Update File: src/app.ts\n*** End Patch",
        })
      ).reasonCode,
      "tool_not_allowed",
    );
  } finally {
    await rm(root, { recursive: true, force: true });
    await rm(outside, { recursive: true, force: true });
  }
});

test("an outside-workspace denial names the exact root and a corrected example path", async () => {
  // A live measured failure: a denial that only says "escapes the
  // workspace" left the model alternating denied reads and unproductive
  // searches for an entire slice budget, never discovering the one thing
  // that would have fixed it -- the exact root to combine with its plan.
  const root = await mkdtemp(join(tmpdir(), "metis-tool-corrective-"));
  try {
    const canonicalRoot = await realpath(root);
    const decision = await evaluateToolCall(root, "read_files", {
      files: [{ path: "/app/config.py" }],
    });
    assert.equal(decision.reasonCode, "outside_workspace");
    assert.match(
      decision.reason ?? "",
      new RegExp(canonicalRoot.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")),
    );
    assert.match(decision.reason ?? "", /absolute path inside it/);
    assert.match(decision.reason ?? "", /\/app\/config\.py/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("editor writes are confined to the current slice's exact allowlist; reads are not", async () => {
  const root = await mkdtemp(join(tmpdir(), "metis-slice-scope-"));
  try {
    await mkdir(join(root, "app"), { recursive: true });
    await writeFile(join(root, "app", "verified.py"), "VALUE = 1\n");
    await writeFile(join(root, "app", "planned.py"), "VALUE = 2\n");
    const allowedPaths = new Set(["app/planned.py"]);

    // In-scope write: allowed.
    assert.deepEqual(
      await evaluateToolCall(
        root,
        "editor",
        { path: "app/planned.py" },
        allowedPaths,
      ),
      { allowed: true },
    );

    // Out-of-scope write, including to a file staged by an earlier, already
    // verified slice: denied and terminal, not merely skippable.
    const outOfScope = await evaluateToolCall(
      root,
      "editor",
      { path: "app/verified.py" },
      allowedPaths,
    );
    assert.equal(outOfScope.allowed, false);
    assert.equal(outOfScope.reasonCode, "outside_slice");

    // Reading an out-of-scope file (e.g. for context on an earlier slice)
    // still succeeds -- only the editor tool is scope-restricted.
    assert.deepEqual(
      await evaluateToolCall(
        root,
        "read_files",
        { files: [{ path: "app/verified.py" }] },
        allowedPaths,
      ),
      { allowed: true },
    );

    // A read-only round (present-and-empty allowlist) refuses every write.
    const readOnlyRound = await evaluateToolCall(
      root,
      "editor",
      { path: "app/planned.py" },
      new Set(),
    );
    assert.equal(readOnlyRound.allowed, false);
    assert.equal(readOnlyRound.reasonCode, "outside_slice");

    // No allowlist at all (undefined) applies no slice restriction -- only
    // the ordinary workspace-containment check still runs.
    assert.deepEqual(
      await evaluateToolCall(
        root,
        "editor",
        { path: "app/verified.py" },
        undefined,
      ),
      { allowed: true },
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("an out-of-slice editor denial stops the turn rather than inviting a retry", async () => {
  const root = await mkdtemp(join(tmpdir(), "metis-slice-scope-hooks-"));
  const observations: SafeToolObservation[] = [];
  try {
    await mkdir(join(root, "app"), { recursive: true });
    await writeFile(join(root, "app", "verified.py"), "VALUE = 1\n");
    const hooks = createFailClosedHooks({
      workspaceRoot: root,
      allowedPaths: new Set(["app/planned.py"]),
      observer: (event) => observations.push(event),
    });
    const decision = await hooks.beforeTool?.({
      tool: { name: "editor" },
      input: { path: "app/verified.py" },
    } as never);
    assert.equal(decision?.skip, true);
    assert.equal(decision?.stop, true);
    assert.deepEqual(observations, [
      { tool: "editor", status: "denied", reasonCode: "outside_slice" },
    ]);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("repeated inspection without an editor call earns one correction, then a bounded stop", async () => {
  // A live measured failure: read_files denials interleaved with individually
  // "successful" search_codebase calls burned an entire 16-iteration slice
  // budget without a single editor call. The SDK's own loop detector only
  // catches identical repeated calls; every one of those searches had
  // different arguments, so nothing in the SDK would have caught this.
  const root = await mkdtemp(join(tmpdir(), "metis-stuck-inspection-"));
  const observations: SafeToolObservation[] = [];
  try {
    const hooks = createFailClosedHooks({
      workspaceRoot: root,
      observer: (event) => observations.push(event),
    });

    let lastAfter: Awaited<ReturnType<NonNullable<typeof hooks.afterTool>>>;
    for (let index = 0; index < 4; index += 1) {
      const before = await hooks.beforeTool?.({
        tool: { name: "search_codebase" },
        input: { queries: [`query-${index}`] },
      } as never);
      assert.equal(before?.skip, undefined);
      lastAfter = await hooks.afterTool?.({
        tool: { name: "search_codebase" },
        result: { output: { success: true, results: [] } },
      } as never);
    }

    // The 4th inspection call (the threshold) is still allowed, but its own
    // result now carries the corrective nudge inline, in the same turn.
    const noticeOutput = (
      lastAfter as
        { result?: { output?: { metis_notice?: unknown } } } | undefined
    )?.result?.output;
    assert.match(
      String(noticeOutput?.metis_notice ?? ""),
      /STOP INSPECTING AND ACT/,
    );

    // A 5th inspection call arrives with still no editor call in between:
    // stop the turn honestly rather than spend the rest of the slice budget
    // repeating inspection that already earned one correction.
    const fifth = await hooks.beforeTool?.({
      tool: { name: "search_codebase" },
      input: { queries: ["query-4"] },
    } as never);
    assert.equal(fifth?.skip, true);
    assert.equal(fifth?.stop, true);
    assert.equal(observations.at(-1)?.reasonCode, "stuck_inspection_loop");

    // Fewer than four allowed searches never reaches the stop -- the guard
    // fires on the count, not on the mere presence of inspection tools.
    assert.equal(
      observations.filter((event) => event.status === "denied").length,
      1,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("denied tool observations expose only a fixed non-sensitive reason code", async () => {
  const root = await mkdtemp(join(tmpdir(), "metis-tool-observer-"));
  const observations: SafeToolObservation[] = [];
  try {
    const hooks = createFailClosedHooks({
      workspaceRoot: root,
      observer: (event) => observations.push(event),
    });
    const recoverable = await hooks.beforeTool?.({
      tool: { name: "read_files" },
      input: { files: [{ path: "/private/secret.txt" }] },
    } as never);

    const terminal = await hooks.beforeTool?.({
      tool: { name: "read_files" },
      input: { files: [{ path: ".env" }] },
    } as never);

    assert.deepEqual(observations, [
      { tool: "read_files", status: "denied", reasonCode: "outside_workspace" },
      { tool: "read_files", status: "denied", reasonCode: "protected_path" },
    ]);
    assert.deepEqual(Object.keys(observations[0] ?? {}).sort(), [
      "reasonCode",
      "status",
      "tool",
    ]);
    assert.equal(
      JSON.stringify(observations).includes("/private/secret.txt"),
      false,
    );
    assert.equal(recoverable?.skip, true);
    assert.equal(recoverable?.stop, undefined);
    assert.equal(terminal?.skip, true);
    assert.equal(terminal?.stop, true);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("successful editor output is compacted but failures and read results remain exact", () => {
  const renderedDiff = "unchanged rendered diff line\n".repeat(1_000);
  const success = {
    output: {
      query: "edit:/tmp/workspace/app.ts",
      result: renderedDiff,
      success: true,
    },
  };
  const compacted = compactSuccessfulEditorResult("editor", success);
  assert.notEqual(compacted, success);
  assert.equal(JSON.stringify(compacted).includes(renderedDiff), false);
  assert.match(
    JSON.stringify(compacted),
    /Metis will verify the final workspace/,
  );
  assert.ok(JSON.stringify(compacted).length < 300);

  const failed = {
    output: JSON.stringify({
      query: "edit:/tmp/workspace/app.ts",
      error: "Editor input too large: 12000 > 6000",
      success: false,
    }),
  };
  assert.equal(compactSuccessfulEditorResult("editor", failed), failed);
  assert.equal(compactSuccessfulEditorResult("read_files", success), success);
});

test("editor-level failures are reported as failed even when the SDK omits isError", async () => {
  const root = await mkdtemp(join(tmpdir(), "metis-editor-observer-"));
  const observations: SafeToolObservation[] = [];
  try {
    const hooks = createFailClosedHooks({
      workspaceRoot: root,
      observer: (event) => observations.push(event),
    });
    await hooks.afterTool?.({
      tool: { name: "editor" },
      result: {
        output: {
          success: false,
          error: "Editor input too large",
        },
      },
    } as never);
    assert.deepEqual(observations, [{ tool: "editor", status: "failed" }]);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

// ── Protected paths: the direct path's write boundary ──────────────────────
//
// A broad-scope session cannot be expressed as an 8-entry allowlist, so it is
// bounded by a deny-list instead. This is the sidecar half; the safe importer
// refuses the same paths again, independently, against the bytes that
// actually arrive.

test("a broad-scope session may read a protected file but never write it", async () => {
  const root = await mkdtemp(join(tmpdir(), "metis-protected-scope-"));
  try {
    await mkdir(join(root, "assets"), { recursive: true });
    await writeFile(join(root, "extractor.py"), "OCI = 1\n");
    await writeFile(join(root, "excel_writer.py"), "COL = 'A'\n");
    await writeFile(join(root, "assets", "DHL_Template.xlsx"), "binary\n");
    await writeFile(join(root, "app.py"), "APP = 1\n");
    const protectedPaths = new Set([
      "extractor.py",
      "excel_writer.py",
      "assets/DHL_Template.xlsx",
    ]);

    // No allowlist at all: the whole project is writable...
    assert.deepEqual(
      await evaluateToolCall(
        root,
        "editor",
        { path: "app.py" },
        undefined,
        protectedPaths,
      ),
      { allowed: true },
    );
    // ...except these.
    for (const path of protectedPaths) {
      const decision = await evaluateToolCall(
        root,
        "editor",
        { path },
        undefined,
        protectedPaths,
      );
      assert.equal(decision.allowed, false, path);
      assert.equal(decision.reasonCode, "protected_path", path);
      // The refusal tells the model what to do instead, or it will simply
      // try the same edit again with the rest of its budget.
      assert.match(
        decision.reason ?? "",
        /read it to understand its interface/,
      );
      assert.match(decision.reason ?? "", /Adapt your own files/);
    }

    // Reading is the whole point of protecting rather than hiding: the port
    // has to match the interface it must not change.
    assert.deepEqual(
      await evaluateToolCall(
        root,
        "read_files",
        { paths: ["extractor.py"] },
        undefined,
        protectedPaths,
      ),
      { allowed: true },
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("a protected path cannot be reached by an alias or a traversal", async () => {
  const root = await mkdtemp(join(tmpdir(), "metis-protected-alias-"));
  try {
    await writeFile(join(root, "extractor.py"), "OCI = 1\n");
    const protectedPaths = new Set(["extractor.py"]);

    for (const path of ["./extractor.py", "app/../extractor.py"]) {
      const decision = await evaluateToolCall(
        root,
        "editor",
        { path },
        undefined,
        protectedPaths,
      );
      assert.equal(decision.allowed, false, path);
      assert.equal(decision.reasonCode, "protected_path", path);
    }
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("the broad writable root still never reaches control, secret or aliased paths", async () => {
  const root = await mkdtemp(join(tmpdir(), "metis-broad-root-"));
  const outside = await mkdtemp(join(tmpdir(), "metis-broad-outside-"));
  try {
    await mkdir(join(root, ".git"), { recursive: true });
    await writeFile(join(root, ".git", "config"), "[core]\n");
    await writeFile(join(root, ".env"), "TOKEN=abc\n");
    await writeFile(join(outside, "elsewhere.py"), "X = 1\n");
    await symlink(join(outside, "elsewhere.py"), join(root, "linked.py"));

    // No allowlist and no deny-list: the standard exclusions still stand.
    for (const path of [".git/config", ".env", "linked.py", "../escape.py"]) {
      const decision = await evaluateToolCall(
        root,
        "editor",
        { path },
        undefined,
        undefined,
      );
      assert.equal(decision.allowed, false, path);
    }
  } finally {
    await rm(root, { recursive: true, force: true });
    await rm(outside, { recursive: true, force: true });
  }
});

// ── broad scope (cline_direct): inspection is not a defect ──────────────────
// The sliced guard above measures the wrong thing when one session owns a
// whole project. These pin the replacement: distinct successful inspection is
// unbounded, and only an identical failing action is stopped.

test("broad scope lets twelve distinct successful inspections precede an edit", async () => {
  const root = await mkdtemp(join(tmpdir(), "metis-broad-inspection-"));
  const observations: SafeToolObservation[] = [];
  try {
    await mkdir(join(root, "app"), { recursive: true });
    for (let index = 0; index < 12; index += 1) {
      await writeFile(join(root, "app", `module_${index}.py`), "VALUE = 1\n");
    }
    const hooks = createFailClosedHooks({
      workspaceRoot: root,
      broadScope: true,
      observer: (event) => observations.push(event),
    });

    for (let index = 0; index < 12; index += 1) {
      // Alternate the two inspection tools, exactly as a real session does.
      const before =
        index % 2 === 0
          ? await hooks.beforeTool?.({
              tool: { name: "read_files" },
              input: { files: [{ path: `app/module_${index}.py` }] },
            } as never)
          : await hooks.beforeTool?.({
              tool: { name: "search_codebase" },
              input: { queries: [`distinct-query-${index}`] },
            } as never);
      assert.equal(before?.skip, undefined, `inspection ${index} was refused`);
      assert.equal(before?.stop, undefined, `inspection ${index} stopped the turn`);
      const after = await hooks.afterTool?.({
        tool: { name: index % 2 === 0 ? "read_files" : "search_codebase" },
        result: { output: { success: true, results: [] } },
      } as never);
      // No corrective nudge is ever attached on this path.
      const output = (
        after as { result?: { output?: { metis_notice?: unknown } } } | undefined
      )?.result?.output;
      assert.equal(output?.metis_notice, undefined);
    }

    // The 13th call is the first edit, and it is allowed.
    const edit = await hooks.beforeTool?.({
      tool: { name: "editor" },
      input: { path: "app/module_0.py" },
    } as never);
    assert.equal(edit?.skip, undefined);
    assert.equal(edit?.stop, undefined);

    // Nothing was ever denied, so nothing was ever counted against the session.
    assert.deepEqual(
      observations.filter((event) => event.status === "denied"),
      [],
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("broad scope bounds an identical failing action but not distinct ones", async () => {
  const root = await mkdtemp(join(tmpdir(), "metis-broad-repeat-"));
  const observations: SafeToolObservation[] = [];
  try {
    const hooks = createFailClosedHooks({
      workspaceRoot: root,
      broadScope: true,
      observer: (event) => observations.push(event),
    });

    // Six DIFFERENT recoverable denials (each escapes the workspace, which is
    // the archetypal model path typo): each is a distinct action, so none of
    // them accumulates toward the repeat bound and the turn stays open.
    for (let index = 0; index < 6; index += 1) {
      const distinct = await hooks.beforeTool?.({
        tool: { name: "read_files" },
        input: { files: [{ path: `/elsewhere/module_${index}.py` }] },
      } as never);
      assert.equal(distinct?.skip, true, `distinct denial ${index} was not refused`);
      assert.equal(
        distinct?.stop,
        undefined,
        `distinct denial ${index} stopped the turn`,
      );
    }

    // The SAME failing call, three times: that is a loop, and it is stopped.
    const identical = { files: [{ path: "/elsewhere/same.py" }] };
    const first = await hooks.beforeTool?.({
      tool: { name: "read_files" },
      input: identical,
    } as never);
    assert.equal(first?.stop, undefined);
    const second = await hooks.beforeTool?.({
      tool: { name: "read_files" },
      // Same action expressed as a fresh object: still the same action.
      input: { files: [{ path: "/elsewhere/same.py" }] },
    } as never);
    assert.equal(second?.stop, undefined);
    const third = await hooks.beforeTool?.({
      tool: { name: "read_files" },
      input: identical,
    } as never);
    assert.equal(third?.skip, true);
    assert.equal(third?.stop, true);
    assert.equal(observations.at(-1)?.reasonCode, "stuck_inspection_loop");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("broad scope keeps every path and security denial terminal", async () => {
  const root = await mkdtemp(join(tmpdir(), "metis-broad-security-"));
  try {
    await writeFile(join(root, "keep.py"), "VALUE = 1\n");
    const hooks = createFailClosedHooks({
      workspaceRoot: root,
      broadScope: true,
      protectedPaths: new Set(["keep.py"]),
    });

    // A protected write, a control-directory read, and a disabled tool each
    // stop the turn on the first attempt, exactly as on the sliced path.
    const protectedWrite = await hooks.beforeTool?.({
      tool: { name: "editor" },
      input: { path: "keep.py" },
    } as never);
    assert.equal(protectedWrite?.stop, true);

    const secret = await hooks.beforeTool?.({
      tool: { name: "read_files" },
      input: { files: [{ path: ".env" }] },
    } as never);
    assert.equal(secret?.stop, true);

    const disabled = await hooks.beforeTool?.({
      tool: { name: "fetch_web_content" },
      input: {},
    } as never);
    assert.equal(disabled?.stop, true);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("the sliced inspection stop is unchanged when broad scope is absent", async () => {
  // The frozen path keeps exactly the behaviour its checkpoints started under.
  const root = await mkdtemp(join(tmpdir(), "metis-sliced-unchanged-"));
  const observations: SafeToolObservation[] = [];
  try {
    const hooks = createFailClosedHooks({
      workspaceRoot: root,
      observer: (event) => observations.push(event),
    });
    for (let index = 0; index < 4; index += 1) {
      await hooks.beforeTool?.({
        tool: { name: "search_codebase" },
        input: { queries: [`query-${index}`] },
      } as never);
      await hooks.afterTool?.({
        tool: { name: "search_codebase" },
        result: { output: { success: true, results: [] } },
      } as never);
    }
    const fifth = await hooks.beforeTool?.({
      tool: { name: "search_codebase" },
      input: { queries: ["query-4"] },
    } as never);
    assert.equal(fifth?.stop, true);
    assert.equal(observations.at(-1)?.reasonCode, "stuck_inspection_loop");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
