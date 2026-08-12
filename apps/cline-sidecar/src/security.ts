import { lstat, realpath } from "node:fs/promises";
import { dirname, isAbsolute, relative, resolve, sep } from "node:path";

import type {
  AgentHooks,
  ToolApprovalRequest,
  ToolApprovalResult,
  ToolPolicy,
} from "@cline/sdk";

import type { HostCheckResult, ToolDenialReasonCode } from "./protocol.js";

type BeforeToolContext = Parameters<NonNullable<AgentHooks["beforeTool"]>>[0];
type AfterToolContext = Parameters<NonNullable<AgentHooks["afterTool"]>>[0];
type ToolResult = AfterToolContext["result"];

export interface SafeToolObservation {
  tool: string;
  status: "started" | "finished" | "failed" | "denied";
  reasonCode?: ToolDenialReasonCode;
}

export type SafeToolObserver = (event: SafeToolObservation) => void;

export const ALLOWED_TOOLS = new Set([
  "read_files",
  "search_codebase",
  "editor",
  // A real custom tool registered through the SDK's `extraTools`, with its own
  // closed `{check: enum}` schema. It is NOT the built-in command tool wearing
  // a different name: `run_commands` is disabled below and hidden from the
  // model, because advertising a shell schema for a non-shell operation is
  // what made two live sessions burn their remaining iterations guessing.
  "run_check",
] as const);

/** The one tool name the model is given for verification. See run-check-tool.ts. */
export const CHECK_TOOL = "run_check";

const DISABLED_TOOLS = [
  // The SDK's shell. Never enabled, never advertised: Metis owns every argv
  // and runs it in its own networkless verifier, reached through `run_check`.
  "run_commands",
  "apply_patch",
  "fetch_web_content",
  "skills",
  "ask_question",
  "submit_and_exit",
  "spawn_agent",
  "team_spawn_teammate",
  "team_shutdown_teammate",
  "team_status",
  "team_task",
  "team_run_task",
  "team_cancel_run",
  "team_list_runs",
  "team_await_runs",
  "team_send_message",
  "team_broadcast",
  "team_read_mailbox",
  "team_mission_log",
  "team_cleanup",
  "team_create_outcome",
  "team_attach_outcome_fragment",
  "team_review_outcome_fragment",
  "team_finalize_outcome",
  "team_list_outcomes",
] as const;

export const FAIL_CLOSED_TOOL_POLICIES: Readonly<Record<string, ToolPolicy>> =
  Object.freeze({
    ...Object.fromEntries(
      [...ALLOWED_TOOLS].map((name) => [
        name,
        { enabled: true, autoApprove: false },
      ]),
    ),
    ...Object.fromEntries(
      DISABLED_TOOLS.map((name) => [
        name,
        { enabled: false, autoApprove: false },
      ]),
    ),
  });

function isWithin(root: string, candidate: string): boolean {
  const result = relative(root, candidate);
  return (
    result === "" ||
    (!result.startsWith(`..${sep}`) && result !== ".." && !isAbsolute(result))
  );
}

function record(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

function outputRecord(value: unknown): Record<string, unknown> | undefined {
  if (typeof value !== "string") return record(value);
  try {
    return record(JSON.parse(value) as unknown);
  } catch {
    return undefined;
  }
}

function toolResultFailed(result: ToolResult): boolean {
  if (result.isError === true) return true;
  const output = outputRecord(result.output);
  return (
    output?.success === false ||
    (typeof output?.error === "string" && output.error.length > 0)
  );
}

/**
 * Cline's editor returns the entire rendered diff after a successful edit.
 * The model already supplied those bytes, and retaining the diff makes every
 * later full-history request resend it. Keep failure details intact, but turn
 * successful editor output into the same small acknowledgement used by mature
 * coding harnesses; Metis verifies the mirror independently.
 */
export function compactSuccessfulEditorResult(
  toolName: string,
  result: ToolResult,
): ToolResult {
  if (toolName !== "editor" || toolResultFailed(result)) return result;
  const output = outputRecord(result.output);
  if (output?.success !== true) return result;
  const query = typeof output.query === "string" ? output.query : "editor";
  return {
    ...result,
    output: {
      success: true,
      query,
      result:
        "Editor change applied. Continue with the planned edits; Metis will verify the final workspace.",
    },
  };
}

function collectReadPaths(input: unknown): string[] {
  if (typeof input === "string") return [input];
  if (Array.isArray(input)) {
    return input.flatMap((item) =>
      typeof item === "string"
        ? [item]
        : typeof record(item)?.path === "string"
          ? [record(item)?.path as string]
          : [],
    );
  }
  const object = record(input);
  if (object === undefined) return [];
  for (const key of ["path", "file_path", "filePath"] as const) {
    if (typeof object[key] === "string") return [object[key] as string];
  }
  for (const key of ["files", "paths", "file_paths"] as const) {
    if (object[key] !== undefined) return collectReadPaths(object[key]);
  }
  return [];
}

function collectToolPaths(toolName: string, input: unknown): string[] {
  if (toolName === "read_files") return collectReadPaths(input);
  if (toolName === "editor") {
    const path = record(input)?.path;
    return typeof path === "string" ? [path] : [];
  }
  return [];
}

function isWorkspaceOnlySearch(input: unknown): boolean {
  if (typeof input === "string") return true;
  if (Array.isArray(input))
    return input.every((query) => typeof query === "string");
  const object = record(input);
  if (object === undefined) return false;
  if (Object.keys(object).some((key) => key !== "queries")) return false;
  return (
    typeof object.queries === "string" ||
    (Array.isArray(object.queries) &&
      object.queries.every((query) => typeof query === "string"))
  );
}

interface ToolDecision {
  allowed: boolean;
  reason?: string;
  reasonCode?: ToolDenialReasonCode;
}

function denied(
  reasonCode: ToolDenialReasonCode,
  reason: string,
): ToolDecision {
  return { allowed: false, reason, reasonCode };
}

/**
 * A denial the model can act on immediately, not just a fact that it failed.
 * Ephemeral: this text becomes the tool's own error result for this live
 * turn (see createFailClosedHooks), never the persisted event stream, which
 * stays path-free (see the fixed reasonCode-only shape in cline-runtime.ts).
 * Fed back inline rather than left to a follow-up search, which cannot
 * discover a workspace root a search result never contains.
 */
function correctivePathGuidance(
  canonicalRoot: string,
  requestedPath: string,
): string {
  return (
    ` The exact workspace root is ${canonicalRoot} -- pass an absolute path inside it, ` +
    `e.g. ${canonicalRoot}/<the-relative-path-from-your-plan>, not "${requestedPath}". ` +
    'Never assume the root is "/" or any other path.'
  );
}

async function validatePath(
  workspaceRoot: string,
  requestedPath: string,
): Promise<ToolDecision | undefined> {
  if (requestedPath.includes("\0"))
    return denied("invalid_path", "Path contains a NUL byte");
  const canonicalRoot = await realpath(workspaceRoot);
  const candidate = resolve(canonicalRoot, requestedPath);
  if (!isWithin(canonicalRoot, candidate)) {
    return denied(
      "outside_workspace",
      `Path escapes the disposable workspace.${correctivePathGuidance(canonicalRoot, requestedPath)}`,
    );
  }
  const requestedRelative = relative(canonicalRoot, candidate);
  const segments = requestedRelative.split(sep).filter(Boolean);
  if (
    segments.some((segment) =>
      new Set([".git", ".cline", ".metis"]).has(segment),
    )
  ) {
    return denied(
      "protected_path",
      "Path enters a protected project-control directory",
    );
  }
  const basename = segments.at(-1) ?? "";
  if (
    basename === ".env" ||
    (basename.startsWith(".env.") && basename !== ".env.example")
  ) {
    return denied(
      "protected_path",
      "Secret-bearing environment files are not available to the coding engine",
    );
  }

  // Refuse symlink traversal even when the link eventually points back inside the mirror.
  const candidateRelative = requestedRelative;
  let current = canonicalRoot;
  for (const component of candidateRelative.split(sep).filter(Boolean)) {
    current = resolve(current, component);
    try {
      const stat = await lstat(current);
      if (stat.isSymbolicLink()) {
        return denied("symlink_path", "Path traverses a symbolic link");
      }
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") break;
      throw error;
    }
  }

  // For a new file, validate the nearest existing parent after symlink checks.
  let existing = candidate;
  while (true) {
    try {
      const canonicalExisting = await realpath(existing);
      if (!isWithin(canonicalRoot, canonicalExisting)) {
        return denied(
          "outside_workspace",
          `Resolved path escapes the workspace.${correctivePathGuidance(canonicalRoot, requestedPath)}`,
        );
      }
      break;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
      const parent = dirname(existing);
      if (parent === existing || !isWithin(canonicalRoot, parent)) {
        return denied(
          "invalid_path",
          "No safe existing parent was found inside the workspace",
        );
      }
      existing = parent;
    }
  }
  return undefined;
}

/**
 * Checked only for `editor` — the one tool that writes. Reads stay
 * unrestricted by slice scope: an earlier, already-verified slice's files
 * remain readable for context, just never editable, from a later slice.
 */
async function validateSliceScope(
  workspaceRoot: string,
  requestedPath: string,
  allowedPaths: ReadonlySet<string>,
): Promise<ToolDecision | undefined> {
  const canonicalRoot = await realpath(workspaceRoot);
  const candidate = resolve(canonicalRoot, requestedPath);
  const relativePosix = relative(canonicalRoot, candidate).split(sep).join("/");
  if (!allowedPaths.has(relativePosix)) {
    return denied(
      "outside_slice",
      "This file is not part of the current vertical slice's write scope",
    );
  }
  return undefined;
}

async function validateProtectedScope(
  workspaceRoot: string,
  requestedPath: string,
  protectedPaths: ReadonlySet<string>,
): Promise<ToolDecision | undefined> {
  const canonicalRoot = await realpath(workspaceRoot);
  const candidate = resolve(canonicalRoot, requestedPath);
  const relativePosix = relative(canonicalRoot, candidate).split(sep).join("/");
  if (protectedPaths.has(relativePosix)) {
    return denied(
      "protected_path",
      `${relativePosix} is protected for this task: you may read it to ` +
        "understand its interface, but it must not change. Adapt your own " +
        "files to it instead.",
    );
  }
  return undefined;
}

export async function evaluateToolCall(
  workspaceRoot: string,
  toolName: string,
  input: unknown,
  allowedPaths?: ReadonlySet<string>,
  protectedPaths?: ReadonlySet<string>,
): Promise<ToolDecision> {
  if (
    !ALLOWED_TOOLS.has(
      toolName as typeof ALLOWED_TOOLS extends Set<infer T> ? T : never,
    )
  ) {
    return denied(
      "tool_not_allowed",
      `Tool ${toolName} is not in the Metis allowlist`,
    );
  }
  if (toolName === "search_codebase") {
    return isWorkspaceOnlySearch(input)
      ? { allowed: true }
      : denied(
          "invalid_search_scope",
          "Search input attempts to override the workspace scope",
        );
  }
  const paths = collectToolPaths(toolName, input);
  if (paths.length === 0) {
    return denied(
      "missing_path",
      `Tool ${toolName} did not provide an auditable workspace path`,
    );
  }
  for (const requestedPath of paths) {
    const decision = await validatePath(workspaceRoot, requestedPath);
    if (decision !== undefined) return decision;
  }
  // A broad-scope session has no 8-entry allowlist to check against, so the
  // deny-list is what stands between it and a file the task said not to
  // touch. Reads are untouched: understanding an interface requires reading
  // the file that defines it.
  if (toolName === "editor" && protectedPaths !== undefined) {
    for (const requestedPath of paths) {
      const decision = await validateProtectedScope(
        workspaceRoot,
        requestedPath,
        protectedPaths,
      );
      if (decision !== undefined) return decision;
    }
  }
  if (toolName === "editor" && allowedPaths !== undefined) {
    for (const requestedPath of paths) {
      const decision = await validateSliceScope(
        workspaceRoot,
        requestedPath,
        allowedPaths,
      );
      if (decision !== undefined) return decision;
    }
  }
  return { allowed: true };
}

export function decideToolApproval(
  request: ToolApprovalRequest,
): ToolApprovalResult {
  return ALLOWED_TOOLS.has(
    request.toolName as typeof ALLOWED_TOOLS extends Set<infer T> ? T : never,
  )
    ? { approved: true }
    : {
        approved: false,
        reason: `Tool ${request.toolName} is disabled by Metis`,
      };
}

function observeTool(
  observer: SafeToolObserver | undefined,
  event: SafeToolObservation,
): void {
  try {
    observer?.(event);
  } catch {
    // Progress observers never gain authority over tool execution.
  }
}

// A path typo is not an authority grant. The attempted call remains skipped,
// but returning its bounded reason to the same agent lets it use the canonical
// workspace root already present in the system prompt and correct itself. A
// protected-path, symlink, or disabled-tool attempt remains terminal because
// retrying those would be a materially different security signal.
const RECOVERABLE_TOOL_DENIALS = new Set([
  "outside_workspace",
  "invalid_path",
  "missing_path",
  "invalid_search_scope",
] as const);

const INSPECTION_TOOLS = new Set(["read_files", "search_codebase"]);

// A live measured failure: repeated denied read_files calls interleaved with
// search_codebase calls that each individually "succeed" burned an entire
// 16-iteration slice budget without a single editor call. Neither threshold
// alone is enough -- the SDK's own loop detector only catches IDENTICAL
// repeated calls, and every one of those denials/searches had different
// arguments. This tracks unproductive *inspection*, not repetition.
//
// SCOPE: sliced sessions only. On the broad-scope `cline_direct` path this
// heuristic is off, because there it measured the wrong thing: a session that
// owns a whole project legitimately reads many distinct files before its first
// edit, and a fixed count of four cannot tell that apart from a loop. A live
// baseline ended at 6 inspections with 0 edits for exactly that reason. What
// remains on the direct path is the SDK's own identical-call loop detection,
// the repeated-identical-failure bound below, the hard iteration/time limits,
// and every path/security denial -- none of which count successful, distinct
// inspection against the session.
const STUCK_DENIAL_THRESHOLD = 2;
const STUCK_INSPECTION_THRESHOLD = 4;

// Broad-scope bound: the SAME call failing over and over is a loop; different
// calls failing are a session exploring a tree it does not know yet. Only the
// former is stopped, and the signature is tool + exact arguments, so one
// changed path makes it a different action.
const REPEATED_IDENTICAL_FAILURE_LIMIT = 3;

/** Stable identity of one attempted action: the tool and its exact arguments. */
function actionSignature(toolName: string, input: unknown): string {
  let rendered: string;
  try {
    rendered = stableStringify(input);
  } catch {
    // An un-serializable input cannot be proven identical to anything, so it
    // is treated as unique rather than folded into whatever came before it.
    rendered = `unrepresentable:${Math.random()}`;
  }
  return `${toolName} ${rendered}`;
}

/** Key-order-independent JSON, so `{a,b}` and `{b,a}` are the same action. */
function stableStringify(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value) ?? "null";
  if (Array.isArray(value)) return `[${value.map(stableStringify).join(",")}]`;
  const entries = Object.entries(value as Record<string, unknown>).sort(
    ([left], [right]) => (left < right ? -1 : left > right ? 1 : 0),
  );
  return `{${entries.map(([key, item]) => `${JSON.stringify(key)}:${stableStringify(item)}`).join(",")}}`;
}

function appendCorrectiveNotice(
  result: ToolResult,
  notice: string,
): ToolResult {
  if (toolResultFailed(result)) return result;
  const output = outputRecord(result.output);
  if (output === undefined) return result;
  const withNotice = { ...output, metis_notice: notice };
  return {
    ...result,
    output:
      typeof result.output === "string"
        ? JSON.stringify(withNotice)
        : withNotice,
  };
}

export type CheckRunner = (check: string) => Promise<HostCheckResult>;

export interface FailClosedHooksOptions {
  workspaceRoot: string;
  /** See StartSliceParams.allowedPaths; undefined applies no slice restriction. */
  allowedPaths?: ReadonlySet<string>;
  /** Readable, never writable. Refused again by Metis at import. */
  protectedPaths?: ReadonlySet<string>;
  observer?: SafeToolObserver;
  /** Runs one host-owned check. Absent means checks are not offered. */
  runCheck?: CheckRunner;
  /**
   * The `cline_direct` path: one session owns the whole authorized scope and
   * chooses its own order. Turns OFF the fixed inspection-count stop and the
   * cross-action denial counter; leaves every path/security denial, the
   * repeated-identical-failure bound, and the hard limits exactly as they are.
   */
  broadScope?: boolean;
}

export function createFailClosedHooks(
  options: FailClosedHooksOptions,
): AgentHooks {
  const {
    workspaceRoot,
    allowedPaths,
    protectedPaths,
    observer,
    runCheck,
    broadScope = false,
  } = options;
  // Per-session state: this closure is installed once at session start and
  // reused for every continuation within that same live session (Metis never
  // rebuilds hooks mid-session), so these counters see the whole turn.
  let consecutiveDenials = 0;
  let inspectionSinceEdit = 0;
  let correctionIssued = false;
  let noticeForThisAllowedCall = false;
  // Broad scope only: how many times each exact failed action has repeated.
  // Successful calls never appear here, so distinct inspection is free.
  const failuresBySignature = new Map<string, number>();

  function correctiveMessage(detail: string): string {
    return (
      `${detail} STOP INSPECTING AND ACT: ${inspectionSinceEdit} read/search call(s) have ` +
      "passed without an editor call. Use the exact workspace root and the authorized file " +
      "paths already named in your instructions, and call editor now for the next owed file."
    );
  }

  return {
    beforeTool: async ({ tool, input }: BeforeToolContext) => {
      const isInspection = INSPECTION_TOOLS.has(tool.name);
      const isEditor = tool.name === "editor";

      // ── run_check ──────────────────────────────────────────────────────
      // A real registered tool now, so it executes itself: run-check-tool.ts
      // validates the name against the same closed enum its schema advertises
      // and asks Metis over the hostCall bridge. Nothing runs in this process.
      //
      // The hook's only job here is to let it through untouched. It carries no
      // path, so there is nothing to resolve; and a check is progress rather
      // than inspection, so it must not push the model toward any stop, nor
      // reset the editor counters.
      if (tool.name === CHECK_TOOL) {
        if (runCheck === undefined) {
          observeTool(observer, {
            tool: CHECK_TOOL,
            status: "denied",
            reasonCode: "tool_not_allowed",
          });
          return {
            skip: true,
            reason:
              "run_check is not available in this session. Finish the edits you " +
              "can make; Metis verifies independently afterwards.",
            policy: { enabled: false, autoApprove: false },
          };
        }
        return { policy: { enabled: true, autoApprove: false } };
      }

      // Corrective guidance already fired once this stuck episode, and the
      // very next call is still not an editor call: stop honestly rather
      // than spend the rest of this slice's budget repeating inspection.
      // (The host reacts to a round that produced no writes by switching to
      // the next coder for the retry; this only ends the turn early.)
      //
      // Sliced sessions only: see STUCK_INSPECTION_THRESHOLD.
      if (!broadScope && correctionIssued && isInspection) {
        observeTool(observer, {
          tool: tool.name,
          status: "denied",
          reasonCode: "stuck_inspection_loop",
        });
        return {
          skip: true,
          stop: true,
          reason:
            "Stopping this turn: corrective guidance already asked for an editor call and " +
            "none followed. Repeating inspection would spend the rest of this slice's budget " +
            "for nothing.",
          policy: { enabled: false, autoApprove: false },
        };
      }

      const decision = await evaluateToolCall(
        workspaceRoot,
        tool.name,
        input,
        allowedPaths,
        protectedPaths,
      );
      noticeForThisAllowedCall = false;

      if (decision.allowed) {
        if (isEditor) {
          consecutiveDenials = 0;
          inspectionSinceEdit = 0;
          correctionIssued = false;
        } else if (isInspection && !broadScope) {
          inspectionSinceEdit += 1;
          consecutiveDenials = 0;
          if (
            !correctionIssued &&
            inspectionSinceEdit >= STUCK_INSPECTION_THRESHOLD
          ) {
            correctionIssued = true;
            noticeForThisAllowedCall = true;
          }
        }
        // Broad scope: an allowed call is progress. Nothing is counted
        // against the session for succeeding, however many distinct files it
        // reads or searches before its first edit.
        observeTool(observer, { tool: tool.name, status: "started" });
        return { policy: { enabled: true, autoApprove: false } };
      }

      const recoverable = RECOVERABLE_TOOL_DENIALS.has(
        decision.reasonCode as typeof RECOVERABLE_TOOL_DENIALS extends Set<
          infer T
        >
          ? T
          : never,
      );
      observeTool(observer, {
        tool: tool.name,
        status: "denied",
        ...(decision.reasonCode === undefined
          ? {}
          : { reasonCode: decision.reasonCode }),
      });

      if (broadScope) {
        // A non-recoverable denial (protected path, symlink, disabled tool)
        // is terminal here exactly as it is for a sliced session: retrying
        // one is a materially different security signal, not a typo.
        if (!recoverable) {
          return {
            skip: true,
            stop: true,
            reason: decision.reason ?? "Tool denied by Metis",
            policy: { enabled: false, autoApprove: false },
          };
        }
        // Recoverable: bound the SAME failing action, and only that. A
        // different path or query is a different action and starts at one.
        const signature = actionSignature(tool.name, input);
        const repeats = (failuresBySignature.get(signature) ?? 0) + 1;
        failuresBySignature.set(signature, repeats);
        if (repeats >= REPEATED_IDENTICAL_FAILURE_LIMIT) {
          observeTool(observer, {
            tool: tool.name,
            status: "denied",
            reasonCode: "stuck_inspection_loop",
          });
          return {
            skip: true,
            stop: true,
            reason:
              `Stopping this turn: the identical ${tool.name} call has now been refused ` +
              `${repeats} times for the same reason. Repeating it cannot succeed. ` +
              (decision.reason ?? "Tool denied by Metis"),
            policy: { enabled: false, autoApprove: false },
          };
        }
        return {
          skip: true,
          reason: decision.reason ?? "Tool denied by Metis",
          policy: { enabled: false, autoApprove: false },
        };
      }

      consecutiveDenials += 1;
      inspectionSinceEdit += 1;
      const stuck =
        consecutiveDenials >= STUCK_DENIAL_THRESHOLD ||
        inspectionSinceEdit >= STUCK_INSPECTION_THRESHOLD;
      if (recoverable && stuck && !correctionIssued) {
        correctionIssued = true;
        return {
          skip: true,
          reason: correctiveMessage(decision.reason ?? "Tool denied by Metis"),
          policy: { enabled: false, autoApprove: false },
        };
      }
      return {
        skip: true,
        ...(recoverable ? {} : { stop: true }),
        reason: decision.reason ?? "Tool denied by Metis",
        policy: { enabled: false, autoApprove: false },
      };
    },
    afterTool: async ({ tool, result }: AfterToolContext) => {
      const failed = toolResultFailed(result);
      observeTool(observer, {
        tool: tool.name,
        status: failed ? "failed" : "finished",
      });
      let compacted = compactSuccessfulEditorResult(tool.name, result);
      if (noticeForThisAllowedCall && !failed) {
        // The call itself succeeded (an ordinary read or search, not a
        // denial), but it was the one that crossed the stuck-inspection
        // threshold. Attach the correction to this call's own result so it
        // reaches the model inline, in the same turn, rather than only on a
        // later denial that may never come.
        compacted = appendCorrectiveNotice(
          compacted,
          correctiveMessage(
            `${tool.name} succeeded, but this is taking too long.`,
          ),
        );
      }
      noticeForThisAllowedCall = false;
      return compacted === result ? undefined : { result: compacted };
    },
  };
}
