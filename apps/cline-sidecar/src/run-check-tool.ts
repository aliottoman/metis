/**
 * `run_check` as a real ClineCore tool.
 *
 * It used to be a disguise: the SDK's built-in `run_commands` was enabled in
 * policy and then intercepted, so the model was shown a shell tool's schema —
 * `command`, `args`, `cwd` — and refused every time it filled that schema in.
 * Two live runs died there. GLM implemented the whole feature, then spent its
 * last three iterations trying to satisfy a contract it had never been shown,
 * and its own words were "The check calls must be the only field. Let me retry
 * with the exact format."
 *
 * So the model is now shown the real thing. One closed enum, one required
 * field, nothing else accepted. Nothing executes here either way: `execute`
 * hands the name to Metis over the existing hostCall bridge, and Metis runs it
 * in the pinned networkless verifier it already owns.
 */

import { createTool } from "@cline/sdk";

import {
  HOST_CHECKS,
  isHostCheck,
  type HostCheck,
  type HostCheckFinding,
  type HostCheckResult,
} from "./protocol.js";
import type { SafeToolObserver } from "./security.js";

export const RUN_CHECK_TOOL_NAME = "run_check";

/** Bounded so one noisy check cannot flood the turn's context. */
const MAX_REPORTED_FINDINGS = 20;
const MAX_DETAIL_CHARS = 2_000;

/**
 * Exactly `{ "check": "<one of HOST_CHECKS>" }`.
 *
 * `additionalProperties: false` is the whole point: a model that reaches for
 * `command`, `argv`, `cwd` or `env` gets a schema violation from its own
 * provider, before any call is made, instead of a refusal it has to guess its
 * way out of.
 */
export const RUN_CHECK_INPUT_SCHEMA: Record<string, unknown> = {
  type: "object",
  properties: {
    check: {
      type: "string",
      enum: [...HOST_CHECKS],
      description:
        "Which host-owned check to run. Metis owns the command and runs it " +
        "in its own networkless verifier against your workspace.",
    },
  },
  required: ["check"],
  additionalProperties: false,
};

const DESCRIPTION =
  "Ask Metis to run one verification check against your current workspace and " +
  "return its findings. You do not run commands and you cannot choose one: " +
  `pass a single field, {"check": "<name>"}, where <name> is exactly one of ` +
  `${HOST_CHECKS.join(", ")}. "imports" resolves imports, "pytest" runs the ` +
  'discovered test slice, "ruff" lints, "acceptance" runs the acceptance ' +
  'probes, and "full" runs the same verification the approval gate runs. ' +
  "Use it after your edits, read the findings, and fix the causes in your " +
  "authorized files.";

/** What the model reads back. Small, structured and already redacted. */
export interface RunCheckToolOutput {
  check: string;
  status: "clean" | "blocked" | "unavailable";
  errors: number;
  warnings: number;
  findings: HostCheckFinding[];
  truncated: boolean;
  detail: string;
}

function bounded(text: string): string {
  const flat = text.replace(/\s+/g, " ").trim();
  return flat.length > MAX_DETAIL_CHARS
    ? `${flat.slice(0, MAX_DETAIL_CHARS)}…`
    : flat;
}

/** Turn one host answer into the model's structured result. */
export function renderRunCheckOutput(
  result: HostCheckResult,
): RunCheckToolOutput {
  if (result.unavailable !== undefined) {
    return {
      check: result.check,
      status: "unavailable",
      errors: 0,
      warnings: 0,
      findings: [],
      truncated: false,
      detail: bounded(
        `The check did not run: ${result.unavailable}. Continue with the work ` +
          "you can do without it.",
      ),
    };
  }
  if (result.ok) {
    return {
      check: result.check,
      status: "clean",
      errors: 0,
      warnings: result.warnings ?? 0,
      findings: (result.findings ?? []).slice(0, MAX_REPORTED_FINDINGS),
      truncated: result.truncated === true,
      detail: bounded(`${result.check} passed with no blocking findings.`),
    };
  }
  const findings = (result.findings ?? []).slice(0, MAX_REPORTED_FINDINGS);
  return {
    check: result.check,
    status: "blocked",
    errors: result.errors ?? 0,
    warnings: result.warnings ?? 0,
    findings,
    truncated: result.truncated === true || (result.findings ?? []).length > MAX_REPORTED_FINDINGS,
    detail: bounded(
      `${result.check} found ${result.errors ?? 0} error(s) and ` +
        `${result.warnings ?? 0} warning(s). Fix the root causes in your ` +
        "authorized files, then run the check again.",
    ),
  };
}

/** The refusal a malformed call reads, naming the only shape that works. */
export function malformedCheckOutput(received: unknown): RunCheckToolOutput {
  const named =
    typeof received === "string" && received.trim() ? `"${received.trim()}"` : "nothing";
  return {
    check: "",
    status: "unavailable",
    errors: 0,
    warnings: 0,
    findings: [],
    truncated: false,
    detail: bounded(
      `No check ran: this call named ${named}. Metis runs checks, you do not. ` +
        `Call run_check again with exactly {"check": "<name>"} where <name> is ` +
        `one of: ${HOST_CHECKS.join(", ")}.`,
    ),
  };
}

export interface RunCheckToolDeps {
  /** Asks Metis for one named check. Never throws; see HostBridge. */
  runCheck: (check: HostCheck) => Promise<HostCheckResult>;
  observer?: SafeToolObserver;
}

/**
 * The tool handed to ClineCore via `extraTools`.
 *
 * `execute` never touches a shell, a path or an argv. It validates the name
 * against the same closed enum the schema advertises and hands it upstream.
 */
export function createRunCheckTool(deps: RunCheckToolDeps) {
  return createTool<{ check?: unknown }, RunCheckToolOutput>({
    name: RUN_CHECK_TOOL_NAME,
    description: DESCRIPTION,
    inputSchema: RUN_CHECK_INPUT_SCHEMA,
    execute: async (input: { check?: unknown }): Promise<RunCheckToolOutput> => {
      const requested = (input ?? {}).check;
      if (!isHostCheck(requested)) {
        // Fail closed, and say what the accepted shape is. A schema-violating
        // provider will usually have stopped this before it reached us.
        deps.observer?.({
          tool: RUN_CHECK_TOOL_NAME,
          status: "denied",
          reasonCode: requested === undefined ? "command_not_permitted" : "unknown_check",
        });
        return malformedCheckOutput(requested);
      }
      deps.observer?.({ tool: RUN_CHECK_TOOL_NAME, status: "started" });
      const result = await deps.runCheck(requested);
      deps.observer?.({
        tool: RUN_CHECK_TOOL_NAME,
        status: result.unavailable !== undefined ? "failed" : "finished",
      });
      return renderRunCheckOutput(result);
    },
  });
}
