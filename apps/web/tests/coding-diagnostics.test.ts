import assert from "node:assert/strict";
import test from "node:test";

import {
  codingRoundSummary,
  codingStepSummary,
  directContractSummary,
  isFailedCodingStep,
} from "../lib/coding-diagnostics.ts";

test("a failed editor step shows the reason, not just the word failed", () => {
  const summary = codingStepSummary({
    tool: "editor",
    status: "tool_call_failed",
    failure_class: "EditApplyError",
    message: "old_text did not match the file at README.md",
    iteration: 3,
  });

  assert.match(summary, /editor could not finish/);
  assert.match(summary, /EditApplyError/);
  assert.match(summary, /old_text did not match the file at README\.md/);
  assert.match(summary, /step 3/);
});

test("a failure the SDK could not explain says so rather than reading as blank", () => {
  const summary = codingStepSummary({
    tool: "editor",
    status: "tool_call_failed",
    message: "SDK returned a failed tool result without diagnostic text",
  });

  assert.match(summary, /editor could not finish/);
  assert.match(summary, /without diagnostic text/);
});

test("a failure with nothing at all still reads as a sentence", () => {
  const summary = codingStepSummary({ tool: "editor", status: "failed" });

  assert.equal(summary, "editor could not finish — no reason was reported");
});

test("failure detection covers the statuses the SDK actually emits", () => {
  for (const status of [
    "tool_call_failed",
    "error",
    "denied",
    "rejected",
    "aborted",
  ]) {
    assert.equal(isFailedCodingStep({ status }), true, status);
  }
  assert.equal(isFailedCodingStep({ status: "tool_call_finished" }), false);
});

test("a usage step shows the running total and what this step added", () => {
  const summary = codingStepSummary({
    status: "usage",
    usage: { totalTokens: 129_764, inputTokens: 124_826, outputTokens: 4_938 },
    usage_delta: { totalTokens: 80_457, first_snapshot: false },
    iteration: 2,
  });

  assert.match(summary, /129,764 tokens so far/);
  assert.match(summary, /80,457 added since the last step/);
  assert.match(summary, /step 2/);
});

test("an unreported count reads as unknown and never as zero", () => {
  const summary = codingStepSummary({
    status: "usage",
    usage: { inputTokens: 10 },
  });

  assert.match(summary, /unknown tokens so far/);
  assert.match(summary, /unknown added since the last step/);
  assert.doesNotMatch(summary, /\b0 tokens\b/);
});

test("a round that wrote nothing says so, with its limit and its cost", () => {
  const summary = codingRoundSummary({
    changed_paths: [],
    iterations: 4,
    controlled_stop_reason: "max_iterations",
    usage: { totalTokens: 134_702 },
  });

  assert.match(summary, /wrote nothing/);
  assert.match(summary, /4 steps/);
  assert.match(summary, /134,702 tokens/);
  assert.match(summary, /stopped at its step limit/);
});

test("a round that wrote files names them", () => {
  const summary = codingRoundSummary({
    changed_paths: ["README.md"],
    iterations: 4,
    usage: { totalTokens: 277_386 },
  });

  assert.match(summary, /wrote README\.md/);
  assert.doesNotMatch(summary, /stopped at its step limit/);
});

test("a round with no reported usage does not claim it was free", () => {
  const summary = codingRoundSummary({
    changed_paths: ["a.py"],
    iterations: 2,
  });

  assert.match(summary, /unknown tokens/);
});

test("the summaries stay in plain language with no internal tokens", () => {
  const summary = codingStepSummary({
    tool: "read_files",
    status: "tool_call_finished",
  });

  assert.equal(summary, "read files");
  assert.doesNotMatch(summary, /_/);
});

test("the contract card names the scope and the protected files before coding", () => {
  const summary = directContractSummary({
    writable_roots: ["."],
    protected_files: ["assets/DHL_Template.xlsx", "extractor.py"],
    unresolved: [],
  });

  assert.match(summary, /one session/);
  assert.match(summary, /can edit anything in this project/);
  assert.match(
    summary,
    /keeping assets\/DHL_Template\.xlsx, extractor\.py unchanged/,
  );
});

test("an unresolved protection reads as waiting, not as a completed step", () => {
  const summary = directContractSummary({
    writable_roots: ["."],
    protected_files: [],
    unresolved: ["payments.py"],
  });

  assert.match(summary, /waiting/);
  assert.match(summary, /payments\.py/);
  assert.doesNotMatch(summary, /one session/);
});

test("a contract with nothing protected says so rather than looking empty", () => {
  const summary = directContractSummary({
    writable_roots: ["."],
    protected_files: [],
  });

  assert.match(summary, /no files singled out/);
});
