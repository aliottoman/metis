import assert from "node:assert/strict";
import test from "node:test";

import { BACKEND_REASON_LABELS, humanizeToken, OPERATION_LABELS } from "../lib/run-event-labels.ts";

test("known backend reason codes read as plain sentences", () => {
  assert.equal(
    humanizeToken("provider_exhausted", BACKEND_REASON_LABELS),
    "its account-wide limit was hit",
  );
  assert.equal(
    humanizeToken("rate_limited", BACKEND_REASON_LABELS),
    "it is rate-limiting requests",
  );
});

test("known operation tokens read as plain sentences", () => {
  assert.equal(
    humanizeToken("clinecore_round", OPERATION_LABELS),
    "building this vertical slice",
  );
});

test("an unrecognized token never leaks its raw snake_case form", () => {
  const humanized = humanizeToken("some_new_internal_code", BACKEND_REASON_LABELS);
  assert.ok(!humanized.includes("_"), `expected no underscores in ${humanized}`);
  assert.equal(humanized, "some new internal code");
});
