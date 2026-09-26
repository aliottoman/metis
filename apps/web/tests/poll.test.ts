import assert from "node:assert/strict";
import test from "node:test";

import { createPollRunner } from "../lib/poll.ts";

test("a slow refresh cannot overlap the next interval or a visibility refresh", async () => {
  let finish!: () => void;
  let calls = 0;
  const run = createPollRunner(() => {
    calls += 1;
    return new Promise<void>((resolve) => { finish = resolve; });
  });

  const initial = run();
  await Promise.all([run(), run()]);
  assert.equal(calls, 1);
  finish();
  await initial;

  const next = run();
  assert.equal(calls, 2);
  finish();
  await next;
});

test("a rejected refresh is contained and the next interval can recover", async () => {
  let attempts = 0;
  const run = createPollRunner(async () => {
    attempts += 1;
    if (attempts === 1) throw new Error("Local service unavailable");
  });

  await assert.doesNotReject(run());
  await run();
  assert.equal(attempts, 2);
});

test("a synchronous callback failure also releases the next refresh", async () => {
  let attempts = 0;
  const run = createPollRunner(() => {
    attempts += 1;
    if (attempts === 1) throw new Error("Refresh could not start");
  });

  await assert.doesNotReject(run());
  await run();
  assert.equal(attempts, 2);
});
