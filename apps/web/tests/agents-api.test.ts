import assert from "node:assert/strict";
import test from "node:test";

import { agentDemoUrl, deployAgent, removeAgent } from "../lib/api.ts";
import {
  EMPTY_BRIEF,
  briefToRequest,
  embedSnippet,
  parseSourceUrls,
  splitLines,
  statusLabel,
  testsSummary,
  validateBrief,
} from "../lib/agents.ts";

const BRIEF = {
  company: "Batelco",
  brief: "Answer billing questions and book a callback.",
  source_urls: "https://www.batelco.com/help\nhttps://www.batelco.com/help, https://www.batelco.com/plans",
  notes: "",
};

// -- the brief form ----------------------------------------------------------

test("the brief needs a company and a purpose, named at the field", () => {
  const problems = validateBrief({ ...EMPTY_BRIEF });
  assert.ok(problems.company);
  assert.ok(problems.brief);
  assert.equal(briefToRequest({ ...EMPTY_BRIEF }), null);
});

test("urls split on lines, commas and spaces, and repeats drop", () => {
  assert.deepEqual(parseSourceUrls(BRIEF.source_urls), [
    "https://www.batelco.com/help",
    "https://www.batelco.com/plans",
  ]);
  assert.deepEqual(briefToRequest(BRIEF)?.source_urls, [
    "https://www.batelco.com/help",
    "https://www.batelco.com/plans",
  ]);
});

test("a non-web address blocks the draft and is named", () => {
  const problems = validateBrief({ ...BRIEF, source_urls: "ftp://nope" });
  assert.match(problems.source_urls ?? "", /ftp:\/\/nope/);
  assert.equal(briefToRequest({ ...BRIEF, source_urls: "ftp://nope" }), null);
  const many = Array.from({ length: 7 }, (_, index) => `https://x.com/${index}`).join("\n");
  assert.match(validateBrief({ ...BRIEF, source_urls: many }).source_urls ?? "", /At most/);
});

test("lines become list items and blanks vanish", () => {
  assert.deepEqual(splitLines("one\n\n two \r\nthree"), ["one", "two", "three"]);
});

// -- words for the page ------------------------------------------------------

test("the embed snippet carries the agent id and the widget script", () => {
  const snippet = embedSnippet("agent_123");
  assert.match(snippet, /agent-id="agent_123"/);
  assert.match(snippet, /convai-widget-embed/);
});

test("status and test summaries read as plain words", () => {
  assert.equal(statusLabel("deployed"), "live");
  assert.equal(statusLabel("failed"), "needs review");
  assert.equal(testsSummary({ tests: [], tests_stage: "" }), "Not run yet");
  assert.equal(testsSummary({ tests: [], tests_stage: "running" }), "Running the simulations");
  assert.equal(
    testsSummary({
      tests_stage: "ready",
      tests: [
        { name: "a", status: "passed", rationale: "" },
        { name: "b", status: "failed", rationale: "" },
      ],
    }),
    "1 of 2 passed",
  );
});

// -- the API calls -----------------------------------------------------------

test("deploying sends the explicit confirmation and nothing else", async () => {
  const original = globalThis.fetch;
  let seen: { url: string; init?: RequestInit } | null = null;
  globalThis.fetch = (async (url: string, init?: RequestInit) => {
    seen = { url, init };
    return new Response(JSON.stringify({ id: "agent_1", status: "deployed" }), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  }) as typeof fetch;
  try {
    const agent = await deployAgent("agent_1");
    assert.equal(agent.status, "deployed");
    assert.match(seen!.url, /\/api\/v1\/agents\/agent_1\/deploy$/);
    assert.equal(seen!.init?.method, "POST");
    assert.deepEqual(JSON.parse(String(seen!.init?.body)), { confirm: true });
  } finally {
    globalThis.fetch = original;
  }
});

test("removing an agent accepts an empty reply", async () => {
  const original = globalThis.fetch;
  globalThis.fetch = (async () => new Response(null, { status: 204 })) as typeof fetch;
  try {
    await removeAgent("agent_1");
  } finally {
    globalThis.fetch = original;
  }
});

test("the demo url is absolute and names the agent", () => {
  assert.match(agentDemoUrl("agent_1"), /^https?:\/\/.+\/api\/v1\/agents\/agent_1\/demo$/);
});
