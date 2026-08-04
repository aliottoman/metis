import assert from "node:assert/strict";
import test from "node:test";

import {
  createCustomerNote,
  deleteCustomerFact,
  editCustomerAction,
  searchCustomerRecords,
  updateCustomer,
  updateCustomerNote,
  updateCustomerSource,
} from "../lib/api.ts";

function jsonResponse(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "content-type": "application/json" },
  });
}

const ACCOUNT_ID = "cust_1234567890abcdef1234";

const NOTE = {
  id: "cnote_1234",
  account_id: ACCOUNT_ID,
  title: "Standing context",
  body: "Runs OCI Ashburn.",
  pinned: true,
  origin: "manual",
  origin_ref: "",
  created_at: "2026-07-31T08:00:00Z",
  updated_at: "2026-07-31T08:00:00Z",
};

/** Runs `call` with fetch stubbed, and reports what the client actually sent. */
async function capture<T>(
  response: Response | (() => Response),
  call: () => Promise<T>,
): Promise<{ url: string; method: string; body: Record<string, unknown>; result: T }> {
  const originalFetch = globalThis.fetch;
  let url = "";
  let method = "";
  let body: Record<string, unknown> = {};
  globalThis.fetch = async (input, init) => {
    url = String(input);
    method = String(init?.method ?? "GET");
    body = init?.body ? JSON.parse(String(init.body)) : {};
    return typeof response === "function" ? response() : response;
  };
  try {
    const result = await call();
    return { url, method, body, result };
  } finally {
    globalThis.fetch = originalFetch;
  }
}

test("createCustomerNote posts to the account-scoped notes endpoint", async () => {
  const { url, method, body, result } = await capture(
    jsonResponse(NOTE, 201),
    () => createCustomerNote(ACCOUNT_ID, { title: "Standing context", body: "Runs OCI Ashburn.", pinned: true }),
  );
  assert.match(url, /\/api\/v1\/customers\/cust_1234567890abcdef1234\/notes$/);
  assert.equal(method, "POST");
  assert.equal(body.pinned, true);
  assert.equal(result.pinned, true);
});

test("a note saved from a conversation carries its origin", async () => {
  const { body } = await capture(
    jsonResponse({ ...NOTE, origin: "chat", origin_ref: "conv_9" }, 201),
    () => createCustomerNote(ACCOUNT_ID, { body: "Answer worth keeping.", origin: "chat", origin_ref: "conv_9" }),
  );
  assert.equal(body.origin, "chat");
  assert.equal(body.origin_ref, "conv_9");
});

test("updateCustomerNote puts to the note endpoint", async () => {
  const { url, method, body } = await capture(
    jsonResponse({ ...NOTE, pinned: false }),
    () => updateCustomerNote(NOTE.id, { title: NOTE.title, body: NOTE.body, pinned: false }),
  );
  assert.match(url, /\/api\/v1\/customers\/notes\/cnote_1234$/);
  assert.equal(method, "PUT");
  assert.equal(body.pinned, false);
});

test("updateCustomer sends the whole profile, aliases included", async () => {
  const { url, method, body } = await capture(
    jsonResponse({ id: ACCOUNT_ID, name: "Northwind Authority" }),
    () => updateCustomer(ACCOUNT_ID, {
      name: "Northwind Authority",
      aliases: ["NWA"],
      industry: "Government",
      region: "UAE",
      status: "paused",
    }),
  );
  assert.match(url, /\/api\/v1\/customers\/cust_1234567890abcdef1234$/);
  assert.equal(method, "PUT");
  assert.deepEqual(body.aliases, ["NWA"]);
  assert.equal(body.status, "paused");
});

test("editCustomerAction puts the full action, not just its status", async () => {
  const { url, method, body } = await capture(
    jsonResponse({ id: "cact_1", description: "Send the shape" }),
    () => editCustomerAction("cact_1", {
      description: "Send the shape",
      owner: "Ali",
      due_at: "2026-08-10T00:00:00Z",
      status: "open",
    }),
  );
  assert.match(url, /\/api\/v1\/customers\/actions\/cact_1$/);
  assert.equal(method, "PUT");
  assert.equal(body.owner, "Ali");
  assert.equal(body.due_at, "2026-08-10T00:00:00Z");
});

test("deleteCustomerFact issues a DELETE and tolerates an empty body", async () => {
  const { url, method } = await capture(
    new Response(null, { status: 204 }),
    () => deleteCustomerFact("cfact_1"),
  );
  assert.match(url, /\/api\/v1\/customers\/facts\/cfact_1$/);
  assert.equal(method, "DELETE");
});

test("updateCustomerSource puts the corrected note", async () => {
  const { url, method, body } = await capture(
    jsonResponse({ id: "csrc_1", title: "Discovery call" }),
    () => updateCustomerSource("csrc_1", { title: "Discovery call", content: "Corrected", source_kind: "meeting" }),
  );
  assert.match(url, /\/api\/v1\/customers\/sources\/csrc_1$/);
  assert.equal(method, "PUT");
  assert.equal(body.source_kind, "meeting");
});

test("searchCustomerRecords encodes the query rather than pasting it into the URL", async () => {
  const { url, method, result } = await capture(
    jsonResponse({ query: "a&b c", hits: [], truncated: false }),
    () => searchCustomerRecords("a&b c", 10),
  );
  assert.match(url, /\/api\/v1\/customers\/search\?/);
  const parsed = new URL(url, "http://127.0.0.1");
  assert.equal(parsed.searchParams.get("q"), "a&b c");
  assert.equal(parsed.searchParams.get("limit"), "10");
  assert.equal(method, "GET");
  assert.equal(result.truncated, false);
});
