import assert from "node:assert/strict";
import test from "node:test";

import { createCustomerWin, deleteCustomerWin, updateCustomerWin } from "../lib/api.ts";

function jsonResponse(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "content-type": "application/json" },
  });
}

const WIN = {
  id: "cwin_1234567890abcdef",
  account_id: "cust_1234567890abcdef1234",
  account_name: "Northwind Authority",
  title: "Identity service Model Import DAC live",
  brief: "Model Import DAC connected through the OpenAI-compatible endpoint.",
  services: ["Generative AI Services", "DAC", "Model-Import"],
  dac_shape: "Model Import DAC (2xA100-40G)",
  yearly_arr: 42000,
  won_at: "2025-12-30T00:00:00Z",
  source_ref: "",
  created_at: "2025-12-30T08:08:29Z",
  updated_at: "2025-12-30T08:08:29Z",
};

test("createCustomerWin posts to the account-scoped wins endpoint", async () => {
  const originalFetch = globalThis.fetch;
  let requestedUrl = "";
  let requestedMethod = "";
  let requestedBody: Record<string, unknown> = {};
  globalThis.fetch = async (input, init) => {
    requestedUrl = String(input);
    requestedMethod = String(init?.method);
    requestedBody = JSON.parse(String(init?.body));
    return jsonResponse(WIN);
  };
  try {
    const win = await createCustomerWin("cust_1234567890abcdef1234", {
      title: "Identity service Model Import DAC live",
      services: ["Generative AI Services", "DAC", "Model-Import"],
      yearly_arr: 42000,
      won_at: "2025-12-30T00:00:00Z",
    });
    assert.match(requestedUrl, /\/api\/v1\/customers\/cust_1234567890abcdef1234\/wins$/);
    assert.equal(requestedMethod, "POST");
    assert.equal(requestedBody.yearly_arr, 42000);
    assert.deepEqual(win.services, ["Generative AI Services", "DAC", "Model-Import"]);
    assert.equal(win.account_name, "Northwind Authority");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("updateCustomerWin puts to the win endpoint", async () => {
  const originalFetch = globalThis.fetch;
  let requestedUrl = "";
  let requestedMethod = "";
  globalThis.fetch = async (input, init) => {
    requestedUrl = String(input);
    requestedMethod = String(init?.method);
    return jsonResponse({ ...WIN, yearly_arr: 90000 });
  };
  try {
    const win = await updateCustomerWin(WIN.id, { title: WIN.title, yearly_arr: 90000 });
    assert.match(requestedUrl, /\/api\/v1\/customers\/wins\/cwin_1234567890abcdef$/);
    assert.equal(requestedMethod, "PUT");
    assert.equal(win.yearly_arr, 90000);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("deleteCustomerWin issues a DELETE and tolerates an empty body", async () => {
  const originalFetch = globalThis.fetch;
  let requestedUrl = "";
  let requestedMethod = "";
  globalThis.fetch = async (input, init) => {
    requestedUrl = String(input);
    requestedMethod = String(init?.method);
    return new Response(null, { status: 204 });
  };
  try {
    await deleteCustomerWin(WIN.id);
    assert.match(requestedUrl, /\/api\/v1\/customers\/wins\/cwin_1234567890abcdef$/);
    assert.equal(requestedMethod, "DELETE");
  } finally {
    globalThis.fetch = originalFetch;
  }
});
