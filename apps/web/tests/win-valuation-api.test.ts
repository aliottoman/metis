import assert from "node:assert/strict";
import test from "node:test";

import {
  acceptWinValuation,
  dismissWinValuation,
  estimateWinValuation,
  getSkuRates,
  saveSkuRates,
} from "../lib/api.ts";

function jsonResponse(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "content-type": "application/json" },
  });
}

const WIN_ID = "cwin_1234567890abcdef";

const VALUATION = {
  id: "cval_1234567890abcdef",
  win_id: WIN_ID,
  estimated_yearly_arr: 374040,
  currency: "USD",
  lines: [
    {
      sku: "B98415",
      part_number: "B98415",
      name: "Oracle Cloud Infrastructure - Compute - GPU H100",
      unit: "GPU Per Hour",
      quantity: 4,
      utilization: 1,
      rate: 10,
      rate_verified: false,
      yearly_amount: 350400,
      basis: "4 × $10.0000/hr × 8,760 hr",
      why: "4xH100 GPU cluster",
    },
  ],
  explanation: "Four H100s running continuously.",
  confidence: "high",
  unpriced: [],
  rates_verified: false,
  model_used: "grok-4.3",
  prompt_version: "win-valuation-v1",
  status: "proposed",
  created_at: "2026-07-29T06:50:43Z",
  updated_at: "2026-07-29T06:50:43Z",
};

test("estimateWinValuation posts to the win-scoped valuation endpoint", async () => {
  const originalFetch = globalThis.fetch;
  let requestedUrl = "";
  let requestedMethod = "";
  globalThis.fetch = async (input, init) => {
    requestedUrl = String(input);
    requestedMethod = String(init?.method);
    return jsonResponse(VALUATION);
  };
  try {
    const valuation = await estimateWinValuation(WIN_ID);
    assert.match(requestedUrl, /\/api\/v1\/customers\/wins\/cwin_1234567890abcdef\/valuation$/);
    assert.equal(requestedMethod, "POST");
    assert.equal(valuation.estimated_yearly_arr, 374040);
    // The proposal must arrive unreviewed — the UI relies on this to keep it
    // out of the ARR total until the user accepts it.
    assert.equal(valuation.status, "proposed");
    assert.equal(valuation.lines[0].rate_verified, false);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("acceptWinValuation sends the estimate through unchanged by default", async () => {
  const originalFetch = globalThis.fetch;
  let requestedBody: Record<string, unknown> = {};
  globalThis.fetch = async (input, init) => {
    requestedBody = JSON.parse(String(init?.body));
    return jsonResponse({ ...VALUATION, status: "accepted" });
  };
  try {
    const accepted = await acceptWinValuation(WIN_ID);
    // A null figure means "use the estimate"; sending 0 would zero the win.
    assert.equal(requestedBody.yearly_arr, null);
    assert.equal(accepted.status, "accepted");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("acceptWinValuation carries a corrected figure", async () => {
  const originalFetch = globalThis.fetch;
  let requestedBody: Record<string, unknown> = {};
  globalThis.fetch = async (input, init) => {
    requestedBody = JSON.parse(String(init?.body));
    return jsonResponse({ ...VALUATION, status: "accepted" });
  };
  try {
    await acceptWinValuation(WIN_ID, 250_000);
    assert.equal(requestedBody.yearly_arr, 250_000);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("dismissWinValuation posts to the dismiss route", async () => {
  const originalFetch = globalThis.fetch;
  let requestedUrl = "";
  globalThis.fetch = async (input) => {
    requestedUrl = String(input);
    return jsonResponse({ ...VALUATION, status: "dismissed" });
  };
  try {
    const dismissed = await dismissWinValuation(WIN_ID);
    assert.match(requestedUrl, /\/customers\/wins\/cwin_1234567890abcdef\/valuation\/dismiss$/);
    assert.equal(dismissed.status, "dismissed");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("saveSkuRates sends only the edited rows", async () => {
  const originalFetch = globalThis.fetch;
  let requestedMethod = "";
  let requestedBody: { updates?: Array<Record<string, unknown>> } = {};
  globalThis.fetch = async (input, init) => {
    requestedMethod = String(init?.method);
    requestedBody = JSON.parse(String(init?.body));
    return jsonResponse({
      currency: "USD",
      hours_per_year: 8760,
      source_urls: [],
      catalog_size: 758,
      rates: [],
    });
  };
  try {
    await saveSkuRates([{ key: "B98415", value: 12.5, verified: true }]);
    assert.equal(requestedMethod, "PUT");
    assert.equal(requestedBody.updates?.length, 1);
    assert.equal(requestedBody.updates?.[0].key, "B98415");
    assert.equal(requestedBody.updates?.[0].verified, true);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("getSkuRates reads the rate card", async () => {
  const originalFetch = globalThis.fetch;
  let requestedUrl = "";
  globalThis.fetch = async (input) => {
    requestedUrl = String(input);
    return jsonResponse({
      currency: "USD",
      hours_per_year: 8760,
      source_urls: ["https://www.oracle.com/cloud/price-list/"],
      catalog_size: 758,
      rates: [
        {
          key: "B98415", part_number: "B98415", unit: "GPU Per Hour",
          value: 10, label: "OCI Compute — GPU H100", verified: false,
          aliases: ["H100"], note: "",
        },
      ],
    });
  };
  try {
    const card = await getSkuRates();
    assert.match(requestedUrl, /\/api\/v1\/sku-rates$/);
    assert.equal(card.catalog_size, 758);
    assert.equal(card.rates[0].verified, false);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
