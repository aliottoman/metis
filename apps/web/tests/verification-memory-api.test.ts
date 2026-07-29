import assert from "node:assert/strict";
import test from "node:test";

import {
  decideProjectVerification,
  getMemoryIndexStatus,
  getProjectVerification,
  setMemoryIndexConsent,
} from "../lib/api.ts";

function jsonResponse(value: unknown): Response {
  return new Response(JSON.stringify(value), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

test("carries the plain-English explanation onto the verification view", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => jsonResponse({
    project_id: "asset_00000000000000000001",
    configured: true,
    approved: false,
    fingerprint: "a".repeat(64),
    checks: [
      {
        name: "test",
        command: ["make", "test"],
        description: "Full suite.",
        explanation: "Runs the `test` target from the project's Makefile.",
        timeout_seconds: 300,
      },
    ],
    explanation: "Approving this lets Metis run 1 reviewed check in this project:",
    boundary: "Checks run as your own macOS user account…",
    error: null,
  });
  try {
    const view = await getProjectVerification("asset_00000000000000000001");
    assert.equal(view.configured, true);
    assert.equal(view.approved, false);
    assert.equal(view.checks.length, 1);
    // The explanation is the artefact the approval decision rests on; losing it
    // in normalization would leave the card showing bare argv.
    assert.match(view.checks[0]!.explanation, /Makefile/);
    assert.equal(view.checks[0]!.timeoutSeconds, 300);
    assert.match(view.boundary, /your own macOS user account/);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("approves and revokes a verification recipe through the v1 routes", async () => {
  const originalFetch = globalThis.fetch;
  const seen: string[] = [];
  globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
    seen.push(`${init?.method ?? "GET"} ${String(input)}`);
    return jsonResponse({
      project_id: "asset_00000000000000000001",
      configured: true,
      approved: String(input).endsWith("/approve"),
      checks: [],
      explanation: "",
      boundary: "",
    });
  };
  try {
    const approved = await decideProjectVerification("asset_00000000000000000001", "approve");
    assert.equal(approved.approved, true);
    const revoked = await decideProjectVerification("asset_00000000000000000001", "revoke");
    assert.equal(revoked.approved, false);
    assert.equal(seen.length, 2);
    assert.ok(
      seen[0]!.startsWith("POST ")
        && seen[0]!.endsWith("/api/v1/projects/asset_00000000000000000001/verification/approve"),
      seen[0],
    );
    assert.ok(
      seen[1]!.startsWith("POST ")
        && seen[1]!.endsWith("/api/v1/projects/asset_00000000000000000001/verification/revoke"),
      seen[1],
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("reports memory search as keyword-only until every precondition holds", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => jsonResponse({
    consent: true,
    consent_reason: "enabled",
    cloud_available: false,
    semantic: false,
    active: 4,
    embedded: 0,
  });
  try {
    const status = await getMemoryIndexStatus();
    assert.equal(status.consent, true);
    // Consent is not the same as working; the UI must be able to say which.
    assert.equal(status.cloudAvailable, false);
    assert.equal(status.semantic, false);
    assert.equal(status.active, 4);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("sends memory consent as an explicit boolean decision", async () => {
  const originalFetch = globalThis.fetch;
  let body = "";
  globalThis.fetch = async (_input: RequestInfo | URL, init?: RequestInit) => {
    body = String(init?.body ?? "");
    return jsonResponse({ consent: false, semantic: false, active: 0, embedded: 0 });
  };
  try {
    const status = await setMemoryIndexConsent(false);
    assert.equal(body, JSON.stringify({ consent: false }));
    assert.equal(status.consent, false);
    assert.equal(status.consentReason, null);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
