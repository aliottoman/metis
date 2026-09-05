import assert from "node:assert/strict";
import test from "node:test";

import {
  bindVoiceSession,
  getSpeechPreference,
  setSpeechPreference,
} from "../lib/api.ts";

function jsonResponse(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "content-type": "application/json" },
  });
}

const STORED = {
  stt_provider: "elevenlabs",
  spoken_confirmation: false,
  voice_model: "deepseek-v4-flash:cloud",
  cohere_available: true,
  elevenlabs_available: true,
  voice_models: ["deepseek-v4-flash:cloud", "gpt-oss:20b-cloud"],
};

test("reads the speech preference with its availability flags", async () => {
  const originalFetch = globalThis.fetch;
  let requested = "";
  globalThis.fetch = async (input: RequestInfo | URL) => {
    requested = String(input);
    return jsonResponse(STORED);
  };
  try {
    const preference = await getSpeechPreference();
    assert.ok(requested.endsWith("/api/v1/settings/speech"));
    assert.equal(preference.stt_provider, "elevenlabs");
    assert.equal(preference.elevenlabs_available, true);
    assert.deepEqual(preference.voice_models, [
      "deepseek-v4-flash:cloud",
      "gpt-oss:20b-cloud",
    ]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("omits the voice model unless one was named", async () => {
  // The distinction the backend depends on: a surface that only knows about
  // dictation must not reset the voice model by saving without it.
  const originalFetch = globalThis.fetch;
  const bodies: Array<Record<string, unknown>> = [];
  globalThis.fetch = async (_input: RequestInfo | URL, init?: RequestInit) => {
    bodies.push(JSON.parse(String(init?.body)));
    return jsonResponse(STORED);
  };
  try {
    await setSpeechPreference("elevenlabs");
    await setSpeechPreference("cohere", true, "");
    assert.deepEqual(bodies[0], {
      stt_provider: "elevenlabs",
      spoken_confirmation: false,
    });
    assert.deepEqual(bodies[1], {
      stt_provider: "cohere",
      spoken_confirmation: true,
      voice_model: "",
    });
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("binds a provider conversation to the exact local voice lease", async () => {
  const originalFetch = globalThis.fetch;
  let requested = "";
  let method = "";
  let body = "";
  globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
    requested = String(input);
    method = String(init?.method);
    body = String(init?.body);
    return jsonResponse({ id: "vs_1", provider_conversation_id: "conv_1" });
  };
  try {
    await bindVoiceSession("vs/1", "conv_1");
    assert.ok(requested.endsWith("/api/v1/voice/sessions/vs%2F1"));
    assert.equal(method, "PATCH");
    assert.deepEqual(JSON.parse(body), {
      provider_conversation_id: "conv_1",
    });
  } finally {
    globalThis.fetch = originalFetch;
  }
});
