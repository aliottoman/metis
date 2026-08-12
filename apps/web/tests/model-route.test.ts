import assert from "node:assert/strict";
import test from "node:test";

import {
  clinePassReady,
  conversationModelRoute,
  projectMappingReady,
  shouldShowLocalSession,
} from "../lib/model-route.ts";
import { activeModelLabel, isCloudActive } from "../lib/model.ts";
import type { ModelPreference } from "../lib/types.ts";

test("a saved Cline provider outranks an old hosted Ollama pin", () => {
  assert.equal(conversationModelRoute("cline", true), "cline");
  assert.equal(conversationModelRoute("local", true), "ollama_cloud");
});

test("ClinePass is selectable only with its key and backend catalog", () => {
  assert.equal(clinePassReady(false, ["cline-pass/glm-5.2"]), false);
  assert.equal(clinePassReady(true, []), false);
  assert.equal(clinePassReady(true, ["cline-pass/glm-5.2"]), true);
});

test("a configured ClinePass lane can map a new project", () => {
  assert.equal(projectMappingReady({
    oci_available: false,
    cohere_available: false,
    cline_available: true,
    cline_models: ["cline-pass/glm-5.2"],
  }), true);
  assert.equal(projectMappingReady({
    oci_available: false,
    cohere_available: false,
    cline_available: true,
    cline_models: [],
  }), false);
});

test("shared hosted-model state treats ClinePass as ready without Ollama", () => {
  const preference = { provider: "cline" } as ModelPreference;
  assert.equal(isCloudActive(preference), true);
  assert.equal(activeModelLabel(preference, null), "ClinePass");
});

test("Cline routes never expose Ollama launch controls", () => {
  assert.equal(
    shouldShowLocalSession(false, "grok_bootstrap_local", "cline", "cline"),
    false,
  );
  assert.equal(
    shouldShowLocalSession(true, "grok_bootstrap_local", "cline", "cline"),
    false,
  );
  assert.equal(
    shouldShowLocalSession(true, "grok_bootstrap_local", "local", "local"),
    true,
  );
});
