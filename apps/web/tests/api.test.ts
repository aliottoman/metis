import assert from "node:assert/strict";
import test from "node:test";

import {
  activateToolVersion,
  createMemoryProposal,
  decideToolImprovement,
  getConversation,
  getNotionConnection,
  getToolImprovementEvidence,
  listRecoverableRuns,
  listProjectWorkspaces,
  listToolDefinitionProposals,
  listToolDefinitions,
  listToolImprovementProposals,
  setModelPreference,
  openProjectWorkspace,
  sendMessage,
  syncNotion,
} from "../lib/api.ts";

function jsonResponse(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "content-type": "application/json" },
  });
}

test("hydrates persisted message run IDs and selects the latest run for replay", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input) => {
    const url = String(input);
    if (url.endsWith("/messages")) {
      return jsonResponse([
        { id: "msg_user_1", role: "user", content: "First", run_id: "run_1" },
        { id: "msg_assistant_1", role: "assistant", content: "Done", run_id: "run_1" },
        { id: "msg_user_2", role: "user", content: "Second", run_id: "run_2" },
        { id: "msg_assistant_2", role: "assistant", content: "Artifacts ready", run_id: "run_2" },
      ]);
    }
    return jsonResponse({ id: "conv_1", title: "Architecture" });
  };

  try {
    const conversation = await getConversation("conv_1");
    assert.equal(conversation.messages[1]?.run_id, "run_1");
    assert.equal(conversation.messages[3]?.run_id, "run_2");
    assert.equal(conversation.latest_run_id, "run_2");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("normalizes persisted approval runs and requests the recovery status filter", async () => {
  const originalFetch = globalThis.fetch;
  let requestedUrl = "";
  globalThis.fetch = async (input) => {
    requestedUrl = String(input);
    return jsonResponse([{
      run: {
        id: "run_1",
        conversation_id: "conv_1",
        user_message_id: "msg_1",
        status: "awaiting_approval",
        graph_schema_version: "1",
        cancel_requested: false,
        result: null,
        created_at: "2026-07-20T10:00:00Z",
        updated_at: "2026-07-20T10:01:00Z",
      },
      approval: {
        id: "appr_1",
        run_id: "run_1",
        title: "Activate tested tool",
        summary: "Promote the immutable candidate.",
        risk_level: "R3",
        permissions: ["write:registry"],
        input_digest: "abc123",
      },
    }]);
  };

  try {
    const runs = await listRecoverableRuns();
    assert.match(requestedUrl, /\/api\/v1\/runs\?status=awaiting_approval$/);
    assert.equal(runs[0]?.run.conversation_id, "conv_1");
    assert.equal(runs[0]?.approval?.id, "appr_1");
    assert.equal(runs[0]?.approval?.action_digest, "abc123");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("sends an idempotent, reasoned prior-version activation", async () => {
  const originalFetch = globalThis.fetch;
  let requestedUrl = "";
  let requestedInit: RequestInit | undefined;
  globalThis.fetch = async (input, init) => {
    requestedUrl = String(input);
    requestedInit = init;
    return jsonResponse({ tool_id: "tool_1", active_version_id: "version_1", prior_version_id: "version_2" });
  };

  try {
    const result = await activateToolVersion("tool_1", "version_1", "restore-key-123", "Restore known-good output");
    assert.match(requestedUrl, /\/tools\/tool_1\/versions\/version_1\/activate$/);
    assert.equal(requestedInit?.method, "POST");
    assert.deepEqual(JSON.parse(String(requestedInit?.body)), {
      idempotency_key: "restore-key-123",
      reason: "Restore known-good output",
    });
    assert.equal(result.active_version_id, "version_1");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("normalizes correction proposals with their regression evidence", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => jsonResponse([{
    id: "timpr_1",
    source_run_id: "run_4",
    tool_id: "tool_2",
    tool_version_id: "version_8",
    content_hash: "deadbeef",
    correction: "Keep database traffic inside the private subnet.",
    regression_eval: {
      id: "regression-1",
      name: "Private subnet correction",
      input: { source_run_id: "run_4" },
      expected_properties: ["Database remains private"],
    },
    status: "pending",
    created_at: "2026-07-20T10:00:00Z",
  }]);

  try {
    const proposals = await listToolImprovementProposals();
    assert.equal(proposals[0]?.correction, "Keep database traffic inside the private subnet.");
    assert.deepEqual(proposals[0]?.regression_eval.expected_properties, ["Database remains private"]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("creates an explicit pending memory proposal with provenance fields", async () => {
  const originalFetch = globalThis.fetch;
  let body: unknown;
  globalThis.fetch = async (_input, init) => {
    body = JSON.parse(String(init?.body));
    return jsonResponse({
      id: "mprop_1",
      kind: "project",
      content: "Use Chicago for OCI Responses.",
      source_run_id: "run_7",
      confidence: 1,
      status: "pending",
      created_at: "2026-07-22T10:00:00Z",
    }, 201);
  };
  try {
    const proposal = await createMemoryProposal("project", "Use Chicago for OCI Responses.", "run_7");
    assert.deepEqual(body, {
      kind: "project",
      content: "Use Chicago for OCI Responses.",
      source_run_id: "run_7",
    });
    assert.equal(proposal.status, "pending");
    assert.equal(proposal.source_run_id, "run_7");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("pins OCI provider and native tools in model preferences", async () => {
  const originalFetch = globalThis.fetch;
  let body: unknown;
  globalThis.fetch = async (_input, init) => {
    body = JSON.parse(String(init?.body));
    return jsonResponse({
      mode: "split",
      model: null,
      provider: "oci",
      oci_tools: ["code_interpreter", "x_search"],
      oci_available: true,
    });
  };
  try {
    const preference = await setModelPreference(
      "split",
      null,
      "oci",
      ["code_interpreter", "x_search"],
    );
    assert.deepEqual(body, {
      mode: "split",
      model: null,
      provider: "oci",
      oci_tools: ["code_interpreter", "x_search"],
    });
    assert.equal(preference.provider, "oci");
    assert.equal(preference.oci_available, true);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("normalizes and opens project workspaces with the selected Grok handoff", async () => {
  const originalFetch = globalThis.fetch;
  const calls: Array<{ url: string; body?: unknown }> = [];
  globalThis.fetch = async (input, init) => {
    calls.push({
      url: String(input),
      body: init?.body ? JSON.parse(String(init.body)) as unknown : undefined,
    });
    return jsonResponse(init?.method === "POST" ? {
      id: "asset_1234567890abcdef1234",
      name: "Demo",
      summary: "A project",
      framework: "Next.js",
      initialized: true,
      manifest_revision: 2,
      file_count: 48,
      metis_md_path: ".metis/METIS.md",
    } : [{
      id: "asset_1234567890abcdef1234",
      name: "Demo",
      summary: "A project",
      initialized: false,
      manifest_revision: 0,
      file_count: 0,
    }]);
  };
  try {
    const projects = await listProjectWorkspaces();
    assert.equal(projects[0]?.manifestRevision, 0);
    const opened = await openProjectWorkspace(
      "asset_1234567890abcdef1234",
      "grok_bootstrap_local",
    );
    assert.equal(opened.fileCount, 48);
    assert.deepEqual(calls[1]?.body, { mode: "grok_bootstrap_local" });
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("pins or clears project mode explicitly in chat messages", async () => {
  const originalFetch = globalThis.fetch;
  const bodies: unknown[] = [];
  globalThis.fetch = async (_input, init) => {
    bodies.push(JSON.parse(String(init?.body)) as unknown);
    return jsonResponse({ run_id: `run_${bodies.length}`, status: "queued" }, 202);
  };
  try {
    await sendMessage("conv_1", "Work here", [], {
      id: "asset_1234567890abcdef1234",
      mode: "grok_continuous",
    });
    await sendMessage("conv_1", "Leave project mode", [], null);
    assert.deepEqual(bodies[0], {
      content: "Work here",
      attachment_ids: [],
      knowledge_scope: "auto",
      project_id: "asset_1234567890abcdef1234",
      project_mode: "grok_continuous",
    });
    assert.deepEqual(bodies[1], {
      content: "Leave project mode",
      attachment_ids: [],
      knowledge_scope: "auto",
      project_id: null,
      project_mode: null,
    });
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("sends the Notion-only grounding scope per chat turn", async () => {
  const originalFetch = globalThis.fetch;
  let body: Record<string, unknown> = {};
  globalThis.fetch = async (_input, init) => {
    body = JSON.parse(String(init?.body)) as Record<string, unknown>;
    return jsonResponse({ run_id: "run_notion", status: "queued" }, 202);
  };
  try {
    await sendMessage("conv_1", "What did I decide?", [], undefined, "notion");
    assert.equal(body.knowledge_scope, "notion");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("normalizes the secret-safe Notion connection and manual sync result", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input) => {
    const url = String(input);
    if (url.endsWith("/sync")) {
      return jsonResponse({
        pages_fetched: 12,
        pages_written: 3,
        pages_removed: 1,
        source: {
          id: "src_notion",
          root_path: "/private/mirror",
          label: "Work Notion",
          kind: "notes",
          provider: "notion",
          consent: true,
          status: "indexed",
          file_count: 12,
          chunk_count: 44,
        },
        index_result: null,
        message: "Synced.",
      });
    }
    return jsonResponse({
      configured: true,
      token_configured: true,
      root_page_ids: ["11111111-2222-3333-4444-555555555555"],
      label: "Work Notion",
      page_count: 12,
      source: {
        id: "src_notion",
        root_path: "/private/mirror",
        label: "Work Notion",
        kind: "notes",
        provider: "notion",
        consent: true,
        status: "indexed",
        file_count: 12,
        chunk_count: 44,
      },
    });
  };
  try {
    const connection = await getNotionConnection();
    assert.equal(connection.source?.provider, "notion");
    assert.equal(connection.page_count, 12);
    const synced = await syncNotion();
    assert.equal(synced.pages_fetched, 12);
    assert.equal(synced.source.chunk_count, 44);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("submits a reasoned idempotent improvement decision without implying activation", async () => {
  const originalFetch = globalThis.fetch;
  const calls: Array<{ url: string; body?: unknown }> = [];
  globalThis.fetch = async (input, init) => {
    calls.push({
      url: String(input),
      body: init?.body ? JSON.parse(String(init.body)) as unknown : undefined,
    });
    return jsonResponse({
      proposal: {
        id: "timpr_1",
        source_run_id: "run_4",
        tool_id: "tool_2",
        tool_version_id: "version_8",
        content_hash: "deadbeef",
        correction: "Keep it private",
        regression_eval: { id: "reg_1", name: "Private", input: {}, expected_properties: ["Private"] },
        status: "approved",
        created_at: "2026-07-20T10:00:00Z",
        outcome: "revision_queued",
      },
      outcome: "revision_queued",
      revision_request: { id: "treq_1", status: "queued" },
    });
  };

  try {
    const result = await decideToolImprovement(
      "timpr_1",
      "approve",
      "stable-key-123",
      "Queue a tested revision",
    );
    assert.match(calls[0]!.url, /tool-improvement-proposals\/timpr_1\/decision$/);
    assert.deepEqual(calls[0]!.body, {
      decision: "approve",
      idempotency_key: "stable-key-123",
      reason: "Queue a tested revision",
    });
    assert.equal(result.outcome, "revision_queued");
    assert.equal(result.activated_version_id, undefined);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("loads verified source, manifest, evaluation, and revision evidence", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => jsonResponse({
    proposal: {
      id: "timpr_1",
      source_run_id: "run_4",
      tool_id: "tool_2",
      tool_version_id: "version_8",
      content_hash: "deadbeef",
      correction: "Keep it private",
      regression_eval: { id: "reg_1", name: "Private", input: {}, expected_properties: ["Private"] },
      status: "pending",
      created_at: "2026-07-20T10:00:00Z",
    },
    base_version: {
      tool_id: "tool_2",
      version_id: "version_8",
      state: "active",
      content_hash: "deadbeef",
      manifest: { version: "1.0.0", permissions: ["artifact:write"] },
      eval_report: { passed: true, score: 1 },
      bundle_verified: true,
      files: [{ path: "src/tool.py", sha256: "abc", size: 12, content: "print('ok')" }],
      evidence_truncated: false,
      source_diff: "",
    },
    eligible_revisions: [],
  });

  try {
    const evidence = await getToolImprovementEvidence("timpr_1");
    assert.equal(evidence.base_version.bundle_verified, true);
    assert.equal(evidence.base_version.files[0]?.path, "src/tool.py");
    assert.deepEqual(evidence.eligible_revisions, []);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("normalizes declarative tool definition records with their capability profile", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => jsonResponse([{
    definition: {
      slug: "architecture-diagram",
      version: "1.0.0",
      name: "Architecture Diagram",
      description: "Render a reference architecture.",
      archetype: "architecture",
      intent_examples: ["diagram my system"],
      input_contract: {},
      output_contract: {},
      route_facts: { existing_risk: "R2", factory_risk: "R3", input_pipeline: "architecture_spec" },
      capability_profile: {
        code_allowlist: "diagram-safe",
        runtime_allowlists: { python: "diagram" },
        model_access: {
          enabled: true,
          roles: ["coder"],
          max_calls_per_run: 3,
          max_tokens_per_call: 2048,
          prompt_templates: { spec: "..." },
        },
        filesystem: "run-io",
        network: "none",
        max_runtime_seconds: 150,
        max_artifact_bytes: 10000000,
      },
      status: "defined",
      content_hash: "abc123def456",
    },
    active: true,
    runnable: true,
    buildable: false,
    disabled: false,
    pending_definition_proposal: false,
    pending_build: false,
  }]);

  try {
    const records = await listToolDefinitions();
    assert.equal(records[0]?.definition.slug, "architecture-diagram");
    assert.equal(records[0]?.runnable, true);
    assert.equal(records[0]?.definition.capability_profile.model_access.enabled, true);
    assert.deepEqual(records[0]?.definition.capability_profile.model_access.roles, ["coder"]);
    assert.equal(records[0]?.definition.capability_profile.runtime_allowlists.python, "diagram");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("filters tool-definition proposals by status and omits the query when unset", async () => {
  const originalFetch = globalThis.fetch;
  const urls: string[] = [];
  globalThis.fetch = async (input) => {
    urls.push(String(input));
    return jsonResponse([{
      id: "tdp_1",
      definition_id: "def_1",
      slug: "text-summary",
      version: "1.0.0",
      status: "pending",
      risk_level: "R3",
      summary: "Summarize attached text.",
      created_at: "2026-07-20T10:00:00Z",
    }]);
  };

  try {
    const pending = await listToolDefinitionProposals("pending");
    assert.match(urls[0]!, /\/tool-definition-proposals\?status=pending$/);
    assert.equal(pending[0]?.slug, "text-summary");
    await listToolDefinitionProposals();
    assert.match(urls[1]!, /\/tool-definition-proposals$/);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
