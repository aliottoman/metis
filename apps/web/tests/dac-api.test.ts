import assert from "node:assert/strict";
import test from "node:test";

import { estimateDac, getDacCatalog, optimizeDac, recommendDac } from "../lib/api.ts";

function jsonResponse(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "content-type": "application/json" },
  });
}

async function withFetch<T>(
  handler: (url: string, init?: RequestInit) => Response,
  run: () => Promise<T>,
): Promise<T> {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit) =>
    handler(String(input), init);
  try {
    return await run();
  } finally {
    globalThis.fetch = originalFetch;
  }
}

const VRAM = {
  weights_gb: 61.02,
  kv_cache_gb: 8.59,
  activations_gb: 0.35,
  overhead_gb: 2,
  total_gb: 71.97,
  capacity_gb: 160,
  usable_gb: 144,
  utilization: 0.4498,
  status: "okay",
  fits: true,
  max_concurrency: 150,
};

const PERFORMANCE = {
  ttft_s: 0.36,
  inference_speed_tps: 34.2,
  token_throughput_tps: 210.5,
  request_latency_s: 6.2,
  request_throughput_rps: 1.05,
  request_throughput_rpm: 63.15,
  total_throughput_tps: 2315.5,
  concurrency: 16,
  prompt_tokens: 2000,
  response_tokens: 200,
};

test("catalog carries validated shapes and calibration provenance through", async () => {
  const catalog = await withFetch(
    () =>
      jsonResponse({
        models: [
          {
            id: "Qwen/Qwen3-32B",
            family: "Alibaba Qwen",
            capability: "TEXT_TO_TEXT",
            validated_shapes: ["A100_80G_X2"],
            supported: true,
            unsupported_reason: null,
            config_source: "Qwen/Qwen3-32B",
            architecture: { params_total: 32_800_000_000, attention_type: "gqa" },
          },
        ],
        shapes: [
          {
            key: "A100_80G_X2",
            gpu: "A100_80G",
            gpu_count: 2,
            ai_units: 6.48,
            total_memory_gb: 160,
            importable: true,
          },
        ],
        gpus: [
          {
            key: "A100_80G",
            label: "A100 80GB",
            memory_gb: 80,
            memory_bandwidth_gb_s: 2039,
            dense_bf16_tflops: 312,
            dense_fp8_tflops: null,
            supports_fp8: false,
          },
        ],
        quantizations: ["bf16", "fp8"],
        pricing: { price_per_ai_unit_hour: { value: 1, verified: false } },
        provenance: { calibration: { fitted: true, decode_median_error: 0.125 } },
      }),
    () => getDacCatalog(),
  );

  assert.equal(catalog.models[0]?.validated_shapes[0], "A100_80G_X2");
  assert.equal(catalog.shapes[0]?.ai_units, 6.48);
  assert.equal(
    (catalog.provenance.calibration as { decode_median_error: number }).decode_median_error,
    0.125,
  );
});

test("estimate posts the configuration and returns the confidence tier intact", async () => {
  let captured: unknown;
  const estimate = await withFetch(
    (url, init) => {
      assert.match(url, /\/api\/v1\/dac\/estimate$/);
      assert.equal(init?.method, "POST");
      captured = JSON.parse(String(init?.body));
      return jsonResponse({
        model_id: "Qwen/Qwen3-32B",
        shape: "A100_80G_X2",
        units: 1,
        oracle_validated: true,
        minimum_shape: "A100_80G_X1",
        vram: VRAM,
        performance: PERFORMANCE,
        cost: {
          ai_units_per_unit: 6.48,
          units: 1,
          hours: 744,
          unit_hours: 4821.12,
          billed_unit_hours: 4821.12,
          minimum_unit_hours: 0,
          cost: 4821.12,
        },
        confidence: {
          tier: "modeled",
          error_margin: 0.225,
          reason: "No published benchmark names this GPU.",
        },
        published: null,
        notes: ["The model fits on A100_80G_X1."],
      });
    },
    () =>
      estimateDac({
        model_id: "Qwen/Qwen3-32B",
        shape: "A100_80G_X2",
        concurrency: 16,
        prompt_tokens: 2000,
        response_tokens: 200,
      }),
  );

  assert.deepEqual(captured, {
    model_id: "Qwen/Qwen3-32B",
    shape: "A100_80G_X2",
    concurrency: 16,
    prompt_tokens: 2000,
    response_tokens: 200,
  });
  assert.equal(estimate.confidence.tier, "modeled");
  assert.equal(estimate.confidence.error_margin, 0.225);
  // The minimum shape differing from the validated one is the whole point of
  // showing both, so it must survive the round trip.
  assert.equal(estimate.minimum_shape, "A100_80G_X1");
  assert.equal(estimate.vram.status, "okay");
  assert.equal(estimate.cost.unit_hours, 4821.12);
});

test("optimize keeps option ordering and the reasons a configuration missed", async () => {
  const result = await withFetch(
    () =>
      jsonResponse({
        model_id: "Qwen/Qwen3-32B",
        options: [
          {
            shape: "A100_80G_X2",
            gpu: "A100_80G",
            gpu_count: 2,
            units: 2,
            oracle_validated: true,
            vram: VRAM,
            performance: PERFORMANCE,
            cost: {
              ai_units_per_unit: 6.48,
              units: 2,
              hours: 744,
              unit_hours: 9642.24,
              billed_unit_hours: 9642.24,
              minimum_unit_hours: 0,
              cost: 9642.24,
            },
            meets_sla: true,
            unmet: [],
          },
          {
            shape: "A100_80G_X2",
            gpu: "A100_80G",
            gpu_count: 2,
            units: 1,
            oracle_validated: true,
            vram: VRAM,
            performance: PERFORMANCE,
            cost: {
              ai_units_per_unit: 6.48,
              units: 1,
              hours: 744,
              unit_hours: 4821.12,
              billed_unit_hours: 4821.12,
              minimum_unit_hours: 0,
              cost: 4821.12,
            },
            meets_sla: false,
            unmet: ["latency 15.20s > 12.00s"],
          },
        ],
        confidence: { tier: "modeled", error_margin: 0.225, reason: "Extrapolated." },
        considered: 2,
        notes: ["Showing only shapes Oracle validated for this model."],
      }),
    () => optimizeDac({ model_id: "Qwen/Qwen3-32B", max_request_latency_s: 12 }),
  );

  assert.equal(result.options.length, 2);
  assert.equal(result.options[0]?.meets_sla, true);
  assert.equal(result.options[1]?.unmet[0], "latency 15.20s > 12.00s");
  assert.equal(result.notes.length, 1);
});

test("recommendation reports when it was ranked without a model call", async () => {
  const offline = await withFetch(
    () =>
      jsonResponse({
        use_case: "coding assistant",
        candidates: [
          {
            model_id: "Qwen/Qwen2.5-Coder-32B-Instruct",
            family: "Alibaba Qwen",
            capability: "TEXT_TO_TEXT",
            score: 6.79,
            shape: "A100_80G_X2",
            units: 1,
            performance: PERFORMANCE,
            cost: null,
            meets_sla: true,
            rationale: null,
          },
        ],
        summary: null,
        model_used: null,
        model_backed: false,
        notes: ["Ranked without a model call — sizing and cost are exact."],
      }),
    () => recommendDac({ use_case: "coding assistant" }),
  );

  assert.equal(offline.model_backed, false);
  assert.equal(offline.candidates[0]?.rationale, null);
  assert.match(offline.notes[0] ?? "", /without a model call/);
});

test("surfaces a rejected configuration as an error rather than a blank result", async () => {
  await withFetch(
    () => jsonResponse({ detail: "unknown model 'nope/none'" }, 400),
    async () => {
      await assert.rejects(
        () => estimateDac({ model_id: "nope/none", shape: "H100_X1" }),
        (error: Error) => error.name === "ApiError",
      );
    },
  );
});

test("renders a field-level validation failure instead of [object Object]", async () => {
  // FastAPI reports 422s as an array of objects; the generic string coercion
  // turns that into "[object Object]", which tells a user nothing about which
  // input was rejected.
  await withFetch(
    () =>
      jsonResponse(
        {
          detail: [
            {
              type: "greater_than_equal",
              loc: ["body", "concurrency"],
              msg: "Input should be greater than or equal to 1",
              input: 0,
            },
          ],
        },
        422,
      ),
    async () => {
      await assert.rejects(
        () => estimateDac({ model_id: "Qwen/Qwen3-32B", shape: "A100_80G_X2", concurrency: 0 }),
        (error: Error) => {
          assert.equal(error.message, "concurrency: Input should be greater than or equal to 1");
          return true;
        },
      );
    },
  );
});
