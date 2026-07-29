"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { SelectMenu, type SelectOption } from "@/components/select-menu";
import { estimateDac, getDacCatalog, optimizeDac, recommendDac } from "@/lib/api";
import type {
  DacCatalog,
  DacConfidence,
  DacEstimate,
  DacOptimizeResult,
  DacRecommendation,
} from "@/lib/types";

/** Oracle's own benchmark scenarios, so a reader can compare like for like. */
const SCENARIOS: Array<{ id: string; label: string; prompt: number; response: number }> = [
  { id: "chat", label: "Chat", prompt: 100, response: 100 },
  { id: "random", label: "Random length", prompt: 480, response: 300 },
  { id: "generation", label: "Generation heavy", prompt: 100, response: 1000 },
  { id: "rag1", label: "RAG · 2K", prompt: 2000, response: 200 },
  { id: "rag2", label: "RAG · 7.8K", prompt: 7800, response: 200 },
  { id: "rag3", label: "RAG · 128K", prompt: 128000, response: 200 },
];

const CONCURRENCIES = [1, 2, 4, 8, 16, 32, 64, 128, 256];

function formatNumber(value: number, digits = 0): string {
  return value.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function formatSeconds(value: number): string {
  if (value < 1) return `${Math.round(value * 1000)} ms`;
  if (value < 60) return `${value.toFixed(2)} s`;
  return `${Math.floor(value / 60)}m ${Math.round(value % 60)}s`;
}

function formatParams(architecture: Record<string, unknown> | null): string {
  const total = architecture?.params_total;
  const active = architecture?.params_active;
  if (typeof total !== "number") return "—";
  const billions = (value: number) => `${(value / 1e9).toFixed(value < 1e10 ? 1 : 0)}B`;
  if (typeof active === "number" && active < total * 0.9) {
    return `${billions(total)} · ${billions(active)} active`;
  }
  return billions(total);
}

function ConfidenceBadge({ confidence }: { confidence: DacConfidence }) {
  const margin =
    confidence.error_margin != null && confidence.error_margin > 0
      ? ` ±${Math.round(confidence.error_margin * 100)}%`
      : "";
  const label =
    confidence.tier === "measured"
      ? "Measured"
      : confidence.tier === "interpolated"
        ? "Interpolated"
        : "Modeled";
  return (
    <span className={`dacConfidence dacConfidence-${confidence.tier}`} title={confidence.reason}>
      {label}
      {margin}
    </span>
  );
}

export function DacSizing() {
  const [catalog, setCatalog] = useState<DacCatalog | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [modelId, setModelId] = useState("");
  const [shape, setShape] = useState("");
  const [units, setUnits] = useState(1);
  const [scenario, setScenario] = useState("rag1");
  const [concurrency, setConcurrency] = useState(16);
  const [quantization, setQuantization] = useState("");
  const [kvQuantization, setKvQuantization] = useState("");
  const [rate, setRate] = useState(1);
  const [validatedOnly, setValidatedOnly] = useState(true);
  const [latencyTarget, setLatencyTarget] = useState<number | "">("");

  const [estimate, setEstimate] = useState<DacEstimate | null>(null);
  const [options, setOptions] = useState<DacOptimizeResult | null>(null);
  const [busy, setBusy] = useState(false);

  const [useCase, setUseCase] = useState("");
  const [recommendation, setRecommendation] = useState<DacRecommendation | null>(null);
  const [recommending, setRecommending] = useState(false);

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const loaded = await getDacCatalog();
        if (!mounted) return;
        setCatalog(loaded);
        const rateValue = (loaded.pricing?.price_per_ai_unit_hour as { value?: number } | undefined)
          ?.value;
        if (typeof rateValue === "number") setRate(rateValue);
        const first = loaded.models.find((model) => model.supported && model.validated_shapes.length);
        if (first) {
          setModelId(first.id);
          setShape(first.validated_shapes[0] ?? "");
        }
      } catch (loadError) {
        if (mounted) {
          setError(
            loadError instanceof Error ? loadError.message : "Could not load the sizing catalog.",
          );
        }
      } finally {
        if (mounted) setLoading(false);
      }
    })();
    return () => {
      mounted = false;
    };
  }, []);

  const models = useMemo(() => catalog?.models ?? [], [catalog]);
  const selected = useMemo(
    () => models.find((model) => model.id === modelId) ?? null,
    [models, modelId],
  );

  const modelOptions = useMemo<SelectOption[]>(
    () =>
      [...models]
        .sort((a, b) => a.family.localeCompare(b.family) || a.id.localeCompare(b.id))
        .map((model) => ({
          value: model.id,
          label: model.id.split("/").pop() ?? model.id,
          hint: model.supported ? formatParams(model.architecture) : "no data",
          group: model.family,
          disabled: !model.supported,
        })),
    [models],
  );

  const quantizationOptions = useMemo<SelectOption[]>(
    () => (catalog?.quantizations ?? []).map((name) => ({ value: name, label: name })),
    [catalog],
  );

  // The shape key already spells out its GPU and count (A100_80G_X2 is two
  // 80GB A100s), so the label is just the key. The hint carries total memory
  // instead — the one fact a reader needs that the name does not already say.
  //
  // Benchmarked-but-not-importable shapes are kept. Oracle's only published
  // measurements run on an OpenAI-reserved shape, so filtering every
  // non-importable shape would make the one tier backed by real measurements
  // impossible to select — the tab would model numbers it could have quoted.
  const shapeOptions = useMemo<SelectOption[]>(() => {
    if (!catalog) return [];
    const validated = new Set(selected?.validated_shapes ?? []);
    const benchmarked = new Set(selected?.benchmarked_shapes ?? []);
    const group = (key: string) =>
      benchmarked.has(key)
        ? "Benchmarked by Oracle"
        : validated.has(key)
          ? "Validated for this model"
          : "Other shapes";
    const rank = (key: string) =>
      benchmarked.has(key) ? 0 : validated.has(key) ? 1 : 2;

    return catalog.shapes
      .filter((item) => item.importable || benchmarked.has(item.key))
      .sort((a, b) =>
        rank(a.key) === rank(b.key)
          ? a.ai_units - b.ai_units
          : rank(a.key) - rank(b.key),
      )
      .map((item) => ({
        value: item.key,
        label: item.key,
        hint: `${formatNumber(item.total_memory_gb)} GB`,
        group: group(item.key),
      }));
  }, [catalog, selected]);

  // Switching model picks a sensible default shape for it. Keyed on the model
  // alone: an earlier version also depended on the current shape and reset any
  // value outside the validated list, which made every unvalidated shape
  // unselectable — the control snapped back the instant it was changed.
  const lastModelId = useRef<string | null>(null);
  useEffect(() => {
    if (!selected || selected.id === lastModelId.current) return;
    lastModelId.current = selected.id;
    if (selected.validated_shapes.length && !selected.validated_shapes.includes(shape)) {
      setShape(selected.validated_shapes[0] ?? "");
    }
  }, [selected, shape]);

  const activeScenario = SCENARIOS.find((item) => item.id === scenario) ?? SCENARIOS[3]!;

  // Every control change fires a fresh estimate, so several can be in flight at
  // once and they do not necessarily resolve in order. Without this guard a
  // slower earlier request lands last and overwrites the current answer — most
  // visibly by restoring an error banner for a configuration already replaced.
  const requestSeq = useRef(0);

  const run = useCallback(async () => {
    if (!modelId || !shape || !selected?.supported) return;
    const sequence = ++requestSeq.current;
    setBusy(true);
    setError(null);
    try {
      const shared = {
        model_id: modelId,
        prompt_tokens: activeScenario.prompt,
        response_tokens: activeScenario.response,
        concurrency,
        quantization: quantization || null,
        price_per_ai_unit_hour: rate,
      };
      const [next, ranked] = await Promise.all([
        estimateDac({
          ...shared,
          shape,
          units,
          kv_quantization: kvQuantization || null,
        }),
        optimizeDac({
          ...shared,
          validated_only: validatedOnly,
          max_request_latency_s: latencyTarget === "" ? null : Number(latencyTarget),
        }),
      ]);
      if (sequence !== requestSeq.current) return;
      setEstimate(next);
      setOptions(ranked);
    } catch (runError) {
      if (sequence !== requestSeq.current) return;
      setError(runError instanceof Error ? runError.message : "Could not compute an estimate.");
    } finally {
      if (sequence === requestSeq.current) setBusy(false);
    }
  }, [
    activeScenario,
    concurrency,
    kvQuantization,
    latencyTarget,
    modelId,
    quantization,
    rate,
    selected,
    shape,
    units,
    validatedOnly,
  ]);

  useEffect(() => {
    if (catalog && modelId && shape) void run();
  }, [catalog, modelId, shape, run]);

  async function askRecommendation() {
    const text = useCase.trim();
    if (!text || recommending) return;
    setRecommending(true);
    setError(null);
    try {
      setRecommendation(
        await recommendDac({
          use_case: text,
          concurrency,
          prompt_tokens: activeScenario.prompt,
          response_tokens: activeScenario.response,
          max_request_latency_s: latencyTarget === "" ? null : Number(latencyTarget),
        }),
      );
    } catch (recommendError) {
      setError(
        recommendError instanceof Error
          ? recommendError.message
          : "Could not produce a recommendation.",
      );
    } finally {
      setRecommending(false);
    }
  }

  const provenance = catalog?.provenance as
    | {
        models?: { generated_at?: string; count?: number; with_architecture?: number };
        benchmarks?: { rows?: number; calibration_grids?: number };
        calibration?: {
          decode_median_error?: number;
          decode_p90_error?: number;
          ttft_median_error?: number;
          gpus?: string[];
          sample_rows?: number;
        };
        pricing?: { rate_verified?: boolean; label?: string };
      }
    | undefined;

  const vram = estimate?.vram;
  const segments = vram
    ? [
        { key: "weights", label: "Weights", value: vram.weights_gb },
        { key: "kv", label: "KV cache", value: vram.kv_cache_gb },
        { key: "act", label: "Activations", value: vram.activations_gb },
        { key: "overhead", label: "Framework", value: vram.overhead_gb },
      ]
    : [];

  return (
    <div className="workspacePage dacPage">
      <header className="pageHeader">
        <div>
          <span className="eyebrow">Dedicated AI Cluster</span>
          <h1>Sizing</h1>
          <p>
            Pick a shape, predict throughput and latency, and price it — for every model Oracle
            validates for import, not just the ones it publishes benchmarks for.
          </p>
        </div>
        <button className="secondaryButton" type="button" onClick={() => void run()} disabled={busy}>
          {busy ? "Computing…" : "Recompute"}
        </button>
      </header>

      {error ? (
        <div className="notice errorNotice" role="alert">
          <strong>Sizing unavailable</strong>
          <span>{error}</span>
        </div>
      ) : null}

      <section className="dacRecommend" aria-labelledby="dac-recommend-title">
        <div>
          <span className="eyebrow">Start from the problem</span>
          <h2 id="dac-recommend-title">What are you building?</h2>
          <p>
            Describe the workload. Metis ranks the models that fit a validated shape at your load,
            then asks your configured model to judge which suits the task.
          </p>
        </div>
        <textarea
          value={useCase}
          onChange={(event) => setUseCase(event.target.value)}
          maxLength={4000}
          rows={3}
          placeholder="For example: a coding assistant for 20 engineers working on internal Python services."
        />
        <div className="dacRecommendActions">
          <button
            className="primaryButton"
            type="button"
            disabled={useCase.trim().length < 8 || recommending}
            onClick={() => void askRecommendation()}
          >
            {recommending ? "Thinking…" : "Recommend a model"}
          </button>
          {recommendation?.model_backed ? (
            <span className="dacHint">Ranked by {recommendation.model_used}</span>
          ) : null}
        </div>

        {recommendation ? (
          <div className="dacCandidates">
            {recommendation.summary ? <p className="dacSummary">{recommendation.summary}</p> : null}
            {recommendation.candidates.map((candidate, index) => (
              <article className="dacCandidate" key={candidate.model_id}>
                <div className="dacCandidateHead">
                  <span className="dacRank">{index + 1}</span>
                  <div>
                    <strong>{candidate.model_id}</strong>
                    <small>
                      {candidate.family} · {candidate.capability.replaceAll("_", " ").toLowerCase()}
                    </small>
                  </div>
                  <button
                    className="secondaryButton"
                    type="button"
                    onClick={() => {
                      setModelId(candidate.model_id);
                      if (candidate.shape) setShape(candidate.shape);
                      setUnits(candidate.units);
                    }}
                  >
                    Size this
                  </button>
                </div>
                {candidate.rationale ? <p>{candidate.rationale}</p> : null}
                <dl className="dacCandidateMeta">
                  <div>
                    <dt>Shape</dt>
                    <dd className="mono">
                      {candidate.shape} × {candidate.units}
                    </dd>
                  </div>
                  <div>
                    <dt>Speed</dt>
                    <dd>
                      {candidate.performance
                        ? `${formatNumber(candidate.performance.inference_speed_tps)} tok/s`
                        : "—"}
                    </dd>
                  </div>
                  <div>
                    <dt>Unit-hours / mo</dt>
                    <dd>{candidate.cost ? formatNumber(candidate.cost.unit_hours) : "—"}</dd>
                  </div>
                </dl>
              </article>
            ))}
            {recommendation.notes.map((note) => (
              <p className="dacNote" key={note}>
                {note}
              </p>
            ))}
          </div>
        ) : null}
      </section>

      <section className="dacConfigure" aria-labelledby="dac-configure-title">
        <h2 id="dac-configure-title" className="visuallyHidden">
          Configure
        </h2>
        <div className="dacControls">
          <SelectMenu
            label="Model"
            value={modelId}
            options={modelOptions}
            onChange={setModelId}
            emptyMessage="No model matches that filter"
          />

          <SelectMenu
            label="Unit shape"
            value={shape}
            options={shapeOptions}
            onChange={setShape}
          />

          <label className="dacNumberField">
            Units
            <input
              type="number"
              min={1}
              max={16}
              value={units}
              onChange={(event) => setUnits(Math.max(1, Number(event.target.value) || 1))}
            />
          </label>

          <SelectMenu
            label="Scenario"
            value={scenario}
            options={SCENARIOS.map((item) => ({
              value: item.id,
              label: item.label,
              hint: `${formatNumber(item.prompt)} in / ${formatNumber(item.response)} out`,
            }))}
            onChange={setScenario}
          />

          <SelectMenu
            label="Concurrency"
            value={String(concurrency)}
            options={CONCURRENCIES.map((item) => ({
              value: String(item),
              label: String(item),
            }))}
            onChange={(next) => setConcurrency(Number(next))}
          />

          <SelectMenu
            label="Weights"
            value={quantization}
            options={[{ value: "", label: "As published" }, ...quantizationOptions]}
            onChange={setQuantization}
          />

          <SelectMenu
            label="KV cache"
            value={kvQuantization}
            options={[{ value: "", label: "Match weights" }, ...quantizationOptions]}
            onChange={setKvQuantization}
          />

          <label className="dacNumberField">
            Latency target (s)
            <input
              type="number"
              min={0}
              step={0.5}
              value={latencyTarget}
              placeholder="none"
              onChange={(event) =>
                setLatencyTarget(event.target.value === "" ? "" : Number(event.target.value))
              }
            />
          </label>

          <label className="dacNumberField">
            $ / AI unit-hour
            <input
              type="number"
              min={0}
              step={0.01}
              value={rate}
              onChange={(event) => setRate(Math.max(0, Number(event.target.value) || 0))}
            />
          </label>
        </div>

        {selected ? (
          <p className="dacHint">
            {formatParams(selected.architecture)} ·{" "}
            {String(selected.architecture?.attention_type ?? "—").toUpperCase()} ·{" "}
            {selected.validated_shapes.length
              ? `Oracle validates ${selected.validated_shapes.join(", ")}`
              : "no validated shape published"}
            {selected.config_source && !selected.config_source.startsWith(selected.id.split("/")[0]!)
              ? ` · architecture read from ${selected.config_source}`
              : ""}
          </p>
        ) : null}
        {selected && !selected.supported ? (
          <div className="notice" role="status">
            <strong>Not modelled</strong>
            <span>{selected.unsupported_reason}</span>
          </div>
        ) : null}
      </section>

      {estimate && vram ? (
        <section className="dacResults" aria-labelledby="dac-results-title">
          <div className="dacResultsHead">
            <h2 id="dac-results-title">
              {estimate.shape} × {estimate.units}
            </h2>
            <ConfidenceBadge confidence={estimate.confidence} />
            {estimate.oracle_validated ? (
              <span className="dacBadge dacBadge-ok">Oracle validated</span>
            ) : (
              <span className="dacBadge dacBadge-warn">Not validated for this model</span>
            )}
          </div>

          <div className="dacVram">
            <div className="dacVramHead">
              <strong>
                {vram.total_gb.toFixed(1)} GB of {formatNumber(vram.capacity_gb)} GB
              </strong>
              <span className={`dacStatus dacStatus-${vram.status}`}>
                {vram.status.replaceAll("_", " ")}
              </span>
            </div>
            <div className="dacVramBar" role="img" aria-label={`${Math.round(vram.utilization * 100)}% of VRAM used`}>
              {segments.map((segment) => (
                <span
                  key={segment.key}
                  className={`dacVramSegment dacVramSegment-${segment.key}`}
                  style={{ width: `${Math.max(0, (segment.value / vram.capacity_gb) * 100)}%` }}
                  title={`${segment.label}: ${segment.value.toFixed(2)} GB`}
                />
              ))}
            </div>
            <dl className="dacVramLegend">
              {segments.map((segment) => (
                <div key={segment.key}>
                  <dt>
                    <span className={`dacSwatch dacVramSegment-${segment.key}`} aria-hidden="true" />
                    {segment.label}
                  </dt>
                  <dd>{segment.value.toFixed(2)} GB</dd>
                </div>
              ))}
              <div>
                <dt>Fits concurrent</dt>
                <dd>{formatNumber(vram.max_concurrency * estimate.units)}</dd>
              </div>
            </dl>
          </div>

          <div className="dacMetrics">
            {[
              { label: "Time to first token", value: formatSeconds(estimate.performance.ttft_s) },
              {
                label: "Speed per user",
                value: `${formatNumber(estimate.performance.inference_speed_tps)} tok/s`,
              },
              {
                label: "Token throughput",
                value: `${formatNumber(estimate.performance.token_throughput_tps)} tok/s`,
              },
              {
                label: "Request latency",
                value: formatSeconds(estimate.performance.request_latency_s),
              },
              {
                label: "Requests / min",
                value: formatNumber(estimate.performance.request_throughput_rpm, 1),
              },
              {
                label: "Total throughput",
                value: `${formatNumber(estimate.performance.total_throughput_tps)} tok/s`,
              },
            ].map((metric) => (
              <div className="dacMetric" key={metric.label}>
                <span>{metric.label}</span>
                <strong>{metric.value}</strong>
              </div>
            ))}
          </div>

          <div className="dacCost">
            <div>
              <span>AI unit-hours</span>
              <strong>{formatNumber(estimate.cost.unit_hours)}</strong>
              <small>
                {estimate.cost.ai_units_per_unit} AI units × {estimate.units} ×{" "}
                {formatNumber(estimate.cost.hours)} h — exact
              </small>
            </div>
            <div>
              <span>Estimated cost</span>
              <strong>${formatNumber(estimate.cost.cost, 2)}</strong>
              <small>
                {provenance?.pricing?.rate_verified
                  ? "at the published rate"
                  : "at your entered rate — Oracle does not publish this rate on a readable page"}
              </small>
            </div>
          </div>

          {estimate.published ? (
            <p className="dacNote">
              Oracle publishes this exact configuration:{" "}
              {formatNumber(Number(estimate.published.inference_speed_tps ?? 0))} tok/s at{" "}
              {formatSeconds(Number(estimate.published.ttft_s ?? 0))} TTFT.
            </p>
          ) : null}
          {estimate.notes.map((note) => (
            <p className="dacNote" key={note}>
              {note}
            </p>
          ))}
        </section>
      ) : null}

      {options && options.options.length ? (
        <section className="dacOptimizer" aria-labelledby="dac-optimizer-title">
          <div className="dacResultsHead">
            <h2 id="dac-optimizer-title">Cheapest configuration that meets the target</h2>
            <label className="dacToggle">
              <input
                type="checkbox"
                checked={validatedOnly}
                onChange={(event) => setValidatedOnly(event.target.checked)}
              />
              Validated shapes only
            </label>
          </div>
          <div className="dacTableScroll">
            <table className="dacTable">
              <thead>
                <tr>
                  <th scope="col">Shape</th>
                  <th scope="col">Units</th>
                  <th scope="col">Unit-hours</th>
                  <th scope="col">Cost</th>
                  <th scope="col">TTFT</th>
                  <th scope="col">Latency</th>
                  <th scope="col">Req/min</th>
                  <th scope="col">VRAM</th>
                  <th scope="col">Target</th>
                </tr>
              </thead>
              <tbody>
                {options.options.map((option) => (
                  <tr
                    key={`${option.shape}-${option.units}`}
                    className={option.meets_sla ? "" : "dacRowMiss"}
                  >
                    <th scope="row">
                      <span className="mono">{option.shape}</span>
                      {option.oracle_validated ? (
                        <span className="dacTick" title="Validated by Oracle for this model">
                          ✓
                        </span>
                      ) : null}
                    </th>
                    <td>{option.units}</td>
                    <td>{formatNumber(option.cost.unit_hours)}</td>
                    <td>${formatNumber(option.cost.cost, 0)}</td>
                    <td>{formatSeconds(option.performance.ttft_s)}</td>
                    <td>{formatSeconds(option.performance.request_latency_s)}</td>
                    <td>{formatNumber(option.performance.request_throughput_rpm, 1)}</td>
                    <td>{Math.round(option.vram.utilization * 100)}%</td>
                    <td>
                      {option.meets_sla ? (
                        <span className="dacBadge dacBadge-ok">meets</span>
                      ) : (
                        <span className="dacBadge dacBadge-warn" title={option.unmet.join("; ")}>
                          {option.unmet[0] ?? "misses"}
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {options.notes.map((note) => (
            <p className="dacNote" key={note}>
              {note}
            </p>
          ))}
        </section>
      ) : null}

      {provenance ? (
        <section className="dacProvenance" aria-labelledby="dac-provenance-title">
          <span className="principleMark">◈</span>
          <div>
            <strong id="dac-provenance-title">Where these numbers come from</strong>
            <p>
              Shapes and validated-model lists are copied from Oracle&apos;s documentation
              ({provenance.models?.with_architecture ?? 0} of {provenance.models?.count ?? 0} models
              have published architecture, read from Hugging Face). The performance model is a
              roofline calibrated against {provenance.benchmarks?.rows ?? 0} published benchmark
              rows, of which only{" "}
              {(provenance.calibration?.sample_rows ?? 0).toLocaleString()} name the GPUs they ran
              on and can be used to fit it.
            </p>
            <p className="dacRationale">
              Against those rows the model&apos;s median error is{" "}
              {Math.round((provenance.calibration?.decode_median_error ?? 0) * 100)}% on decode speed
              and {Math.round((provenance.calibration?.ttft_median_error ?? 0) * 100)}% on time to
              first token, with a 90th-percentile decode error of{" "}
              {Math.round((provenance.calibration?.decode_p90_error ?? 0) * 100)}%. Both calibration
              models are mixture-of-experts on {(provenance.calibration?.gpus ?? []).join(", ")},
              so dense models and other GPUs are extrapolation — which is what the
              &ldquo;modeled&rdquo; badge means. Nothing here calls out to the network; run{" "}
              <code>make dac-catalog</code> to refresh it.
            </p>
          </div>
        </section>
      ) : null}

      {loading ? <div className="skeletonRow" /> : null}
    </div>
  );
}
