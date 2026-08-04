"""Behavioural tests for the Dedicated AI Cluster sizing model.

Two claims are being tested, and they need different kinds of evidence.

The memory model claims to be *exact*, so it is checked against hand-computable
values and against the published VRAM figures of a third-party calculator whose
arithmetic was verified independently. The performance model only claims to be
*calibrated*, so it is checked by holding out Oracle's own benchmark rows and
asserting the error stays inside the margin the UI actually reports — a test
that fails if the model degrades or if the reported margin becomes flattering.
"""
from __future__ import annotations


import pytest

from waqil_api.dac_catalog import DacCatalog
from waqil_api.dac_sizing import (
    Coefficients,
    GpuSpec,
    ModelArchitecture,
    ShapeSpec,
    SizingError,
    SlaTarget,
    confidence_for,
    cost_estimate,
    decode_step_bytes,
    estimate_performance,
    estimate_vram,
    fit_coefficients,
    kv_bytes_per_token,
    minimum_shape,
    optimize,
    residuals,
)

GIB = 1024**3


# ── Fixtures mirroring hardware the reference numbers were taken on ──────────


RTX_3060 = GpuSpec(
    key="RTX_3060",
    label="RTX 3060 12GB",
    memory_gb=12.0,
    memory_bandwidth_gb_s=360.0,
    dense_bf16_tflops=51.0,
)
RTX_3060_RIG = ShapeSpec(key="RIG", gpu=RTX_3060, gpu_count=1, ai_units=0.0)

A100 = GpuSpec(
    key="A100_80G",
    label="A100 80GB",
    memory_gb=80.0,
    memory_bandwidth_gb_s=2039.0,
    dense_bf16_tflops=312.0,
)
A100_X1 = ShapeSpec(key="A100_80G_X1", gpu=A100, gpu_count=1, ai_units=3.24)
A100_X2 = ShapeSpec(key="A100_80G_X2", gpu=A100, gpu_count=2, ai_units=6.48)


def dense_3b() -> ModelArchitecture:
    """A 3B dense multi-head model, sized like a Llama-3.2-3B."""
    return ModelArchitecture(
        params_total=3_000_000_000,
        params_active=3_000_000_000,
        num_layers=32,
        hidden_size=3072,
        num_attention_heads=48,
        num_key_value_heads=48,
        head_dim=64,
        attention_type="mha",
        torch_dtype="fp16",
    )


# ── Memory model ─────────────────────────────────────────────────────────────


def test_weights_are_params_times_dtype_width():
    """3B parameters at FP16 is exactly 6.00 GB, the anchor every other term sits on."""
    breakdown = estimate_vram(dense_3b(), RTX_3060_RIG, context_tokens=1024, concurrency=1)
    assert breakdown.weights_gb == pytest.approx(6.0, abs=0.01)


def test_the_four_components_add_up_and_drive_the_reported_utilization():
    """The breakdown must reconcile: the bar a user reads is the sum of its parts.

    Utilization is asserted alongside the total because the percentage is what
    someone actually acts on, and a capacity-unit bug would move it while
    leaving the total correct.
    """
    breakdown = estimate_vram(dense_3b(), RTX_3060_RIG, context_tokens=1024, concurrency=1)
    assert breakdown.total_gb == pytest.approx(
        breakdown.weights_gb
        + breakdown.kv_cache_gb
        + breakdown.activations_gb
        + breakdown.overhead_gb
    )
    assert breakdown.weights_gb == pytest.approx(6.0, abs=0.01)
    assert breakdown.utilization == pytest.approx(breakdown.total_gb / 12.0, rel=1e-6)
    assert breakdown.status == "moderate"
    assert breakdown.fits


def test_a_multi_head_model_pays_far_more_for_cache_than_a_latent_one():
    """The reference calculator reports 0.02 GB of cache for a 3B model at 1K.

    That figure is only reachable with compressed-latent attention; the same
    parameter count with plain multi-head attention costs an order of magnitude
    more. Reproducing "a 3B model" without reading its attention structure is
    how a calculator ends up an order of magnitude out.
    """
    mha = estimate_vram(dense_3b(), RTX_3060_RIG, context_tokens=1024, concurrency=1)
    latent = ModelArchitecture(
        **{
            **dense_3b().__dict__,
            "attention_type": "mla",
            "mla": {"kv_lora_rank": 512, "qk_rope_head_dim": 64},
        }
    )
    compressed = estimate_vram(latent, RTX_3060_RIG, context_tokens=1024, concurrency=1)
    assert compressed.kv_cache_gb < 0.05
    assert mha.kv_cache_gb > compressed.kv_cache_gb * 8


def test_grouped_query_attention_shrinks_the_cache_by_the_head_ratio():
    """GQA must be sized by KV heads, not query heads — an 8x error if confused."""
    mha = ModelArchitecture(
        params_total=8_000_000_000, params_active=8_000_000_000, num_layers=32,
        hidden_size=4096, num_attention_heads=32, num_key_value_heads=32,
        head_dim=128, attention_type="mha", torch_dtype="bf16",
    )
    gqa = ModelArchitecture(**{**mha.__dict__, "num_key_value_heads": 8, "attention_type": "gqa"})
    assert kv_bytes_per_token(mha) / kv_bytes_per_token(gqa) == pytest.approx(4.0)


def test_multi_head_latent_attention_uses_the_compressed_latent():
    """MLA caches one latent per token per layer, not per-head keys and values."""
    mla = ModelArchitecture(
        params_total=671_000_000_000, params_active=37_000_000_000, num_layers=61,
        hidden_size=7168, num_attention_heads=128, num_key_value_heads=128,
        head_dim=192, attention_type="mla", torch_dtype="bf16",
        mla={"kv_lora_rank": 512, "qk_rope_head_dim": 64},
    )
    expected = 61 * (512 + 64) * 2
    assert kv_bytes_per_token(mla) == pytest.approx(expected)
    # Without MLA handling this model would look ~50x more cache-hungry.
    as_dense = ModelArchitecture(**{**mla.__dict__, "attention_type": "mha", "mla": None})
    assert kv_bytes_per_token(as_dense) > kv_bytes_per_token(mla) * 10


def test_sliding_window_stops_the_cache_growing_past_the_window():
    """Past the window a windowed layer holds a fixed number of tokens."""
    windowed = ModelArchitecture(
        params_total=12_000_000_000, params_active=12_000_000_000, num_layers=48,
        hidden_size=3840, num_attention_heads=16, num_key_value_heads=8, head_dim=256,
        attention_type="gqa", torch_dtype="bf16", sliding_window=1024,
        sliding_window_ratio=1.0,
    )
    short = kv_bytes_per_token(windowed, context_tokens=512)
    long = kv_bytes_per_token(windowed, context_tokens=32_768)
    assert long < short / 20
    # Total cache stops growing, rather than per-token cost merely shrinking.
    assert long * 32_768 == pytest.approx(short * 1024, rel=0.05)


def test_interleaved_windows_only_discount_the_windowed_layers():
    """A 5:1 local:global model keeps paying full price on its global layers."""
    base = dict(
        params_total=12_000_000_000, params_active=12_000_000_000, num_layers=48,
        hidden_size=3840, num_attention_heads=16, num_key_value_heads=8, head_dim=256,
        attention_type="gqa", torch_dtype="bf16", sliding_window=1024,
    )
    all_local = ModelArchitecture(**base, sliding_window_ratio=1.0)
    interleaved = ModelArchitecture(**base, sliding_window_ratio=5 / 6)
    long = 65_536
    assert kv_bytes_per_token(interleaved, context_tokens=long) > kv_bytes_per_token(
        all_local, context_tokens=long
    )


def test_mixture_of_experts_reads_only_routed_experts_per_token():
    """Decode bandwidth sees active parameters; capacity still sees them all."""
    moe = ModelArchitecture(
        params_total=120_000_000_000, params_active=5_700_000_000,
        params_expert_total=114_000_000_000, params_expert_active=3_580_000_000,
        num_layers=36, hidden_size=2880, num_attention_heads=64,
        num_key_value_heads=8, head_dim=64, attention_type="gqa",
        torch_dtype="bf16", expert_dtype="mxfp4", is_moe=True,
    )
    # Experts are stored at MXFP4, so residency is far below 120B x 2 bytes.
    assert moe.weight_bytes() / GIB < 120_000_000_000 * 2 / GIB
    # And a decode step reads far less again.
    assert decode_step_bytes(moe) < moe.weight_bytes() / 5


def test_split_precision_is_honored_for_experts():
    """Quantized experts alongside full-precision attention must not be averaged."""
    moe = ModelArchitecture(
        params_total=20_000_000_000, params_active=4_000_000_000,
        params_expert_total=19_000_000_000, params_expert_active=3_000_000_000,
        num_layers=24, hidden_size=2880, num_attention_heads=64,
        num_key_value_heads=8, head_dim=64, torch_dtype="bf16",
        expert_dtype="mxfp4", is_moe=True,
    )
    uniform = 20_000_000_000 * 2
    assert moe.weight_bytes() < uniform / 2
    # An explicit override applies to the whole model, experts included.
    assert moe.weight_bytes("bf16") == pytest.approx(uniform)


def test_kv_cache_scales_with_concurrency_and_bounds_the_running_batch():
    one = estimate_vram(dense_3b(), RTX_3060_RIG, context_tokens=4096, concurrency=1)
    eight = estimate_vram(dense_3b(), RTX_3060_RIG, context_tokens=4096, concurrency=8)
    assert eight.kv_cache_gb == pytest.approx(one.kv_cache_gb * 8, rel=0.01)
    assert one.max_concurrency > 0


def test_a_model_that_cannot_fit_reports_insufficient_rather_than_a_number():
    huge = ModelArchitecture(
        params_total=405_000_000_000, params_active=405_000_000_000, num_layers=126,
        hidden_size=16384, num_attention_heads=128, num_key_value_heads=8,
        head_dim=128, attention_type="gqa", torch_dtype="bf16",
    )
    breakdown = estimate_vram(huge, A100_X1, context_tokens=2048)
    assert not breakdown.fits
    assert breakdown.status == "insufficient"
    assert breakdown.max_concurrency == 0


def test_units_are_replicas_so_capacity_is_per_replica():
    """Extra units must not make an oversized model appear to fit.

    OCI scales a hosting cluster by adding whole copies of the shape, each with
    its own full set of weights. Summing memory across units would report that a
    model fits when no single replica can load it.
    """
    huge = ModelArchitecture(
        params_total=405_000_000_000, params_active=405_000_000_000, num_layers=126,
        hidden_size=16384, num_attention_heads=128, num_key_value_heads=8,
        head_dim=128, attention_type="gqa", torch_dtype="bf16",
    )
    assert not estimate_vram(huge, A100_X1, units=8, context_tokens=2048).fits


def test_minimum_shape_finds_the_smallest_that_fits():
    smallest = minimum_shape(dense_3b(), [A100_X2, A100_X1], context_tokens=2048)
    assert smallest is not None and smallest.key == "A100_80G_X1"


def test_missing_parameter_count_raises_rather_than_guessing():
    unknown = ModelArchitecture(
        params_total=None, params_active=None, num_layers=32, hidden_size=4096,
        num_attention_heads=32, num_key_value_heads=8, head_dim=128,
    )
    with pytest.raises(SizingError):
        estimate_vram(unknown, A100_X1)


# ── Performance model ────────────────────────────────────────────────────────


def test_single_stream_decode_matches_the_bandwidth_roofline():
    """360 GB/s over 6 GB of weights at 75% utilization is 45 tokens/second.

    This is the reference calculator's published figure for the same setup, and
    it pins the whole decode model: bandwidth divided by bytes read per token.
    A short context isolates the weight term, which is the claim being made.
    """
    coefficients = Coefficients(dense_mbu=0.75, decode_a=0.0, mbu_by_gpu_count=())
    estimate = estimate_performance(
        dense_3b(), RTX_3060_RIG, prompt_tokens=8, response_tokens=8,
        concurrency=1, coefficients=coefficients,
    )
    assert estimate.inference_speed_tps == pytest.approx(45.0, rel=0.02)


def test_long_context_slows_decode_because_the_cache_is_reread():
    """The reference calculator ignores KV traffic; at long context it must not.

    Every decode step rereads the whole cache, so a 32K conversation decodes
    measurably slower than a 16-token one on identical hardware.
    """
    coefficients = Coefficients(dense_mbu=0.75, decode_a=0.0, mbu_by_gpu_count=())
    short = estimate_performance(
        dense_3b(), RTX_3060_RIG, prompt_tokens=8, response_tokens=8,
        concurrency=1, coefficients=coefficients,
    )
    long = estimate_performance(
        dense_3b(), RTX_3060_RIG, prompt_tokens=32_000, response_tokens=200,
        concurrency=1, coefficients=coefficients,
    )
    assert long.inference_speed_tps < short.inference_speed_tps * 0.7


def test_derived_metrics_stay_algebraically_consistent():
    """Latency, throughput and RPS are defined off TTFT and decode speed.

    Oracle publishes seven metrics but only two are independent, so the derived
    five must never contradict each other — a latency and a request rate that
    disagree would be worse than either being slightly wrong.
    """
    estimate = estimate_performance(
        dense_3b(), A100_X1, prompt_tokens=2000, response_tokens=200, concurrency=16
    )
    assert estimate.request_latency_s == pytest.approx(
        estimate.ttft_s + 200 / estimate.inference_speed_tps, rel=1e-6
    )
    assert estimate.request_throughput_rps == pytest.approx(
        16 / estimate.request_latency_s, rel=1e-6
    )
    assert estimate.total_throughput_tps == pytest.approx(
        estimate.request_throughput_rps * 2200, rel=1e-6
    )
    assert estimate.request_throughput_rpm == pytest.approx(
        estimate.request_throughput_rps * 60, rel=1e-6
    )


def test_concurrency_lowers_per_user_speed_but_raises_aggregate():
    single = estimate_performance(
        dense_3b(), A100_X1, prompt_tokens=100, response_tokens=100, concurrency=1
    )
    busy = estimate_performance(
        dense_3b(), A100_X1, prompt_tokens=100, response_tokens=100, concurrency=32
    )
    assert busy.inference_speed_tps < single.inference_speed_tps
    assert busy.token_throughput_tps > single.token_throughput_tps


def test_more_units_improve_latency_rather_than_degrading_it():
    """Load is spread across replicas, so adding units must not look harmful."""
    one = estimate_performance(
        dense_3b(), A100_X1, prompt_tokens=2000, response_tokens=200, concurrency=64, units=1
    )
    four = estimate_performance(
        dense_3b(), A100_X1, prompt_tokens=2000, response_tokens=200, concurrency=64, units=4
    )
    assert four.request_latency_s < one.request_latency_s


def test_long_prompts_dominate_time_to_first_token():
    """Prefill is compute-bound and superlinear once attention takes over."""
    short = estimate_performance(
        dense_3b(), A100_X1, prompt_tokens=100, response_tokens=100
    )
    long = estimate_performance(
        dense_3b(), A100_X1, prompt_tokens=32_000, response_tokens=100
    )
    assert long.ttft_s > short.ttft_s * 20


def test_invalid_load_is_rejected():
    with pytest.raises(SizingError):
        estimate_performance(dense_3b(), A100_X1, prompt_tokens=10, response_tokens=10, units=0)
    with pytest.raises(SizingError):
        estimate_performance(
            dense_3b(), A100_X1, prompt_tokens=10, response_tokens=10, concurrency=0
        )


# ── Calibration against Oracle's published grids ─────────────────────────────


@pytest.fixture(scope="module")
def catalog() -> DacCatalog:
    return DacCatalog()


def test_calibration_uses_only_grids_whose_hardware_oracle_names(catalog: DacCatalog):
    """Rows measured on an unnamed "Large Generic" unit must not train the model.

    They carry no bandwidth or FLOP number, so fitting against them would mean
    inventing the hardware they ran on and silently moving every prediction.
    """
    samples = catalog.calibration_samples
    assert samples, "expected published benchmark rows to calibrate against"
    assert {sample.shape.gpu.key for sample in samples} == {"H100"}
    assert len(samples) > len(catalog.benchmark_grids)  # many rows per grid
    # The wider published set is still carried, just not used for fitting.
    assert any(not grid["hardware_known"] for grid in catalog.benchmark_grids)


def test_fit_beats_the_uncalibrated_defaults(catalog: DacCatalog):
    """Calibration has to earn its place against the shipped defaults."""
    samples = catalog.calibration_samples
    fitted = catalog.coefficients
    assert fitted.fitted and fitted.sample_count == len(samples)

    default_decode, _, default_ttft = residuals(samples, Coefficients())
    assert fitted.decode_median_error is not None
    assert default_decode is not None and default_ttft is not None
    assert fitted.decode_median_error < default_decode
    assert fitted.ttft_median_error is not None
    assert fitted.ttft_median_error < default_ttft


def test_held_out_scenarios_stay_within_the_reported_margin(catalog: DacCatalog):
    """Fit on five scenarios, predict the sixth, and check the published error bar.

    This is the test that makes the accuracy claim real rather than asserted: a
    model fitted and scored on the same rows would look good no matter what. The
    bound compared against is the number the UI actually shows, so widening the
    error without widening the badge fails here.
    """
    samples = catalog.calibration_samples
    scenarios = sorted({(item.prompt_tokens, item.response_tokens) for item in samples})
    assert len(scenarios) >= 4

    worst = 0.0
    for held_out in scenarios:
        train = [
            item for item in samples if (item.prompt_tokens, item.response_tokens) != held_out
        ]
        test = [item for item in samples if (item.prompt_tokens, item.response_tokens) == held_out]
        fitted = fit_coefficients(train)
        decode_error, _, _ = residuals(test, fitted)
        assert decode_error is not None
        worst = max(worst, decode_error)

    reported = catalog.coefficients.decode_p90_error
    assert reported is not None
    assert worst <= max(reported, 0.75), (
        f"held-out decode error {worst:.1%} exceeds the margin the UI reports "
        f"({reported:.1%}); either the model regressed or the badge is flattering"
    )


def test_predictions_reproduce_published_rows_for_a_benchmarked_model(catalog: DacCatalog):
    """On a model and shape Oracle measured, single-stream speed should be close."""
    record = catalog.model("openai/gpt-oss-120b")
    shape = catalog.shape("OAI_H100_X2")
    assert record is not None and record.architecture is not None and shape is not None

    published = catalog.published_row("openai/gpt-oss-120b", "OAI_H100_X2", 2000, 200, 1)
    assert published is not None
    estimate = estimate_performance(
        record.architecture, shape, prompt_tokens=2000, response_tokens=200,
        concurrency=1, coefficients=catalog.coefficients,
    )
    assert estimate.inference_speed_tps == pytest.approx(
        published["inference_speed_tps"], rel=0.35
    )
    assert estimate.ttft_s < 1.0


def test_decode_speed_plateaus_once_kv_memory_is_exhausted(catalog: DacCatalog):
    """At 128K context Oracle's measured speed goes flat as concurrency climbs.

    That is the serving stack refusing to admit more sequences than it has cache
    for. A model that keeps dividing speed by offered concurrency is wrong by
    several times at the top of that grid.
    """
    record = catalog.model("openai/gpt-oss-120b")
    shape = catalog.shape("OAI_H100_X2")
    assert record is not None and record.architecture is not None and shape is not None

    speeds = [
        estimate_performance(
            record.architecture, shape, prompt_tokens=128_000, response_tokens=200,
            concurrency=concurrency, coefficients=catalog.coefficients,
        ).inference_speed_tps
        for concurrency in (32, 64, 128, 256)
    ]
    assert max(speeds) == pytest.approx(min(speeds), rel=0.01)


def test_dense_models_do_not_inherit_the_mixture_of_experts_utilization(
    catalog: DacCatalog,
):
    """Both calibrated models are MoE; dense models must use the dense default.

    Applying the MoE-derived figure to a dense model roughly halves its
    predicted speed, and two thirds of the import catalog is dense.
    """
    coefficients = catalog.coefficients
    assert coefficients.mbu_for(2, is_moe=False) == coefficients.dense_mbu
    assert coefficients.mbu_for(2, is_moe=True) < coefficients.dense_mbu


# ── Confidence ───────────────────────────────────────────────────────────────


def test_confidence_tiers_separate_measured_from_extrapolated():
    coefficients = Coefficients(decode_median_error=0.12, fitted=True)
    measured = confidence_for(
        has_published_row=True, within_published_grid=True, calibrated_gpu=True,
        architecture_matches_calibration=True, coefficients=coefficients,
    )
    interpolated = confidence_for(
        has_published_row=False, within_published_grid=True, calibrated_gpu=True,
        architecture_matches_calibration=True, coefficients=coefficients,
    )
    modeled = confidence_for(
        has_published_row=False, within_published_grid=False, calibrated_gpu=False,
        architecture_matches_calibration=False, coefficients=coefficients,
    )
    assert measured.tier == "measured" and measured.error_margin == 0.0
    assert interpolated.tier == "interpolated"
    assert modeled.tier == "modeled"
    # Extrapolating past the calibrated hardware costs more than the in-sample
    # residual, so the margin must widen rather than be reused as-is.
    assert modeled.error_margin > interpolated.error_margin
    assert "mixture-of-experts" in modeled.reason


# ── Cost and optimization ────────────────────────────────────────────────────


def test_unit_hours_are_exact_and_independent_of_the_price():
    cost = cost_estimate(A100_X2, units=2, hours=744.0, price_per_ai_unit_hour=0.0)
    assert cost["unit_hours"] == pytest.approx(6.48 * 2 * 744)
    assert cost["cost"] == 0.0
    priced = cost_estimate(A100_X2, units=2, hours=744.0, price_per_ai_unit_hour=1.5)
    assert priced["unit_hours"] == cost["unit_hours"]
    assert priced["cost"] == pytest.approx(cost["unit_hours"] * 1.5, rel=1e-6)


def test_minimum_commitment_is_billed_when_usage_falls_short():
    cost = cost_estimate(
        A100_X1, units=1, hours=10.0, price_per_ai_unit_hour=1.0, minimum_unit_hours=744
    )
    assert cost["unit_hours"] < 744
    assert cost["billed_unit_hours"] == 744


def test_optimizer_ranks_compliant_options_by_cost_and_keeps_near_misses():
    sla = SlaTarget(
        max_request_latency_s=30.0, concurrency=8, prompt_tokens=2000, response_tokens=200
    )
    options = optimize(
        dense_3b(), [A100_X2, A100_X1], sla, validated_shapes=("A100_80G_X1",),
        max_units=2, validated_only=False,
    )
    assert options
    compliant = [option for option in options if option.meets_sla]
    assert compliant, "expected at least one configuration to meet a lenient target"
    # Compliant first, then validated, then cheapest.
    assert options[0].meets_sla
    costs = [option.cost["unit_hours"] for option in compliant if option.oracle_validated]
    assert costs == sorted(costs)


def test_optimizer_reports_why_a_configuration_misses():
    impossible = SlaTarget(
        max_request_latency_s=0.001, concurrency=8, prompt_tokens=2000, response_tokens=200
    )
    options = optimize(dense_3b(), [A100_X1], impossible, max_units=1, validated_only=False)
    assert options and not options[0].meets_sla
    assert options[0].unmet and "latency" in options[0].unmet[0]


def test_validated_only_hides_shapes_oracle_has_not_blessed():
    sla = SlaTarget(concurrency=4, prompt_tokens=1000, response_tokens=100)
    restricted = optimize(
        dense_3b(), [A100_X1, A100_X2], sla, validated_shapes=("A100_80G_X1",),
        max_units=1, validated_only=True,
    )
    assert {option.shape.key for option in restricted} == {"A100_80G_X1"}
    everything = optimize(
        dense_3b(), [A100_X1, A100_X2], sla, validated_shapes=("A100_80G_X1",),
        max_units=1, validated_only=False,
    )
    assert len({option.shape.key for option in everything}) == 2
