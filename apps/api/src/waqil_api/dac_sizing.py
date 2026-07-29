"""Memory and performance model for OCI Dedicated AI Cluster hosting.

The question this answers is "which validated unit shape do I need for this
model, will it hit my latency target, and what does it cost" — for the whole
model-import catalog, not just the handful of models Oracle publishes numbers
for.

Two layers, kept separate because they fail differently:

* A **memory model** — weights, KV cache, activations, framework overhead. This
  is close to exact. Every term is arithmetic over published architecture, and
  the only judgement call is the serving stack's usable-VRAM fraction.
* A **performance model** — time to first token and decode speed. This is a
  roofline (prefill is compute-bound, decode is bandwidth-bound) whose
  efficiency coefficients are *fitted* to Oracle's published benchmark grids
  rather than assumed.

Only two of Oracle's benchmark pages name the GPUs behind their cluster unit
(the gpt-oss pages, on OAI_H100_X1 and OAI_H100_X2); every other page reports an
opaque marketing unit. So the fit is anchored on two MoE models on H100, and
everything else — dense models, other GPUs — is extrapolation. That limitation
is not hidden: `confidence_for` downgrades those predictions and the reported
error bar comes from held-out residuals, not from optimism.

Oracle publishes seven metrics, but only two of them are independent. The rest
follow algebraically from TTFT and decode speed, so only those two are modeled
and the remainder are derived — which also means the derived metrics stay
mutually consistent instead of drifting apart under separate fits.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Sequence

# Bytes per stored parameter. MXFP4 is 4-bit data plus a shared exponent per
# 32-element block, hence slightly over 0.5.
DTYPE_BYTES: dict[str, float] = {
    "fp32": 4.0,
    "float32": 4.0,
    "fp16": 2.0,
    "float16": 2.0,
    "bf16": 2.0,
    "bfloat16": 2.0,
    "fp8": 1.0,
    "float8": 1.0,
    "int8": 1.0,
    "int4": 0.5,
    "mxfp4": 0.53,
    "nvfp4": 0.53,
    "q4_k_m": 0.55,
}
DEFAULT_DTYPE = "bf16"

# vLLM reserves a slice of each GPU for the CUDA context, allocator headroom and
# fragmentation; `gpu_memory_utilization` defaults to 0.90 and OME does not
# raise it. Anything above this fraction of HBM is not addressable in practice.
USABLE_VRAM_FRACTION = 0.90

# Roughly fixed per-GPU cost of the runtime itself: CUDA context, NCCL buffers,
# captured CUDA graphs. Measured in the 1-2 GB range across vLLM deployments and
# treated as constant because it does not scale with model or batch.
FRAMEWORK_OVERHEAD_GB_PER_GPU = 1.0

# Gigabytes are decimal (1e9), not GiB, and so are the GPU capacities in
# gpus.json. The two must agree: mixing a GiB model size against a decimal card
# size is a silent 7% error in the direction that makes things look like they
# fit. Decimal also matches how weights are universally quoted — a 3B model at
# FP16 is "6 GB" everywhere — and reading a vendor's "80GB" as 80e9 bytes errs
# toward under-reporting capacity, which is the safe direction for sizing.
BYTES_PER_GB = 1e9


def dtype_bytes(name: str | None) -> float:
    if not name:
        return DTYPE_BYTES[DEFAULT_DTYPE]
    return DTYPE_BYTES.get(str(name).strip().lower(), DTYPE_BYTES[DEFAULT_DTYPE])


class SizingError(ValueError):
    """Raised when an estimate is asked for with incoherent inputs."""


# ── Inputs ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GpuSpec:
    key: str
    label: str
    memory_gb: float
    memory_bandwidth_gb_s: float
    dense_bf16_tflops: float
    dense_fp8_tflops: float | None = None
    supports_fp8: bool = False

    def compute_tflops(self, quantization: str | None) -> float:
        """Peak dense tensor throughput for the precision the weights run at."""
        if self.supports_fp8 and self.dense_fp8_tflops and dtype_bytes(quantization) <= 1.0:
            return self.dense_fp8_tflops
        return self.dense_bf16_tflops


@dataclass(frozen=True)
class ShapeSpec:
    key: str
    gpu: GpuSpec
    gpu_count: int
    ai_units: float
    importable: bool = True

    @property
    def total_memory_gb(self) -> float:
        return self.gpu.memory_gb * self.gpu_count

    @property
    def usable_memory_gb(self) -> float:
        return self.total_memory_gb * USABLE_VRAM_FRACTION

    @property
    def total_bandwidth_gb_s(self) -> float:
        return self.gpu.memory_bandwidth_gb_s * self.gpu_count


@dataclass(frozen=True)
class ModelArchitecture:
    """The subset of a Hugging Face config the sizing math actually reads."""

    params_total: int | None
    params_active: int | None
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    # Expert weights are tracked apart from the rest because they are commonly
    # stored at a lower precision than attention and embeddings.
    params_expert_total: int = 0
    params_expert_active: int = 0
    expert_dtype: str | None = None
    attention_type: Literal["mha", "gqa", "mla"] = "gqa"
    vocab_size: int | None = None
    max_position_embeddings: int | None = None
    sliding_window: int | None = None
    sliding_window_ratio: float | None = None
    torch_dtype: str | None = None
    is_moe: bool = False
    moe: dict[str, Any] | None = None
    mla: dict[str, Any] | None = None

    @property
    def effective_active_params(self) -> int:
        """Parameters read per decoded token — what decode bandwidth sees."""
        if self.params_active:
            return self.params_active
        return self.params_total or 0

    def weight_bytes(self, quantization: str | None = None) -> float:
        """Bytes every resident weight occupies, honoring a split expert dtype.

        An explicit `quantization` override applies to the whole model, because
        that is what a user asking "what if I run this in FP8" means.
        """
        total = self.params_total or self.effective_active_params
        if quantization:
            return total * dtype_bytes(quantization)
        experts = min(self.params_expert_total, total)
        dense = max(0, total - experts)
        return dense * dtype_bytes(self.torch_dtype) + experts * dtype_bytes(
            self.expert_dtype or self.torch_dtype
        )

    def active_weight_bytes(self, quantization: str | None = None) -> float:
        """Bytes read to decode one token, honoring a split expert dtype."""
        active = self.effective_active_params
        if quantization:
            return active * dtype_bytes(quantization)
        experts = min(self.params_expert_active, active)
        dense = max(0, active - experts)
        return dense * dtype_bytes(self.torch_dtype) + experts * dtype_bytes(
            self.expert_dtype or self.torch_dtype
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelArchitecture":
        required = ("num_layers", "hidden_size", "num_attention_heads")
        missing = [key for key in required if not raw.get(key)]
        if missing:
            raise SizingError(f"architecture is missing {', '.join(missing)}")
        heads = int(raw["num_attention_heads"])
        return cls(
            params_total=raw.get("params_total"),
            params_active=raw.get("params_active"),
            num_layers=int(raw["num_layers"]),
            hidden_size=int(raw["hidden_size"]),
            num_attention_heads=heads,
            num_key_value_heads=int(raw.get("num_key_value_heads") or heads),
            head_dim=int(raw.get("head_dim") or (int(raw["hidden_size"]) // heads)),
            params_expert_total=int(raw.get("params_expert_total") or 0),
            params_expert_active=int(raw.get("params_expert_active") or 0),
            expert_dtype=raw.get("expert_dtype"),
            attention_type=raw.get("attention_type") or "gqa",
            vocab_size=raw.get("vocab_size"),
            max_position_embeddings=raw.get("max_position_embeddings"),
            sliding_window=raw.get("sliding_window"),
            sliding_window_ratio=raw.get("sliding_window_ratio"),
            torch_dtype=raw.get("torch_dtype"),
            is_moe=bool(raw.get("is_moe")),
            moe=raw.get("moe"),
            mla=raw.get("mla"),
        )


# ── Memory ───────────────────────────────────────────────────────────────────


def kv_bytes_per_token(
    architecture: ModelArchitecture,
    *,
    kv_quantization: str | None = None,
    context_tokens: int | None = None,
) -> float:
    """Bytes of KV cache one token of context costs.

    Three architectural facts change this by an order of magnitude, so each is
    handled rather than folded into an average:

    * **GQA** — the cache is sized by *key/value* heads, not query heads. Using
      `num_attention_heads` here is the single most common way to overstate KV
      memory (8x on a typical 32:4 model).
    * **MLA** — DeepSeek caches one compressed latent per token plus a small
      RoPE part, instead of per-head K and V. That is a different formula, not a
      scaled one.
    * **Sliding window** — a windowed layer never caches more than its window,
      so past a few thousand tokens those layers stop growing entirely. Models
      that interleave windowed and global layers only get the discount on the
      windowed fraction.
    """
    element = dtype_bytes(kv_quantization or architecture.torch_dtype)
    layers = architecture.num_layers

    if architecture.attention_type == "mla" and architecture.mla:
        latent = architecture.mla.get("kv_lora_rank") or 0
        rope = architecture.mla.get("qk_rope_head_dim") or 0
        if latent:
            # One shared latent vector plus the decoupled RoPE key, per layer.
            return float(layers) * (latent + rope) * element

    per_layer = 2 * architecture.num_key_value_heads * architecture.head_dim * element

    window = architecture.sliding_window
    if window and context_tokens and context_tokens > window:
        # Windowed layers hold `window` tokens no matter how long the context
        # gets, so their share is rescaled to an equivalent per-token cost.
        ratio = architecture.sliding_window_ratio
        ratio = 1.0 if ratio is None else max(0.0, min(1.0, ratio))
        windowed = ratio * per_layer * (window / context_tokens)
        globaled = (1.0 - ratio) * per_layer
        per_layer = windowed + globaled

    return float(layers) * per_layer


@dataclass(frozen=True)
class VramBreakdown:
    weights_gb: float
    kv_cache_gb: float
    activations_gb: float
    overhead_gb: float
    total_gb: float
    capacity_gb: float
    usable_gb: float
    utilization: float
    status: Literal["okay", "moderate", "high", "very_high", "insufficient"]
    fits: bool
    max_concurrency: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "weights_gb": round(self.weights_gb, 3),
            "kv_cache_gb": round(self.kv_cache_gb, 3),
            "activations_gb": round(self.activations_gb, 3),
            "overhead_gb": round(self.overhead_gb, 3),
            "total_gb": round(self.total_gb, 3),
            "capacity_gb": round(self.capacity_gb, 3),
            "usable_gb": round(self.usable_gb, 3),
            "utilization": round(self.utilization, 4),
            "status": self.status,
            "fits": self.fits,
            "max_concurrency": self.max_concurrency,
        }


def _status(utilization: float, fits: bool) -> str:
    if not fits:
        return "insufficient"
    if utilization <= 0.50:
        return "okay"
    if utilization <= 0.75:
        return "moderate"
    if utilization <= 0.90:
        return "high"
    return "very_high"


def estimate_vram(
    architecture: ModelArchitecture,
    shape: ShapeSpec,
    *,
    units: int = 1,
    quantization: str | None = None,
    kv_quantization: str | None = None,
    context_tokens: int = 4096,
    concurrency: int = 1,
) -> VramBreakdown:
    """Memory footprint of hosting `architecture` on `units` x `shape`.

    Units are replicas, not a bigger pool: OCI scales a hosting cluster by
    adding whole copies of the shape, each holding its own full set of weights.
    So capacity for one replica is what decides whether the model fits, and unit
    count multiplies throughput rather than memory. Reporting the aggregate
    memory instead would say a model fits when no single replica can load it.
    """
    if units < 1:
        raise SizingError("units must be at least 1")
    if concurrency < 1:
        raise SizingError("concurrency must be at least 1")
    context_tokens = max(1, context_tokens)

    if not (architecture.params_total or architecture.effective_active_params):
        raise SizingError("architecture has no parameter count")
    weights_gb = architecture.weight_bytes(quantization) / BYTES_PER_GB
    element_bytes = dtype_bytes(quantization or architecture.torch_dtype)

    per_token = kv_bytes_per_token(
        architecture, kv_quantization=kv_quantization, context_tokens=context_tokens
    )
    kv_gb = per_token * context_tokens * concurrency / BYTES_PER_GB

    # Peak activation working set. Serving stacks cap prefill into chunks, so
    # this tracks the chunk rather than the whole prompt; the constant folds in
    # the handful of hidden-sized buffers a transformer block keeps live.
    chunk = min(context_tokens, 2048)
    activations_gb = 18 * chunk * architecture.hidden_size * element_bytes / BYTES_PER_GB

    overhead_gb = FRAMEWORK_OVERHEAD_GB_PER_GPU * shape.gpu_count

    total_gb = weights_gb + kv_gb + activations_gb + overhead_gb
    capacity_gb = shape.total_memory_gb
    usable_gb = shape.usable_memory_gb
    fits = total_gb <= usable_gb

    # How many sequences the leftover memory can actually hold at this context.
    spare = usable_gb - (weights_gb + activations_gb + overhead_gb)
    per_sequence_gb = per_token * context_tokens / BYTES_PER_GB
    max_concurrency = (
        int(spare / per_sequence_gb) if spare > 0 and per_sequence_gb > 0 else 0
    )

    return VramBreakdown(
        weights_gb=weights_gb,
        kv_cache_gb=kv_gb,
        activations_gb=activations_gb,
        overhead_gb=overhead_gb,
        total_gb=total_gb,
        capacity_gb=capacity_gb,
        usable_gb=usable_gb,
        utilization=total_gb / capacity_gb if capacity_gb else 0.0,
        status=_status(total_gb / capacity_gb if capacity_gb else 1.0, fits),  # type: ignore[arg-type]
        fits=fits,
        max_concurrency=max(0, max_concurrency),
    )


def minimum_shape(
    architecture: ModelArchitecture,
    shapes: Iterable[ShapeSpec],
    **kwargs: Any,
) -> ShapeSpec | None:
    """Smallest shape (by AI units) the model actually fits on.

    Oracle's recommended shape frequently exceeds this — Qwen3-0.6B is listed
    against a full A100 80GB — because the recommendation carries validation and
    throughput headroom, not just capacity. Surfacing both lets a reader see the
    gap instead of assuming the recommendation is a memory floor.
    """
    candidates = [shape for shape in shapes if shape.importable]
    for shape in sorted(candidates, key=lambda item: (item.ai_units, item.gpu_count)):
        try:
            if estimate_vram(architecture, shape, **kwargs).fits:
                return shape
        except SizingError:
            return None
    return None


# ── Performance ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Coefficients:
    """Fitted efficiency constants for the roofline model.

    `mbu` and `mfu` are the fraction of peak memory bandwidth and peak compute a
    real serving stack reaches. The rest describe how per-user decode speed
    decays as concurrent sequences contend, and how prefill queues behind other
    prefills.
    """

    mbu: float = 0.24
    # Measured bandwidth utilization per GPU count, where Oracle's data covers
    # it. A single global MBU misses that the two calibrated configurations sit
    # ~1.9x apart, and the two differ in GPU count *and* expert count at once —
    # so which factor drives the gap is not identifiable from two points. Rather
    # than attribute it to one and extrapolate a made-up law, the measured value
    # is used where it exists and clamped to the nearest measured count outside
    # that range; aggregate bandwidth then still scales with GPUs, it just stops
    # pretending to know how per-GPU efficiency keeps changing.
    mbu_by_gpu_count: tuple[tuple[int, float], ...] = ()
    # Both hardware-anchored calibration models are mixture-of-experts, and MoE
    # decode is markedly less bandwidth-efficient than dense at small batch —
    # the router gathers a few experts scattered across a large weight pool.
    # Two thirds of the import catalog is dense, so applying the MoE-derived
    # figure to them would understate a dense 8B model by roughly half. Dense
    # models instead use this published-roofline default, and `confidence_for`
    # reports them as uncalibrated rather than implying the fit covers them.
    dense_mbu: float = 0.70
    mfu: float = 0.55
    decode_a: float = 0.135
    decode_b: float = 0.72
    ttft_overhead_s: float = 0.06
    queue_factor: float = 0.30
    queue_exponent: float = 0.85
    fitted: bool = False
    sample_count: int = 0
    decode_median_error: float | None = None
    decode_p90_error: float | None = None
    ttft_median_error: float | None = None

    def mbu_for(self, gpu_count: int, *, is_moe: bool = True) -> float:
        """Bandwidth utilization for this configuration.

        Dense models take the published default at any GPU count: their tensor-
        parallel scaling is not calibrated either, and inventing a derate from
        two MoE measurements would be a guess wearing a fitted number's clothes.
        Aggregate bandwidth still grows with GPU count regardless.
        """
        if not is_moe:
            return self.dense_mbu
        if not self.mbu_by_gpu_count:
            return self.mbu
        measured = sorted(self.mbu_by_gpu_count)
        for count, value in measured:
            if gpu_count <= count:
                return value
        return measured[-1][1]

    def as_dict(self) -> dict[str, Any]:
        return {
            "mbu": round(self.mbu, 4),
            "mbu_by_gpu_count": {str(count): round(value, 4) for count, value in self.mbu_by_gpu_count},
            "dense_mbu": round(self.dense_mbu, 4),
            "mfu": round(self.mfu, 4),
            "decode_a": round(self.decode_a, 4),
            "decode_b": round(self.decode_b, 4),
            "ttft_overhead_s": round(self.ttft_overhead_s, 4),
            "queue_factor": round(self.queue_factor, 4),
            "queue_exponent": round(self.queue_exponent, 4),
            "fitted": self.fitted,
            "sample_count": self.sample_count,
            "decode_median_error": (
                round(self.decode_median_error, 4)
                if self.decode_median_error is not None
                else None
            ),
            "decode_p90_error": (
                round(self.decode_p90_error, 4) if self.decode_p90_error is not None else None
            ),
            "ttft_median_error": (
                round(self.ttft_median_error, 4)
                if self.ttft_median_error is not None
                else None
            ),
        }


DEFAULT_COEFFICIENTS = Coefficients()


def decode_bytes_per_token(
    architecture: ModelArchitecture, *, quantization: str | None = None
) -> float:
    """Weight bytes read to decode one token.

    MoE reads only the routed experts per token, so the active parameter count
    is what bandwidth sees — using the total would understate a 120B/5.7B model
    by more than 20x.
    """
    return architecture.active_weight_bytes(quantization)


def decode_step_bytes(
    architecture: ModelArchitecture,
    *,
    batch: int = 1,
    context_tokens: int = 0,
    quantization: str | None = None,
    kv_quantization: str | None = None,
) -> float:
    """Bytes read from HBM to advance a decode step for `batch` sequences.

    Weights are read once and shared by the whole batch; every sequence's KV
    cache is read on top of that. Treating long context as a multiplicative
    slowdown instead of these additive bytes is what makes naive calculators
    wrong at 128K: the honest form predicts gpt-oss-20b at 200 tokens/second
    single-stream against Oracle's measured 209, where the multiplicative form
    predicts 109.
    """
    weights = architecture.active_weight_bytes(quantization)
    if weights <= 0:
        raise SizingError("cannot determine decode weight bytes")
    if context_tokens <= 0 or batch <= 0:
        return weights
    per_token = kv_bytes_per_token(
        architecture, kv_quantization=kv_quantization, context_tokens=context_tokens
    )
    return weights + batch * per_token * context_tokens


def peak_decode_speed(
    architecture: ModelArchitecture,
    shape: ShapeSpec,
    *,
    context_tokens: int = 0,
    batch: int = 1,
    quantization: str | None = None,
    kv_quantization: str | None = None,
    coefficients: Coefficients = DEFAULT_COEFFICIENTS,
) -> float:
    """The bandwidth roof for one decode step, derated by measured MBU."""
    per_step = decode_step_bytes(
        architecture,
        batch=batch,
        context_tokens=context_tokens,
        quantization=quantization,
        kv_quantization=kv_quantization,
    )
    utilization = coefficients.mbu_for(shape.gpu_count, is_moe=architecture.is_moe)
    return shape.total_bandwidth_gb_s * 1e9 * utilization / per_step


def running_batch_size(
    architecture: ModelArchitecture,
    shape: ShapeSpec,
    *,
    concurrency: int,
    context_tokens: int,
    quantization: str | None = None,
    kv_quantization: str | None = None,
) -> int:
    """Sequences actually decoding at once, after KV memory runs out.

    Offered concurrency and running batch are not the same thing. A serving
    stack admits only as many sequences as it has KV cache for and queues the
    rest, so past that point extra load lengthens the queue instead of slowing
    each token down.

    This is visible in Oracle's own numbers: gpt-oss-120b at 128K context holds
    a decode rate of ~27 tokens/second from concurrency 8 all the way to 256,
    while request latency climbs from 39s to 605s. Modeling contention against
    offered concurrency instead predicts a decode rate that keeps falling, which
    is wrong by 4x at the top of that grid.
    """
    try:
        vram = estimate_vram(
            architecture,
            shape,
            quantization=quantization,
            kv_quantization=kv_quantization,
            context_tokens=context_tokens,
            concurrency=1,
        )
    except SizingError:
        return max(1, concurrency)
    if not vram.fits:
        return max(1, concurrency)
    return max(1, min(concurrency, vram.max_concurrency or concurrency))


def decode_speed(
    architecture: ModelArchitecture,
    shape: ShapeSpec,
    *,
    concurrency: int,
    context_tokens: int,
    quantization: str | None = None,
    kv_quantization: str | None = None,
    coefficients: Coefficients = DEFAULT_COEFFICIENTS,
) -> float:
    """Per-user decode rate under contention.

    Concurrency does not simply divide throughput: batching amortizes the weight
    read across sequences, so aggregate rises while per-user speed falls. The
    decay is a power law in the *running* batch, whose strength grows with
    context — capturing both the KV traffic that scales with context and the
    widening set of MoE experts a larger batch touches.
    """
    batch = running_batch_size(
        architecture,
        shape,
        concurrency=concurrency,
        context_tokens=context_tokens,
        quantization=quantization,
        kv_quantization=kv_quantization,
    )
    peak = peak_decode_speed(
        architecture,
        shape,
        context_tokens=context_tokens,
        batch=batch,
        quantization=quantization,
        kv_quantization=kv_quantization,
        coefficients=coefficients,
    )
    # Whatever batching costs beyond the bytes it moves — scheduling, kernel
    # efficiency, widening MoE expert activation. Context is already accounted
    # for in the bytes, so this term only has to explain batch size.
    contention = coefficients.decode_a * (batch**coefficients.decode_b)
    return peak / (1.0 + contention)


def prefill_flops(architecture: ModelArchitecture, prompt_tokens: int) -> float:
    """FLOPs to prefill a prompt, including the quadratic attention term.

    The linear `2 * params * tokens` term alone is fine at chat lengths and badly
    wrong at 128K, where attention dominates. Sliding-window layers stay linear
    because each query only attends over its window.
    """
    tokens = max(1, prompt_tokens)
    linear = 2.0 * architecture.effective_active_params * tokens

    attention_width = architecture.num_attention_heads * architecture.head_dim
    ratio = architecture.sliding_window_ratio
    windowed_fraction = 0.0 if ratio is None else max(0.0, min(1.0, ratio))
    if not architecture.sliding_window:
        windowed_fraction = 0.0
    global_layers = architecture.num_layers * (1.0 - windowed_fraction)
    windowed_layers = architecture.num_layers * windowed_fraction

    quadratic = 4.0 * global_layers * (tokens**2) * attention_width
    window = architecture.sliding_window or 0
    windowed = 4.0 * windowed_layers * tokens * min(tokens, window) * attention_width
    return linear + quadratic + windowed


def time_to_first_token(
    architecture: ModelArchitecture,
    shape: ShapeSpec,
    *,
    prompt_tokens: int,
    concurrency: int = 1,
    quantization: str | None = None,
    coefficients: Coefficients = DEFAULT_COEFFICIENTS,
) -> float:
    """Prefill latency, including the queueing other concurrent prefills cause."""
    flops = prefill_flops(architecture, prompt_tokens)
    compute = shape.gpu.compute_tflops(quantization or architecture.torch_dtype)
    peak = compute * 1e12 * shape.gpu_count * coefficients.mfu
    if peak <= 0:
        raise SizingError("shape has no compute rating")
    queued = 1.0 + coefficients.queue_factor * (max(1, concurrency) - 1) ** coefficients.queue_exponent
    return coefficients.ttft_overhead_s + (flops / peak) * queued


@dataclass(frozen=True)
class PerformanceEstimate:
    ttft_s: float
    inference_speed_tps: float
    token_throughput_tps: float
    request_latency_s: float
    request_throughput_rps: float
    total_throughput_tps: float
    concurrency: int
    prompt_tokens: int
    response_tokens: int

    @property
    def request_throughput_rpm(self) -> float:
        """Requests per minute — the unit Oracle uses on its Cohere and Meta pages."""
        return self.request_throughput_rps * 60.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "ttft_s": round(self.ttft_s, 4),
            "inference_speed_tps": round(self.inference_speed_tps, 2),
            "token_throughput_tps": round(self.token_throughput_tps, 2),
            "request_latency_s": round(self.request_latency_s, 4),
            "request_throughput_rps": round(self.request_throughput_rps, 4),
            "request_throughput_rpm": round(self.request_throughput_rpm, 2),
            "total_throughput_tps": round(self.total_throughput_tps, 2),
            "concurrency": self.concurrency,
            "prompt_tokens": self.prompt_tokens,
            "response_tokens": self.response_tokens,
        }


def estimate_performance(
    architecture: ModelArchitecture,
    shape: ShapeSpec,
    *,
    prompt_tokens: int,
    response_tokens: int,
    concurrency: int = 1,
    units: int = 1,
    quantization: str | None = None,
    coefficients: Coefficients = DEFAULT_COEFFICIENTS,
) -> PerformanceEstimate:
    """The seven metrics Oracle publishes, for one configuration.

    Concurrency is the load offered to the whole cluster, so it is divided
    across replicas before the per-replica model runs; the resulting throughputs
    are then multiplied back up. Feeding total concurrency to a single replica
    would make extra units look like they slow the service down.
    """
    if units < 1:
        raise SizingError("units must be at least 1")
    if concurrency < 1:
        raise SizingError("concurrency must be at least 1")
    prompt_tokens = max(1, prompt_tokens)
    response_tokens = max(1, response_tokens)

    per_replica = max(1, math.ceil(concurrency / units))
    context = prompt_tokens + response_tokens

    ttft = time_to_first_token(
        architecture,
        shape,
        prompt_tokens=prompt_tokens,
        concurrency=per_replica,
        quantization=quantization,
        coefficients=coefficients,
    )
    speed = decode_speed(
        architecture,
        shape,
        concurrency=per_replica,
        context_tokens=context,
        quantization=quantization,
        coefficients=coefficients,
    )
    latency = ttft + response_tokens / speed
    # Oracle's remaining five metrics are all defined off these two, so they are
    # derived rather than modeled: keeping them algebraic guarantees a prediction
    # can never report a latency and a throughput that disagree.
    request_rps = concurrency / latency if latency > 0 else 0.0
    token_throughput = concurrency * response_tokens / latency if latency > 0 else 0.0
    total_throughput = request_rps * (prompt_tokens + response_tokens)

    return PerformanceEstimate(
        ttft_s=ttft,
        inference_speed_tps=speed,
        token_throughput_tps=token_throughput,
        request_latency_s=latency,
        request_throughput_rps=request_rps,
        total_throughput_tps=total_throughput,
        concurrency=concurrency,
        prompt_tokens=prompt_tokens,
        response_tokens=response_tokens,
    )


# ── Calibration ──────────────────────────────────────────────────────────────


@dataclass
class CalibrationSample:
    architecture: ModelArchitecture
    shape: ShapeSpec
    prompt_tokens: int
    response_tokens: int
    concurrency: int
    ttft_s: float
    inference_speed_tps: float


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _least_squares(matrix: list[list[float]], target: list[float]) -> list[float] | None:
    """Tiny normal-equation solve, so calibration needs no numpy at import."""
    columns = len(matrix[0]) if matrix else 0
    if not matrix or len(matrix) < columns:
        return None
    normal = [[0.0] * columns for _ in range(columns)]
    rhs = [0.0] * columns
    for row, value in zip(matrix, target, strict=True):
        for i in range(columns):
            rhs[i] += row[i] * value
            for j in range(columns):
                normal[i][j] += row[i] * row[j]
    # Gaussian elimination with partial pivoting.
    for i in range(columns):
        pivot = max(range(i, columns), key=lambda r: abs(normal[r][i]))
        if abs(normal[pivot][i]) < 1e-12:
            return None
        normal[i], normal[pivot] = normal[pivot], normal[i]
        rhs[i], rhs[pivot] = rhs[pivot], rhs[i]
        for r in range(i + 1, columns):
            factor = normal[r][i] / normal[i][i]
            if factor:
                for c in range(i, columns):
                    normal[r][c] -= factor * normal[i][c]
                rhs[r] -= factor * rhs[i]
    solution = [0.0] * columns
    for i in reversed(range(columns)):
        total = rhs[i] - sum(normal[i][j] * solution[j] for j in range(i + 1, columns))
        solution[i] = total / normal[i][i]
    return solution


def _fit_prefill(
    samples: Sequence[CalibrationSample], base: Coefficients
) -> tuple[float, float, float, float]:
    """Fit (overhead, MFU, queue factor, queue exponent) for time to first token.

    Fitting TTFT by ordinary least squares on seconds does not work: a 128K-token
    prompt takes ~600s at high concurrency while a chat prompt takes 0.04s, so
    the squared error is entirely decided by the long-context rows and the fitted
    fixed overhead lands near two seconds — thirty times the real value, and
    visibly wrong on every short prompt.

    So the fit minimizes *relative* error, and is staged. Single-request rows
    isolate overhead and MFU with no queueing in play; the concurrent rows then
    only have to explain what queueing adds on top.
    """
    overhead, mfu = base.ttft_overhead_s, base.mfu

    def compute_seconds(item: CalibrationSample) -> float:
        flops = prefill_flops(item.architecture, item.prompt_tokens)
        compute = item.shape.gpu.compute_tflops(item.architecture.torch_dtype)
        peak = compute * 1e12 * item.shape.gpu_count
        return flops / peak if peak > 0 else 0.0

    single = [item for item in samples if item.concurrency == 1 and item.ttft_s > 0]
    if single:
        best = None
        # Overhead spans a few milliseconds to a second; MFU cannot exceed 1.
        for overhead_candidate in [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3]:
            for mfu_candidate in [0.05 * step for step in range(1, 21)]:
                error = 0.0
                for item in single:
                    predicted = overhead_candidate + compute_seconds(item) / mfu_candidate
                    error += (math.log(max(predicted, 1e-9) / item.ttft_s)) ** 2
                if best is None or error < best[0]:
                    best = (error, overhead_candidate, mfu_candidate)
        if best:
            _, overhead, mfu = best

    # Queueing is fitted against each scenario's own measured single-request
    # row rather than against the modeled one. Dividing by a modeled baseline
    # is unstable exactly where it matters: on short prompts the compute term is
    # ~10ms against an 80ms fixed overhead, so a few ms of error in the baseline
    # multiplies the inferred queueing by an order of magnitude. That is what
    # drove an earlier fit to a queue factor of 3.0, which then predicted a
    # 2.8-second TTFT for an 8B model where vLLM measures under 0.2.
    reference: dict[tuple[str, int, int], float] = {}
    for item in samples:
        if item.concurrency == 1 and item.ttft_s > 0:
            reference[(item.shape.key, item.prompt_tokens, item.response_tokens)] = item.ttft_s

    rows: list[list[float]] = []
    target: list[float] = []
    for item in samples:
        if item.concurrency <= 1 or item.ttft_s <= 0:
            continue
        single_ttft = reference.get((item.shape.key, item.prompt_tokens, item.response_tokens))
        baseline = compute_seconds(item) / mfu
        if single_ttft is None or baseline <= 0:
            continue
        excess = (item.ttft_s - single_ttft) / baseline
        if excess <= 0:
            continue
        rows.append([1.0, math.log(item.concurrency - 1)])
        target.append(math.log(excess))

    solved = _least_squares(rows, target) if len(rows) >= 4 else None
    if solved:
        # A request cannot wait behind more prefill than the other requests
        # actually issue, so the factor is capped at mild super-serialization
        # (prefill also contends with in-flight decode) rather than left free.
        return overhead, mfu, min(2.0, math.exp(solved[0])), min(1.2, max(0.4, solved[1]))
    return overhead, mfu, base.queue_factor, base.queue_exponent


def fit_coefficients(
    samples: Sequence[CalibrationSample],
    *,
    base: Coefficients = DEFAULT_COEFFICIENTS,
) -> Coefficients:
    """Fit MBU, MFU and the contention terms to measured benchmark rows.

    Both halves are linearized so the fit is a closed-form least squares rather
    than an iterative search: the decode model rearranges to
    `log(peak/v - 1) = log(a) + e*log(1 + ctx/ref) + b*log(C)`, and the prefill
    model is linear in FLOPs once the queueing factor is divided out.

    A failed fit returns the caller's baseline unchanged rather than a partly
    updated one. Half-fitted coefficients would still be reported as calibrated,
    which is worse than honestly falling back to the defaults.
    """
    if not samples:
        return base

    # MBU first: at concurrency 1 there is no contention, so measured decode
    # speed divided by the bandwidth roof is the bandwidth utilization directly.
    single = [item for item in samples if item.concurrency == 1]
    mbu = base.mbu
    by_count: dict[int, list[float]] = {}
    if single:
        ratios = []
        for item in single:
            per_step = decode_step_bytes(
                item.architecture,
                batch=1,
                context_tokens=item.prompt_tokens + item.response_tokens,
            )
            roof = item.shape.total_bandwidth_gb_s * 1e9 / per_step
            if roof > 0:
                ratio = item.inference_speed_tps / roof
                ratios.append(ratio)
                by_count.setdefault(item.shape.gpu_count, []).append(ratio)
        if ratios:
            mbu = _median(ratios)
    mbu_by_gpu_count = tuple(
        sorted((count, _median(values)) for count, values in by_count.items())
    )

    probe = Coefficients(
        **{**base.__dict__, "mbu": mbu, "mbu_by_gpu_count": mbu_by_gpu_count}
    )

    rows: list[list[float]] = []
    target: list[float] = []
    for item in samples:
        context = item.prompt_tokens + item.response_tokens
        batch = running_batch_size(
            item.architecture, item.shape, concurrency=item.concurrency, context_tokens=context
        )
        if batch <= 1:
            continue
        peak = peak_decode_speed(
            item.architecture,
            item.shape,
            context_tokens=context,
            batch=batch,
            coefficients=probe,
        )
        if peak <= item.inference_speed_tps:
            continue  # measured faster than the roof: nothing to attribute
        contention = peak / item.inference_speed_tps - 1.0
        if contention <= 0:
            continue
        rows.append([1.0, math.log(batch)])
        target.append(math.log(contention))

    decode = _least_squares(rows, target) if len(rows) >= 4 else None
    if decode:
        log_a, concurrency_exponent = decode
        decode_a = math.exp(log_a)
    else:
        decode_a = base.decode_a
        concurrency_exponent = base.decode_b

    overhead, mfu, queue_factor, queue_exponent = _fit_prefill(samples, base)

    fitted = Coefficients(
        mbu=mbu,
        mbu_by_gpu_count=mbu_by_gpu_count,
        mfu=mfu,
        decode_a=decode_a,
        decode_b=concurrency_exponent,
        ttft_overhead_s=overhead,
        queue_factor=queue_factor,
        queue_exponent=queue_exponent,
        fitted=True,
        sample_count=len(samples),
    )
    decode_error, decode_tail, ttft_error = residuals(samples, fitted)
    return Coefficients(
        **{
            **fitted.__dict__,
            "decode_median_error": decode_error,
            "decode_p90_error": decode_tail,
            "ttft_median_error": ttft_error,
        }
    )


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(len(ordered) * fraction)))
    return ordered[index]


def residuals(
    samples: Sequence[CalibrationSample], coefficients: Coefficients
) -> tuple[float | None, float | None, float | None]:
    """Relative error of the model against measured rows: (median, p90, ttft).

    Both a median and a tail are reported because the median alone flatters the
    model: it is comfortably low while the worst configurations — 128K context,
    and the deliberately stochastic Random Length scenario — sit several times
    higher. A single "±x%" built from the median would understate the risk on
    exactly the configurations a reader is least able to check.
    """
    decode_errors: list[float] = []
    ttft_errors: list[float] = []
    for item in samples:
        context = item.prompt_tokens + item.response_tokens
        predicted = decode_speed(
            item.architecture,
            item.shape,
            concurrency=item.concurrency,
            context_tokens=context,
            coefficients=coefficients,
        )
        if item.inference_speed_tps > 0:
            decode_errors.append(abs(predicted - item.inference_speed_tps) / item.inference_speed_tps)
        predicted_ttft = time_to_first_token(
            item.architecture,
            item.shape,
            prompt_tokens=item.prompt_tokens,
            concurrency=item.concurrency,
            coefficients=coefficients,
        )
        if item.ttft_s > 0:
            ttft_errors.append(abs(predicted_ttft - item.ttft_s) / item.ttft_s)
    return (
        _median(decode_errors) if decode_errors else None,
        _percentile(decode_errors, 0.9) if decode_errors else None,
        _median(ttft_errors) if ttft_errors else None,
    )


# ── Confidence ───────────────────────────────────────────────────────────────

Confidence = Literal["measured", "interpolated", "modeled"]


@dataclass(frozen=True)
class ConfidenceVerdict:
    tier: Confidence
    error_margin: float | None
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "error_margin": round(self.error_margin, 3) if self.error_margin is not None else None,
            "reason": self.reason,
        }


def confidence_for(
    *,
    has_published_row: bool,
    within_published_grid: bool,
    calibrated_gpu: bool,
    architecture_matches_calibration: bool,
    coefficients: Coefficients,
) -> ConfidenceVerdict:
    """How much to trust one prediction, and why.

    The distinction that matters is not "did we compute this carefully" but "how
    far is this configuration from something Oracle actually measured". A number
    read straight off a published table and a number extrapolated to a dense
    model on a GPU nobody benchmarked are both produced by this module, and
    presenting them identically would be the dishonest part.
    """
    margin = coefficients.decode_median_error
    if has_published_row:
        return ConfidenceVerdict("measured", 0.0, "Oracle publishes this exact row.")
    if within_published_grid:
        return ConfidenceVerdict(
            "interpolated",
            margin,
            "Between published benchmark points for this model and shape.",
        )
    reasons = []
    if not calibrated_gpu:
        reasons.append("no published benchmark names this GPU")
    if not architecture_matches_calibration:
        reasons.append("calibration models are mixture-of-experts, this one is dense")
    if not coefficients.fitted:
        reasons.append("running on default coefficients, not a fit")
    detail = "; ".join(reasons) if reasons else "extrapolated from the calibrated roofline"
    # Extrapolating past the calibrated hardware costs more than the in-sample
    # residual suggests, so the published margin is widened rather than reused.
    penalty = 1.0 + (0.0 if calibrated_gpu else 0.5) + (0.0 if architecture_matches_calibration else 0.3)
    widened = margin * penalty if margin is not None else None
    return ConfidenceVerdict("modeled", widened, detail.capitalize() + ".")


# ── Cost and optimization ────────────────────────────────────────────────────


def cost_estimate(
    shape: ShapeSpec,
    *,
    units: int,
    hours: float,
    price_per_ai_unit_hour: float,
    minimum_unit_hours: int = 0,
) -> dict[str, Any]:
    """AI unit-hours (exact) and the dollar figure they imply (rate-dependent)."""
    unit_hours = shape.ai_units * units * hours
    billed = max(unit_hours, minimum_unit_hours) if minimum_unit_hours else unit_hours
    return {
        "ai_units_per_unit": shape.ai_units,
        "units": units,
        "hours": hours,
        "unit_hours": round(unit_hours, 3),
        "billed_unit_hours": round(billed, 3),
        "minimum_unit_hours": minimum_unit_hours,
        "cost": round(billed * price_per_ai_unit_hour, 2),
    }


@dataclass
class SlaTarget:
    max_ttft_s: float | None = None
    max_request_latency_s: float | None = None
    min_inference_speed_tps: float | None = None
    min_request_throughput_rps: float | None = None
    concurrency: int = 1
    prompt_tokens: int = 2000
    response_tokens: int = 200

    def unmet(self, estimate: PerformanceEstimate) -> list[str]:
        failures: list[str] = []
        if self.max_ttft_s is not None and estimate.ttft_s > self.max_ttft_s:
            failures.append(f"TTFT {estimate.ttft_s:.2f}s > {self.max_ttft_s:.2f}s")
        if (
            self.max_request_latency_s is not None
            and estimate.request_latency_s > self.max_request_latency_s
        ):
            failures.append(
                f"latency {estimate.request_latency_s:.2f}s > {self.max_request_latency_s:.2f}s"
            )
        if (
            self.min_inference_speed_tps is not None
            and estimate.inference_speed_tps < self.min_inference_speed_tps
        ):
            failures.append(
                f"speed {estimate.inference_speed_tps:.1f} < {self.min_inference_speed_tps:.1f} tok/s"
            )
        if (
            self.min_request_throughput_rps is not None
            and estimate.request_throughput_rps < self.min_request_throughput_rps
        ):
            failures.append(
                f"throughput {estimate.request_throughput_rps:.2f} < "
                f"{self.min_request_throughput_rps:.2f} rps"
            )
        return failures


@dataclass
class SizingOption:
    shape: ShapeSpec
    units: int
    vram: VramBreakdown
    performance: PerformanceEstimate
    cost: dict[str, Any]
    meets_sla: bool
    unmet: list[str] = field(default_factory=list)
    oracle_validated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "shape": self.shape.key,
            "gpu": self.shape.gpu.key,
            "gpu_count": self.shape.gpu_count,
            "units": self.units,
            "oracle_validated": self.oracle_validated,
            "vram": self.vram.as_dict(),
            "performance": self.performance.as_dict(),
            "cost": self.cost,
            "meets_sla": self.meets_sla,
            "unmet": self.unmet,
        }


def optimize(
    architecture: ModelArchitecture,
    shapes: Sequence[ShapeSpec],
    sla: SlaTarget,
    *,
    validated_shapes: Sequence[str] = (),
    max_units: int = 8,
    hours: float = 744.0,
    price_per_ai_unit_hour: float = 0.0,
    quantization: str | None = None,
    kv_quantization: str | None = None,
    coefficients: Coefficients = DEFAULT_COEFFICIENTS,
    validated_only: bool = True,
) -> list[SizingOption]:
    """Every shape x unit count that fits, cheapest compliant option first.

    Shapes Oracle has validated for this model are preferred, because a
    configuration that is cheaper but unvalidated is not actually available to
    deploy; `validated_only` exists so a reader can deliberately look past that.
    Options that miss the SLA are still returned, ranked below the ones that
    meet it, so a target nothing can hit shows near misses rather than nothing.
    """
    validated = {name.upper() for name in validated_shapes}
    options: list[SizingOption] = []

    for shape in shapes:
        if not shape.importable:
            continue
        is_validated = shape.key.upper() in validated
        if validated_only and validated and not is_validated:
            continue
        for units in range(1, max_units + 1):
            try:
                vram = estimate_vram(
                    architecture,
                    shape,
                    units=units,
                    quantization=quantization,
                    kv_quantization=kv_quantization,
                    context_tokens=sla.prompt_tokens + sla.response_tokens,
                    concurrency=max(1, math.ceil(sla.concurrency / units)),
                )
            except SizingError:
                continue
            if not vram.fits:
                continue
            performance = estimate_performance(
                architecture,
                shape,
                prompt_tokens=sla.prompt_tokens,
                response_tokens=sla.response_tokens,
                concurrency=sla.concurrency,
                units=units,
                quantization=quantization,
                coefficients=coefficients,
            )
            unmet = sla.unmet(performance)
            options.append(
                SizingOption(
                    shape=shape,
                    units=units,
                    vram=vram,
                    performance=performance,
                    cost=cost_estimate(
                        shape,
                        units=units,
                        hours=hours,
                        price_per_ai_unit_hour=price_per_ai_unit_hour,
                    ),
                    meets_sla=not unmet,
                    unmet=unmet,
                    oracle_validated=is_validated,
                )
            )
            if not unmet:
                break  # more units of this shape only cost more

    options.sort(
        key=lambda option: (
            not option.meets_sla,
            not option.oracle_validated,
            option.cost["unit_hours"],
            option.shape.ai_units * option.units,
        )
    )
    return options
