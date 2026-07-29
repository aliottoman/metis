"""Request handling for the Dedicated AI Cluster sizing tab.

Sits between the API routes and the pure math in `dac_sizing`, so the routes
stay thin and the math stays free of contracts and model plumbing.

The recommender here is deliberately two-stage. A deterministic scorer always
runs and always produces a ranking; the user's own configured model — local
Ollama or OCI, whichever `ModelPreferenceStore` resolves — is then given that
shortlist to reorder and explain. Sizing arithmetic is something a host can do
exactly, and judging whether a model suits a use case is not; keeping them apart
means the tab still answers offline, and that a model failure costs the prose
rather than the recommendation.
"""
from __future__ import annotations

import json
import math
from typing import Any

from .contracts import (
    DacCandidateV1,
    DacCatalogV1,
    DacConfidenceV1,
    DacEstimateRequestV1,
    DacEstimateV1,
    DacGpuV1,
    DacModelV1,
    DacOptimizeRequestV1,
    DacOptimizeResultV1,
    DacOptionV1,
    DacPerformanceV1,
    DacRecommendationV1,
    DacRecommendRequestV1,
    DacShapeV1,
    DacVramBreakdownV1,
    ModelRequestV1,
)
from .dac_catalog import DacCatalog
from .dac_sizing import (
    DTYPE_BYTES,
    SizingError,
    SlaTarget,
    confidence_for,
    cost_estimate,
    estimate_performance,
    estimate_vram,
    minimum_shape,
    optimize,
)

# Offered to the UI as weight/KV precision choices. Ordered from most to least
# precise so the control reads as a quality dial.
QUANTIZATION_CHOICES = ("fp32", "bf16", "fp16", "fp8", "int8", "mxfp4", "int4")

# Cheap use-case signals, matched against the model id. Only distinctive markers
# belong here: a family name like "qwen" matches every Qwen model and turns the
# bonus into noise, and "instruct" matches nearly the whole catalog. Judging
# which model is actually good at a task is the language model's job, and this
# table is kept narrow so it cannot quietly grow into a worse version of one.
USE_CASE_HINTS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("code", "coding", "program", "developer", "sql", "refactor"), ("coder",)),
    (("reason", "math", "proof", "agent", "plan"), ("qwq", "r1", "reasoning", "thinking")),
)

VISION_WORDS = ("vision", "image", "screenshot", "diagram", "ocr", "document", "visual")
EMBED_WORDS = ("embed", "embedding", "vector", "retrieval index", "semantic search")
RERANK_WORDS = ("rerank", "re-rank", "reranking")


class DacService:
    def __init__(self, catalog: DacCatalog, model: Any = None, preference: Any = None) -> None:
        self._catalog = catalog
        self._model = model
        self._preference = preference

    # ── Catalog ──────────────────────────────────────────────────────────────

    def catalog(self) -> DacCatalogV1:
        catalog = self._catalog
        return DacCatalogV1(
            models=[
                DacModelV1(
                    **record.as_dict(),
                    benchmarked_shapes=catalog.benchmarked_shapes_for(record.id),
                )
                for record in catalog.models.values()
            ],
            shapes=[
                DacShapeV1(
                    key=shape.key,
                    gpu=shape.gpu.key,
                    gpu_count=shape.gpu_count,
                    ai_units=shape.ai_units,
                    total_memory_gb=shape.total_memory_gb,
                    importable=shape.importable,
                )
                for shape in catalog.shapes.values()
            ],
            gpus=[
                DacGpuV1(
                    key=gpu.key,
                    label=gpu.label,
                    memory_gb=gpu.memory_gb,
                    memory_bandwidth_gb_s=gpu.memory_bandwidth_gb_s,
                    dense_bf16_tflops=gpu.dense_bf16_tflops,
                    dense_fp8_tflops=gpu.dense_fp8_tflops,
                    supports_fp8=gpu.supports_fp8,
                )
                for gpu in catalog.gpus.values()
            ],
            quantizations=[name for name in QUANTIZATION_CHOICES if name in DTYPE_BYTES],
            pricing=catalog.pricing,
            provenance=catalog.provenance(),
        )

    # ── Estimate ─────────────────────────────────────────────────────────────

    def _confidence(self, model_id: str, shape_key: str, request: Any) -> DacConfidenceV1:
        catalog = self._catalog
        record = catalog.model(model_id)
        shape = catalog.shape(shape_key)
        published = None
        if shape is not None:
            published = catalog.published_row(
                model_id,
                shape.key,
                request.prompt_tokens,
                request.response_tokens,
                request.concurrency,
            )
        verdict = confidence_for(
            has_published_row=published is not None,
            within_published_grid=shape is not None and catalog.has_grid(model_id, shape.key),
            calibrated_gpu=shape is not None and shape.gpu.key in catalog.calibrated_gpus,
            architecture_matches_calibration=bool(
                record and record.architecture and record.architecture.is_moe
            ),
            coefficients=catalog.coefficients,
        )
        return DacConfidenceV1(**verdict.as_dict())

    def estimate(self, request: DacEstimateRequestV1) -> DacEstimateV1:
        catalog = self._catalog
        record = catalog.model(request.model_id)
        if record is None:
            raise SizingError(f"unknown model {request.model_id!r}")
        if record.architecture is None:
            raise SizingError(record.unsupported_reason or "model has no architecture data")
        shape = catalog.shape(request.shape)
        if shape is None:
            raise SizingError(f"unknown shape {request.shape!r}")

        architecture = record.architecture
        context = request.prompt_tokens + request.response_tokens
        vram = estimate_vram(
            architecture,
            shape,
            units=request.units,
            quantization=request.quantization,
            kv_quantization=request.kv_quantization,
            context_tokens=context,
            concurrency=max(1, math.ceil(request.concurrency / request.units)),
        )
        performance = estimate_performance(
            architecture,
            shape,
            prompt_tokens=request.prompt_tokens,
            response_tokens=request.response_tokens,
            concurrency=request.concurrency,
            units=request.units,
            quantization=request.quantization,
            coefficients=catalog.coefficients,
        )
        price = (
            request.price_per_ai_unit_hour
            if request.price_per_ai_unit_hour is not None
            else catalog.price_per_ai_unit_hour
        )
        cost = cost_estimate(
            shape, units=request.units, hours=request.hours, price_per_ai_unit_hour=price
        )
        smallest = minimum_shape(
            architecture,
            catalog.importable_shapes,
            quantization=request.quantization,
            kv_quantization=request.kv_quantization,
            context_tokens=context,
            concurrency=1,
        )
        published = catalog.published_row(
            request.model_id,
            shape.key,
            request.prompt_tokens,
            request.response_tokens,
            request.concurrency,
        )

        notes: list[str] = []
        metrics = DacPerformanceV1(**performance.as_dict())
        if published and request.units == 1:
            # Oracle measured this exact configuration, so report what it
            # measured. A badge that says "measured" above a modeled number is
            # the one case where the confidence label would actively mislead —
            # and the model is ~13% off here, which a reader has no way to see.
            metrics = _measured_performance(published, request)
            notes.append(
                "These are Oracle's published measurements for this exact "
                "configuration, not modeled values."
            )
        validated = shape.key.upper() in {name.upper() for name in record.validated_shapes}
        if record.validated_shapes and not validated:
            notes.append(
                f"Oracle has not validated {shape.key} for this model. "
                f"Validated: {', '.join(record.validated_shapes)}."
            )
        if smallest and record.validated_shapes and smallest.key not in record.validated_shapes:
            notes.append(
                f"The model fits on {smallest.key}, smaller than Oracle's recommendation — "
                "the recommended shape carries throughput and validation headroom, "
                "not just capacity."
            )
        if not vram.fits:
            notes.append(
                "Does not fit: weights plus KV cache exceed the usable memory of one replica."
            )
        if vram.max_concurrency and request.concurrency > vram.max_concurrency * request.units:
            notes.append(
                f"KV cache holds about {vram.max_concurrency * request.units} concurrent "
                "sequences at this context; beyond that requests queue rather than run."
            )

        return DacEstimateV1(
            model_id=request.model_id,
            shape=shape.key,
            units=request.units,
            oracle_validated=validated,
            minimum_shape=smallest.key if smallest else None,
            vram=DacVramBreakdownV1(**vram.as_dict()),
            performance=metrics,
            cost=cost,
            confidence=self._confidence(request.model_id, shape.key, request),
            published=published,
            notes=notes,
        )

    # ── Optimize ─────────────────────────────────────────────────────────────

    def optimize(self, request: DacOptimizeRequestV1) -> DacOptimizeResultV1:
        catalog = self._catalog
        record = catalog.model(request.model_id)
        if record is None:
            raise SizingError(f"unknown model {request.model_id!r}")
        if record.architecture is None:
            raise SizingError(record.unsupported_reason or "model has no architecture data")

        sla = SlaTarget(
            max_ttft_s=request.max_ttft_s,
            max_request_latency_s=request.max_request_latency_s,
            min_inference_speed_tps=request.min_inference_speed_tps,
            min_request_throughput_rps=request.min_request_throughput_rps,
            concurrency=request.concurrency,
            prompt_tokens=request.prompt_tokens,
            response_tokens=request.response_tokens,
        )
        price = (
            request.price_per_ai_unit_hour
            if request.price_per_ai_unit_hour is not None
            else catalog.price_per_ai_unit_hour
        )
        options = optimize(
            record.architecture,
            catalog.importable_shapes,
            sla,
            validated_shapes=record.validated_shapes,
            max_units=request.max_units,
            hours=request.hours,
            price_per_ai_unit_hour=price,
            quantization=request.quantization,
            kv_quantization=request.kv_quantization,
            coefficients=catalog.coefficients,
            validated_only=request.validated_only,
        )

        notes: list[str] = []
        if request.validated_only and record.validated_shapes:
            notes.append(
                "Showing only shapes Oracle validated for this model. "
                "Turn that off to see cheaper unvalidated configurations."
            )
        if not any(option.meets_sla for option in options):
            notes.append("No configuration meets the target; the closest misses are listed first.")

        best = options[0].shape.key if options else (record.validated_shapes[0] if record.validated_shapes else "")
        return DacOptimizeResultV1(
            model_id=request.model_id,
            options=[DacOptionV1(**option.as_dict()) for option in options[:24]],
            confidence=self._confidence(request.model_id, best, request),
            considered=len(options),
            notes=notes,
        )

    # ── Recommend ────────────────────────────────────────────────────────────

    def _target_capability(self, use_case: str) -> str | None:
        lowered = use_case.lower()
        if any(word in lowered for word in RERANK_WORDS):
            return "RERANK"
        if any(word in lowered for word in EMBED_WORDS):
            return "EMBEDDING"
        if any(word in lowered for word in VISION_WORDS):
            return "IMAGE_TEXT_TO_TEXT"
        return None

    def _shortlist(
        self,
        request: DacRecommendRequestV1,
        *,
        candidate_limit: int | None = None,
    ) -> list[DacCandidateV1]:
        """Rank deployable models for the use case, without a model call.

        Scoring is intentionally about deployability rather than intelligence:
        does it fit a validated shape, does it hit the latency target, what does
        it cost, and how big is it. Parameter count is a crude capability proxy
        and is weighted lowest, because it is the part a language model is
        actually better placed to judge.
        """
        catalog = self._catalog
        wanted = request.capability or self._target_capability(request.use_case)
        lowered = request.use_case.lower()
        keywords: set[str] = set()
        for triggers, hints in USE_CASE_HINTS:
            if any(trigger in lowered for trigger in triggers):
                keywords.update(hints)

        sla = SlaTarget(
            max_request_latency_s=request.max_request_latency_s,
            concurrency=request.concurrency,
            prompt_tokens=request.prompt_tokens,
            response_tokens=request.response_tokens,
        )
        context = request.prompt_tokens + request.response_tokens
        candidates: list[tuple[float, DacCandidateV1]] = []

        for record in catalog.models.values():
            if record.architecture is None or not record.validated_shapes:
                continue
            if wanted and record.capability != wanted:
                continue
            if not wanted and record.capability not in ("TEXT_TO_TEXT", "IMAGE_TEXT_TO_TEXT"):
                continue

            best: tuple[float, Any, Any, Any, int] | None = None
            for shape in catalog.validated_shapes_for(record.id):
                for units in (1, 2, 4):
                    try:
                        vram = estimate_vram(
                            record.architecture,
                            shape,
                            units=units,
                            context_tokens=context,
                            concurrency=max(1, math.ceil(request.concurrency / units)),
                        )
                    except SizingError:
                        continue
                    if not vram.fits:
                        continue
                    performance = estimate_performance(
                        record.architecture,
                        shape,
                        prompt_tokens=request.prompt_tokens,
                        response_tokens=request.response_tokens,
                        concurrency=request.concurrency,
                        units=units,
                        coefficients=catalog.coefficients,
                    )
                    unit_hours = shape.ai_units * units
                    meets = not sla.unmet(performance)
                    # Cheapest configuration that meets the target; failing that,
                    # the cheapest that runs at all.
                    rank = (0 if meets else 1, unit_hours)
                    if best is None or rank < best[0]:
                        best = (rank, shape, performance, vram, units)  # type: ignore[assignment]
            if best is None:
                continue
            _, shape, performance, _vram, units = best  # type: ignore[misc]
            meets = not sla.unmet(performance)

            params = record.architecture.params_total or 0
            size_score = math.log10(max(params, 1e8)) / 12.0
            cost_score = 1.0 / (1.0 + shape.ai_units * units / 10.0)
            keyword_score = (
                1.0 if any(word in record.id.lower() for word in keywords) else 0.0
            )
            score = (
                (3.0 if meets else 0.0)
                + 2.0 * keyword_score
                + 1.5 * cost_score
                + 1.0 * size_score
            )
            candidates.append(
                (
                    score,
                    DacCandidateV1(
                        model_id=record.id,
                        family=record.family,
                        capability=record.capability,
                        score=round(score, 3),
                        shape=shape.key,
                        units=units,
                        performance=DacPerformanceV1(**performance.as_dict()),
                        cost=cost_estimate(
                            shape,
                            units=units,
                            hours=744.0,
                            price_per_ai_unit_hour=catalog.price_per_ai_unit_hour,
                        ),
                        meets_sla=meets,
                    ),
                )
            )

        candidates.sort(key=lambda item: -item[0])
        limit = candidate_limit or request.limit

        # OCI validates many near-identical generations from some families. If
        # those occupy the whole shortlist, the model-backed judge never sees
        # alternatives from other vendors (or newer models whose ids do not
        # contain an old task marker such as "-VL-"). Keep the deterministic
        # list useful offline while preserving score order within each family.
        selected: list[DacCandidateV1] = []
        selected_ids: set[str] = set()
        family_counts: dict[str, int] = {}
        for _, candidate in candidates:
            if family_counts.get(candidate.family, 0) >= 2:
                continue
            selected.append(candidate)
            selected_ids.add(candidate.model_id)
            family_counts[candidate.family] = family_counts.get(candidate.family, 0) + 1
            if len(selected) >= limit:
                return selected

        # A narrow capability may have only a few families. Fill any remaining
        # slots by score rather than returning fewer results than requested.
        for _, candidate in candidates:
            if candidate.model_id in selected_ids:
                continue
            selected.append(candidate)
            if len(selected) >= limit:
                break
        return selected

    async def recommend(self, request: DacRecommendRequestV1) -> DacRecommendationV1:
        # The language model needs a broad enough choice to judge task quality;
        # request.limit is the number of answers the user wants, not the number
        # of models the judge should be allowed to consider.
        judge_limit = min(20, max(12, request.limit * 3))
        shortlist = self._shortlist(request, candidate_limit=judge_limit)
        notes: list[str] = []
        if not shortlist:
            notes.append(
                "No validated model matches that use case at this load. "
                "Try relaxing the latency target or the concurrency."
            )
            return DacRecommendationV1(
                use_case=request.use_case, candidates=[], notes=notes
            )

        reranked, summary, model_used = await self._model_rerank(request, shortlist)
        if model_used is None:
            notes.append(
                "Ranked without a model call — sizing and cost are exact, but the "
                "fit of each model to your use case is not judged."
            )
        return DacRecommendationV1(
            use_case=request.use_case,
            candidates=reranked[: request.limit],
            summary=summary,
            model_used=model_used,
            model_backed=model_used is not None,
            notes=notes,
        )

    async def _model_rerank(
        self, request: DacRecommendRequestV1, shortlist: list[DacCandidateV1]
    ) -> tuple[list[DacCandidateV1], str | None, str | None]:
        """Let the configured model reorder and explain the shortlist.

        Every failure path returns the deterministic ranking untouched. The
        model is allowed to reorder models it was given and to write prose; it
        is never allowed to introduce a model, a shape, or a number, so a
        hallucination can change the order of a list the host already validated
        but cannot invent a configuration that does not exist.
        """
        if self._model is None:
            return shortlist, None, None
        aliases: dict[str, str] = {}
        if self._preference is not None:
            try:
                aliases = self._preference.resolve_aliases()
            except Exception:  # noqa: BLE001 - fall back to provider defaults
                aliases = {}

        allowed = {candidate.model_id for candidate in shortlist}
        payload = {
            "use_case": request.use_case,
            "candidates": [
                {
                    "model_id": candidate.model_id,
                    "family": candidate.family,
                    "shape": candidate.shape,
                    "units": candidate.units,
                    "meets_latency_target": candidate.meets_sla,
                    "tokens_per_second": (
                        candidate.performance.inference_speed_tps if candidate.performance else None
                    ),
                    "ai_unit_hours": (candidate.cost or {}).get("unit_hours"),
                }
                for candidate in shortlist
            ],
        }
        system = (
            "You advise on picking a model to host on Oracle Cloud Infrastructure. "
            "You are given a shortlist that is already known to fit and to be "
            "validated by Oracle. Reorder it by how well each model suits the "
            "stated use case, and explain the top choice in two sentences. Treat "
            "multiple numbered use cases as distinct requirements and prefer a "
            "model that serves all of them well, or explain the tradeoff. Consider "
            "model generation and task specialization; do not assume the largest "
            "or most familiar model is best. "
            "Use only the model ids given. Do not invent models, shapes or numbers. "
            'Reply with JSON only: {"order": ["model id", ...], '
            '"rationales": {"model id": "one sentence"}, "summary": "two sentences"}'
        )
        try:
            result = await self._model.generate(
                ModelRequestV1(
                    role="planner",
                    system_prompt=system,
                    user_prompt=json.dumps(payload, ensure_ascii=False),
                ),
                model_aliases=aliases,
            )
        except Exception:  # noqa: BLE001 - a model outage must not lose the ranking
            return shortlist, None, None

        parsed = _parse_json_object(result.content or "")
        if not parsed:
            return shortlist, None, None

        by_id = {candidate.model_id: candidate for candidate in shortlist}
        rationales = parsed.get("rationales")
        if isinstance(rationales, dict):
            for model_id, text in rationales.items():
                candidate = by_id.get(model_id)
                if candidate is not None and isinstance(text, str):
                    candidate.rationale = text.strip()[:400]

        order = parsed.get("order")
        ordered: list[DacCandidateV1] = []
        if isinstance(order, list):
            for model_id in order:
                if isinstance(model_id, str) and model_id in allowed and model_id in by_id:
                    ordered.append(by_id.pop(model_id))
        # Anything the model dropped keeps its deterministic position at the end,
        # so a truncated reply narrows the ordering rather than the shortlist.
        ordered.extend(by_id[key] for key in [c.model_id for c in shortlist] if key in by_id)

        summary = parsed.get("summary")
        return (
            ordered or shortlist,
            summary.strip()[:800] if isinstance(summary, str) else None,
            result.model,
        )


def _measured_performance(
    published: dict[str, Any], request: DacEstimateRequestV1
) -> DacPerformanceV1:
    """Oracle's own published row, shaped like a computed estimate.

    Only applied when the request matches a published row exactly, and only for
    a single unit — Oracle benchmarks one unit, and scaling its measurement to
    a multi-unit cluster would be a model again, wearing a measurement's badge.
    """
    rps = float(published.get("request_throughput_rps") or 0.0)
    return DacPerformanceV1(
        ttft_s=float(published.get("ttft_s") or 0.0),
        inference_speed_tps=float(published.get("inference_speed_tps") or 0.0),
        token_throughput_tps=float(published.get("token_throughput_tps") or 0.0),
        request_latency_s=float(published.get("request_latency_s") or 0.0),
        request_throughput_rps=rps,
        request_throughput_rpm=rps * 60.0,
        total_throughput_tps=float(published.get("total_throughput_tps") or 0.0),
        concurrency=request.concurrency,
        prompt_tokens=request.prompt_tokens,
        response_tokens=request.response_tokens,
    )


def _parse_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort JSON object out of a model reply that may be fenced."""
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.lower().startswith("json"):
            candidate = candidate[4:]
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(candidate[start : end + 1])
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None
