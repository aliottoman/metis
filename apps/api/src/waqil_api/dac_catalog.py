"""Loads the vendored OCI sizing catalog and calibrates the roofline model.

Everything here reads from `data/dac/*.json`, which `scripts/build_dac_catalog.py`
regenerates by hand. Nothing in this module touches the network: the Sizing tab
has to give the same answer on a plane as it does online, and a catalog that
silently refreshed itself would also silently change the numbers a user had
already written down.

The calibration runs once at construction. It is a closed-form least squares
over ~100 rows, so it costs microseconds, and doing it here rather than baking
constants into source means the coefficients always match the benchmark data
actually shipped alongside them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from .dac_sizing import (
    CalibrationSample,
    Coefficients,
    DEFAULT_COEFFICIENTS,
    GpuSpec,
    ModelArchitecture,
    ShapeSpec,
    SizingError,
    fit_coefficients,
)

DATA_DIR = Path(__file__).resolve().parent / "data" / "dac"

# Capabilities the roofline model applies to. A diffusion or speech model has no
# KV cache and no autoregressive decode, so the catalog keeps it (Oracle does
# validate it for import) but the calculator declines to estimate rather than
# printing a number derived from the wrong physics.
TEXT_GENERATION_CAPABILITIES = frozenset({"TEXT_TO_TEXT", "IMAGE_TEXT_TO_TEXT"})


@dataclass(frozen=True)
class CatalogModel:
    id: str
    family: str
    capability: str
    validated_shapes: tuple[str, ...]
    architecture: ModelArchitecture | None
    architecture_raw: dict[str, Any] | None
    unsupported_reason: str | None

    @property
    def supported(self) -> bool:
        return self.architecture is not None

    def as_dict(self) -> dict[str, Any]:
        architecture = None
        if self.architecture_raw:
            architecture = {
                key: value
                for key, value in self.architecture_raw.items()
                if key not in ("config_source",)
            }
        return {
            "id": self.id,
            "family": self.family,
            "capability": self.capability,
            "validated_shapes": list(self.validated_shapes),
            "supported": self.supported,
            "unsupported_reason": self.unsupported_reason,
            "config_source": (self.architecture_raw or {}).get("config_source"),
            "architecture": architecture,
        }


def _unsupported_reason(capability: str, architecture: dict[str, Any] | None) -> str | None:
    if architecture is not None:
        return None
    if capability not in TEXT_GENERATION_CAPABILITIES:
        return (
            f"{capability.replace('_', ' ').title()} models do not decode "
            "autoregressively, so the token-throughput model does not apply."
        )
    return "No published architecture for this repository."


class DacCatalog:
    """The vendored shapes, GPUs, models, benchmarks and fitted coefficients."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self._dir = data_dir or DATA_DIR
        self._gpus_raw = self._load("gpus.json")
        self._shapes_raw = self._load("shapes.json")
        self._models_raw = self._load("models.json")
        self._benchmarks_raw = self._load("benchmarks.json")
        self._pricing_raw = self._load("pricing.json")

    def _load(self, name: str) -> dict[str, Any]:
        path = self._dir / name
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # A missing or corrupt file degrades the tab to "no data" rather
            # than taking the whole API down at import time.
            return {}

    # ── Hardware ─────────────────────────────────────────────────────────────

    @cached_property
    def gpus(self) -> dict[str, GpuSpec]:
        return {
            key: GpuSpec(
                key=key,
                label=value.get("label", key),
                memory_gb=float(value["memory_gb"]),
                memory_bandwidth_gb_s=float(value["memory_bandwidth_gb_s"]),
                dense_bf16_tflops=float(value["dense_bf16_tflops"]),
                dense_fp8_tflops=(
                    float(value["dense_fp8_tflops"]) if value.get("dense_fp8_tflops") else None
                ),
                supports_fp8=bool(value.get("supports_fp8")),
            )
            for key, value in (self._gpus_raw.get("gpus") or {}).items()
        }

    @cached_property
    def shapes(self) -> dict[str, ShapeSpec]:
        shapes: dict[str, ShapeSpec] = {}
        for key, value in (self._shapes_raw.get("shapes") or {}).items():
            gpu = self.gpus.get(value.get("gpu"))
            if gpu is None:
                continue
            shapes[key] = ShapeSpec(
                key=key,
                gpu=gpu,
                gpu_count=int(value["gpu_count"]),
                ai_units=float(value["ai_units"]),
                importable=bool(value.get("importable", True)),
            )
        return shapes

    @cached_property
    def shape_aliases(self) -> dict[str, str]:
        return {
            key.upper(): value
            for key, value in (self._shapes_raw.get("aliases") or {}).items()
        }

    def shape(self, key: str | None) -> ShapeSpec | None:
        if not key:
            return None
        name = key.strip().upper()
        return self.shapes.get(name) or self.shapes.get(self.shape_aliases.get(name, ""))

    @property
    def importable_shapes(self) -> list[ShapeSpec]:
        return [shape for shape in self.shapes.values() if shape.importable]

    # ── Models ───────────────────────────────────────────────────────────────

    @cached_property
    def models(self) -> dict[str, CatalogModel]:
        models: dict[str, CatalogModel] = {}
        for record in self._models_raw.get("models") or []:
            raw = record.get("architecture")
            architecture: ModelArchitecture | None = None
            if isinstance(raw, dict):
                try:
                    architecture = ModelArchitecture.from_dict(raw)
                except SizingError:
                    architecture = None
            capability = str(record.get("capability") or "").upper()
            models[record["id"]] = CatalogModel(
                id=record["id"],
                family=record.get("family") or "Other",
                capability=capability,
                validated_shapes=tuple(record.get("validated_shapes") or ()),
                architecture=architecture,
                architecture_raw=raw if isinstance(raw, dict) else None,
                unsupported_reason=_unsupported_reason(capability, raw if architecture else None),
            )
        return models

    def model(self, model_id: str) -> CatalogModel | None:
        return self.models.get(model_id)

    def validated_shapes_for(self, model_id: str) -> list[ShapeSpec]:
        record = self.model(model_id)
        if record is None:
            return []
        resolved = [self.shape(name) for name in record.validated_shapes]
        return [shape for shape in resolved if shape is not None]

    # ── Benchmarks and calibration ───────────────────────────────────────────

    @property
    def benchmark_grids(self) -> list[dict[str, Any]]:
        return list(self._benchmarks_raw.get("grids") or [])

    @cached_property
    def calibration_samples(self) -> list[CalibrationSample]:
        """Published rows whose GPU shape and model architecture are both known.

        Everything else Oracle publishes is still shown to the user, but cannot
        train the model: a row measured on "one Large Generic unit" has no
        bandwidth or FLOP number attached to it, and guessing which GPU that is
        would quietly move every downstream prediction.
        """
        samples: list[CalibrationSample] = []
        for grid in self.benchmark_grids:
            if not grid.get("hardware_known"):
                continue
            shape = self.shape(grid.get("shape"))
            record = self.model(grid.get("hf_id") or "")
            if shape is None or record is None or record.architecture is None:
                continue
            prompt = grid.get("prompt_tokens")
            response = grid.get("response_tokens")
            if not prompt or not response:
                continue
            for row in grid.get("rows") or []:
                samples.append(
                    CalibrationSample(
                        architecture=record.architecture,
                        shape=shape,
                        prompt_tokens=int(prompt),
                        response_tokens=int(response),
                        concurrency=int(row["concurrency"]),
                        ttft_s=float(row["ttft_s"]),
                        inference_speed_tps=float(row["inference_speed_tps"]),
                    )
                )
        return samples

    @cached_property
    def coefficients(self) -> Coefficients:
        samples = self.calibration_samples
        if not samples:
            return DEFAULT_COEFFICIENTS
        return fit_coefficients(samples)

    @cached_property
    def calibrated_gpus(self) -> frozenset[str]:
        return frozenset(sample.shape.gpu.key for sample in self.calibration_samples)

    def published_row(
        self, model_id: str, shape_key: str, prompt_tokens: int, response_tokens: int, concurrency: int
    ) -> dict[str, Any] | None:
        """An exact published measurement, if Oracle happens to have one."""
        for grid in self.benchmark_grids:
            if grid.get("hf_id") != model_id or grid.get("shape") != shape_key:
                continue
            if grid.get("prompt_tokens") != prompt_tokens:
                continue
            if grid.get("response_tokens") != response_tokens:
                continue
            for row in grid.get("rows") or []:
                if int(row.get("concurrency", -1)) == concurrency:
                    return {**row, "scenario": grid.get("scenario"), "source_url": grid.get("source_url")}
        return None

    def benchmarked_shapes_for(self, model_id: str) -> list[str]:
        """Shapes Oracle published measurements for on this model.

        These are not always importable — the gpt-oss grids run on an
        OpenAI-reserved shape — but they are the only configurations that can
        ever be reported as `measured`, so they have to stay reachable rather
        than being filtered out along with the rest of the non-importable shapes.
        """
        return sorted(
            {
                str(grid["shape"])
                for grid in self.benchmark_grids
                if grid.get("hf_id") == model_id and grid.get("shape")
            }
        )

    def has_grid(self, model_id: str, shape_key: str) -> bool:
        return any(
            grid.get("hf_id") == model_id and grid.get("shape") == shape_key
            for grid in self.benchmark_grids
        )

    # ── Pricing ──────────────────────────────────────────────────────────────

    @property
    def price_per_ai_unit_hour(self) -> float:
        return float((self._pricing_raw.get("price_per_ai_unit_hour") or {}).get("value") or 0.0)

    @property
    def pricing(self) -> dict[str, Any]:
        return dict(self._pricing_raw)

    # ── Provenance ───────────────────────────────────────────────────────────

    def provenance(self) -> dict[str, Any]:
        return {
            "models": {
                "generated_at": self._models_raw.get("generated_at"),
                "count": self._models_raw.get("model_count"),
                "with_architecture": self._models_raw.get("architecture_count"),
                "source_urls": self._models_raw.get("source_urls", [])[:1],
            },
            "benchmarks": {
                "generated_at": self._benchmarks_raw.get("generated_at"),
                "grids": self._benchmarks_raw.get("grid_count"),
                "rows": self._benchmarks_raw.get("row_count"),
                "calibration_grids": self._benchmarks_raw.get("calibration_grid_count"),
            },
            "shapes": {"generated_at": self._shapes_raw.get("generated_at"), "count": len(self.shapes)},
            "pricing": {
                "generated_at": self._pricing_raw.get("generated_at"),
                "rate_verified": bool(
                    (self._pricing_raw.get("price_per_ai_unit_hour") or {}).get("verified")
                ),
                "label": (self._pricing_raw.get("price_per_ai_unit_hour") or {}).get("label"),
            },
            "calibration": {
                **self.coefficients.as_dict(),
                "gpus": sorted(self.calibrated_gpus),
                "sample_rows": len(self.calibration_samples),
            },
        }
