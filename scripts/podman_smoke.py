#!/usr/bin/env python3
"""Render and validate one canonical diagram through the real Podman boundary."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import tempfile
from pathlib import Path

from waqil_api.config import Settings
from waqil_api.contracts import ArchitectureSpecV1
from waqil_api.diagram_source import canonical_diagram_source
from waqil_api.reference_architecture import ReferenceArchitectureRunner


EXPECTED_FILES = {
    "architecture-spec.json",
    "diagram.py",
    "architecture.svg",
    "architecture.png",
    "validation-report.json",
}


async def run_smoke(image: str) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="metis-podman-smoke-") as temporary_name:
        data_dir = Path(temporary_name) / "data"
        settings = Settings(
            data_dir=data_dir,
            reference_runner_mode="podman",
            reference_runner_image=image,
            allow_test_backends=False,
        )
        runner = ReferenceArchitectureRunner(settings)
        resolved_image = await runner.resolve_image_ref(image)
        spec = ArchitectureSpecV1.model_validate(
            {
                "title": "Metis Podman Verification",
                "provider": "onprem",
                "direction": "LR",
                "components": [
                    {"id": "client", "label": "Client", "kind": "client"},
                    {"id": "api", "label": "API", "kind": "service"},
                    {"id": "database", "label": "Database", "kind": "database"},
                ],
                "edges": [
                    {"source": "client", "target": "api", "label": "HTTPS"},
                    {"source": "api", "target": "database", "label": "SQL"},
                ],
                "boundaries": [
                    {
                        "id": "private",
                        "label": "Private application boundary",
                        "component_ids": ["api", "database"],
                    }
                ],
                "assumptions": [],
                "unresolved_ambiguities": [],
            }
        )
        source = canonical_diagram_source(spec, ["svg", "png"])
        output = await runner.run(
            "podman-smoke",
            "Render the canonical Podman verification architecture.",
            spec,
            diagram_code=source,
            action_id="podman-smoke:v1",
            image_ref=resolved_image,
        )
        names = {path.name for path in output.files}
        if names != EXPECTED_FILES:
            raise RuntimeError(
                f"sandbox artifact contract mismatch: expected {sorted(EXPECTED_FILES)}, "
                f"received {sorted(names)}"
            )
        artifacts = []
        for path in sorted(output.files):
            content = path.read_bytes()
            if not content:
                raise RuntimeError(f"sandbox created an empty artifact: {path.name}")
            artifacts.append(
                {
                    "name": path.name,
                    "size": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
        if not output.eval_report.passed:
            raise RuntimeError(output.eval_report.model_dump_json())
        return {
            "status": "passed",
            "rootless_runner": True,
            "image": resolved_image,
            "deployment_hash": output.deployment_hash,
            "renderer": output.envelope.get("renderer"),
            "validation": output.envelope.get("validation", {}),
            "artifacts": artifacts,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        default=os.environ.get(
            "WAQIL_REFERENCE_RUNNER_IMAGE",
            "localhost/metis/reference-architecture-tool:0.3.0",
        ),
    )
    arguments = parser.parse_args()
    print(json.dumps(asyncio.run(run_smoke(arguments.image)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
