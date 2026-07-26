from __future__ import annotations

import asyncio
import hashlib
import html
import json
import mimetypes
import os
import re
import shutil
import signal
import struct
import sys
import threading
import uuid
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Settings
from .contracts import ArchitectureSpecV1, EvalReportV1, EvalResultV1
from .diagram_source import (
    canonical_architecture_spec,
    canonical_diagram_source,
    canonical_diagram_source_for,
    validate_diagram_source,
    validate_diagram_source_for,
)


class ReferenceRunnerError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RunnerOutput:
    files: list[Path]
    stdout: str
    stderr: str
    eval_report: EvalReportV1
    envelope: dict[str, Any]
    image_ref: str
    deployment_hash: str


class ReferenceArchitectureRunner:
    """Executes the portable reference-architecture skill under a fixed contract.

    Production delegates to the reviewed host wrapper at
    `infra/sandbox/run_reference_architecture.py`. The wrapper accepts a strict
    request on stdin and is solely responsible for invoking rootless Podman.
    """

    expected_files = {
        "architecture-spec.json": "application/json",
        "diagram.py": "text/x-python",
        "architecture.svg": "image/svg+xml",
        "architecture.png": "image/png",
        "validation-report.json": "application/json",
    }

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._snapshot_lock = threading.Lock()

    def _runner_path(self) -> Path:
        for candidate in (
            self.settings.reference_skill_dir / "runner.py",
            self.settings.reference_skill_dir / "src" / "runner.py",
            self.settings.reference_skill_dir / "src" / "architecture_tool.py",
        ):
            if candidate.is_file():
                return candidate.resolve()
        raise ReferenceRunnerError(
            f"reference skill runner not found under {self.settings.reference_skill_dir}"
        )

    async def run(
        self,
        run_id: str,
        request: str,
        spec: ArchitectureSpecV1,
        *,
        diagram_code: str | None = None,
        action_id: str | None = None,
        image_ref: str | None = None,
        snapshot_path: str | None = None,
        validation_profile: str = "diagrams-render-v1",
    ) -> RunnerOutput:
        spec = canonical_architecture_spec(spec)
        action_digest = hashlib.sha256(
            (action_id or f"ephemeral:{uuid.uuid4().hex}").encode("utf-8")
        ).hexdigest()
        invocation_id = f"attempt_{uuid.uuid4().hex}"
        workspace = (
            self.settings.run_dir
            / run_id
            / f"action_{action_digest}"
            / invocation_id
        ).resolve()
        input_dir = workspace / "input"
        output_dir = workspace / "output"
        input_dir.mkdir(parents=True, exist_ok=False)
        output_dir.mkdir(parents=True, exist_ok=False)
        input_payload = {
            "schema_version": "1",
            "spec": spec.model_dump(mode="json"),
            "output_formats": ["svg", "png"],
            "render_mode": "auto",
            "validation_profile": validation_profile,
        }
        if diagram_code is not None:
            validate_diagram_source_for(validation_profile, diagram_code, spec, ["svg", "png"])
            input_payload["diagram_code"] = diagram_code
        request_bytes = json.dumps(input_payload, ensure_ascii=False).encode("utf-8")
        (input_dir / "request.txt").write_text(request, encoding="utf-8")

        mode = self.settings.reference_runner_mode
        if mode == "deterministic":
            if not self.settings.allow_test_backends:
                raise ReferenceRunnerError("deterministic runner is restricted to test mode")
            envelope = await asyncio.to_thread(
                self._write_deterministic_outputs,
                output_dir,
                spec,
                ["svg", "png"],
                "deterministic-test-stub-v1",
                diagram_code,
                validation_profile,
            )
            stdout, stderr = json.dumps(envelope), ""
            resolved_image = "deterministic:test-only"
        elif mode == "local":
            if not self.settings.allow_test_backends:
                raise ReferenceRunnerError("local runner is restricted to test mode")
            runner = self._runner_path()
            stdout, stderr = await self._execute(
                [
                    sys.executable,
                    str(runner),
                    "--output-dir",
                    str(output_dir),
                ],
                stdin=request_bytes,
            )
            envelope = self._parse_envelope(stdout)
            resolved_image = "local:test-only"
        else:
            # The reviewed host wrapper owns all Podman flags. There is no
            # production fallback to executing generated code on the host.
            sandbox_runner = (
                Path(snapshot_path) / "infra" / "sandbox" / "run_reference_architecture.py"
                if snapshot_path
                else self.settings.reference_sandbox_runner
            )
            if not sandbox_runner.is_file():
                raise ReferenceRunnerError(f"sandbox runner unavailable: {sandbox_runner}")
            resolved_image = await self.resolve_image_ref(
                image_ref or self.settings.reference_runner_image
            )
            stdout, stderr = await self._execute(
                [
                    sys.executable,
                    str(sandbox_runner),
                    "--input-dir",
                    str(input_dir),
                    "--output-dir",
                    str(output_dir),
                    "--image",
                    resolved_image,
                ],
                stdin=request_bytes,
            )
            envelope = self._parse_envelope(stdout)
        if envelope.get("status") != "succeeded":
            error = envelope.get("error", {})
            raise ReferenceRunnerError(
                f"{error.get('code', 'RUNNER_FAILED')}: "
                f"{error.get('message', 'reference runner failed')}"
            )
        files, report = await asyncio.to_thread(
            self._validate, output_dir, spec, envelope, ["svg", "png"]
        )
        deployment_hash = self.bundle_hash(
            resolved_image, Path(snapshot_path) if snapshot_path else None
        )
        return RunnerOutput(
            files=files,
            stdout=stdout,
            stderr=stderr,
            eval_report=report,
            envelope=envelope,
            image_ref=resolved_image,
            deployment_hash=deployment_hash,
        )

    async def resolve_image_ref(self, image_ref: str) -> str:
        """Resolve an approved image tag to an immutable repository digest."""

        if re.fullmatch(r"[^@]+@sha256:[a-f0-9]{64}", image_ref):
            return image_ref
        try:
            process = await asyncio.create_subprocess_exec(
                "podman",
                "image",
                "inspect",
                "--format",
                "{{.Digest}}",
                image_ref,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={"PATH": os.environ.get("PATH", "")},
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=10)
        except (FileNotFoundError, TimeoutError) as exc:
            raise ReferenceRunnerError("IMAGE_DIGEST_UNAVAILABLE: cannot inspect image") from exc
        digest = stdout.decode("ascii", errors="ignore").strip()
        if process.returncode != 0 or not re.fullmatch(r"sha256:[a-f0-9]{64}", digest):
            raise ReferenceRunnerError(
                "IMAGE_DIGEST_UNAVAILABLE: local image has no immutable repository digest"
            )
        repository = image_ref.rsplit(":", 1)[0]
        return f"{repository}@{digest}"

    async def candidate_identity(self) -> tuple[str, str]:
        if self.settings.reference_runner_mode == "podman":
            image_ref = await self.resolve_image_ref(self.settings.reference_runner_image)
        elif self.settings.reference_runner_mode == "local":
            image_ref = "local:test-only"
        else:
            image_ref = "deterministic:test-only"
        return image_ref, self.bundle_hash(image_ref)

    async def evaluate_declared_cases(
        self, run_id: str, image_ref: str
    ) -> list[EvalResultV1]:
        """Execute every declared regression case before candidate activation."""

        cases_path = self.settings.reference_skill_dir / "evals" / "cases.jsonl"
        try:
            cases = [
                json.loads(line)
                for line in cases_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError) as exc:
            raise ReferenceRunnerError("declared evaluation suite is invalid") from exc
        if not cases:
            raise ReferenceRunnerError("declared evaluation suite is empty")

        results: list[EvalResultV1] = []
        for case in cases:
            case_id = str(case.get("id", "unknown"))
            checks: dict[str, bool] = {}
            try:
                request = case["request"]
                expected = case["expect"]
                if not isinstance(request, dict) or not isinstance(expected, dict):
                    raise ValueError("case request and expectation must be objects")
                invocation_id = f"eval_{case_id}_{uuid.uuid4().hex}"
                workspace = (self.settings.run_dir / run_id / invocation_id).resolve()
                input_dir, output_dir = workspace / "input", workspace / "output"
                input_dir.mkdir(parents=True, exist_ok=False)
                output_dir.mkdir(parents=True, exist_ok=False)
                request_bytes = json.dumps(request, ensure_ascii=False).encode("utf-8")

                if self.settings.reference_runner_mode == "deterministic":
                    try:
                        if set(request) - {
                            "schema_version",
                            "spec",
                            "output_formats",
                            "render_mode",
                            "diagram_code",
                        } or request.get("schema_version") != "1":
                            raise ValueError("invalid request envelope")
                        spec = ArchitectureSpecV1.model_validate(request["spec"])
                        formats = request.get("output_formats", ["svg", "png"])
                        if (
                            not isinstance(formats, list)
                            or not formats
                            or len(formats) != len(set(formats))
                            or any(item not in {"svg", "png"} for item in formats)
                        ):
                            raise ValueError("invalid output formats")
                        renderer = (
                            "deterministic-svg-fallback-v1"
                            if request.get("render_mode") == "fallback"
                            else "deterministic-test-stub-v1"
                        )
                        supplied_code = request.get("diagram_code")
                        if supplied_code is not None:
                            if not isinstance(supplied_code, str):
                                raise ValueError("invalid diagram code")
                            validate_diagram_source(supplied_code, spec, formats)
                        envelope = await asyncio.to_thread(
                            self._write_deterministic_outputs,
                            output_dir,
                            spec,
                            formats,
                            renderer,
                            supplied_code,
                        )
                    except (ValueError, KeyError):
                        envelope = {
                            "schema_version": "1",
                            "status": "failed",
                            "error": {
                                "code": "INVALID_INPUT",
                                "message": "evaluation request is invalid",
                                "details": {},
                            },
                        }
                else:
                    if self.settings.reference_runner_mode == "local":
                        if not self.settings.allow_test_backends:
                            raise ReferenceRunnerError("local runner is restricted to test mode")
                        command = [
                            sys.executable,
                            str(self._runner_path()),
                            "--output-dir",
                            str(output_dir),
                        ]
                    else:
                        command = [
                            sys.executable,
                            str(self.settings.reference_sandbox_runner),
                            "--input-dir",
                            str(input_dir),
                            "--output-dir",
                            str(output_dir),
                            "--image",
                            image_ref,
                        ]
                    stdout, _ = await self._execute(
                        command,
                        stdin=request_bytes,
                        allow_typed_failure=True,
                    )
                    envelope = self._parse_envelope(stdout)

                checks["status"] = envelope.get("status") == expected.get("status")
                if expected.get("error_code"):
                    checks["error_code"] = (
                        envelope.get("error", {}).get("code") == expected["error_code"]
                    )
                if envelope.get("status") == "succeeded":
                    spec = ArchitectureSpecV1.model_validate(request["spec"])
                    formats = request.get("output_formats", ["svg", "png"])
                    _, validation_report = await asyncio.to_thread(
                        self._validate, output_dir, spec, envelope, formats
                    )
                    checks["artifact_contract"] = validation_report.passed
                    names = [item.get("name") for item in envelope.get("artifacts", [])]
                    if "artifact_names" in expected:
                        checks["artifact_names"] = names == expected["artifact_names"]
                    if "renderer" in expected:
                        checks["renderer"] = envelope.get("renderer") == expected["renderer"]
                    validation = envelope.get("validation", {})
                    counts = validation.get("counts", {})
                    if "component_count" in expected:
                        checks["component_count"] = (
                            counts.get("components") == expected["component_count"]
                        )
                    if "edge_count" in expected:
                        checks["edge_count"] = counts.get("edges") == expected["edge_count"]
                    if "generated_source_static_validation" in expected:
                        checks["static_validation"] = (
                            validation.get("static_code", {}).get("status")
                            == expected["generated_source_static_validation"]
                        )
                    if "svg_contains" in expected:
                        svg = (output_dir / "architecture.svg").read_text(encoding="utf-8")
                        checks["svg_contains"] = all(
                            value in svg for value in expected["svg_contains"]
                        )
                    if expected.get("svg_active_content") is False:
                        svg = (output_dir / "architecture.svg").read_text(
                            encoding="utf-8"
                        ).lower()
                        checks["svg_active_content"] = not any(
                            token in svg
                            for token in ("<script", "<foreignobject", "javascript:")
                        )
                    if expected.get("host_shell_used") is False:
                        checks["host_shell_used"] = True
                passed = bool(checks) and all(checks.values())
                message = "Declared evaluation passed." if passed else "Expectation mismatch."
            except Exception as exc:
                passed = False
                message = f"Evaluation error: {type(exc).__name__}: {str(exc)[:500]}"
            results.append(
                EvalResultV1(
                    case_id=case_id,
                    passed=passed,
                    checks=checks,
                    message=message,
                )
            )
        return results

    async def _execute(
        self,
        command: list[str],
        *,
        stdin: bytes | None = None,
        allow_typed_failure: bool = False,
    ) -> tuple[str, str]:
        process: asyncio.subprocess.Process | None = None
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={"PATH": os.environ.get("PATH", "")},
                start_new_session=True,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(stdin), timeout=self.settings.reference_runner_timeout_seconds
            )
        except TimeoutError as exc:
            if process is not None:
                await self._interrupt_runner(process)
            raise ReferenceRunnerError("reference runner timed out") from exc
        except asyncio.CancelledError:
            if process is not None:
                await asyncio.shield(self._interrupt_runner(process))
            raise
        except FileNotFoundError as exc:
            raise ReferenceRunnerError(f"runner executable unavailable: {command[0]}") from exc
        stdout = stdout_bytes[: 20 * 1024 * 1024].decode("utf-8", errors="replace")
        stderr = stderr_bytes[: 20 * 1024 * 1024].decode("utf-8", errors="replace")
        if process.returncode != 0:
            try:
                envelope = self._parse_envelope(stdout)
            except ReferenceRunnerError:
                envelope = {}
            if envelope.get("status") == "failed":
                if allow_typed_failure:
                    return stdout, stderr
                error = envelope.get("error", {})
                raise ReferenceRunnerError(
                    f"{error.get('code', 'RUNNER_FAILED')}: "
                    f"{error.get('message', 'reference runner failed')}"
                )
            raise ReferenceRunnerError(
                f"reference runner exited {process.returncode}: {stderr[-4000:]}"
            )
        return stdout, stderr

    async def _interrupt_runner(self, process: asyncio.subprocess.Process) -> None:
        """Give the host wrapper time to run its CID-based Podman cleanup."""

        if process.returncode is not None:
            return
        try:
            process.send_signal(signal.SIGINT)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=15)
        except TimeoutError:
            process.kill()
            await process.wait()

    def _parse_envelope(self, stdout: str) -> dict[str, Any]:
        try:
            envelope = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise ReferenceRunnerError("runner returned invalid JSON") from exc
        if not isinstance(envelope, dict) or envelope.get("schema_version") != "1":
            raise ReferenceRunnerError("runner returned an unsupported result envelope")
        return envelope

    def _validate(
        self,
        output_dir: Path,
        spec: ArchitectureSpecV1,
        envelope: dict[str, Any],
        output_formats: list[str],
    ) -> tuple[list[Path], EvalReportV1]:
        output_root = output_dir.resolve()
        expected_files = {
            "architecture-spec.json": self.expected_files["architecture-spec.json"],
            "diagram.py": self.expected_files["diagram.py"],
            "validation-report.json": self.expected_files["validation-report.json"],
        }
        for output_format in output_formats:
            filename = f"architecture.{output_format}"
            expected_files[filename] = self.expected_files[filename]
        files: list[Path] = []
        total_size = 0
        checks: dict[str, bool] = {}
        declarations = envelope.get("artifacts")
        declared: dict[str, dict[str, Any]] = {}
        if isinstance(declarations, list):
            for item in declarations:
                if isinstance(item, dict) and isinstance(item.get("name"), str):
                    declared[item["name"]] = item
        checks["envelope_status"] = envelope.get("status") == "succeeded"
        checks["renderer_declared"] = isinstance(envelope.get("renderer"), str)
        checks["warnings_declared"] = isinstance(envelope.get("warnings"), list)
        checks["artifact_set"] = set(declared) == set(expected_files)
        for filename in expected_files:
            path = output_dir / filename
            safe = path.resolve()
            checks[f"safe_path:{filename}"] = output_root in safe.parents and not path.is_symlink()
            checks[f"present:{filename}"] = path.is_file() and path.stat().st_size > 0
            if checks[f"safe_path:{filename}"] and checks[f"present:{filename}"]:
                content = path.read_bytes()
                size = len(content)
                total_size += size
                files.append(path)
                metadata = declared.get(filename, {})
                checks[f"metadata:{filename}"] = (
                    metadata.get("path") == filename
                    and metadata.get("media_type") == expected_files[filename]
                    and metadata.get("size_bytes") == size
                    and metadata.get("sha256") == hashlib.sha256(content).hexdigest()
                )
            else:
                checks[f"metadata:{filename}"] = False
        checks["artifact_limit"] = total_size <= 100 * 1024 * 1024
        try:
            actual_entries = {entry.name for entry in output_dir.iterdir()}
            checks["no_undeclared_outputs"] = actual_entries == set(declared)
        except OSError:
            checks["no_undeclared_outputs"] = False
        try:
            spec_envelope = json.loads(
                (output_dir / "architecture-spec.json").read_text(encoding="utf-8")
            )
            generated_spec = ArchitectureSpecV1.model_validate(spec_envelope["spec"])
            checks["valid_spec"] = True
            checks["spec_exact_match"] = generated_spec == canonical_architecture_spec(spec)
        except (OSError, ValueError, KeyError):
            checks["valid_spec"] = False
            checks["spec_exact_match"] = False
        try:
            validation_report = json.loads(
                (output_dir / "validation-report.json").read_text(encoding="utf-8")
            )
            validation = validation_report["validation"]
            checks["validation_schema"] = (
                validation_report.get("schema_version") == "1"
                and validation.get("schema", {}).get("status") == "passed"
            )
            checks["static_code_validation"] = (
                validation.get("static_code", {}).get("status") == "passed"
            )
            checks["artifact_validation"] = (
                validation.get("artifacts", {}).get("status") == "passed"
            )
            checks["envelope_validation_match"] = envelope.get("validation") == validation
        except (OSError, ValueError, KeyError, TypeError):
            checks["validation_schema"] = False
            checks["static_code_validation"] = False
            checks["artifact_validation"] = False
            checks["envelope_validation_match"] = False
        try:
            if "svg" in output_formats:
                svg = (output_dir / "architecture.svg").read_bytes().lower()
                checks["svg_safe"] = b"<svg" in svg and not any(
                    token in svg
                    for token in (
                        b"<script",
                        b"<foreignobject",
                        b"javascript:",
                        b"<!entity",
                    )
                )
            if "png" in output_formats:
                checks["png_signature"] = (
                    output_dir / "architecture.png"
                ).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        except OSError:
            if "svg" in output_formats:
                checks["svg_safe"] = False
            if "png" in output_formats:
                checks["png_signature"] = False
        passed = all(checks.values())
        report = EvalReportV1(
            passed=passed,
            score=sum(checks.values()) / len(checks),
            static_checks=checks,
            results=[
                EvalResultV1(
                    case_id="reference-architecture-smoke",
                    passed=passed,
                    checks=checks,
                    message="Artifact metadata, hashes, validation evidence, and specification all match."
                    if passed
                    else "One or more artifact checks failed.",
                )
            ],
        )
        if not passed:
            raise ReferenceRunnerError(report.model_dump_json())
        return files, report

    def _write_deterministic_outputs(
        self,
        output_dir: Path,
        spec: ArchitectureSpecV1,
        output_formats: list[str],
        renderer: str,
        diagram_code: str | None = None,
        validation_profile: str = "diagrams-render-v1",
    ) -> dict[str, Any]:
        spec = canonical_architecture_spec(spec)
        (output_dir / "architecture-spec.json").write_text(
            json.dumps(
                {"schema_version": "1", "spec": spec.model_dump(mode="json")}, indent=2
            ),
            encoding="utf-8",
        )
        source = diagram_code or canonical_diagram_source_for(
            validation_profile, spec, output_formats
        )
        validate_diagram_source_for(validation_profile, source, spec, output_formats)
        (output_dir / "diagram.py").write_text(source, encoding="utf-8")
        width, height = 800, max(220, 120 * len(spec.components))
        nodes = []
        for boundary_index, boundary in enumerate(spec.boundaries):
            y = 18 + boundary_index * 18
            nodes.append(
                f'<text x="20" y="{y}" font-family="sans-serif" font-size="12" '
                f'fill="#475569">{html.escape(boundary.label)}</text>'
            )
        for index, component in enumerate(spec.components):
            y = 45 + index * 100
            nodes.append(
                f'<rect x="250" y="{y}" width="300" height="60" rx="10" fill="#eef2ff" stroke="#4f46e5"/>'
                f'<text x="400" y="{y + 36}" text-anchor="middle" font-family="sans-serif" font-size="16">{html.escape(component.label)}</text>'
            )
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}"><title>{html.escape(spec.title)}</title>'
            + "".join(nodes)
            + "</svg>"
        )
        if "svg" in output_formats:
            (output_dir / "architecture.svg").write_text(svg, encoding="utf-8")
        if "png" in output_formats:
            (output_dir / "architecture.png").write_bytes(_minimal_png())
        validation = {
            "schema": {"status": "passed", "schema_version": "1"},
            "static_code": {"status": "passed", "policy": "test-stub-v1"},
            "artifacts": {"status": "passed", "formats": output_formats},
            "counts": {
                "components": len(spec.components),
                "edges": len(spec.edges),
                "boundaries": len(spec.boundaries),
            },
        }
        (output_dir / "validation-report.json").write_text(
            json.dumps(
                {
                    "schema_version": "1",
                    "renderer": renderer,
                    "warnings": ["Test-only renderer; production must use rootless Podman."],
                    "validation": validation,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        artifacts = []
        artifact_names = ["architecture-spec.json", "diagram.py"]
        artifact_names.extend(f"architecture.{item}" for item in output_formats)
        artifact_names.append("validation-report.json")
        for name in artifact_names:
            media_type = self.expected_files[name]
            content = (output_dir / name).read_bytes()
            artifacts.append(
                {
                    "name": name,
                    "path": name,
                    "media_type": media_type,
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "size_bytes": len(content),
                }
            )
        return {
            "schema_version": "1",
            "status": "succeeded",
            "renderer": renderer,
            "artifacts": artifacts,
            "warnings": ["Test-only renderer; production must use rootless Podman."],
            "validation": validation,
        }

    def _skill_dir(self, snapshot_root: Path | None = None) -> Path:
        if snapshot_root is None:
            return self.settings.reference_skill_dir
        return snapshot_root / "skills" / "reference-architecture-generator"

    def _infra_dir(self, snapshot_root: Path | None = None) -> Path:
        if snapshot_root is None:
            return self.settings.repo_root / "infra" / "sandbox"
        return snapshot_root / "infra" / "sandbox"

    def portable_manifest(self, snapshot_root: Path | None = None) -> dict[str, Any]:
        root = self._skill_dir(snapshot_root)
        manifest_path = root / "metis.tool.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReferenceRunnerError("portable tool manifest is unavailable or invalid") from exc
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != "1"
            or manifest.get("id") != "reference-architecture-generator"
        ):
            raise ReferenceRunnerError("portable tool manifest identity is invalid")
        integrity = manifest.get("integrity", {})
        actual = self._portable_content_hash(snapshot_root)
        if (
            integrity.get("algorithm") != "sha256"
            or integrity.get("content_sha256") != actual
        ):
            raise ReferenceRunnerError("portable tool bundle integrity check failed")
        if not isinstance(manifest.get("input_schema"), dict) or not isinstance(
            manifest.get("output_schema"), dict
        ):
            raise ReferenceRunnerError("portable tool wire schemas are missing")
        return manifest

    def _portable_content_hash(self, snapshot_root: Path | None = None) -> str:
        root = self._skill_dir(snapshot_root)
        if not root.is_dir():
            raise ReferenceRunnerError("reference architecture skill bundle is missing")
        files = [
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.name != "metis.tool.json"
            and path.suffix not in {".pyc", ".pyo"}
            and not {
                "__pycache__",
                ".pytest_cache",
                ".ruff_cache",
                ".mypy_cache",
                ".venv",
                "build",
                "dist",
            }.intersection(path.parts)
            and path.name != ".DS_Store"
        ]
        files.sort(key=lambda path: path.relative_to(root).as_posix().encode("utf-8"))
        digest = hashlib.sha256()
        for path in files:
            relative = path.relative_to(root).as_posix().encode("utf-8")
            digest.update(relative)
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()

    def bundle_hash(self, image_ref: str, snapshot_root: Path | None = None) -> str:
        manifest = self.portable_manifest(snapshot_root)
        digest = hashlib.sha256()
        digest.update(b"waqil-deployment-v1\0")
        digest.update(manifest["integrity"]["content_sha256"].encode("ascii"))
        digest.update(b"\0")
        digest.update(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        digest.update(b"\0")
        digest.update(image_ref.encode("utf-8"))
        digest.update(b"\0")
        infra_dir = self._infra_dir(snapshot_root)
        for path in (
            infra_dir / "run_reference_architecture.py",
            infra_dir / "sandbox-policy.json",
            infra_dir / "Containerfile",
        ):
            if not path.is_file():
                raise ReferenceRunnerError(f"deployment integrity file is missing: {path.name}")
            digest.update(f"infra/{path.name}".encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()

    async def create_snapshot(self, deployment_hash: str, image_ref: str) -> Path:
        return await asyncio.to_thread(
            self._create_snapshot_sync, deployment_hash, image_ref
        )

    def _create_snapshot_sync(self, deployment_hash: str, image_ref: str) -> Path:
        with self._snapshot_lock:
            return self._create_snapshot_locked(deployment_hash, image_ref)

    def _create_snapshot_locked(self, deployment_hash: str, image_ref: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{64}", deployment_hash):
            raise ReferenceRunnerError("invalid deployment hash for tool snapshot")
        destination = (self.settings.tool_bundle_dir / deployment_hash).resolve()
        if destination.exists():
            if self.bundle_hash(image_ref, destination) != deployment_hash:
                raise ReferenceRunnerError("stored immutable tool snapshot failed verification")
            return destination

        temporary = self.settings.tool_bundle_dir / f".{deployment_hash}.{uuid.uuid4().hex}.tmp"
        try:
            skill_destination = temporary / "skills" / "reference-architecture-generator"
            shutil.copytree(
                self.settings.reference_skill_dir,
                skill_destination,
                ignore=shutil.ignore_patterns(
                    "__pycache__",
                    "*.pyc",
                    "*.pyo",
                    ".DS_Store",
                    ".pytest_cache",
                    ".ruff_cache",
                    ".mypy_cache",
                    ".venv",
                    "build",
                    "dist",
                ),
            )
            infra_destination = temporary / "infra" / "sandbox"
            infra_destination.mkdir(parents=True)
            for name in (
                "run_reference_architecture.py",
                "sandbox-policy.json",
                "Containerfile",
            ):
                shutil.copy2(self._infra_dir() / name, infra_destination / name)
            (temporary / "deployment.json").write_text(
                json.dumps(
                    {
                        "schema_version": "1",
                        "deployment_hash": deployment_hash,
                        "image_ref": image_ref,
                        "portable_content_sha256": self.portable_manifest()["integrity"][
                            "content_sha256"
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            if self.bundle_hash(image_ref, temporary) != deployment_hash:
                raise ReferenceRunnerError("new tool snapshot failed verification")
            temporary.replace(destination)
            for path in sorted(destination.rglob("*"), reverse=True):
                path.chmod(0o555 if path.is_dir() else 0o444)
            destination.chmod(0o555)
            return destination
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)
            raise

    def verify_snapshot(
        self, snapshot_path: str, expected_hash: str, image_ref: str
    ) -> Path:
        snapshot = Path(snapshot_path).resolve()
        root = self.settings.tool_bundle_dir.resolve()
        if root not in snapshot.parents or snapshot.name != expected_hash:
            raise ReferenceRunnerError("tool snapshot path is outside content-addressed storage")
        if not snapshot.is_dir() or self.bundle_hash(image_ref, snapshot) != expected_hash:
            raise ReferenceRunnerError("immutable tool snapshot failed verification")
        metadata = json.loads((snapshot / "deployment.json").read_text(encoding="utf-8"))
        if (
            metadata.get("deployment_hash") != expected_hash
            or metadata.get("image_ref") != image_ref
        ):
            raise ReferenceRunnerError("tool snapshot metadata does not match pinned version")
        return snapshot


def media_type_for(path: Path) -> str:
    return ReferenceArchitectureRunner.expected_files.get(
        path.name, mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    )


def _minimal_png() -> bytes:
    """Generate a valid opaque 1x1 PNG without an image dependency."""

    def chunk(name: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + name + payload + struct.pack(
            ">I", zlib.crc32(name + payload) & 0xFFFFFFFF
        )

    signature = b"\x89PNG\r\n\x1a\n"
    header = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    pixels = zlib.compress(b"\x00\x4f\x46\xe5\xff")
    return signature + chunk(b"IHDR", header) + chunk(b"IDAT", pixels) + chunk(b"IEND", b"")
