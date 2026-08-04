"""Reviewed host wrapper that runs the project-verify sandbox under Podman.

This file owns every container flag, exactly as the reference-architecture
wrapper does for diagrams: the API process never invokes Podman itself, so the
containment a project verification runs under can be reviewed by reading one
place. There is no fallback to executing project code on the host — if the
sandbox is unavailable the wrapper says so and the caller degrades to the static
checks it can do without executing anything.

Contract: a JSON request on stdin, a project directory to mount read-only, and a
single JSON envelope on stdout.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1"
POLICY_PATH = Path(__file__).with_name("sandbox-policy.json")
SAFE_DIGEST_RE = re.compile(r"sha256:[a-f0-9]{64}")


class RunnerFailure(Exception):
    """A refusal to run, carrying the code and exit status the host reads."""

    def __init__(
        self, code: str, message: str, *, exit_code: int, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.exit_code = exit_code
        self.details = details or {}


def _fail(code: str, message: str, *, exit_code: int, details: dict[str, Any] | None = None):
    """Abort with a structured failure rather than a traceback."""
    raise RunnerFailure(code, message, exit_code=exit_code, details=details)


def _load_policy() -> dict[str, Any]:
    """The reviewed containment policy this wrapper is required to apply."""
    try:
        policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _fail("POLICY_UNREADABLE", "sandbox policy could not be read", exit_code=10,
              details={"type": type(exc).__name__})
    if policy.get("schema_version") != SCHEMA_VERSION or policy.get("network") != "none":
        _fail("POLICY_INVALID", "sandbox policy is not the reviewed shape", exit_code=10)
    return policy


def _read_request(maximum: int) -> bytes:
    """Read the host's request, refusing anything larger than the policy allows."""
    raw = sys.stdin.buffer.read(maximum + 1)
    if len(raw) > maximum:
        _fail("REQUEST_TOO_LARGE", f"request exceeds {maximum} bytes", exit_code=10)
    try:
        json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        _fail("REQUEST_INVALID", "request is not valid JSON", exit_code=10,
              details={"type": type(exc).__name__})
    return raw


def _resolve_project(path: Path, limits: dict[str, Any]) -> Path:
    """Check the directory to mount exists and is within the policy's bounds."""
    resolved = path.expanduser().resolve(strict=False)
    if not resolved.is_dir():
        _fail("INPUT_UNAVAILABLE", f"project directory not found: {resolved}", exit_code=10)
    files = 0
    total = 0
    for entry in resolved.rglob("*"):
        if entry.is_symlink():
            _fail("INPUT_UNSAFE", "project directory contains a symlink", exit_code=10)
        if not entry.is_file():
            continue
        files += 1
        total += entry.stat().st_size
        if files > limits["max_input_files"] or total > limits["max_input_bytes"]:
            _fail("INPUT_TOO_LARGE", "project exceeds the sandbox input limits", exit_code=10,
                  details={"files": files, "bytes": total})
    return resolved


def _validate_image(image: str, expected: str) -> None:
    """Allow only the policy's image, or a digest-pinned form of that image."""
    if image == expected:
        return
    repository = expected.rsplit(":", 1)[0]
    prefix = f"{repository}@"
    if image.startswith(prefix) and SAFE_DIGEST_RE.fullmatch(image[len(prefix):]):
        return
    _fail("UNAPPROVED_IMAGE",
          "image must be the policy tag or a digest-pinned form of the same repository",
          exit_code=10)


def _require_rootless_podman(podman: str) -> None:
    """Refuse to run anywhere but a reachable, rootless Podman."""
    try:
        result = subprocess.run(
            [podman, "info", "--format", "json"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=15, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _fail("PODMAN_UNAVAILABLE", "cannot query Podman", exit_code=11,
              details={"type": type(exc).__name__})
    if result.returncode != 0:
        _fail("PODMAN_UNAVAILABLE", "Podman is not ready", exit_code=11,
              details={"stderr": result.stderr.decode("utf-8", errors="replace")[:1_000]})
    try:
        info = json.loads(result.stdout)
        rootless = info["host"]["security"]["rootless"]
    except (KeyError, TypeError, ValueError) as exc:
        _fail("PODMAN_INFO_INVALID", "cannot verify Podman rootless mode", exit_code=11,
              details={"type": type(exc).__name__})
    if rootless is not True:
        _fail("ROOTLESS_REQUIRED", "sandbox execution requires rootless Podman", exit_code=11)


def _cleanup_container(podman: str, cidfile: Path) -> None:
    """Remove the container even if the run was interrupted before --rm fired."""
    try:
        container_id = cidfile.read_text(encoding="utf-8").strip()
    except OSError:
        return
    if not container_id:
        return
    subprocess.run(
        [podman, "rm", "--force", container_id],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        timeout=30, check=False,
    )


def _run_bounded(
    command: list[str], request: bytes, *, timeout_seconds: int, max_output_bytes: int
) -> tuple[int, bytes, bytes, bool]:
    """Run the container to completion, or kill its whole process group trying."""
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(input=request, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(os.getpgid(process.pid), 9)
        except (ProcessLookupError, PermissionError):
            process.kill()
        stdout, stderr = process.communicate()
    return process.returncode, stdout[:max_output_bytes], stderr[:max_output_bytes], timed_out


def run(arguments: argparse.Namespace) -> tuple[dict[str, Any], int]:
    """Apply the policy, run the sandbox, and return its envelope."""
    policy = _load_policy()
    limits = policy["limits"]
    request = _read_request(limits["max_stdin_bytes"])
    project = _resolve_project(arguments.project_dir, limits)
    _validate_image(arguments.image, policy["image"])

    podman = shutil.which("podman")
    if podman is None:
        _fail("PODMAN_UNAVAILABLE", "podman executable is not available", exit_code=11)
    _require_rootless_podman(podman)

    tmpfs = policy["tmpfs"]
    uid, gid = os.getuid(), os.getgid()
    with tempfile.TemporaryDirectory(prefix="metis-verify-") as control_dir:
        cidfile = Path(control_dir) / "container.cid"
        command = [
            podman, "run", "--rm", "--interactive",
            "--pull=never",
            f"--cidfile={cidfile}",
            "--network=none",
            "--read-only",
            "--read-only-tmpfs=false",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--ipc=none",
            "--log-driver=none",
            "--userns=keep-id",
            f"--user={uid}:{gid}",
            f"--pids-limit={limits['pids']}",
            f"--cpus={limits['cpus']}",
            f"--memory={limits['memory_mib']}m",
            f"--memory-swap={limits['memory_mib']}m",
            f"--tmpfs={tmpfs['path']}:{','.join(tmpfs['options'])},size={tmpfs['size_mib']}m",
            f"--mount=type=bind,src={project},target=/input,ro=true",
            "--workdir=/opt/metis/verify",
            arguments.image,
        ]
        try:
            returncode, stdout, stderr, timed_out = _run_bounded(
                command, request,
                timeout_seconds=limits["timeout_seconds"],
                max_output_bytes=limits["max_logs_bytes"],
            )
        except OSError as exc:
            _fail("PODMAN_START_FAILED", "could not start the sandbox", exit_code=11,
                  details={"type": type(exc).__name__})
        finally:
            _cleanup_container(podman, cidfile)

    if timed_out:
        _fail("SANDBOX_TIMEOUT", f"sandbox exceeded {limits['timeout_seconds']} seconds",
              exit_code=11)
    try:
        envelope = json.loads(stdout.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        _fail("INVALID_TOOL_RESULT", "sandbox did not return a JSON envelope", exit_code=12,
              details={
                  "type": type(exc).__name__,
                  "returncode": returncode,
                  "stderr": stderr.decode("utf-8", errors="replace")[:2_000],
              })
    if not isinstance(envelope, dict) or envelope.get("schema_version") != SCHEMA_VERSION:
        _fail("INVALID_TOOL_RESULT", "sandbox envelope is not the expected shape", exit_code=12)
    if envelope.get("status") != "succeeded":
        return envelope, 1
    return envelope, 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run, and always print exactly one JSON envelope."""
    parser = argparse.ArgumentParser(description="Run the Metis project verify sandbox")
    parser.add_argument("--project-dir", type=Path, required=True,
                        help="materialized project directory, mounted read-only")
    parser.add_argument("--image", default="localhost/metis/project-verify:0.2.0",
                        help="approved local image tag or digest-pinned equivalent")
    arguments = parser.parse_args(argv)
    try:
        envelope, exit_code = run(arguments)
    except RunnerFailure as exc:
        envelope = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "error": {"code": exc.code, "message": exc.message, "details": exc.details},
            "checks": [],
            "routes": [],
        }
        exit_code = exc.exit_code
    print(json.dumps(envelope))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
