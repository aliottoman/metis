#!/usr/bin/env python3
"""Invoke the reference-architecture tool in a locked-down rootless Podman container.

Protocol: read one JSON object from stdin and emit one JSON result object to stdout.
All diagnostics are represented as typed failures so API callers never need to parse
Podman text.  This runner intentionally has no option that enables networking,
additional mounts, secrets, or elevated privileges.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterator, NoReturn


SCHEMA_VERSION = "1"
POLICY_PATH = Path(__file__).with_name("sandbox-policy.json")
SAFE_DIGEST_RE = re.compile(r"sha256:[a-f0-9]{64}\Z")
CONTAINER_ID_RE = re.compile(r"[a-f0-9]{12,64}\Z")


class RunnerFailure(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        exit_code: int,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.exit_code = exit_code
        self.details = details or {}


class RunnerSignal(BaseException):
    """A catchable process-termination request used to guarantee cleanup."""

    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


@contextmanager
def _termination_signal_handlers() -> Iterator[None]:
    previous: dict[int, Any] = {}
    received: int | None = None

    def handle(signum: int, _frame: Any) -> None:
        nonlocal received
        # Let the first catchable termination signal unwind through cleanup.
        # Ignore repeats while cleanup is in progress; SIGKILL remains absolute.
        if received is None:
            received = signum
            raise RunnerSignal(signum)

    handled = (signal.SIGINT, signal.SIGTERM)
    try:
        for signum in handled:
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, handle)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


@contextmanager
def _block_termination_signals() -> Iterator[None]:
    """Close the spawn-before-assignment race for catchable termination signals."""

    if not hasattr(signal, "pthread_sigmask"):
        yield
        return
    handled = {signal.SIGINT, signal.SIGTERM}
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, handled)
    try:
        yield
    finally:
        # A pending signal is delivered here, after the caller has stored the
        # Popen object and can therefore terminate its new process group.
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def _fail(
    code: str,
    message: str,
    *,
    exit_code: int,
    details: dict[str, Any] | None = None,
) -> NoReturn:
    raise RunnerFailure(code, message, exit_code=exit_code, details=details)


def _load_policy() -> dict[str, Any]:
    try:
        policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail("SANDBOX_POLICY_INVALID", "cannot load sandbox policy", exit_code=10, details={"type": type(exc).__name__})
    if policy.get("schema_version") != SCHEMA_VERSION:
        _fail("SANDBOX_POLICY_INVALID", "unsupported sandbox policy schema", exit_code=10)
    return policy


def _read_request(maximum: int) -> tuple[bytes, dict[str, Any]]:
    data = sys.stdin.buffer.read(maximum + 1)
    if len(data) > maximum:
        _fail("INPUT_TOO_LARGE", f"stdin exceeds {maximum} bytes", exit_code=10)
    if not data.strip():
        _fail("INVALID_JSON", "stdin must contain one JSON object", exit_code=10)
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail("INVALID_JSON", "stdin is not valid UTF-8 JSON", exit_code=10, details={"reason": str(exc)})
    if not isinstance(value, dict):
        _fail("INVALID_JSON", "stdin JSON must be an object", exit_code=10)
    if value.get("schema_version") != SCHEMA_VERSION:
        _fail("UNSUPPORTED_SCHEMA", "schema_version must be '1'", exit_code=10)
    return data, value


def _resolve_existing_directory(path: Path, label: str) -> Path:
    if not path.is_absolute():
        _fail("UNSAFE_MOUNT", f"{label} must be an absolute path", exit_code=10)
    if path.is_symlink() or not path.exists() or not path.is_dir():
        _fail("UNSAFE_MOUNT", f"{label} must be an existing non-symlink directory", exit_code=10)
    resolved = path.resolve(strict=True)
    if resolved in {Path("/"), Path.home().resolve()}:
        _fail("UNSAFE_MOUNT", f"{label} may not be a broad filesystem root", exit_code=10)
    if "," in str(resolved):
        _fail("UNSAFE_MOUNT", f"{label} may not contain a comma", exit_code=10)
    return resolved


def _inspect_input_tree(path: Path, *, max_files: int, max_bytes: int) -> None:
    entries = 0
    total = 0
    for entry in path.rglob("*"):
        entries += 1
        if entries > max_files:
            _fail("INPUT_TREE_LIMIT", f"input directory exceeds {max_files} entries", exit_code=10)
        if entry.is_symlink():
            _fail("UNSAFE_MOUNT", "input directory may not contain symlinks", exit_code=10)
        if entry.is_file():
            total += entry.stat().st_size
            if total > max_bytes:
                _fail("INPUT_TREE_LIMIT", f"input directory exceeds {max_bytes} bytes", exit_code=10)
        elif not entry.is_dir():
            _fail("UNSAFE_MOUNT", "input directory may contain only regular files and directories", exit_code=10)


def _inspect_output_tree(path: Path, allowed: dict[str, str]) -> None:
    for entry in path.iterdir():
        if entry.name not in allowed:
            _fail("UNSAFE_OUTPUT", f"output directory contains undeclared entry {entry.name!r}", exit_code=10)
        if entry.is_symlink() or not entry.is_file():
            _fail("UNSAFE_OUTPUT", f"output entry {entry.name!r} is not a regular file", exit_code=10)


def _validate_mount_separation(input_dir: Path, output_dir: Path) -> None:
    if input_dir == output_dir or input_dir.is_relative_to(output_dir) or output_dir.is_relative_to(input_dir):
        _fail("UNSAFE_MOUNT", "input and output directories must not overlap", exit_code=10)


def _validate_image(image: str, expected: str) -> None:
    repository = expected.rsplit(":", 1)[0]
    if image == expected:
        return
    prefix = f"{repository}@"
    if image.startswith(prefix) and SAFE_DIGEST_RE.fullmatch(image[len(prefix) :]):
        return
    _fail(
        "UNAPPROVED_IMAGE",
        "image must be the policy tag or a digest-pinned form of the same repository",
        exit_code=10,
    )


def _require_rootless_podman(podman: str) -> None:
    try:
        result = subprocess.run(
            [podman, "info", "--format", "json"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _fail("PODMAN_UNAVAILABLE", "cannot query Podman", exit_code=11, details={"type": type(exc).__name__})
    if result.returncode != 0:
        _fail(
            "PODMAN_UNAVAILABLE",
            "Podman is not ready",
            exit_code=11,
            details={"stderr": result.stderr.decode("utf-8", errors="replace")[:2_000]},
        )
    try:
        info = json.loads(result.stdout)
        rootless = info["host"]["security"]["rootless"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        _fail("PODMAN_INFO_INVALID", "cannot verify Podman rootless mode", exit_code=11, details={"type": type(exc).__name__})
    if rootless is not True:
        _fail("ROOTLESS_REQUIRED", "sandbox execution requires rootless Podman", exit_code=11)


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    timed_out: bool
    output_limited: bool


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_process_group(
    process: subprocess.Popen[bytes], *, grace_seconds: float = 2.0
) -> None:
    """Terminate the isolated child session and reap its leader.

    `start_new_session=True` makes the child's PID its process-group ID. Killing
    that group covers the Podman client and any helpers even when no CID has yet
    been written. CID-based container removal remains a second cleanup layer.
    """

    process_group_id = process.pid
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except PermissionError:
        try:
            process.terminate()
        except ProcessLookupError:
            pass

    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        process.poll()
        if not _process_group_exists(process_group_id):
            break
        time.sleep(0.05)

    if _process_group_exists(process_group_id):
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                process.kill()
            except ProcessLookupError:
                pass

    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def _run_bounded(
    command: list[str],
    request: bytes,
    *,
    timeout_seconds: int,
    max_output_bytes: int,
) -> ProcessResult:
    process: subprocess.Popen[bytes] | None = None
    selector: selectors.BaseSelector | None = None
    buffers: dict[str, bytearray] = {"stdout": bytearray(), "stderr": bytearray()}
    timed_out = False
    output_limited = False
    with tempfile.TemporaryFile() as request_file:
        request_file.write(request)
        request_file.seek(0)
        try:
            with _block_termination_signals():
                process = subprocess.Popen(
                    command,
                    stdin=request_file,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True,
                )
            assert process.stdout is not None
            assert process.stderr is not None
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ, "stdout")
            selector.register(process.stderr, selectors.EVENT_READ, "stderr")
            deadline = time.monotonic() + timeout_seconds

            while selector.get_map():
                now = time.monotonic()
                if now >= deadline:
                    timed_out = True
                    _terminate_process_group(process)
                    break
                events = selector.select(timeout=min(0.2, max(0.0, deadline - now)))
                if not events and process.poll() is not None:
                    events = [
                        (key, selectors.EVENT_READ)
                        for key in list(selector.get_map().values())
                    ]
                for key, _ in events:
                    try:
                        chunk = os.read(key.fileobj.fileno(), 65_536)
                    except OSError:
                        chunk = b""
                    if not chunk:
                        try:
                            selector.unregister(key.fileobj)
                        except KeyError:
                            pass
                        continue
                    buffers[key.data].extend(chunk)
                    if len(buffers["stdout"]) + len(buffers["stderr"]) > max_output_bytes:
                        output_limited = True
                        _terminate_process_group(process)
                        break
                if output_limited:
                    break

            if process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    _terminate_process_group(process)
            returncode = process.returncode if process.returncode is not None else -signal.SIGKILL
            return ProcessResult(
                returncode=returncode,
                stdout=bytes(buffers["stdout"][:max_output_bytes]),
                stderr=bytes(buffers["stderr"][:max_output_bytes]),
                timed_out=timed_out,
                output_limited=output_limited,
            )
        except BaseException:
            if process is not None:
                _terminate_process_group(process)
            raise
        finally:
            if selector is not None:
                selector.close()
            if process is not None:
                if process.stdout is not None:
                    process.stdout.close()
                if process.stderr is not None:
                    process.stderr.close()


def _cleanup_container(podman: str, cidfile: Path) -> None:
    try:
        container_id = cidfile.read_text(encoding="ascii").strip()
    except OSError:
        return
    if not CONTAINER_ID_RE.fullmatch(container_id):
        return
    try:
        subprocess.run(
            [podman, "rm", "--force", container_id],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_success(
    result: dict[str, Any],
    output_dir: Path,
    allowed: dict[str, str],
    request: dict[str, Any],
    max_artifact_bytes: int,
) -> None:
    allowed_keys = {"schema_version", "status", "renderer", "artifacts", "warnings", "validation"}
    if set(result) - allowed_keys:
        _fail("INVALID_TOOL_RESULT", "success result contains unknown fields", exit_code=12)
    artifacts = result.get("artifacts")
    if not isinstance(result.get("renderer"), str) or not isinstance(artifacts, list):
        _fail("INVALID_TOOL_RESULT", "success result is missing renderer or artifacts", exit_code=12)
    if not isinstance(result.get("warnings"), list) or not isinstance(result.get("validation"), dict):
        _fail("INVALID_TOOL_RESULT", "success result is missing validation evidence", exit_code=12)

    requested_formats = request.get("output_formats", ["svg", "png"])
    required = {"architecture-spec.json", "diagram.py", "validation-report.json"}
    if isinstance(requested_formats, list):
        required.update(f"architecture.{item}" for item in requested_formats if item in {"svg", "png"})

    declared: set[str] = set()
    total = 0
    for artifact in artifacts:
        if not isinstance(artifact, dict) or set(artifact) != {"name", "path", "media_type", "sha256", "size_bytes"}:
            _fail("INVALID_TOOL_RESULT", "artifact metadata does not match the contract", exit_code=12)
        name = artifact.get("name")
        if not isinstance(name, str) or name not in allowed or artifact.get("path") != name:
            _fail("INVALID_TOOL_RESULT", "artifact path is not allowlisted and relative", exit_code=12)
        if name in declared:
            _fail("INVALID_TOOL_RESULT", f"artifact {name!r} is duplicated", exit_code=12)
        declared.add(name)
        path = output_dir / name
        if path.is_symlink() or not path.is_file():
            _fail("INVALID_TOOL_RESULT", f"artifact {name!r} is not a regular file", exit_code=12)
        size = path.stat().st_size
        if size <= 0 or size != artifact.get("size_bytes"):
            _fail("INVALID_TOOL_RESULT", f"artifact {name!r} size does not match", exit_code=12)
        total += size
        if total > max_artifact_bytes:
            _fail("ARTIFACT_LIMIT_EXCEEDED", "combined artifacts exceed policy", exit_code=12)
        if artifact.get("media_type") != allowed[name] or artifact.get("sha256") != _sha256(path):
            _fail("INVALID_TOOL_RESULT", f"artifact {name!r} metadata does not match", exit_code=12)

    if not required.issubset(declared):
        _fail("INVALID_TOOL_RESULT", "required artifacts are missing", exit_code=12, details={"missing": sorted(required - declared)})
    actual = {entry.name for entry in output_dir.iterdir()}
    if actual != declared:
        _fail("INVALID_TOOL_RESULT", "output directory and artifact declaration differ", exit_code=12)

    if "architecture.svg" in declared:
        svg = (output_dir / "architecture.svg").read_bytes().lower()
        if b"<svg" not in svg or any(item in svg for item in (b"<script", b"<foreignobject", b"javascript:", b"<!entity")):
            _fail("INVALID_TOOL_RESULT", "SVG failed the host active-content check", exit_code=12)
    if "architecture.png" in declared:
        with (output_dir / "architecture.png").open("rb") as handle:
            if handle.read(8) != b"\x89PNG\r\n\x1a\n":
                _fail("INVALID_TOOL_RESULT", "PNG signature is invalid", exit_code=12)


def _parse_tool_result(stdout: bytes) -> dict[str, Any]:
    try:
        text = stdout.decode("utf-8")
        result = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail("INVALID_TOOL_RESULT", "sandbox did not emit exactly one JSON result", exit_code=12, details={"type": type(exc).__name__})
    if not isinstance(result, dict) or result.get("schema_version") != SCHEMA_VERSION:
        _fail("INVALID_TOOL_RESULT", "sandbox result schema is invalid", exit_code=12)
    if result.get("status") not in {"succeeded", "failed"}:
        _fail("INVALID_TOOL_RESULT", "sandbox result status is invalid", exit_code=12)
    return result


def _failure_envelope(exc: RunnerFailure) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        "error": {
            "code": exc.code,
            "message": exc.message,
            "details": exc.details,
        },
    }


def run(arguments: argparse.Namespace) -> tuple[dict[str, Any], int]:
    policy = _load_policy()
    limits = policy["limits"]
    allowed = policy["allowed_artifacts"]
    request_bytes, request = _read_request(limits["max_stdin_bytes"])
    input_dir = _resolve_existing_directory(arguments.input_dir, "input directory")
    output_dir = _resolve_existing_directory(arguments.output_dir, "output directory")
    _validate_mount_separation(input_dir, output_dir)
    _inspect_input_tree(input_dir, max_files=limits["max_input_files"], max_bytes=limits["max_input_bytes"])
    _inspect_output_tree(output_dir, allowed)
    _validate_image(arguments.image, policy["image"])

    podman = shutil.which("podman")
    if podman is None:
        _fail("PODMAN_UNAVAILABLE", "podman executable is not available", exit_code=11)
    _require_rootless_podman(podman)

    uid = os.getuid()
    gid = os.getgid()
    with tempfile.TemporaryDirectory(prefix="metis-sandbox-") as control_dir_name:
        cidfile = Path(control_dir_name) / "container.cid"
        command = [
            podman,
            "run",
            "--rm",
            "--interactive",
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
            f"--ulimit=nofile={limits['open_files']}:{limits['open_files']}",
            f"--tmpfs=/tmp:rw,noexec,nosuid,nodev,size={policy['tmpfs']['size_mib']}m,mode=1777",
            f"--mount=type=bind,src={input_dir},target=/input,ro=true",
            f"--mount=type=bind,src={output_dir},target=/output,rw=true",
            "--workdir=/opt/metis/tool",
            arguments.image,
        ]
        try:
            outcome = _run_bounded(
                command,
                request_bytes,
                timeout_seconds=limits["timeout_seconds"],
                max_output_bytes=limits["max_logs_bytes"],
            )
        except OSError as exc:
            _fail("PODMAN_START_FAILED", "could not start sandbox process", exit_code=11, details={"type": type(exc).__name__})
        finally:
            try:
                _cleanup_container(podman, cidfile)
            except RunnerSignal:
                # The first signal may land inside CID cleanup rather than the
                # child wait. The installed handler ignores repeats, so retry
                # cleanup before propagating the cancellation to main().
                _cleanup_container(podman, cidfile)
                raise

    if outcome.timed_out:
        _fail("SANDBOX_TIMEOUT", f"sandbox exceeded {limits['timeout_seconds']} seconds", exit_code=11)
    if outcome.output_limited:
        _fail("SANDBOX_LOG_LIMIT", "sandbox output exceeded policy", exit_code=11)

    try:
        result = _parse_tool_result(outcome.stdout)
    except RunnerFailure as exc:
        if outcome.stderr:
            exc.details["stderr"] = outcome.stderr.decode("utf-8", errors="replace")[:4_000]
        raise

    if result["status"] == "failed":
        if outcome.returncode == 0:
            _fail("INVALID_TOOL_RESULT", "failed tool result used a zero exit code", exit_code=12)
        return result, outcome.returncode if 1 <= outcome.returncode <= 125 else 11
    if outcome.returncode != 0:
        _fail(
            "SANDBOX_PROCESS_FAILED",
            "sandbox process exited unsuccessfully despite a success result",
            exit_code=11,
            details={"returncode": outcome.returncode},
        )
    _validate_success(result, output_dir, allowed, request, limits["max_artifacts_bytes"])
    return result, 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Metis reference architecture sandbox")
    parser.add_argument("--input-dir", type=Path, required=True, help="existing read-only input directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="existing run-specific artifact directory")
    parser.add_argument(
        "--image",
        default="localhost/metis/reference-architecture-tool:0.3.0",
        help="approved local image tag or digest-pinned equivalent",
    )
    arguments = parser.parse_args(argv)
    try:
        with _termination_signal_handlers():
            result, exit_code = run(arguments)
    except RunnerSignal as exc:
        signal_name = signal.Signals(exc.signum).name
        result = _failure_envelope(
            RunnerFailure(
                "SANDBOX_CANCELLED",
                f"sandbox runner received {signal_name}",
                exit_code=128 + exc.signum,
                details={"signal": signal_name},
            )
        )
        exit_code = 128 + exc.signum
    except RunnerFailure as exc:
        result = _failure_envelope(exc)
        exit_code = exc.exit_code
    sys.stdout.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
