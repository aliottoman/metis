from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER_PATH = REPO_ROOT / "infra" / "sandbox" / "run_reference_architecture.py"
spec = importlib.util.spec_from_file_location("metis_sandbox_runner", RUNNER_PATH)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


class SandboxRunnerPolicyTests(unittest.TestCase):
    def test_policy_contains_required_isolation(self) -> None:
        policy = json.loads((REPO_ROOT / "infra" / "sandbox" / "sandbox-policy.json").read_text())
        self.assertEqual(policy["network"], "none")
        self.assertTrue(policy["read_only_root"])
        self.assertTrue(policy["non_root"])
        self.assertIn("ALL", policy["drop_capabilities"])
        self.assertIn("no-new-privileges", policy["security_options"])
        self.assertEqual(policy["mounts"]["/input"], "read-only")
        self.assertEqual(policy["mounts"]["/output"], "read-write")

    def test_image_override_must_remain_in_approved_repository(self) -> None:
        expected = "localhost/metis/reference-architecture-tool:0.3.0"
        runner._validate_image(expected, expected)
        runner._validate_image(
            "localhost/metis/reference-architecture-tool@sha256:" + "a" * 64,
            expected,
        )
        with self.assertRaises(runner.RunnerFailure):
            runner._validate_image("docker.io/untrusted/tool:latest", expected)

    def test_container_dependencies_are_minimal_and_immutable(self) -> None:
        infra = REPO_ROOT / "infra" / "sandbox"
        containerfile = (infra / "Containerfile").read_text(encoding="utf-8")
        build_script = (infra / "build_reference_architecture_image.sh").read_text(
            encoding="utf-8"
        )
        skill = REPO_ROOT / "skills" / "reference-architecture-generator"
        lock = (skill / "requirements-runtime.lock").read_text(encoding="utf-8")
        manifest = json.loads((skill / "metis.tool.json").read_text(encoding="utf-8"))

        base_digest = "sha256:4c2cf9917bd1cbacc5e9b07320025bdb7cdf2df7b0ceaccb55e9dd7e30987419"
        self.assertIn(f"python:3.13.5-slim-bookworm@{base_digest}", containerfile)
        self.assertIn('"graphviz=2.42.2-7+deb12u1"', containerfile)
        self.assertIn("--no-deps", containerfile)
        self.assertIn("--no-compile", containerfile)
        self.assertIn("--only-binary=:all:", containerfile)
        self.assertIn("--require-hashes", containerfile)
        self.assertIn("--platform \"$platform\"", build_script)
        self.assertIn("--timestamp 0", build_script)

        expected_packages = {
            "diagrams==0.25.1",
            "graphviz==0.20.3",
            "Jinja2==3.1.6",
            "MarkupSafe==3.0.2",
        }
        for package in expected_packages:
            self.assertIn(package, lock)
        for unwanted in ("pre-commit", "virtualenv", "nodeenv"):
            self.assertNotIn(f"{unwanted}==", lock.lower())

        dependencies = manifest["dependencies"]
        self.assertEqual(set(dependencies["python_packages"]), expected_packages)
        self.assertEqual(dependencies["system_packages"], ["graphviz=2.42.2-7+deb12u1"])
        self.assertTrue(dependencies["installation"]["require_hashes"])
        self.assertTrue(dependencies["installation"]["binary_only"])
        self.assertFalse(dependencies["installation"]["resolve_dependencies"])

    def test_rejects_symlink_in_input_mount(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "target.txt").write_text("safe", encoding="utf-8")
            (root / "link.txt").symlink_to(root / "target.txt")
            with self.assertRaises(runner.RunnerFailure):
                runner._inspect_input_tree(root, max_files=10, max_bytes=1_000)

    def test_rejects_undeclared_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "unexpected.bin").write_bytes(b"x")
            with self.assertRaises(runner.RunnerFailure):
                runner._inspect_output_tree(root, {"architecture.svg": "image/svg+xml"})

    def test_base_exception_terminates_spawned_process_group(self) -> None:
        class SimulatedCancellation(BaseException):
            pass

        captured: list[subprocess.Popen[bytes]] = []
        real_popen = subprocess.Popen

        def capture_process(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            captured.append(process)
            return process

        def interrupt_wait(*_args, **_kwargs):
            time.sleep(0.1)
            raise SimulatedCancellation()

        command = [sys.executable, "-c", "import time; time.sleep(60)"]
        with mock.patch.object(runner.subprocess, "Popen", side_effect=capture_process), mock.patch.object(
            runner.selectors.DefaultSelector, "select", side_effect=interrupt_wait
        ):
            with self.assertRaises(SimulatedCancellation):
                runner._run_bounded(
                    command,
                    b"{}",
                    timeout_seconds=30,
                    max_output_bytes=1024,
                )

        self.assertEqual(len(captured), 1)
        self.assertIsNotNone(captured[0].returncode)
        with self.assertRaises(ProcessLookupError):
            os.killpg(captured[0].pid, 0)

    def test_sigterm_is_converted_to_catchable_runner_signal(self) -> None:
        with self.assertRaises(runner.RunnerSignal) as caught:
            with runner._termination_signal_handlers():
                signal.raise_signal(signal.SIGTERM)
        self.assertEqual(caught.exception.signum, signal.SIGTERM)


if __name__ == "__main__":
    unittest.main()
