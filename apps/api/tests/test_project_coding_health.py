from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from waqil_api.config import Settings
from waqil_api.main import create_app


def _settings(
    tmp_path: Path,
    *,
    engine: Literal["legacy", "clinecore"],
    entrypoint: Path,
) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        data_dir=tmp_path / "data",
        repo_root=tmp_path,
        model_backend="deterministic",
        reference_runner_mode="deterministic",
        allow_test_backends=True,
        project_coding_engine=engine,
        cline_sidecar_entrypoint=entrypoint,
    )


def _health(settings: Settings) -> dict:
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/v1/health")
    assert response.status_code == 200
    return response.json()


def _valid_sidecar(path: Path) -> None:
    path.write_text(
        """
const readline = require("node:readline");
const input = readline.createInterface({ input: process.stdin });
input.on("line", (line) => {
  const request = JSON.parse(line);
  let result;
  if (request.method === "getInfo") {
    result = {
      protocolVersion: "1",
      engine: "clinecore",
      runtime: "cline",
      sdkVersion: "0.0.86",
      policyVersion: "1",
      allowedTools: ["editor", "read_files", "run_check", "search_codebase"],
    };
  } else if (request.method === "shutdown") {
    result = { state: "shutting_down" };
  } else {
    throw new Error(`unexpected method: ${request.method}`);
  }
  process.stdout.write(JSON.stringify({ version: "1", id: request.id, result }) + "\\n");
  if (request.method === "shutdown") input.close();
});
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_legacy_project_engine_does_not_require_the_sidecar(tmp_path: Path) -> None:
    missing = tmp_path / "missing-sidecar.js"

    payload = _health(_settings(tmp_path, engine="legacy", entrypoint=missing))

    assert payload["status"] == "ok"
    assert payload["details"]["project_coding_engine"] == {
        "configured": "legacy",
        "ready": True,
        "sidecar_built": False,
        "sidecar_readable": False,
        "reason": None,
        "handshake": None,
    }


def test_legacy_project_engine_is_rejected_outside_test_backends(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValidationError, match="legacy project coding engine is retired"
    ):
        Settings(
            _env_file=None,  # type: ignore[call-arg]
            data_dir=tmp_path / "data",
            repo_root=tmp_path,
            model_backend="ollama",
            allow_test_backends=False,
            project_coding_engine="legacy",
        )


def test_selected_clinecore_without_a_built_sidecar_is_degraded(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing-sidecar.js"

    payload = _health(_settings(tmp_path, engine="clinecore", entrypoint=missing))
    coding = payload["details"]["project_coding_engine"]

    assert payload["status"] == "degraded"
    assert coding["configured"] == "clinecore"
    assert coding["ready"] is False
    assert coding["sidecar_built"] is False
    assert coding["sidecar_readable"] is False
    assert "has not been built" in coding["reason"]


def test_selected_clinecore_with_a_readable_sidecar_is_ready(tmp_path: Path) -> None:
    entrypoint = tmp_path / "sidecar.js"
    _valid_sidecar(entrypoint)

    payload = _health(_settings(tmp_path, engine="clinecore", entrypoint=entrypoint))
    coding = payload["details"]["project_coding_engine"]

    assert payload["status"] == "ok"
    assert coding == {
        "configured": "clinecore",
        "ready": True,
        "sidecar_built": True,
        "sidecar_readable": True,
        "reason": None,
        "handshake": {
            "protocolVersion": "1",
            "engine": "clinecore",
            "runtime": "cline",
            "sdkVersion": "0.0.86",
            "policyVersion": "1",
            "allowedTools": [
                "editor",
                "read_files",
                "run_check",
                "search_codebase",
            ],
        },
    }


def test_readable_but_invalid_sidecar_is_not_ready(tmp_path: Path) -> None:
    entrypoint = tmp_path / "invalid-sidecar.js"
    entrypoint.write_text("// readable is not a protocol handshake\n", encoding="utf-8")

    payload = _health(_settings(tmp_path, engine="clinecore", entrypoint=entrypoint))
    coding = payload["details"]["project_coding_engine"]

    assert payload["status"] == "degraded"
    assert coding["sidecar_built"] is True
    assert coding["sidecar_readable"] is True
    assert coding["ready"] is False
    assert coding["handshake"] is None
    # The user-facing reason stays plain-language; the raw protocol/parse
    # error is logged server-side, not echoed into this API response.
    assert "could not reach its local coding service" in coding["reason"]
    assert "invalid-sidecar" not in coding["reason"]
    assert "handshake" not in coding["reason"]


def test_responsive_sidecar_with_wrong_sdk_version_is_not_ready(
    tmp_path: Path,
) -> None:
    entrypoint = tmp_path / "wrong-version-sidecar.js"
    _valid_sidecar(entrypoint)
    entrypoint.write_text(
        entrypoint.read_text(encoding="utf-8").replace(
            'sdkVersion: "0.0.86"', 'sdkVersion: "0.0.87"'
        ),
        encoding="utf-8",
    )

    payload = _health(_settings(tmp_path, engine="clinecore", entrypoint=entrypoint))
    coding = payload["details"]["project_coding_engine"]

    assert payload["status"] == "degraded"
    assert coding["ready"] is False
    assert coding["handshake"] is None
    assert "could not reach its local coding service" in coding["reason"]
    assert "0.0.73" not in coding["reason"]
    assert "invalid result" not in coding["reason"]
