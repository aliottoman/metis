from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

API_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_ROOT / "src"))

from waqil_api.config import Settings  # noqa: E402
from waqil_api.main import create_app  # noqa: E402


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    # `_env_file=None` makes the suite hermetic: it must not inherit the
    # developer's `.env` (e.g. real OCI creds or cloud embeddings turned on),
    # so cloud retrieval defaults off and tests stay deterministic.
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        repo_root=Path(__file__).resolve().parents[3],
        model_backend="deterministic",
        reference_runner_mode="deterministic",
        reference_runner_timeout_seconds=150,
        allow_test_backends=True,
    )


@pytest.fixture
def client(settings: Settings):
    with TestClient(create_app(settings)) as value:
        yield value
