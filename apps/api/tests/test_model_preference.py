from __future__ import annotations

import pytest

from waqil_api.config import Settings
from waqil_api.model_preference import ModelPreferenceStore


def _settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        allow_test_backends=True,
        allow_oci_responses=False,
        oci_responses_project_id="",
        planner_model="qwen3.6:35b-mlx",
        coder_model="north-mini-code-1.0:mlx-nvfp4",
        quality_model="north-mini-code-1.0:mlx-mxfp8",
    )


def test_defaults_to_split_per_role_models(tmp_path) -> None:
    store = ModelPreferenceStore(_settings(tmp_path))
    preference = store.load()
    assert preference.mode == "split"
    assert preference.model is None
    assert store.resolve_aliases() == {
        "planner": "qwen3.6:35b-mlx",
        "coder": "north-mini-code-1.0:mlx-nvfp4",
        "quality": "north-mini-code-1.0:mlx-mxfp8",
        "_provider": "local",
        "_oci_tools": "code_interpreter",
    }


def test_pinning_one_model_routes_every_role_to_it(tmp_path) -> None:
    store = ModelPreferenceStore(_settings(tmp_path))
    saved = store.save("pinned", "qwen3.6:35b-mlx")
    assert saved.mode == "pinned"
    assert saved.model == "qwen3.6:35b-mlx"
    assert store.load() == saved
    assert store.resolve_aliases() == {
        "planner": "qwen3.6:35b-mlx",
        "coder": "qwen3.6:35b-mlx",
        "quality": "qwen3.6:35b-mlx",
        "_provider": "local",
        "_oci_tools": "",
    }


def test_switching_back_to_split_clears_the_pin(tmp_path) -> None:
    store = ModelPreferenceStore(_settings(tmp_path))
    store.save("pinned", "north-mini-code-1.0:mlx-nvfp4")
    reverted = store.save("split", None)
    assert reverted.mode == "split"
    assert reverted.model is None
    assert store.resolve_aliases()["coder"] == "north-mini-code-1.0:mlx-nvfp4"
    assert store.resolve_aliases()["planner"] == "qwen3.6:35b-mlx"


def test_pinned_mode_requires_a_model(tmp_path) -> None:
    store = ModelPreferenceStore(_settings(tmp_path))
    with pytest.raises(ValueError):
        store.save("pinned", None)


def test_invalid_mode_is_rejected(tmp_path) -> None:
    store = ModelPreferenceStore(_settings(tmp_path))
    with pytest.raises(ValueError):
        store.save("sometimes", "qwen3.6:35b-mlx")


def test_oci_requires_explicit_configuration_and_pins_native_tools(tmp_path) -> None:
    disabled = ModelPreferenceStore(_settings(tmp_path))
    with pytest.raises(ValueError, match="OCI Responses requires"):
        disabled.save("split", None, provider="oci", oci_tools=["code_interpreter"])

    settings = _settings(tmp_path)
    settings.allow_oci_responses = True
    settings.oci_responses_project_id = "ocid1.generativeaiproject.oc1..test"
    enabled = ModelPreferenceStore(settings)
    saved = enabled.save(
        "split",
        None,
        provider="oci",
        oci_tools=["code_interpreter", "x_search", "code_interpreter"],
    )
    assert saved.provider == "oci"
    assert saved.oci_tools == ["code_interpreter", "x_search"]
    assert saved.oci_available is True
    assert enabled.resolve_aliases()["_provider"] == "oci"
    assert enabled.resolve_aliases()["_oci_tools"] == "code_interpreter,x_search"
