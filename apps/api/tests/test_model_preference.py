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


def test_the_grok_lane_kill_switch_disables_oci_without_touching_cohere(tmp_path) -> None:
    """One flag sidelines the Grok lane, re-enablable later: with credentials
    fully configured, grok_lane_enabled=False must read as unavailable — the
    same wire fact the whole web UI disables its Grok controls on — while
    Cohere is untouched. A stored oci preference collapses to local on read."""
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        allow_test_backends=True,
        allow_oci_responses=True,
        oci_responses_project_id="ocid1.aiproject.oc1.test",
        cohere_api_key="test-key",
        grok_lane_enabled=False,
    )
    store = ModelPreferenceStore(settings)
    assert store.oci_available is False
    assert store.cohere_available is True
    with pytest.raises(ValueError):
        store.save("split", None, provider="oci")

    # The identical configuration with the lane on: available again — the
    # switch is the only difference, so re-enabling is one env var.
    enabled = ModelPreferenceStore(
        Settings(
            _env_file=None,
            data_dir=tmp_path / "data2",
            allow_test_backends=True,
            allow_oci_responses=True,
            oci_responses_project_id="ocid1.aiproject.oc1.test",
            cohere_api_key="test-key",
            grok_lane_enabled=True,
        )
    )
    assert enabled.oci_available is True


def test_a_stale_oci_preference_collapses_to_local_when_the_lane_is_off(tmp_path) -> None:
    on = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        allow_test_backends=True,
        allow_oci_responses=True,
        oci_responses_project_id="ocid1.aiproject.oc1.test",
        grok_lane_enabled=True,
    )
    ModelPreferenceStore(on).save("split", None, provider="oci")
    off = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        allow_test_backends=True,
        allow_oci_responses=True,
        oci_responses_project_id="ocid1.aiproject.oc1.test",
        grok_lane_enabled=False,
    )
    preference = ModelPreferenceStore(off).load()
    assert preference.provider == "local"
