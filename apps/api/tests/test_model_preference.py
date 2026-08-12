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


def test_clinepass_catalog_is_backend_owned_and_excludes_paid_models(tmp_path) -> None:
    settings = _settings(tmp_path)
    settings.cline_api_key = "subscription-key"
    preference = ModelPreferenceStore(settings).load()
    assert preference.cline_models[0] == "cline-pass/qwen3.7-plus"
    assert "cline-pass/deepseek-v4-pro" in preference.cline_models
    assert "cline-pass/kimi-k3" in preference.cline_models
    assert all(model.startswith("cline-pass/") for model in preference.cline_models)


def test_clinepass_defaults_have_role_specific_safety_ladders(tmp_path) -> None:
    import json

    settings = _settings(tmp_path)
    settings.cline_api_key = "subscription-key"
    store = ModelPreferenceStore(settings)
    store.save("split", None, provider="cline")

    aliases = store.resolve_aliases()
    planners = json.loads(aliases["_chain_planner"])
    coders = json.loads(aliases["_fallbacks_coder"])
    assert [item["model"] for item in planners] == [
        "cline-pass/qwen3.7-plus",
        "cline-pass/glm-5.2",
    ]
    assert [item["model"] for item in coders] == [
        "cline-pass/kimi-k3",
        "cline-pass/kimi-k2.7-code",
    ]
    advertised = set(store.load().cline_models)
    assert {item["model"] for item in planners + coders} <= advertised


@pytest.mark.parametrize(
    ("coder_primary", "expected_backups"),
    [
        ("cline-pass/kimi-k3", ["cline-pass/kimi-k2.7-code"]),
        ("cline-pass/kimi-k2.7-code", ["cline-pass/kimi-k3"]),
    ],
)
def test_synthesized_cline_ladders_do_not_repeat_env_selected_primaries(
    tmp_path, coder_primary: str, expected_backups: list[str]
) -> None:
    import json

    settings = _settings(tmp_path)
    settings.cline_api_key = "subscription-key"
    settings.cline_orchestrator_model = "cline-pass/glm-5.2"
    settings.cline_coder_model = coder_primary
    store = ModelPreferenceStore(settings)
    store.save("split", None, provider="cline")

    aliases = store.resolve_aliases()
    planners = json.loads(aliases["_chain_planner"])
    backups = json.loads(aliases["_fallbacks_coder"])

    assert planners == [{"provider": "cline", "model": "cline-pass/glm-5.2"}]
    assert [item["model"] for item in backups] == expected_backups
    assert coder_primary not in {item["model"] for item in backups}


def test_an_explicit_cline_planner_chain_is_never_rewritten(tmp_path) -> None:
    import json

    from waqil_api.contracts import RoleChainEntryV1

    settings = _settings(tmp_path)
    settings.cline_api_key = "subscription-key"
    store = ModelPreferenceStore(settings)
    store.save(
        "split",
        None,
        provider="cline",
        role_chains={
            "planner": [
                RoleChainEntryV1(provider="cline", model="cline-pass/qwen3.7-max")
            ]
        },
    )

    assert json.loads(store.resolve_aliases()["_chain_planner"]) == [
        {"provider": "cline", "model": "cline-pass/qwen3.7-max"}
    ]


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
        # The synthesized coder safety fallback: with no Cohere key in these
        # settings, the ladder holds only the default local coder, so an
        # outage on the pinned model degrades a build instead of ending it.
        "_fallbacks_coder": '[{"provider": "local", "model": "north-mini-code-1.0:mlx-nvfp4"}]',
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


def test_the_grok_lane_kill_switch_disables_oci_without_touching_cohere(
    tmp_path,
) -> None:
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


def test_a_stale_oci_preference_collapses_to_local_when_the_lane_is_off(
    tmp_path,
) -> None:
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


def test_role_chains_round_trip_and_set_the_primary(tmp_path) -> None:
    """An explicit chain is the selection, not a decoration on it: its first
    entry becomes the role's primary, and the whole ladder rides into the
    aliases every run is frozen with."""
    import json as _json

    from waqil_api.contracts import RoleChainEntryV1

    store = ModelPreferenceStore(_settings(tmp_path))
    saved = store.save(
        "pinned",
        "qwen3.6:35b-mlx",
        role_chains={
            "coder": [
                RoleChainEntryV1(provider="local", model="glm-5.2:cloud"),
                RoleChainEntryV1(provider="local", model="qwen3-coder:30b"),
            ]
        },
    )
    assert [entry.model for entry in saved.role_chains["coder"]] == [
        "glm-5.2:cloud",
        "qwen3-coder:30b",
    ]
    aliases = store.resolve_aliases()
    # chain[0] IS the coder, even under a different pin.
    assert aliases["coder"] == "glm-5.2:cloud"
    chain = _json.loads(aliases["_chain_coder"])
    assert [entry["model"] for entry in chain] == ["glm-5.2:cloud", "qwen3-coder:30b"]

    # None leaves chains untouched; {} clears them.
    kept = store.save("pinned", "qwen3.6:35b-mlx")
    assert "coder" in kept.role_chains
    cleared = store.save("pinned", "qwen3.6:35b-mlx", role_chains={})
    assert cleared.role_chains == {}


def test_role_chain_validation_refuses_broken_ladders(tmp_path) -> None:
    """A rung that cannot answer is refused where the choice is made — a
    ladder that fails exactly when it is needed is worse than none."""
    import pytest as _pytest

    from waqil_api.contracts import RoleChainEntryV1

    store = ModelPreferenceStore(_settings(tmp_path))
    # A hosted model measured to ignore tool calling.
    with _pytest.raises(ValueError, match="does not honour tool calling"):
        store.save(
            "split",
            None,
            role_chains={
                "coder": [RoleChainEntryV1(provider="local", model="minimax-m3:cloud")]
            },
        )
    # A lane with no key behind it.
    with _pytest.raises(ValueError, match="Cohere, which is not configured"):
        store.save(
            "split",
            None,
            role_chains={"coder": [RoleChainEntryV1(provider="cohere")]},
        )
