"""Opening a project shifts the coder to the hosted model — but never forces it.

Whole-application builds are the one workload where the local models measurably
fall short, so project runs default to the hosted coder. The rule that keeps it
from being a lock-in: an explicitly pinned model is the user answering this
question themselves, and their answer wins.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from waqil_api.config import Settings
from waqil_api.model_preference import ModelPreferenceStore, is_cloud_model


def _service(tmp_path: Path, **overrides: object) -> ModelPreferenceStore:
    # The shipped default is off (routing a build to someone's Ollama Cloud
    # account is an opt-in decision); these tests are about the routing rule,
    # so they opt in.
    overrides.setdefault("project_cloud_coder", True)
    settings = Settings(_env_file=None, data_dir=tmp_path, **overrides)  # type: ignore[arg-type]
    return ModelPreferenceStore(settings)


def test_a_project_run_defaults_to_the_hosted_coder(tmp_path: Path) -> None:
    assert _service(tmp_path).project_coder() == "gpt-oss:120b-cloud"


def test_a_local_pin_does_not_block_the_default(tmp_path: Path) -> None:
    """Pinning is not the statement it looks like: launching a local model
    session pins the preference as a side effect, so gating on it meant the
    default never fired for anyone who had ever started a local model."""
    service = _service(tmp_path)
    service.save("pinned", "qwen3-coder:30b")
    assert service.project_coder() == "gpt-oss:120b-cloud"


def test_pinning_a_cloud_model_is_a_real_choice_and_wins(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.save("pinned", "glm-5.2:cloud")
    assert service.project_coder() == ""


def test_split_preference_still_gets_the_hosted_coder(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.save("split", None)
    assert service.project_coder() == "gpt-oss:120b-cloud"


def test_the_setting_turns_it_off_entirely(tmp_path: Path) -> None:
    assert _service(tmp_path, project_cloud_coder=False).project_coder() == ""


def test_the_shipped_default_is_off(tmp_path: Path) -> None:
    """The provider can drive a hosted coder now — over tool calling, since
    Ollama Cloud ignores the step grammar — but routing a build to someone's
    cloud account stays an explicit opt-in, not a shipped default."""
    settings = Settings(_env_file=None, data_dir=tmp_path)
    assert settings.project_cloud_coder is False
    assert ModelPreferenceStore(settings).project_coder() == ""


def test_the_model_name_is_configurable(tmp_path: Path) -> None:
    service = _service(tmp_path, project_cloud_coder_model="glm-5.2:cloud")
    assert service.project_coder() == "glm-5.2:cloud"


def test_only_the_coder_alias_moves(tmp_path: Path) -> None:
    """The planner routes and the reviewer critiques; neither is the step
    that struggles, and both stay wherever the preference put them."""
    service = _service(tmp_path)
    aliases = service.resolve_aliases()
    aliases["coder"] = service.project_coder()
    assert aliases["coder"] == "gpt-oss:120b-cloud"
    assert aliases["planner"] == "qwen3.6:35b-mlx"
    assert aliases["quality"] == "north-mini-code-1.0:mlx-mxfp8"


def test_a_corrupt_preference_file_does_not_break_the_default(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service._settings.model_preference_path.parent.mkdir(parents=True, exist_ok=True)
    service._settings.model_preference_path.write_text("{not json", encoding="utf-8")
    assert service.project_coder() == "gpt-oss:120b-cloud"
    # And a pinned cloud model still wins after the file is repaired.
    service._settings.model_preference_path.write_text(
        json.dumps({"mode": "pinned", "model": "kimi-k2.7-code:cloud"}), encoding="utf-8"
    )
    assert service.project_coder() == ""


def test_an_unusable_hosted_coder_is_refused_at_selection_time(tmp_path: Path) -> None:
    """M6: minimax-m3 ignores tool calls on the same endpoint where gemma4
    honours them, so routing a build to it can only end in malformed replies
    at step five. The refusal happens where the route is chosen, and it names
    the configuration mistake instead of blaming the model."""
    service = _service(tmp_path, project_cloud_coder_model="minimax-m3:cloud")
    with pytest.raises(ValueError, match="does not honour tool calling"):
        service.project_coder()


def test_a_hosted_model_not_yet_measured_gets_the_benefit_of_the_doubt(
    tmp_path: Path,
) -> None:
    """The record holds measurements, not guesses: a subscription model nobody
    has tested routes normally, and the malformed-streak breaker bounds a
    wrong guess. Only a model measured to fail is refused."""
    service = _service(tmp_path, project_cloud_coder_model="kimi-k2.7-code:cloud")
    assert service.project_coder() == "kimi-k2.7-code:cloud"


def test_pinning_an_unusable_hosted_model_is_refused_where_it_is_saved(
    tmp_path: Path,
) -> None:
    """A pinned model drives every role, so an unusable one would break the
    planner too. The save is the selection moment, so the save refuses."""
    service = _service(tmp_path)
    with pytest.raises(ValueError, match="does not honour tool calling"):
        service.save("pinned", "minimax-m3:cloud")
    # And the usable hosted models still pin exactly as before.
    service.save("pinned", "gpt-oss:120b-cloud")
    assert service.load().model == "gpt-oss:120b-cloud"


def test_both_cloud_tag_shapes_are_recognized() -> None:
    """Ollama spells hosted models two ways; matching only one silently
    misread every model of the other kind as local."""
    for name in ("gpt-oss:120b-cloud", "glm-5.2:cloud", "kimi-k2.7-code:cloud"):
        assert is_cloud_model(name), name
    for name in ("qwen3-coder:30b", "qwen3.6:35b-mlx", "cloudy:7b", ""):
        assert not is_cloud_model(name), name


@pytest.mark.asyncio
async def test_a_hosted_model_needs_no_local_session(tmp_path: Path) -> None:
    """The guard that failed every hosted build.

    The session keeps two large models out of unified memory, so everything
    it enforces is about local weights. A hosted model has none — asking the
    user to "launch gpt-oss:120b-cloud" is asking for something that cannot
    be launched — and it must not mark the local session busy either.
    """
    from waqil_api.local_model_session import LocalModelSessionManager

    settings = Settings(_env_file=None, data_dir=tmp_path, model_backend="ollama")
    session = LocalModelSessionManager(settings, ModelPreferenceStore(settings))

    # Nothing is loaded: a local model is refused, a hosted one is allowed.
    with pytest.raises(Exception):
        await session.require_ready("qwen3-coder:30b")
    await session.require_ready("gpt-oss:120b-cloud")

    async with session.use("gpt-oss:120b-cloud"):
        state = await session.status(include_models=False)
        assert state.state == "off", "a hosted call must not report the local session busy"
