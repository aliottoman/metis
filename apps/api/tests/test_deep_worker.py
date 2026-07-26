from __future__ import annotations

from waqil_api.deep_worker import ConstrainedDeepWorkerFactory
from waqil_api.model_provider import OllamaModelProvider


def test_deep_agent_worker_compiles_without_side_effect_tools(settings) -> None:
    provider = OllamaModelProvider(settings)
    factory = ConstrainedDeepWorkerFactory(provider)
    worker = factory.create()
    assert worker is not None
    assert factory.policy.allow_host_filesystem is False
    assert factory.policy.allow_shell is False
    assert factory.policy.allow_network is False
    assert factory.policy.allow_registry_write is False
