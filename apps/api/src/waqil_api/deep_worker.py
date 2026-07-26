from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from .model_provider import OllamaModelProvider


WORKER_SYSTEM_PROMPT = """You are a constrained Metis worker. You may reason, plan, and
edit only the virtual files held in your graph state. You have no host shell,
network, registry, policy, activation, secrets, or long-term-memory authority.
Return proposed files and actions to the root LangGraph for validation. Never
claim that a proposed tool has been tested, approved, activated, or executed."""


@dataclass(frozen=True, slots=True)
class DeepWorkerPolicy:
    allow_host_filesystem: bool = False
    allow_shell: bool = False
    allow_network: bool = False
    allow_registry_write: bool = False
    allow_memory_write: bool = False


class ConstrainedDeepWorkerFactory:
    """Builds Deep Agents workers backed only by graph-state virtual files.

    The root control plane owns execution, persistence, approval, and tool
    promotion. No external tools or persistent store are passed to Deep Agents.
    """

    policy = DeepWorkerPolicy()

    def __init__(self, provider: OllamaModelProvider) -> None:
        self.provider = provider

    def create(
        self, role: str = "coder", model_aliases: dict[str, str] | None = None
    ) -> Any:
        from deepagents import create_deep_agent
        from deepagents.backends import StateBackend

        # StateBackend exposes a virtual workspace in LangGraph state. Passing no
        # custom tools, stores, or sandbox prevents host and network side effects.
        return create_deep_agent(
            model=self.provider.langchain_model(
                role,
                model_aliases=model_aliases,
                max_output_tokens=min(768, self.provider.settings.max_output_tokens),
            ),
            tools=[],
            system_prompt=WORKER_SYSTEM_PROMPT,
            backend=StateBackend(),
            subagents=[],
            skills=[],
            memory=[],
            permissions=[],
        )

    async def propose(
        self, prompt: str, *, model_aliases: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """Run one bounded virtual-workspace proposal under the broker's model slot."""

        worker = self.create("coder", model_aliases)
        # Deep Agents invokes the ChatOllama model directly, so hold the same slot
        # used by typed broker calls to preserve the one-heavy-generation invariant.
        async with self.provider._semaphore:
            async with asyncio.timeout(
                self.provider.settings.deep_worker_timeout_seconds
            ):
                result = await worker.ainvoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ]
                    },
                    config={"recursion_limit": 6},
                )
        files = result.get("files", {}) if isinstance(result, dict) else {}
        return {
            "status": "completed",
            "virtual_files": sorted(str(name) for name in files)[:50]
            if isinstance(files, dict)
            else [],
        }


def build_deep_worker_factory(provider: Any) -> ConstrainedDeepWorkerFactory | None:
    """Return a production Deep Agents factory when the Ollama adapter is active."""

    if not isinstance(provider, OllamaModelProvider):
        return None
    return ConstrainedDeepWorkerFactory(provider)
