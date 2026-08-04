"""Check that every schema Metis decodes with actually compiles on this backend.

A local runtime turns a JSON schema into a decoding grammar, and rejects the
request outright when it cannot. Run this after changing a contract, upgrading
Ollama, or switching to a model with a different runtime.
"""

from __future__ import annotations

import asyncio
import sys

from waqil_api.config import Settings
from waqil_api.model_provider import OllamaModelProvider, local_decode_grammars


async def main() -> None:
    """Compile every local decode schema and report the ones the backend refuses."""
    settings = Settings()
    provider = OllamaModelProvider(settings)
    health = await provider.health()
    if not health.get("reachable"):
        raise SystemExit(f"Ollama is not reachable: {health.get('error', 'unknown error')}")

    failures = await provider.preflight_schemas()
    grammars = local_decode_grammars()
    for label, _schema, _constraint in grammars:
        problem = failures.get(label)
        print(f"{'FAIL' if problem else 'ok  '}  {label}" + (f"  {problem}" if problem else ""))

    if failures:
        print(
            f"\n{len(failures)} of {len(grammars)} schemas will not compile on "
            f"{settings.coder_model!r}/{settings.planner_model!r}. Every feature that decodes "
            "with one of them fails on this backend before the model ever runs.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print(f"\nAll {len(grammars)} grammars compile.")


if __name__ == "__main__":
    asyncio.run(main())
