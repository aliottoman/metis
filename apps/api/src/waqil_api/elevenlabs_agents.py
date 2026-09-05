"""The ElevenLabs Agents Platform API, as far as the agent factory needs it.

A thin client and nothing more: every method is one documented call, the
error is one exception type, and the httpx client is borrowed from the speech
provider so there is exactly one holder of the API key. The factory decides
what to create; this file only knows how to ask for it.

Paths were checked against the platform's OpenAPI document rather than the
prose docs, which lag it. Two facts worth keeping: a document deleted twice
answers 404, which is success for a caller cleaning up; and a test run is
asynchronous — `run_tests` returns an invocation whose runs are `pending`
until the platform finishes them.
"""

from __future__ import annotations

from typing import Any

from .contracts import AgentTestV1


class ElevenLabsAgentsError(RuntimeError):
    """One call to the Agents Platform failed. The message says which."""


class ElevenLabsAgents:
    def __init__(self, client: Any) -> None:
        # An httpx.AsyncClient with the base URL and the xi-api-key header set.
        self._client = client

    # -- agents ------------------------------------------------------------

    async def create_agent(self, body: dict[str, Any]) -> str:
        reply = await self._call("POST", "/v1/convai/agents/create", json=body)
        agent_id = str(reply.get("agent_id") or "").strip()
        if not agent_id:
            raise ElevenLabsAgentsError("ElevenLabs created no agent id")
        return agent_id

    async def update_agent(self, agent_id: str, body: dict[str, Any]) -> None:
        await self._call("PATCH", f"/v1/convai/agents/{agent_id}", json=body)

    async def delete_agent(self, agent_id: str) -> None:
        await self._call("DELETE", f"/v1/convai/agents/{agent_id}", missing_ok=True)

    # -- knowledge base ----------------------------------------------------

    async def create_document(
        self, *, kind: str, name: str, url: str = "", text: str = ""
    ) -> str:
        if kind == "url":
            reply = await self._call(
                "POST", "/v1/convai/knowledge-base/url", json={"url": url, "name": name}
            )
        else:
            reply = await self._call(
                "POST",
                "/v1/convai/knowledge-base/text",
                json={"text": text, "name": name},
            )
        document_id = str(reply.get("id") or "").strip()
        if not document_id:
            raise ElevenLabsAgentsError("ElevenLabs created no document id")
        return document_id

    async def delete_document(self, document_id: str) -> None:
        await self._call(
            "DELETE", f"/v1/convai/knowledge-base/{document_id}", missing_ok=True
        )

    # -- tests -------------------------------------------------------------

    async def create_simulation_test(self, test: AgentTestV1) -> str:
        reply = await self._call(
            "POST",
            "/v1/convai/agent-testing/create",
            json={
                "type": "simulation",
                "name": test.name,
                "simulation_scenario": test.scenario,
                "success_conditions": list(test.success_conditions),
                "simulation_max_turns": test.max_turns,
            },
        )
        test_id = str(reply.get("id") or "").strip()
        if not test_id:
            raise ElevenLabsAgentsError("ElevenLabs created no test id")
        return test_id

    async def delete_test(self, test_id: str) -> None:
        await self._call(
            "DELETE", f"/v1/convai/agent-testing/{test_id}", missing_ok=True
        )

    async def run_tests(self, agent_id: str, test_ids: list[str]) -> dict[str, Any]:
        return await self._call(
            "POST",
            f"/v1/convai/agents/{agent_id}/run-tests",
            json={"tests": [{"test_id": test_id} for test_id in test_ids]},
        )

    async def get_test_invocation(self, invocation_id: str) -> dict[str, Any]:
        return await self._call("GET", f"/v1/convai/test-invocations/{invocation_id}")

    # -- the one request path ---------------------------------------------

    async def _call(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        missing_ok: bool = False,
    ) -> dict[str, Any]:
        try:
            response = await self._client.request(method, path, json=json)
        except Exception as exc:  # noqa: BLE001 - network errors become one error type
            raise ElevenLabsAgentsError(
                f"Could not reach ElevenLabs: {str(exc)[:200]}"
            ) from exc
        if response.status_code == 404 and missing_ok:
            return {}
        if response.status_code >= 400:
            raise ElevenLabsAgentsError(
                f"ElevenLabs refused {method} {path}: HTTP {response.status_code} "
                f"{response.text[:300]}"
            )
        if not response.content:
            return {}
        try:
            payload = response.json()
        except ValueError as exc:
            raise ElevenLabsAgentsError("ElevenLabs returned a non-JSON reply") from exc
        return payload if isinstance(payload, dict) else {}
