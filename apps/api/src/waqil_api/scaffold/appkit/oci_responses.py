"""The OCI Responses integration, vendored so no build reinvents it.

This mirrors the client Metis itself runs on: an httpx client carries the
OCI signing auth flow, the OpenAI SDK constructor carries ``project=``
(non-OpenAI models are refused with HTTP 400 without it), and text comes
from ``response.output_text``. Application code composes these calls; it
does not rebuild them.

Calls are synchronous on purpose. OCI request-signing covers exactly one
HTTP request — a ``background=True`` job is re-executed later by a service
worker that has no API key to present, and every such job dies with "No or
an invalid authentication header" (verified live against the Chicago
endpoint, 2026-08-04). Deferral belongs to the application (FastAPI
``BackgroundTasks`` around these calls), never to the provider call itself.

The SDK imports live inside the client builder, so importing this module —
and the application that uses it — needs neither cloud packages resolved at
import time nor any OCI environment. The first actual call raises a clear
``ConfigError`` naming whichever variable is missing.
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

from .config import OciResponsesConfig
from .uploads import sniff_mime


class ExtractionError(RuntimeError):
    """A model call ended without usable output; status and detail preserved.

    ``status`` is the terminal Responses status (or "timeout"), ``detail``
    whatever the service said about why. Keep both when re-raising — they are
    the difference between a diagnosable failure and a shrug.
    """

    def __init__(self, message: str, *, status: str = "", detail: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.detail = detail


class OciResponses:
    """Lazy client for the OpenAI-compatible OCI Responses surface."""

    def __init__(self, config: OciResponsesConfig | None = None) -> None:
        self._config = config
        self._client: Any | None = None
        self._lock = asyncio.Lock()

    async def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        async with self._lock:
            if self._client is not None:
                return self._client
            # Imports live here so the app imports — and its health route
            # serves — on a machine with no cloud SDKs or OCI environment.
            import httpx
            from oci_genai_auth import OciUserPrincipalAuth
            from openai import AsyncOpenAI

            config = self._config or OciResponsesConfig.from_env()
            self._config = config
            auth_kwargs: dict[str, Any] = {"profile_name": config.profile}
            if config.config_file:
                auth_kwargs["config_file"] = config.config_file
            self._client = AsyncOpenAI(
                api_key="not-used",  # constructor requires it; OCI signing ignores it
                base_url=config.base_url,
                project=config.project_id,  # OpenAI-Project header; 400 without it
                http_client=httpx.AsyncClient(
                    auth=OciUserPrincipalAuth(**auth_kwargs),
                    timeout=120.0,
                ),
            )
            return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def generate(
        self,
        prompt: str,
        *,
        timeout_seconds: float = 300.0,
    ) -> str:
        """Text in, text out, in one signed request."""
        return await self._run(prompt, timeout_seconds=timeout_seconds)

    async def extract_document(
        self,
        prompt: str,
        image_bytes: bytes,
        *,
        timeout_seconds: float = 300.0,
    ) -> str:
        """Run a vision prompt over one image and return the reply text.

        The image travels as a data URL inside an ``input_image`` part, with
        the MIME type sniffed from its actual bytes — a PNG labelled JPEG is
        a decoding failure waiting to happen.
        """
        mime = sniff_mime(image_bytes)
        encoded = base64.b64encode(image_bytes).decode("ascii")
        message = {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {
                    "type": "input_image",
                    "image_url": f"data:{mime};base64,{encoded}",
                },
            ],
        }
        return await self._run([message], timeout_seconds=timeout_seconds)

    async def _run(self, input_payload: Any, *, timeout_seconds: float) -> str:
        client = await self._get_client()
        config = self._config
        assert config is not None  # _get_client resolved it
        try:
            async with asyncio.timeout(timeout_seconds):
                response = await client.responses.create(
                    model=config.model_id, input=input_payload
                )
        except TimeoutError:
            raise ExtractionError(
                f"model call did not finish within {timeout_seconds:g}s",
                status="timeout",
            ) from None
        # A synchronous create still reports a status; anything terminal
        # other than completed carries the service's own reason with it.
        status = getattr(response, "status", "") or "completed"
        if status != "completed":
            error = getattr(response, "error", None)
            detail = str(getattr(error, "message", None) or error or "")
            raise ExtractionError(
                f"model call ended {status}", status=status, detail=detail
            )
        text = getattr(response, "output_text", "") or ""
        if not text.strip():
            raise ExtractionError(
                "model returned an empty completed response", status=status
            )
        return text


def parse_json_output(text: str) -> Any:
    """The JSON value inside a model reply — or ExtractionError, never {}.

    Swallowing a parse failure into an empty object turns "the extraction
    broke" into "the document was blank", which downstream checks then
    happily approve. A reply that is not JSON is an error with the reply
    attached, so it can be seen and fixed.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                pass
    raise ExtractionError(
        "model reply was not valid JSON", status="completed", detail=stripped[:500]
    )
