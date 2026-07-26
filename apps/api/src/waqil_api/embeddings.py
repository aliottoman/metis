"""OCI Cohere retrieval — embeddings (embed-v4) and rerank (rerank-v3.5).

Both use OCI Generative AI *on-demand* serving (pay-per-call, no dedicated
endpoint). This is an OPT-IN capability: `Settings.allow_cloud_embeddings` must
be true, `oci_compartment_id` must be set, and the `oci` SDK must import. When
any of those is false the caller falls back to the local FTS/keyword path, so a
default Metis install sends nothing off the device.

The vector index is stored LOCALLY (SQLite, see `corpus.py`); only the text
being embedded or reranked leaves the Mac, and only for a source the user has
explicitly consented to index. Auth is read from ~/.oci/config
(`Settings.oci_profile`); the private key never passes through this process.

`import oci` is deliberately lazy (inside methods) so this module — and the
whole API — imports cleanly when the heavy SDK is not installed.
"""
from __future__ import annotations

import json
import random
import re
import time
from collections.abc import Callable
from typing import Any, TypeVar

from . import entity_graph
from .config import Settings

_T = TypeVar("_T")


def _parse_grok_review(text: str) -> dict[str, Any]:
    """Defensively extract the review verdict from Grok's reply. A malformed reply
    is treated as 'reviewed but no verdict' (safe=True, no improvement) so the
    host AST-gate remains the decision-maker."""
    candidate = None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text or "", re.DOTALL)
    if fence:
        candidate = fence.group(1)
    else:
        start, end = (text or "").find("{"), (text or "").rfind("}")
        if start != -1 and end > start:
            candidate = text[start : end + 1]
    parsed: dict[str, Any] = {}
    if candidate:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                parsed = value
        except (json.JSONDecodeError, ValueError):
            parsed = {}
    reasons = parsed.get("reasons")
    improved = parsed.get("improved_code")
    return {
        "safe": bool(parsed.get("safe", True)),
        "reasons": [str(r)[:300] for r in reasons][:12] if isinstance(reasons, list) else [],
        "improved_code": improved if isinstance(improved, str) and improved.strip() else "",
    }

# Transient failures worth retrying, matched on text because the SDK wraps
# socket errors and throttling in its own exception types.
_TRANSIENT_MARKERS = (
    "connection reset",
    "connection aborted",
    "broken pipe",
    "timed out",
    "timeout",
    "temporarily unavailable",
    "too many requests",
    "429",
    "500",
    "502",
    "503",
    "504",
)


def _is_transient(error: BaseException) -> bool:
    text = f"{type(error).__name__}: {error}".lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


class CohereUnavailable(RuntimeError):
    """A cloud retrieval call was attempted while it was disabled or unconfigured."""


class CohereRetrieval:
    """Thin, synchronous wrapper over OCI GenAI inference for embed + rerank.

    Instances are cheap and hold no client; each call builds a fresh client so a
    settings/profile change takes effect immediately and no socket is retained.
    Run calls off the event loop (``asyncio.to_thread``) — the SDK is blocking.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def available(self) -> bool:
        """True only if cloud retrieval is opted-in, configured, and importable."""
        settings = self._settings
        if not settings.allow_cloud_embeddings or not settings.oci_compartment_id:
            return False
        try:
            import oci  # noqa: F401
        except Exception:
            return False
        return True

    def _require(self) -> None:
        settings = self._settings
        if not settings.allow_cloud_embeddings:
            raise CohereUnavailable(
                "cloud embeddings are disabled (set WAQIL_ALLOW_CLOUD_EMBEDDINGS=true)"
            )
        if not settings.oci_compartment_id:
            raise CohereUnavailable("WAQIL_OCI_COMPARTMENT_ID is not configured")

    def _client(self, endpoint: str | None = None):
        import oci

        config = oci.config.from_file(profile_name=self._settings.oci_profile)
        service_endpoint = endpoint or self._settings.oci_genai_endpoint
        kwargs = {"service_endpoint": service_endpoint} if service_endpoint else {}
        return oci.generative_ai_inference.GenerativeAiInferenceClient(config, **kwargs)

    def _with_retry(self, label: str, call: Callable[[], _T]) -> _T:
        """Run one blocking OCI call, retrying transient failures with capped
        exponential backoff and jitter. Non-transient errors (auth, bad request)
        raise immediately; transient ones raise only after the attempts are spent.

        Callers run inside ``asyncio.to_thread`` already, so a blocking sleep here
        holds only the worker thread, not the event loop.
        """
        settings = self._settings
        attempts = settings.cloud_max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                return call()
            except Exception as error:  # noqa: BLE001 - re-raised below
                if attempt >= attempts or not _is_transient(error):
                    raise
                delay = min(
                    settings.cloud_retry_max_seconds,
                    settings.cloud_retry_base_seconds * (2 ** (attempt - 1)),
                )
                # Full jitter avoids a thundering herd of same-timed retries when
                # a batch of chunks all fail against a throttling endpoint at once.
                time.sleep(random.uniform(0, delay))
        raise AssertionError(f"unreachable retry exit for {label}")  # pragma: no cover

    def embed(
        self, texts: list[str], input_type: str = "search_document"
    ) -> list[list[float]]:
        """Embed `texts` with Cohere embed-v4. Use `search_document` when indexing
        and `search_query` when retrieving — Cohere needs the asymmetric hint."""
        import oci

        self._require()
        if not texts:
            return []
        models = oci.generative_ai_inference.models
        input_type_value = getattr(
            models.EmbedTextDetails,
            f"INPUT_TYPE_{input_type.upper()}",
            input_type.upper(),
        )
        batch = self._settings.embed_batch
        out: list[list[float]] = []
        for start in range(0, len(texts), batch):
            details = models.EmbedTextDetails(
                inputs=texts[start : start + batch],
                serving_mode=models.OnDemandServingMode(
                    model_id=self._settings.oci_embed_model
                ),
                compartment_id=self._settings.oci_compartment_id,
                input_type=input_type_value,
                truncate=models.EmbedTextDetails.TRUNCATE_END,
            )
            # Fresh client per attempt: a reset socket can leave the SDK's
            # connection pool unusable, so a retry must not reuse it.
            response = self._with_retry(
                "embed_text", lambda details=details: self._client().embed_text(details)
            )
            out.extend(response.data.embeddings)
        return out

    def rerank(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[tuple[int, float]]:
        """Rerank `documents` by relevance to `query` with Cohere rerank.

        Returns ``[(original_index, relevance_score), ...]`` sorted most-relevant
        first, length ``min(top_n, len(documents))``. The scores are absolute
        relevance in [0, 1], not the recall similarities — that is the whole
        point of a second stage over lexical/vector recall.
        """
        import oci

        self._require()
        if not documents:
            return []
        models = oci.generative_ai_inference.models
        # The API rejects a top_n larger than the candidate count, so clamp it here.
        effective_top_n = max(1, min(top_n or len(documents), len(documents)))
        details = models.RerankTextDetails(
            input=query,
            documents=list(documents),
            compartment_id=self._settings.oci_compartment_id,
            serving_mode=models.OnDemandServingMode(
                model_id=self._settings.oci_rerank_model
            ),
            top_n=effective_top_n,
        )
        rerank_endpoint = self._settings.oci_rerank_endpoint or None
        response = self._with_retry(
            "rerank_text",
            lambda: self._client(rerank_endpoint).rerank_text(details),
        )
        ranks = response.data.document_ranks
        return [(int(rank.index), float(rank.relevance_score)) for rank in ranks]

    def command_a(self, prompt: str, *, max_tokens: int | None = None) -> str:
        """Single-shot Cohere Command A completion over OCI GenAI chat (on-demand).

        Returns the model's text. SDK shapes were pinned against the installed
        `oci` package: ChatDetails(chat_request=CohereChatRequest(...)) → chat()
        → data.chat_response.text."""
        import oci

        self._require()
        models = oci.generative_ai_inference.models
        chat_request = models.CohereChatRequest(
            message=prompt,
            max_tokens=max_tokens or self._settings.cloud_max_tokens,
            temperature=0.0,
        )
        details = models.ChatDetails(
            compartment_id=self._settings.oci_compartment_id,
            serving_mode=models.OnDemandServingMode(
                model_id=self._settings.oci_command_a_model
            ),
            chat_request=chat_request,
        )
        response = self._with_retry("chat", lambda: self._client().chat(details))
        return response.data.chat_response.text or ""

    def extract_entities(self, text: str) -> entity_graph.EntityExtraction:
        """Command A → entities + relationships for a prose document (Stage 2).
        Parsing is defensive, so a malformed reply yields an empty extraction."""
        if not (text or "").strip():
            return entity_graph.EntityExtraction()
        raw = self.command_a(entity_graph.build_prompt(text))
        return entity_graph.parse(raw)

    # Optional cloud review of model-authored tool code.

    def tool_review_available(self) -> bool:
        """True only if the Grok code reviewer is opted-in AND cloud is configured.
        Enabling it sends tool CODE to xAI Grok (us-chicago-1) — cloud egress."""
        return bool(getattr(self._settings, "allow_tool_code_review", False)) and self.available()

    def grok_review(self, code: str, task: dict[str, Any]) -> dict[str, Any]:
        """Ask xAI Grok (OCI GenAI, us-chicago-1, on-demand) to review authored
        tool code: check it for safety, improve it if warranted, and return a JSON
        verdict. Uses the *generic* chat request shape (Cohere's differs). The host
        AST-gate still validates whatever code is used — this never widens caps.

        Returns ``{safe, reasons, improved_code}``. Callers treat any exception as
        fail-soft (review skipped) since the AST-gate is the load-bearing control."""
        import oci

        self._require()
        settings = self._settings
        models = oci.generative_ai_inference.models
        system = (
            "You are a strict security reviewer for a local agent that runs "
            "model-authored Python tools in a locked-down sandbox. The tool is a "
            "single `run(inputs, model)` function that may only use pure-stdlib "
            "logic and an injected `model()` bridge — no network, files, os, eval, "
            "exec, imports outside a small allowlist, or dunder access. Review the "
            "code for safety and correctness against the task. Reply with ONLY a "
            "JSON object: {\"safe\": bool, \"reasons\": [string], \"improved_code\": "
            "string}. Set improved_code to a better full `run(inputs, model)` "
            "source if you can improve it (same constraints), else an empty string. "
            "Never add capabilities the constraints forbid."
        )
        user = json.dumps({"task": task, "code": code}, ensure_ascii=False)
        messages = [
            models.Message(role="SYSTEM", content=[models.TextContent(text=system)]),
            models.Message(role="USER", content=[models.TextContent(text=user)]),
        ]
        chat_request = models.GenericChatRequest(
            api_format=models.GenericChatRequest.API_FORMAT_GENERIC,
            messages=messages,
            max_tokens=min(settings.cloud_max_tokens, 4096),
            temperature=0.0,
        )
        details = models.ChatDetails(
            compartment_id=settings.oci_compartment_id,
            serving_mode=models.OnDemandServingMode(model_id=settings.oci_grok_model),
            chat_request=chat_request,
        )
        endpoint = settings.oci_chicago_endpoint or settings.oci_genai_endpoint
        response = self._with_retry(
            "grok_review", lambda: self._client(endpoint=endpoint).chat(details)
        )
        text = response.data.chat_response.choices[0].message.content[0].text or ""
        return _parse_grok_review(text)
