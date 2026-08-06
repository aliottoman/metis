"""Model routing preference — one heavyweight local model at a time.

By default Metis splits roles across models (a planner, a coder, a quality
reviewer). Each role switch makes Ollama unload one model and load another
into unified memory, which is slow and doubles the resident footprint on a
single Mac. This lets a user pin one model for every role instead, so nothing
ever swaps mid-session. Stored as a small local JSON file the user owns,
mirroring `profile.py`.
"""
from __future__ import annotations

import json

from .config import Settings
from .contracts import ModelPreferenceV1


def is_cloud_model(model: str) -> bool:
    """Whether a model name names a hosted model rather than a local one.

    Ollama spells the hosted variants two ways — the suffix form
    ``gpt-oss:120b-cloud`` and the bare tag ``glm-5.2:cloud`` — and matching
    only the first silently misread every model of the second kind as local.
    """
    name = model.strip()
    return name.endswith("-cloud") or name.endswith(":cloud")


# Whether each hosted model honours tool calling, measured — not assumed —
# against the real project-step contract through the Ollama daemon on
# 2026-08-05. Tool calling is a training property, not a platform one:
# minimax-m3 fails it on the same endpoint where gemma4 succeeds, so a new
# subscription model belongs here only after it has been tested the same way.
# A model absent from this table gets the benefit of the doubt — the loop's
# malformed-streak breaker still bounds a wrong guess — but a model measured
# to ignore tool calls is refused at selection time, because the alternative
# is three malformed replies at step five of somebody's build.
HOSTED_MODEL_TOOL_CALLING: dict[str, bool] = {
    "gpt-oss:120b-cloud": True,
    "gpt-oss:20b-cloud": True,
    "gemma4:31b-cloud": True,
    "minimax-m3:cloud": False,
}


def hosted_model_capability_error(model: str) -> str:
    """Why this hosted model cannot drive structured work, or "" when it can.

    Hosted decode rides entirely on tool calling — Ollama Cloud enforces no
    other structure — so a hosted model that does not honour it cannot produce
    one readable step. Local names always pass: they are grammar-constrained
    and never consult this record.
    """
    name = model.strip()
    if not is_cloud_model(name) or HOSTED_MODEL_TOOL_CALLING.get(name, True):
        return ""
    return (
        f"{name} does not honour tool calling on Ollama Cloud, so it cannot "
        "return a readable structured step. Choose a hosted model that does, "
        "such as gpt-oss:120b-cloud."
    )


class ModelPreferenceStore:
    """Read/write the local model-routing preference at `Settings.model_preference_path`."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def load(self) -> ModelPreferenceV1:
        try:
            raw = json.loads(
                self._settings.model_preference_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            raw = {}
        if not isinstance(raw, dict):
            raw = {}
        mode = raw.get("mode") if raw.get("mode") in ("split", "pinned") else "split"
        model = raw.get("model") if isinstance(raw.get("model"), str) else None
        if mode == "pinned" and not model:
            mode = "split"
        provider = (
            raw.get("provider")
            if raw.get("provider") in ("local", "oci", "cohere")
            else "local"
        )
        if provider == "oci" and not self.oci_available:
            provider = "local"
        if provider == "cohere" and not self.cohere_available:
            provider = "local"
        raw_tools = raw.get("oci_tools")
        oci_tools = (
            [item for item in raw_tools if item in ("x_search", "code_interpreter")]
            if isinstance(raw_tools, list)
            else ["code_interpreter"]
        )
        return ModelPreferenceV1(
            mode=mode,
            model=model,
            provider=provider,
            oci_tools=list(dict.fromkeys(oci_tools)),
            oci_available=self.oci_available,
            cohere_available=self.cohere_available,
        )

    @property
    def oci_available(self) -> bool:
        return bool(
            self._settings.allow_oci_responses
            and self._settings.oci_responses_project_id.strip()
        )

    @property
    def cohere_available(self) -> bool:
        return bool(self._settings.cohere_api_key.strip())

    def save(
        self,
        mode: str,
        model: str | None,
        *,
        provider: str = "local",
        oci_tools: list[str] | None = None,
    ) -> ModelPreferenceV1:
        if mode not in ("split", "pinned"):
            raise ValueError("mode must be 'split' or 'pinned'")
        if mode == "pinned" and not (model and model.strip()):
            raise ValueError("pinned mode requires a model name")
        if mode == "pinned" and model:
            # Refused here, where the choice is made, rather than at step five
            # of a build: a pinned model drives every role, and a hosted model
            # that ignores tool calls cannot answer a single structured call.
            capability_error = hosted_model_capability_error(model)
            if capability_error:
                raise ValueError(capability_error)
        if provider not in ("local", "oci", "cohere"):
            raise ValueError("provider must be 'local', 'oci' or 'cohere'")
        if provider == "oci" and not self.oci_available:
            raise ValueError(
                "OCI Responses requires WAQIL_ALLOW_OCI_RESPONSES=true and "
                "WAQIL_OCI_RESPONSES_PROJECT_ID"
            )
        if provider == "cohere" and not self.cohere_available:
            raise ValueError("Cohere requires WAQIL_COHERE_API_KEY")
        selected_tools = list(dict.fromkeys(oci_tools or []))
        if any(item not in ("x_search", "code_interpreter") for item in selected_tools):
            raise ValueError("unsupported OCI native tool")
        path = self._settings.model_preference_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "mode": mode,
                    "model": model.strip() if model else None,
                    "provider": provider,
                    "oci_tools": selected_tools,
                }
            ),
            encoding="utf-8",
        )
        return self.load()

    def project_coder(self) -> str:
        """The coder a project build should use, or "" to keep the default.

        A whole-application build is the one workload where the local models
        measurably fall short: benchmarked on the same specification they
        deliver every file and still land two to seven defects, and a repair
        turn takes upwards of forty minutes because each structured step costs
        about a minute. The same step against the hosted model comes back in
        about five seconds, so project runs default to it.

        A pinned preference deliberately does NOT block this. Pinning is not
        the statement it looks like: launching a local model session pins the
        preference as a side effect, so anyone who has ever started a local
        model has one, and gating on it meant the default never fired for the
        people it was written for. Pinning a *cloud* model is a real choice
        about this workload, and that one is honored.

        The opt-outs are the settings: project_cloud_coder turns it off, and
        project_cloud_coder_model chooses a different hosted model.
        """
        if not self._settings.project_cloud_coder:
            return ""
        preference = self.load()
        if preference.mode == "pinned" and is_cloud_model(preference.model or ""):
            return ""
        coder = self._settings.project_cloud_coder_model
        capability_error = hosted_model_capability_error(coder)
        if capability_error:
            # A configuration error, surfaced where the route is chosen. The
            # alternative — routing the build and letting it die on malformed
            # replies at step five — reports a settings mistake as the model
            # replying unintelligibly.
            raise ValueError(capability_error)
        return coder

    def resolve_aliases(self) -> dict[str, str]:
        """The `model_aliases` a new run should use, honoring the preference."""
        preference = self.load()
        provider_aliases = {
            "_provider": preference.provider,
            "_oci_tools": ",".join(preference.oci_tools),
        }
        if preference.mode == "pinned" and preference.model:
            return {
                "planner": preference.model,
                "coder": preference.model,
                "quality": preference.model,
                **provider_aliases,
            }
        return {
            "planner": self._settings.planner_model,
            "coder": self._settings.coder_model,
            "quality": self._settings.quality_model,
            **provider_aliases,
        }
