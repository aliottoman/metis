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
        provider = raw.get("provider") if raw.get("provider") in ("local", "oci") else "local"
        if provider == "oci" and not self.oci_available:
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
        )

    @property
    def oci_available(self) -> bool:
        return bool(
            self._settings.allow_oci_responses
            and self._settings.oci_responses_project_id.strip()
        )

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
        if provider not in ("local", "oci"):
            raise ValueError("provider must be 'local' or 'oci'")
        if provider == "oci" and not self.oci_available:
            raise ValueError(
                "OCI Responses requires WAQIL_ALLOW_OCI_RESPONSES=true and "
                "WAQIL_OCI_RESPONSES_PROJECT_ID"
            )
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
