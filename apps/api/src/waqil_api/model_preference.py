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
from .contracts import MODEL_ROLES, ModelPreferenceV1, RoleChainEntryV1


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
    # Measured 2026-08-08 against the five-scenario project-step probe
    # (first build step, mid-build, clarify-invited, plain question, finish):
    # each returned 5/5 well-formed function calls with required arguments
    # present, including correct use of the new respond talk tool. Their
    # *judgment* differed — deepseek-v4-flash answered every scenario with
    # list_files — but this record is about honouring tool calling, and the
    # coder choice is made elsewhere.
    # kimi-k3:cloud is deliberately absent: its probe was blocked by the
    # plan's extra-usage gate (HTTP 402), and this record only holds measured
    # results. Absent already means benefit of the doubt.
    "kimi-k2.7-code:cloud": True,
    "glm-5.2:cloud": True,
    "deepseek-v4-flash:cloud": True,
}


# Cline's subscription-backed catalog, exposed by the API so the web client
# never has to guess which names are included in ClinePass. The original list
# passed a live ProjectDirectionV1 probe on 2026-08-10. On 2026-09-26 the
# gateway returned model-not-found for kimi-k2.6 and deepseek-v4-flash, while
# mimo-v2.6-flash passed a live structured route probe. Paid Anthropic/xAI
# routes deliberately stay out of this list: they remain valid explicit model
# IDs, but presenting them beside subscription models made a zero-credit lane
# look healthy until the first HTTP 402.
CLINEPASS_MODELS: tuple[str, ...] = (
    "cline-pass/qwen3.7-plus",
    "cline-pass/glm-5.2",
    "cline-pass/kimi-k3",
    "cline-pass/kimi-k2.7-code",
    "cline-pass/deepseek-v4-pro",
    "cline-pass/minimax-m3",
    "cline-pass/mimo-v2.6-flash",
    "cline-pass/mimo-v2.5-pro",
    "cline-pass/mimo-v2.5",
    "cline-pass/qwen3.7-max",
)


def is_cline_gateway_model(model: str) -> bool:
    """Cline gateway IDs are provider/model names, not Ollama model tags."""
    provider, separator, name = model.strip().partition("/")
    return bool(separator and provider and name and not is_cloud_model(model))


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
            if raw.get("provider") in ("local", "oci", "cohere", "cline")
            else "local"
        )
        if provider == "oci" and not self.oci_available:
            provider = "local"
        if provider == "cohere" and not self.cohere_available:
            provider = "local"
        if provider == "cline" and not self.cline_available:
            provider = "local"
        if (
            provider == "cline"
            and mode == "pinned"
            and model
            and not is_cline_gateway_model(model)
        ):
            # Old preferences could carry an Ollama pin across a provider
            # switch. Display the effective split mode instead of claiming a
            # pin that the Cline gateway cannot serve.
            mode = "split"
            model = None
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
            role_chains=self._parse_chains(raw.get("role_chains")),
            oci_available=self.oci_available,
            cohere_available=self.cohere_available,
            cline_available=self.cline_available,
            cline_models=list(CLINEPASS_MODELS) if self.cline_available else [],
        )

    @staticmethod
    def _parse_chains(raw: object) -> dict[str, list[RoleChainEntryV1]]:
        """Stored chains, with anything unparseable dropped rather than fatal.

        The file is user-owned JSON: a hand-edit that breaks one entry should
        cost that entry, not every model preference in the app.
        """
        if not isinstance(raw, dict):
            return {}
        chains: dict[str, list[RoleChainEntryV1]] = {}
        for role, entries in raw.items():
            if role not in MODEL_ROLES or not isinstance(entries, list):
                continue
            parsed: list[RoleChainEntryV1] = []
            for entry in entries[:4]:
                try:
                    parsed.append(RoleChainEntryV1.model_validate(entry))
                except Exception:  # noqa: BLE001 - one bad entry costs itself
                    continue
            if parsed:
                chains[role] = parsed
        return chains

    @property
    def oci_available(self) -> bool:
        # Settings owns the formula (grok_lane_available) so the provider and
        # this store can never disagree about whether the lane exists.
        return self._settings.grok_lane_available

    @property
    def cohere_available(self) -> bool:
        return bool(self._settings.cohere_api_key.strip())

    @property
    def cline_available(self) -> bool:
        # One key covers both seats — the ClinePass coding models and the
        # Anthropic/xAI models behind the same gateway — so there is one flag.
        return bool(self._settings.cline_api_key.strip())

    def save(
        self,
        mode: str,
        model: str | None,
        *,
        provider: str = "local",
        oci_tools: list[str] | None = None,
        role_chains: dict[str, list[RoleChainEntryV1]] | None = None,
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
        if provider not in ("local", "oci", "cohere", "cline"):
            raise ValueError("provider must be 'local', 'oci', 'cohere' or 'cline'")
        if (
            provider == "cline"
            and mode == "pinned"
            and model
            and not is_cline_gateway_model(model)
        ):
            raise ValueError("a pinned Cline model must use a provider/model ID")
        if provider == "oci" and not self.oci_available:
            raise ValueError(
                "OCI Responses requires WAQIL_ALLOW_OCI_RESPONSES=true and "
                "WAQIL_OCI_RESPONSES_PROJECT_ID"
            )
        if provider == "cohere" and not self.cohere_available:
            raise ValueError("Cohere requires WAQIL_COHERE_API_KEY")
        if provider == "cline" and not self.cline_available:
            raise ValueError("the Cline lane requires WAQIL_CLINE_API_KEY")
        selected_tools = list(dict.fromkeys(oci_tools or []))
        if any(item not in ("x_search", "code_interpreter") for item in selected_tools):
            raise ValueError("unsupported OCI native tool")
        # None keeps what is stored; {} clears. The distinction lets a save
        # from a surface that predates chains leave them untouched.
        if role_chains is None:
            stored = self.load().role_chains
        else:
            stored = self._validated_chains(role_chains)
        path = self._settings.model_preference_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "mode": mode,
                    "model": model.strip() if model else None,
                    "provider": provider,
                    "oci_tools": selected_tools,
                    "role_chains": {
                        role: [entry.model_dump(mode="json") for entry in chain]
                        for role, chain in stored.items()
                    },
                }
            ),
            encoding="utf-8",
        )
        return self.load()

    def _validated_chains(
        self, chains: dict[str, list[RoleChainEntryV1]]
    ) -> dict[str, list[RoleChainEntryV1]]:
        """Chains fit to store, refused loudly where the choice is made.

        The same principle as the pinned-model gate above: a hosted model that
        ignores tool calling, or a lane with no key behind it, is refused at
        selection time — the alternative is a fallback ladder that "works"
        until the day it is needed and then fails exactly like no ladder.
        """
        cleaned: dict[str, list[RoleChainEntryV1]] = {}
        for role, chain in chains.items():
            if role not in MODEL_ROLES:
                raise ValueError(f"unknown model role: {role}")
            entries: list[RoleChainEntryV1] = []
            for entry in chain[:4]:
                if entry.provider == "oci" and not self.oci_available:
                    raise ValueError(
                        f"the {role} chain names OCI, which is not configured"
                    )
                if entry.provider == "cohere" and not self.cohere_available:
                    raise ValueError(
                        f"the {role} chain names Cohere, which is not configured"
                    )
                if entry.provider == "cline" and not self.cline_available:
                    raise ValueError(
                        f"the {role} chain names Cline, which is not configured"
                    )
                if entry.provider == "local" and entry.model:
                    capability_error = hosted_model_capability_error(entry.model)
                    if capability_error:
                        raise ValueError(capability_error)
                entries.append(entry)
            if entries:
                cleaned[role] = entries
        return cleaned

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
        """The `model_aliases` a new run should use, honoring the preference.

        Chains ride in the aliases as `_chain_<role>` JSON so they are frozen
        into the run row with everything else: a run that started under one
        ladder keeps it, exactly as it keeps its models. An explicit chain's
        first entry IS that role's primary — "completely controllable" means
        the ladder is the selection, not a decoration on it. Without an
        explicit coder chain a safety ladder is synthesized (Cohere when its
        key exists, then the default local coder) so an outage degrades a
        build instead of ending it; synthesized fallbacks never change the
        primary, so absent chains behave exactly as before on a healthy lane.
        """
        preference = self.load()
        provider_aliases = {
            "_provider": preference.provider,
            "_oci_tools": ",".join(preference.oci_tools),
        }
        if preference.mode == "pinned" and preference.model:
            aliases = {
                "planner": preference.model,
                "coder": preference.model,
                "quality": preference.model,
                **provider_aliases,
            }
            if preference.provider == "cline":
                # ClinePass resolves its models through _cline_model rather
                # than the local planner/coder aliases. Without this, the
                # model selected in the UI never reaches ordinary chat.
                aliases["_cline_model"] = preference.model
        else:
            aliases = {
                "planner": self._settings.planner_model,
                "coder": self._settings.coder_model,
                "quality": self._settings.quality_model,
                **provider_aliases,
            }
        if (
            preference.provider == "cline"
            and preference.mode == "split"
            and self._settings.cline_chat_model.strip()
        ):
            # The synthesis call alone may use this alias. Leave the planner
            # primary and any explicit planner chain intact for evidence and
            # project work. A planner-chain choice does not select chat answers.
            aliases["_cline_chat_model"] = self._settings.cline_chat_model.strip()
        for role in MODEL_ROLES:
            chain = preference.role_chains.get(role) or []
            if chain:
                primary = chain[0]
                if primary.provider == "local" and primary.model:
                    aliases[role] = primary.model
                aliases[f"_chain_{role}"] = json.dumps(
                    [entry.model_dump(mode="json") for entry in chain]
                )
            elif role == "coder":
                fallbacks: list[dict[str, str | None]] = []
                if preference.provider == "cline":
                    # The configured Cline coder is the implicit first rung.
                    # An env override may choose one of these safety models as
                    # that primary; do not retry the identical model while
                    # reporting that a fallback occurred.
                    fallbacks.extend(
                        {"provider": "cline", "model": model}
                        for model in (
                            "cline-pass/kimi-k3",
                            "cline-pass/kimi-k2.7-code",
                        )
                        if model != self._settings.cline_coder_model
                    )
                elif self.cohere_available and preference.provider != "cohere":
                    fallbacks.append({"provider": "cohere", "model": None})
                if self._settings.coder_model != aliases["coder"]:
                    fallbacks.append(
                        {"provider": "local", "model": self._settings.coder_model}
                    )
                if fallbacks:
                    aliases["_fallbacks_coder"] = json.dumps(fallbacks)
        if preference.provider == "cline" and not preference.role_chains.get("planner"):
            # Two production-shaped 16-file manifests returned no usable GLM
            # reply while Qwen completed both. The configured model remains
            # first so an explicit pin or WAQIL_CLINE_ORCHESTRATOR_MODEL stays
            # authoritative; the safety rung is GLM, de-duplicated when it is
            # already the selected primary.
            planner_primary = (
                preference.model
                if preference.mode == "pinned" and preference.model
                else self._settings.cline_orchestrator_model
            )
            planner_models = list(
                dict.fromkeys(
                    (
                        planner_primary,
                        "cline-pass/glm-5.2",
                    )
                )
            )
            aliases["_chain_planner"] = json.dumps(
                [{"provider": "cline", "model": model} for model in planner_models]
            )
        return aliases
