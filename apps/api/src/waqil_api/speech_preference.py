"""Speech preference — which service listens, and how Metis speaks back.

A separate store from `model_preference.py` rather than another field on it,
because these are separate decisions: dictating through one service says
nothing about which model should answer a chat message, and one file would
have made the two look like one choice. The shape is the one this codebase
already uses for user-owned settings — a small local JSON file, hand-editable,
degraded rather than fatal when it does not parse, and never holding a key.

Environment values are startup defaults; what is stored here is the choice the
owner made in Settings. A browser radio does not edit `.env`, and this store is
the reason it does not have to pretend to.
"""

from __future__ import annotations

import json

from .config import Settings
from .contracts import SpeechPreferenceV1

STT_PROVIDERS = ("cohere", "elevenlabs")

# Models allowed to reason during a spoken conversation. Server-owned: the
# voice model is not the caller's to choose, and an ElevenLabs request naming
# a model gets that name discarded before it reaches routing.
#
# Two criteria, both of which the local models fail. A voice turn has to come
# back inside the pause a person leaves after speaking, and a local 35B answers
# in tens of seconds; and the voice graph decodes structured replies, so a
# model must be one measured to honour tool calling (HOSTED_MODEL_TOOL_CALLING
# in model_preference.py). The default leads the list.
VOICE_MODELS: tuple[str, ...] = (
    "deepseek-v4-flash:cloud",
    "gpt-oss:20b-cloud",
    "gpt-oss:120b-cloud",
    "glm-5.2:cloud",
)


class SpeechPreferenceStore:
    """Read/write the local speech preference at `Settings.speech_preference_path`."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def cohere_available(self) -> bool:
        return bool(self._settings.cohere_api_key.strip())

    @property
    def elevenlabs_available(self) -> bool:
        return bool(self._settings.elevenlabs_api_key.strip())

    def load(self) -> SpeechPreferenceV1:
        try:
            raw = json.loads(
                self._settings.speech_preference_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            raw = {}
        if not isinstance(raw, dict):
            raw = {}
        stored = raw.get("stt_provider")
        provider = stored if stored in STT_PROVIDERS else "cohere"
        voice_model = raw.get("voice_model")
        if not isinstance(voice_model, str) or voice_model not in VOICE_MODELS:
            voice_model = ""
        return SpeechPreferenceV1(
            stt_provider=self._reachable(provider),
            spoken_confirmation=bool(raw.get("spoken_confirmation")),
            voice_model=voice_model or self.voice_model(),
            cohere_available=self.cohere_available,
            elevenlabs_available=self.elevenlabs_available,
            voice_models=list(VOICE_MODELS),
        )

    def _reachable(self, provider: str) -> str:
        """The stored choice, or the other provider when it has no key.

        The same degrade the model preference makes, for the same reason: a key
        can be removed long after the choice was saved, and dictation failing
        with "not configured" while a working transcriber sits beside it is a
        worse answer than quietly using the one that works. With no ElevenLabs
        key at all this can only ever return "cohere", which is exactly the
        behavior that existed before there was a choice.
        """
        if provider == "elevenlabs" and not self.elevenlabs_available:
            return "cohere" if self.cohere_available else provider
        if provider == "cohere" and not self.cohere_available:
            return "elevenlabs" if self.elevenlabs_available else provider
        return provider

    def voice_model(self) -> str:
        """The configured default, refused if it is not on the allowlist."""
        configured = self._settings.voice_model.strip()
        return configured if configured in VOICE_MODELS else VOICE_MODELS[0]

    def save(
        self,
        stt_provider: str,
        *,
        spoken_confirmation: bool = False,
        voice_model: str | None = None,
    ) -> SpeechPreferenceV1:
        if stt_provider not in STT_PROVIDERS:
            raise ValueError("stt_provider must be 'cohere' or 'elevenlabs'")
        if stt_provider == "cohere" and not self.cohere_available:
            raise ValueError("Cohere dictation requires WAQIL_COHERE_API_KEY")
        if stt_provider == "elevenlabs" and not self.elevenlabs_available:
            raise ValueError("ElevenLabs dictation requires WAQIL_ELEVENLABS_API_KEY")
        # None keeps what is stored; "" returns to the configured default. The
        # distinction lets a surface that only knows about dictation save
        # without silently resetting the voice model.
        if voice_model is None:
            selected = self.load().voice_model
        elif not voice_model.strip():
            selected = self.voice_model()
        elif voice_model.strip() in VOICE_MODELS:
            selected = voice_model.strip()
        else:
            raise ValueError(f"{voice_model.strip()} is not a voice model")
        path = self._settings.speech_preference_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "stt_provider": stt_provider,
                    "spoken_confirmation": bool(spoken_confirmation),
                    "voice_model": selected,
                }
            ),
            encoding="utf-8",
        )
        return self.load()
