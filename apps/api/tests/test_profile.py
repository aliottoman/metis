from __future__ import annotations

from waqil_api.config import Settings
from waqil_api.profile import ProfileStore


def test_profile_roundtrip(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path / "data", allow_test_backends=True)
    store = ProfileStore(settings)
    assert store.load().content == ""
    saved = store.save("  I am Ali, I work at Oracle and like Cohere.  ")
    assert saved.content == "I am Ali, I work at Oracle and like Cohere."
    assert saved.characters == len(saved.content)
    assert store.load().content == saved.content


def test_injection_text_is_bounded(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data", profile_max_chars=20, allow_test_backends=True
    )
    store = ProfileStore(settings)
    store.save("x" * 500)
    # The editor still sees the full stored content...
    assert store.load().characters == 500
    # ...but the injected slice is capped for the prompt budget.
    assert len(store.injection_text()) == 20
