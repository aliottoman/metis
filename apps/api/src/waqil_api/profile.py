"""Tier-0 personal profile — a small, always-on, user-curated file.

Unlike the Tier-1 corpus (probabilistic just-in-time retrieval), the profile is
injected verbatim on every turn. Stable facts the agent must never miss — who
the user is, their role, writing style, hard preferences — belong here rather
than in a long system prompt or a fragile retrieval hit. It is bounded on
injection so a hand-grown file can never blow the context budget, and it stays a
plain local markdown file the user fully owns and can edit by hand.
"""
from __future__ import annotations

from datetime import UTC, datetime

from .config import Settings
from .contracts import PersonalProfileV1


class ProfileStore:
    """Read/write the local Tier-0 profile markdown at `Settings.profile_path`."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def _raw_text(self) -> str:
        try:
            return self._settings.profile_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return ""

    def injection_text(self) -> str:
        """The bounded text injected into the prompt (≈ profile_max_chars)."""
        return self._raw_text()[: self._settings.profile_max_chars].strip()

    def load(self) -> PersonalProfileV1:
        """The full stored profile plus metadata, for the editor UI."""
        text = self._raw_text().strip()
        updated_at: datetime | None = None
        try:
            mtime = self._settings.profile_path.stat().st_mtime
            updated_at = datetime.fromtimestamp(mtime, UTC)
        except OSError:
            updated_at = None
        return PersonalProfileV1(
            content=text, characters=len(text), updated_at=updated_at
        )

    def save(self, content: str) -> PersonalProfileV1:
        """Persist the profile (contract-bounded upstream) and return the reload."""
        content = (content or "").strip()
        path = self._settings.profile_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content + ("\n" if content else ""), encoding="utf-8")
        return self.load()
