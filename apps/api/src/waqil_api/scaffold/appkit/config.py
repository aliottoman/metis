"""Configuration that fails features, never imports.

Importing the application must succeed on a machine with no cloud
environment at all — otherwise even the health route dies with it. So
nothing here reads the environment at import time: a required value is
fetched when the feature needing it first runs, and a missing one raises
``ConfigError`` naming the exact variable, which the route turns into a
clear error instead of a stack trace.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class ConfigError(RuntimeError):
    """A required setting is absent. The message names it; nothing guesses."""


def load_dotenv(path: str | Path = ".env") -> None:
    """Read KEY=VALUE lines into os.environ without overriding real values.

    A deliberate tiny subset of python-dotenv — no interpolation, no
    multiline values — so standalone `uvicorn app.main:app` runs pick up a
    local .env. Under Metis the environment is injected and this is a no-op.
    """
    file = Path(path)
    if not file.is_file():
        return
    for line in file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        name = name.strip()
        value = value.strip().strip("'\"")
        if name and name not in os.environ:
            os.environ[name] = value


def optional(name: str, default: str = "") -> str:
    """An environment value, or the default when unset or blank."""
    return os.environ.get(name, "").strip() or default


def require(name: str) -> str:
    """An environment value that must exist — raising here, at use time."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(
            f"{name} is not set. Add it to the environment or to .env — "
            "see .env.example for every setting this application reads."
        )
    return value


@dataclass(frozen=True)
class OciResponsesConfig:
    """Everything the OCI Responses adapter needs, read lazily via from_env."""

    base_url: str
    project_id: str
    model_id: str
    profile: str
    config_file: str

    @classmethod
    def from_env(cls) -> "OciResponsesConfig":
        load_dotenv()
        return cls(
            base_url=require("OCI_RESPONSES_BASE_URL"),
            project_id=require("OCI_RESPONSES_PROJECT_ID"),
            model_id=require("OCI_RESPONSES_MODEL_ID"),
            profile=optional("OCI_PROFILE", "DEFAULT"),
            config_file=optional("OCI_CONFIG_FILE"),
        )
