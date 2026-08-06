"""One-click launch recipes, drafted by Command A+.

The claim under test is not "the model writes good recipes" — that is judged
live — but the machinery around it: the context the model sees is bounded and
read-only, the draft must survive the scanner's OWN parser before it touches
disk, an existing manifest is never overwritten, and a generated recipe
arrives exactly as untrusted as a hand-written one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from waqil_api.asset_recipe import RecipeError, gather_recipe_context, write_recipe
from waqil_api.config import Settings
from waqil_api.contracts import AssetRecipeV1
from waqil_api.main import create_app


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "projects"
    project = root / "invoice-extractor"
    project.mkdir(parents=True)
    (project / "README.md").write_text("# Invoice Extractor\n\nA Streamlit tool.\n")
    (project / "requirements.txt").write_text("streamlit==1.31\n")
    (project / "app.py").write_text("import streamlit as st\nst.title('hi')\n")
    (project / ".env").write_text("SECRET_TOKEN=do-not-read\n")
    junk = project / "node_modules" / "left-pad"
    junk.mkdir(parents=True)
    (junk / "index.js").write_text("module.exports = 1\n")
    return project


STREAMLIT_RECIPE = AssetRecipeV1(
    entrypoint="app.py",
    launch_command=[
        "{uv}", "run", "--isolated", "--no-project", "--no-env-file",
        "--with-requirements", "requirements.txt", "--with", "streamlit",
        "--", "python", "-m", "streamlit", "run", "app.py",
        "--server.address", "{host}", "--server.port", "{port}",
        "--server.headless", "true",
    ],
    env_keys=["INVOICE_API_KEY"],
)


def test_the_context_is_bounded_and_never_reads_env_values(tmp_path) -> None:
    context = gather_recipe_context(_project(tmp_path))
    assert "app.py" in context["files"]
    # Build detritus and dotfiles never reach the model.
    assert not any("node_modules" in item for item in context["files"])
    assert not any(item.startswith(".env") for item in context["files"])
    assert "do-not-read" not in json.dumps(context)
    assert "requirements.txt" in context["config_files"]
    assert "app.py" in context["entry_file_heads"]


def test_a_valid_draft_lands_as_the_scanner_expects(tmp_path) -> None:
    project = _project(tmp_path)
    body = write_recipe(project, STREAMLIT_RECIPE)
    stored = json.loads((project / ".metis" / "asset.json").read_text())
    assert stored == body
    assert stored["launch"]["command"][0] == "{uv}"
    assert stored["schema_version"] == "1"


def test_an_existing_manifest_is_never_overwritten(tmp_path) -> None:
    project = _project(tmp_path)
    write_recipe(project, STREAMLIT_RECIPE)
    with pytest.raises(RecipeError, match="delete it first"):
        write_recipe(project, STREAMLIT_RECIPE)


def test_a_draft_the_scanner_would_reject_never_touches_disk(tmp_path) -> None:
    project = _project(tmp_path)
    # PATH is a reserved environment key, so the scanner's parser drops the
    # command — and a manifest that looks configured but refuses to launch is
    # exactly the artifact this guard exists to prevent.
    poisoned = STREAMLIT_RECIPE.model_copy(update={"env_keys": ["PATH"]})
    with pytest.raises(RecipeError, match="did not survive validation"):
        write_recipe(project, poisoned)
    assert not (project / ".metis" / "asset.json").exists()


class _FakeCohere:
    available = True

    def __init__(self) -> None:
        self.saw_context: dict | None = None

    async def draft_asset_recipe(self, context: dict) -> AssetRecipeV1:
        self.saw_context = context
        return STREAMLIT_RECIPE


def test_the_endpoint_generates_validates_and_leaves_trust_ungranted(
    settings: Settings, tmp_path
) -> None:
    project = _project(tmp_path)
    configured = settings.model_copy(update={"asset_roots": [project.parent]})
    app = create_app(configured)
    with TestClient(app) as client:
        fake = _FakeCohere()
        app.state.runtime.model.cohere = fake

        catalog = client.post("/api/v1/assets/scan").json()
        asset = next(item for item in catalog if item["name"] == "Invoice Extractor")
        assert asset["launch_configured"] is False

        generated = client.post(f"/api/v1/assets/{asset['id']}/manifest/generate")
        assert generated.status_code == 200, generated.text
        body = generated.json()
        # Configured by the draft; trust still belongs to the human review.
        assert body["launch_configured"] is True
        assert body["launch_approved"] is False
        assert body["launch_command"][0] == "{uv}"
        # The model saw the bounded context, not the raw folder.
        assert fake.saw_context is not None
        assert "do-not-read" not in json.dumps(fake.saw_context)

        # Second click: the existing manifest is refused, not clobbered.
        again = client.post(f"/api/v1/assets/{asset['id']}/manifest/generate")
        assert again.status_code == 409


def test_the_endpoint_names_the_missing_key_without_a_provider(
    settings: Settings, tmp_path
) -> None:
    project = _project(tmp_path)
    configured = settings.model_copy(update={"asset_roots": [project.parent]})
    app = create_app(configured)
    with TestClient(app) as client:
        catalog = client.post("/api/v1/assets/scan").json()
        asset = next(item for item in catalog if item["name"] == "Invoice Extractor")
        response = client.post(f"/api/v1/assets/{asset['id']}/manifest/generate")
        assert response.status_code == 503
        assert "WAQIL_COHERE_API_KEY" in response.json()["detail"]
