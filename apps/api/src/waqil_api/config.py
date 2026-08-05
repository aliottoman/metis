from __future__ import annotations

import ipaddress
import json
import os
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings. Unsafe capabilities remain opt-in."""

    model_config = SettingsConfigDict(
        env_prefix="WAQIL_",
        env_file=".env",
        extra="ignore",
    )

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    data_dir: Path = Path(".data")
    repo_root: Path = Path(__file__).resolve().parents[4]
    model_backend: str = "auto"
    ollama_base_url: str = "http://127.0.0.1:11434"
    planner_model: str = "qwen3.6:35b-mlx"
    coder_model: str = "north-mini-code-1.0:mlx-nvfp4"
    quality_model: str = "north-mini-code-1.0:mlx-mxfp8"
    context_window: int = Field(default=32768, ge=4096, le=262144)
    max_output_tokens: int = Field(default=8192, ge=256, le=32768)
    # Turns a wedged model runtime into a typed failure instead of a stuck run.
    # For a streamed answer this covers prompt evaluation and the first token;
    # after that the stall timeout below takes over, so a slow-but-progressing
    # local answer is never cut off mid-sentence.
    model_call_timeout_seconds: float = Field(default=600.0, ge=30.0, le=1800.0)
    # Longest silence allowed between two streamed chunks before the call fails.
    model_stall_timeout_seconds: float = Field(default=120.0, ge=15.0, le=600.0)
    # Streams the planner's thinking to the run timeline as a separate channel,
    # so it can be read without ever being mixed into the answer text.
    stream_model_reasoning: bool = True
    # Bounds the advisory proposal worker so it cannot stall the root graph.
    deep_worker_timeout_seconds: int = Field(default=120, ge=15, le=300)
    # Unload after each call so two models never share unified memory.
    ollama_keep_alive: str = "0"
    # A manually launched model stays warm briefly after its last completed use.
    # The session control may override this per launch, but never starts a model.
    local_model_idle_seconds: int = Field(default=300, ge=60, le=86_400)
    # How long the API waits, with no client calling it at all, before unloading
    # the model it launched. The UI polls the session while it is open and
    # visible, so a gap this long means every window is closed or in the
    # background — the case where holding the weights costs memory and battery
    # and buys nothing. 0 disables it and leaves Ollama's keep_alive in charge.
    model_release_after_idle_seconds: int = Field(default=180, ge=0, le=86_400)
    max_upload_bytes: int = Field(
        default=10 * 1024 * 1024, ge=1024, le=100 * 1024 * 1024
    )
    max_text_attachment_bytes: int = Field(
        default=64 * 1024, ge=1024, le=512 * 1024
    )
    # Project builds default to a hosted coder. Ollama serves cloud models
    # through the same loopback API as local ones — the name carries the
    # "-cloud" suffix and the daemon proxies it — so this is a model-name
    # change, not a second provider. Benchmarked, a structured build step
    # costs about a minute locally and about five seconds hosted, and the
    # local models leave two to seven defects on the same specification.
    # Set project_cloud_coder=false to keep whole-application builds local;
    # an explicitly pinned model always outranks this either way.
    project_cloud_coder: bool = True
    project_cloud_coder_model: str = "gpt-oss:120b-cloud"

    reference_runner_mode: str = "podman"
    reference_runner_image: str = "localhost/metis/reference-architecture-tool:0.3.0"
    # The inner sandbox stops at 120s; the rest is Podman startup and cleanup.
    reference_runner_timeout_seconds: int = Field(default=150, ge=135, le=600)
    allow_test_backends: bool = False
    cors_origins: list[str] = ["http://127.0.0.1:3000", "http://localhost:3000"]

    # Cloud retrieval (OCI Cohere embed, rerank, Command A). Opt-in; any unmet
    # precondition falls back to local keyword search. Vectors are stored locally.
    allow_cloud_embeddings: bool = False
    oci_profile: str = "DEFAULT"
    # Empty uses the SDK default ~/.oci/config; set to point somewhere else.
    oci_config_file: str = ""
    oci_compartment_id: str = ""
    # Empty lets the SDK derive the endpoint from the profile region.
    oci_genai_endpoint: str = ""
    oci_chicago_endpoint: str = (
        "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
    )
    # Rerank may live in a different region than embed; empty reuses the embed one.
    oci_rerank_endpoint: str = ""
    oci_embed_model: str = "cohere.embed-v4.0"
    oci_rerank_model: str = "cohere.rerank-v3.5"
    oci_command_a_model: str = "cohere.command-a-03-2025"
    embed_batch: int = Field(default=96, ge=1, le=96)
    cloud_max_tokens: int = Field(default=2048, ge=256, le=8192)
    # Bounded retry so one transient network failure cannot abort a whole index run.
    cloud_max_retries: int = Field(default=4, ge=0, le=10)
    cloud_retry_base_seconds: float = Field(default=1.0, ge=0.0, le=30.0)
    cloud_retry_max_seconds: float = Field(default=20.0, ge=1.0, le=120.0)

    # Lets a tool's model author its diagram code; needs the v2 sandbox image.
    tool_model_authoring: bool = False

    # Kill-switches for the tool lifecycle: the whole factory, the drafting entry
    # point, the per-run model-call ceiling, and individual tool slugs.
    tool_factory_enabled: bool = True
    tool_definition_enabled: bool = True
    # Lets an explicit build request finish in one run inside the trusted boundary.
    tool_trusted_auto_activation: bool = True
    tool_global_max_broker_calls: int = Field(default=4, ge=0, le=16)
    tool_disabled_slugs: list[str] = Field(default_factory=list)

    # Bounds each model-authored tool run in the restricted executor.
    tool_authored_timeout_seconds: int = Field(default=10, ge=1, le=120)
    tool_authored_memory_mb: int = Field(default=512, ge=64, le=4096)
    # Wall-clock allowance for ONE brokered model call made from inside a tool.
    # Separate from the code budget above: a local 35B reply takes far longer than
    # the few seconds of CPU an authored tool should ever need.
    tool_authored_model_call_timeout_seconds: int = Field(default=300, ge=5, le=900)
    # Optional Grok review of authored code; sends that code to the cloud.
    allow_tool_code_review: bool = False
    oci_grok_model: str = "xai.grok-4.3"

    # Per-run cloud provider. Metis keeps memory authoritative via store=False.
    allow_oci_responses: bool = False
    oci_responses_project_id: str = ""
    oci_responses_base_url: str = (
        "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/openai/v1"
    )
    oci_responses_max_output_tokens: int = Field(default=16_384, ge=256, le=131_072)
    oci_recent_history_chars: int = Field(default=64_000, ge=12_000, le=400_000)
    oci_memory_context_chars: int = Field(default=24_000, ge=8_000, le=100_000)

    # Bounds the project act→observe→decide loop. Writes are staged into an
    # overlay as the loop runs and reach disk only through the single
    # batch approval at the end, so a large step budget spends model time,
    # never unreviewed filesystem authority.
    project_agent_max_steps: int = Field(default=48, ge=2, le=200)
    project_manifest_max_files: int = Field(default=8_000, ge=100, le=50_000)
    project_manifest_sample_chars: int = Field(default=80_000, ge=8_000, le=400_000)
    project_tool_result_chars: int = Field(default=48_000, ge=4_000, le=200_000)
    project_max_write_bytes: int = Field(default=1_000_000, ge=1_024, le=8_000_000)
    # Caps one turn's staged changeset. Files is the count of distinct paths;
    # bytes is the total staged content held in graph state (which is
    # checkpointed, so this also bounds checkpoint growth).
    project_staged_max_files: int = Field(default=48, ge=1, le=256)
    project_staged_max_bytes: int = Field(default=4_000_000, ge=10_000, le=32_000_000)

    # Reviewed verification checks. The agent may only name a check declared in
    # the project's own .metis/verify.json, and the recipe is approved once by
    # fingerprint; runs per turn are bounded so a failing check cannot loop.
    project_verify_enabled: bool = True
    project_verify_timeout_seconds: int = Field(default=300, ge=5, le=1_800)
    project_verify_output_chars: int = Field(default=12_000, ge=500, le=120_000)
    project_verify_max_runs: int = Field(default=6, ge=0, le=30)

    # Verified API facts injected into every build turn, deterministically.
    #
    # This is deliberately NOT retrieval. The reference was indexed as a corpus
    # source first, and measured: a build prompt's nearest neighbours are the
    # transcripts of previous build prompts, so all 31 retrieved passages were
    # run history and none were the reference. Worse, it compounds — every build
    # indexes its own transcript, growing the very corpus that outranks it.
    # A coding reference for the stack being built on is not "possibly
    # relevant", so it is read from disk and always sent.
    # Budgets are per build step. The first values (14k/6k) were set by eye and
    # measured wrong: the whole library is ~14.2k, so the cloud budget dropped
    # the OCI reference by 222 characters on an OCI build, and the local budget
    # was under the size of a single document so local builds got nothing at
    # all. Sized to fit the library whole, with headroom for it to grow.
    project_reference_enabled: bool = True
    project_reference_max_chars: int = Field(default=40_000, ge=0, le=120_000)
    project_reference_max_chars_local: int = Field(default=9_000, ge=0, le=60_000)

    # The build loop's own checks on a staged changeset, before the user is ever
    # offered it. The wiring gate is pure AST and always runs; the sandbox
    # actually imports the project inside the reviewed container and is the only
    # place model-authored project code is executed. Either can be turned off
    # without a code change, and a sandbox that cannot run degrades to the
    # wiring gate rather than passing the build silently.
    # Ruff and mypy over the staged changeset, resolved against the packages the
    # project will actually run on. This is the rung that knows things the code
    # cannot say about itself: a keyword argument the callee does not accept
    # parses perfectly and imports perfectly, and was invented independently by
    # a frontier model and a local one. Neither tool executes what it reads.
    project_typecheck_enabled: bool = True
    project_typecheck_timeout_seconds: int = Field(default=60, ge=5, le=600)
    project_wiring_gate_enabled: bool = True
    project_sandbox_enabled: bool = True
    project_sandbox_image: str = "localhost/metis/project-verify:0.2.0"
    project_sandbox_timeout_seconds: int = Field(default=150, ge=30, le=600)
    project_sandbox_max_modules: int = Field(default=40, ge=1, le=200)
    # Booting the Podman VM costs about ten seconds, once per laptop boot. A
    # verification that silently does not happen is the failure this whole gate
    # exists to remove, so the default is to start it rather than skip the check.
    project_sandbox_autostart: bool = True
    # How long the VM may sit unused before Metis stops it again, mirroring what
    # the model session does with its weights. Deliberately not per-request: a
    # build turn verifies two or three times, and a stop between them would pay
    # the ten-second boot repeatedly to reclaim 1.7 GB for a few seconds. Metis
    # only ever stops a machine it started itself. 0 leaves it running.
    project_sandbox_release_after_idle_seconds: int = Field(default=600, ge=0, le=86_400)

    # Containers whose child directories become Assets on an explicit scan.
    # NoDecode accepts a single path, a separated list, or a JSON array.
    asset_roots: Annotated[list[Path], NoDecode] = Field(default_factory=list)

    # Personal knowledge: an always-on profile plus a just-in-time corpus.
    profile_max_chars: int = Field(default=3_200, ge=0, le=16_000)
    corpus_chunk_chars: int = Field(default=1_200, ge=200, le=8_000)
    corpus_chunk_overlap: int = Field(default=150, ge=0, le=2_000)
    corpus_max_file_bytes: int = Field(default=1_000_000, ge=1_024, le=16_000_000)
    corpus_recall_k: int = Field(default=40, ge=1, le=400)
    corpus_top_k: int = Field(default=8, ge=1, le=50)
    corpus_context_chars: int = Field(default=8_000, ge=0, le=32_000)
    # Keeps rerank's weakest hits out of an answer; /corpus/search stays ungated.
    corpus_min_relevance: float = Field(default=0.05, ge=0.0, le=1.0)
    # Local call graph parsed during indexing; expansion adds neighbours of the
    # top vector hits so multi-hop code questions find related definitions.
    corpus_graph_enabled: bool = True
    corpus_graph_expand: bool = True
    corpus_graph_expand_seeds: int = Field(default=6, ge=0, le=50)
    corpus_graph_expand_k: int = Field(default=12, ge=0, le=100)

    # Same-document expansion: after rerank picks winners, pull the rest of the
    # top documents so "summarize this page" sees the page, not just its hits.
    corpus_page_expand: bool = True
    corpus_page_expand_pages: int = Field(default=2, ge=0, le=10)
    corpus_page_expand_k: int = Field(default=12, ge=0, le=100)

    # Completed runs become corpus documents so past work is retrievable. The
    # documents are always local; indexing them still needs source consent.
    run_history_enabled: bool = True
    run_history_max_chars: int = Field(default=12_000, ge=500, le=80_000)
    # Durable facts are proposed from a finished run, never activated by it.
    memory_harvest_enabled: bool = True
    memory_harvest_max_candidates: int = Field(default=3, ge=0, le=10)

    # Entity graph over prose. Off by default: it costs a cloud call per file.
    corpus_entity_graph: bool = False
    corpus_entity_max_chars: int = Field(default=8_000, ge=200, le=48_000)

    # Read-only Notion mirror. Tokens stay local and are never returned by the API.
    notion_token: str = ""
    notion_api_version: str = "2026-03-11"
    notion_sync_max_pages: int = Field(default=5_000, ge=1, le=50_000)

    # A deterministic verifier sends one bounded revision back to the generator
    # when strongly relevant passages went uncited. It makes no model call itself.
    answer_grounding_review: bool = True
    answer_max_revisions: int = Field(default=1, ge=0, le=3)
    answer_grounding_min_score: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("host")
    @classmethod
    def loopback_host_only(cls, value: str) -> str:
        if value not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Metis v1 may bind only to a loopback interface")
        return value

    @field_validator("ollama_base_url")
    @classmethod
    def loopback_ollama_only(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("ollama_base_url must be an HTTP(S) URL")
        try:
            is_loopback = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            is_loopback = parsed.hostname == "localhost"
        if not is_loopback:
            raise ValueError("Ollama must be accessed over loopback in v1")
        return value.rstrip("/")

    @field_validator("model_backend")
    @classmethod
    def valid_model_backend(cls, value: str) -> str:
        if value not in {"auto", "ollama", "deterministic"}:
            raise ValueError("model_backend must be auto, ollama, or deterministic")
        return value

    @field_validator("oci_responses_base_url")
    @classmethod
    def valid_oci_responses_base_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or not parsed.hostname.endswith(".oci.oraclecloud.com")
        ):
            raise ValueError("OCI Responses must use an HTTPS oci.oraclecloud.com endpoint")
        return value.rstrip("/")

    @field_validator("reference_runner_mode")
    @classmethod
    def valid_runner_mode(cls, value: str) -> str:
        if value not in {"podman", "local", "deterministic"}:
            raise ValueError("reference_runner_mode must be podman, local, or deterministic")
        return value

    @field_validator("asset_roots", mode="before")
    @classmethod
    def parse_asset_roots(cls, value: object) -> object:
        if value is None or value == "":
            return []
        if not isinstance(value, str):
            return value
        raw = value.strip()
        if not raw:
            return []
        if raw.startswith("["):
            try:
                decoded = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError("asset_roots must be a path or a JSON path array") from exc
            if not isinstance(decoded, list):
                raise ValueError("asset_roots JSON value must be an array")
            return decoded
        separator = "\n" if "\n" in raw else "," if "," in raw else os.pathsep
        if separator in raw:
            return [item.strip() for item in raw.split(separator) if item.strip()]
        return [raw]

    @property
    def database_path(self) -> Path:
        return self.data_dir / "waqil.db"

    @property
    def checkpoint_path(self) -> Path:
        return self.data_dir / "checkpoints.db"

    @property
    def blob_dir(self) -> Path:
        return self.data_dir / "blobs"

    @property
    def run_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def asset_approval_path(self) -> Path:
        return self.data_dir / "asset-launch-approvals.json"

    @property
    def asset_catalog_path(self) -> Path:
        return self.data_dir / "asset-catalog.json"

    @property
    def project_verify_approval_path(self) -> Path:
        return self.data_dir / "project-verify-approvals.json"

    @property
    def tool_bundle_dir(self) -> Path:
        return self.data_dir / "tool-bundles"

    @property
    def reference_skill_dir(self) -> Path:
        return self.repo_root / "skills" / "reference-architecture-generator"

    @property
    def project_reference_dir(self) -> Path:
        """The verified coding reference every build turn is given."""
        return self.repo_root / "reference"

    @property
    def reference_sandbox_runner(self) -> Path:
        return self.repo_root / "infra" / "sandbox" / "run_reference_architecture.py"

    @property
    def project_sandbox_runner(self) -> Path:
        return (
            self.repo_root / "infra" / "sandbox" / "project-verify" / "run_project_verify.py"
        )

    @property
    def corpus_dir(self) -> Path:
        """Local home for corpus state (embeddings live in SQLite, not here)."""
        return self.data_dir / "corpus"

    @property
    def notion_config_path(self) -> Path:
        return self.data_dir / "notion.json"

    @property
    def notion_mirror_dir(self) -> Path:
        return self.corpus_dir / "notion"

    @property
    def profile_path(self) -> Path:
        """The Tier-0 always-on personal profile (local, user-owned markdown)."""
        return self.data_dir / "profile.md"

    @property
    def model_preference_path(self) -> Path:
        """Which model(s) to route requests to (local, user-owned JSON)."""
        return self.data_dir / "model_preference.json"

    @property
    def model_session_path(self) -> Path:
        """Last explicit local-model session choices (never credentials)."""
        return self.data_dir / "model_session.json"

    @property
    def sku_rates_path(self) -> Path:
        """The Oracle SKU rate card (local, user-owned JSON).

        Seeded once from the copy vendored beside the SKU catalog, then owned by
        the user — a rate they verified or replaced with their contracted price
        must survive an update that ships a new seed.
        """
        return self.data_dir / "sku_rates.json"

    def prepare_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.blob_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.tool_bundle_dir.mkdir(parents=True, exist_ok=True)
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.notion_mirror_dir.mkdir(parents=True, exist_ok=True)
