from __future__ import annotations

import ipaddress
import json
import os
from pathlib import Path
from typing import Annotated, Literal, Self
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
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
    max_text_attachment_bytes: int = Field(default=64 * 1024, ge=1024, le=512 * 1024)
    # Routes project builds to a hosted Ollama model, over tool-calling
    # decode: Ollama Cloud ignores `format` grammars (measured on three model
    # families — a live Ledger build died on three unreadable replies proving
    # it) but enforces tool calling, so hosted models get the same function
    # schemas the OCI provider sends, and local models keep their grammar.
    #
    # Still OFF by default: sending a build to someone's Ollama Cloud account
    # is an opt-in spend/privacy decision, not a routing default. The speed
    # case for opting in is real — a build step costs about a minute locally
    # and seconds hosted. Models measured to ignore tool calling
    # (HOSTED_MODEL_TOOL_CALLING in model_preference.py) are refused at
    # selection time.
    project_cloud_coder: bool = False
    project_cloud_coder_model: str = "gpt-oss:120b-cloud"

    # Compiles a loose whole-application request into a prescriptive spec
    # before the build plan is taken. Measured on the same model, same day,
    # same pipeline: the conversational prompt produced 38 blocking findings,
    # its prescriptive rewrite 11. Deliberately NOT a per-message switch: a
    # request at or past the length below — or one already carrying spec
    # structure — passes through untouched, so real specs are never rewritten
    # and loose asks always are. The rewrite is emitted as a run event and
    # its assumptions are named in the build's final response.
    project_spec_rewrite: bool = True
    project_spec_rewrite_max_chars: int = Field(default=1800, ge=200, le=20_000)

    reference_runner_mode: str = "podman"
    reference_runner_image: str = "localhost/metis/reference-architecture-tool:0.3.0"
    # The inner sandbox stops at 120s; the rest is Podman startup and cleanup.
    reference_runner_timeout_seconds: int = Field(default=150, ge=135, le=600)
    allow_test_backends: bool = False
    # 3000 is the packaged app; 3001 is the UI dev server (see docs on the dev loop).
    cors_origins: list[str] = [
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "http://127.0.0.1:3001",
        "http://localhost:3001",
    ]

    # Cohere's own API (Command A family), opt-in like every cloud provider:
    # absent key means the provider simply is not offered. Distinct from the
    # OCI-hosted Cohere retrieval models below — this is chat/agents, keyed
    # directly against api.cohere.com.
    cohere_api_key: str = ""
    cohere_model: str = "command-a-plus-05-2026"
    cohere_max_output_tokens: int = Field(default=8192, ge=256, le=32768)

    # The Cline gateway: one key reaching both the ClinePass open-weight coding
    # models and the Anthropic/xAI models behind the same endpoint. The split
    # of defaults stays entirely on the subscription. In two production-shaped
    # 16-file manifest runs GLM 5.2 returned neither a typed call nor usable text,
    # while Qwen3.7 Plus completed both plans, so Qwen owns specification,
    # planning and per-file direction and GLM is its bounded fallback. DeepSeek
    # V4 Pro writes. Paid Anthropic/xAI routes remain explicit choices, never a
    # default that fails with HTTP 402 for an otherwise healthy subscription.
    cline_api_key: str = ""
    cline_base_url: str = "https://api.cline.bot/api/v1"
    cline_orchestrator_model: str = "cline-pass/qwen3.7-plus"
    cline_coder_model: str = "cline-pass/deepseek-v4-pro"
    cline_max_output_tokens: int = Field(default=32_768, ge=256, le=200_000)

    # Speech to text, on the same key. Dictation is the one place a cloud call
    # is hard to argue with even in a local-first app: the audio is a few
    # seconds of the user's own voice, not the conversation, and the local
    # alternative would hold a second model resident for the whole session.
    # The 25 MB ceiling is Cohere's, restated here so an oversized clip is
    # refused before it is read into memory rather than after a round trip.
    cohere_transcribe_model: str = "cohere-transcribe-03-2026"
    cohere_transcribe_language: str = "en"
    cohere_transcribe_max_bytes: int = Field(
        default=25 * 1024 * 1024, ge=1024, le=25 * 1024 * 1024
    )

    # ElevenLabs: the second speech provider, and the one voice mode is built
    # on. Dictation may use either — which one is a runtime preference the
    # owner sets in Settings, not an environment variable — while everything
    # *spoken* is ElevenLabs only. Absent key means the provider is simply not
    # offered, exactly the posture the Cohere key takes.
    #
    # The voice id names a public stock voice rather than authenticating
    # anything, so unlike the key it ships as a default.
    elevenlabs_api_key: str = ""
    # The Agents Platform agent voice mode talks through. Created by the owner
    # in the ElevenLabs console, never by this code: it is an external resource
    # with its own billing, and a program that creates one on your behalf is a
    # program that surprises you.
    elevenlabs_agent_id: str = ""
    elevenlabs_stt_model: str = "scribe_v1"
    elevenlabs_tts_model: str = "eleven_flash_v2_5"
    elevenlabs_voice_id: str = "r1KmysJdVYZjJCm4mL3b"
    # A dictation ceiling, not the service's: ElevenLabs accepts clips far
    # larger than this. Stated here so an oversized recording is refused
    # before it is read into memory rather than after a round trip.
    elevenlabs_transcribe_max_bytes: int = Field(
        default=25 * 1024 * 1024, ge=1024, le=100 * 1024 * 1024
    )
    # The private Agents Platform agent behind the /interviews page (Chiron).
    # Deliberately a different agent from voice mode: voice talks through a
    # Custom LLM pointed back at this machine, while the interviewer runs
    # entirely on ElevenLabs' hosted model and never calls home. Created by
    # the owner in the console per docs/interviews-elevenlabs-agent.md, never
    # by this code.
    interview_elevenlabs_agent_id: str = ""

    # Meetings. A recording is minutes or hours of audio, so it gets its own
    # ceilings: the dictation limits are sized for a sentence and would refuse
    # a stand-up.
    meeting_max_bytes: int = Field(
        default=500 * 1024 * 1024, ge=1024, le=2 * 1024 * 1024 * 1024
    )
    meeting_transcribe_timeout_seconds: float = Field(
        default=1_800.0, ge=60.0, le=7_200.0
    )
    # Cleaning the room out of a recording costs a second call over the whole
    # file. Off by default: it improves a noisy transcript and is pure spend on
    # a clean one, and which is which is the owner's call, not a guess.
    meeting_audio_isolation: bool = False

    # Voice mode's reasoning model, held apart from the chat preference: a
    # spoken turn has to come back in seconds, and the model chosen to write a
    # build plan is not that model. This is the startup default; the owner may
    # pick another from the allowlist in speech_preference.py.
    voice_model: str = "deepseek-v4-flash:cloud"

    # Interactive voice. Off is a real state, not a degraded one: dictation and
    # the spoken brief both work without any of this.
    voice_enabled: bool = True
    # The bearer the isolated ingress checks. Empty mints one locally on first
    # use (0600, beside the rest of the local state) rather than defaulting to
    # something guessable — see VoiceSessionService.shared_secret.
    voice_shared_secret: str = ""
    # The model name ElevenLabs is configured to send. Checked by the ingress,
    # never used to route: which model reasons is the stored preference.
    voice_public_model_alias: str = "metis-voice"
    voice_ingress_host: str = "127.0.0.1"
    voice_ingress_port: int = Field(default=8788, ge=1, le=65535)
    # The interpreter the ingress runs under. Its own process, never a thread
    # of this one — a compromise of the internet-facing adapter must not land
    # inside the process holding the database and every credential.
    # Empty means "the interpreter running Metis". Using a bare ``python3``
    # here is unsafe on macOS: Finder and terminal launches can resolve it to a
    # different Python where the ingress package is not installed.
    voice_ingress_command: str = ""
    cloudflared_path: str = "cloudflared"
    voice_tunnel_name: str = "metis-voice"
    voice_tunnel_hostname: str = ""
    # The A.3 go-ahead. This Mac is Oracle-managed, and the brief's appendix
    # asks for one further explicit authorization at the moment a connector is
    # first started here rather than treating the recorded decision as that
    # authorization. Absent, voice refuses to start and says why.
    voice_tunnel_authorized: bool = False
    # How long a browser lease lasts before it must be renewed. Short on
    # purpose: it is the only thing that closes the tunnel after a crashed tab,
    # and a long lease is a long time to leave one open for nobody.
    voice_lease_seconds: int = Field(default=45, ge=15, le=600)
    # A hard ceiling on one conversation, so a forgotten open tab cannot bill
    # all afternoon.
    voice_session_max_seconds: int = Field(default=1_800, ge=60, le=14_400)

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

    # How often the Notion mirror re-syncs on its own. Syncing was manual, so
    # the mirror drifted until someone remembered — and stale pages are worse
    # than missing ones, because retrieval answers confidently from them. The
    # sync re-embeds the consented source as part of its own work. 0 disables.
    notion_refresh_hours: int = Field(default=12, ge=0, le=168)

    # The answer bank: reusable answers harvested from finished runs. Every
    # atom is proposed, never stored silently, and harvesting only considers a
    # run whose answer was actually grounded and cited.
    answer_bank_enabled: bool = True
    answer_bank_min_citations: int = Field(default=1, ge=0, le=10)

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
    # Kill-switch for the Grok reasoning lane (OCI Responses as a Metis
    # provider), independent of allow_oci_responses on purpose: that flag also
    # governs whether *generated apps* may be handed OCI Responses credentials
    # (project_env.py), a different product surface with a different owner.
    # Off, the lane disables cleanly everywhere — preference collapses to
    # local, the web controls grey out, project modes 409 with a named reason
    # — and turning it back on is this one flag, no migration.
    grok_lane_enabled: bool = True
    oci_responses_project_id: str = ""
    oci_responses_base_url: str = (
        "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/openai/v1"
    )
    # Grok 4.3 carries a very large context, and these defaults were set for a
    # smaller one: they bound what Metis is willing to SEND and generate, not
    # what the model can hold, so a conservative value simply wastes the
    # window. Raised to use it — long documents, long threads, whole-file
    # answers — while staying inside the validated ceilings.
    oci_responses_max_output_tokens: int = Field(default=65_536, ge=256, le=131_072)
    oci_recent_history_chars: int = Field(default=240_000, ge=12_000, le=400_000)
    oci_memory_context_chars: int = Field(default=80_000, ge=8_000, le=100_000)

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

    # ClineCore is the sole engine for new project runs. It remains a local
    # child process: only its configured model provider leaves the machine.
    # The old loop survives only inside ControlPlane for frozen legacy
    # checkpoints created before this migration; it is no longer selectable.
    project_coding_engine: Literal["legacy", "clinecore"] = "clinecore"
    cline_sidecar_node_executable: str = "node"
    # None resolves to the pinned workspace package; an explicit override must
    # be absolute so launch never depends on an ambient working directory.
    cline_sidecar_entrypoint: Path | None = None
    cline_sidecar_start_timeout_seconds: float = Field(default=10.0, ge=1.0, le=60.0)
    cline_sidecar_request_timeout_seconds: float = Field(
        default=900.0, ge=10.0, le=3_600.0
    )
    # How many times a rejected slice topology may be handed back to the
    # planner with its exact findings before the turn stops. One correction is
    # cheap (a planner call, no coder session) and fixes the ordinary case: a
    # planner that sliced tests or docs on their own. Zero is fail-fast, which
    # is what a qualification run wants — a corrected plan there would hide
    # the planner defect the run exists to measure.
    project_plan_corrections: int = Field(default=1, ge=0, le=2)
    # Evaluator-only: run the real planner, normalization and topology gate,
    # then stop before any mirror, coding session, or coder inference. It
    # exists to measure planner reliability cheaply -- the expensive half of a
    # build is everything after the plan -- and is refused outside a
    # test-backend configuration so a production run can never be silently
    # turned into a plan that never builds.
    project_plan_only: bool = False
    # The simplified path: one persistent Cline Plan->Act session owns
    # inspection, ordering, editing, checks and repair; Metis stays the
    # contract, security, verification, persistence and approval boundary.
    # The planner/slice path is retained, frozen, behind project_build_path
    # so an in-flight checkpoint and a rollback both keep working.
    project_build_path: Literal["cline_direct", "planner_slices"] = "cline_direct"
    # Slices are opt-in checkpoints for explicitly large work, never inferred
    # from file count and never a reason to refuse ordinary work.
    project_slices_enabled: bool = False
    # How many host-owned checks one coding session may ask for. A check is
    # cheap next to an inference round, but it is not free.
    project_run_check_budget: int = Field(default=12, ge=0, le=60)
    project_run_check_timeout_seconds: float = Field(default=300.0, ge=10.0, le=900.0)
    cline_sidecar_max_iterations: int = Field(default=24, ge=1, le=200)
    # Host-owned convergence bounds around Cline's internal tool loop: one
    # implementation round plus at most three exact verifier-guided repairs,
    # and an honest stop when the normalized finding signature repeats.
    cline_sidecar_max_rounds: int = Field(default=4, ge=1, le=8)
    cline_sidecar_unchanged_findings_limit: int = Field(default=2, ge=1, le=3)
    cline_sidecar_shutdown_timeout_seconds: float = Field(default=5.0, ge=1.0, le=30.0)
    cline_sidecar_max_frame_bytes: int = Field(
        default=1024 * 1024, ge=4_096, le=16 * 1024 * 1024
    )
    # Replay may prepend one synthetic overflow marker. Keep the host queue at
    # least one slot larger than the sidecar journal so a full replay cannot
    # fail closed before the consumer sees its subscribe response.
    cline_sidecar_event_queue_size: int = Field(default=256, ge=3, le=10_000)
    # Artifact cleanup is host-only maintenance: it performs no provider/model
    # call. A full day protects a just-created mirror that crashed before its
    # coding-session ownership row could be committed.
    cline_cleanup_interval_seconds: float = Field(default=60.0, ge=5.0, le=3_600.0)
    cline_orphan_workspace_age_seconds: int = Field(
        default=86_400,
        ge=3_600,
        le=2_592_000,
    )
    # The same conservative shape for event journals. A live measured leak: a
    # model-switch fork's journal outlived its run because the fork identity
    # was only ever a per-round argument, so terminal cleanup deleted the
    # parent's journal and never knew the child's. Ancestry is durable now;
    # this age gate is the backstop for a crash between minting an identity
    # and committing it, and it never touches a referenced journal.
    cline_orphan_journal_age_seconds: int = Field(
        default=86_400,
        ge=3_600,
        le=2_592_000,
    )

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

    # The ranked symbol map sent with every project step. It replaces reads the
    # model would otherwise have to spend steps on, so it earns its budget only
    # if it is measured doing that — see docs/orchestration-plan.md. Set the
    # budget to 0 to send no map at all, which is also the before-side of that
    # measurement.
    project_repo_map_enabled: bool = True
    project_repo_map_max_chars: int = Field(default=12_000, ge=0, le=80_000)
    project_repo_map_max_chars_local: int = Field(default=5_000, ge=0, le=40_000)

    # The orchestrator/coder split. One model (the `planner` role) decides which
    # file is written next and what it must contain; another (the `coder` role)
    # writes exactly that file with reads closed. Off returns the loop to the
    # single-model arc, which is also the control for measuring this.
    # A write step's own output ceiling, separate from chat's. A whole-file
    # rewrite has to EMIT the file: measured, a 28,819-character stylesheet needs
    # roughly 8-10k tokens of output, so the 8,192 shared with chat made the
    # write physically inexpressible — and the model that could not emit it had
    # no way to say so, which reads exactly like a model that would not try.
    # The ClinePass models allow 131k-384k output; this is not the binding
    # constraint it was pretending to be.
    project_write_max_output_tokens: int = Field(default=32_768, ge=1_024, le=200_000)

    project_orchestrator_enabled: bool = True
    # How many times one file may be directed before the host stops asking. A
    # repair is the orchestrator naming the same path again, which is right and
    # necessary — and without a cap it is also an infinite loop.
    project_orchestrator_max_attempts: int = Field(default=3, ge=1, le=8)

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
    project_sandbox_image: str = "localhost/metis/project-verify:0.3.0"
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
    project_sandbox_release_after_idle_seconds: int = Field(
        default=600, ge=0, le=86_400
    )

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

    # Web research: search + page reading for Web scope and public questions in
    # Auto that clearly need current information. Private-context questions
    # stay local in Auto unless the user explicitly asks for the web.
    web_research_enabled: bool = True
    # Optional Brave Search LLM Context. Without WAQIL_BRAVE_SEARCH_API_KEY,
    # the existing keyless DuckDuckGo search remains available.
    brave_search_api_key: str = ""
    brave_search_timeout_seconds: float = Field(default=15.0, ge=2.0, le=30.0)
    web_search_max_results: int = Field(default=4, ge=1, le=8)
    web_fetch_timeout_seconds: float = Field(default=8.0, ge=2.0, le=30.0)
    # Per-page prompt budget. Four pages at this cap stay well inside every
    # provider's context while leaving room for history and memory.
    web_page_max_chars: int = Field(default=3_500, ge=500, le=20_000)
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

    @field_validator("cline_sidecar_node_executable")
    @classmethod
    def valid_sidecar_executable(cls, value: str) -> str:
        if not value.strip() or any(character in value for character in "\r\n\x00"):
            raise ValueError("cline_sidecar_node_executable must be a non-empty line")
        return value

    @model_validator(mode="after")
    def legacy_engine_is_test_only(self) -> Self:
        if self.project_plan_only and not self.allow_test_backends:
            raise ValueError(
                "project_plan_only is an evaluation mode and requires "
                "allow_test_backends"
            )
        if self.project_coding_engine == "legacy" and not self.allow_test_backends:
            raise ValueError(
                "the legacy project coding engine is retired; use clinecore"
            )
        return self

    @field_validator("cline_sidecar_entrypoint")
    @classmethod
    def absolute_sidecar_entrypoint(cls, value: Path | None) -> Path | None:
        if value is not None and not value.is_absolute():
            raise ValueError("cline_sidecar_entrypoint must be absolute")
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
            raise ValueError(
                "OCI Responses must use an HTTPS oci.oraclecloud.com endpoint"
            )
        return value.rstrip("/")

    @field_validator("reference_runner_mode")
    @classmethod
    def valid_runner_mode(cls, value: str) -> str:
        if value not in {"podman", "local", "deterministic"}:
            raise ValueError(
                "reference_runner_mode must be podman, local, or deterministic"
            )
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
                raise ValueError(
                    "asset_roots must be a path or a JSON path array"
                ) from exc
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
    def cline_sidecar_path(self) -> Path:
        return self.cline_sidecar_entrypoint or (
            self.repo_root / "apps" / "cline-sidecar" / "dist" / "src" / "index.js"
        )

    @property
    def cline_sidecar_data_dir(self) -> Path:
        return self.data_dir / "coding-sessions"

    @property
    def cline_event_journal_dir(self) -> Path:
        """Metis's own bounded event journals, written by the sidecar.

        Named here because it is the ONLY directory the journal sweep is
        allowed to touch: the SDK's session store lives beside it under the
        same parent, and a sweep that wandered into that would delete
        transcripts a recovery still needs.
        """
        return self.cline_sidecar_data_dir / "metis-events-v1"

    @property
    def coding_workspace_dir(self) -> Path:
        """Private parent for disposable project mirrors owned by Metis."""
        return self.data_dir / "coding-workspaces"

    @property
    def cline_sidecar_command(self) -> tuple[str, ...]:
        replay_events = min(4_096, self.cline_sidecar_event_queue_size - 1)
        return (
            self.cline_sidecar_node_executable,
            str(self.cline_sidecar_path),
            "--stdio",
            "--runtime",
            "cline",
            "--data-dir",
            str(self.cline_sidecar_data_dir.resolve()),
            "--max-replay-events",
            str(replay_events),
        )

    @property
    def uploads_mirror_dir(self) -> Path:
        # Documents added to the knowledge base from chat land here as one
        # managed corpus source, mirroring how Notion pages are mirrored.
        return self.data_dir / "corpus" / "uploads"

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
            self.repo_root
            / "infra"
            / "sandbox"
            / "project-verify"
            / "run_project_verify.py"
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
    def grok_lane_available(self) -> bool:
        """The one formula for "may Metis call Grok over OCI Responses".

        Written once, here, because it used to exist twice — the preference
        store and the provider each hand-rolled the same conjunction — and a
        new conjunct (the lane kill-switch) landing in one copy but not the
        other would disagree about reality in the worst possible place.
        """
        return bool(
            self.grok_lane_enabled
            and self.allow_oci_responses
            and self.oci_responses_project_id.strip()
        )

    @property
    def model_preference_path(self) -> Path:
        """Which model(s) to route requests to (local, user-owned JSON)."""
        return self.data_dir / "model_preference.json"

    @property
    def model_session_path(self) -> Path:
        """Last explicit local-model session choices (never credentials)."""
        return self.data_dir / "model_session.json"

    @property
    def voice_secret_path(self) -> Path:
        """The locally minted ingress bearer. Never returned by the API."""
        return self.data_dir / "voice-secret"

    @property
    def voice_cache_dir(self) -> Path:
        """Rendered speech, kept so the same words cost one call, not many."""
        return self.data_dir / "voice-audio"

    @property
    def speech_preference_path(self) -> Path:
        """Which speech provider to dictate through (local, user-owned JSON).

        Separate from the model preference on purpose: speech is a different
        decision from reasoning, and the environment only supplies this file's
        startup defaults. Nothing here is a credential.
        """
        return self.data_dir / "speech_preference.json"

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
        self.cline_sidecar_data_dir.mkdir(parents=True, exist_ok=True)
        self.coding_workspace_dir.mkdir(parents=True, exist_ok=True)
        self.tool_bundle_dir.mkdir(parents=True, exist_ok=True)
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.notion_mirror_dir.mkdir(parents=True, exist_ok=True)
