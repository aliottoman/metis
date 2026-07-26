# Metis API

The local FastAPI/LangGraph control plane for Metis. It binds to loopback,
persists domain state and graph checkpoints separately, and streams durable run
events over SSE.

From this directory, after installing the locked project dependencies into a
Python 3.13 environment:

```bash
waqil-api
```

Useful environment variables:

- `WAQIL_DATA_DIR`: application data directory (default: `./.data`)
- `WAQIL_REPO_ROOT`: repository root used to find `skills/`
- `WAQIL_MODEL_BACKEND`: `ollama`, `deterministic`, or `auto`
- `WAQIL_OLLAMA_BASE_URL`: loopback Ollama URL
- `WAQIL_ALLOW_OCI_RESPONSES`: explicitly enables the optional Grok Responses provider
- `WAQIL_OCI_RESPONSES_PROJECT_ID`: OCI Enterprise AI project OCID used by Responses
- `WAQIL_OCI_RESPONSES_BASE_URL`: regional OpenAI-compatible `/openai/v1` base URL
- `WAQIL_TOOL_TRUSTED_AUTO_ACTIVATION`: automatically activates an approved,
  passing build only when its registered profile is network-free, run-IO-only,
  and R2-or-lower at execution (default: `true`)
- `WAQIL_REFERENCE_RUNNER_MODE`: `podman`, `local`, or `deterministic`
- `WAQIL_REFERENCE_RUNNER_IMAGE`: prebuilt Podman image name (default:
  `localhost/metis/reference-architecture-tool:0.3.0`)
- `WAQIL_ASSET_ROOTS`: one asset-container path, an OS-path-separated list, or
  a JSON array of paths. Only immediate, non-hidden child directories appear in
  the Metis asset library after an explicit `POST /api/v1/assets/scan`.

`deterministic` model and runner modes are intended only for tests and smoke
checks. Production-like local runs should use Ollama and the rootless Podman
runner.

OCI Responses calls are stateless from OCI's perspective (`store=false`): Metis
continues to own conversation history and approved memory locally. Provider and
native-tool choices are pinned into each run. The cloud provider can plan,
synthesize, and author candidate tool code, while the `ModelBroker` used inside
executing tools is always constructed with the local Ollama provider.

Tool review endpoints expose the hash-verified manifest, evaluation report,
reviewable source files, and source diff for immutable versions. Corrective
feedback creates a pending improvement proposal. Approving one without naming
an exact passing revision creates a queued revision request and leaves the
active version unchanged; only an explicitly selected, verified, evaluated
version can move the registry pointer. Improvement decisions use idempotency
keys and are retained in an append-only audit table.

An explicit human tool-build request is itself authorization for a host-hardened
definition inside the trusted local boundary. Metis defines, evaluates,
activates, and serves that request in one run. Planner-inferred definitions and
broader execution risk fall back to explicit review.

## Asset launch manifests

Discovery is metadata-only and never turns README instructions into executable
commands. To configure an asset for launch, add `.metis/asset.json` inside it
with an explicit argv array:

```json
{
  "name": "Customer dashboard",
  "summary": "Explore customer health and service metrics.",
  "category": "Analytics",
  "tags": ["dashboard", "oci"],
  "env": ["DAC_OCID"],
  "launch": {
    "command": ["python", "-m", "streamlit", "run", "app.py", "--server.port", "{port}"]
  }
}
```

`{port}` and `{host}` become an allocated loopback port and `127.0.0.1`.
`{python}` and `{uv}` resolve to the Python and uv executables bundled with the
running Metis API, so recipes do not depend on the shell's active environment.
Environment keys must be declared in the manifest or detected from bounded
README, `.env.example`, and shallow source metadata; process-control keys such
as `PATH`, `HOST`, and `PORT` are reserved. Values supplied at launch are never
included in catalog/start responses and are redacted from captured logs.

The exact current manifest recipe must then be approved through
`POST /api/v1/assets/{id}/approval`. Approval is stored by fingerprint, so any
change to the command, launch path, or allowed environment names requires a new
review. `DELETE` on the same endpoint revokes it. Approved assets currently run
as trusted host subprocesses with the macOS user's filesystem and network
authority; this is distinct from the generated-tool Podman sandbox.
