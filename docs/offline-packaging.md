# Offline environment bundles

Metis can create a platform-specific environment bundle while the packaging
machine is online, verify it without contacting a registry, and use it to install
the exact locked backend and frontend dependencies on an offline machine.

The bundle accompanies an existing Metis source checkout. It does not contain
the repository, user data, Ollama itself, or Ollama model blobs.

## What is included

`make offline-bundle` creates a ZIP containing:

- `apps/api/uv.lock`, `apps/api/pyproject.toml`, the pnpm lockfile, workspace
  file, and package manifests used during creation;
- a freshly populated `uv` cache that was proven with an offline frozen sync;
- the exact `uv` executable used to populate that cache;
- a freshly populated pnpm store that was proven with an offline frozen install;
- an OCI archive of the locally reviewed sandbox image, resolved and recorded by
  repository digest;
- the base-image reference, sandbox build inputs, host/platform versions, and a
  SHA-256/size/mode record for every member.

Archive entry order, timestamps, and modes are normalized. Package-manager cache
metadata and OCI archives can nevertheless differ byte-for-byte across tool
versions, operating systems, or container stores. Reproducibility here means
that the exact lockfiles and cache contents are recorded, integrity checked, and
offline-install tested, not that independently created archives are universally
bit-identical.

## Create on the connected machine

First establish the frozen development environment and build the reviewed
sandbox image:

```bash
make setup
make sandbox-image
make offline-bundle OFFLINE_BUNDLE=/path/to/metis-offline-bundle.zip
make offline-verify-release OFFLINE_BUNDLE=/path/to/metis-offline-bundle.zip
make offline-smoke-install OFFLINE_BUNDLE=/path/to/metis-offline-bundle.zip
```

Creation uses network access only to populate fresh package caches. It refuses
to overwrite an existing output, rejects non-regular or escaping links, preserves
uv's verified internal relative cache links, and publishes the ZIP only after
both caches work in offline mode. Machine-local pnpm checkout backlinks are
removed because they are not dependency content and would make the store
non-portable.

If the final Podman image cannot be present on the packaging machine, create a
dependency-only bundle instead:

```bash
make offline-bundle-prerequisites OFFLINE_BUNDLE=/path/to/metis-dependencies.zip
make offline-verify OFFLINE_BUNDLE=/path/to/metis-dependencies.zip
```

That variant records the exact Containerfile, base-image digest, policy, build
script, runtime lock, and skill manifest, but it is not a release-complete
offline package. `make offline-verify-release` intentionally rejects it. The
sandbox image must be built on a connected machine or supplied as a separately
verified OCI archive before Metis can execute generated code.

## Verify and extract on the offline machine

Copy the ZIP and the exact source checkout to the target machine, then run:

```bash
make offline-verify-release OFFLINE_BUNDLE=/media/metis-offline-bundle.zip
make offline-extract \
  OFFLINE_BUNDLE=/media/metis-offline-bundle.zip \
  OFFLINE_EXTRACT=/opt/metis-offline
```

Verification checks every archived byte and also checks that the checkout's six
dependency metadata files match those used to build the bundle. For transport
verification without a checkout comparison, use:

```bash
python3.13 scripts/offline_bundle.py verify \
  /media/metis-offline-bundle.zip --no-project-check --require-image
```

Extraction verifies first, rejects traversal, duplicates, and links whose targets
escape or are absent from the archive, refuses to overwrite its destination, and
restores the recorded internal links and executable mode for bundled `uv`.

## Install with networking disabled

The target must already provide the same Python 3.13 patch release, OS/CPU,
Node.js release, and pnpm release recorded in `bundle-manifest.json`. Podman must
be installed and configured rootless. The verifier's `--smoke-install` option
enforces these host versions and performs disposable offline Python and frontend
installs without changing the checkout:

```bash
python3.13 scripts/offline_bundle.py verify \
  /media/metis-offline-bundle.zip \
  --project-root . \
  --require-image \
  --smoke-install
```

To install into the checkout, after extraction:

```bash
python3.13 -m venv .venv
UV_CACHE_DIR=/opt/metis-offline/python/uv-cache \
UV_PROJECT_ENVIRONMENT="$PWD/.venv" \
/opt/metis-offline/tooling/uv sync \
  --offline --project apps/api --frozen --extra dev --inexact --no-python-downloads

pnpm install \
  --offline --frozen-lockfile \
  --store-dir /opt/metis-offline/frontend/pnpm-store

podman load --input /opt/metis-offline/podman/reference-architecture.oci.tar
```

Compare `podman image inspect --format '{{.Digest}}'` for the loaded image with
the `sandbox.resolved_image` digest in `bundle-manifest.json` before starting
Metis. The application resolves the configured tag to that immutable digest at
execution time.

## Deliberate exclusions

- Ollama model blobs are not bundled. Provision the configured Qwen and North
  models separately and verify them with `make verify-ollama`.
- The Podman machine/VM is not bundled.
- Application data is not bundled. Use the separate `make export-data` workflow.
- The package is platform specific; it is not a cross-platform wheel/npm mirror.
- A prerequisite-only bundle cannot render diagrams until the separately built
  sandbox image is available locally.
