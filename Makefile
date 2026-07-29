SHELL := /bin/zsh

.PHONY: setup dev api web start run stop refresh install-agent uninstall-agent test verify-lock build sandbox-image acceptance \
	verify-ollama verify-podman verify-live offline-bundle \
	verify-restart \
	offline-bundle-prerequisites offline-verify offline-verify-release \
	offline-smoke-install offline-extract export-data verify-export clean-data dac-catalog

EXPORT ?= metis-export.zip
ACCEPTANCE_URL ?= http://127.0.0.1:8000
UV_VERSION ?= 0.11.29
WAQIL_UV_CACHE ?= $(CURDIR)/.uv-cache
SANDBOX_IMAGE ?= localhost/metis/reference-architecture-tool:0.3.0
OFFLINE_BUNDLE ?= metis-offline-bundle.zip
OFFLINE_EXTRACT ?= metis-offline-bundle

setup:
	python3.13 -m venv .venv
	.venv/bin/python -m pip install "uv==$(UV_VERSION)"
	UV_CACHE_DIR="$(WAQIL_UV_CACHE)" UV_PROJECT_ENVIRONMENT="$(CURDIR)/.venv" .venv/bin/uv sync --project apps/api --frozen --extra dev --extra cloud --inexact --no-python-downloads
	pnpm install --frozen-lockfile

dev:
	@echo "Run 'make api' and 'make web' in separate terminals."

api:
	.venv/bin/waqil-api

web:
	pnpm --dir apps/web dev

start: build
	.venv/bin/python scripts/run_local.py

# Same as `start`, but frees stale ports first, skips the build when nothing
# changed, and opens the browser. This is what Metis.app runs.
run:
	./scripts/metis

stop:
	@lsof -ti -i tcp:3000 -i tcp:8000 2>/dev/null | xargs -r kill || true

# Rebuild and restart the background service after code changes.
refresh:
	@launchctl kickstart -k gui/$$UID/com.metis.local 2>/dev/null \
		&& echo "Metis restarting with your latest changes." \
		|| $(MAKE) run

install-agent:
	./scripts/install-agent

uninstall-agent:
	./scripts/install-agent --remove

test:
	$(MAKE) verify-lock
	.venv/bin/pytest apps/api/tests tests
	.venv/bin/pytest skills/reference-architecture-generator/tests
	pnpm --dir apps/web test

verify-lock:
	UV_CACHE_DIR="$(WAQIL_UV_CACHE)" UV_PROJECT_ENVIRONMENT="$(CURDIR)/.venv" .venv/bin/uv sync --project apps/api --frozen --extra dev --extra cloud --inexact --no-python-downloads --check

build:
	pnpm --dir apps/web build

sandbox-image:
	./infra/sandbox/build_reference_architecture_image.sh

# Refreshes the vendored OCI sizing catalog from Oracle's docs and Hugging Face.
# The only networked step in the project; the app itself never calls out. Run it
# when Oracle publishes newly validated models, then commit the JSON it writes.
dac-catalog:
	.venv/bin/python scripts/build_dac_catalog.py

acceptance:
	.venv/bin/python scripts/acceptance_smoke.py --base-url "$(ACCEPTANCE_URL)" --readme README.md

verify-ollama:
	PYTHONPATH="$(CURDIR)/apps/api/src" .venv/bin/python scripts/ollama_smoke.py

verify-podman:
	PYTHONPATH="$(CURDIR)/apps/api/src" .venv/bin/python scripts/podman_smoke.py --image "$(SANDBOX_IMAGE)"

verify-restart:
	PYTHONPATH="$(CURDIR)/apps/api/src" .venv/bin/python scripts/restart_smoke.py --image "$(SANDBOX_IMAGE)"

verify-live: verify-ollama verify-podman verify-restart

offline-bundle:
	python3.13 scripts/offline_bundle.py create \
		--project-root "$(CURDIR)" \
		--output "$(OFFLINE_BUNDLE)" \
		--uv-bin "$(CURDIR)/.venv/bin/uv" \
		--pnpm-bin pnpm \
		--image "$(SANDBOX_IMAGE)"

offline-bundle-prerequisites:
	python3.13 scripts/offline_bundle.py create \
		--project-root "$(CURDIR)" \
		--output "$(OFFLINE_BUNDLE)" \
		--uv-bin "$(CURDIR)/.venv/bin/uv" \
		--pnpm-bin pnpm \
		--image "$(SANDBOX_IMAGE)" \
		--without-image

offline-verify:
	python3.13 scripts/offline_bundle.py verify "$(OFFLINE_BUNDLE)" --project-root "$(CURDIR)"

offline-verify-release:
	python3.13 scripts/offline_bundle.py verify "$(OFFLINE_BUNDLE)" --project-root "$(CURDIR)" --require-image

offline-smoke-install:
	python3.13 scripts/offline_bundle.py verify "$(OFFLINE_BUNDLE)" --project-root "$(CURDIR)" --require-image --smoke-install

offline-extract:
	python3.13 scripts/offline_bundle.py extract "$(OFFLINE_BUNDLE)" \
		--project-root "$(CURDIR)" \
		--destination "$(OFFLINE_EXTRACT)"

export-data:
	.venv/bin/python scripts/export_data.py create --data-dir .data --output "$(EXPORT)"

verify-export:
	.venv/bin/python scripts/export_data.py verify "$(EXPORT)"

clean-data:
	@echo "Refusing to delete local state automatically. Remove ./.data manually after backing it up."
