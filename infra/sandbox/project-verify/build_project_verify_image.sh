#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/../../.." && pwd)
image=${METIS_VERIFY_IMAGE:-localhost/metis/project-verify:0.2.0}
platform=${METIS_VERIFY_PLATFORM:-linux/arm64}

podman build \
  --file "$script_dir/Containerfile" \
  --ignorefile "$script_dir/containerignore" \
  --platform "$platform" \
  --timestamp 0 \
  --tag "$image" \
  "$repo_root"

podman image inspect --format 'id={{.Id}} digest={{.Digest}} platform={{.Os}}/{{.Architecture}}' "$image"
