#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${VERARAG_GPU_HOST:-windows-gpu}"
REMOTE_PROJECT="${VERARAG_GPU_PROJECT:-~/projects/VeraRAG}"

quote_remote() {
  printf "%q" "$1"
}

REMOTE_PROJECT_QUOTED="$(quote_remote "${REMOTE_PROJECT}")"
REMOTE_PROJECT_RESOLVED="$(
  ssh "${REMOTE_HOST}" \
    "VERARAG_REMOTE_PROJECT=${REMOTE_PROJECT_QUOTED} bash -s" <<'REMOTE'
set -euo pipefail

project="${VERARAG_REMOTE_PROJECT/#\~/$HOME}"
mkdir -p "${project}"
printf "%s" "${project}"
REMOTE
)"
REMOTE_PROJECT_RSYNC="$(quote_remote "${REMOTE_PROJECT_RESOLVED}")"

rsync -az --delete \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  --exclude '.mypy_cache' \
  --exclude 'dist' \
  --exclude 'build' \
  --exclude '*.egg-info' \
  --exclude 'outputs' \
  ./ "${REMOTE_HOST}:${REMOTE_PROJECT_RSYNC}/"

echo "Synced VeraRAG to ${REMOTE_HOST}:${REMOTE_PROJECT_RESOLVED}"
