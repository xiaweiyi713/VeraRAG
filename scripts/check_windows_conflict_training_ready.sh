#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${VERARAG_GPU_HOST:-windows-gpu}"
REMOTE_PROJECT="${VERARAG_GPU_PROJECT:-~/projects/VeraRAG}"
CONDA_ENV="${VERARAG_GPU_CONDA_ENV:-train}"
BASE_MODEL="${VERARAG_GPU_BASE_MODEL:-~/models/verarag/cross-encoder_nli-distilroberta-base}"
OFFLINE="${VERARAG_GPU_OFFLINE:-1}"
SSH_CONNECT_TIMEOUT="${VERARAG_GPU_SSH_CONNECT_TIMEOUT:-10}"

quote_remote() {
  printf "%q" "$1"
}

if ! ssh \
  -o "ConnectTimeout=${SSH_CONNECT_TIMEOUT}" \
  -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=2 \
  "${REMOTE_HOST}" \
  "VERARAG_REMOTE_PROJECT=$(quote_remote "${REMOTE_PROJECT}") \
   VERARAG_REMOTE_CONDA_ENV=$(quote_remote "${CONDA_ENV}") \
   VERARAG_REMOTE_BASE_MODEL=$(quote_remote "${BASE_MODEL}") \
   VERARAG_REMOTE_OFFLINE=$(quote_remote "${OFFLINE}") \
   bash -s" <<'REMOTE'
set -euo pipefail

project="${VERARAG_REMOTE_PROJECT/#\~/$HOME}"
base_model="${VERARAG_REMOTE_BASE_MODEL/#\~/$HOME}"
failures=0

check() {
  local label="$1"
  shift
  if "$@" >/dev/null 2>&1; then
    printf '[ok] %s\n' "${label}"
  else
    printf '[fail] %s\n' "${label}" >&2
    failures=$((failures + 1))
  fi
}

check "project directory exists: ${project}" test -d "${project}"
check "tmux is installed" command -v tmux
check "conda profile exists" test -f "${HOME}/miniconda3/etc/profile.d/conda.sh"

if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  if conda env list | awk '{print $1}' | grep -Fxq "${VERARAG_REMOTE_CONDA_ENV}"; then
    printf '[ok] conda env exists: %s\n' "${VERARAG_REMOTE_CONDA_ENV}"
    conda activate "${VERARAG_REMOTE_CONDA_ENV}"
    if [ -d "${project}" ]; then
      cd "${project}"
      if python - <<'PY'
import importlib.util
import sys

missing = [
    name
    for name in ("torch", "sentence_transformers")
    if importlib.util.find_spec(name) is None
]
if missing:
    print("missing optional train dependencies: " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(1)
PY
      then
        printf '[ok] train dependencies importable\n'
      else
        failures=$((failures + 1))
      fi
      if python - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
      then
        printf '[ok] torch can see CUDA\n'
      else
        printf '[fail] torch cannot see CUDA\n' >&2
        failures=$((failures + 1))
      fi
    fi
  else
    printf '[fail] conda env missing: %s\n' "${VERARAG_REMOTE_CONDA_ENV}" >&2
    failures=$((failures + 1))
  fi
fi

if [ "${VERARAG_REMOTE_OFFLINE}" = "1" ]; then
  check "offline base model exists: ${base_model}" test -e "${base_model}"
fi

if [ "${failures}" -gt 0 ]; then
  printf '\nRemote conflict-training preflight failed with %s issue(s).\n' "${failures}" >&2
  printf 'Start Windows, log in, wait for Tailscale/WSL keepalive, sync the repo, and ensure the train env/model are ready.\n' >&2
  exit 2
fi

printf '\nRemote conflict-training preflight passed.\n'
REMOTE
then
  cat >&2 <<EOF

Remote conflict-training preflight could not complete for host: ${REMOTE_HOST}

Check that Windows is powered on, logged in, Tailscale is online, the WSL
keepalive task has started, and the repository has been synced with:

  scripts/sync_windows_gpu.sh

You can bypass this preflight for manual repair only with:

  VERARAG_GPU_SKIP_PREFLIGHT=1 scripts/start_windows_conflict_training.sh
EOF
  exit 2
fi
