#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${VERARAG_GPU_HOST:-windows-gpu}"
REMOTE_PROJECT="${VERARAG_GPU_PROJECT:-~/projects/VeraRAG}"
SESSIONS="${VERARAG_GPU_SESSIONS:-verarag-conflict-train verarag-conflict-train-matrix verarag-verabench-eval}"
GPU_INTERVAL="${VERARAG_GPU_WATCH_INTERVAL:-1}"
MODE="${1:-status}"

quote_remote() {
  printf "%q" "$1"
}

usage() {
  cat <<'USAGE'
Usage: scripts/windows_gpu_status.sh [status|gpu]

Environment:
  VERARAG_GPU_HOST            Remote SSH host (default: windows-gpu)
  VERARAG_GPU_PROJECT         Remote project path (default: ~/projects/VeraRAG)
  VERARAG_GPU_SESSIONS        Space-separated tmux sessions to inspect
  VERARAG_GPU_WATCH_INTERVAL  nvidia-smi watch interval in seconds (default: 1)

Modes:
  status  Print tmux, GPU, disk, and recent artifact status once.
  gpu     Watch nvidia-smi on the remote host.
USAGE
}

if [ "${MODE}" = "-h" ] || [ "${MODE}" = "--help" ]; then
  usage
  exit 0
fi

if [ "${MODE}" = "gpu" ]; then
  ssh -t "${REMOTE_HOST}" \
    "if [ -x /usr/lib/wsl/lib/nvidia-smi ]; then watch -n $(quote_remote "${GPU_INTERVAL}") /usr/lib/wsl/lib/nvidia-smi; else watch -n $(quote_remote "${GPU_INTERVAL}") nvidia-smi; fi"
  exit 0
fi

if [ "${MODE}" != "status" ]; then
  usage >&2
  exit 2
fi

ssh "${REMOTE_HOST}" \
  "VERARAG_REMOTE_HOST=$(quote_remote "${REMOTE_HOST}") \
   VERARAG_REMOTE_PROJECT=$(quote_remote "${REMOTE_PROJECT}") \
   VERARAG_REMOTE_SESSIONS=$(quote_remote "${SESSIONS}") \
   bash -s" <<'REMOTE'
set -euo pipefail

project="${VERARAG_REMOTE_PROJECT/#\~/$HOME}"
read -r -a sessions <<< "${VERARAG_REMOTE_SESSIONS}"

section() {
  printf '\n== %s ==\n' "$1"
}

section "Host"
printf 'host: %s\n' "${VERARAG_REMOTE_HOST}"
printf 'project: %s\n' "${project}"
printf 'time: %s\n' "$(date -Is)"

section "Project"
if [ -d "${project}" ]; then
  cd "${project}"
  printf 'path: present\n'
  if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    printf 'git: %s\n' "$(git rev-parse --short HEAD)"
  else
    printf 'git: unavailable (the sync helper excludes .git)\n'
  fi
else
  printf 'path: missing\n'
fi

section "tmux"
if command -v tmux >/dev/null 2>&1; then
  if tmux ls 2>/dev/null; then
    :
  else
    printf 'no tmux sessions\n'
  fi
  for session in "${sessions[@]}"; do
    if tmux has-session -t "${session}" 2>/dev/null; then
      printf 'attach: ssh %s "tmux attach -t %s"\n' "${VERARAG_REMOTE_HOST}" "${session}"
    fi
  done
else
  printf 'tmux is not installed or not on PATH\n'
fi

section "GPU"
if [ -x /usr/lib/wsl/lib/nvidia-smi ]; then
  /usr/lib/wsl/lib/nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits \
    || /usr/lib/wsl/lib/nvidia-smi
elif command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits \
    || nvidia-smi
else
  printf 'nvidia-smi not found\n'
fi

section "Disk"
if [ -d "${project}" ]; then
  df -h "${project}" | tail -n 1
else
  df -h "${HOME}" | tail -n 1
fi

section "Recent Artifacts"
if [ -d "${project}/outputs" ]; then
  find "${project}/outputs" -maxdepth 4 -type f \( \
      -name 'train.log' \
      -o -name 'training_metrics.json' \
      -o -name 'training_metadata.json' \
      -o -name '*audit*.json' \
      -o -name 'verabench*.json' \
    \) -printf '%T@ %p\n' \
    | sort -nr \
    | head -n 12 \
    | cut -d' ' -f2- \
    || true
else
  printf 'no outputs directory\n'
fi
REMOTE

echo
echo "Watch GPU: scripts/windows_gpu_status.sh gpu"
