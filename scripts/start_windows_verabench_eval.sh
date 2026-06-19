#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${VERARAG_GPU_HOST:-windows-gpu}"
REMOTE_PROJECT="${VERARAG_GPU_PROJECT:-~/projects/VeraRAG}"
CONDA_ENV="${VERARAG_GPU_CONDA_ENV:-train}"
SESSION="${VERARAG_GPU_TMUX_SESSION:-verarag-verabench-eval}"
CONFIG="${VERARAG_EVAL_CONFIG:-configs/deepseek_run.yaml}"
OUTPUT="${VERARAG_EVAL_OUTPUT:-outputs/remote_results/verabench_v112_full.json}"
TYPES="${VERARAG_EVAL_TYPES:-}"
IDS="${VERARAG_EVAL_IDS:-}"
MAX_QUESTIONS="${VERARAG_EVAL_MAX:-}"
RESTART="${VERARAG_EVAL_RESTART:-1}"
EXTRA_ARGS="${VERARAG_EVAL_EXTRA_ARGS:-}"

quote_remote() {
  printf "%q" "$1"
}

read_deepseek_key() {
  if [ -n "${VERARAG_DEEPSEEK_API_KEY:-}" ]; then
    printf "%s" "${VERARAG_DEEPSEEK_API_KEY}"
    return
  fi
  if [ ! -t 0 ]; then
    echo "VERARAG_DEEPSEEK_API_KEY is required when stdin is not interactive." >&2
    return 2
  fi
  local key
  read -r -s -p "DeepSeek API key: " key
  echo >&2
  if [ -z "${key}" ]; then
    echo "DeepSeek API key cannot be empty." >&2
    return 2
  fi
  printf "%s" "${key}"
}

deepseek_key="$(read_deepseek_key)"
trap 'unset deepseek_key' EXIT

safe_session="${SESSION//[^A-Za-z0-9_.-]/_}"
KEY_FIFO="/tmp/verarag-deepseek-key-${safe_session}-$$"

ssh "${REMOTE_HOST}" \
  "VERARAG_REMOTE_PROJECT=$(quote_remote "${REMOTE_PROJECT}") \
   VERARAG_REMOTE_CONDA_ENV=$(quote_remote "${CONDA_ENV}") \
   VERARAG_REMOTE_SESSION=$(quote_remote "${SESSION}") \
   VERARAG_REMOTE_CONFIG=$(quote_remote "${CONFIG}") \
   VERARAG_REMOTE_OUTPUT=$(quote_remote "${OUTPUT}") \
   VERARAG_REMOTE_TYPES=$(quote_remote "${TYPES}") \
   VERARAG_REMOTE_IDS=$(quote_remote "${IDS}") \
   VERARAG_REMOTE_MAX=$(quote_remote "${MAX_QUESTIONS}") \
   VERARAG_REMOTE_RESTART=$(quote_remote "${RESTART}") \
   VERARAG_REMOTE_EXTRA_ARGS=$(quote_remote "${EXTRA_ARGS}") \
   VERARAG_REMOTE_KEY_FIFO=$(quote_remote "${KEY_FIFO}") \
   bash -s" <<'REMOTE'
set -euo pipefail

project="${VERARAG_REMOTE_PROJECT/#\~/$HOME}"
key_fifo="${VERARAG_REMOTE_KEY_FIFO}"

if [ -n "${VERARAG_REMOTE_TYPES}" ] && [ -n "${VERARAG_REMOTE_IDS}" ]; then
  echo "Set either VERARAG_EVAL_TYPES or VERARAG_EVAL_IDS, not both." >&2
  exit 2
fi

if tmux has-session -t "${VERARAG_REMOTE_SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${VERARAG_REMOTE_SESSION}" >&2
  echo "Attach with: tmux attach -t ${VERARAG_REMOTE_SESSION}" >&2
  exit 2
fi

rm -f "${key_fifo}"
mkfifo "${key_fifo}"
chmod 600 "${key_fifo}"

driver="$(mktemp /tmp/verarag-verabench-eval.XXXXXX.sh)"
{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  printf 'VERARAG_REMOTE_PROJECT=%q\n' "${VERARAG_REMOTE_PROJECT}"
  printf 'VERARAG_REMOTE_CONDA_ENV=%q\n' "${VERARAG_REMOTE_CONDA_ENV}"
  printf 'VERARAG_REMOTE_CONFIG=%q\n' "${VERARAG_REMOTE_CONFIG}"
  printf 'VERARAG_REMOTE_OUTPUT=%q\n' "${VERARAG_REMOTE_OUTPUT}"
  printf 'VERARAG_REMOTE_TYPES=%q\n' "${VERARAG_REMOTE_TYPES}"
  printf 'VERARAG_REMOTE_IDS=%q\n' "${VERARAG_REMOTE_IDS}"
  printf 'VERARAG_REMOTE_MAX=%q\n' "${VERARAG_REMOTE_MAX}"
  printf 'VERARAG_REMOTE_RESTART=%q\n' "${VERARAG_REMOTE_RESTART}"
  printf 'VERARAG_REMOTE_EXTRA_ARGS=%q\n' "${VERARAG_REMOTE_EXTRA_ARGS}"
  printf 'VERARAG_REMOTE_KEY_FIFO=%q\n' "${VERARAG_REMOTE_KEY_FIFO}"
  cat <<'DRIVER'

key_fifo="${VERARAG_REMOTE_KEY_FIFO}"
if ! DEEPSEEK_API_KEY="$(
  timeout 30s bash -c 'IFS= read -r key < "$1"; printf "%s" "${key}"' \
    bash "${key_fifo}"
)"; then
  rm -f "${key_fifo}"
  echo "Failed to read DeepSeek API key from secure FIFO." >&2
  exit 2
fi
rm -f "${key_fifo}"
export DEEPSEEK_API_KEY

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${VERARAG_REMOTE_CONDA_ENV}"

project="${VERARAG_REMOTE_PROJECT/#\~/$HOME}"
cd "${project}"

mkdir -p "$(dirname "${VERARAG_REMOTE_OUTPUT}")"
log_path="${VERARAG_REMOTE_OUTPUT%.json}.log"

args=(
  python experiments/run_verabench.py
  --config "${VERARAG_REMOTE_CONFIG}"
  --output "${VERARAG_REMOTE_OUTPUT}"
)

if [ "${VERARAG_REMOTE_RESTART}" = "1" ] || [ "${VERARAG_REMOTE_RESTART}" = "true" ]; then
  args+=(--restart)
fi

if [ -n "${VERARAG_REMOTE_TYPES}" ]; then
  read -r -a types <<< "${VERARAG_REMOTE_TYPES}"
  args+=(--types "${types[@]}")
fi

if [ -n "${VERARAG_REMOTE_IDS}" ]; then
  read -r -a ids <<< "${VERARAG_REMOTE_IDS}"
  args+=(--ids "${ids[@]}")
fi

if [ -n "${VERARAG_REMOTE_MAX}" ]; then
  args+=(--max "${VERARAG_REMOTE_MAX}")
fi

if [ -n "${VERARAG_REMOTE_EXTRA_ARGS}" ]; then
  read -r -a extra_args <<< "${VERARAG_REMOTE_EXTRA_ARGS}"
  args+=("${extra_args[@]}")
fi

echo "Starting VeraBench evaluation."
echo "Config: ${VERARAG_REMOTE_CONFIG}"
echo "Output: ${VERARAG_REMOTE_OUTPUT}"
echo "Log: ${log_path}"
"${args[@]}" 2>&1 | tee "${log_path}"
DRIVER
} > "${driver}"

chmod +x "${driver}"
tmux new -d -s "${VERARAG_REMOTE_SESSION}" "bash '${driver}'"

tmux ls
REMOTE

printf "%s\n" "${deepseek_key}" | ssh "${REMOTE_HOST}" "timeout 30s cat > $(quote_remote "${KEY_FIFO}")"
unset deepseek_key

echo "Started tmux session ${SESSION} on ${REMOTE_HOST}"
echo "Attach with: ssh ${REMOTE_HOST} 'tmux attach -t ${SESSION}'"
echo "Output: ${OUTPUT}"
