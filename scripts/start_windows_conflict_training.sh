#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${VERARAG_GPU_HOST:-windows-gpu}"
REMOTE_PROJECT="${VERARAG_GPU_PROJECT:-~/projects/VeraRAG}"
CONDA_ENV="${VERARAG_GPU_CONDA_ENV:-train}"
SESSION="${VERARAG_GPU_TMUX_SESSION:-verarag-conflict-train}"
BASE_MODEL="${VERARAG_GPU_BASE_MODEL:-~/models/verarag/cross-encoder_nli-distilroberta-base}"
EPOCHS="${VERARAG_GPU_EPOCHS:-2}"
BATCH_SIZE="${VERARAG_GPU_BATCH_SIZE:-16}"
WARMUP_STEPS="${VERARAG_GPU_WARMUP_STEPS:-10}"
THRESHOLD_OBJECTIVE="${VERARAG_GPU_THRESHOLD_OBJECTIVE:-f1}"
MIN_THRESHOLD_PRECISION="${VERARAG_GPU_MIN_THRESHOLD_PRECISION:-0.0}"
MIN_THRESHOLD_RECALL="${VERARAG_GPU_MIN_THRESHOLD_RECALL:-0.0}"
SEED="${VERARAG_GPU_SEED:-13}"
OFFLINE="${VERARAG_GPU_OFFLINE:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

quote_remote() {
  printf "%q" "$1"
}

if [ "${VERARAG_GPU_SKIP_PREFLIGHT:-0}" != "1" ]; then
  "${SCRIPT_DIR}/check_windows_conflict_training_ready.sh"
fi

ssh "${REMOTE_HOST}" \
  "VERARAG_REMOTE_PROJECT=$(quote_remote "${REMOTE_PROJECT}") \
   VERARAG_REMOTE_CONDA_ENV=$(quote_remote "${CONDA_ENV}") \
   VERARAG_REMOTE_SESSION=$(quote_remote "${SESSION}") \
   VERARAG_REMOTE_BASE_MODEL=$(quote_remote "${BASE_MODEL}") \
   VERARAG_REMOTE_EPOCHS=$(quote_remote "${EPOCHS}") \
   VERARAG_REMOTE_BATCH_SIZE=$(quote_remote "${BATCH_SIZE}") \
   VERARAG_REMOTE_WARMUP_STEPS=$(quote_remote "${WARMUP_STEPS}") \
   VERARAG_REMOTE_THRESHOLD_OBJECTIVE=$(quote_remote "${THRESHOLD_OBJECTIVE}") \
   VERARAG_REMOTE_MIN_THRESHOLD_PRECISION=$(quote_remote "${MIN_THRESHOLD_PRECISION}") \
   VERARAG_REMOTE_MIN_THRESHOLD_RECALL=$(quote_remote "${MIN_THRESHOLD_RECALL}") \
   VERARAG_REMOTE_SEED=$(quote_remote "${SEED}") \
   VERARAG_REMOTE_OFFLINE=$(quote_remote "${OFFLINE}") \
   bash -s" <<'REMOTE'
set -euo pipefail

if tmux has-session -t "${VERARAG_REMOTE_SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${VERARAG_REMOTE_SESSION}" >&2
  echo "Attach with: tmux attach -t ${VERARAG_REMOTE_SESSION}" >&2
  exit 2
fi

driver="$(mktemp /tmp/verarag-conflict-train.XXXXXX.sh)"
{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  printf 'VERARAG_REMOTE_PROJECT=%q\n' "${VERARAG_REMOTE_PROJECT}"
  printf 'VERARAG_REMOTE_CONDA_ENV=%q\n' "${VERARAG_REMOTE_CONDA_ENV}"
  printf 'VERARAG_REMOTE_BASE_MODEL=%q\n' "${VERARAG_REMOTE_BASE_MODEL}"
  printf 'VERARAG_REMOTE_EPOCHS=%q\n' "${VERARAG_REMOTE_EPOCHS}"
  printf 'VERARAG_REMOTE_BATCH_SIZE=%q\n' "${VERARAG_REMOTE_BATCH_SIZE}"
  printf 'VERARAG_REMOTE_WARMUP_STEPS=%q\n' "${VERARAG_REMOTE_WARMUP_STEPS}"
  printf 'VERARAG_REMOTE_THRESHOLD_OBJECTIVE=%q\n' "${VERARAG_REMOTE_THRESHOLD_OBJECTIVE}"
  printf 'VERARAG_REMOTE_MIN_THRESHOLD_PRECISION=%q\n' "${VERARAG_REMOTE_MIN_THRESHOLD_PRECISION}"
  printf 'VERARAG_REMOTE_MIN_THRESHOLD_RECALL=%q\n' "${VERARAG_REMOTE_MIN_THRESHOLD_RECALL}"
  printf 'VERARAG_REMOTE_SEED=%q\n' "${VERARAG_REMOTE_SEED}"
  printf 'VERARAG_REMOTE_OFFLINE=%q\n' "${VERARAG_REMOTE_OFFLINE}"
  cat <<'DRIVER'

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${VERARAG_REMOTE_CONDA_ENV}"

project="${VERARAG_REMOTE_PROJECT/#\~/$HOME}"
base_model="${VERARAG_REMOTE_BASE_MODEL/#\~/$HOME}"
cd "${project}"

export TRANSFORMERS_OFFLINE="${VERARAG_REMOTE_OFFLINE}"
export HF_HUB_OFFLINE="${VERARAG_REMOTE_OFFLINE}"

mkdir -p outputs
python experiments/build_conflict_training_data.py --output-dir outputs/conflict_pairs
python experiments/train_conflict_cross_encoder.py \
  --train outputs/conflict_pairs/train.jsonl \
  --val outputs/conflict_pairs/val.jsonl \
  --test outputs/conflict_pairs/test.jsonl \
  --output-dir outputs/conflict_cross_encoder \
  --model "${base_model}" \
  --device cuda \
  --epochs "${VERARAG_REMOTE_EPOCHS}" \
  --batch-size "${VERARAG_REMOTE_BATCH_SIZE}" \
  --warmup-steps "${VERARAG_REMOTE_WARMUP_STEPS}" \
  --threshold-objective "${VERARAG_REMOTE_THRESHOLD_OBJECTIVE}" \
  --min-threshold-precision "${VERARAG_REMOTE_MIN_THRESHOLD_PRECISION}" \
  --min-threshold-recall "${VERARAG_REMOTE_MIN_THRESHOLD_RECALL}" \
  --seed "${VERARAG_REMOTE_SEED}"
DRIVER
} > "${driver}"

chmod +x "${driver}"
tmux new -d -s "${VERARAG_REMOTE_SESSION}" "bash '${driver}'"
  tmux ls
REMOTE

echo "Started tmux session ${SESSION} on ${REMOTE_HOST}"
echo "Attach with: ssh ${REMOTE_HOST} 'tmux attach -t ${SESSION}'"
