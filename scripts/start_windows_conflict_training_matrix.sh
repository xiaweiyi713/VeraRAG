#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${VERARAG_GPU_HOST:-windows-gpu}"
REMOTE_PROJECT="${VERARAG_GPU_PROJECT:-~/projects/VeraRAG}"
CONDA_ENV="${VERARAG_GPU_CONDA_ENV:-train}"
SESSION="${VERARAG_GPU_TMUX_SESSION:-verarag-conflict-train-matrix}"
BASE_MODEL="${VERARAG_GPU_BASE_MODEL:-~/models/verarag/cross-encoder_nli-distilroberta-base}"
EPOCHS="${VERARAG_GPU_EPOCHS:-3}"
BATCH_SIZE="${VERARAG_GPU_BATCH_SIZE:-16}"
WARMUP_STEPS="${VERARAG_GPU_WARMUP_STEPS:-10}"
SEEDS="${VERARAG_GPU_SEEDS:-13 17 23}"
OFFLINE="${VERARAG_GPU_OFFLINE:-1}"
DATASET_DIR="${VERARAG_GPU_DATASET_DIR:-outputs/conflict_pairs_v112_leakfree}"
OUTPUT_PREFIX="${VERARAG_GPU_OUTPUT_PREFIX:-outputs/conflict_cross_encoder_v112_leakfree}"
ABLATION_OUTPUT="${VERARAG_GPU_ABLATION_OUTPUT:-outputs/conflict_detector_v112_leakfree_matrix_test.json}"
AUDIT_OUTPUT="${VERARAG_GPU_AUDIT_OUTPUT:-outputs/conflict_model_promotion_audit_matrix.json}"
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
   VERARAG_REMOTE_SEEDS=$(quote_remote "${SEEDS}") \
   VERARAG_REMOTE_OFFLINE=$(quote_remote "${OFFLINE}") \
   VERARAG_REMOTE_DATASET_DIR=$(quote_remote "${DATASET_DIR}") \
   VERARAG_REMOTE_OUTPUT_PREFIX=$(quote_remote "${OUTPUT_PREFIX}") \
   VERARAG_REMOTE_ABLATION_OUTPUT=$(quote_remote "${ABLATION_OUTPUT}") \
   VERARAG_REMOTE_AUDIT_OUTPUT=$(quote_remote "${AUDIT_OUTPUT}") \
   bash -s" <<'REMOTE'
set -euo pipefail

project="${VERARAG_REMOTE_PROJECT/#\~/$HOME}"
base_model="${VERARAG_REMOTE_BASE_MODEL/#\~/$HOME}"
read -r -a seeds <<< "${VERARAG_REMOTE_SEEDS}"
if [ "${#seeds[@]}" -lt 1 ]; then
  echo "VERARAG_GPU_SEEDS must contain at least one seed" >&2
  exit 2
fi

if tmux has-session -t "${VERARAG_REMOTE_SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${VERARAG_REMOTE_SESSION}" >&2
  echo "Attach with: tmux attach -t ${VERARAG_REMOTE_SESSION}" >&2
  exit 2
fi

driver="$(mktemp /tmp/verarag-conflict-matrix.XXXXXX.sh)"
{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  printf 'VERARAG_REMOTE_PROJECT=%q\n' "${VERARAG_REMOTE_PROJECT}"
  printf 'VERARAG_REMOTE_CONDA_ENV=%q\n' "${VERARAG_REMOTE_CONDA_ENV}"
  printf 'VERARAG_REMOTE_BASE_MODEL=%q\n' "${VERARAG_REMOTE_BASE_MODEL}"
  printf 'VERARAG_REMOTE_EPOCHS=%q\n' "${VERARAG_REMOTE_EPOCHS}"
  printf 'VERARAG_REMOTE_BATCH_SIZE=%q\n' "${VERARAG_REMOTE_BATCH_SIZE}"
  printf 'VERARAG_REMOTE_WARMUP_STEPS=%q\n' "${VERARAG_REMOTE_WARMUP_STEPS}"
  printf 'VERARAG_REMOTE_SEEDS=%q\n' "${VERARAG_REMOTE_SEEDS}"
  printf 'VERARAG_REMOTE_OFFLINE=%q\n' "${VERARAG_REMOTE_OFFLINE}"
  printf 'VERARAG_REMOTE_DATASET_DIR=%q\n' "${VERARAG_REMOTE_DATASET_DIR}"
  printf 'VERARAG_REMOTE_OUTPUT_PREFIX=%q\n' "${VERARAG_REMOTE_OUTPUT_PREFIX}"
  printf 'VERARAG_REMOTE_ABLATION_OUTPUT=%q\n' "${VERARAG_REMOTE_ABLATION_OUTPUT}"
  printf 'VERARAG_REMOTE_AUDIT_OUTPUT=%q\n' "${VERARAG_REMOTE_AUDIT_OUTPUT}"
  cat <<'DRIVER'

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${VERARAG_REMOTE_CONDA_ENV}"

project="${VERARAG_REMOTE_PROJECT/#\~/$HOME}"
base_model="${VERARAG_REMOTE_BASE_MODEL/#\~/$HOME}"
read -r -a seeds <<< "${VERARAG_REMOTE_SEEDS}"
cd "${project}"

export TRANSFORMERS_OFFLINE="${VERARAG_REMOTE_OFFLINE}"
export HF_HUB_OFFLINE="${VERARAG_REMOTE_OFFLINE}"

mkdir -p "${VERARAG_REMOTE_DATASET_DIR}"
python experiments/build_conflict_training_data.py \
  --output-dir "${VERARAG_REMOTE_DATASET_DIR}" \
  | tee "${VERARAG_REMOTE_DATASET_DIR}/build_summary.json"

run_dirs=()
for seed in "${seeds[@]}"; do
  output_dir="${VERARAG_REMOTE_OUTPUT_PREFIX}_seed${seed}"
  mkdir -p "${output_dir}"
  run_dirs+=("${output_dir}")
  python experiments/train_conflict_cross_encoder.py \
    --train "${VERARAG_REMOTE_DATASET_DIR}/train.jsonl" \
    --val "${VERARAG_REMOTE_DATASET_DIR}/val.jsonl" \
    --test "${VERARAG_REMOTE_DATASET_DIR}/test.jsonl" \
    --output-dir "${output_dir}" \
    --model "${base_model}" \
    --device cuda \
    --epochs "${VERARAG_REMOTE_EPOCHS}" \
    --batch-size "${VERARAG_REMOTE_BATCH_SIZE}" \
    --warmup-steps "${VERARAG_REMOTE_WARMUP_STEPS}" \
    --seed "${seed}" \
    2>&1 | tee "${output_dir}/train.log"
done

primary_seed="${seeds[0]}"
primary_dir="${VERARAG_REMOTE_OUTPUT_PREFIX}_seed${primary_seed}"
primary_threshold="$(
  python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_threshold"])' \
    "${primary_dir}/training_metrics.json"
)"

python experiments/compare_conflict_detectors.py \
  --split test \
  --learned-model-path "${primary_dir}" \
  --learned-threshold "${primary_threshold}" \
  --output "${VERARAG_REMOTE_ABLATION_OUTPUT}"

python experiments/audit_conflict_model.py \
  --runs "${run_dirs[@]}" \
  --ablation "${VERARAG_REMOTE_ABLATION_OUTPUT}" \
  --allow-internal-heldout \
  --report-only \
  --output "${VERARAG_REMOTE_AUDIT_OUTPUT}"

echo "Conflict training matrix finished."
echo "A/B report: ${VERARAG_REMOTE_ABLATION_OUTPUT}"
echo "Promotion audit: ${VERARAG_REMOTE_AUDIT_OUTPUT}"
DRIVER
} > "${driver}"

chmod +x "${driver}"
tmux new -d -s "${VERARAG_REMOTE_SESSION}" "bash '${driver}'"
tmux ls
REMOTE

echo "Started tmux session ${SESSION} on ${REMOTE_HOST}"
echo "Attach with: ssh ${REMOTE_HOST} 'tmux attach -t ${SESSION}'"
