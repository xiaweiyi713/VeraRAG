#!/bin/bash
# Evaluate VeraRAG results

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default values
RESULTS_PATH=""
DATASET="hotpotqa"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --results-path)
            RESULTS_PATH="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$RESULTS_PATH" ]; then
    echo "Error: --results-path is required"
    exit 1
fi

echo "Evaluating results from: $RESULTS_PATH"
echo "Dataset: $DATASET"

cd "$PROJECT_DIR"

python - <<EOF
import sys
import json
sys.path.insert(0, "src")

from src.evaluation.answer_metrics import AnswerMetrics
from src.evidence.evidence_metrics import EvidenceMetrics

# Load results
with open("$RESULTS_PATH", "r") as f:
    results = json.load(f)

predictions = results.get("predictions", [])

if not predictions:
    print("No predictions found in results")
    sys.exit(1)

# Calculate metrics
em_scores = [p.get("exact_match", 0) for p in predictions]
f1_scores = [p.get("f1", 0) for p in predictions]

avg_em = sum(em_scores) / len(em_scores)
avg_f1 = sum(f1_scores) / len(f1_scores)

print("\n=== Evaluation Results ===")
print(f"Num samples: {len(predictions)}")
print(f"Exact Match: {avg_em:.4f}")
print(f"F1 Score: {avg_f1:.4f}")

# Evidence metrics (if available)
if any("evidence_precision" in p for p in predictions):
    ev_prec = [p.get("evidence_precision", 0) for p in predictions if "evidence_precision" in p]
    ev_rec = [p.get("evidence_recall", 0) for p in predictions if "evidence_recall" in p]

    if ev_prec:
        avg_ep = sum(ev_prec) / len(ev_prec)
        avg_er = sum(ev_rec) / len(ev_rec)
        avg_ef1 = 2 * avg_ep * avg_er / (avg_ep + avg_er) if (avg_ep + avg_er) > 0 else 0

        print(f"Evidence Precision: {avg_ep:.4f}")
        print(f"Evidence Recall: {avg_er:.4f}")
        print(f"Evidence F1: {avg_ef1:.4f}")

# Print summary
print("\n=== Summary ===")
print(f"Results evaluated successfully!")
EOF
