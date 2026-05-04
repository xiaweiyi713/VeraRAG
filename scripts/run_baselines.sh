#!/bin/bash
# Run baseline experiments

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default values
DATASET="hotpotqa"
DATA_PATH=""
OUTPUT_DIR="results/baselines"

# Baselines to run
BASELINES=("vanilla_rag" "hybrid_rag" "self_rag")

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --baselines)
            IFS=',' read -ra BASELINES <<< "$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$DATA_PATH" ]; then
    echo "Error: --data-path is required"
    exit 1
fi

echo "Running baselines for dataset: $DATASET"
echo "Baselines: ${BASELINES[@]}"
echo "Output directory: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

# Run each baseline
for baseline in "${BASELINES[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running baseline: $baseline"
    echo "=========================================="

    case $baseline in
        vanilla_rag)
            python -m experiments.baselines.vanilla_rag \
                --dataset "$DATASET" \
                --data-path "$DATA_PATH" \
                --output "$OUTPUT_DIR/${baseline}_${DATASET}.json"
            ;;
        hybrid_rag)
            python -m experiments.baselines.hybrid_rag \
                --dataset "$DATASET" \
                --data-path "$DATA_PATH" \
                --output "$OUTPUT_DIR/${baseline}_${DATASET}.json"
            ;;
        self_rag)
            python -m experiments.baselines.self_rag \
                --dataset "$DATASET" \
                --data-path "$DATA_PATH" \
                --output "$OUTPUT_DIR/${baseline}_${DATASET}.json"
            ;;
        *)
            echo "Unknown baseline: $baseline"
            ;;
    esac
done

echo ""
echo "All baselines complete!"
echo "Results saved to: $OUTPUT_DIR"
