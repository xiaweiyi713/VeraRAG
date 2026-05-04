#!/bin/bash
# Build retrieval index for a dataset

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default values
DATASET="hotpotqa"
DATA_PATH=""
OUTPUT_DIR="data/indexes"

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
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$DATA_PATH" ]; then
    echo "Error: --data-path is required"
    echo "Usage: $0 --dataset <name> --data-path <path> [--output-dir <dir>]"
    exit 1
fi

echo "Building index for dataset: $DATASET"
echo "Data path: $DATA_PATH"
echo "Output directory: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build index using Python
cd "$PROJECT_DIR"

python - <<EOF
import sys
sys.path.insert(0, "src")

from src.retriever.hybrid import HybridRetriever
from src.configs import get_dataset_config
import json

# Load dataset-specific config
config = get_dataset_config("$DATASET")

# Load data
print(f"Loading data from $DATA_PATH...")
with open("$DATA_PATH", "r") as f:
    data = json.load(f)

# Prepare documents
print(f"Preparing {len(data)} documents for indexing...")
documents = []

if "$DATASET" == "hotpotqa":
    for sample in data:
        context = sample.get("context", [])
        for title, sentences in context:
            documents.append({
                "id": f"doc_{sample.get('_id', '')}_{title}",
                "title": title,
                "text": " ".join(sentences),
                "source": "wikipedia"
            })
elif "$DATASET" == "fever":
    for sample in data:
        # FEVER uses pages as context
        for page in sample.get("predicted_pages", []):
            documents.append({
                "id": page,
                "title": page,
                "text": sample.get("claim", ""),
                "source": "wikipedia"
            })
else:
    # Generic document preparation
    for i, doc in enumerate(data):
        documents.append({
            "id": doc.get("id", f"doc_{i}"),
            "title": doc.get("title", ""),
            "text": doc.get("text", ""),
            "source": doc.get("source", "unknown")
        })

print(f"Prepared {len(documents)} documents")

# Build retriever and index
print("Building retriever index...")
retriever = HybridRetriever(config=config.get("retriever", {}))
retriever.index_documents(documents)

# Save index
index_path = "$OUTPUT_DIR/${DATASET}_index"
print(f"Saving index to {index_path}...")
retriever.save_index(index_path)

print("Index built successfully!")
EOF

echo "Index building complete!"
