"""Run VeraRAG on FEVER dataset."""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.pipeline.verarag import VeraRAG
from src.configs import get_dataset_config
from src.utils.data_structures import VerificationStatus


def load_fever(data_path: str) -> List[Dict[str, Any]]:
    """
    Load FEVER dataset.

    Args:
        data_path: Path to FEVER data file

    Returns:
        List of FEVER samples
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    return data


def evaluate_fever(
    verarag: VeraRAG,
    fever_data: List[Dict[str, Any]],
    num_samples: int = 100,
    output_path: str = None
) -> Dict[str, Any]:
    """
    Evaluate VeraRAG on FEVER dataset.

    Args:
        verarag: VeraRAG instance
        fever_data: FEVER samples
        num_samples: Number of samples to evaluate
        output_path: Path to save detailed results

    Returns:
        Evaluation results
    """
    results = {
        "predictions": [],
        "metrics": {}
    }

    samples = fever_data[:num_samples]

    print(f"Evaluating on {len(samples)} samples...")

    label_map = {
        "SUPPORTS": "supported",
        "REFUTES": "refuted",
        "NOT ENOUGH INFO": "not_enough_info"
    }

    for i, sample in enumerate(samples):
        claim = sample.get("claim", "")
        gold_label = sample.get("label", "")

        print(f"\n[{i+1}/{len(samples)}] Claim: {claim[:100]}...")

        try:
            # Use verification-focused query
            query = f"Is this claim true or false: {claim}"
            output = verarag.query(query)

            # Map verification status to label
            pred_status = output.verification_report.overall_status.value if output.verification_report else "not_enough_info"

            # Convert to FEVER label
            pred_label = None
            for fever_label, status in label_map.items():
                if status == pred_status:
                    pred_label = fever_label
                    break

            if pred_label is None:
                # Use confidence to decide
                if output.confidence > 0.7:
                    pred_label = "SUPPORTS"
                elif output.confidence < 0.3:
                    pred_label = "REFUTES"
                else:
                    pred_label = "NOT ENOUGH INFO"

            correct = 1 if pred_label == gold_label else 0

            result = {
                "id": sample.get("id", ""),
                "claim": claim,
                "gold_label": gold_label,
                "predicted_label": pred_label,
                "correct": correct,
                "confidence": output.confidence,
                "verification_status": pred_status
            }

            results["predictions"].append(result)

            print(f"  Gold: {gold_label}, Pred: {pred_label}, Correct: {correct}")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Calculate aggregate metrics
    predictions = results["predictions"]

    if predictions:
        num_correct = sum(p["correct"] for p in predictions)
        accuracy = num_correct / len(predictions)

        # Calculate per-class metrics
        label_counts = {}
        label_correct = {}

        for p in predictions:
            label = p["gold_label"]
            label_counts[label] = label_counts.get(label, 0) + 1
            if p["correct"]:
                label_correct[label] = label_correct.get(label, 0) + 1

        class_accuracies = {}
        for label in label_counts:
            class_accuracies[label] = label_correct.get(label, 0) / label_counts[label]

        results["metrics"] = {
            "num_samples": len(predictions),
            "accuracy": accuracy,
            "class_accuracies": class_accuracies,
            "label_distribution": label_counts
        }

        print("\n=== Aggregate Results ===")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Class Accuracies:")
        for label, acc in class_accuracies.items():
            print(f"    {label}: {acc:.4f}")

    # Save results
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run VeraRAG on FEVER")
    parser.add_argument("--data-path", type=str, required=True, help="Path to FEVER data file")
    parser.add_argument("--config", type=str, default="fever", help="Config name")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--output", type=str, default="results/fever_results.json", help="Output path")

    args = parser.parse_args()

    # Load config
    config = get_dataset_config(args.config)

    # Create VeraRAG instance
    print("Creating VeraRAG instance...")
    verarag = VeraRAG(config)

    # Load data
    print(f"Loading data from {args.data_path}...")
    fever_data = load_fever(args.data_path)

    # Evaluate
    evaluate_fever(
        verarag,
        fever_data,
        num_samples=args.num_samples,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
