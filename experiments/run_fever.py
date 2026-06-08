"""Run VeraRAG on FEVER dataset."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from configs import get_dataset_config  # noqa: E402
from src.pipeline.verarag import VeraRAG  # noqa: E402


def load_fever(data_path: str) -> list[dict[str, Any]]:
    """
    Load FEVER dataset.

    Args:
        data_path: Path to FEVER data file

    Returns:
        List of FEVER samples
    """
    with open(data_path, encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    return data


def evaluate_fever(
    verarag: VeraRAG,
    fever_data: list[dict[str, Any]],
    num_samples: int = 100,
    output_path: str | None = None
) -> dict[str, Any]:
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
        print("  Class Accuracies:")
        for label, acc in class_accuracies.items():
            print(f"    {label}: {acc:.4f}")

    # Save results
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_path}")

    return results


def generate_demo_data() -> list[dict[str, Any]]:
    """Generate synthetic FEVER-like data for demo mode."""
    return [
        {
            "id": 1,
            "claim": "The Eiffel Tower is located in Paris, France.",
            "label": "SUPPORTS",
            "evidence": [
                ["Eiffel Tower", 0, "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France."]
            ]
        },
        {
            "id": 2,
            "claim": "The Great Wall of China is visible from the Moon with the naked eye.",
            "label": "REFUTES",
            "evidence": [
                ["Great Wall of China", 0, "The Great Wall of China is a series of fortifications in northern China."],
                ["Great Wall visibility", 0, "The claim that the Great Wall is visible from the Moon has been debunked by multiple sources."]
            ]
        },
        {
            "id": 3,
            "claim": "Albert Einstein failed his math class in school.",
            "label": "REFUTES",
            "evidence": [
                ["Albert Einstein", 0, "Albert Einstein was a German-born theoretical physicist."],
                ["Einstein education", 0, "Einstein excelled in math and physics from a young age, mastering differential and integral calculus by age 15."]
            ]
        },
        {
            "id": 4,
            "claim": "The planet Mars has two moons named Phobos and Deimos.",
            "label": "SUPPORTS",
            "evidence": [
                ["Moons of Mars", 0, "The two moons of Mars are Phobos and Deimos."]
            ]
        },
        {
            "id": 5,
            "claim": "The human body has exactly 206 bones at birth.",
            "label": "REFUTES",
            "evidence": [
                ["Human skeleton", 0, "A typical adult human skeleton consists of 206 bones."],
                ["Baby bones", 0, "Babies are born with approximately 300 bones, many of which fuse together as they grow."]
            ]
        }
    ]


def run_demo(output_path: str | None = None):
    """Run demo evaluation with synthetic data (no API needed)."""
    print("=" * 60)
    print("  FEVER Demo Mode (Synthetic Data)")
    print("=" * 60)

    demo_data = generate_demo_data()
    print(f"\nGenerated {len(demo_data)} demo samples")

    results = {
        "predictions": [],
        "metrics": {},
        "demo_mode": True
    }

    label_map = {
        "SUPPORTS": "supported",
        "REFUTES": "refuted",
        "NOT ENOUGH INFO": "not_enough_info"
    }

    # Simulated predictions (mostly correct with one error for realism)
    simulated_predictions = {
        1: "SUPPORTS",
        2: "REFUTES",
        3: "SUPPORTS",  # Simulated error: model incorrectly says SUPPORTS
        4: "SUPPORTS",
        5: "REFUTES"
    }

    for i, sample in enumerate(demo_data):
        claim = sample.get("claim", "")
        gold_label = sample.get("label", "")

        print(f"\n[{i+1}/{len(demo_data)}] Claim: {claim}")

        pred_label = simulated_predictions.get(sample["id"], "NOT ENOUGH INFO")
        correct = 1 if pred_label == gold_label else 0

        # Simulated verification status
        pred_status = label_map.get(pred_label, "not_enough_info")

        result = {
            "id": sample.get("id", ""),
            "claim": claim,
            "gold_label": gold_label,
            "predicted_label": pred_label,
            "correct": correct,
            "confidence": 0.82 if correct else 0.65,
            "verification_status": pred_status
        }

        results["predictions"].append(result)

        print(f"  Gold: {gold_label}, Pred: {pred_label}, Correct: {correct}")

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
        print("  Class Accuracies:")
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
    parser.add_argument("--demo", action="store_true", help="Run demo with synthetic data (no API needed)")
    parser.add_argument("--data-path", type=str, default=None, help="Path to FEVER data file")
    parser.add_argument("--config", type=str, default="fever", help="Config name")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--output", type=str, default="results/external/fever_results.json", help="Output path")

    args = parser.parse_args()

    # Demo mode: use synthetic data, no API calls
    if args.demo:
        return run_demo(output_path=args.output)

    # Normal mode: requires data path
    if not args.data_path:
        parser.error("--data-path is required when not using --demo mode")

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
