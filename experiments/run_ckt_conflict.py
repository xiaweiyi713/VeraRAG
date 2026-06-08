"""Run VeraRAG on CKT-Conflict dataset."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs import get_dataset_config
from src.evaluation.answer_metrics import AnswerMetrics
from src.evaluation.calibration_metrics import CalibrationMetrics
from src.evaluation.conflict_metrics import ConflictMetrics
from src.pipeline.verarag import VeraRAG


def load_ckt_conflict(data_path: str) -> list[dict[str, Any]]:
    """
    Load CKT-Conflict dataset.

    Args:
        data_path: Path to CKT-Conflict data file

    Returns:
        List of CKT-Conflict samples
    """
    with open(data_path, encoding='utf-8') as f:
        data = json.load(f)

    return data.get("samples", data)


def evaluate_ckt_conflict(
    verarag: VeraRAG,
    ckt_data: list[dict[str, Any]],
    num_samples: int = 100,
    output_path: str | None = None
) -> dict[str, Any]:
    """
    Evaluate VeraRAG on CKT-Conflict dataset.

    Args:
        verarag: VeraRAG instance
        ckt_data: CKT-Conflict samples
        num_samples: Number of samples to evaluate
        output_path: Path to save detailed results

    Returns:
        Evaluation results
    """
    results = {
        "predictions": [],
        "metrics": {}
    }

    samples = ckt_data[:num_samples]

    print(f"Evaluating on {len(samples)} samples...")

    for i, sample in enumerate(samples):
        question = sample.get("question", "")
        answer = sample.get("answer", "")

        print(f"\n[{i+1}/{len(samples)}] Question: {question[:80]}...")

        try:
            output = verarag.query(question)

            # Answer metrics
            em = AnswerMetrics.exact_match(output.answer, answer)
            f1 = AnswerMetrics.f1_score(output.answer, answer)

            # Conflict detection metrics
            gold_conflicts = sample.get("conflicting_evidence", [])
            predicted_conflicts = []

            if output.conflict_report.get("conflicts"):
                for conflict in output.conflict_report["conflicts"]:
                    predicted_conflicts.append((conflict.get("source_id", ""), conflict.get("target_id", "")))

            gold_conflict_pairs = [
                (c.get("source", ""), c.get("target", ""))
                for c in gold_conflicts
            ]

            conflict_f1 = ConflictMetrics.conflict_detection_f1(predicted_conflicts, gold_conflict_pairs)

            # Uncertainty calibration
            is_correct = em == 1.0

            result = {
                "id": sample.get("id", ""),
                "question": question,
                "gold_answer": answer,
                "predicted_answer": output.answer,
                "exact_match": em,
                "f1": f1,
                "conflict_f1": conflict_f1,
                "confidence": output.confidence,
                "uncertainty": output.uncertainty.to_dict(),
                "num_evidence": len(output.evidence),
                "num_conflicts": output.metadata.get("num_conflicts", 0),
                "correct": is_correct
            }

            results["predictions"].append(result)

            print(f"  EM: {em}, F1: {f1:.2f}, Conflict F1: {conflict_f1:.2f}, Conf: {output.confidence:.2f}")

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Calculate aggregate metrics
    predictions = results["predictions"]

    if predictions:
        # Basic metrics
        avg_em = sum(p["exact_match"] for p in predictions) / len(predictions)
        avg_f1 = sum(p["f1"] for p in predictions) / len(predictions)
        avg_conflict_f1 = sum(p["conflict_f1"] for p in predictions) / len(predictions)
        avg_confidence = sum(p["confidence"] for p in predictions) / len(predictions)

        # Calibration metrics
        confidences = [p["confidence"] for p in predictions]
        correct = [p["correct"] for p in predictions]

        ece = CalibrationMetrics.expected_calibration_error(confidences, correct)
        brier = CalibrationMetrics.brier_score(confidences, correct)

        # Unsupported claim rate (hallucination)
        from src.evaluation.hallucination_metrics import HallucinationMetrics

        unsupported_rates = []
        for p in predictions:
            answer_claims = HallucinationMetrics.extract_claims_from_text(p["predicted_answer"])
            evidence_texts = [ev.text_span for ev in sample.get("supporting_evidence", [])]
            supported_claims = set()

            for claim in answer_claims:
                for ev_text in evidence_texts:
                    if claim.lower() in ev_text.lower():
                        supported_claims.add(claim)
                        break

            rate = HallucinationMetrics.unsupported_claim_rate(answer_claims, supported_claims)
            unsupported_rates.append(rate)

        avg_unsupported = sum(unsupported_rates) / len(unsupported_rates) if unsupported_rates else 0

        results["metrics"] = {
            "num_samples": len(predictions),
            "exact_match": avg_em,
            "f1_score": avg_f1,
            "conflict_detection_f1": avg_conflict_f1,
            "avg_confidence": avg_confidence,
            "expected_calibration_error": ece,
            "brier_score": brier,
            "unsupported_claim_rate": avg_unsupported
        }

        # Calculate primary score (weighted combination)
        primary_score = (
            avg_f1 * 0.25 +
            avg_conflict_f1 * 0.30 +
            (1 - ece) * 0.20 +
            (1 - avg_unsupported) * 0.15 +
            avg_confidence * 0.10
        )

        results["metrics"]["primary_score"] = primary_score

        print("\n=== Aggregate Results ===")
        for key, value in results["metrics"].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        print(f"\nPrimary Score: {primary_score:.4f}")
        print(f"Pass Threshold (0.60): {'PASS' if primary_score >= 0.60 else 'FAIL'}")

    # Save results
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run VeraRAG on CKT-Conflict")
    parser.add_argument("--data-path", type=str, required=True, help="Path to CKT-Conflict data file")
    parser.add_argument("--config", type=str, default="ckt_conflict", help="Config name")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--output", type=str, default="results/ckt_conflict_results.json", help="Output path")

    args = parser.parse_args()

    # Load config
    config = get_dataset_config(args.config)

    # Create VeraRAG instance
    print("Creating VeraRAG instance...")
    verarag = VeraRAG(config)

    # Load data
    print(f"Loading data from {args.data_path}...")
    ckt_data = load_ckt_conflict(args.data_path)

    # Evaluate
    evaluate_ckt_conflict(
        verarag,
        ckt_data,
        num_samples=args.num_samples,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
