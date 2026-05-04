"""Run VeraRAG on HotpotQA dataset."""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.pipeline.verarag import VeraRAG, create_verarag
from src.configs import get_dataset_config
from src.evaluation.answer_metrics import AnswerMetrics
from src.evaluation.evidence_metrics import EvidenceMetrics


def load_hotpotqa(data_path: str, split: str = "dev") -> List[Dict[str, Any]]:
    """
    Load HotpotQA dataset.

    Args:
        data_path: Path to HotpotQA data file
        split: Dataset split (dev, train, test)

    Returns:
        List of HotpotQA samples
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def prepare_documents(hotpotqa_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Prepare documents for indexing from HotpotQA data.

    Args:
        hotpotqa_data: HotpotQA samples

    Returns:
        List of documents for indexing
    """
    documents = []

    for sample in hotpotqa_data:
        context = sample.get("context", [])

        for title, sentences in context:
            doc_text = " ".join(sentences)
            documents.append({
                "id": f"doc_{sample.get('_id', '')}_{title}",
                "title": title,
                "text": doc_text,
                "source": "wikipedia"
            })

    return documents


def evaluate_hotpotqa(
    verarag: VeraRAG,
    hotpotqa_data: List[Dict[str, Any]],
    num_samples: int = 100,
    output_path: str = None
) -> Dict[str, Any]:
    """
    Evaluate VeraRAG on HotpotQA dataset.

    Args:
        verarag: VeraRAG instance
        hotpotqa_data: HotpotQA samples
        num_samples: Number of samples to evaluate
        output_path: Path to save detailed results

    Returns:
        Evaluation results
    """
    results = {
        "predictions": [],
        "metrics": {}
    }

    # Limit samples if specified
    samples = hotpotqa_data[:num_samples]

    print(f"Evaluating on {len(samples)} samples...")

    for i, sample in enumerate(samples):
        question = sample.get("question", "")
        answer = sample.get("answer", "")

        print(f"\n[{i+1}/{len(samples)}] Question: {question}")

        try:
            output = verarag.query(question)

            # Extract predicted answer
            predicted_answer = output.answer

            # Get supporting facts for evidence evaluation
            supporting_facts = sample.get("supporting_facts", [])
            gold_fact_ids = [f"doc_{sample.get('_id', '')}_{fact[0]}" for fact in supporting_facts]

            # Get retrieved evidence IDs
            retrieved_ids = [ev.evidence_id for ev in output.evidence]

            # Calculate metrics
            em = AnswerMetrics.exact_match(predicted_answer, answer)
            f1 = AnswerMetrics.f1_score(predicted_answer, answer)
            evidence_precision = EvidenceMetrics.evidence_precision(retrieved_ids, gold_fact_ids)
            evidence_recall = EvidenceMetrics.evidence_recall(retrieved_ids, gold_fact_ids)

            result = {
                "id": sample.get("_id", ""),
                "question": question,
                "gold_answer": answer,
                "predicted_answer": predicted_answer,
                "exact_match": em,
                "f1": f1,
                "evidence_precision": evidence_precision,
                "evidence_recall": evidence_recall,
                "confidence": output.confidence,
                "num_evidence": len(output.evidence),
                "num_conflicts": output.metadata.get("num_conflicts", 0)
            }

            results["predictions"].append(result)

            print(f"  EM: {em:.2f}, F1: {f1:.2f}, Ev. Prec: {evidence_precision:.2f}, Ev. Rec: {evidence_recall:.2f}")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Calculate aggregate metrics
    predictions = results["predictions"]

    if predictions:
        avg_em = sum(p["exact_match"] for p in predictions) / len(predictions)
        avg_f1 = sum(p["f1"] for p in predictions) / len(predictions)
        avg_ev_prec = sum(p["evidence_precision"] for p in predictions) / len(predictions)
        avg_ev_rec = sum(p["evidence_recall"] for p in predictions) / len(predictions)
        avg_confidence = sum(p["confidence"] for p in predictions) / len(predictions)

        results["metrics"] = {
            "num_samples": len(predictions),
            "exact_match": avg_em,
            "f1_score": avg_f1,
            "evidence_precision": avg_ev_prec,
            "evidence_recall": avg_ev_rec,
            "evidence_f1": 2 * avg_ev_prec * avg_ev_rec / (avg_ev_prec + avg_ev_rec) if (avg_ev_prec + avg_ev_rec) > 0 else 0,
            "avg_confidence": avg_confidence
        }

        print("\n=== Aggregate Results ===")
        for key, value in results["metrics"].items():
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    # Save results
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run VeraRAG on HotpotQA")
    parser.add_argument("--data-path", type=str, required=True, help="Path to HotpotQA data file")
    parser.add_argument("--config", type=str, default="hotpotqa", help="Config name (hotpotqa, fever, etc.)")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--output", type=str, default="results/hotpotqa_results.json", help="Output path")
    parser.add_argument("--index-only", action="store_true", help="Only build index, don't evaluate")

    args = parser.parse_args()

    # Load config
    config = get_dataset_config(args.config)

    # Create VeraRAG instance
    print("Creating VeraRAG instance...")
    verarag = VeraRAG(config)

    # Load data
    print(f"Loading data from {args.data_path}...")
    hotpotqa_data = load_hotpotqa(args.data_path)

    # Build index
    print("Preparing documents for indexing...")
    documents = prepare_documents(hotpotqa_data)
    print(f"Indexing {len(documents)} documents...")
    verarag.index_documents(documents)

    if args.index_only:
        print("Index built. Exiting.")
        return

    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Evaluate
    evaluate_hotpotqa(
        verarag,
        hotpotqa_data,
        num_samples=args.num_samples,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
