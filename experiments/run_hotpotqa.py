"""Run VeraRAG on HotpotQA dataset."""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.pipeline.verarag import VeraRAG, create_verarag
from configs import get_dataset_config
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


def generate_demo_data() -> List[Dict[str, Any]]:
    """Generate synthetic HotpotQA-like data for demo mode."""
    return [
        {
            "_id": "demo_001",
            "question": "What government position was held by the woman who portrayed media mogul Logan Roy in the HBO series Succession?",
            "answer": "Prime Minister",
            "context": [
                ["Succession (TV series)", [
                    "Succession is an American satirical comedy-drama television series.",
                    "The character Logan Roy is a media mogul.",
                    "The actress who portrayed Logan Roy in the show is a well-known figure.",
                    "The series aired on HBO from 2018 to 2023."
                ]],
                ["Brian Cox (actor)", [
                    "Brian Cox portrayed Logan Roy in Succession.",
                    "Brian Cox is a Scottish actor.",
                    "He has won multiple awards for his performance."
                ]]
            ],
            "supporting_facts": [
                ["Succession (TV series)", 0],
                ["Brian Cox (actor)", 0]
            ]
        },
        {
            "_id": "demo_002",
            "question": "Were both the inventor of the telephone and the inventor of the light bulb born in the same country?",
            "answer": "no",
            "context": [
                ["Alexander Graham Bell", [
                    "Alexander Graham Bell was a Scottish-born inventor.",
                    "He is credited with inventing the telephone.",
                    "Bell was born in Edinburgh, Scotland on March 3, 1847."
                ]],
                ["Thomas Edison", [
                    "Thomas Edison was an American inventor.",
                    "He is known for inventing the light bulb.",
                    "Edison was born in Milan, Ohio, United States."
                ]]
            ],
            "supporting_facts": [
                ["Alexander Graham Bell", 2],
                ["Thomas Edison", 2]
            ]
        },
        {
            "_id": "demo_003",
            "question": "Which has more members, the band that released 'Abbey Road' or the band that released 'Rumours'?",
            "answer": "equal",
            "context": [
                ["The Beatles", [
                    "The Beatles were an English rock band formed in Liverpool in 1960.",
                    "The band had four members: John Lennon, Paul McCartney, George Harrison, and Ringo Starr.",
                    "They released the album Abbey Road in 1969."
                ]],
                ["Fleetwood Mac", [
                    "Fleetwood Mac are a British-American rock band formed in London in 1967.",
                    "The band had four core members during the Rumours era.",
                    "They released the album Rumours in 1977."
                ]]
            ],
            "supporting_facts": [
                ["The Beatles", 1],
                ["Fleetwood Mac", 1]
            ]
        }
    ]


def run_demo(output_path: str = None):
    """Run demo evaluation with synthetic data (no API needed)."""
    from src.utils.data_structures import (
        Evidence, AnswerClaim, VeraRAGOutput,
        VerificationReport, VerificationStatus, UncertaintyBreakdown,
        ReasoningStep
    )

    print("=" * 60)
    print("  HotpotQA Demo Mode (Synthetic Data)")
    print("=" * 60)

    demo_data = generate_demo_data()
    print(f"\nGenerated {len(demo_data)} demo samples")

    results = {
        "predictions": [],
        "metrics": {},
        "demo_mode": True
    }

    for i, sample in enumerate(demo_data):
        question = sample.get("question", "")
        gold_answer = sample.get("answer", "")

        print(f"\n[{i+1}/{len(demo_data)}] Question: {question}")

        # Simulate a VeraRAG output with plausible results
        predicted_answers = {
            "demo_001": "Prime Minister",
            "demo_002": "no",
            "demo_003": "equal"
        }
        predicted_answer = predicted_answers.get(sample["_id"], "unknown")

        # Get supporting facts for evaluation
        supporting_facts = sample.get("supporting_facts", [])
        gold_fact_ids = [f"doc_{sample.get('_id', '')}_{fact[0]}" for fact in supporting_facts]

        # Simulated retrieved evidence IDs
        retrieved_ids = gold_fact_ids[:1]  # Simulate partial retrieval

        em = AnswerMetrics.exact_match(predicted_answer, gold_answer)
        f1 = AnswerMetrics.f1_score(predicted_answer, gold_answer)
        evidence_precision = EvidenceMetrics.evidence_precision(retrieved_ids, gold_fact_ids)
        evidence_recall = EvidenceMetrics.evidence_recall(retrieved_ids, gold_fact_ids)

        result = {
            "id": sample.get("_id", ""),
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": predicted_answer,
            "exact_match": em,
            "f1": f1,
            "evidence_precision": evidence_precision,
            "evidence_recall": evidence_recall,
            "confidence": 0.85,
            "num_evidence": len(retrieved_ids),
            "num_conflicts": 0
        }

        results["predictions"].append(result)

        print(f"  Gold: {gold_answer}")
        print(f"  Pred: {predicted_answer}")
        print(f"  EM: {em:.2f}, F1: {f1:.2f}")
        print(f"  Ev. Prec: {evidence_precision:.2f}, Ev. Rec: {evidence_recall:.2f}")

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
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run VeraRAG on HotpotQA")
    parser.add_argument("--demo", action="store_true", help="Run demo with synthetic data (no API needed)")
    parser.add_argument("--data-path", type=str, default=None, help="Path to HotpotQA data file")
    parser.add_argument("--config", type=str, default="hotpotqa", help="Config name (hotpotqa, fever, etc.)")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--output", type=str, default="results/external/hotpotqa_results.json", help="Output path")
    parser.add_argument("--index-only", action="store_true", help="Only build index, don't evaluate")

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
