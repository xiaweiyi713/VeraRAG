"""
Run baseline experiments for comparison with VeraRAG.

Supports three baselines:
  - Vanilla RAG  (BM25 retrieve -> single LLM generate)
  - Hybrid RAG   (BM25 + dense retrieve -> single LLM generate)
  - Self-RAG      (BM25 retrieve -> generate -> self-critique -> revise)

Modes:
  --demo    : Run on 3 synthetic questions (no LLM / no external data required)
  --full    : Run on a real dataset with a live LLM

Usage:
  python experiments/run_baselines.py --demo
  python experiments/run_baselines.py --full --data-path data/benchmark.json --num-samples 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.answer_metrics import AnswerMetrics
from src.utils.data_structures import VeraRAGOutput

# ---------------------------------------------------------------------------
# Baseline registry
# ---------------------------------------------------------------------------

_BASELINES: dict[str, type] = {}


def _register(name: str):
    """Decorator to register a baseline class."""
    def wrapper(cls):
        _BASELINES[name] = cls
        return cls
    return wrapper


# Lazy imports to avoid breaking when optional deps are missing
def _import_baselines():
    from experiments.baselines.vanilla_rag import VanillaRAG
    from experiments.baselines.hybrid_rag import HybridRAG
    from experiments.baselines.self_rag import SelfRAG
    _BASELINES["vanilla_rag"] = VanillaRAG
    _BASELINES["hybrid_rag"] = HybridRAG
    _BASELINES["self_rag"] = SelfRAG


def get_baseline(name: str, config: dict[str, Any] | None = None):
    """Instantiate a named baseline."""
    if not _BASELINES:
        _import_baselines()
    if name not in _BASELINES:
        raise ValueError(f"Unknown baseline: {name}. Available: {list(_BASELINES.keys())}")
    return _BASELINES[name](config=config)


# ---------------------------------------------------------------------------
# Demo data (synthetic)
# ---------------------------------------------------------------------------

DEMO_QUESTIONS = [
    {
        "question": "What is the capital of France?",
        "answer": "Paris",
        "documents": [
            {"id": "d1", "title": "France", "text": "France is a country in Western Europe. Its capital is Paris, the largest city in the country."},
            {"id": "d2", "title": "European Capitals", "text": "Berlin is the capital of Germany. Madrid is the capital of Spain."},
        ],
    },
    {
        "question": "Who wrote the novel '1984'?",
        "answer": "George Orwell",
        "documents": [
            {"id": "d3", "title": "George Orwell", "text": "George Orwell was an English novelist. He wrote the novel 1984, published in 1949."},
            {"id": "d4", "title": "Famous Novels", "text": "The novel Brave New World was written by Aldous Huxley."},
        ],
    },
    {
        "question": "What is the speed of light in vacuum?",
        "answer": "299,792,458 meters per second",
        "documents": [
            {"id": "d5", "title": "Speed of Light", "text": "The speed of light in vacuum is 299,792,458 meters per second, approximately 3 x 10^8 m/s."},
            {"id": "d6", "title": "Physical Constants", "text": "Planck's constant is 6.626 x 10^-34 J*s."},
        ],
    },
]


# ---------------------------------------------------------------------------
# Demo mode (no LLM needed — mocks the generate call)
# ---------------------------------------------------------------------------

class _MockLLMClient:
    """Minimal mock that returns canned responses for demo mode."""

    def __init__(self, **_kwargs):
        pass

    def generate(self, prompt: str, **_kwargs) -> str:
        # Heuristic: return a plausible short answer based on question keywords
        prompt_lower = prompt.lower()
        if "capital of france" in prompt_lower:
            return "Paris"
        if "1984" in prompt_lower or "orwell" in prompt_lower:
            return "George Orwell"
        if "speed of light" in prompt_lower:
            return "299,792,458 meters per second"
        # Generic fallback
        return "Based on the context, the answer is unclear."


def _run_demo(config: dict[str, Any]) -> dict[str, Any]:
    """Run baselines on synthetic demo questions (no real LLM)."""
    from experiments.baselines.vanilla_rag import VanillaRAG
    from experiments.baselines.hybrid_rag import HybridRAG
    from experiments.baselines.self_rag import SelfRAG

    results: dict[str, Any] = {"mode": "demo", "baselines": {}}

    baseline_classes = {
        "VanillaRAG": VanillaRAG,
        "HybridRAG": HybridRAG,
        "SelfRAG": SelfRAG,
    }

    for bl_name, BlClass in baseline_classes.items():
        print(f"\n{'=' * 60}")
        print(f"  Baseline: {bl_name}")
        print(f"{'=' * 60}")

        instance = BlClass(config=config)
        # Patch the LLM client with mock for demo mode
        instance.llm_client = _MockLLMClient()
        # For HybridRAG, replace the hybrid retriever with BM25-only
        # since sentence_transformers is not available in demo mode
        if bl_name == "HybridRAG":
            from src.retriever.bm25 import BM25Retriever
            instance.retriever = BM25Retriever()

        baseline_results = []
        for item in DEMO_QUESTIONS:
            docs = item["documents"]

            # Index documents for each question
            instance.index_documents(docs)

            output = instance.query(item["question"])

            em = AnswerMetrics.exact_match(output.answer, item["answer"])
            f1 = AnswerMetrics.f1_score(output.answer, item["answer"])

            print(
                f"  Q: {item['question'][:60]}...\n"
                f"    Gold:     {item['answer']}\n"
                f"    Predicted:{output.answer}\n"
                f"    EM={em:.1f}  F1={f1:.3f}  "
                f"Ev={len(output.evidence)}  Conf={output.confidence:.2f}"
            )

            baseline_results.append({
                "question": item["question"],
                "gold_answer": item["answer"],
                "predicted_answer": output.answer,
                "exact_match": em,
                "f1": f1,
                "num_evidence": len(output.evidence),
                "confidence": output.confidence,
            })

        avg_em = sum(r["exact_match"] for r in baseline_results) / len(baseline_results)
        avg_f1 = sum(r["f1"] for r in baseline_results) / len(baseline_results)
        avg_conf = sum(r["confidence"] for r in baseline_results) / len(baseline_results)

        results["baselines"][bl_name] = {
            "predictions": baseline_results,
            "metrics": {
                "num_samples": len(baseline_results),
                "exact_match": avg_em,
                "f1_score": avg_f1,
                "avg_confidence": avg_conf,
            },
        }

        print(f"\n  >> {bl_name} avg  EM={avg_em:.3f}  F1={avg_f1:.3f}  Conf={avg_conf:.3f}")

    # Summary table
    print(f"\n{'=' * 60}")
    print("  Summary (demo mode)")
    print(f"{'=' * 60}")
    print(f"  {'Baseline':<15} {'EM':>6} {'F1':>8} {'Conf':>8}")
    print(f"  {'-' * 37}")
    for bl_name, bl_data in results["baselines"].items():
        m = bl_data["metrics"]
        print(f"  {bl_name:<15} {m['exact_match']:>6.3f} {m['f1_score']:>8.3f} {m['avg_confidence']:>8.3f}")

    return results


# ---------------------------------------------------------------------------
# Full mode (real LLM)
# ---------------------------------------------------------------------------

def _load_dataset(data_path: str) -> list[dict[str, Any]]:
    """Load benchmark dataset."""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("samples", data.get("questions", [data]))
    return data


def _run_full(
    data_path: str,
    num_samples: int,
    baselines: list[str] | None,
    output_path: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run baselines on a real dataset with a live LLM."""
    if not _BASELINES:
        _import_baselines()

    dataset = _load_dataset(data_path)
    dataset = dataset[:num_samples]

    print(f"Loaded {len(dataset)} samples from {data_path}")

    # Collect all documents from the dataset for indexing
    all_docs: list[dict[str, Any]] = []
    for item in dataset:
        for doc in item.get("documents", []):
            all_docs.append(doc)
    if not all_docs:
        print("WARNING: No documents found in dataset. Baselines will have no retrieval context.")

    chosen = baselines or list(_BASELINES.keys())
    results: dict[str, Any] = {"mode": "full", "baselines": {}}

    for bl_name in chosen:
        print(f"\n{'=' * 60}")
        print(f"  Baseline: {bl_name}")
        print(f"{'=' * 60}")

        instance = get_baseline(bl_name, config=config)
        instance.index_documents(all_docs)

        baseline_results = []
        for i, item in enumerate(dataset):
            question = item.get("question", "")
            gold_answer = item.get("answer", "")

            print(f"  [{i + 1}/{len(dataset)}] {question[:70]}...")

            try:
                output = instance.query(question)
                em = AnswerMetrics.exact_match(output.answer, gold_answer)
                f1 = AnswerMetrics.f1_score(output.answer, gold_answer)

                baseline_results.append({
                    "question": question,
                    "gold_answer": gold_answer,
                    "predicted_answer": output.answer,
                    "exact_match": em,
                    "f1": f1,
                    "num_evidence": len(output.evidence),
                    "confidence": output.confidence,
                })

                print(f"    EM={em:.1f}  F1={f1:.3f}  Ev={len(output.evidence)}  Conf={output.confidence:.2f}")

            except Exception as exc:
                print(f"    ERROR: {exc}")
                continue

        if baseline_results:
            avg_em = sum(r["exact_match"] for r in baseline_results) / len(baseline_results)
            avg_f1 = sum(r["f1"] for r in baseline_results) / len(baseline_results)
            avg_conf = sum(r["confidence"] for r in baseline_results) / len(baseline_results)

            results["baselines"][bl_name] = {
                "predictions": baseline_results,
                "metrics": {
                    "num_samples": len(baseline_results),
                    "exact_match": avg_em,
                    "f1_score": avg_f1,
                    "avg_confidence": avg_conf,
                },
            }
            print(f"\n  >> {bl_name} avg  EM={avg_em:.3f}  F1={avg_f1:.3f}  Conf={avg_conf:.3f}")

    # Save results
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run baseline RAG experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--demo", action="store_true", help="Run in demo mode (no LLM needed)")
    parser.add_argument("--full", action="store_true", help="Run in full mode (requires LLM)")
    parser.add_argument("--data-path", type=str, help="Path to benchmark dataset (full mode)")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of samples (full mode)")
    parser.add_argument("--baselines", nargs="+", help="Which baselines to run (default: all)")
    parser.add_argument("--output", type=str, default="results/baselines/results.json", help="Output path")
    args = parser.parse_args()

    if not args.demo and not args.full:
        parser.error("Specify --demo or --full")

    config: dict[str, Any] = {}

    if args.demo:
        results = _run_demo(config)
        # Also save demo results
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nDemo results saved to {args.output}")
    else:
        if not args.data_path:
            parser.error("--full requires --data-path")
        results = _run_full(
            data_path=args.data_path,
            num_samples=args.num_samples,
            baselines=args.baselines,
            output_path=args.output,
            config=config,
        )


if __name__ == "__main__":
    main()
