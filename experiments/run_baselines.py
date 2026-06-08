"""Run baseline comparisons against VeraRAG on VeraBench.

Usage:
  # Demo mode (no LLM):
  python experiments/run_baselines.py --demo

  # Full mode:
  python experiments/run_baselines.py --config configs/model.yaml

  # Specific baselines:
  python experiments/run_baselines.py --demo --baselines vanilla_rag hybrid_rag

  # Output:
  python experiments/run_baselines.py --demo --output results/baselines.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.benchmark.loader import load_verabench  # noqa: E402
from src.ingestion.pipeline import IngestionPipeline  # noqa: E402

BASELINES = {
    "vanilla_rag": {"label": "Vanilla RAG", "module": "vanilla_rag", "class": "MockVanillaRAG"},
    "hybrid_rag": {"label": "Hybrid RAG", "module": "hybrid_rag", "class": "MockHybridRAG"},
    "self_rag": {"label": "Self-RAG", "module": "self_rag", "class": "MockSelfRAG"},
}


def _index_verabench(baseline):
    """Index VeraBench corpus into a baseline's retriever."""
    corpus_path = os.path.join(_project_root, "data", "verabench", "corpus.jsonl")
    pipeline = IngestionPipeline(chunk_size=512, chunk_overlap=64, chunk_strategy="fixed")
    chunks, _ = pipeline.ingest_and_index(corpus_path, retriever_type="bm25")

    docs = [c.to_index_doc() for c in chunks]
    baseline.index_documents(docs)
    return len(docs)


def _score_answer(predicted: str, ground_truth: str) -> float:
    """Simple token-overlap F1."""
    pred_tokens = set(predicted.lower().split())
    gt_tokens = set(ground_truth.lower().split())
    if not pred_tokens or not gt_tokens:
        return 0.0
    overlap = pred_tokens & gt_tokens
    if not overlap:
        return 0.0
    precision = len(overlap) / len(pred_tokens)
    recall = len(overlap) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def run_demo_baselines(
    baseline_names: list[str],
    max_questions: int | None = None,
) -> dict[str, Any]:
    """Run baselines in demo mode (Mock LLM, real BM25 retrieval)."""
    benchmark = load_verabench()
    questions = benchmark.questions
    if max_questions:
        questions = questions[:max_questions]

    all_results = {}

    for name in baseline_names:
        info = BASELINES[name]
        print(f"\n{'='*50}")
        print(f"Running baseline: {name} ({info['label']})")

        # Import and instantiate
        mod = __import__(f"experiments.baselines.{info['module']}", fromlist=[info["class"]])
        cls = getattr(mod, info["class"])
        baseline = cls()

        num_docs = _index_verabench(baseline)
        print(f"  Indexed {num_docs} chunks")

        per_question = []
        for q in questions:
            t0 = time.time()
            result = baseline.query(q.question)
            latency = time.time() - t0

            f1 = _score_answer(result["answer"], q.ground_truth_answer)
            num_ev = len(result.get("evidence", []))

            per_question.append({
                "question_id": q.id,
                "type": q.type,
                "answer_f1": round(f1, 3),
                "num_evidence": num_ev,
                "confidence": result.get("confidence", 0),
                "latency": round(latency, 3),
            })

        n = len(per_question)
        def avg(key: str) -> float:
            return round(sum(p[key] for p in per_question) / n, 3) if n else 0

        all_results[name] = {
            "baseline": name,
            "label": info["label"],
            "num_questions": n,
            "answer_f1": avg("answer_f1"),
            "avg_evidence": avg("num_evidence"),
            "avg_confidence": avg("confidence"),
            "avg_latency": avg("latency"),
            "per_question": per_question,
        }
        print(f"  F1={all_results[name]['answer_f1']:.3f}  "
              f"Conf={all_results[name]['avg_confidence']:.3f}  "
              f"Evidence={all_results[name]['avg_evidence']:.1f}")

    return {
        "mode": "demo",
        "baselines": list(all_results.values()),
    }


def run_full_baselines(
    config: dict[str, Any],
    baseline_names: list[str],
    max_questions: int | None = None,
) -> dict[str, Any]:
    """Run baselines with real LLM."""
    benchmark = load_verabench()
    questions = benchmark.questions
    if max_questions:
        questions = questions[:max_questions]

    all_results = {}

    for name in baseline_names:
        info = BASELINES[name]
        print(f"\n{'='*50}")
        print(f"Running baseline: {name} ({info['label']})")

        mod = __import__(f"experiments.baselines.{info['module']}", fromlist=[info["class"]])
        # Use real class (not Mock) for full mode
        real_cls_name = info["class"].replace("Mock", "")
        cls = getattr(mod, real_cls_name, getattr(mod, info["class"]))
        baseline = cls(config)

        num_docs = _index_verabench(baseline)
        print(f"  Indexed {num_docs} chunks")

        per_question = []
        for q in questions:
            t0 = time.time()
            result = baseline.query(q.question)
            latency = time.time() - t0

            f1 = _score_answer(result["answer"], q.ground_truth_answer)
            per_question.append({
                "question_id": q.id,
                "type": q.type,
                "answer_f1": round(f1, 3),
                "num_evidence": len(result.get("evidence", [])),
                "confidence": result.get("confidence", 0),
                "latency": round(latency, 3),
            })

        n = len(per_question)
        def avg(key: str) -> float:
            return round(sum(p[key] for p in per_question) / n, 3) if n else 0

        all_results[name] = {
            "baseline": name,
            "label": info["label"],
            "num_questions": n,
            "answer_f1": avg("answer_f1"),
            "avg_evidence": avg("num_evidence"),
            "avg_confidence": avg("confidence"),
            "avg_latency": avg("latency"),
        }

    return {"mode": "full", "baselines": list(all_results.values())}


def print_comparison_table(summary: dict[str, Any]):
    """Print formatted comparison table."""
    baselines = summary["baselines"]
    print(f"\n{'Baseline':<16} {'Label':<14} {'F1':>8} {'Conf':>8} {'Evidence':>10} {'Latency':>10}")
    print("-" * 70)
    for b in baselines:
        print(f"{b['baseline']:<16} {b['label']:<14} "
              f"{b['answer_f1']:>8.3f} {b['avg_confidence']:>8.3f} "
              f"{b['avg_evidence']:>10.1f} {b['avg_latency']:>10.3f}")


def main():
    parser = argparse.ArgumentParser(description="VeraRAG Baseline Comparison")
    parser.add_argument("--demo", action="store_true", help="Demo mode (no LLM)")
    parser.add_argument("--config", type=str, help="Pipeline config (YAML)")
    parser.add_argument("--baselines", nargs="+", choices=list(BASELINES.keys()),
                        default=list(BASELINES.keys()))
    parser.add_argument("--max", type=int, help="Max questions")
    parser.add_argument("--output", type=str, help="Output JSON path")
    args = parser.parse_args()

    if args.demo:
        print("Running baselines in DEMO mode...")
        summary = run_demo_baselines(args.baselines, args.max)
    else:
        if not args.config:
            print("Error: --config required for full mode (or use --demo)")
            sys.exit(1)
        import yaml
        with open(args.config, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        summary = run_full_baselines(config, args.baselines, args.max)

    print_comparison_table(summary)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        import subprocess
        import time as _t
        try:
            _gh = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            _gh = "unknown"
        summary["metadata"] = {
            "git_commit": _gh,
            "mode": "demo" if args.demo else "full",
            "config_path": args.config or "",
            "timestamp": _t.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "baselines": args.baselines or "all",
            "max_questions": args.max or "all",
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
