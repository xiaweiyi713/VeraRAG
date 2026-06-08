#!/usr/bin/env python3
"""VeraBench evaluation runner.

Usage:
    # Run demo (ground truth vs ground truth, validates benchmark)
    python experiments/run_verabench.py --demo

    # Run with pipeline (requires LLM config)
    python experiments/run_verabench.py --config configs/model.yaml

    # Run specific question types only
    python experiments/run_verabench.py --demo --types conflict temporal

    # Limit number of questions
    python experiments/run_verabench.py --demo --max 10

    # Output JSON report
    python experiments/run_verabench.py --demo --output results.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure src is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark.evaluator import VeraBenchEvaluator  # noqa: E402
from src.benchmark.loader import QUESTION_TYPES, load_verabench  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("verabench")


def print_progress(current: int, total: int, q):
    pct = (current + 1) / total * 100
    print(f"  [{current+1}/{total}] ({pct:.0f}%) {q.type:20s} | {q.question[:50]}...")


def print_report(report):
    print("\n" + "=" * 70)
    print("VeraBench Evaluation Report")
    print("=" * 70)
    print(f"  Total: {report.total_questions}  |  Completed: {report.completed}  |  Errors: {report.errors}")
    print(f"  Answer EM:      {report.overall_answer_em:.4f}")
    print(f"  Answer F1:      {report.overall_answer_f1:.4f}")
    print(f"  Evidence Recall: {report.overall_evidence_recall:.4f}")
    print(f"  Evidence Prec:  {report.overall_evidence_precision:.4f}")
    print(f"  Conflict F1:    {report.overall_conflict_f1:.4f}")
    print(f"  Behavior Acc:   {report.behavior_accuracy:.4f}")
    print(f"  Avg Confidence: {report.avg_confidence:.4f}")
    print(f"  ECE:            {report.ece:.4f}")
    print(f"  Brier Score:    {report.brier_score:.4f}")
    print(f"  Avg Latency:    {report.avg_latency:.2f}s")

    if report.by_type:
        print("\n--- By Question Type ---")
        print(f"  {'Type':<20s} {'Count':>5s} {'EM':>6s} {'F1':>6s} {'EvRec':>6s} {'BhvAcc':>6s}")
        print("  " + "-" * 55)
        for qtype in QUESTION_TYPES:
            if qtype in report.by_type:
                d = report.by_type[qtype]
                print(f"  {qtype:<20s} {d['count']:>5d} {d['answer_em']:>6.3f} {d['answer_f1']:>6.3f} {d['evidence_recall']:>6.3f} {d['behavior_accuracy']:>6.3f}")

    if report.by_difficulty:
        print("\n--- By Difficulty ---")
        for diff in ["easy", "medium", "hard"]:
            if diff in report.by_difficulty:
                d = report.by_difficulty[diff]
                print(f"  {diff:<10s} count={d['count']:<3d} EM={d['answer_em']:.3f} F1={d['answer_f1']:.3f} BhvAcc={d['behavior_accuracy']:.3f}")

    if report.failure_summary:
        summary = report.failure_summary
        print("\n--- Failure Diagnostics ---")
        print(f"  Behavior failures:     {summary.get('behavior_failure_count', 0)}")
        print(f"  Low evidence recall:   {summary.get('low_evidence_recall_count', 0)}")
        print(f"  Conflict failures:     {summary.get('conflict_failure_count', 0)}")
        by_type = summary.get("behavior_failures_by_type", {})
        if by_type:
            formatted = ", ".join(f"{k}={v}" for k, v in sorted(by_type.items()))
            print(f"  Behavior failures by type: {formatted}")

    if report.conflict_summary:
        c = report.conflict_summary
        print("\n--- Conflict Diagnostics ---")
        print(f"  Gold / Predicted: {c.get('gold_conflicts', 0)} / {c.get('predicted_conflicts', 0)}")
        print(f"  TP / FP / FN:     {c.get('true_positives', 0)} / {c.get('false_positives', 0)} / {c.get('false_negatives', 0)}")
        print(f"  Precision/Recall: {c.get('precision', 0):.4f} / {c.get('recall', 0):.4f}")
        print(f"  Dominant failure: {c.get('dominant_failure', 'unknown')}")

    if report.behavior_confusion:
        print("\n--- Behavior Confusion (expected -> actual) ---")
        for expected, actual_counts in sorted(report.behavior_confusion.items()):
            actual = ", ".join(f"{k}:{v}" for k, v in sorted(actual_counts.items()))
            print(f"  {expected:<26s} -> {actual}")

    # Error details
    errored = [r for r in report.results if r.error]
    if errored:
        print(f"\n--- Errors ({len(errored)}) ---")
        for r in errored:
            print(f"  {r.question_id}: {r.error[:80]}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="VeraBench Evaluation Runner")
    parser.add_argument("--demo", action="store_true", help="Demo mode: score ground truth against itself")
    parser.add_argument("--config", type=str, help="Pipeline config YAML path")
    parser.add_argument("--data-dir", type=str, help="VeraBench data directory")
    parser.add_argument("--types", nargs="+", choices=QUESTION_TYPES, help="Run specific question types only")
    parser.add_argument("--max", type=int, help="Max number of questions to evaluate")
    parser.add_argument("--output", type=str, help="Output JSON report path")
    parser.add_argument("--checkpoint", type=str,
                        help="Checkpoint JSONL path for incremental save / resume. "
                             "Defaults to '<output>.ckpt.jsonl' when --output is set.")
    parser.add_argument("--restart", action="store_true",
                        help="Ignore and overwrite any existing checkpoint (start fresh).")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Disable incremental checkpointing entirely.")
    args = parser.parse_args()

    print("Loading VeraBench...")
    benchmark = load_verabench(args.data_dir)
    stats = benchmark.stats()
    print(f"  Corpus: {stats['total_documents']} documents")
    print(f"  Questions: {stats['total_questions']}")
    print(f"  Types: {stats['questions_by_type']}")
    print(f"  Multi-hop: {stats['multi_hop_count']}")
    print(f"  With conflicts: {stats['conflict_count']}")

    pipeline_factory = None
    if not args.demo and args.config:
        try:
            import yaml

            from src.pipeline.verarag import VeraRAG

            with open(args.config) as f:
                config = yaml.safe_load(f)

            # Auto-index VeraBench corpus into the pipeline.
            # Build the pipeline ONCE and reuse it across questions: loading the
            # embedding/reranker/NLI models and re-indexing the corpus per question
            # is hugely wasteful (and query() does not mutate pipeline state).
            _pipeline_cache: dict = {}

            def make_pipeline():
                if "p" in _pipeline_cache:
                    return _pipeline_cache["p"]
                p = VeraRAG(config)
                corpus_path = str(PROJECT_ROOT / "data" / "verabench" / "corpus.jsonl")
                if Path(corpus_path).exists():
                    from src.ingestion.pipeline import IngestionPipeline
                    ip = IngestionPipeline(chunk_size=512, chunk_overlap=64, chunk_strategy="fixed")
                    chunks, _retriever = ip.ingest_and_index(corpus_path, retriever_type="bm25")
                    index_docs = [c.to_index_doc() for c in chunks]
                    p.retriever.index_documents(index_docs)
                    logger.info(f"Indexed {len(chunks)} chunks from VeraBench corpus")
                _pipeline_cache["p"] = p
                return p

            pipeline_factory = make_pipeline
            print(f"Pipeline loaded from {args.config}")
        except Exception as e:
            print(f"Failed to load pipeline: {e}")
            print("Falling back to demo mode.")
            args.demo = True

    if args.demo:
        print("\nRunning in demo mode (ground truth self-evaluation)...")

    # Resolve checkpoint path: enables incremental save + resume for pipeline runs.
    # Demo mode is instant, so checkpointing there is pointless and skipped.
    checkpoint_path = None
    if not args.demo and not args.no_checkpoint:
        checkpoint_path = args.checkpoint or (
            (args.output + ".ckpt.jsonl") if args.output else "results/verabench.ckpt.jsonl"
        )
        if args.restart and Path(checkpoint_path).exists():
            Path(checkpoint_path).unlink()
            print(f"--restart: removed existing checkpoint {checkpoint_path}")
        if Path(checkpoint_path).exists():
            print(f"Resuming from checkpoint {checkpoint_path} (already-done questions will be skipped)")
        else:
            print(f"Incremental checkpoint: {checkpoint_path} (interrupted runs can be resumed)")

    evaluator = VeraBenchEvaluator(
        benchmark=benchmark,
        pipeline_factory=pipeline_factory if not args.demo else None,
    )

    t0 = time.time()
    report = evaluator.evaluate(
        question_types=args.types,
        max_questions=args.max,
        callback=print_progress,
        checkpoint_path=checkpoint_path,
    )
    elapsed = time.time() - t0
    print(f"\nEvaluation completed in {elapsed:.1f}s")

    print_report(report)

    if args.output:
        report_dict = report.to_dict()
        # Add reproducibility metadata
        import subprocess
        try:
            git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            git_hash = "unknown"
        report_dict["metadata"] = {
            "git_commit": git_hash,
            "mode": "demo" if args.demo else "pipeline",
            "config_path": args.config or "",
            "data_dir": args.data_dir or "default",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "question_types": args.types or "all",
            "max_questions": args.max or "all",
        }
        if not args.demo and args.config:
            try:
                import yaml
                with open(args.config) as f:
                    cfg = yaml.safe_load(f)
                llm = cfg.get("llm", {})
                report_dict["metadata"]["model"] = llm.get("model", "")
                report_dict["metadata"]["provider"] = llm.get("provider", "")
                report_dict["metadata"]["temperature"] = llm.get("temperature", "")
                report_dict["metadata"]["max_tokens"] = llm.get("max_tokens", "")
            except Exception:
                pass
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, ensure_ascii=False, indent=2)
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
