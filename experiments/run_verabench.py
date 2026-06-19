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
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Ensure src is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.validate_verabench import build_audit_report  # noqa: E402
from src.benchmark.evaluator import VeraBenchEvaluator  # noqa: E402
from src.benchmark.loader import QUESTION_TYPES, VeraBenchLoader  # noqa: E402
from src.evaluation.answer_metrics import AnswerMetrics  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("verabench")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_sha256(project_root: Path = PROJECT_ROOT) -> str:
    """Fingerprint evaluation code so stale checkpoints cannot be reused."""
    digest = hashlib.sha256()
    roots = [
        project_root / "src",
        project_root / "verarag",
        project_root / "experiments" / "run_verabench.py",
    ]
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        elif root.exists():
            files.extend(sorted(root.rglob("*.py")))
    for path in sorted(files):
        digest.update(str(path.relative_to(project_root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _build_run_signature(
    *,
    benchmark_metadata: dict[str, Any],
    config_path: str | None,
    question_types: list[str] | None,
    question_ids: list[str] | None,
    max_questions: int | None,
) -> dict[str, Any]:
    config_sha256 = ""
    if config_path:
        path = Path(config_path)
        if path.exists():
            config_sha256 = _sha256(path)
    return {
        "schema_version": 1,
        "benchmark_version": benchmark_metadata["version"],
        "benchmark_corpus_sha256": benchmark_metadata["fingerprints"]["corpus_sha256"],
        "benchmark_questions_sha256": benchmark_metadata["fingerprints"]["questions_sha256"],
        "implementation_sha256": _implementation_sha256(),
        "config_sha256": config_sha256,
        "question_types": sorted(question_types) if question_types else "all",
        "question_ids": sorted(question_ids) if question_ids else "all",
        "max_questions": max_questions if max_questions is not None else "all",
    }


def _prepare_checkpoint(
    checkpoint_path: str,
    signature: dict[str, Any],
    *,
    restart: bool,
) -> bool:
    """Prepare a checkpoint and reject stale or incompatible resumptions.

    Returns True when an existing compatible checkpoint will be resumed.
    """
    path = Path(checkpoint_path)
    metadata_path = Path(f"{checkpoint_path}.meta.json")
    if restart:
        path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)

    if path.exists():
        if not metadata_path.exists():
            raise ValueError(
                f"Checkpoint metadata missing for {checkpoint_path}; "
                "use --restart to avoid reusing unverifiable results"
            )
        saved = json.loads(metadata_path.read_text(encoding="utf-8"))
        if saved != signature:
            differing = sorted(
                key
                for key in set(saved) | set(signature)
                if saved.get(key) != signature.get(key)
            )
            raise ValueError(
                f"Checkpoint signature mismatch for {checkpoint_path}: "
                f"{', '.join(differing)}; use --restart"
            )
        return True

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(signature, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return False


def _unknown_question_ids(question_ids: list[str] | None, known_ids: set[str]) -> list[str]:
    """Return requested VeraBench question ids that are not present."""
    if not question_ids:
        return []
    return sorted(set(question_ids) - known_ids)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _read_config_run_metadata(config_path: str) -> dict[str, Any]:
    """Read report metadata from an evaluation config without hiding failures."""
    try:
        import yaml

        with open(config_path, encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
        if not isinstance(cfg, dict):
            raise ValueError("top-level YAML object is not a mapping")
        llm = cfg.get("llm", {}) or {}
        if not isinstance(llm, dict):
            raise ValueError("llm config is not a mapping")
        retriever = cfg.get("retriever", {}) or {}
        if not isinstance(retriever, dict):
            raise ValueError("retriever config is not a mapping")
        metadata = {
            "model": llm.get("model", ""),
            "provider": llm.get("provider", ""),
            "temperature": llm.get("temperature", ""),
            "max_tokens": llm.get("max_tokens", ""),
        }
        for key in (
            "type",
            "top_k_policy",
            "precision_cap_top_k",
            "adaptive_simple_top_k",
            "adaptive_medium_top_k",
            "adaptive_complex_top_k",
        ):
            if key in retriever:
                metadata[f"retriever_{key}"] = retriever[key]
        return metadata
    except Exception as exc:
        warning = f"failed to read config metadata from {config_path}: {exc}"
        logger.warning(warning)
        return {"config_metadata_warning": warning}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VeraBench Evaluation Runner")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--demo",
        action="store_true",
        help="Demo mode: validate report plumbing by scoring gold labels against themselves",
    )
    mode.add_argument("--config", type=str, help="Pipeline config YAML path")
    parser.add_argument("--data-dir", type=str, help="VeraBench data directory")
    parser.add_argument(
        "--types",
        nargs="+",
        choices=QUESTION_TYPES,
        help="Run specific question types only",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        help="Run specific VeraBench question ids, e.g. V036 V048 V084",
    )
    parser.add_argument("--max", type=_positive_int, help="Max number of questions to evaluate")
    parser.add_argument("--output", type=str, help="Output JSON report path")
    parser.add_argument(
        "--checkpoint",
        type=str,
        help=(
            "Checkpoint JSONL path for incremental save / resume. "
            "Defaults to '<output>.ckpt.jsonl' when --output is set."
        ),
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore and overwrite any existing checkpoint (start fresh).",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable incremental checkpointing entirely.",
    )
    return parser


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

    confidence_intervals = report.confidence_intervals
    interval_metrics = confidence_intervals.get("metrics", {})
    if interval_metrics:
        confidence_level = (
            float(confidence_intervals.get("confidence_level", 0.95)) * 100
        )
        print(f"\n--- Stratified Bootstrap {confidence_level:.1f}% Intervals ---")
        labels = (
            ("answer_f1", "Answer F1"),
            ("evidence_recall", "Evidence Recall"),
            ("behavior_accuracy", "Behavior Accuracy"),
            ("conflict_micro_f1", "Conflict micro-F1"),
            ("ece", "ECE"),
            ("brier_score", "Brier Score"),
        )
        for key, label in labels:
            interval = interval_metrics.get(key)
            if interval:
                print(
                    f"  {label:<20s} "
                    f"[{interval['lower']:.4f}, {interval['upper']:.4f}]"
                )
        print(
            "  "
            f"method={confidence_intervals.get('method', 'unknown')}, "
            f"resamples={confidence_intervals.get('resamples', 0)}, "
            f"seed={confidence_intervals.get('seed', 0)}"
        )

    dependency_intervals = report.dependency_robust_confidence_intervals
    dependency_metrics = dependency_intervals.get("metrics", {})
    if dependency_metrics:
        confidence_level = (
            float(dependency_intervals.get("confidence_level", 0.95)) * 100
        )
        clusters = int(dependency_intervals.get("clusters", 0))
        print(
            f"\n--- Evidence-Cluster Bootstrap {confidence_level:.1f}% "
            f"Intervals ({clusters} clusters) ---"
        )
        for key, label in (
            ("answer_f1", "Answer F1"),
            ("evidence_recall", "Evidence Recall"),
            ("behavior_accuracy", "Behavior Accuracy"),
            ("conflict_micro_f1", "Conflict micro-F1"),
            ("ece", "ECE"),
            ("brier_score", "Brier Score"),
        ):
            interval = dependency_metrics.get(key)
            if interval:
                print(
                    f"  {label:<20s} "
                    f"[{interval['lower']:.4f}, {interval['upper']:.4f}]"
                )
        print(
            "  "
            f"method={dependency_intervals.get('method', 'unknown')}, "
            f"resamples={dependency_intervals.get('resamples', 0)}, "
            f"seed={dependency_intervals.get('seed', 0)}"
        )

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

    if report.premise_refutation_summary:
        p = report.premise_refutation_summary
        print("\n--- Premise Refutation Diagnostics ---")
        print(f"  Expected / Detected: {p.get('expected', 0)} / {p.get('detected', 0)}")
        print(f"  TP / FP / FN:        {p.get('true_positives', 0)} / {p.get('false_positives', 0)} / {p.get('false_negatives', 0)}")
        print(f"  Precision/Recall:    {p.get('precision', 0):.4f} / {p.get('recall', 0):.4f}")

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


def main(argv: list[str] | None = None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    print("Loading VeraBench...")
    loader = VeraBenchLoader(args.data_dir)
    benchmark = loader.load()
    package_data_dir = PROJECT_ROOT / "src" / "benchmark" / "data" / "verabench"
    benchmark_metadata = build_audit_report(
        loader.data_dir,
        package_data_dir if args.data_dir is None and package_data_dir.exists() else None,
    )
    stats = benchmark.stats()
    print(f"  Corpus: {stats['total_documents']} documents")
    print(f"  Questions: {stats['total_questions']}")
    print(f"  Types: {stats['questions_by_type']}")
    print(f"  Multi-hop: {stats['multi_hop_count']}")
    print(f"  With conflicts: {stats['conflict_count']}")
    unknown_ids = _unknown_question_ids(args.ids, {q.id for q in benchmark.questions})
    if unknown_ids:
        parser.error(f"unknown VeraBench question id(s): {', '.join(unknown_ids)}")

    pipeline_factory = None
    if args.config:
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
                corpus_path = str(loader.data_dir / "corpus.jsonl")
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
            parser.error(
                f"failed to load pipeline config '{args.config}': {e}. "
                "VeraBench never falls back to demo mode for a requested real run"
            )

    if args.demo:
        print("\nRunning in demo mode (ground truth self-evaluation)...")

    # Resolve checkpoint path: enables incremental save + resume for pipeline runs.
    # Demo mode is instant, so checkpointing there is pointless and skipped.
    checkpoint_path = None
    run_signature = _build_run_signature(
        benchmark_metadata=benchmark_metadata,
        config_path=args.config,
        question_types=args.types,
        question_ids=args.ids,
        max_questions=args.max,
    )
    if not args.demo and not args.no_checkpoint:
        checkpoint_path = args.checkpoint or (
            (args.output + ".ckpt.jsonl") if args.output else "results/verabench.ckpt.jsonl"
        )
        resumed = _prepare_checkpoint(
            checkpoint_path,
            run_signature,
            restart=args.restart,
        )
        if args.restart:
            print(f"--restart: initialized fresh checkpoint {checkpoint_path}")
        if resumed:
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
        question_ids=args.ids,
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
            "question_ids": args.ids or "all",
            "max_questions": args.max or "all",
            "benchmark": benchmark_metadata,
            "run_signature": run_signature,
            "metric_versions": {
                "answer": AnswerMetrics.VERSION,
                "behavior": VeraBenchEvaluator.BEHAVIOR_METRIC_VERSION,
                "conflict": VeraBenchEvaluator.CONFLICT_METRIC_VERSION,
            },
        }
        if not args.demo and args.config:
            report_dict["metadata"].update(_read_config_run_metadata(args.config))
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report_dict, f, ensure_ascii=False, indent=2)
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
