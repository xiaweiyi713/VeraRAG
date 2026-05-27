"""
VeraRAG Ablation Experiment Runner.

Systematically disables pipeline components to measure their individual
contribution to answer quality, evidence recall, and conflict detection.

Ablation groups:
  1. full          — all features enabled (baseline)
  2. no_conflict   — disable conflict graph
  3. no_uncertainty — disable uncertainty estimation
  4. no_verification — disable verification layer
  5. no_repair     — disable repair agent
  6. minimal       — all optional features disabled
  7. single_round  — limit retrieval to 1 round

Usage:
  # Demo mode (no LLM required, simulates ablation results):
  python experiments/run_ablation.py --demo

  # Full mode with pipeline:
  python experiments/run_ablation.py --config configs/model.yaml

  # Filter groups and limit questions:
  python experiments/run_ablation.py --demo --groups full no_conflict minimal --max 10

  # Output to file:
  python experiments/run_ablation.py --demo --output results/ablation.json
"""

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root on path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from configs import merge_configs


# --- Ablation group definitions ---

ABLATION_GROUPS = {
    "full": {
        "label": "完整流水线",
        "overrides": {},
    },
    "no_conflict": {
        "label": "无冲突检测",
        "overrides": {
            "pipeline": {"enable_conflict_graph": False},
        },
    },
    "no_uncertainty": {
        "label": "无不确定性估计",
        "overrides": {
            "pipeline": {"enable_uncertainty": False},
        },
    },
    "no_verification": {
        "label": "无验证层",
        "overrides": {
            "pipeline": {"enable_verification": False},
        },
    },
    "no_repair": {
        "label": "无修复",
        "overrides": {
            "pipeline": {"enable_repair": False},
        },
    },
    "minimal": {
        "label": "最小配置",
        "overrides": {
            "pipeline": {
                "enable_conflict_graph": False,
                "enable_uncertainty": False,
                "enable_verification": False,
                "enable_repair": False,
            },
        },
    },
    "single_round": {
        "label": "单轮检索",
        "overrides": {
            "pipeline": {"max_retrieval_rounds": 1},
        },
    },
}


def build_config(base: Dict[str, Any], group_name: str) -> Dict[str, Any]:
    """Build pipeline config for an ablation group by merging overrides."""
    group = ABLATION_GROUPS[group_name]
    return merge_configs(deepcopy(base), group["overrides"])


# --- Demo mode: simulate ablation results ---

def run_demo_ablation(
    groups: List[str],
    max_questions: Optional[int] = None,
) -> Dict[str, Any]:
    """Run simulated ablation without real LLM.

    Generates plausible score distributions: the full pipeline scores best,
    each disabled component degrades a specific metric.
    """
    from src.benchmark.loader import load_verabench

    benchmark = load_verabench()
    questions = benchmark.questions
    if max_questions:
        questions = questions[:max_questions]

    # Simulated scores per group: (answer_f1, evidence_recall, conflict_f1, confidence)
    score_profiles = {
        "full":             (0.85, 0.88, 0.82, 0.84),
        "no_conflict":      (0.83, 0.87, 0.45, 0.81),
        "no_uncertainty":   (0.80, 0.82, 0.78, 0.76),
        "no_verification":  (0.74, 0.85, 0.79, 0.70),
        "no_repair":        (0.79, 0.86, 0.80, 0.77),
        "minimal":          (0.62, 0.65, 0.30, 0.58),
        "single_round":     (0.72, 0.60, 0.74, 0.68),
    }

    import random
    random.seed(42)

    results = {}
    all_group_results = []

    for group_name in groups:
        profile = score_profiles.get(group_name, score_profiles["full"])
        base_f1, base_ev, base_cf, base_conf = profile
        n = len(questions)

        per_question = []
        for q in questions:
            # Add small noise per question
            noise = lambda base, sigma=0.08: max(0, min(1, base + random.gauss(0, sigma)))
            per_question.append({
                "question_id": q.id,
                "question_type": q.type,
                "answer_f1": round(noise(base_f1), 3),
                "evidence_recall": round(noise(base_ev), 3),
                "conflict_f1": round(noise(base_cf, 0.12), 3),
                "confidence": round(noise(base_conf), 3),
            })

        # Aggregate
        avg = lambda key: round(sum(p[key] for p in per_question) / n, 3) if n else 0
        group_result = {
            "group": group_name,
            "label": ABLATION_GROUPS[group_name]["label"],
            "num_questions": n,
            "answer_f1": avg("answer_f1"),
            "evidence_recall": avg("evidence_recall"),
            "conflict_f1": avg("conflict_f1"),
            "confidence": avg("confidence"),
            "per_question": per_question,
        }
        results[group_name] = group_result
        all_group_results.append(group_result)

    # Compute delta vs full baseline
    baseline = results.get("full")
    if baseline:
        for g in all_group_results:
            g["delta_answer_f1"] = round(g["answer_f1"] - baseline["answer_f1"], 3)
            g["delta_evidence_recall"] = round(g["evidence_recall"] - baseline["evidence_recall"], 3)
            g["delta_conflict_f1"] = round(g["conflict_f1"] - baseline["conflict_f1"], 3)
            g["delta_confidence"] = round(g["confidence"] - baseline["confidence"], 3)

    # Summary table
    summary = {
        "mode": "demo",
        "num_questions": len(questions),
        "groups": groups,
        "metrics": ["answer_f1", "evidence_recall", "conflict_f1", "confidence"],
        "results": all_group_results,
    }
    return summary


# --- Full mode: real pipeline evaluation ---

def run_full_ablation(
    base_config: Dict[str, Any],
    groups: List[str],
    max_questions: Optional[int] = None,
    question_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run ablation with real VeraRAG pipeline on VeraBench."""
    from src.benchmark.loader import load_verabench
    from src.benchmark.evaluator import VeraBenchEvaluator
    from src.pipeline.verarag import VeraRAG

    benchmark = load_verabench()
    all_group_results = []

    for group_name in groups:
        config = build_config(base_config, group_name)
        label = ABLATION_GROUPS[group_name]["label"]

        print(f"\n{'='*60}")
        print(f"Running ablation group: {group_name} ({label})")
        print(f"{'='*60}")

        factory = lambda cfg=config: VeraRAG(cfg)
        evaluator = VeraBenchEvaluator(
            benchmark=benchmark,
            pipeline_factory=factory,
        )

        t0 = time.time()
        report = evaluator.evaluate(
            question_types=question_types,
            max_questions=max_questions,
        )
        elapsed = time.time() - t0

        group_result = {
            "group": group_name,
            "label": label,
            "num_questions": len(report.results),
            "answer_f1": report.overall_answer_f1,
            "evidence_recall": report.overall_evidence_recall,
            "conflict_f1": report.overall_conflict_f1,
            "confidence": report.avg_confidence,
            "avg_latency": report.avg_latency,
            "elapsed_seconds": round(elapsed, 2),
            "by_type": {k: v for k, v in report.by_type.items()},
        }
        all_group_results.append(group_result)
        print(f"  answer_f1={group_result['answer_f1']:.3f}  "
              f"evidence_recall={group_result['evidence_recall']:.3f}  "
              f"conflict_f1={group_result['conflict_f1']:.3f}  "
              f"confidence={group_result['confidence']:.3f}")

    # Delta vs full baseline
    baseline = next((g for g in all_group_results if g["group"] == "full"), None)
    if baseline:
        for g in all_group_results:
            g["delta_answer_f1"] = round(g["answer_f1"] - baseline["answer_f1"], 3)
            g["delta_evidence_recall"] = round(g["evidence_recall"] - baseline["evidence_recall"], 3)
            g["delta_conflict_f1"] = round(g["conflict_f1"] - baseline["conflict_f1"], 3)
            g["delta_confidence"] = round(g["confidence"] - baseline["confidence"], 3)

    return {
        "mode": "full",
        "groups": groups,
        "metrics": ["answer_f1", "evidence_recall", "conflict_f1", "confidence"],
        "results": all_group_results,
    }


def print_table(summary: Dict[str, Any]):
    """Print ablation results as a formatted table."""
    results = summary["results"]
    metrics = summary["metrics"]

    # Header
    header = f"{'Group':<20} {'Label':<14}"
    for m in metrics:
        header += f" {m:>18}"
    header += f" {'Δ answer_f1':>12} {'Δ conflict':>12}"
    print(f"\n{header}")
    print("-" * len(header))

    for r in results:
        row = f"{r['group']:<20} {r['label']:<14}"
        for m in metrics:
            row += f" {r.get(m, 0):>18.3f}"
        delta_a = r.get("delta_answer_f1", 0)
        delta_c = r.get("delta_conflict_f1", 0)
        row += f" {delta_a:>+12.3f} {delta_c:>+12.3f}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="VeraRAG Ablation Experiment Runner")
    parser.add_argument("--demo", action="store_true", help="Demo mode (no LLM required)")
    parser.add_argument("--config", type=str, help="Path to base pipeline config (YAML)")
    parser.add_argument("--groups", nargs="+", choices=list(ABLATION_GROUPS.keys()),
                        default=list(ABLATION_GROUPS.keys()),
                        help="Ablation groups to run")
    parser.add_argument("--max", type=int, help="Max questions per group")
    parser.add_argument("--types", nargs="+", help="Filter by question type")
    parser.add_argument("--output", type=str, help="Output JSON path")
    args = parser.parse_args()

    if args.demo:
        print("Running ablation in DEMO mode...")
        summary = run_demo_ablation(
            groups=args.groups,
            max_questions=args.max,
        )
    else:
        if not args.config:
            print("Error: --config required for full mode (or use --demo)")
            sys.exit(1)

        import yaml
        base_config = yaml.safe_load(open(args.config))

        summary = run_full_ablation(
            base_config=base_config,
            groups=args.groups,
            max_questions=args.max,
            question_types=args.types,
        )

    print_table(summary)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        import subprocess, time as _t
        try:
            _gh = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            _gh = "unknown"
        summary["metadata"] = {
            "git_commit": _gh,
            "mode": "demo" if args.demo else "full",
            "config_path": args.config or "",
            "timestamp": _t.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "groups": args.groups or "all",
            "max_questions": args.max or "all",
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
