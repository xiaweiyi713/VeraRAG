#!/usr/bin/env python3
"""Run a paired VeraBench ablation for conflict detector variants.

The script writes two derived configs from a shared base config:

* ``rules`` disables the learned detector while keeping the rule/NLI graph.
* ``rules_plus_learned`` enables the learned CrossEncoder detector.

Use ``--plan-only`` to create the configs and command manifest without making
LLM calls. Omit it when an API key is available and you want to execute the two
VeraBench runs.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


SUMMARY_METRICS = [
    "completed",
    "errors",
    "overall_answer_em",
    "overall_answer_f1",
    "overall_evidence_recall",
    "overall_evidence_precision",
    "overall_conflict_f1",
    "behavior_accuracy",
    "avg_confidence",
    "ece",
    "brier_score",
    "avg_latency",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def build_variant_config(
    base_config: dict[str, Any],
    *,
    enable_learned: bool,
    learned_model_path: str | None,
    learned_threshold: float,
) -> dict[str, Any]:
    """Return a derived config with the requested conflict detector variant."""
    config = deepcopy(base_config)
    conflict_graph = config.setdefault("conflict_graph", {})
    conflict_graph["enable_learned_detector"] = enable_learned
    if learned_model_path:
        conflict_graph["learned_model_path"] = learned_model_path
    conflict_graph["learned_threshold"] = learned_threshold
    return config


def build_run_command(
    *,
    config_path: Path,
    output_path: Path,
    data_dir: str | None,
    question_types: list[str],
    max_questions: int | None,
    restart: bool,
    no_checkpoint: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "experiments.run_verabench",
        "--config",
        str(config_path),
        "--output",
        str(output_path),
        "--types",
        *question_types,
    ]
    if data_dir:
        command.extend(["--data-dir", data_dir])
    if max_questions is not None:
        command.extend(["--max", str(max_questions)])
    if restart:
        command.append("--restart")
    if no_checkpoint:
        command.append("--no-checkpoint")
    return command


def _run_command(command: list[str]) -> dict[str, Any]:
    started = time.time()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "elapsed_seconds": round(time.time() - started, 2),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_report(report: dict[str, Any]) -> dict[str, Any]:
    """Extract stable, comparison-oriented metrics from a run_verabench report."""
    summary = {metric: report.get(metric) for metric in SUMMARY_METRICS if metric in report}
    conflict_summary = report.get("conflict_summary") or {}
    if conflict_summary:
        summary["conflict_summary"] = {
            key: conflict_summary.get(key)
            for key in [
                "gold_conflicts",
                "predicted_conflicts",
                "true_positives",
                "false_positives",
                "false_negatives",
                "precision",
                "recall",
                "f1",
                "dominant_failure",
            ]
            if key in conflict_summary
        }
    if report.get("by_type"):
        summary["by_type"] = report["by_type"]
    return summary


def compare_summaries(
    baseline: dict[str, Any],
    learned: dict[str, Any],
) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for metric in SUMMARY_METRICS:
        left = baseline.get(metric)
        right = learned.get(metric)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            delta[metric] = round(right - left, 6)

    base_conflict = baseline.get("conflict_summary") or {}
    learned_conflict = learned.get("conflict_summary") or {}
    conflict_delta = {}
    for metric in ["predicted_conflicts", "true_positives", "false_positives", "false_negatives", "precision", "recall", "f1"]:
        left = base_conflict.get(metric)
        right = learned_conflict.get(metric)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            conflict_delta[metric] = round(right - left, 6)
    if conflict_delta:
        delta["conflict_summary"] = conflict_delta
    return delta


def build_ablation_plan(args: argparse.Namespace) -> dict[str, Any]:
    base_config_path = Path(args.config)
    base_config = _load_yaml(base_config_path)
    output_dir = Path(args.output_dir)
    config_dir = output_dir / "configs"
    report_dir = output_dir / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    learned_model_path = args.learned_model_path or os.environ.get("VERARAG_CONFLICT_MODEL")
    if not learned_model_path:
        raise SystemExit("Error: --learned-model-path or VERARAG_CONFLICT_MODEL is required")

    variants = {
        "rules": build_variant_config(
            base_config,
            enable_learned=False,
            learned_model_path=learned_model_path,
            learned_threshold=args.learned_threshold,
        ),
        "rules_plus_learned": build_variant_config(
            base_config,
            enable_learned=True,
            learned_model_path=learned_model_path,
            learned_threshold=args.learned_threshold,
        ),
    }

    planned_runs = []
    for name, config in variants.items():
        config_path = config_dir / f"{name}.yaml"
        output_path = report_dir / f"{name}.json"
        _write_yaml(config_path, config)
        planned_runs.append({
            "name": name,
            "config_path": str(config_path),
            "output_path": str(output_path),
            "command": build_run_command(
                config_path=config_path,
                output_path=output_path,
                data_dir=args.data_dir,
                question_types=args.types,
                max_questions=args.max,
                restart=args.restart,
                no_checkpoint=args.no_checkpoint,
            ),
        })

    return {
        "mode": "plan" if args.plan_only else "run",
        "base_config": str(base_config_path),
        "output_dir": str(output_dir),
        "question_types": args.types,
        "max_questions": args.max or "all",
        "learned_model_path": learned_model_path,
        "learned_threshold": args.learned_threshold,
        "runs": planned_runs,
    }


def execute_plan(plan: dict[str, Any]) -> list[dict[str, Any]]:
    executions = []
    for run in plan["runs"]:
        print(f"\nRunning {run['name']}...")
        result = _run_command(run["command"])
        executions.append({"name": run["name"], **result})
        if result["returncode"] != 0:
            break
    return executions


def write_summary(
    *,
    plan: dict[str, Any],
    executions: list[dict[str, Any]] | None,
    summary_path: Path,
) -> dict[str, Any]:
    payload = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "git_commit": _git_commit(),
        },
        "plan": plan,
    }
    if executions is not None:
        payload["executions"] = executions

    reports: dict[str, Any] = {}
    for run in plan["runs"]:
        output_path = Path(run["output_path"])
        if output_path.exists():
            reports[run["name"]] = summarize_report(_read_json(output_path))

    if reports:
        payload["reports"] = reports
    if "rules" in reports and "rules_plus_learned" in reports:
        payload["delta_rules_plus_learned_minus_rules"] = compare_summaries(
            reports["rules"],
            reports["rules_plus_learned"],
        )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def print_result(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    summary_path = payload.get("summary_path")
    if summary_path:
        print(f"\nSummary saved to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rules-only vs rules+learned VeraBench pipeline ablation")
    parser.add_argument(
        "--config",
        default="configs/verabench_v112_canonical.yaml",
        help="Base pipeline config YAML",
    )
    parser.add_argument("--data-dir", help="VeraBench data directory")
    parser.add_argument("--output-dir", default="results/conflict_pipeline_ablation", help="Directory for configs, reports, and summary")
    parser.add_argument("--types", nargs="+", default=["conflict", "misleading"], help="Question types passed to run_verabench.py")
    parser.add_argument("--max", type=int, help="Max questions per variant")
    parser.add_argument("--learned-model-path", help="Trained conflict detector path; defaults to VERARAG_CONFLICT_MODEL")
    parser.add_argument("--learned-threshold", type=float, default=0.5, help="Learned detector probability threshold")
    parser.add_argument("--summary", help="Summary JSON path; defaults to <output-dir>/summary.json")
    parser.add_argument("--plan-only", action="store_true", help="Write configs and command plan without running VeraBench")
    parser.add_argument("--restart", action="store_true", help="Pass --restart to each VeraBench run")
    parser.add_argument("--no-checkpoint", action="store_true", help="Pass --no-checkpoint to each VeraBench run")
    args = parser.parse_args()

    plan = build_ablation_plan(args)
    summary_path = Path(args.summary) if args.summary else Path(args.output_dir) / "summary.json"
    executions = None if args.plan_only else execute_plan(plan)
    payload = write_summary(plan=plan, executions=executions, summary_path=summary_path)
    payload["summary_path"] = str(summary_path)
    print_result(payload)
    failed = [
        result for result in executions or []
        if result.get("returncode") != 0
    ]
    if failed:
        raise SystemExit(
            f"{failed[0]['name']} failed with exit code {failed[0]['returncode']}. "
            f"See {summary_path} for captured stdout/stderr tails."
        )


if __name__ == "__main__":
    main()
