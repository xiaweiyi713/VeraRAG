#!/usr/bin/env python3
"""Plan the VeraBench fixed-vs-adaptive retrieval-policy ablation."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_BASELINE_CONFIG = "configs/verabench_v112_canonical.yaml"
DEFAULT_CANDIDATE_CONFIG = "configs/verabench_v112_retrieval_adaptive_top3.yaml"


def _load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{config_path}: config root must be a mapping")
    return payload


def _canonical_run(config: dict[str, Any], label: str) -> dict[str, Any]:
    run = config.get("canonical_run")
    if not isinstance(run, dict):
        raise ValueError(f"{label}: missing canonical_run mapping")
    for field in ("name", "benchmark_version", "output", "checkpoint"):
        if not run.get(field):
            raise ValueError(f"{label}: canonical_run.{field} is required")
    return run


def _section(config: dict[str, Any], label: str, name: str) -> dict[str, Any]:
    section = config.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"{label}: missing {name} mapping")
    return section


def _assert_equal(name: str, baseline: Any, candidate: Any) -> dict[str, Any]:
    if baseline != candidate:
        raise ValueError(f"{name} differs between baseline and candidate")
    return {"name": name, "status": "matched"}


def build_plan(
    *,
    baseline_config_path: str = DEFAULT_BASELINE_CONFIG,
    candidate_config_path: str = DEFAULT_CANDIDATE_CONFIG,
    comparison_output: str | None = None,
    restart: bool = False,
) -> dict[str, Any]:
    baseline_config = _load_config(baseline_config_path)
    candidate_config = _load_config(candidate_config_path)
    baseline_run = _canonical_run(baseline_config, "baseline")
    candidate_run = _canonical_run(candidate_config, "candidate")
    baseline_retriever = _section(baseline_config, "baseline", "retriever")
    candidate_retriever = _section(candidate_config, "candidate", "retriever")

    checks = [
        _assert_equal(
            "benchmark_version",
            baseline_run["benchmark_version"],
            candidate_run["benchmark_version"],
        ),
        _assert_equal("llm", baseline_config.get("llm"), candidate_config.get("llm")),
        _assert_equal(
            "pipeline",
            baseline_config.get("pipeline"),
            candidate_config.get("pipeline"),
        ),
        _assert_equal(
            "retriever.type",
            baseline_retriever.get("type"),
            candidate_retriever.get("type"),
        ),
    ]
    baseline_policy = str(baseline_retriever.get("top_k_policy", "fixed"))
    candidate_policy = str(candidate_retriever.get("top_k_policy", "fixed"))
    baseline_retrieval_top_k = baseline_retriever.get("retrieval_top_k", 10)
    candidate_retrieval_top_k = candidate_retriever.get("retrieval_top_k", 10)
    if baseline_policy == candidate_policy:
        raise ValueError(
            "baseline and candidate use the same retriever.top_k_policy; "
            "this is not a retrieval-policy ablation"
        )
    checks.append({
        "name": "retriever.top_k_policy",
        "status": "differs",
        "baseline": baseline_policy,
        "candidate": candidate_policy,
    })
    checks.append({
        "name": "retriever.retrieval_top_k",
        "status": "differs"
        if baseline_retrieval_top_k != candidate_retrieval_top_k
        else "matched",
        "baseline": baseline_retrieval_top_k,
        "candidate": candidate_retrieval_top_k,
    })

    comparison_output = comparison_output or (
        "outputs/remote_results/verabench_v112_retrieval_adaptive_comparison.md"
    )
    baseline_run_args = [
        "python",
        "experiments/run_verabench.py",
        "--config",
        baseline_config_path,
        "--output",
        str(baseline_run["output"]),
        "--checkpoint",
        str(baseline_run["checkpoint"]),
    ]
    candidate_run_args = [
        "python",
        "experiments/run_verabench.py",
        "--config",
        candidate_config_path,
        "--output",
        str(candidate_run["output"]),
        "--checkpoint",
        str(candidate_run["checkpoint"]),
    ]
    if restart:
        baseline_run_args.append("--restart")
        candidate_run_args.append("--restart")
    compare_args = [
        "python",
        "experiments/compare_verabench_reports.py",
        str(baseline_run["output"]),
        str(candidate_run["output"]),
        "--baseline-label",
        baseline_policy,
        "--candidate-label",
        candidate_policy,
        "--output",
        comparison_output,
    ]
    return {
        "schema_version": "retrieval-ablation-plan-v1",
        "baseline": {
            "config": baseline_config_path,
            "run_name": baseline_run["name"],
            "output": baseline_run["output"],
            "checkpoint": baseline_run["checkpoint"],
            "top_k_policy": baseline_policy,
            "retrieval_top_k": baseline_retrieval_top_k,
        },
        "candidate": {
            "config": candidate_config_path,
            "run_name": candidate_run["name"],
            "output": candidate_run["output"],
            "checkpoint": candidate_run["checkpoint"],
            "top_k_policy": candidate_policy,
            "retrieval_top_k": candidate_retrieval_top_k,
        },
        "checks": checks,
        "commands": {
            "baseline_run": {
                "argv": baseline_run_args,
                "shell": shlex.join(baseline_run_args),
            },
            "candidate_run": {
                "argv": candidate_run_args,
                "shell": shlex.join(candidate_run_args),
            },
            "compare": {
                "argv": compare_args,
                "shell": shlex.join(compare_args),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", default=DEFAULT_BASELINE_CONFIG)
    parser.add_argument("--candidate-config", default=DEFAULT_CANDIDATE_CONFIG)
    parser.add_argument("--comparison-output")
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Include --restart in planned run commands.",
    )
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    try:
        plan = build_plan(
            baseline_config_path=args.baseline_config,
            candidate_config_path=args.candidate_config,
            comparison_output=args.comparison_output,
            restart=args.restart,
        )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))

    rendered = json.dumps(plan, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
