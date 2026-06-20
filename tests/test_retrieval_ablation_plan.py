import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from experiments.plan_retrieval_ablation import build_plan


def test_default_retrieval_ablation_plan_uses_fixed_vs_adaptive_configs():
    plan = build_plan()

    assert plan["schema_version"] == "retrieval-ablation-plan-v1"
    assert plan["baseline"]["top_k_policy"] == "fixed"
    assert plan["baseline"]["retrieval_top_k"] == 10
    assert plan["candidate"]["top_k_policy"] == "complexity_adaptive"
    assert plan["candidate"]["retrieval_top_k"] == 3
    assert plan["candidate"]["config"] == "configs/verabench_v112_retrieval_adaptive_top3.yaml"
    assert "compare_verabench_reports.py" in plan["commands"]["compare"]["shell"]
    assert plan["commands"]["baseline_run"]["argv"][0:2] == [
        "python",
        "experiments/run_verabench.py",
    ]
    assert {
        "name": "retriever.retrieval_top_k",
        "status": "differs",
        "baseline": 10,
        "candidate": 3,
    } in plan["checks"]
    assert all(check["status"] in {"matched", "differs"} for check in plan["checks"])


def test_retrieval_ablation_plan_rejects_same_policy(tmp_path):
    baseline = yaml.safe_load(
        Path("configs/verabench_v112_canonical.yaml").read_text(encoding="utf-8")
    )
    candidate = yaml.safe_load(
        Path("configs/verabench_v112_canonical.yaml").read_text(encoding="utf-8")
    )
    baseline_path = tmp_path / "baseline.yaml"
    candidate_path = tmp_path / "candidate.yaml"
    baseline_path.write_text(yaml.safe_dump(baseline), encoding="utf-8")
    candidate_path.write_text(yaml.safe_dump(candidate), encoding="utf-8")

    with pytest.raises(ValueError, match=r"same retriever\.top_k_policy"):
        build_plan(
            baseline_config_path=str(baseline_path),
            candidate_config_path=str(candidate_path),
        )


def test_plan_retrieval_ablation_cli_writes_json(tmp_path):
    output = tmp_path / "plan.json"

    subprocess.run(
        [
            sys.executable,
            "experiments/plan_retrieval_ablation.py",
            "--restart",
            "--output",
            str(output),
        ],
        check=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["commands"]["baseline_run"]["argv"][-1] == "--restart"
    assert payload["commands"]["candidate_run"]["argv"][-1] == "--restart"
