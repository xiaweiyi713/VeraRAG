import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.rescore_verabench import rescore_report


def test_rescore_report_recomputes_answer_metrics_and_calibration():
    report = {
        "metadata": {"provider": "deepseek"},
        "question_results": [
            {
                "question_id": "V002",
                "question_type": "single_evidence",
                "question": "中国生成式AI管理暂行办法是什么时候施行的？",
                "ground_truth": "2023年8月15日。",
                "predicted": "该办法于2023年8月15日正式施行。",
                "expected_behavior": "answer_with_citation",
                "actual_behavior": "answer_with_citation",
                "correct": False,
                "answer_f1": 0.1,
                "confidence": 0.8,
                "difficulty": "easy",
            },
        ],
    }

    rescored = rescore_report(report, allow_unverified=True)

    assert rescored["overall_answer_f1"] == 1.0
    assert rescored["question_results"][0]["correct"] is True
    assert rescored["metadata"]["metric_versions"]["answer"] == "soft-f1-v2"
    assert rescored["metadata"]["metric_versions"]["behavior"] == "behavior-v2"
    assert rescored["metadata"]["metric_versions"]["conflict"] == "gold-evidence-pair-micro-f1-v2"
    assert rescored["metadata"]["rescored_offline"] is True


def test_rescore_report_recomputes_premise_refutation_diagnostics():
    report = {
        "question_results": [
            {
                "question_id": "V011",
                "question_type": "multi_evidence",
                "question": "比较谷歌和IBM在量子计算路线上的不同策略。",
                "ground_truth": "谷歌强调纠错突破，IBM强调模块化扩展。",
                "predicted": "谷歌通过增加量子比特降低错误率，IBM强调模块化扩展。",
                "expected_behavior": "answer_with_citation",
                "actual_behavior": "answer_with_citation",
                "correct": True,
                "premise_refutation_detected": True,
                "premise_refutation_correct": False,
                "confidence": 0.7,
                "difficulty": "hard",
            },
        ],
    }

    rescored = rescore_report(report, allow_unverified=True)
    row = rescored["question_results"][0]

    assert row["premise_refutation_expected"] is False
    assert row["premise_refutation_detected"] is False
    assert row["premise_refutation_correct"] is True


def test_rescore_report_recomputes_conflicts_against_gold_evidence_only():
    report = {
        "question_results": [
            {
                "question_id": "V001",
                "question_type": "single_evidence",
                "question": "欧盟AI法案将违规罚款上限设定为多少？",
                "ground_truth": "3500万欧元或全球年营业额7%。",
                "predicted": "3500万欧元或全球年营业额7%。",
                "expected_behavior": "answer_with_citation",
                "actual_behavior": "answer_with_citation",
                "correct": True,
                "predicted_conflicts": 1,
                "conflict_false_positives": 1,
                "confidence": 0.7,
                "difficulty": "easy",
                "diagnostics": {
                    "predicted_conflict_pairs": ["D006_c0-E1"],
                    "gold_conflict_pairs": [],
                },
            },
        ],
    }

    rescored = rescore_report(report, allow_unverified=True)
    row = rescored["question_results"][0]

    assert row["predicted_conflicts"] == 0
    assert row["conflict_false_positives"] == 0
    assert row["diagnostics"]["scored_predicted_conflict_pairs"] == []
    assert row["diagnostics"]["unscored_extraneous_conflict_pairs"] == ["D006_c0-E1"]


def test_rescore_rejects_missing_or_mismatched_benchmark_metadata():
    report = {"question_results": []}
    with pytest.raises(ValueError, match="missing benchmark"):
        rescore_report(report)

    report["metadata"] = {
        "benchmark": {
            "version": "1.1",
            "fingerprints": {
                "corpus_sha256": "old-corpus",
                "questions_sha256": "old-questions",
            },
        },
    }
    with pytest.raises(ValueError, match="does not match"):
        rescore_report(report)

    rescored = rescore_report(report, allow_benchmark_mismatch=True)
    assert rescored["metadata"]["benchmark_mismatch_allowed"] is True
    assert rescored["metadata"]["source_benchmark"]["version"] == "1.1"
    assert rescored["metadata"]["rescored_against_benchmark"]["version"] == "1.1.2"


def test_rescore_cli_fails_cleanly_without_benchmark_provenance(tmp_path):
    source = tmp_path / "legacy.json"
    output = tmp_path / "rescored.json"
    source.write_text(
        json.dumps({"question_results": []}),
        encoding="utf-8",
    )
    script = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "rescore_verabench.py"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            str(source),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "missing benchmark version/fingerprints" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()
