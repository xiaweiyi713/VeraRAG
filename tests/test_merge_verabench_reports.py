from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import pytest

from experiments.merge_verabench_reports import merge_reports
from experiments.validate_verabench import build_audit_report


@lru_cache(maxsize=1)
def _benchmark_metadata() -> dict:
    project_root = Path(__file__).resolve().parents[1]
    audit = build_audit_report(
        data_dir=project_root / "data" / "verabench",
    )
    return {
        "version": audit["version"],
        "fingerprints": audit["fingerprints"],
    }


def _report(question_id: str, question_type: str = "single_evidence") -> dict:
    return {
        "completed": 1,
        "errors": 0,
        "metadata": {
            "provider": "deepseek",
            "model": "deepseek-chat",
            "question_types": [question_type],
            "timestamp": "2026-06-14T12:00:00+08:00",
            "benchmark": deepcopy(_benchmark_metadata()),
            "run_signature": {
                "implementation_sha256": "implementation",
                "config_sha256": "config",
            },
            "metric_versions": {"answer": "soft-f1-v2"},
        },
        "question_results": [
            {
                "question_id": question_id,
                "question_type": question_type,
                "question": "placeholder",
                "ground_truth": "placeholder",
                "predicted": "placeholder",
                "expected_behavior": "answer_with_citation",
                "actual_behavior": "answer_with_citation",
                "correct": True,
                "difficulty": "easy",
            },
        ],
    }


def test_merge_reports_rescores_and_preserves_partition_provenance():
    merged = merge_reports(
        [_report("V001"), _report("V002")],
        labels=["part-a.json", "part-b.json"],
    )

    assert merged["completed"] == 2
    assert merged["metadata"]["mode"] == "partitioned_pipeline"
    assert merged["metadata"]["run_signature"]["partitioned"] is True
    assert [item["label"] for item in merged["metadata"]["partition_reports"]] == [
        "part-a.json",
        "part-b.json",
    ]
    assert [row["question_id"] for row in merged["question_results"]] == [
        "V001",
        "V002",
    ]


def test_merge_reports_rejects_duplicate_question_ids():
    with pytest.raises(ValueError, match="Duplicate question_id"):
        merge_reports([_report("V001"), _report("V001")])


def test_merge_reports_rejects_incompatible_signatures():
    other = deepcopy(_report("V002"))
    other["metadata"]["run_signature"]["implementation_sha256"] = "different"

    with pytest.raises(ValueError, match="implementation_sha256"):
        merge_reports([_report("V001"), other])


def test_merge_reports_rejects_source_benchmark_mismatch():
    first = _report("V001")
    second = _report("V002")
    first["metadata"]["benchmark"]["version"] = "1.1"
    second["metadata"]["benchmark"]["version"] = "1.1"

    with pytest.raises(ValueError, match="does not match"):
        merge_reports([first, second])


def test_merge_reports_can_require_full_benchmark_coverage():
    with pytest.raises(ValueError, match="incomplete"):
        merge_reports(
            [_report("V001"), _report("V002")],
            require_complete=True,
        )
