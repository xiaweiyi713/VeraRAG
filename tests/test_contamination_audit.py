"""Tests for local VeraBench contamination auditing."""

import json
import subprocess
import sys

import pytest

from src.benchmark.contamination import (
    AuditText,
    ReferenceText,
    _best_reference_ngram_excerpt,
    _excerpt,
    _near_match_basis,
    _near_matches,
    audit_verabench_contamination,
)
from tests.test_benchmark import SAMPLE_CORPUS, SAMPLE_QUESTIONS


def _write_sample_benchmark(path):
    (path / "corpus.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in SAMPLE_CORPUS),
        encoding="utf-8",
    )
    (path / "questions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in SAMPLE_QUESTIONS),
        encoding="utf-8",
    )


def test_contamination_audit_detects_question_and_answer_overlap(tmp_path):
    _write_sample_benchmark(tmp_path)
    reference = tmp_path / "reference.txt"
    reference.write_text(
        "训练集中意外包含：该系统2024年的预算是多少？答案是500万元。",
        encoding="utf-8",
    )

    report = audit_verabench_contamination(
        tmp_path,
        reference_paths=[reference],
    )

    assert report["status"] == "complete"
    assert report["summary"]["high_risk_exact_matches"] >= 1
    assert "T001" in report["summary"]["matched_item_ids"]
    question_matches = [
        match for match in report["matches"]["exact"]
        if match["item_id"] == "T001" and match["kind"] == "question"
    ]
    assert question_matches
    assert "该系统2024年的预算是多少" in question_matches[0]["reference_excerpt"]
    assert question_matches[0]["reference_start_char"] < question_matches[0]["reference_end_char"]


def test_contamination_audit_reports_no_references():
    report = audit_verabench_contamination()

    assert report["status"] == "no_references"
    assert report["coverage"]["reference_texts"] == 0
    assert report["summary"]["exact_matches"] == 0


def test_contamination_audit_high_risk_counts_survive_match_truncation(tmp_path):
    _write_sample_benchmark(tmp_path)
    reference = tmp_path / "reference.txt"
    corpus_contents = "\n".join(row["content"] for row in SAMPLE_CORPUS)
    reference.write_text(
        corpus_contents + "\n该系统2024年的预算是多少？",
        encoding="utf-8",
    )

    report = audit_verabench_contamination(
        tmp_path,
        reference_paths=[reference],
        max_matches=1,
    )

    assert len(report["matches"]["exact"]) == 1
    assert report["summary"]["matches_truncated"] is True
    assert report["summary"]["exact_matches"] > report["summary"]["returned_exact_matches"]
    assert report["summary"]["high_risk_exact_matches"] >= 1
    assert "T001" in report["summary"]["matched_item_ids"]


def test_contamination_audit_detects_short_item_contained_in_long_reference(tmp_path):
    _write_sample_benchmark(tmp_path)
    reference = tmp_path / "long_reference.txt"
    filler = "这是一段无关的长篇训练语料。" * 200
    reference.write_text(
        f"{filler} 该系统2024年预算是多少 {filler}",
        encoding="utf-8",
    )

    report = audit_verabench_contamination(
        tmp_path,
        reference_paths=[reference],
        near_threshold=0.95,
        containment_threshold=0.8,
        ngram_size=2,
    )

    assert report["summary"]["high_risk_near_duplicate_matches"] >= 1
    question_matches = [
        match for match in report["matches"]["near_duplicate"]
        if match["item_id"] == "T001" and match["kind"] == "question"
    ]
    assert question_matches
    assert question_matches[0]["match_basis"] == "item_containment"
    assert question_matches[0]["jaccard"] < 0.95
    assert question_matches[0]["item_ngram_containment"] >= 0.8
    assert "该系统2024年预算是多少" in question_matches[0]["reference_excerpt"]


def test_contamination_audit_cli_writes_report_and_can_fail_on_overlap(tmp_path):
    reference = tmp_path / "reference.jsonl"
    output = tmp_path / "audit.json"
    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    _write_sample_benchmark(benchmark_dir)
    reference.write_text(
        json.dumps(
            {"text": "该系统2024年的预算是多少？这是泄漏的问题文本。"},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "experiments/audit_verabench_contamination.py",
            "--data-dir",
            str(benchmark_dir),
            "--reference",
            str(reference),
            "--output",
            str(output),
            "--fail-on-high-risk-match",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert result.returncode == 1
    assert payload == saved
    assert payload["summary"]["high_risk_exact_matches"] >= 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"near_threshold": -0.1}, "near_threshold"),
        ({"containment_threshold": 1.1}, "containment_threshold"),
        ({"ngram_size": 1}, "ngram_size"),
        ({"min_exact_chars": 0}, "min_exact_chars"),
        ({"max_matches": -1}, "max_matches"),
    ],
)
def test_contamination_audit_rejects_invalid_options(kwargs, message):
    with pytest.raises(ValueError, match=message):
        audit_verabench_contamination(**kwargs)


def test_contamination_audit_reports_reference_path_errors(tmp_path):
    _write_sample_benchmark(tmp_path)
    unsupported = tmp_path / "reference.bin"
    unsupported.write_bytes(b"not text")

    with pytest.raises(FileNotFoundError):
        audit_verabench_contamination(tmp_path, reference_paths=[tmp_path / "missing.txt"])

    with pytest.raises(ValueError, match="unsupported reference file type"):
        audit_verabench_contamination(tmp_path, reference_paths=[unsupported])


def test_contamination_audit_reads_directory_json_jsonl_and_fallback_text(tmp_path):
    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    _write_sample_benchmark(benchmark_dir)

    references_dir = tmp_path / "references"
    references_dir.mkdir()
    nested_dir = references_dir / "nested"
    nested_dir.mkdir()
    (references_dir / "data.json").write_text(
        json.dumps(
            {
                "records": [
                    {"question": "该系统2024年的预算是多少？"},
                    {"answer": "500万元"},
                ],
                "ignored_number": 42,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (references_dir / "mixed.jsonl").write_text(
        "\n"
        + json.dumps({"text": "预算是500万元。"}, ensure_ascii=False)
        + "\n"
        + "not-json but still searchable 该系统2024年的预算是多少？\n",
        encoding="utf-8",
    )
    (nested_dir / "broken.json").write_text(
        "{not-json but includes 该系统2024年的预算是多少？}",
        encoding="utf-8",
    )
    (references_dir / "ignored.csv").write_text(
        "该系统2024年的预算是多少？",
        encoding="utf-8",
    )

    report = audit_verabench_contamination(
        benchmark_dir,
        reference_paths=[references_dir],
        max_matches=20,
    )

    assert report["coverage"]["reference_files"] == 3
    assert report["coverage"]["reference_texts"] >= 4
    source_ids = {
        fingerprint["source_id"]
        for fingerprint in report["reference_fingerprints"]
    }
    assert "data.json:$.records[0].question" in source_ids
    assert "mixed.jsonl:2:$.text" in source_ids
    assert "mixed.jsonl:3" in source_ids
    assert "broken.json" in source_ids
    assert report["summary"]["high_risk_exact_matches"] >= 1


def test_contamination_audit_distinguishes_near_match_basis(tmp_path):
    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    _write_sample_benchmark(benchmark_dir)
    exactish = tmp_path / "exactish.txt"
    exactish.write_text("该系统2024年的预算是多少？", encoding="utf-8")
    reference = tmp_path / "reference.txt"
    reference.write_text(
        "该系统2024年的预算是多少？" + "完全无关内容" * 80,
        encoding="utf-8",
    )

    exactish_report = audit_verabench_contamination(
        benchmark_dir,
        reference_paths=[exactish],
        near_threshold=0.5,
        containment_threshold=0.5,
        ngram_size=2,
    )
    embedded_report = audit_verabench_contamination(
        benchmark_dir,
        reference_paths=[reference],
        near_threshold=0.99,
        containment_threshold=0.5,
        ngram_size=2,
    )

    exactish_match = next(
        match for match in exactish_report["matches"]["near_duplicate"]
        if match["item_id"] == "T001" and match["kind"] == "question"
    )
    embedded_match = next(
        match for match in embedded_report["matches"]["near_duplicate"]
        if match["item_id"] == "T001" and match["kind"] == "question"
    )
    assert exactish_match["match_basis"] == "jaccard_and_item_containment"
    assert embedded_match["match_basis"] == "item_containment"
    assert _near_match_basis(
        0.8,
        0.2,
        near_threshold=0.7,
        containment_threshold=0.9,
    ) == "jaccard"


def test_best_reference_ngram_excerpt_handles_sparse_and_short_references():
    assert _best_reference_ngram_excerpt(
        {"abc"},
        "!!!",
        ngram_size=3,
        item_normalized_chars=3,
    ) == "!!!"
    assert _best_reference_ngram_excerpt(
        {"abcdef"},
        "abc",
        ngram_size=6,
        item_normalized_chars=6,
    ) == "abc"
    assert _best_reference_ngram_excerpt(
        {"zzz"},
        "abcdef",
        ngram_size=3,
        item_normalized_chars=3,
    ) == "abcdef"
    assert "abcde" in _best_reference_ngram_excerpt(
        {"abc", "bcd", "cde", "xyz"},
        "abcde " + ("filler " * 40) + "xyz",
        ngram_size=3,
        item_normalized_chars=5,
    )


def test_low_level_near_match_and_excerpt_edge_cases():
    assert _near_matches(
        [AuditText(item_id="punct", kind="question", text="!!!")],
        [ReferenceText(source_id="ref", path="ref.txt", text="anything", sha256="0")],
        near_threshold=0.0,
        containment_threshold=0.0,
        ngram_size=3,
    ) == []
    truncated = _excerpt("word " * 40, limit=20)
    assert truncated.endswith("…")
    assert len(truncated) == 20
