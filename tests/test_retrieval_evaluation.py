import json
import subprocess
import sys

import pytest

import experiments.evaluate_retrieval as retrieval_eval
from experiments.evaluate_retrieval import build_matrix_report, build_report


@pytest.fixture
def sample_bench_dir(tmp_path):
    corpus = [
        {
            "doc_id": "D001",
            "title": "测试文档1",
            "source": "report",
            "content": "verarag_budget_truth 2024年1月，某系统正式发布，预算为500万元。",
        },
        {
            "doc_id": "D002",
            "title": "测试文档2",
            "source": "report",
            "content": "2022年6月，该系统预算为300万元。",
        },
        {
            "doc_id": "D003",
            "title": "不实报道",
            "source": "blog",
            "content": "verarag_false_report 该系统预算为800万元。",
        },
    ]
    questions = [
        {
            "id": "T001",
            "type": "single_evidence",
            "question": "verarag_budget_truth 该系统2024年的预算是多少？",
            "ground_truth_answer": "500万元。",
            "expected_behavior": "answer_with_citation",
            "evidence": [
                {
                    "evidence_id": "E1",
                    "doc_id": "D001",
                    "text_span": "预算为500万元",
                    "category": "supporting",
                }
            ],
        },
        {
            "id": "T002",
            "type": "conflict",
            "question": "verarag_budget_truth 和 verarag_false_report 对该系统预算有什么冲突？",
            "ground_truth_answer": "根据2024年文档为500万元，但有不实报道称800万元。",
            "expected_behavior": "answer_with_conflict_note",
            "evidence": [
                {
                    "evidence_id": "E1",
                    "doc_id": "D001",
                    "text_span": "预算为500万元",
                    "category": "supporting",
                },
                {
                    "evidence_id": "E3",
                    "doc_id": "D003",
                    "text_span": "该系统预算为800万元",
                    "category": "conflicting",
                },
            ],
            "expected_conflicts": [
                {"pair": ["E1", "E3"], "conflict_type": "numeric_conflict"}
            ],
        },
        {
            "id": "T003",
            "type": "unanswerable",
            "question": "该系统用了什么编程语言？",
            "ground_truth_answer": "无法回答。文档中没有相关信息。",
            "expected_behavior": "abstain",
            "evidence": [],
        },
    ]
    (tmp_path / "corpus.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in corpus),
        encoding="utf-8",
    )
    (tmp_path / "questions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in questions),
        encoding="utf-8",
    )
    return str(tmp_path)


def test_build_retrieval_report_scores_sample_benchmark(sample_bench_dir):
    report = build_report(
        data_dir=sample_bench_dir,
        retriever_name="bm25",
        top_k=3,
    )

    assert report["schema_version"] == "retrieval-eval-v1"
    assert report["retriever"] == "bm25"
    assert report["selected_questions"] == 3
    assert report["evaluated_questions"] == 2
    assert report["summary"]["macro_recall"] == 1.0
    assert report["summary"]["micro_recall"] == 1.0
    assert "single_evidence" in report["by_type"]
    assert "conflict" in report["by_type"]
    assert all(row["top_k_policy"] == "fixed" for row in report["question_results"])


def test_complexity_adaptive_policy_records_selected_top_k(sample_bench_dir):
    report = build_report(
        data_dir=sample_bench_dir,
        retriever_name="bm25",
        top_k=10,
        top_k_policy="complexity_adaptive",
    )

    by_id = {row["question_id"]: row for row in report["question_results"]}
    assert by_id["T001"]["selected_top_k"] == 2
    assert by_id["T002"]["selected_top_k"] == 5
    assert report["top_k_policy"] == "complexity_adaptive"


def test_dense_retriever_variant_can_be_evaluated(sample_bench_dir, monkeypatch):
    class FakeDenseRetriever:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def index_documents(self, documents):
            self.documents = documents

        def retrieve(self, query, top_k=10):
            from src.retriever.base import RetrievalResult

            return [
                RetrievalResult(
                    doc_id=document["id"],
                    content=document["text"],
                    title=document.get("title", ""),
                    score=float(len(self.documents) - index),
                    metadata=document,
                )
                for index, document in enumerate(self.documents[:top_k])
            ]

    monkeypatch.setattr(retrieval_eval, "DenseRetriever", FakeDenseRetriever)

    report = build_report(
        data_dir=sample_bench_dir,
        retriever_name="dense",
        top_k=2,
    )

    assert report["retriever"] == "dense"
    assert report["evaluated_questions"] == 2
    assert report["summary"]["hit_rate"] == 1.0
    assert report["dense_model_name"] == "BAAI/bge-base-en-v1.5"
    assert report["dense_local_files_only"] is True


def test_build_matrix_report_compares_retriever_policy_grid(sample_bench_dir):
    report = build_matrix_report(
        data_dir=sample_bench_dir,
        retriever_names=["bm25"],
        top_k_values=[1, 3],
        top_k_policies=["fixed", "complexity_adaptive"],
    )

    assert report["schema_version"] == "retrieval-matrix-v1"
    assert len(report["variants"]) == 4
    assert {variant["status"] for variant in report["variants"]} == {"ok"}
    assert report["best_by_macro_f1"]["summary"]["macro_f1"] >= 0.0
    assert {
        (variant["top_k"], variant["top_k_policy"])
        for variant in report["variants"]
    } == {
        (1, "fixed"),
        (1, "complexity_adaptive"),
        (3, "fixed"),
        (3, "complexity_adaptive"),
    }


def test_evaluate_retrieval_cli_writes_json_with_sweep(sample_bench_dir, tmp_path):
    output = tmp_path / "retrieval.json"

    subprocess.run(
        [
            sys.executable,
            "experiments/evaluate_retrieval.py",
            "--data-dir",
            sample_bench_dir,
            "--retriever",
            "bm25",
            "--top-k",
            "3",
            "--sweep-top-k",
            "1",
            "3",
            "--output",
            str(output),
        ],
        check=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["evaluated_questions"] == 2
    assert payload["summary"]["hit_rate"] == 1.0
    assert [row["top_k"] for row in payload["sweep"]] == [1, 3]


def test_evaluate_retrieval_cli_writes_matrix_json(sample_bench_dir, tmp_path):
    output = tmp_path / "matrix.json"

    subprocess.run(
        [
            sys.executable,
            "experiments/evaluate_retrieval.py",
            "--data-dir",
            sample_bench_dir,
            "--matrix",
            "--matrix-retrievers",
            "bm25",
            "--matrix-top-k",
            "1",
            "3",
            "--matrix-policies",
            "fixed",
            "precision_cap",
            "--output",
            str(output),
        ],
        check=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "retrieval-matrix-v1"
    assert len(payload["variants"]) == 4
    assert payload["best_by_macro_f1"]["status"] == "ok"
