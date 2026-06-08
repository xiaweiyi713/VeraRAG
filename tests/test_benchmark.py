"""Tests for VeraBench benchmark module."""

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark.evaluator import QuestionResult, VeraBenchEvaluator
from src.benchmark.loader import (
    QUESTION_TYPES,
    BenchmarkQuestion,
    CorpusDocument,
    VeraBenchLoader,
    load_verabench,
)

# --- Fixtures ---

SAMPLE_CORPUS = [
    {
        "doc_id": "D001",
        "title": "测试文档1",
        "source": "report",
        "date": "2024-01-01",
        "author": "测试",
        "url": "",
        "content": "2024年1月，某系统正式发布，预算为500万元。",
        "entities": ["测试系统"],
        "tags": ["测试"],
    },
    {
        "doc_id": "D002",
        "title": "测试文档2（旧版）",
        "source": "report",
        "date": "2022-06-01",
        "author": "测试",
        "url": "",
        "content": "2022年6月，该系统预算为300万元。",
        "entities": ["测试系统"],
        "tags": ["测试"],
    },
    {
        "doc_id": "D003",
        "title": "不实报道",
        "source": "blog",
        "date": "2024-03-01",
        "author": "某媒体",
        "url": "",
        "content": "该系统预算为800万元。",
        "entities": ["测试系统"],
        "tags": ["测试", "不实"],
    },
]

SAMPLE_QUESTIONS = [
    {
        "id": "T001",
        "type": "single_evidence",
        "question": "该系统2024年的预算是多少？",
        "ground_truth_answer": "500万元。",
        "ground_truth_claims": [
            {"claim": "预算500万元", "status": "supported", "evidence_ids": ["E1"]}
        ],
        "evidence": [
            {"evidence_id": "E1", "doc_id": "D001", "text_span": "预算为500万元", "category": "supporting"}
        ],
        "expected_conflicts": [],
        "difficulty": "easy",
        "requires_multi_hop": False,
        "expected_behavior": "answer_with_citation",
        "tags": ["测试"],
    },
    {
        "id": "T002",
        "type": "conflict",
        "question": "该系统预算到底是多少？",
        "ground_truth_answer": "根据2024年文档为500万元，但有不实报道称800万元。",
        "ground_truth_claims": [
            {"claim": "预算500万元", "status": "supported", "evidence_ids": ["E1"]},
            {"claim": "报道称800万不准确", "status": "refuted", "evidence_ids": ["E3"]}
        ],
        "evidence": [
            {"evidence_id": "E1", "doc_id": "D001", "text_span": "预算为500万元", "category": "supporting"},
            {"evidence_id": "E3", "doc_id": "D003", "text_span": "该系统预算为800万元", "category": "conflicting"}
        ],
        "expected_conflicts": [
            {"pair": ["E1", "E3"], "conflict_type": "numeric_conflict"}
        ],
        "difficulty": "medium",
        "requires_multi_hop": False,
        "expected_behavior": "answer_with_conflict_note",
        "tags": ["测试", "冲突"],
    },
    {
        "id": "T003",
        "type": "unanswerable",
        "question": "该系统用了什么编程语言？",
        "ground_truth_answer": "无法回答。文档中没有相关信息。",
        "ground_truth_claims": [],
        "evidence": [],
        "expected_conflicts": [],
        "difficulty": "easy",
        "requires_multi_hop": False,
        "expected_behavior": "abstain",
        "tags": ["测试"],
    },
]


@pytest.fixture
def temp_bench_dir():
    """Create a temporary directory with sample benchmark data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_path = os.path.join(tmpdir, "corpus.jsonl")
        questions_path = os.path.join(tmpdir, "questions.jsonl")

        with open(corpus_path, "w", encoding="utf-8") as f:
            for doc in SAMPLE_CORPUS:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")

        with open(questions_path, "w", encoding="utf-8") as f:
            for q in SAMPLE_QUESTIONS:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")

        yield tmpdir


# --- Loader Tests ---

class TestCorpusDocument:
    def test_from_dict(self):
        doc = CorpusDocument.from_dict(SAMPLE_CORPUS[0])
        assert doc.doc_id == "D001"
        assert doc.title == "测试文档1"
        assert doc.source == "report"
        assert doc.date == "2024-01-01"
        assert doc.entities == ["测试系统"]

    def test_from_dict_minimal(self):
        d = {"doc_id": "D099", "title": "T", "source": "wiki", "content": "C"}
        doc = CorpusDocument.from_dict(d)
        assert doc.date is None
        assert doc.entities == []


class TestBenchmarkQuestion:
    def test_from_dict_full(self):
        q = BenchmarkQuestion.from_dict(SAMPLE_QUESTIONS[0])
        assert q.id == "T001"
        assert q.type == "single_evidence"
        assert len(q.ground_truth_claims) == 1
        assert len(q.evidence) == 1
        assert q.expected_behavior == "answer_with_citation"

    def test_from_dict_conflict(self):
        q = BenchmarkQuestion.from_dict(SAMPLE_QUESTIONS[1])
        assert q.type == "conflict"
        assert len(q.expected_conflicts) == 1
        assert q.expected_conflicts[0].conflict_type == "numeric_conflict"

    def test_from_dict_unanswerable(self):
        q = BenchmarkQuestion.from_dict(SAMPLE_QUESTIONS[2])
        assert q.type == "unanswerable"
        assert q.expected_behavior == "abstain"
        assert len(q.evidence) == 0


class TestVeraBenchLoader:
    def test_load_success(self, temp_bench_dir):
        loader = VeraBenchLoader(temp_bench_dir)
        bench = loader.load()
        assert len(bench.corpus) == 3
        assert len(bench.questions) == 3

    def test_load_missing_corpus(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Only create questions, no corpus
            with open(os.path.join(tmpdir, "questions.jsonl"), "w") as f:
                f.write("{}\n")
            loader = VeraBenchLoader(tmpdir)
            with pytest.raises(FileNotFoundError, match="Corpus not found"):
                loader.load()

    def test_load_package_data_fallback(self, monkeypatch, tmp_path):
        package_data = Path("src/benchmark/data/verabench").resolve()
        monkeypatch.setattr(VeraBenchLoader, "DEFAULT_PATH", tmp_path / "missing-verabench")
        monkeypatch.setattr(VeraBenchLoader, "PACKAGE_DATA_PATH", package_data)

        bench = VeraBenchLoader().load()

        assert len(bench.corpus) == 57
        assert len(bench.questions) == 152

    def test_validation_bad_type(self, temp_bench_dir):
        bad_q = {
            "id": "T999", "type": "invalid_type", "question": "?",
            "ground_truth_answer": "A", "expected_behavior": "answer_with_citation",
        }
        with open(os.path.join(temp_bench_dir, "questions.jsonl"), "a") as f:
            f.write(json.dumps(bad_q, ensure_ascii=False) + "\n")
        loader = VeraBenchLoader(temp_bench_dir)
        with pytest.raises(ValueError, match="unknown type"):
            loader.load()

    def test_validation_bad_doc_ref(self, temp_bench_dir):
        bad_q = {
            "id": "T998", "type": "single_evidence", "question": "?",
            "ground_truth_answer": "A", "expected_behavior": "answer_with_citation",
            "evidence": [{"evidence_id": "E1", "doc_id": "MISSING", "text_span": "x", "category": "supporting"}],
        }
        with open(os.path.join(temp_bench_dir, "questions.jsonl"), "a") as f:
            f.write(json.dumps(bad_q, ensure_ascii=False) + "\n")
        loader = VeraBenchLoader(temp_bench_dir)
        with pytest.raises(ValueError, match="missing doc"):
            loader.load()


class TestVeraBench:
    def test_stats(self, temp_bench_dir):
        bench = VeraBenchLoader(temp_bench_dir).load()
        stats = bench.stats()
        assert stats["total_documents"] == 3
        assert stats["total_questions"] == 3
        assert stats["questions_by_type"]["single_evidence"] == 1
        assert stats["questions_by_type"]["conflict"] == 1
        assert stats["questions_by_type"]["unanswerable"] == 1

    def test_get_by_type(self, temp_bench_dir):
        bench = VeraBenchLoader(temp_bench_dir).load()
        conflict_qs = bench.get_questions_by_type("conflict")
        assert len(conflict_qs) == 1
        assert conflict_qs[0].id == "T002"

    def test_get_document(self, temp_bench_dir):
        bench = VeraBenchLoader(temp_bench_dir).load()
        doc = bench.get_document("D001")
        assert doc is not None
        assert doc.title == "测试文档1"
        assert bench.get_document("MISSING") is None


class TestLoadVeraBench:
    def test_load_default(self):
        bench = load_verabench()
        assert bench.total_documents if hasattr(bench, 'total_documents') else len(bench.corpus) > 0
        stats = bench.stats()
        assert stats["total_questions"] >= 100
        for qtype in QUESTION_TYPES:
            assert qtype in stats["questions_by_type"]


# --- Evaluator Tests ---

class TestVeraBenchEvaluator:
    def test_demo_evaluate(self, temp_bench_dir):
        bench = VeraBenchLoader(temp_bench_dir).load()
        evaluator = VeraBenchEvaluator(benchmark=bench)
        report = evaluator.evaluate()
        assert report.total_questions == 3
        assert report.completed == 3
        assert report.errors == 0
        assert report.overall_answer_em == 1.0

    def test_evaluate_by_type(self, temp_bench_dir):
        bench = VeraBenchLoader(temp_bench_dir).load()
        evaluator = VeraBenchEvaluator(benchmark=bench)
        report = evaluator.evaluate(question_types=["conflict"])
        assert report.total_questions == 1
        assert report.results[0].question_id == "T002"

    def test_evaluate_max_questions(self, temp_bench_dir):
        bench = VeraBenchLoader(temp_bench_dir).load()
        evaluator = VeraBenchEvaluator(benchmark=bench)
        report = evaluator.evaluate(max_questions=2)
        assert report.total_questions == 2

    def test_baseline_evaluate(self, temp_bench_dir):
        bench = VeraBenchLoader(temp_bench_dir).load()
        evaluator = VeraBenchEvaluator(benchmark=bench)

        def dummy_baseline(question):
            return "我不知道"

        report = evaluator.evaluate_baseline(dummy_baseline)
        assert report.total_questions == 3
        assert report.overall_answer_em == 0.0

    def test_behavior_classification_abstain(self, temp_bench_dir):
        bench = VeraBenchLoader(temp_bench_dir).load()
        evaluator = VeraBenchEvaluator(benchmark=bench)
        q = bench.questions[2]  # unanswerable
        assert evaluator._classify_behavior(q, "无法回答") == "abstain"
        assert evaluator._classify_behavior(q, "语料库中没有相关信息") == "abstain"

    def test_behavior_classification_conflict(self, temp_bench_dir):
        bench = VeraBenchLoader(temp_bench_dir).load()
        evaluator = VeraBenchEvaluator(benchmark=bench)
        q = bench.questions[1]  # conflict with expected_conflicts
        assert evaluator._classify_behavior(q, "有矛盾的信息") == "answer_with_conflict_note"

    def test_report_to_dict(self, temp_bench_dir):
        bench = VeraBenchLoader(temp_bench_dir).load()
        evaluator = VeraBenchEvaluator(benchmark=bench)
        report = evaluator.evaluate()
        d = report.to_dict()
        assert "total_questions" in d
        assert "by_type" in d
        assert "behavior_confusion" in d
        assert "failure_summary" in d
        assert d["overall_answer_em"] == 1.0

    def test_report_failure_diagnostics(self, temp_bench_dir):
        bench = VeraBenchLoader(temp_bench_dir).load()
        evaluator = VeraBenchEvaluator(benchmark=bench)
        rows = [
            QuestionResult(
                question_id="T001", question_type="conflict",
                question="?", ground_truth="A", predicted="A",
                expected_behavior="answer_with_conflict_note",
                actual_behavior="answer_with_citation", correct=False,
                evidence_recall=0.25, conflict_detection_f1=0.0,
            ),
            QuestionResult(
                question_id="T002", question_type="single_evidence",
                question="?", ground_truth="A", predicted="A",
                expected_behavior="answer_with_citation",
                actual_behavior="answer_with_citation", correct=True,
                evidence_recall=1.0, conflict_detection_f1=0.0,
            ),
        ]
        report = evaluator._build_report(rows)
        assert report.behavior_confusion["answer_with_conflict_note"]["answer_with_citation"] == 1
        assert report.failure_summary["behavior_failure_count"] == 1
        assert report.failure_summary["low_evidence_recall_count"] == 1
        assert report.failure_summary["conflict_failure_count"] == 1
        assert report.conflict_summary["gold_conflicts"] == 0
        assert report.conflict_summary["available"] is True
        assert len(report.calibration_bins) == 10

    def test_report_conflict_counts(self, temp_bench_dir):
        bench = VeraBenchLoader(temp_bench_dir).load()
        evaluator = VeraBenchEvaluator(benchmark=bench)
        rows = [
            QuestionResult(
                question_id="T001", question_type="conflict",
                question="?", ground_truth="A", predicted="A",
                expected_behavior="answer_with_conflict_note",
                actual_behavior="answer_with_conflict_note", correct=True,
                predicted_conflicts=3, gold_conflicts=1,
                conflict_true_positives=1,
                conflict_false_positives=2,
                conflict_false_negatives=0,
                confidence=0.8,
            ),
            QuestionResult(
                question_id="T002", question_type="conflict",
                question="?", ground_truth="A", predicted="A",
                expected_behavior="answer_with_conflict_note",
                actual_behavior="answer_with_citation", correct=False,
                predicted_conflicts=0, gold_conflicts=1,
                conflict_true_positives=0,
                conflict_false_positives=0,
                conflict_false_negatives=1,
                confidence=0.2,
            ),
        ]
        report = evaluator._build_report(rows)
        assert report.conflict_summary["gold_conflicts"] == 2
        assert report.conflict_summary["predicted_conflicts"] == 3
        assert report.conflict_summary["true_positives"] == 1
        assert report.conflict_summary["false_positives"] == 2
        assert report.conflict_summary["false_negatives"] == 1
        assert report.conflict_summary["dominant_failure"] == "mixed"

    def test_conflict_scoring_ignores_support_edges(self, temp_bench_dir):
        bench = VeraBenchLoader(temp_bench_dir).load()
        evaluator = VeraBenchEvaluator(benchmark=bench)
        q = bench.questions[1]  # gold conflict pair E1-E3

        output = SimpleNamespace(
            answer=q.ground_truth_answer,
            confidence=0.8,
            evidence=[],
            verification_report=None,
            conflict_report={
                "nodes": [
                    {"node_id": "C1", "evidence_ids": ["D001_c0"]},
                    {"node_id": "C2", "evidence_ids": ["D003_c0"]},
                    {"node_id": "C3", "evidence_ids": ["D002_c0"]},
                ],
                "edges": [
                    {
                        "source_id": "C1",
                        "target_id": "C3",
                        "conflict_type": "support",
                    },
                    {
                        "source_id": "C1",
                        "target_id": "C2",
                        "conflict_type": "numeric_conflict",
                    },
                ],
            },
        )

        result = evaluator._score_pipeline_output(q, output, latency=0.0)
        assert result.predicted_conflicts == 1
        assert result.gold_conflicts == 1
        assert result.conflict_true_positives == 1
        assert result.conflict_false_positives == 0
        assert result.conflict_false_negatives == 0
        assert result.conflict_detection_f1 == 1.0

    def test_offline_result_analysis(self):
        from experiments.analyze_verabench_results import analyze

        report = {
            "total_questions": 2,
            "behavior_accuracy": 0.5,
            "overall_answer_f1": 0.4,
            "overall_evidence_recall": 0.6,
            "overall_conflict_f1": 0.0,
            "question_results": [
                {
                    "question_id": "T001",
                    "question_type": "unanswerable",
                    "question": "Q1",
                    "expected_behavior": "abstain",
                    "actual_behavior": "answer_with_citation",
                    "answer_f1": 0.0,
                    "evidence_recall": 0.0,
                    "conflict_detection_f1": 0.0,
                    "confidence": 0.1,
                    "predicted_conflicts": 0,
                    "gold_conflicts": 0,
                },
                {
                    "question_id": "T002",
                    "question_type": "single_evidence",
                    "question": "Q2",
                    "expected_behavior": "answer_with_citation",
                    "actual_behavior": "answer_with_citation",
                    "answer_f1": 0.8,
                    "evidence_recall": 1.0,
                    "conflict_detection_f1": 0.0,
                    "confidence": 0.9,
                    "predicted_conflicts": 2,
                    "gold_conflicts": 1,
                    "conflict_true_positives": 1,
                    "conflict_false_positives": 1,
                },
            ],
        }
        analysis = analyze(report)
        assert analysis["behavior_confusion"]["abstain"]["answer_with_citation"] == 1
        assert analysis["failure_summary"]["behavior_failure_count"] == 1
        assert analysis["failure_summary"]["behavior_failures_by_type"]["unanswerable"] == 1
        assert analysis["conflict_summary"]["predicted_conflicts"] == 2
        assert analysis["conflict_summary"]["false_positives"] == 1
        assert analysis["conflict_summary"]["available"] is True
        assert len(analysis["calibration_bins"]) == 10

    def test_offline_result_analysis_marks_legacy_conflict_counts_unavailable(self):
        from experiments.analyze_verabench_results import analyze

        report = {
            "question_results": [
                {
                    "question_id": "T001",
                    "question_type": "conflict",
                    "expected_behavior": "answer_with_conflict_note",
                    "actual_behavior": "answer_with_citation",
                    "correct": False,
                    "confidence": 0.2,
                }
            ]
        }
        analysis = analyze(report)
        assert analysis["conflict_summary"]["available"] is False

    def test_calibration_curve_loads_report_json(self, tmp_path):
        from experiments.calibration_curve import load_confidence_rows

        path = tmp_path / "report.json"
        path.write_text(json.dumps({
            "question_results": [
                {"confidence": 0.2, "correct": False},
                {"confidence": 0.8, "correct": True},
            ]
        }), encoding="utf-8")
        predicted, actual = load_confidence_rows(str(path))
        assert predicted.tolist() == [0.2, 0.8]
        assert actual.tolist() == [0.0, 1.0]


class TestQuestionResult:
    def test_defaults(self):
        r = QuestionResult(
            question_id="T001", question_type="single_evidence",
            question="?", ground_truth="A", predicted="A",
            expected_behavior="answer_with_citation",
            actual_behavior="answer_with_citation", correct=True,
        )
        assert r.answer_em == 0.0
        assert r.confidence == 0.0
        assert r.error is None
        assert r.difficulty == "medium"
