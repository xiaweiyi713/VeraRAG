"""Tests for VeraBench benchmark module."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.migrate_verabench_ontology_v11 import migrate_question
from scripts.migrate_verabench_span_traceability_v112 import (
    SPAN_UPDATES,
)
from scripts.migrate_verabench_span_traceability_v112 import (
    migrate_question as migrate_span_traceability,
)
from src.benchmark.evaluator import QuestionResult, VeraBenchEvaluator
from src.benchmark.loader import (
    QUESTION_TYPES,
    BenchmarkQuestion,
    CorpusDocument,
    VeraBenchLoader,
    evidence_dependency_groups,
    evidence_span_match_kind,
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

    def test_from_dict_annotation_rationale(self):
        raw = {
            **SAMPLE_QUESTIONS[0],
            "annotation_rationale": "Direct factual question with one supporting source.",
            "difficulty_rationale": "One supporting evidence item and no conflict.",
        }
        q = BenchmarkQuestion.from_dict(raw)
        assert q.annotation_rationale == "Direct factual question with one supporting source."
        assert q.difficulty_rationale == "One supporting evidence item and no conflict."


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

    def test_validation_conflict_requires_gold_pair(self, temp_bench_dir):
        bad_q = {
            **SAMPLE_QUESTIONS[1],
            "id": "T997",
            "expected_conflicts": [],
        }
        with open(os.path.join(temp_bench_dir, "questions.jsonl"), "a") as f:
            f.write(json.dumps(bad_q, ensure_ascii=False) + "\n")

        with pytest.raises(ValueError, match="conflict question has no expected conflicts"):
            VeraBenchLoader(temp_bench_dir).load()

    def test_validation_conflict_pair_must_reference_question_evidence(self, temp_bench_dir):
        bad_q = {
            **SAMPLE_QUESTIONS[1],
            "id": "T996",
            "expected_conflicts": [
                {"pair": ["E1", "MISSING"], "conflict_type": "numeric_conflict"}
            ],
        }
        with open(os.path.join(temp_bench_dir, "questions.jsonl"), "a") as f:
            f.write(json.dumps(bad_q, ensure_ascii=False) + "\n")

        with pytest.raises(ValueError, match="unknown evidence ids"):
            VeraBenchLoader(temp_bench_dir).load()

    def test_validation_type_behavior_alignment(self, temp_bench_dir):
        bad_q = {
            **SAMPLE_QUESTIONS[0],
            "id": "T995",
            "expected_behavior": "correct_premise",
        }
        with open(os.path.join(temp_bench_dir, "questions.jsonl"), "a") as f:
            f.write(json.dumps(bad_q, ensure_ascii=False) + "\n")

        with pytest.raises(ValueError, match="requires behavior 'answer_with_citation'"):
            VeraBenchLoader(temp_bench_dir).load()

    def test_validation_duplicate_question_id(self, temp_bench_dir):
        with open(os.path.join(temp_bench_dir, "questions.jsonl"), "a") as f:
            f.write(json.dumps(SAMPLE_QUESTIONS[0], ensure_ascii=False) + "\n")

        with pytest.raises(ValueError, match="duplicate question id"):
            VeraBenchLoader(temp_bench_dir).load()

    def test_validation_duplicate_corpus_id(self, temp_bench_dir):
        with open(os.path.join(temp_bench_dir, "corpus.jsonl"), "a") as f:
            f.write(json.dumps(SAMPLE_CORPUS[0], ensure_ascii=False) + "\n")

        with pytest.raises(ValueError, match="duplicate corpus id"):
            VeraBenchLoader(temp_bench_dir).load()


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

    def test_v11_ontology_distribution(self):
        bench = load_verabench()
        stats = bench.stats()

        assert stats["total_questions"] == 152
        assert stats["questions_by_type"]["conflict"] == 11
        assert stats["questions_by_type"]["misleading"] == 37
        assert stats["conflict_count"] == 13

    def test_v11_migration_is_idempotent(self):
        benchmark = load_verabench()
        audited_ids = {
            "V017", "V067", "V068", "V069", "V070", "V071", "V072", "V074",
            "V120", "V121", "V122", "V123", "V124", "V125", "V126", "V127",
        }

        for question in benchmark.questions:
            if question.id not in audited_ids:
                continue
            raw = json.loads(
                next(
                    line
                    for line in Path("data/verabench/questions.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                    if json.loads(line)["id"] == question.id
                )
            )
            assert migrate_question(raw) == raw
            assert question.annotation_rationale

    def test_v112_span_migration_is_idempotent(self):
        assert len(SPAN_UPDATES) == 25
        for line in Path("data/verabench/questions.jsonl").read_text(
            encoding="utf-8",
        ).splitlines():
            raw = json.loads(line)
            assert migrate_span_traceability(raw) == raw

    def test_v11_audit_report(self):
        from experiments.validate_verabench import build_audit_report

        report = build_audit_report(
            Path("data/verabench"),
            Path("src/benchmark/data/verabench"),
        )

        assert report["valid"] is True
        assert report["version"] == "1.1.2"
        assert report["package_data_in_sync"] is True
        assert report["total_questions"] == 152
        assert report["expected_conflict_pairs"] == 15
        assert report["self_conflict_pairs"] == 9
        assert report["annotated_ontology_corrections"] == 17
        assert report["annotated_difficulty_corrections"] == 8
        assert report["evidence_traceability"] == {
            "total": 208,
            "exact": 129,
            "segmented": 79,
            "untraceable": 0,
        }
        assert report["evidence_dependency_groups"]["count"] == 27
        assert report["evidence_dependency_groups"]["largest_size"] == 19
        assert len(report["near_duplicate_question_pairs"]["pairs"]) == 4
        assert len(report["fingerprints"]["questions_sha256"]) == 64

    def test_validator_cli_works_outside_project_root(self, tmp_path):
        output = tmp_path / "audit.json"
        script = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "validate_verabench.py"
        )

        subprocess.run(
            [sys.executable, str(script), "--output", str(output)],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )

        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["version"] == "1.1.2"
        assert report["package_data_in_sync"] is True
        assert report["evidence_traceability"]["untraceable"] == 0

    def test_evidence_span_traceability_and_dependency_groups(self):
        assert evidence_span_match_kind(
            "预算为500万元",
            SAMPLE_CORPUS[0]["content"],
        ) == "exact"
        assert evidence_span_match_kind(
            "2024年1月...预算为500万元",
            SAMPLE_CORPUS[0]["content"],
        ) == "segmented"
        assert evidence_span_match_kind(
            "预算为800万元",
            SAMPLE_CORPUS[0]["content"],
        ) == "untraceable"

        questions = [
            BenchmarkQuestion.from_dict(question)
            for question in SAMPLE_QUESTIONS
        ]
        groups = evidence_dependency_groups(questions)
        assert groups["T001"] == groups["T002"]
        assert groups["T003"] != groups["T001"]

    def test_loader_rejects_untraceable_evidence_span(self, tmp_path):
        corpus_path = tmp_path / "corpus.jsonl"
        questions_path = tmp_path / "questions.jsonl"
        corpus_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n"
                for row in SAMPLE_CORPUS
            ),
            encoding="utf-8",
        )
        broken = [dict(question) for question in SAMPLE_QUESTIONS]
        broken[0] = {
            **broken[0],
            "evidence": [{
                **broken[0]["evidence"][0],
                "text_span": "语料中不存在的证据",
            }],
        }
        questions_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n"
                for row in broken
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="not traceable"):
            VeraBenchLoader(str(tmp_path)).load()

    def test_run_signature_includes_sorted_question_ids(self):
        from experiments.run_verabench import _build_run_signature

        benchmark_metadata = {
            "version": "1.1",
            "fingerprints": {
                "corpus_sha256": "c",
                "questions_sha256": "q",
            },
        }

        signature = _build_run_signature(
            benchmark_metadata=benchmark_metadata,
            config_path=None,
            question_types=["misleading", "unanswerable"],
            question_ids=["V048", "V036"],
            max_questions=None,
        )

        assert signature["question_ids"] == ["V036", "V048"]
        assert signature["question_types"] == ["misleading", "unanswerable"]

    def test_unknown_question_ids_are_sorted_and_deduped(self):
        from experiments.run_verabench import _unknown_question_ids

        assert _unknown_question_ids(["V999", "V001", "V999"], {"V001", "V002"}) == ["V999"]
        assert _unknown_question_ids(None, {"V001"}) == []

    def test_checkpoint_signature_rejects_stale_results(self, tmp_path):
        from experiments.run_verabench import _prepare_checkpoint

        checkpoint = tmp_path / "run.ckpt.jsonl"
        signature = {"schema_version": 1, "benchmark_questions_sha256": "a"}

        assert _prepare_checkpoint(str(checkpoint), signature, restart=False) is False
        checkpoint.write_text('{"question_id":"T001"}\n', encoding="utf-8")
        assert _prepare_checkpoint(str(checkpoint), signature, restart=False) is True

        with pytest.raises(ValueError, match="signature mismatch"):
            _prepare_checkpoint(
                str(checkpoint),
                {"schema_version": 1, "benchmark_questions_sha256": "b"},
                restart=False,
            )

    def test_checkpoint_signature_requires_metadata(self, tmp_path):
        from experiments.run_verabench import _prepare_checkpoint

        checkpoint = tmp_path / "legacy.ckpt.jsonl"
        checkpoint.write_text('{"question_id":"T001"}\n', encoding="utf-8")

        with pytest.raises(ValueError, match="metadata missing"):
            _prepare_checkpoint(
                str(checkpoint),
                {"schema_version": 1},
                restart=False,
            )

    def test_checkpoint_restart_replaces_stale_signature(self, tmp_path):
        from experiments.run_verabench import _prepare_checkpoint

        checkpoint = tmp_path / "run.ckpt.jsonl"
        checkpoint.write_text('{"question_id":"T001"}\n', encoding="utf-8")
        metadata = Path(f"{checkpoint}.meta.json")
        metadata.write_text('{"schema_version":0}\n', encoding="utf-8")

        signature = {"schema_version": 1}
        assert _prepare_checkpoint(str(checkpoint), signature, restart=True) is False
        assert not checkpoint.exists()
        assert json.loads(metadata.read_text(encoding="utf-8")) == signature


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
        assert report.overall_answer_f1 == 1.0
        assert report.overall_evidence_recall == 1.0
        assert report.overall_evidence_precision == 1.0
        assert report.overall_conflict_f1 == 1.0
        assert report.behavior_accuracy == 1.0
        assert report.avg_confidence == 1.0
        assert report.ece == 0.0
        assert report.brier_score == 0.0
        assert report.confidence_intervals["method"] == (
            "stratified-question-bootstrap-v1"
        )
        assert report.confidence_intervals["resamples"] == 2000
        assert report.confidence_intervals["metrics"]["answer_f1"] == {
            "estimate": 1.0,
            "lower": 1.0,
            "upper": 1.0,
        }
        assert report.confidence_intervals["metrics"]["conflict_micro_f1"] == {
            "estimate": 1.0,
            "lower": 1.0,
            "upper": 1.0,
        }
        robust = report.dependency_robust_confidence_intervals
        assert robust["method"] == "evidence-cluster-bootstrap-v1"
        assert robust["clusters"] == 2
        assert robust["cluster_sizes"] == [2, 1]
        assert robust["metrics"]["answer_f1"]["lower"] == 1.0
        assert report.conflict_summary["true_positives"] == 1
        assert report.conflict_summary["false_positives"] == 0
        assert report.conflict_summary["false_negatives"] == 0
        assert all(
            row.diagnostics["evaluation_mode"] == "demo_gold_identity"
            for row in report.results
        )

    def test_cli_requires_explicit_evaluation_mode(self):
        from experiments.run_verabench import _build_parser

        parser = _build_parser()

        with pytest.raises(SystemExit):
            parser.parse_args([])
        with pytest.raises(SystemExit):
            parser.parse_args(["--demo", "--config", "configs/model.yaml"])

        assert parser.parse_args(["--demo"]).demo is True
        assert parser.parse_args(["--config", "configs/model.yaml"]).config == (
            "configs/model.yaml"
        )

    def test_cli_max_requires_positive_integer(self):
        from experiments.run_verabench import _build_parser

        parser = _build_parser()

        with pytest.raises(SystemExit):
            parser.parse_args(["--demo", "--max", "0"])
        with pytest.raises(SystemExit):
            parser.parse_args(["--demo", "--max", "-1"])

        assert parser.parse_args(["--demo", "--max", "3"]).max == 3

    def test_config_run_metadata_reads_llm_settings(self, tmp_path):
        from experiments.run_verabench import _read_config_run_metadata

        config = tmp_path / "model.yaml"
        config.write_text(
            "\n".join(
                [
                    "llm:",
                    "  provider: deepseek",
                    "  model: deepseek-v4-flash",
                    "  temperature: 0",
                    "  max_tokens: 4000",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        metadata = _read_config_run_metadata(str(config))

        assert metadata == {
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "temperature": 0,
            "max_tokens": 4000,
        }

    def test_config_run_metadata_records_read_failure(self, tmp_path):
        from experiments.run_verabench import _read_config_run_metadata

        missing = tmp_path / "missing.yaml"

        metadata = _read_config_run_metadata(str(missing))

        assert set(metadata) == {"config_metadata_warning"}
        assert "failed to read config metadata" in metadata["config_metadata_warning"]
        assert str(missing) in metadata["config_metadata_warning"]

    def test_evaluate_by_type(self, temp_bench_dir):
        bench = VeraBenchLoader(temp_bench_dir).load()
        evaluator = VeraBenchEvaluator(benchmark=bench)
        report = evaluator.evaluate(question_types=["conflict"])
        assert report.total_questions == 1
        assert report.results[0].question_id == "T002"

    def test_evaluate_by_question_ids(self, temp_bench_dir):
        bench = VeraBenchLoader(temp_bench_dir).load()
        evaluator = VeraBenchEvaluator(benchmark=bench)

        report = evaluator.evaluate(question_ids=["T003", "T001"])

        assert report.total_questions == 2
        assert [result.question_id for result in report.results] == ["T001", "T003"]

    def test_evaluate_by_question_ids_rejects_unknown(self, temp_bench_dir):
        bench = VeraBenchLoader(temp_bench_dir).load()
        evaluator = VeraBenchEvaluator(benchmark=bench)

        with pytest.raises(ValueError, match="Unknown VeraBench question id"):
            evaluator.evaluate(question_ids=["T404"])

    def test_evaluate_max_questions(self, temp_bench_dir):
        bench = VeraBenchLoader(temp_bench_dir).load()
        evaluator = VeraBenchEvaluator(benchmark=bench)
        report = evaluator.evaluate(max_questions=2)
        assert report.total_questions == 2

    def test_checkpoint_reuse_skips_stale_and_out_of_scope_rows(self, temp_bench_dir, tmp_path, caplog):
        bench = VeraBenchLoader(temp_bench_dir).load()
        evaluator = VeraBenchEvaluator(benchmark=bench)
        checkpoint = tmp_path / "checkpoint.jsonl"
        q1, q2, q3 = bench.questions
        valid = QuestionResult(
            question_id=q1.id,
            question_type=q1.type,
            question=q1.question,
            ground_truth=q1.ground_truth_answer,
            predicted="cached answer",
            expected_behavior=q1.expected_behavior,
            actual_behavior="answer_with_citation",
            correct=False,
        )
        stale = QuestionResult(
            question_id=q2.id,
            question_type=q2.type,
            question="old question text",
            ground_truth=q2.ground_truth_answer,
            predicted="stale answer",
            expected_behavior=q2.expected_behavior,
            actual_behavior="answer_with_conflict_note",
            correct=True,
        )
        out_of_scope = QuestionResult(
            question_id=q3.id,
            question_type=q3.type,
            question=q3.question,
            ground_truth=q3.ground_truth_answer,
            predicted="out of scope answer",
            expected_behavior=q3.expected_behavior,
            actual_behavior="abstain",
            correct=True,
        )
        for row in (valid, stale, out_of_scope):
            evaluator._append_checkpoint(str(checkpoint), row)
        checkpoint.write_text(
            checkpoint.read_text(encoding="utf-8") + "{bad json\n",
            encoding="utf-8",
        )

        with caplog.at_level("WARNING", logger="verabench"):
            report = evaluator.evaluate(
                question_ids=[q1.id, q2.id],
                checkpoint_path=str(checkpoint),
            )

        assert [result.question_id for result in report.results] == [q1.id, q2.id]
        assert report.results[0].predicted == "cached answer"
        assert report.results[1].predicted == q2.ground_truth_answer
        assert "Skipping stale checkpoint result" in caplog.text
        assert "outside current run" in caplog.text
        assert "Skipping malformed checkpoint line" in caplog.text

    def test_pipeline_output_confidence_is_coerced(self, temp_bench_dir):
        bench = VeraBenchLoader(temp_bench_dir).load()
        evaluator = VeraBenchEvaluator(benchmark=bench)
        q = bench.questions[0]

        output = SimpleNamespace(
            answer=q.ground_truth_answer,
            confidence="1.7",
            evidence=[],
            verification_report=None,
            conflict_report={"nodes": [], "edges": []},
            metadata={},
        )

        result = evaluator._score_pipeline_output(q, output, latency=0.0)
        assert result.confidence == 1.0

        output.confidence = float("nan")
        result = evaluator._score_pipeline_output(q, output, latency=0.0)
        assert result.confidence == 0.0

        output.confidence = -0.25
        result = evaluator._score_pipeline_output(q, output, latency=0.0)
        assert result.confidence == 0.0

    def test_report_coerces_loaded_result_confidence_before_calibration(self, temp_bench_dir):
        bench = VeraBenchLoader(temp_bench_dir).load()
        evaluator = VeraBenchEvaluator(benchmark=bench)
        rows = [
            QuestionResult(
                question_id="T001",
                question_type="single_evidence",
                question="?",
                ground_truth="A",
                predicted="A",
                expected_behavior="answer_with_citation",
                actual_behavior="answer_with_citation",
                correct=True,
                confidence="0.75",  # type: ignore[arg-type]
            ),
            QuestionResult(
                question_id="T002",
                question_type="conflict",
                question="?",
                ground_truth="A",
                predicted="A",
                expected_behavior="answer_with_conflict_note",
                actual_behavior="answer_with_conflict_note",
                correct=False,
                confidence=float("inf"),
            ),
        ]

        report = evaluator._build_report(rows)

        assert [row.confidence for row in report.results] == [0.75, 0.0]
        assert report.avg_confidence == 0.375
        assert report.calibration_bins

    def test_pipeline_factory_errors_are_reported_per_question(self, temp_bench_dir):
        bench = VeraBenchLoader(temp_bench_dir).load()
        evaluator = VeraBenchEvaluator(
            benchmark=bench,
            pipeline_factory=lambda: (_ for _ in ()).throw(RuntimeError("pipeline unavailable")),
        )

        report = evaluator.evaluate(max_questions=1)

        assert report.total_questions == 1
        assert report.completed == 0
        assert report.errors == 1
        assert report.results[0].actual_behavior == "error"
        assert "pipeline unavailable" in (report.results[0].error or "")

    def test_baseline_errors_are_reported_per_question(self, temp_bench_dir):
        bench = VeraBenchLoader(temp_bench_dir).load()
        evaluator = VeraBenchEvaluator(benchmark=bench)

        report = evaluator.evaluate_baseline(
            lambda _question: (_ for _ in ()).throw(RuntimeError("baseline failed")),
            max_questions=1,
        )

        assert report.total_questions == 1
        assert report.completed == 0
        assert report.errors == 1
        assert report.results[0].actual_behavior == "error"
        assert "baseline failed" in (report.results[0].error or "")

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
        assert evaluator._classify_behavior(
            q,
            "根据现有证据，无法回答该问题。",
        ) == "abstain"
        assert evaluator._classify_behavior(
            q,
            "证据存在冲突。综合判断，根据现有证据，无法回答该问题。",
        ) == "abstain"
        assert evaluator._classify_behavior(
            q,
            "该办法于2023年施行；其他实施细节因证据不足无法提供。",
        ) == "answer_with_citation"

    def test_behavior_classification_conflict(self, temp_bench_dir):
        bench = VeraBenchLoader(temp_bench_dir).load()
        evaluator = VeraBenchEvaluator(benchmark=bench)
        q = bench.questions[1]  # conflict with expected_conflicts
        assert evaluator._classify_behavior(q, "有矛盾的信息") == "answer_with_conflict_note"
        assert evaluator._classify_behavior(
            q,
            "两条证据存在口径冲突，因此无法直接比较单一销量口径。",
        ) == "answer_with_conflict_note"
        assert evaluator._classify_behavior(
            q,
            "3nm已不再代表实际物理尺寸，而更多是商业命名。",
        ) == "answer_with_conflict_note"

    def test_behavior_classification_misleading_prefers_premise_correction(self, temp_bench_dir):
        bench = VeraBenchLoader(temp_bench_dir).load()
        q = bench.questions[1]
        q.type = "misleading"
        q.expected_behavior = "correct_premise"
        evaluator = VeraBenchEvaluator(benchmark=bench)

        assert evaluator._classify_behavior(
            q,
            "该说法不准确，证据中存在冲突，应纠正这个前提。",
        ) == "correct_premise"
        assert evaluator._classify_behavior(
            q,
            "尚未证实，不能证明该说法成立。",
        ) == "correct_premise"
        assert evaluator._classify_behavior(
            q,
            "该说法不完全准确，不能完全替代。",
        ) == "correct_premise"

    def test_report_to_dict(self, temp_bench_dir):
        bench = VeraBenchLoader(temp_bench_dir).load()
        evaluator = VeraBenchEvaluator(benchmark=bench)
        report = evaluator.evaluate()
        d = report.to_dict()
        assert "total_questions" in d
        assert "by_type" in d
        assert "behavior_confusion" in d
        assert "failure_summary" in d
        assert "confidence_intervals" in d
        assert "dependency_robust_confidence_intervals" in d
        assert d["question_results"][0]["dependency_group"]
        assert d["question_results"][0]["gold_document_ids"] == ["D001"]
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
        assert report.failure_summary["conflict_failures_by_type"]["conflict"] == 1
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
        assert report.conflict_summary["f1"] == 0.4
        assert report.overall_conflict_f1 == 0.4
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

    def test_conflict_scoring_ignores_non_gold_evidence_pairs(self, temp_bench_dir):
        bench = VeraBenchLoader(temp_bench_dir).load()
        evaluator = VeraBenchEvaluator(benchmark=bench)
        q = bench.questions[0]  # only E1 is gold evidence

        output = SimpleNamespace(
            answer=q.ground_truth_answer,
            confidence=0.8,
            evidence=[],
            verification_report=None,
            conflict_report={
                "nodes": [
                    {"node_id": "C1", "evidence_ids": ["D001_c0"]},
                    {"node_id": "C2", "evidence_ids": ["D003_c0"]},
                ],
                "edges": [
                    {
                        "source_id": "C1",
                        "target_id": "C2",
                        "conflict_type": "numeric_conflict",
                    },
                ],
            },
        )

        result = evaluator._score_pipeline_output(q, output, latency=0.0)

        assert result.predicted_conflicts == 0
        assert result.conflict_false_positives == 0
        assert result.diagnostics["predicted_conflict_pairs"] == ["D003_c0-E1"]
        assert result.diagnostics["scored_predicted_conflict_pairs"] == []
        assert result.diagnostics["unscored_extraneous_conflict_pairs"] == ["D003_c0-E1"]

    def test_conflict_scoring_ignores_unannotated_self_pairs(self, temp_bench_dir):
        bench = VeraBenchLoader(temp_bench_dir).load()
        evaluator = VeraBenchEvaluator(benchmark=bench)
        q = bench.questions[0]  # E1 is gold evidence, but no self-conflict is annotated

        output = SimpleNamespace(
            answer=q.ground_truth_answer,
            confidence=0.8,
            evidence=[],
            verification_report=None,
            conflict_report={
                "nodes": [
                    {"node_id": "C1", "evidence_ids": ["D001_c0"]},
                ],
                "edges": [
                    {
                        "source_id": "C1",
                        "target_id": "C1",
                        "conflict_type": "numeric_conflict",
                    },
                ],
            },
        )

        result = evaluator._score_pipeline_output(q, output, latency=0.0)

        assert result.predicted_conflicts == 0
        assert result.conflict_false_positives == 0
        assert result.diagnostics["scored_predicted_conflict_pairs"] == []
        assert result.diagnostics["unscored_extraneous_conflict_pairs"] == ["E1-E1"]

    def test_premise_refutation_scored_separately_from_conflict_edges(self, temp_bench_dir):
        bench = VeraBenchLoader(temp_bench_dir).load()
        evaluator = VeraBenchEvaluator(benchmark=bench)
        q = BenchmarkQuestion(
            id="P001",
            type="conflict",
            question="AI医疗诊断的准确率已经达到98%，这是否意味着可以取代医生了？",
            ground_truth_answer="不意味着可以取代医生。",
            expected_behavior="answer_with_conflict_note",
            expected_conflicts=[],
            difficulty="hard",
        )
        output = SimpleNamespace(
            answer="这并不意味着AI可以取代医生，AI应作为辅助工具而非替代工具使用。",
            confidence=0.8,
            evidence=[],
            verification_report=None,
            conflict_report={"nodes": [], "edges": []},
            metadata={},
        )

        result = evaluator._score_pipeline_output(q, output, latency=0.0)

        assert result.predicted_conflicts == 0
        assert result.gold_conflicts == 0
        assert result.premise_refutation_expected is True
        assert result.premise_refutation_detected is True
        assert result.premise_refutation_correct is True
        assert result.diagnostics["premise_refutation_expected"] is True
        assert result.diagnostics["premise_refutation_detected"] is True

    def test_premise_refutation_handles_yes_no_scope_questions(self, temp_bench_dir):
        bench = VeraBenchLoader(temp_bench_dir).load()
        evaluator = VeraBenchEvaluator(benchmark=bench)

        scope_q = BenchmarkQuestion(
            id="P002",
            type="conflict",
            question="欧盟AI法案是否禁止所有人脸识别？",
            ground_truth_answer="不准确，欧盟AI法案只禁止部分高风险用法。",
            expected_behavior="answer_with_conflict_note",
            expected_conflicts=[],
            difficulty="hard",
        )
        factual_q = BenchmarkQuestion(
            id="P003",
            type="conflict",
            question="全球气候敏感度（ECS）的最佳估计值是多少？",
            ground_truth_answer="不同来源给出不同估计。",
            expected_behavior="answer_with_conflict_note",
            expected_conflicts=[],
            difficulty="hard",
        )

        assert evaluator._premise_refutation_expected(scope_q) is True
        assert evaluator._premise_refutation_expected(factual_q) is False
        assert evaluator._premise_refutation_expected(
            BenchmarkQuestion(
                id="P004",
                type="conflict",
                question="固态电池已经实现大规模量产了吗？",
                ground_truth_answer="尚未实现大规模量产。",
                expected_behavior="answer_with_conflict_note",
                expected_conflicts=[],
                difficulty="medium",
            )
        ) is True
        assert evaluator._premise_refutation_detected("该说法不准确，不能说它禁止所有人脸识别。") is True
        assert evaluator._premise_refutation_detected("因此，无法给出唯一的最佳估计值。") is False
        assert evaluator._premise_refutation_detected("量子纠错的错误率持续降低。") is False
        assert evaluator._premise_refutation_detected("欧盟将其列为不可接受风险。") is False
        assert evaluator._premise_refutation_detected("基础RAG已不足以应对复杂任务。") is False
        assert evaluator._premise_refutation_detected("这个说法为时过早。") is True
        assert evaluator._premise_refutation_detected("该报道的数据有误，应以正式财报为准。") is True

    def test_conflict_behavior_accepts_premise_refutation(self, temp_bench_dir):
        bench = VeraBenchLoader(temp_bench_dir).load()
        evaluator = VeraBenchEvaluator(benchmark=bench)
        q = BenchmarkQuestion(
            id="P005",
            type="conflict",
            question="中国已经量产了7nm芯片，这是否意味着中国的芯片制造已经达到了世界先进水平？",
            ground_truth_answer="不意味着。",
            expected_behavior="answer_with_conflict_note",
            expected_conflicts=[],
            difficulty="hard",
        )

        assert evaluator._classify_behavior(
            q,
            "不意味着。仅凭量产7nm芯片不能代表中国芯片制造达到世界先进水平。",
        ) == "answer_with_conflict_note"

    def test_report_premise_refutation_counts(self, temp_bench_dir):
        bench = VeraBenchLoader(temp_bench_dir).load()
        evaluator = VeraBenchEvaluator(benchmark=bench)
        rows = [
            QuestionResult(
                question_id="P001", question_type="conflict",
                question="?", ground_truth="A", predicted="A",
                expected_behavior="answer_with_conflict_note",
                actual_behavior="answer_with_conflict_note", correct=True,
                premise_refutation_expected=True,
                premise_refutation_detected=True,
                premise_refutation_correct=True,
            ),
            QuestionResult(
                question_id="P002", question_type="conflict",
                question="?", ground_truth="A", predicted="A",
                expected_behavior="answer_with_conflict_note",
                actual_behavior="answer_with_citation", correct=False,
                premise_refutation_expected=True,
                premise_refutation_detected=False,
                premise_refutation_correct=False,
            ),
            QuestionResult(
                question_id="P003", question_type="single_evidence",
                question="?", ground_truth="A", predicted="A",
                expected_behavior="answer_with_citation",
                actual_behavior="answer_with_citation", correct=True,
                premise_refutation_expected=False,
                premise_refutation_detected=True,
                premise_refutation_correct=False,
            ),
        ]

        report = evaluator._build_report(rows)

        summary = report.premise_refutation_summary
        assert summary["expected"] == 2
        assert summary["detected"] == 2
        assert summary["true_positives"] == 1
        assert summary["false_positives"] == 1
        assert summary["false_negatives"] == 1
        assert summary["precision"] == 0.5
        assert summary["recall"] == 0.5
        assert summary["by_type"]["conflict"]["fn"] == 1

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
                    "premise_refutation_expected": True,
                    "premise_refutation_detected": True,
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
        assert analysis["premise_refutation_summary"]["expected"] == 1
        assert analysis["premise_refutation_summary"]["detected"] == 1
        assert analysis["premise_refutation_summary"]["precision"] == 1.0
        assert analysis["confidence_intervals"]["method"] == (
            "stratified-question-bootstrap-v1"
        )
        assert "answer_f1" in analysis["confidence_intervals"]["metrics"]
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
        assert analysis["premise_refutation_summary"]["available"] is False

    def test_offline_analysis_script_runs_outside_project_root(self, tmp_path):
        report_path = tmp_path / "report.json"
        report_path.write_text(json.dumps({
            "question_results": [{
                "question_id": "T001",
                "question_type": "single_evidence",
                "expected_behavior": "answer_with_citation",
                "actual_behavior": "answer_with_citation",
                "answer_f1": 1.0,
                "evidence_recall": 1.0,
                "confidence": 1.0,
                "correct": True,
            }]
        }), encoding="utf-8")
        script = (
            Path(__file__).resolve().parent.parent
            / "experiments"
            / "analyze_verabench_results.py"
        )

        result = subprocess.run(
            [sys.executable, str(script), str(report_path), "--json"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )

        analysis = json.loads(result.stdout)
        assert analysis["confidence_intervals"]["metrics"]["answer_f1"][
            "estimate"
        ] == 1.0

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
