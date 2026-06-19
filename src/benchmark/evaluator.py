"""VeraBench evaluator: runs pipeline on benchmark and computes metrics."""

import json
import logging
import math
import re
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from src.evaluation.answer_metrics import AnswerMetrics
from src.evaluation.evidence_metrics import EvidenceMetrics
from src.evaluation.statistics import (
    dependency_cluster_bootstrap_confidence_intervals,
    stratified_bootstrap_confidence_intervals,
)

from .loader import (
    BenchmarkQuestion,
    VeraBench,
    evidence_dependency_groups,
    load_verabench,
)

logger = logging.getLogger("verabench")


@dataclass
class QuestionResult:
    question_id: str
    question_type: str
    question: str
    ground_truth: str
    predicted: str
    expected_behavior: str
    actual_behavior: str
    correct: bool
    answer_em: float = 0.0
    answer_f1: float = 0.0
    evidence_recall: float = 0.0
    evidence_precision: float = 0.0
    conflict_detection_f1: float = 0.0
    predicted_conflicts: int = 0
    gold_conflicts: int = 0
    conflict_true_positives: int = 0
    conflict_false_positives: int = 0
    conflict_false_negatives: int = 0
    premise_refutation_expected: bool = False
    premise_refutation_detected: bool = False
    premise_refutation_correct: bool = False
    confidence: float = 0.0
    latency_seconds: float = 0.0
    claims_supported: int = 0
    claims_refuted: int = 0
    claims_nei: int = 0
    difficulty: str = "medium"
    error: str | None = None
    dependency_group: str = ""
    gold_document_ids: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    total_questions: int = 0
    completed: int = 0
    errors: int = 0
    overall_answer_em: float = 0.0
    overall_answer_f1: float = 0.0
    overall_evidence_recall: float = 0.0
    overall_evidence_precision: float = 0.0
    overall_conflict_f1: float = 0.0
    behavior_accuracy: float = 0.0
    avg_confidence: float = 0.0
    avg_latency: float = 0.0
    ece: float = 0.0
    brier_score: float = 0.0
    calibration_bins: list[dict[str, Any]] = field(default_factory=list)
    confidence_intervals: dict[str, Any] = field(default_factory=dict)
    dependency_robust_confidence_intervals: dict[str, Any] = field(
        default_factory=dict
    )
    conflict_summary: dict[str, Any] = field(default_factory=dict)
    premise_refutation_summary: dict[str, Any] = field(default_factory=dict)
    behavior_confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    failure_summary: dict[str, Any] = field(default_factory=dict)
    by_type: dict[str, dict[str, float]] = field(default_factory=dict)
    by_difficulty: dict[str, dict[str, float]] = field(default_factory=dict)
    results: list[QuestionResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "total_questions": self.total_questions,
            "completed": self.completed,
            "errors": self.errors,
            "overall_answer_em": round(self.overall_answer_em, 4),
            "overall_answer_f1": round(self.overall_answer_f1, 4),
            "overall_evidence_recall": round(self.overall_evidence_recall, 4),
            "overall_evidence_precision": round(self.overall_evidence_precision, 4),
            "overall_conflict_f1": round(self.overall_conflict_f1, 4),
            "behavior_accuracy": round(self.behavior_accuracy, 4),
            "avg_confidence": round(self.avg_confidence, 4),
            "avg_latency": round(self.avg_latency, 4),
            "ece": round(self.ece, 4),
            "brier_score": round(self.brier_score, 4),
            "calibration_bins": self.calibration_bins,
            "confidence_intervals": self.confidence_intervals,
            "dependency_robust_confidence_intervals": (
                self.dependency_robust_confidence_intervals
            ),
            "conflict_summary": self.conflict_summary,
            "premise_refutation_summary": self.premise_refutation_summary,
            "behavior_confusion": self.behavior_confusion,
            "failure_summary": self.failure_summary,
            "by_type": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in self.by_type.items()},
            "by_difficulty": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in self.by_difficulty.items()},
            "question_results": [
                {
                    "question_id": r.question_id,
                    "question_type": r.question_type,
                    "question": r.question,
                    "ground_truth": r.ground_truth,
                    "predicted": r.predicted,
                    "expected_behavior": r.expected_behavior,
                    "actual_behavior": r.actual_behavior,
                    "correct": r.correct,
                    "answer_em": r.answer_em,
                    "answer_f1": r.answer_f1,
                    "evidence_recall": round(r.evidence_recall, 4),
                    "evidence_precision": round(r.evidence_precision, 4),
                    "conflict_detection_f1": round(r.conflict_detection_f1, 4),
                    "predicted_conflicts": r.predicted_conflicts,
                    "gold_conflicts": r.gold_conflicts,
                    "conflict_true_positives": r.conflict_true_positives,
                    "conflict_false_positives": r.conflict_false_positives,
                    "conflict_false_negatives": r.conflict_false_negatives,
                    "premise_refutation_expected": r.premise_refutation_expected,
                    "premise_refutation_detected": r.premise_refutation_detected,
                    "premise_refutation_correct": r.premise_refutation_correct,
                    "confidence": round(r.confidence, 4),
                    "latency_seconds": round(r.latency_seconds, 2),
                    "difficulty": r.difficulty,
                    "error": r.error,
                    "dependency_group": r.dependency_group,
                    "gold_document_ids": r.gold_document_ids,
                    "diagnostics": r.diagnostics,
                }
                for r in self.results
            ],
        }
        return d


class VeraBenchEvaluator:
    BEHAVIOR_METRIC_VERSION = "behavior-v2"
    CONFLICT_METRIC_VERSION = "gold-evidence-pair-micro-f1-v2"

    def __init__(
        self,
        benchmark: VeraBench | None = None,
        data_dir: str | None = None,
        pipeline_factory: Callable | None = None,
    ):
        if benchmark:
            self.benchmark = benchmark
        else:
            self.benchmark = load_verabench(data_dir)

        self.pipeline_factory = pipeline_factory
        self.dependency_groups = evidence_dependency_groups(
            self.benchmark.questions,
        )

    def evaluate(
        self,
        question_types: list[str] | None = None,
        question_ids: list[str] | None = None,
        max_questions: int | None = None,
        callback: Callable | None = None,
        checkpoint_path: str | None = None,
    ) -> BenchmarkReport:
        """Evaluate the pipeline over the benchmark.

        If ``checkpoint_path`` is given, each question's result is appended to
        that JSONL file as soon as it completes, and any questions already
        present there are reused instead of re-run. This makes an interrupted
        run resumable: just launch with the same checkpoint to "continue where
        it left off".
        """
        questions = self.benchmark.questions
        if question_types:
            questions = [q for q in questions if q.type in question_types]
        if question_ids:
            requested_ids = set(question_ids)
            known_ids = {q.id for q in self.benchmark.questions}
            unknown_ids = sorted(requested_ids - known_ids)
            if unknown_ids:
                raise ValueError(f"Unknown VeraBench question id(s): {', '.join(unknown_ids)}")
            questions = [q for q in questions if q.id in requested_ids]
        if max_questions:
            questions = questions[:max_questions]

        done: dict[str, QuestionResult] = {}
        if checkpoint_path:
            done = self._load_checkpoint(
                checkpoint_path,
                expected_questions={q.id: q for q in questions},
            )
            if done:
                logger.info(f"Resuming from checkpoint: {len(done)} question(s) already done")

        results: list[QuestionResult] = []
        total = len(questions)
        for i, q in enumerate(questions):
            if callback:
                callback(i, total, q)
            if q.id in done:
                results.append(done[q.id])  # reuse cached result, skip re-running
                continue
            result = self._evaluate_one(q)
            results.append(result)
            if checkpoint_path:
                self._append_checkpoint(checkpoint_path, result)

        return self._build_report(results)

    @staticmethod
    def _load_checkpoint(
        path: str,
        expected_questions: dict[str, BenchmarkQuestion] | None = None,
    ) -> dict[str, "QuestionResult"]:
        """Load previously-saved per-question results from a JSONL checkpoint."""
        loaded: dict[str, QuestionResult] = {}
        p = Path(path)
        if not p.exists():
            return loaded
        valid = {f.name for f in fields(QuestionResult)}
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = {k: v for k, v in json.loads(line).items() if k in valid}
                    r = QuestionResult(**d)
                    if expected_questions is not None:
                        expected = expected_questions.get(r.question_id)
                        if expected is None:
                            logger.warning(
                                "Skipping checkpoint result for question outside current run: %s",
                                r.question_id,
                            )
                            continue
                        if not VeraBenchEvaluator._checkpoint_result_matches_question(r, expected):
                            logger.warning(
                                "Skipping stale checkpoint result for changed question: %s",
                                r.question_id,
                            )
                            continue
                    loaded[r.question_id] = r  # last write wins
                except Exception as e:  # tolerate a partially-written final line
                    logger.warning(f"Skipping malformed checkpoint line: {e}")
        return loaded

    @staticmethod
    def _checkpoint_result_matches_question(
        result: QuestionResult,
        question: BenchmarkQuestion,
    ) -> bool:
        """Return whether a saved row still belongs to the current benchmark row."""
        return (
            result.question == question.question
            and result.question_type == question.type
            and result.ground_truth == question.ground_truth_answer
            and result.expected_behavior == question.expected_behavior
        )

    @staticmethod
    def _append_checkpoint(path: str, result: "QuestionResult") -> None:
        """Append one question result to the JSONL checkpoint (atomic per line)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
            f.flush()

    def evaluate_baseline(self, answer_fn: Callable[[str], str], **kwargs) -> BenchmarkReport:
        """Evaluate a baseline function (takes question, returns answer)."""
        questions = self.benchmark.questions
        if kwargs.get("question_types"):
            questions = [q for q in questions if q.type in kwargs["question_types"]]
        if kwargs.get("max_questions"):
            questions = questions[:kwargs["max_questions"]]

        results = []
        for q in questions:
            t0 = time.time()
            try:
                answer = answer_fn(q.question)
                latency = time.time() - t0
                result = self._score_answer(q, answer, latency)
            except Exception as e:
                result = QuestionResult(
                    question_id=q.id, question_type=q.type, question=q.question,
                    ground_truth=q.ground_truth_answer, predicted="",
                    expected_behavior=q.expected_behavior, actual_behavior="error",
                    correct=False, difficulty=q.difficulty, error=str(e),
                )
            results.append(result)

        return self._build_report(results)

    def _evaluate_one(self, q: BenchmarkQuestion) -> QuestionResult:
        if not self.pipeline_factory:
            return self._demo_evaluate(q)

        t0 = time.time()
        try:
            pipeline = self.pipeline_factory()
            output = pipeline.query(q.question)
            latency = time.time() - t0
            return self._score_pipeline_output(q, output, latency)
        except Exception as e:
            return QuestionResult(
                question_id=q.id, question_type=q.type, question=q.question,
                ground_truth=q.ground_truth_answer, predicted="",
                expected_behavior=q.expected_behavior, actual_behavior="error",
                correct=False, difficulty=q.difficulty, error=str(e),
            )

    def _demo_evaluate(self, q: BenchmarkQuestion) -> QuestionResult:
        """Construct a complete gold-vs-gold result for report plumbing checks.

        Demo mode is an identity check, not a model or retrieval evaluation.
        Populate every supervised field from the benchmark so zero-valued
        metrics cannot be mistaken for failed retrieval, conflict detection, or
        calibration.
        """
        gold_evidence_ids = [e.evidence_id for e in q.evidence]
        gold_conflict_pairs = {
            f"{pair[0]}-{pair[1]}"
            for pair in (
                tuple(sorted(conflict.pair))
                for conflict in q.expected_conflicts
            )
        }
        evidence_precision = (
            EvidenceMetrics.evidence_precision(
                gold_evidence_ids,
                gold_evidence_ids,
            )
            if gold_evidence_ids
            else 1.0
        )
        evidence_recall = EvidenceMetrics.evidence_recall(
            gold_evidence_ids,
            gold_evidence_ids,
        )
        supported = sum(
            1 for claim in q.ground_truth_claims
            if claim.status == "supported"
        )
        refuted = sum(
            1 for claim in q.ground_truth_claims
            if claim.status == "refuted"
        )
        nei = len(q.ground_truth_claims) - supported - refuted
        premise_refutation_expected = self._premise_refutation_expected(q)

        return QuestionResult(
            question_id=q.id,
            question_type=q.type,
            question=q.question,
            ground_truth=q.ground_truth_answer,
            predicted=q.ground_truth_answer,
            expected_behavior=q.expected_behavior,
            actual_behavior=q.expected_behavior,
            correct=True,
            answer_em=1.0,
            answer_f1=1.0,
            evidence_recall=evidence_recall,
            evidence_precision=evidence_precision,
            conflict_detection_f1=1.0 if gold_conflict_pairs else 0.0,
            predicted_conflicts=len(gold_conflict_pairs),
            gold_conflicts=len(gold_conflict_pairs),
            conflict_true_positives=len(gold_conflict_pairs),
            conflict_false_positives=0,
            conflict_false_negatives=0,
            premise_refutation_expected=premise_refutation_expected,
            premise_refutation_detected=premise_refutation_expected,
            premise_refutation_correct=True,
            confidence=1.0,
            latency_seconds=0.0,
            claims_supported=supported,
            claims_refuted=refuted,
            claims_nei=nei,
            difficulty=q.difficulty,
            diagnostics={
                "evaluation_mode": "demo_gold_identity",
                "evidence_ids": gold_evidence_ids,
                "predicted_conflict_pairs": sorted(gold_conflict_pairs),
                "scored_predicted_conflict_pairs": sorted(gold_conflict_pairs),
                "unscored_extraneous_conflict_pairs": [],
                "gold_conflict_pairs": sorted(gold_conflict_pairs),
            },
        )

    def _score_answer(self, q: BenchmarkQuestion, answer: str, latency: float) -> QuestionResult:
        em = AnswerMetrics.exact_match(answer, q.ground_truth_answer)
        f1 = AnswerMetrics.soft_f1_score(answer, q.ground_truth_answer)

        predicted_evidence_ids: list[str] = []
        gold_evidence_ids = [e.evidence_id for e in q.evidence]
        ep = EvidenceMetrics.evidence_precision(predicted_evidence_ids, gold_evidence_ids)
        er = EvidenceMetrics.evidence_recall(predicted_evidence_ids, gold_evidence_ids)

        actual_behavior = self._classify_behavior(q, answer)
        premise_refutation_expected = self._premise_refutation_expected(q)
        premise_refutation_detected = self._premise_refutation_detected(answer)
        gold_conflict_count = len({
            f"{pair[0]}-{pair[1]}"
            for pair in (tuple(sorted(c.pair)) for c in q.expected_conflicts)
        })

        return QuestionResult(
            question_id=q.id, question_type=q.type, question=q.question,
            ground_truth=q.ground_truth_answer, predicted=answer,
            expected_behavior=q.expected_behavior, actual_behavior=actual_behavior,
            correct=self._is_correct(q, f1, actual_behavior),
            answer_em=em, answer_f1=f1,
            evidence_recall=er, evidence_precision=ep,
            gold_conflicts=gold_conflict_count,
            conflict_false_negatives=gold_conflict_count,
            premise_refutation_expected=premise_refutation_expected,
            premise_refutation_detected=premise_refutation_detected,
            premise_refutation_correct=(
                premise_refutation_expected == premise_refutation_detected
            ),
            latency_seconds=latency,
            difficulty=q.difficulty,
        )

    def _score_pipeline_output(self, q: BenchmarkQuestion, output: Any, latency: float) -> QuestionResult:
        answer = output.answer if hasattr(output, "answer") else str(output)
        confidence = self._coerce_confidence(getattr(output, "confidence", 0.0))

        em = AnswerMetrics.exact_match(answer, q.ground_truth_answer)
        f1 = AnswerMetrics.soft_f1_score(answer, q.ground_truth_answer)

        # Evidence matching — map pipeline evidence_id (D006_c0) → original doc_id (D006) → gold evidence_id (E1)
        gold_evidence_ids = [e.evidence_id for e in q.evidence]
        doc_to_gold_eid = {ref.doc_id: ref.evidence_id for ref in q.evidence}

        def _chunk_id_to_doc_id(chunk_id: str) -> str:
            """Strip chunk suffix: D006_c0 → D006"""
            if "_c" in chunk_id:
                return chunk_id.rsplit("_c", 1)[0]
            return chunk_id

        pred_evidence_ids = []
        if hasattr(output, "evidence"):
            pred_evidence_ids = [
                doc_to_gold_eid.get(_chunk_id_to_doc_id(e.evidence_id), e.evidence_id)
                for e in output.evidence
            ]
        ep = EvidenceMetrics.evidence_precision(pred_evidence_ids, gold_evidence_ids)
        er = EvidenceMetrics.evidence_recall(pred_evidence_ids, gold_evidence_ids)

        # Conflict detection — map claim IDs → evidence IDs → gold evidence IDs
        # Build claim_id → pipeline evidence_id mapping from conflict report nodes
        claim_to_ev_id: dict[str, str] = {}
        if hasattr(output, "conflict_report") and output.conflict_report:
            for node in output.conflict_report.get("nodes", []):
                nid = node.get("node_id", "")
                ev_ids = node.get("evidence_ids", [])
                if nid and ev_ids:
                    claim_to_ev_id[nid] = ev_ids[0]

        pred_conflicts = []
        conflict_edge_diagnostics: list[dict[str, Any]] = []
        if hasattr(output, "conflict_report") and output.conflict_report:
            edges = output.conflict_report.get("edges", [])
            for edge in edges:
                if not self._is_conflict_edge(edge):
                    continue
                src_claim = edge.get("source_id", "")
                tgt_claim = edge.get("target_id", "")
                src_ev = claim_to_ev_id.get(src_claim, src_claim)
                tgt_ev = claim_to_ev_id.get(tgt_claim, tgt_claim)
                # Map pipeline evidence_id (D006_c0) → doc_id (D006) → gold evidence_id (E1)
                src_doc = _chunk_id_to_doc_id(src_ev)
                tgt_doc = _chunk_id_to_doc_id(tgt_ev)
                src_gold = doc_to_gold_eid.get(src_doc, src_ev)
                tgt_gold = doc_to_gold_eid.get(tgt_doc, tgt_ev)
                pair = tuple(sorted([src_gold, tgt_gold]))
                pred_conflicts.append(pair)
                if len(conflict_edge_diagnostics) < 20:
                    conflict_edge_diagnostics.append({
                        "source_id": src_claim,
                        "target_id": tgt_claim,
                        "source_evidence_id": src_ev,
                        "target_evidence_id": tgt_ev,
                        "mapped_pair": list(pair),
                        "conflict_type": edge.get("conflict_type", ""),
                        "rationale": edge.get("rationale", ""),
                    })

        gold_conflicts = []
        for c in q.expected_conflicts:
            pair = tuple(sorted(c.pair))
            gold_conflicts.append(pair)

        raw_pred_conflict_set = {f"{p[0]}-{p[1]}" for p in pred_conflicts}
        gold_conflict_set = {f"{p[0]}-{p[1]}" for p in gold_conflicts}
        pred_conflict_set, extraneous_conflict_set = self._scored_predicted_conflicts(
            raw_pred_conflict_set,
            set(gold_evidence_ids),
            gold_conflict_set,
        )
        conflict_f1 = 0.0
        conflict_tp = len(pred_conflict_set & gold_conflict_set)
        conflict_fp = len(pred_conflict_set - gold_conflict_set)
        conflict_fn = len(gold_conflict_set - pred_conflict_set)
        if gold_conflicts or pred_conflicts:
            conflict_f1 = EvidenceMetrics.evidence_f1(
                list(pred_conflict_set),
                list(gold_conflict_set),
            )

        # Claim verification
        claims_supported = claims_refuted = claims_nei = 0
        if hasattr(output, "verification_report") and output.verification_report:
            for cv in output.verification_report.claim_verifications:
                st = cv.get("verification_status", "not_enough_info")
                if st == "supported":
                    claims_supported += 1
                elif st == "refuted":
                    claims_refuted += 1
                else:
                    claims_nei += 1

        actual_behavior = self._classify_behavior(q, answer)
        premise_refutation_expected = self._premise_refutation_expected(q)
        premise_refutation_detected = self._premise_refutation_detected(answer)
        correct = self._is_correct(q, f1, actual_behavior)

        output_metadata = getattr(output, "metadata", {}) or {}
        conflict_report = getattr(output, "conflict_report", {}) or {}
        diagnostics = {
            "evidence_ids": [
                getattr(e, "evidence_id", "")
                for e in getattr(output, "evidence", [])[:20]
            ],
            "predicted_conflict_pairs": sorted(raw_pred_conflict_set),
            "scored_predicted_conflict_pairs": sorted(pred_conflict_set),
            "unscored_extraneous_conflict_pairs": sorted(extraneous_conflict_set),
            "gold_conflict_pairs": sorted(gold_conflict_set),
            "conflict_edges": conflict_edge_diagnostics,
            "premise_refutation_expected": premise_refutation_expected,
            "premise_refutation_detected": premise_refutation_detected,
            "conflict_report_nodes": len(conflict_report.get("nodes", [])),
            "conflict_report_edges": len(conflict_report.get("edges", [])),
            "output_metadata": output_metadata,
        }

        return QuestionResult(
            question_id=q.id, question_type=q.type, question=q.question,
            ground_truth=q.ground_truth_answer, predicted=answer,
            expected_behavior=q.expected_behavior, actual_behavior=actual_behavior,
            correct=correct, answer_em=em, answer_f1=f1,
            evidence_recall=er, evidence_precision=ep,
            conflict_detection_f1=conflict_f1,
            predicted_conflicts=len(pred_conflict_set),
            gold_conflicts=len(gold_conflict_set),
            conflict_true_positives=conflict_tp,
            conflict_false_positives=conflict_fp,
            conflict_false_negatives=conflict_fn,
            premise_refutation_expected=premise_refutation_expected,
            premise_refutation_detected=premise_refutation_detected,
            premise_refutation_correct=(
                premise_refutation_expected == premise_refutation_detected
            ),
            confidence=confidence, latency_seconds=latency,
            claims_supported=claims_supported,
            claims_refuted=claims_refuted,
            claims_nei=claims_nei,
            difficulty=q.difficulty,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _coerce_confidence(value: Any) -> float:
        """Normalize model confidence into the finite probability range."""
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(confidence):
            return 0.0
        return min(1.0, max(0.0, confidence))

    @staticmethod
    def _is_correct(
        question: BenchmarkQuestion,
        answer_f1: float,
        actual_behavior: str,
    ) -> bool:
        if question.type == "unanswerable" and actual_behavior == "abstain":
            return True
        if question.type == "misleading" and actual_behavior == "correct_premise":
            return True
        return answer_f1 > 0.3

    def rescore_results(self, results: list[QuestionResult]) -> BenchmarkReport:
        """Recompute answer and aggregate metrics without rerunning a pipeline."""
        questions = {question.id: question for question in self.benchmark.questions}
        for result in results:
            question = questions.get(result.question_id)
            if question is None or result.error is not None:
                continue
            result.answer_em = AnswerMetrics.exact_match(
                result.predicted,
                question.ground_truth_answer,
            )
            result.answer_f1 = AnswerMetrics.soft_f1_score(
                result.predicted,
                question.ground_truth_answer,
            )
            result.actual_behavior = self._classify_behavior(
                question,
                result.predicted,
            )
            result.premise_refutation_expected = self._premise_refutation_expected(
                question,
            )
            result.premise_refutation_detected = self._premise_refutation_detected(
                result.predicted,
            )
            result.premise_refutation_correct = (
                result.premise_refutation_expected
                == result.premise_refutation_detected
            )
            result.correct = self._is_correct(
                question,
                result.answer_f1,
                result.actual_behavior,
            )
            self._rescore_conflicts_from_diagnostics(question, result)
        return self._build_report(results)

    @staticmethod
    def _scored_predicted_conflicts(
        predicted_pairs: set[str],
        gold_evidence_ids: set[str],
        gold_conflict_pairs: set[str],
    ) -> tuple[set[str], set[str]]:
        """Score only predicted pairs that are fully inside question gold evidence.

        A RAG pipeline can retrieve extra distractor evidence and detect real
        disagreement there. VeraBench does not exhaustively annotate every
        possible conflict outside the question's gold evidence set, so those
        pairs are useful diagnostics but should not be counted as false
        positives in the pair-F1 metric.
        """
        if not gold_evidence_ids:
            return set(), set(predicted_pairs)
        scored = {
            pair for pair in predicted_pairs
            if set(pair.split("-", 1)) <= gold_evidence_ids
            and (len(set(pair.split("-", 1))) > 1 or pair in gold_conflict_pairs)
        }
        return scored, set(predicted_pairs) - scored

    def _rescore_conflicts_from_diagnostics(
        self,
        question: BenchmarkQuestion,
        result: QuestionResult,
    ) -> None:
        predicted_pairs = set(result.diagnostics.get("predicted_conflict_pairs") or [])
        if not predicted_pairs and not result.diagnostics.get("scored_predicted_conflict_pairs"):
            return
        gold_evidence_ids = {ref.evidence_id for ref in question.evidence}
        gold_pairs = {
            f"{pair[0]}-{pair[1]}"
            for pair in (tuple(sorted(conflict.pair)) for conflict in question.expected_conflicts)
        }
        scored_pairs, extraneous_pairs = self._scored_predicted_conflicts(
            predicted_pairs,
            gold_evidence_ids,
            gold_pairs,
        )
        result.predicted_conflicts = len(scored_pairs)
        result.gold_conflicts = len(gold_pairs)
        result.conflict_true_positives = len(scored_pairs & gold_pairs)
        result.conflict_false_positives = len(scored_pairs - gold_pairs)
        result.conflict_false_negatives = len(gold_pairs - scored_pairs)
        if scored_pairs or gold_pairs:
            result.conflict_detection_f1 = EvidenceMetrics.evidence_f1(
                list(scored_pairs),
                list(gold_pairs),
            )
        else:
            result.conflict_detection_f1 = 0.0
        result.diagnostics["scored_predicted_conflict_pairs"] = sorted(scored_pairs)
        result.diagnostics["unscored_extraneous_conflict_pairs"] = sorted(extraneous_pairs)

    @staticmethod
    def _is_conflict_edge(edge: dict[str, Any]) -> bool:
        """Return True for edges that should count toward conflict detection.

        The conflict graph can contain support and partial-support edges for
        reasoning context. VeraBench conflict F1 should only score actual
        disagreement edges; otherwise high-quality support detection becomes
        false positives in the conflict metric.
        """
        conflict_type = edge.get("conflict_type", "")
        return conflict_type in {
            "refute",
            "numeric_conflict",
            "temporal_conflict",
            "entity_mismatch",
            "source_disagreement",
            "definitional_conflict",
            "scope_conflict",
            "causal_conflict",
            "granularity_conflict",
        }

    @staticmethod
    def _premise_refutation_expected(q: BenchmarkQuestion) -> bool:
        """Whether the question asks to validate a premise or overgeneralization."""
        if q.expected_behavior not in {"answer_with_conflict_note", "correct_premise"}:
            return False
        question = q.question.lower()
        premise_markers = (
            "是否意味着",
            "这是否意味着",
            "是不是意味着",
            "是否代表",
            "是不是代表",
            "是否说明",
            "是不是说明",
            "是否正确",
            "是否禁止",
            "是否完全",
            "是不是完全",
            "是否已经",
            "是不是已经",
            "已经证实",
            "了吗",
            "了么",
            "对吗",
            "对吧",
            "既然",
            "完全替代",
            "取代",
            "越多",
        )
        yes_no_markers = ("是否", "是不是", "能否", "会不会", "可以吗", "对吗", "对吧")
        universal_markers = (
            "所有",
            "全部",
            "完全",
            "一律",
            "任何",
        )
        return any(marker in question for marker in premise_markers) or (
            any(marker in question for marker in yes_no_markers)
            and any(marker in question for marker in universal_markers)
        )

    @staticmethod
    def _premise_refutation_detected(answer: str) -> bool:
        """Whether the answer explicitly rejects or narrows the user's premise."""
        lower = answer.lower()
        markers = (
            "不意味着",
            "并不意味着",
            "不能说明",
            "并不能说明",
            "不能认为",
            "不能简单",
            "不能简单地认为",
            "不能简单地说",
            "不能完全",
            "不能等同",
            "不代表",
            "并不代表",
            "不等于",
            "不应",
            "不能取代",
            "不准确",
            "不完全准确",
            "不能完全替代",
            "无法完全替代",
            "尚未实现",
            "不再代表",
            "并非实际",
            "不是实际",
            "商业命名",
            "尚未证实",
            "并未证实",
            "未能证实",
            "不能证明",
            "不能证实",
            "为时过早",
            "前提不成立",
            "推论不成立",
            "说法不成立",
            "说法不准确",
            "作为辅助",
            "而非替代",
        )
        if any(marker in lower for marker in markers):
            return True
        correction_patterns = (
            r"(?:说法|前提|信息|报道|数据|结论|理解).{0,8}(?:错误|有误)",
            r"(?:错误|有误).{0,8}(?:说法|前提|信息|报道|数据|结论|理解)",
        )
        return any(re.search(pattern, lower) for pattern in correction_patterns)

    def _classify_behavior(self, q: BenchmarkQuestion, answer: str) -> str:
        if not answer or not answer.strip():
            return "abstain"

        lower = answer.lower()

        if q.type == "misleading":
            correction_keywords = [
                "不正确", "错误", "不准确", "这个前提", "这个说法", "这个推论", "这个理解",
                "不意味着", "不代表", "不等于", "不能说明", "不能证明", "不能简单",
                "尚未证实", "并未证实", "未证实", "不完全准确", "不能完全替代",
                "无法完全替代", "尚未实现",
            ]
            if any(kw in lower for kw in correction_keywords):
                return "correct_premise"

        if (
            (q.type == "conflict" or q.expected_conflicts)
            and self._premise_refutation_detected(answer)
        ):
            return "answer_with_conflict_note"

        conflict_keywords = ["冲突", "矛盾", "不一致", "争议", "不同", "错误", "不准确"]
        if (q.type == "conflict" or q.expected_conflicts) and any(
            kw in lower for kw in conflict_keywords
        ):
            return "answer_with_conflict_note"

        if self._has_abstention(answer, anywhere=q.type == "unanswerable"):
            return "abstain"

        return "answer_with_citation"

    @staticmethod
    def _has_abstention(answer: str, *, anywhere: bool = False) -> bool:
        lower = answer.lower()
        stripped = lower.strip(" \t\r\n，,。:：；;")
        prefix_patterns = (
            r"^无法(?:回答|给出|确定)",
            r"^语料库中没有",
            r"^现有信息不足",
            r"^根据(?:现有|提供的)?证据[，,\s]*(?:无法|不能)",
            r"^不知道",
        )
        if any(re.search(pattern, stripped) for pattern in prefix_patterns):
            return True
        if not anywhere:
            return False
        window = lower[:500]
        anywhere_patterns = (
            r"无法(?:回答|给出|确定)",
            r"根据(?:现有|提供的)?证据[，,\s]*(?:无法|不能)",
            r"(?:证据|语料|材料|文档).{0,20}(?:未|没有)(?:提供|涉及|包含)",
        )
        return any(re.search(pattern, window) for pattern in anywhere_patterns)

    def _build_report(self, results: list[QuestionResult]) -> BenchmarkReport:
        if not results:
            return BenchmarkReport()

        questions = {question.id: question for question in self.benchmark.questions}
        for result in results:
            result.confidence = self._coerce_confidence(result.confidence)
            question = questions.get(result.question_id)
            if question is None:
                result.dependency_group = (
                    result.dependency_group
                    or f"unmapped-question-{result.question_id}"
                )
                continue
            result.dependency_group = self.dependency_groups[result.question_id]
            result.gold_document_ids = sorted({
                evidence.doc_id for evidence in question.evidence
            })

        completed = [r for r in results if r.error is None]
        errored = [r for r in results if r.error is not None]

        def avg(field_name: str) -> float:
            vals = [getattr(r, field_name) for r in completed]
            return sum(vals) / len(vals) if vals else 0.0

        report = BenchmarkReport(
            total_questions=len(results),
            completed=len(completed),
            errors=len(errored),
            overall_answer_em=avg("answer_em"),
            overall_answer_f1=avg("answer_f1"),
            overall_evidence_recall=avg("evidence_recall"),
            overall_evidence_precision=avg("evidence_precision"),
            overall_conflict_f1=avg("conflict_detection_f1"),
            behavior_accuracy=sum(1 for r in completed if r.actual_behavior == r.expected_behavior) / len(completed) if completed else 0.0,
            avg_confidence=avg("confidence"),
            avg_latency=avg("latency_seconds"),
            results=results,
        )
        report.calibration_bins = self._build_calibration_bins(completed)
        report.confidence_intervals = stratified_bootstrap_confidence_intervals(
            completed,
        )
        report.dependency_robust_confidence_intervals = (
            dependency_cluster_bootstrap_confidence_intervals(completed)
            if completed
            else {}
        )
        report.conflict_summary = self._build_conflict_summary(completed)
        report.overall_conflict_f1 = float(report.conflict_summary.get("f1", 0.0))
        report.premise_refutation_summary = self._build_premise_refutation_summary(completed)
        report.behavior_confusion = self._build_behavior_confusion(completed)
        report.failure_summary = self._build_failure_summary(completed, errored)

        # Calibration metrics (ECE, Brier Score)
        if completed:
            from ..uncertainty.calibrator import ConfidenceCalibrator
            calibrator = ConfidenceCalibrator()
            confidences = [r.confidence for r in completed]
            correct_flags = [r.correct for r in completed]
            cal_metrics = calibrator.compute_calibration_metrics(confidences, correct_flags)
            report.ece = cal_metrics["ece"]
            report.brier_score = cal_metrics["brier_score"]

        # Breakdown by type
        by_type = defaultdict(list)
        by_difficulty = defaultdict(list)
        for r in completed:
            by_type[r.question_type].append(r)
            by_difficulty[r.difficulty].append(r)

        for group_name, group_data in [("by_type", by_type), ("by_difficulty", by_difficulty)]:
            target = getattr(report, group_name)
            for key, items in group_data.items():
                n = len(items)
                target[key] = {
                    "count": n,
                    "answer_em": sum(r.answer_em for r in items) / n,
                    "answer_f1": sum(r.answer_f1 for r in items) / n,
                    "evidence_recall": sum(r.evidence_recall for r in items) / n,
                    "behavior_accuracy": sum(1 for r in items if r.actual_behavior == r.expected_behavior) / n,
                    "avg_confidence": sum(r.confidence for r in items) / n,
                }

        return report

    @staticmethod
    def _build_calibration_bins(
        results: list[QuestionResult],
        n_bins: int = 10,
    ) -> list[dict[str, Any]]:
        """Aggregate confidence calibration bins for reliability diagrams."""
        if not results:
            return []

        bins: list[dict[str, Any]] = []
        for i in range(n_bins):
            lower = i / n_bins
            upper = (i + 1) / n_bins
            if i == n_bins - 1:
                rows = [r for r in results if lower <= r.confidence <= upper]
            else:
                rows = [r for r in results if lower <= r.confidence < upper]

            if rows:
                avg_confidence = sum(r.confidence for r in rows) / len(rows)
                accuracy = sum(1 for r in rows if r.correct) / len(rows)
            else:
                avg_confidence = (lower + upper) / 2
                accuracy = 0.0

            bins.append({
                "bin": i + 1,
                "lower": round(lower, 4),
                "upper": round(upper, 4),
                "count": len(rows),
                "avg_confidence": round(avg_confidence, 4),
                "accuracy": round(accuracy, 4),
                "gap": round(abs(avg_confidence - accuracy), 4),
            })
        return bins

    @staticmethod
    def _build_conflict_summary(results: list[QuestionResult]) -> dict[str, Any]:
        """Aggregate conflict detection counts and likely failure mode."""
        total_gold = sum(r.gold_conflicts for r in results)
        total_pred = sum(r.predicted_conflicts for r in results)
        total_tp = sum(r.conflict_true_positives for r in results)
        total_fp = sum(r.conflict_false_positives for r in results)
        total_fn = sum(r.conflict_false_negatives for r in results)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        if total_fp > total_fn * 2:
            dominant_failure = "over_detection"
        elif total_fn > total_fp * 2:
            dominant_failure = "under_detection"
        elif total_fp or total_fn:
            dominant_failure = "mixed"
        else:
            dominant_failure = "none"

        by_type: dict[str, dict[str, int]] = {}
        for r in results:
            bucket = by_type.setdefault(r.question_type, {
                "gold": 0,
                "predicted": 0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
            })
            bucket["gold"] += r.gold_conflicts
            bucket["predicted"] += r.predicted_conflicts
            bucket["tp"] += r.conflict_true_positives
            bucket["fp"] += r.conflict_false_positives
            bucket["fn"] += r.conflict_false_negatives

        return {
            "available": True,
            "gold_conflicts": total_gold,
            "predicted_conflicts": total_pred,
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "dominant_failure": dominant_failure,
            "by_type": by_type,
        }

    @staticmethod
    def _build_premise_refutation_summary(results: list[QuestionResult]) -> dict[str, Any]:
        """Aggregate premise-refutation diagnostics separately from evidence conflicts."""
        total_expected = sum(1 for r in results if r.premise_refutation_expected)
        total_detected = sum(1 for r in results if r.premise_refutation_detected)
        total_correct = sum(1 for r in results if r.premise_refutation_correct)
        true_positives = sum(
            1 for r in results
            if r.premise_refutation_expected and r.premise_refutation_detected
        )
        false_positives = sum(
            1 for r in results
            if not r.premise_refutation_expected and r.premise_refutation_detected
        )
        false_negatives = sum(
            1 for r in results
            if r.premise_refutation_expected and not r.premise_refutation_detected
        )
        precision = (
            true_positives / (true_positives + false_positives)
            if true_positives + false_positives
            else 0.0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if true_positives + false_negatives
            else 0.0
        )
        by_type: dict[str, dict[str, int]] = {}
        for r in results:
            bucket = by_type.setdefault(r.question_type, {
                "expected": 0,
                "detected": 0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
            })
            if r.premise_refutation_expected:
                bucket["expected"] += 1
            if r.premise_refutation_detected:
                bucket["detected"] += 1
            if r.premise_refutation_expected and r.premise_refutation_detected:
                bucket["tp"] += 1
            elif r.premise_refutation_detected:
                bucket["fp"] += 1
            elif r.premise_refutation_expected:
                bucket["fn"] += 1

        return {
            "available": True,
            "expected": total_expected,
            "detected": total_detected,
            "correct": total_correct,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "by_type": by_type,
        }

    @staticmethod
    def _build_behavior_confusion(results: list[QuestionResult]) -> dict[str, dict[str, int]]:
        """Build expected-behavior -> actual-behavior counts for diagnosis."""
        confusion: dict[str, dict[str, int]] = {}
        for r in results:
            expected = r.expected_behavior or "unknown"
            actual = r.actual_behavior or "unknown"
            confusion.setdefault(expected, {})
            confusion[expected][actual] = confusion[expected].get(actual, 0) + 1
        return confusion

    @staticmethod
    def _build_failure_summary(
        completed: list[QuestionResult],
        errored: list[QuestionResult],
        max_examples: int = 10,
    ) -> dict[str, Any]:
        """Summarize the most useful failure modes for report readers."""
        behavior_failures = [
            r for r in completed
            if r.actual_behavior != r.expected_behavior
        ]
        low_evidence_recall = [
            r for r in completed
            if r.evidence_recall < 0.5 and r.expected_behavior != "abstain"
        ]
        conflict_failures = [
            r for r in completed
            if (
                r.conflict_false_positives > 0
                or r.conflict_false_negatives > 0
                or (
                    r.question_type == "conflict"
                    and r.actual_behavior != r.expected_behavior
                )
            )
        ]

        def by_type(rows: list[QuestionResult]) -> dict[str, int]:
            counts: dict[str, int] = {}
            for r in rows:
                counts[r.question_type] = counts.get(r.question_type, 0) + 1
            return dict(sorted(counts.items()))

        def example(r: QuestionResult) -> dict[str, Any]:
            predicted = r.predicted.replace("\n", " ").strip()
            return {
                "question_id": r.question_id,
                "question_type": r.question_type,
                "difficulty": r.difficulty,
                "expected_behavior": r.expected_behavior,
                "actual_behavior": r.actual_behavior,
                "answer_f1": round(r.answer_f1, 4),
                "evidence_recall": round(r.evidence_recall, 4),
                "conflict_detection_f1": round(r.conflict_detection_f1, 4),
                "confidence": round(r.confidence, 4),
                "question": r.question,
                "predicted_preview": predicted[:240],
            }

        # Rank examples by severity: behavior mismatch first, then weak evidence/conflict scores.
        ranked_failures = sorted(
            behavior_failures,
            key=lambda r: (
                r.answer_f1,
                r.evidence_recall,
                r.conflict_detection_f1,
                -r.latency_seconds,
            ),
        )

        return {
            "behavior_failure_count": len(behavior_failures),
            "behavior_failures_by_type": by_type(behavior_failures),
            "low_evidence_recall_count": len(low_evidence_recall),
            "low_evidence_recall_by_type": by_type(low_evidence_recall),
            "conflict_failure_count": len(conflict_failures),
            "conflict_failures_by_type": by_type(conflict_failures),
            "errored_count": len(errored),
            "top_behavior_failures": [
                example(r) for r in ranked_failures[:max_examples]
            ],
        }
