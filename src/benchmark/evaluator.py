"""VeraBench evaluator: runs pipeline on benchmark and computes metrics."""

import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from src.evaluation.answer_metrics import AnswerMetrics
from src.evaluation.evidence_metrics import EvidenceMetrics
from src.evidence.conflict_graph import ConflictType

from .loader import VeraBench, BenchmarkQuestion, load_verabench

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
    confidence: float = 0.0
    latency_seconds: float = 0.0
    claims_supported: int = 0
    claims_refuted: int = 0
    claims_nei: int = 0
    difficulty: str = "medium"
    error: Optional[str] = None


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
    by_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_difficulty: Dict[str, Dict[str, float]] = field(default_factory=dict)
    results: List[QuestionResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
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
                    "confidence": round(r.confidence, 4),
                    "latency_seconds": round(r.latency_seconds, 2),
                    "difficulty": r.difficulty,
                    "error": r.error,
                }
                for r in self.results
            ],
        }
        return d


class VeraBenchEvaluator:
    def __init__(
        self,
        benchmark: Optional[VeraBench] = None,
        data_dir: Optional[str] = None,
        pipeline_factory: Optional[Callable] = None,
    ):
        if benchmark:
            self.benchmark = benchmark
        else:
            self.benchmark = load_verabench(data_dir)

        self.pipeline_factory = pipeline_factory

    def evaluate(
        self,
        question_types: Optional[List[str]] = None,
        max_questions: Optional[int] = None,
        callback: Optional[Callable] = None,
    ) -> BenchmarkReport:
        questions = self.benchmark.questions
        if question_types:
            questions = [q for q in questions if q.type in question_types]
        if max_questions:
            questions = questions[:max_questions]

        results: List[QuestionResult] = []
        for i, q in enumerate(questions):
            if callback:
                callback(i, len(questions), q)
            result = self._evaluate_one(q)
            results.append(result)

        return self._build_report(results)

    def evaluate_baseline(self, answer_fn: Callable[[str], str], **kwargs) -> BenchmarkReport:
        """Evaluate a baseline function (takes question, returns answer)."""
        questions = self.benchmark.questions
        if "question_types" in kwargs and kwargs["question_types"]:
            questions = [q for q in questions if q.type in kwargs["question_types"]]
        if "max_questions" in kwargs and kwargs["max_questions"]:
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
        """Score ground truth against itself as a sanity check."""
        return self._score_answer(q, q.ground_truth_answer, 0.0)

    def _score_answer(self, q: BenchmarkQuestion, answer: str, latency: float) -> QuestionResult:
        em = AnswerMetrics.exact_match(answer, q.ground_truth_answer)
        f1 = AnswerMetrics.soft_f1_score(answer, q.ground_truth_answer)

        predicted_evidence_ids = []
        gold_evidence_ids = [e.evidence_id for e in q.evidence]
        ep = EvidenceMetrics.evidence_precision(predicted_evidence_ids, gold_evidence_ids)
        er = EvidenceMetrics.evidence_recall(predicted_evidence_ids, gold_evidence_ids)

        actual_behavior = self._classify_behavior(q, answer)

        return QuestionResult(
            question_id=q.id, question_type=q.type, question=q.question,
            ground_truth=q.ground_truth_answer, predicted=answer,
            expected_behavior=q.expected_behavior, actual_behavior=actual_behavior,
            correct=(f1 > 0.3),
            answer_em=em, answer_f1=f1,
            evidence_recall=er, evidence_precision=ep,
            latency_seconds=latency,
            difficulty=q.difficulty,
        )

    def _score_pipeline_output(self, q: BenchmarkQuestion, output: Any, latency: float) -> QuestionResult:
        answer = output.answer if hasattr(output, "answer") else str(output)
        confidence = output.confidence if hasattr(output, "confidence") else 0.0

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
        if hasattr(output, "conflict_report") and output.conflict_report:
            edges = output.conflict_report.get("edges", [])
            for edge in edges:
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

        gold_conflicts = []
        for c in q.expected_conflicts:
            pair = tuple(sorted(c.pair))
            gold_conflicts.append(pair)

        conflict_f1 = 0.0
        if gold_conflicts or pred_conflicts:
            conflict_f1 = EvidenceMetrics.evidence_f1(
                [f"{p[0]}-{p[1]}" for p in pred_conflicts],
                [f"{p[0]}-{p[1]}" for p in gold_conflicts],
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
        correct = (f1 > 0.3)

        if q.type == "unanswerable" and actual_behavior == "abstain":
            correct = True
        if q.type == "misleading" and actual_behavior == "correct_premise":
            correct = True

        return QuestionResult(
            question_id=q.id, question_type=q.type, question=q.question,
            ground_truth=q.ground_truth_answer, predicted=answer,
            expected_behavior=q.expected_behavior, actual_behavior=actual_behavior,
            correct=correct, answer_em=em, answer_f1=f1,
            evidence_recall=er, evidence_precision=ep,
            conflict_detection_f1=conflict_f1,
            confidence=confidence, latency_seconds=latency,
            claims_supported=claims_supported,
            claims_refuted=claims_refuted,
            claims_nei=claims_nei,
            difficulty=q.difficulty,
        )

    def _classify_behavior(self, q: BenchmarkQuestion, answer: str) -> str:
        if not answer or not answer.strip():
            return "abstain"

        lower = answer.lower()

        abstention_keywords = ["无法回答", "无法给出", "无法确定", "语料库中没有", "信息不足", "无法", "不知道"]
        if any(kw in lower for kw in abstention_keywords):
            return "abstain"

        if q.type == "misleading":
            correction_keywords = ["不正确", "错误", "不准确", "这个前提", "这个说法", "这个推论", "这个理解"]
            if any(kw in lower for kw in correction_keywords):
                return "correct_premise"

        if q.type == "conflict" or q.expected_conflicts:
            conflict_keywords = ["冲突", "矛盾", "不一致", "争议", "不同", "错误", "不准确"]
            if any(kw in lower for kw in conflict_keywords):
                return "answer_with_conflict_note"

        return "answer_with_citation"

    def _build_report(self, results: List[QuestionResult]) -> BenchmarkReport:
        if not results:
            return BenchmarkReport()

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
