#!/usr/bin/env python3
"""Compare VeraBench conflict detectors on gold evidence only.

This is an offline A/B check for the conflict graph layer. It does not call an
LLM, does not run retrieval, and does not require API keys. The goal is to
verify whether a learned conflict detector improves conflict edge recall before
spending time on full pipeline runs.
"""

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark.conflict_pairs import (  # noqa: E402
    SPLIT_STRATEGY,
    _dependency_group_split_maps,
)
from src.benchmark.loader import BenchmarkQuestion, VeraBench, load_verabench  # noqa: E402
from src.evidence.conflict_graph import ConflictGraphBuilder  # noqa: E402
from src.evidence.extractor import EvidenceExtractor  # noqa: E402
from src.utils.data_structures import ConflictType, Evidence  # noqa: E402

CONFLICT_EDGE_TYPES = {
    ConflictType.REFUTE,
    ConflictType.NUMERIC_CONFLICT,
    ConflictType.TEMPORAL_CONFLICT,
    ConflictType.ENTITY_MISMATCH,
    ConflictType.SOURCE_DISAGREEMENT,
    ConflictType.DEFINITIONAL_CONFLICT,
    ConflictType.SCOPE_CONFLICT,
    ConflictType.CAUSAL_CONFLICT,
    ConflictType.GRANULARITY_CONFLICT,
}


@dataclass(frozen=True)
class ConflictQuestionScore:
    question_id: str
    question_type: str
    gold: int
    predicted: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    predicted_pairs: list[list[str]]
    gold_pairs: list[list[str]]


def _gold_pairs(question: BenchmarkQuestion) -> set[tuple[str, str]]:
    return {tuple(sorted(conflict.pair)) for conflict in question.expected_conflicts}


def _evidence_for_question(
    benchmark: VeraBench,
    question: BenchmarkQuestion,
) -> tuple[list[Evidence], dict[str, str]]:
    """Build graph-ready Evidence objects and claim-id -> evidence-id map."""
    extractor = EvidenceExtractor()
    claim_to_evidence: dict[str, str] = {}
    evidence_list: list[Evidence] = []

    for ref in question.evidence:
        doc = benchmark.get_document(ref.doc_id)
        entities = doc.entities if doc else []
        title = doc.title if doc else ref.doc_id
        claims = extractor._extract_claims(ref.text_span)
        title_entities = extractor._extract_entities(title)
        title_times = extractor._extract_temporal_expressions(title)
        for claim in claims:
            claim_text = claim.claim.lower()
            entity_anchors = set(title_entities)
            entity_anchors.update(
                entity for entity in entities
                if entity.lower() in claim_text
            )
            if claim.source_span in {"reported_claim", "corrective_claim"}:
                entity_anchors.update(entities)
            claim.entities = list(dict.fromkeys([*claim.entities, *entity_anchors]))
            claim.time_expressions = list(dict.fromkeys([*claim.time_expressions, *title_times]))
            claim_to_evidence[claim.claim_id] = ref.evidence_id

        evidence_list.append(
            Evidence(
                evidence_id=ref.evidence_id,
                source=doc.source if doc else ref.category,
                title=title,
                text_span=ref.text_span,
                date=doc.date if doc else None,
                author=doc.author if doc else None,
                url=doc.url if doc else None,
                entities=entities,
                claims=claims,
            )
        )

    return evidence_list, claim_to_evidence


def _predicted_pairs(
    benchmark: VeraBench,
    question: BenchmarkQuestion,
    builder: ConflictGraphBuilder,
) -> set[tuple[str, str]]:
    evidence_list, claim_to_evidence = _evidence_for_question(benchmark, question)
    if len(evidence_list) < 1:
        return set()

    graph = builder.build_graph(evidence_list, use_llm=False)
    pairs: set[tuple[str, str]] = set()
    for edge in graph.edges:
        if edge.conflict_type not in CONFLICT_EDGE_TYPES:
            continue
        source = claim_to_evidence.get(edge.source_id, edge.source_id)
        target = claim_to_evidence.get(edge.target_id, edge.target_id)
        pairs.add(tuple(sorted([source, target])))
    return pairs


def _score_sets(
    question: BenchmarkQuestion,
    predicted: set[tuple[str, str]],
    gold: set[tuple[str, str]],
) -> ConflictQuestionScore:
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return ConflictQuestionScore(
        question_id=question.id,
        question_type=question.type,
        gold=len(gold),
        predicted=len(predicted),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        predicted_pairs=[list(pair) for pair in sorted(predicted)],
        gold_pairs=[list(pair) for pair in sorted(gold)],
    )


def _summarize(scores: list[ConflictQuestionScore]) -> dict[str, Any]:
    gold = sum(score.gold for score in scores)
    predicted = sum(score.predicted for score in scores)
    tp = sum(score.true_positives for score in scores)
    fp = sum(score.false_positives for score in scores)
    fn = sum(score.false_negatives for score in scores)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "questions": len(scores),
        "gold_conflicts": gold,
        "predicted_conflicts": predicted,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _dataset_identity(data_dir: str | None) -> dict[str, Any]:
    if not data_dir:
        return {
            "source": "bundled-verabench",
            "path": None,
            "fingerprints": None,
        }
    root = Path(data_dir).resolve()
    fingerprints = {}
    for name in ("corpus.jsonl", "questions.jsonl"):
        path = root / name
        if not path.exists():
            raise ValueError(f"Missing independent evaluation file: {path}")
        fingerprints[f"{path.stem}_sha256"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return {
        "source": "external-data-dir",
        "path": str(root),
        "fingerprints": fingerprints,
    }


def evaluate_variant(
    benchmark: VeraBench,
    *,
    name: str,
    config: dict[str, Any],
    question_types: set[str] | None = None,
    question_ids: set[str] | None = None,
    max_questions: int | None = None,
) -> dict[str, Any]:
    builder = ConflictGraphBuilder(config=config)
    questions = [
        question for question in benchmark.questions
        if question.expected_conflicts
        and (question_types is None or question.type in question_types)
        and (question_ids is None or question.id in question_ids)
    ]
    if max_questions is not None:
        questions = questions[:max_questions]

    scores = [
        _score_sets(question, _predicted_pairs(benchmark, question, builder), _gold_pairs(question))
        for question in questions
    ]
    return {
        "name": name,
        "learned_available": bool(getattr(builder, "_learned_available", False)),
        "summary": _summarize(scores),
        "questions": [asdict(score) for score in scores],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare VeraBench conflict detector variants offline")
    parser.add_argument("--data-dir", help="Optional VeraBench data directory")
    parser.add_argument("--learned-model-path", help="Optional trained conflict CrossEncoder directory")
    parser.add_argument("--learned-threshold", type=float, default=0.5)
    parser.add_argument("--types", nargs="+", help="Optional question types to include")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        help="Evaluate only the dependency-aware training-data split.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--independent-test",
        action="store_true",
        help=(
            "Declare a separately maintained --data-dir as an independent "
            "test set; cannot be combined with --split."
        ),
    )
    parser.add_argument(
        "--evaluation-id",
        help="Stable identifier for an independent evaluation release.",
    )
    parser.add_argument("--max", type=int, help="Max conflict-bearing questions to evaluate")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    if args.independent_test and not args.data_dir:
        parser.error("--independent-test requires --data-dir")
    if args.independent_test and args.split:
        parser.error("--independent-test cannot be combined with --split")
    if args.independent_test and not args.evaluation_id:
        parser.error("--independent-test requires --evaluation-id")

    benchmark = load_verabench(args.data_dir)
    question_types = set(args.types) if args.types else None
    question_ids = None
    if args.split:
        try:
            split_by_question, _ = _dependency_group_split_maps(
                benchmark.questions,
                args.train_ratio,
                args.val_ratio,
            )
        except ValueError as exc:
            parser.error(str(exc))
        question_ids = {
            question_id
            for question_id, split in split_by_question.items()
            if split == args.split
        }
    variants = [
        evaluate_variant(
            benchmark,
            name="rules",
            config={"conflict_graph": {"enable_learned_detector": False, "enable_nli": False}},
            question_types=question_types,
            question_ids=question_ids,
            max_questions=args.max,
        )
    ]
    if args.learned_model_path:
        variants.append(
            evaluate_variant(
                benchmark,
                name="rules_plus_learned",
                config={
                    "conflict_graph": {
                        "enable_learned_detector": True,
                        "learned_model_path": args.learned_model_path,
                        "learned_threshold": args.learned_threshold,
                        "enable_nli": False,
                    }
                },
                question_types=question_types,
                question_ids=question_ids,
                max_questions=args.max,
            )
        )

    payload = {
        "evaluation_scope": {
            "split": args.split or "all",
            "split_strategy": SPLIT_STRATEGY if args.split else None,
            "train_ratio": args.train_ratio if args.split else None,
            "val_ratio": args.val_ratio if args.split else None,
            "independent_test": args.independent_test,
            "evaluation_id": args.evaluation_id,
            "dataset": _dataset_identity(args.data_dir),
        },
        "variants": variants,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
