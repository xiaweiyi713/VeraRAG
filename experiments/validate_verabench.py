"""Validate VeraBench structure, ontology, packaged-data sync, and fingerprints."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark.loader import (  # noqa: E402
    VERABENCH_VERSION,
    VeraBenchLoader,
    evidence_dependency_groups,
    evidence_span_match_kind,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _character_bigrams(text: str) -> set[str]:
    normalized = "".join(character.lower() for character in text if character.isalnum())
    return {
        normalized[index:index + 2]
        for index in range(max(0, len(normalized) - 1))
    }


def _near_duplicate_questions(
    questions: list[Any],
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    pairs = []
    for first, second in combinations(questions, 2):
        first_grams = _character_bigrams(first.question)
        second_grams = _character_bigrams(second.question)
        union = first_grams | second_grams
        similarity = len(first_grams & second_grams) / len(union) if union else 1.0
        if similarity >= threshold:
            pairs.append({
                "question_ids": [first.id, second.id],
                "character_bigram_jaccard": round(similarity, 4),
            })
    return sorted(
        pairs,
        key=lambda pair: (
            -pair["character_bigram_jaccard"],
            pair["question_ids"],
        ),
    )


def build_audit_report(
    data_dir: Path,
    package_data_dir: Path | None = None,
) -> dict[str, Any]:
    """Build a reproducible audit report and raise on ontology violations."""
    benchmark = VeraBenchLoader(str(data_dir)).load()
    stats = benchmark.stats()
    expected_conflicts = [
        conflict
        for question in benchmark.questions
        for conflict in question.expected_conflicts
    ]
    dependency_group_by_question = evidence_dependency_groups(
        benchmark.questions,
    )
    dependency_sizes = Counter(dependency_group_by_question.values())
    traceability = Counter(
        evidence_span_match_kind(
            evidence.text_span,
            benchmark.corpus[evidence.doc_id].content,
        )
        for question in benchmark.questions
        for evidence in question.evidence
    )
    document_question_counts = Counter(
        evidence.doc_id
        for question in benchmark.questions
        for evidence in question.evidence
    )
    report: dict[str, Any] = {
        "valid": True,
        "version": VERABENCH_VERSION,
        "data_dir": str(data_dir),
        "total_documents": stats["total_documents"],
        "total_questions": stats["total_questions"],
        "questions_by_type": stats["questions_by_type"],
        "multi_hop_count": stats["multi_hop_count"],
        "questions_with_conflicts": stats["conflict_count"],
        "expected_conflict_pairs": len(expected_conflicts),
        "self_conflict_pairs": sum(
            1 for conflict in expected_conflicts if conflict.pair[0] == conflict.pair[1]
        ),
        "annotated_ontology_corrections": sum(
            1 for question in benchmark.questions if question.annotation_rationale
        ),
        "annotated_difficulty_corrections": sum(
            1 for question in benchmark.questions if question.difficulty_rationale
        ),
        "evidence_traceability": {
            "total": sum(traceability.values()),
            "exact": traceability["exact"],
            "segmented": traceability["segmented"],
            "untraceable": traceability["untraceable"],
        },
        "evidence_dependency_groups": {
            "count": len(dependency_sizes),
            "singleton_count": sum(
                size == 1 for size in dependency_sizes.values()
            ),
            "largest_size": max(dependency_sizes.values(), default=0),
            "sizes": sorted(dependency_sizes.values(), reverse=True),
            "definition": "connected components induced by shared gold document IDs",
        },
        "gold_document_reuse": {
            "referenced_documents": len(document_question_counts),
            "max_questions_per_document": max(
                document_question_counts.values(),
                default=0,
            ),
            "documents_used_by_multiple_questions": sum(
                count > 1 for count in document_question_counts.values()
            ),
        },
        "near_duplicate_question_pairs": {
            "method": "character-bigram-jaccard",
            "threshold": 0.5,
            "pairs": _near_duplicate_questions(benchmark.questions),
        },
        "fingerprints": {
            "corpus_sha256": _sha256(data_dir / "corpus.jsonl"),
            "questions_sha256": _sha256(data_dir / "questions.jsonl"),
        },
        "package_data_in_sync": None,
    }

    if package_data_dir is not None:
        VeraBenchLoader(str(package_data_dir)).load()
        files = ("corpus.jsonl", "questions.jsonl")
        in_sync = all(
            (data_dir / filename).read_bytes()
            == (package_data_dir / filename).read_bytes()
            for filename in files
        )
        report["package_data_in_sync"] = in_sync
        if not in_sync:
            report["valid"] = False
            report["error"] = "repository and packaged VeraBench data differ"

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Optional VeraBench data directory; defaults to repository or package data.",
    )
    parser.add_argument(
        "--package-data-dir",
        type=Path,
        default=None,
        help="Optional second VeraBench data directory to compare byte-for-byte.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON audit report path.")
    args = parser.parse_args()

    loader = VeraBenchLoader(str(args.data_dir) if args.data_dir else None)
    data_dir = loader.data_dir
    package_data_dir = args.package_data_dir
    repository_data_dir = PROJECT_ROOT / "data" / "verabench"
    bundled_data_dir = PROJECT_ROOT / "src" / "benchmark" / "data" / "verabench"
    if (
        package_data_dir is None
        and repository_data_dir.exists()
        and bundled_data_dir.exists()
        and data_dir.resolve() == repository_data_dir.resolve()
    ):
        package_data_dir = bundled_data_dir

    report = build_audit_report(data_dir, package_data_dir)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
