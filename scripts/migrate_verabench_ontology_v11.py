#!/usr/bin/env python3
"""Apply the VeraBench v1.1 conflict-ontology corrections.

The original annotations mixed three distinct tasks:

1. evidence-evidence disagreement,
2. evidence that refutes an overgeneralized user premise, and
3. ordinary factual comparison.

This migration keeps ``conflict`` only for questions with an explicit gold
evidence conflict. Premise corrections move to ``misleading`` and direct
factual questions move to their evidence-count type.
"""

import argparse
import json
from pathlib import Path
from typing import Any

PREMISE_REFUTATION_IDS = {
    "V067",
    "V068",
    "V069",
    "V071",
    "V072",
    "V074",
    "V121",
    "V123",
    "V124",
    "V125",
    "V126",
    "V127",
}

PREMISE_RATIONALE = (
    "Premise refutation: the evidence narrows or rejects the user's inference; "
    "the cited evidence items do not make mutually incompatible claims."
)

DIFFICULTY_OVERRIDES = {
    "V017": (
        "medium",
        "Conflict resolution is required; the v1.1 rubric excludes conflicts from easy.",
    ),
    "V019": (
        "medium",
        "Conflict resolution is required; the v1.1 rubric excludes conflicts from easy.",
    ),
    "V021": (
        "medium",
        "Single-hop question with one conflict pair under the v1.1 difficulty rubric.",
    ),
    "V041": (
        "medium",
        "Conflict resolution is required; the v1.1 rubric excludes conflicts from easy.",
    ),
    "V048": (
        "medium",
        "Single-hop premise correction with one evidence item under the v1.1 rubric.",
    ),
    "V074": (
        "medium",
        "Single-hop premise correction with one evidence item under the v1.1 rubric.",
    ),
    "V075": (
        "medium",
        "Single-hop question with one conflict pair under the v1.1 difficulty rubric.",
    ),
    "V151": (
        "medium",
        "Single-hop premise correction with one evidence item under the v1.1 rubric.",
    ),
}


def migrate_question(question: dict[str, Any]) -> dict[str, Any]:
    """Return a migrated copy of one VeraBench question."""
    migrated = dict(question)
    question_id = migrated.get("id")

    if question_id in PREMISE_REFUTATION_IDS:
        migrated["type"] = "misleading"
        migrated["expected_behavior"] = "correct_premise"
        migrated["expected_conflicts"] = []
        migrated["tags"] = ["误导" if tag == "冲突" else tag for tag in migrated.get("tags", [])]
        if "误导" not in migrated["tags"]:
            migrated["tags"].append("误导")
        migrated["annotation_rationale"] = PREMISE_RATIONALE

    if question_id == "V070":
        migrated["type"] = "single_evidence"
        migrated["expected_behavior"] = "answer_with_citation"
        migrated["expected_conflicts"] = []
        migrated["tags"] = ["比较" if tag == "冲突" else tag for tag in migrated.get("tags", [])]
        migrated["annotation_rationale"] = (
            "Direct factual comparison supported by one evidence item; "
            "no evidence disagreement or premise correction is required."
        )

    if question_id == "V120":
        migrated["type"] = "multi_evidence"
        migrated["expected_behavior"] = "answer_with_citation"
        migrated["expected_conflicts"] = []
        migrated["tags"] = ["定义" if tag == "冲突" else tag for tag in migrated.get("tags", [])]
        migrated["annotation_rationale"] = (
            "Technical distinction and regulatory treatment are complementary "
            "dimensions, not contradictory evidence."
        )

    if question_id == "V017":
        migrated["expected_conflicts"] = [
            {
                "pair": ["E2", "E2"],
                "conflict_type": "source_disagreement",
            }
        ]
        migrated["annotation_rationale"] = (
            "The reported 'indefinitely shelved' claim and its correction "
            "co-occur inside E2; E1 and E2 themselves agree that the law passed."
        )

    if question_id == "V122":
        migrated["expected_conflicts"] = [
            {
                "pair": ["E1", "E1"],
                "conflict_type": "temporal_conflict",
            }
        ]
        migrated["annotation_rationale"] = (
            "The original 2025 milestone and revised 2030 estimate both occur "
            "inside E1; E2 describes ITER's power target and is not conflicting."
        )

    if question_id in DIFFICULTY_OVERRIDES:
        difficulty, rationale = DIFFICULTY_OVERRIDES[question_id]
        migrated["difficulty"] = difficulty
        migrated["difficulty_rationale"] = rationale

    return migrated


def migrate_file(path: Path, *, check: bool = False) -> int:
    """Migrate one JSONL file and return the number of changed questions."""
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    migrated_rows = [migrate_question(row) for row in rows]
    changed = sum(before != after for before, after in zip(rows, migrated_rows, strict=True))

    if not check:
        content = "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in migrated_rows
        )
        path.write_text(content, encoding="utf-8")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[
            Path("data/verabench/questions.jsonl"),
            Path("src/benchmark/data/verabench/questions.jsonl"),
        ],
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report pending migrations without writing files.",
    )
    args = parser.parse_args()

    total_changed = 0
    for path in args.paths:
        changed = migrate_file(path, check=args.check)
        total_changed += changed
        print(f"{path}: {changed} question(s) {'pending' if args.check else 'migrated'}")

    if args.check and total_changed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
