"""Validate difficulty labels in VeraBench questions.

Rules:
  easy: ≤1 hop, no conflicts, single evidence
  medium: ≤2 hops, ≤2 conflicts
  hard: ≥2 hops or ≥2 conflicts or multi_evidence with 3+ sources
"""
import json
import sys
from pathlib import Path


def validate_question(q: dict) -> list[str]:
    issues = []
    qtype = q.get("type", "")
    diff = q.get("difficulty", "")
    multi_hop = q.get("requires_multi_hop", False)
    n_conflicts = len(q.get("expected_conflicts", []))
    n_evidence = len(q.get("evidence", []))

    if diff == "easy":
        if multi_hop:
            issues.append("easy should not require multi-hop")
        if n_conflicts > 0:
            issues.append(f"easy should have 0 conflicts, got {n_conflicts}")
        if qtype == "multi_evidence" and n_evidence > 2:
            issues.append(f"easy should have ≤2 evidence, got {n_evidence}")

    elif diff == "medium":
        if n_conflicts > 2:
            issues.append(f"medium should have ≤2 conflicts, got {n_conflicts}")

    elif diff == "hard":
        if not multi_hop and n_conflicts < 2 and qtype != "multi_evidence":
            issues.append("hard should require multi-hop OR ≥2 conflicts OR multi_evidence")

    return issues


def main():
    path = Path(__file__).resolve().parent.parent / "data" / "verabench" / "questions.jsonl"
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    issues_count = 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            q = json.loads(line)
            issues = validate_question(q)
            if issues:
                issues_count += len(issues)
                print(f"[{q.get('id', i)}] {q.get('difficulty', '?')} — {', '.join(issues)}")

    if issues_count == 0:
        print("All difficulty labels are consistent.")
    else:
        print(f"\n{issues_count} issues found.")
        sys.exit(1)


if __name__ == "__main__":
    main()
