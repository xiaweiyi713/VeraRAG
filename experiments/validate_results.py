#!/usr/bin/env python3
"""Validate the published VeraBench results and leaderboard contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ResultsAudit:
    valid: bool
    errors: list[str]
    checks: list[str]
    docs_results_path: str
    corpus_sha256: str
    questions_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "checks": self.checks,
            "docs_results_path": self.docs_results_path,
            "corpus_sha256": self.corpus_sha256,
            "questions_sha256": self.questions_sha256,
        }


def validate_results(project_root: str | Path = ".") -> ResultsAudit:
    root = Path(project_root)
    results_path = root / "docs" / "RESULTS.md"
    evaluation_path = root / "docs" / "EVALUATION.md"
    card_path = root / "docs" / "VERABENCH_CARD.md"
    corpus_path = root / "data" / "verabench" / "corpus.jsonl"
    questions_path = root / "data" / "verabench" / "questions.jsonl"

    errors: list[str] = []
    checks: list[str] = []
    results = _read(results_path)
    evaluation = _read(evaluation_path)
    card = _read(card_path)

    checks.append("docs/RESULTS.md exists and declares its generation command")
    if not results.strip():
        errors.append("docs/RESULTS.md is missing or empty")
    command = _generation_command(results)
    if command is None:
        errors.append("docs/RESULTS.md missing Generation command block")
    else:
        for snippet in (
            "python experiments/build_verabench_leaderboard.py",
            "--output docs/RESULTS.md",
            "--allow-unverified",
            "--allow-mixed-benchmarks",
        ):
            if snippet not in command:
                errors.append(
                    "docs/RESULTS.md generation command missing " + snippet
                )

    corpus_sha256 = _sha256(corpus_path)
    questions_sha256 = _sha256(questions_path)
    checks.append("published VeraBench v1.1.2 fingerprints match repository data")
    for label, digest in (
        ("Corpus SHA-256", corpus_sha256),
        ("Questions SHA-256", questions_sha256),
    ):
        if digest not in results:
            errors.append(f"docs/RESULTS.md missing current {label}: {digest}")
    for snippet in (
        "Current repository data is VeraBench v1.1.2",
        "v1.0 and v1.1 rows are historical and must not be",
        "v1.1.2 targeted",
        "configs/verabench_v112_canonical.yaml",
        "verabench_v112_canonical_deepseek.json",
    ):
        if snippet not in results:
            errors.append(f"docs/RESULTS.md missing result provenance note: {snippet}")

    checks.append("historical and diagnostic tables are labeled non-leaderboard")
    for snippet in (
        "legacy, non-comparable historical runs",
        "Historical VeraBench v1.1 Full Run",
        "Historical v1.0 Leaderboard",
        "Focused Conflict Smoke",
        "not a full leaderboard entry",
    ):
        if snippet not in results:
            errors.append(f"docs/RESULTS.md missing historical label: {snippet}")

    checks.append("published tables expose reproducibility and conflict columns")
    required_table_headers = (
        "| Run | Model | Questions | Errors | Behavior Acc | Answer F1 | Evidence Recall | Conflict micro-F1 |",
        "| Rank | Run | Model | Questions | Errors | Behavior Acc | Answer F1 | Evidence Recall | Conflict F1 | ECE | Avg Latency | Commit |",
        "| Run | Provider | Model | Config | Timestamp | Result Path |",
    )
    for header in required_table_headers:
        if header not in results:
            errors.append(f"docs/RESULTS.md missing required table header: {header}")

    checks.append("formal leaderboard docs explain publication integrity checks")
    docs_text = "\n".join([evaluation, card])
    for snippet in (
        "rejects demo",
        "incomplete or errored",
        "missing reproducibility metadata",
        "mixed benchmark fingerprints",
        "--allow-unverified",
        "--allow-mixed-benchmarks",
    ):
        if snippet not in docs_text:
            errors.append(f"leaderboard documentation missing integrity rule: {snippet}")

    return ResultsAudit(
        valid=not errors,
        errors=errors,
        checks=checks,
        docs_results_path=str(results_path),
        corpus_sha256=corpus_sha256,
        questions_sha256=questions_sha256,
    )


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _generation_command(markdown: str) -> str | None:
    match = re.search(
        r"Generation command:\n\n```bash\n(?P<command>.*?)\n```",
        markdown,
        re.DOTALL,
    )
    if match is None:
        return None
    return re.sub(r"\\\n\s*", " ", match.group("command")).strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".", help="Project root to validate.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args(argv)

    audit = validate_results(args.project_root)
    if args.json:
        print(json.dumps(audit.to_dict(), indent=2, ensure_ascii=False))
    elif audit.valid:
        print(
            "Published results validated: "
            f"{audit.docs_results_path} "
            f"({len(audit.checks)} checks)."
        )
    else:
        print("Published results validation failed:")
        for error in audit.errors:
            print(f"- {error}")
    return 0 if audit.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
