"""Validate independently annotated external conflict benchmark sets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/external/conflict_mini_v1"),
        help="External conflict set directory containing corpus/questions/manifest/annotations/adjudications.",
    )
    parser.add_argument(
        "--internal-data-dir",
        type=Path,
        default=Path("data/verabench"),
        help="Internal VeraBench directory used only for fingerprint separation checks.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON audit report path.")
    parser.add_argument(
        "--min-questions",
        type=int,
        default=1,
        help="Minimum question count required for this audit.",
    )
    parser.add_argument(
        "--min-annotators-per-question",
        type=int,
        default=2,
        help="Minimum independent annotator labels required per question.",
    )
    parser.add_argument(
        "--min-conflict-kappa",
        type=float,
        default=0.6,
        help="Minimum Cohen's kappa for binary conflict-present labels.",
    )
    args = parser.parse_args()

    from src.benchmark.external_annotations import audit_external_conflict_set

    report = audit_external_conflict_set(
        args.data_dir,
        internal_data_dir=args.internal_data_dir,
        min_questions=args.min_questions,
        min_annotators_per_question=args.min_annotators_per_question,
        min_conflict_kappa=args.min_conflict_kappa,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
