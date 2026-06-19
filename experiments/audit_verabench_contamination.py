#!/usr/bin/env python3
"""Audit VeraBench overlap against caller-supplied local reference corpora."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Optional VeraBench data directory; defaults to packaged/repository data.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        action="append",
        default=[],
        help="Local reference file or directory. Supports txt, md, json, and jsonl.",
    )
    parser.add_argument("--near-threshold", type=float, default=0.85)
    parser.add_argument("--containment-threshold", type=float, default=0.85)
    parser.add_argument("--ngram-size", type=int, default=13)
    parser.add_argument("--min-exact-chars", type=int, default=8)
    parser.add_argument("--max-matches", type=int, default=50)
    parser.add_argument("--output", type=Path, help="Optional JSON audit report path.")
    parser.add_argument(
        "--fail-on-high-risk-match",
        action="store_true",
        help="Exit non-zero if question or ground-truth-answer overlap is detected.",
    )
    args = parser.parse_args()

    from src.benchmark.contamination import audit_verabench_contamination

    report = audit_verabench_contamination(
        args.data_dir,
        reference_paths=args.reference,
        near_threshold=args.near_threshold,
        containment_threshold=args.containment_threshold,
        ngram_size=args.ngram_size,
        min_exact_chars=args.min_exact_chars,
        max_matches=args.max_matches,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    summary = report["summary"]
    high_risk = (
        summary["high_risk_exact_matches"]
        + summary["high_risk_near_duplicate_matches"]
    )
    if args.fail_on_high_risk_match and high_risk:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
