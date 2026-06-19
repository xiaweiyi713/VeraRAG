"""Compile completed external conflict annotation packets into audit files."""

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
        "--packet-dir",
        type=Path,
        required=True,
        help="Directory containing packet_manifest.json and completed templates.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where audit-ready corpus/questions/annotations/adjudications are written.",
    )
    parser.add_argument(
        "--adjudicator-id",
        default="adjudicator",
        help="Default adjudicator id used when template rows leave it blank.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing compiled output files.",
    )
    args = parser.parse_args()

    from src.benchmark.external_annotations import compile_external_annotation_packet

    compiled = compile_external_annotation_packet(
        args.packet_dir,
        args.output_dir,
        adjudicator_id=args.adjudicator_id,
        overwrite=args.overwrite,
    )
    print(json.dumps(compiled, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
