"""Build blind annotation packets for external conflict benchmark sets."""

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
        required=True,
        help="VeraBench-compatible dataset directory with corpus.jsonl and questions.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where annotation templates and packet_manifest.json are written.",
    )
    parser.add_argument(
        "--annotator",
        action="append",
        dest="annotators",
        required=True,
        help="Annotator id. Repeat for each independent annotator.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing packet_manifest.json and templates.",
    )
    args = parser.parse_args()

    from src.benchmark.external_annotations import build_external_annotation_packet

    packet = build_external_annotation_packet(
        args.data_dir,
        args.output_dir,
        annotator_ids=args.annotators,
        overwrite=args.overwrite,
    )
    print(json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
