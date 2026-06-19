#!/usr/bin/env python
"""Download external benchmark datasets for VeraRAG evaluation.

Downloads:
  - HotpotQA (distractor subset)
  - FEVER (v1.0 shared task)
  - CKT-Conflict (if available)

Usage:
  python scripts/download_datasets.py
  python scripts/download_datasets.py --datasets hotpotqa fever
  python scripts/download_datasets.py --output data/raw
"""

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path

DATASETS = {
    "hotpotqa": {
        "url": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
        "filename": "hotpot_dev_distractor_v1.json",
        "description": "HotpotQA dev set (distractor, ~7.4K questions)",
        "fallback_url": "https://huggingface.co/datasets/hotpot_qa/resolve/main/hotpot_dev_distractor_v1.json",
    },
    "fever": {
        "url": "https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_dev.jsonl",
        "filename": "fever_dev_v1.0.jsonl",
        "description": "FEVER v1.0 dev set (~6.7K claims)",
    },
}


def download_file(url: str, output_path: str):
    """Download a file with progress."""
    print(f"  Downloading: {url}")
    urllib.request.urlretrieve(url, output_path)
    print(f"  Saved to: {output_path}")


def download_dataset(name: str, output_dir: str):
    """Download a specific dataset with fallback URL and MD5 validation."""
    info = DATASETS[name]
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, info["filename"])

    if os.path.exists(output_path):
        print(f"  [{name}] Already exists: {output_path}")
        return

    print(f"  [{name}] {info['description']}")
    urls = [info["url"]]
    if "fallback_url" in info:
        urls.append(info["fallback_url"])

    for url in urls:
        try:
            download_file(url, output_path)
            size = os.path.getsize(output_path)
            if size < 1000:
                raise ValueError(f"File too small ({size} bytes), likely an error page")
            expected_md5 = info.get("md5")
            if expected_md5:
                with open(output_path, "rb") as f:
                    actual_md5 = hashlib.md5(f.read()).hexdigest()
                if actual_md5 != expected_md5:
                    raise ValueError(f"MD5 mismatch: expected {expected_md5}, got {actual_md5}")
            print(f"  Size: {size / 1024:.1f} KB")
            return
        except Exception as e:
            print(f"  Failed: {url} -- {e}")
            if os.path.exists(output_path):
                os.remove(output_path)

    print(f"  [{name}] All download attempts failed")


def create_sample_ckt_conflict(output_dir: str):
    """Create a sample CKT-Conflict dataset from VeraBench corpus."""
    ckt_dir = os.path.join(output_dir, "ckt_conflict")
    os.makedirs(ckt_dir, exist_ok=True)
    output_path = os.path.join(ckt_dir, "samples.json")

    if os.path.exists(output_path):
        print("  [ckt_conflict] Already exists")
        return

    project_root = Path(__file__).resolve().parent.parent
    corpus_path = project_root / "data" / "verabench" / "corpus.jsonl"

    if not corpus_path.exists():
        print("  [ckt_conflict] VeraBench corpus not found, skipping")
        return

    # Generate CKT-Conflict format samples from VeraBench conflict questions
    questions_path = project_root / "data" / "verabench" / "questions.jsonl"
    samples = []
    if questions_path.exists():
        with open(questions_path, encoding="utf-8") as f:
            for line in f:
                q = json.loads(line.strip())
                if q.get("type") == "conflict":
                    samples.append(
                        {
                            "id": q["id"],
                            "question": q["question"],
                            "answer": q.get("ground_truth_answer", ""),
                            "conflicting_evidence": [
                                {"source": c.get("doc_id", ""), "target": c.get("doc_id", "")}
                                for c in q.get("evidence", [])[:2]
                            ],
                        }
                    )

    if samples:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"samples": samples, "count": len(samples)}, f, ensure_ascii=False, indent=2)
        print(f"  [ckt_conflict] Created {len(samples)} samples")
    else:
        print("  [ckt_conflict] No conflict samples found")


def main():
    parser = argparse.ArgumentParser(description="Download benchmark datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[*list(DATASETS.keys()), "ckt_conflict", "all"],
        default=["all"],
        help="Datasets to download",
    )
    parser.add_argument("--output", type=str, default="data/raw", help="Output directory")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    output_dir = os.path.join(project_root, args.output)

    datasets = args.datasets
    if "all" in datasets:
        datasets = [*list(DATASETS.keys()), "ckt_conflict"]

    print(f"Downloading datasets to {output_dir}\n")

    for name in datasets:
        if name == "ckt_conflict":
            create_sample_ckt_conflict(output_dir)
        elif name in DATASETS:
            download_dataset(name, output_dir)
        else:
            print(f"  Unknown dataset: {name}")

    print("\nDone!")


if __name__ == "__main__":
    main()
