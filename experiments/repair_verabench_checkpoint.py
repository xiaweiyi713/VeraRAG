#!/usr/bin/env python3
"""Repair VeraBench JSONL checkpoints for resumable reruns.

The runner treats every valid checkpoint row as completed, including rows where
``error`` is set. This helper removes failed rows after making a backup, so the
next resume reruns only those questions.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


def _default_backup_path(path: Path) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return path.with_name(f"{path.name}.bak.{timestamp}")


def _read_checkpoint(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}: malformed JSONL at line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}: line {line_number} is not a JSON object")
            rows.append(row)
    return rows


def repair_checkpoint(
    checkpoint_path: str | Path,
    *,
    backup_path: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Remove errored rows from a VeraBench checkpoint.

    Returns a summary with row counts and removed question ids. The original
    checkpoint is copied to ``backup_path`` before replacement unless
    ``dry_run`` is true or no errored rows are present.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(path)
    rows = _read_checkpoint(path)
    kept = [row for row in rows if not row.get("error")]
    removed = [row for row in rows if row.get("error")]
    summary = {
        "checkpoint": str(path),
        "dry_run": dry_run,
        "total_rows": len(rows),
        "kept_rows": len(kept),
        "removed_rows": len(removed),
        "removed_question_ids": [
            row.get("question_id") or row.get("id") or ""
            for row in removed
        ],
        "backup_path": None,
        "changed": False,
    }
    if not removed or dry_run:
        return summary

    backup = Path(backup_path) if backup_path else _default_backup_path(path)
    if backup.exists():
        raise FileExistsError(backup)
    backup.parent.mkdir(parents=True, exist_ok=True)
    backup.write_bytes(path.read_bytes())

    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept),
        encoding="utf-8",
    )
    os.replace(tmp_path, path)
    summary["backup_path"] = str(backup)
    summary["changed"] = True
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Remove errored rows from a resumable VeraBench checkpoint",
    )
    parser.add_argument("checkpoint", help="VeraBench checkpoint JSONL path")
    parser.add_argument(
        "--backup",
        help=(
            "Explicit backup path. Defaults to '<checkpoint>.bak.<timestamp>'. "
            "Existing backups are never overwritten."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report rows that would be removed without writing files.",
    )
    args = parser.parse_args(argv)
    summary = repair_checkpoint(
        args.checkpoint,
        backup_path=args.backup,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
