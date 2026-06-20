import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.repair_verabench_checkpoint import repair_checkpoint


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_repair_checkpoint_dry_run_reports_errors_without_writing(tmp_path):
    checkpoint = tmp_path / "run.ckpt.jsonl"
    rows = [
        {"question_id": "V001", "error": None},
        {"question_id": "V084", "error": "Connection error."},
    ]
    _write_jsonl(checkpoint, rows)

    summary = repair_checkpoint(checkpoint, dry_run=True)

    assert summary["changed"] is False
    assert summary["removed_rows"] == 1
    assert summary["removed_question_ids"] == ["V084"]
    assert [json.loads(line) for line in checkpoint.read_text().splitlines()] == rows


def test_repair_checkpoint_backs_up_and_removes_errored_rows(tmp_path):
    checkpoint = tmp_path / "run.ckpt.jsonl"
    backup = tmp_path / "run.ckpt.jsonl.bak"
    _write_jsonl(
        checkpoint,
        [
            {"question_id": "V001", "error": None},
            {"question_id": "V084", "error": "Connection error."},
            {"question_id": "V085", "error": "Connection error."},
        ],
    )

    summary = repair_checkpoint(checkpoint, backup_path=backup)

    assert summary["changed"] is True
    assert summary["backup_path"] == str(backup)
    assert summary["kept_rows"] == 1
    assert summary["removed_question_ids"] == ["V084", "V085"]
    assert [json.loads(line) for line in checkpoint.read_text().splitlines()] == [
        {"question_id": "V001", "error": None},
    ]
    assert "V085" in backup.read_text(encoding="utf-8")


def test_repair_checkpoint_refuses_to_overwrite_backup(tmp_path):
    checkpoint = tmp_path / "run.ckpt.jsonl"
    backup = tmp_path / "backup.jsonl"
    _write_jsonl(checkpoint, [{"question_id": "V084", "error": "boom"}])
    backup.write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        repair_checkpoint(checkpoint, backup_path=backup)


def test_repair_checkpoint_cli_outputs_summary(tmp_path):
    checkpoint = tmp_path / "run.ckpt.jsonl"
    backup = tmp_path / "backup.jsonl"
    _write_jsonl(checkpoint, [{"question_id": "V084", "error": "boom"}])
    script = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "repair_verabench_checkpoint.py"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            str(checkpoint),
            "--backup",
            str(backup),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["changed"] is True
    assert payload["removed_question_ids"] == ["V084"]
    assert checkpoint.read_text(encoding="utf-8") == ""
