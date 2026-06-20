import json
import subprocess
import sys
from pathlib import Path

from experiments.verabench_status import render_text, summarize_status


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_summarize_status_reports_checkpoint_progress(tmp_path):
    checkpoint = tmp_path / "run.ckpt.jsonl"
    _write_jsonl(
        checkpoint,
        [
            {"question_id": "V001", "question_type": "single_evidence"},
            {
                "question_id": "V084",
                "question_type": "temporal",
                "error": "Connection error.",
            },
        ],
    )

    summary = summarize_status(checkpoint_path=checkpoint)

    assert summary["checkpoint"]["rows"] == 2
    assert summary["checkpoint"]["errors"] == 1
    assert summary["checkpoint"]["last_question_id"] == "V084"
    assert summary["checkpoint"]["error_question_ids"] == ["V084"]
    assert "rows=2 errors=1 last=V084" in render_text(summary)


def test_summarize_status_reports_final_report_errors(tmp_path):
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps({
            "total_questions": 2,
            "completed": 2,
            "errors": 1,
            "question_results": [
                {"question_id": "V001"},
                {"question_id": "V085", "error": "Connection error."},
            ],
        }),
        encoding="utf-8",
    )

    summary = summarize_status(report_path=report)

    assert summary["report"]["exists"] is True
    assert summary["report"]["completed"] == 2
    assert summary["report"]["error_question_ids"] == ["V085"]


def test_verabench_status_cli_outputs_json(tmp_path):
    checkpoint = tmp_path / "run.ckpt.jsonl"
    _write_jsonl(checkpoint, [{"question_id": "V001"}])
    script = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "verabench_status.py"
    )

    result = subprocess.run(
        [sys.executable, str(script), "--checkpoint", str(checkpoint), "--json"],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["checkpoint"]["rows"] == 1
    assert payload["checkpoint"]["last_question_id"] == "V001"
