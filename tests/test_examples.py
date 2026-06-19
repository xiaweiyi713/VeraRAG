"""Tests for runnable examples."""

import json
import subprocess
import sys
from pathlib import Path

from experiments.validate_examples import validate_examples


def test_quickstart_no_key_demo_runs():
    result = subprocess.run(
        [sys.executable, "examples/quickstart.py", "--max-demo-questions", "2"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "VeraBench loaded" in result.stdout
    assert "documents: 57" in result.stdout
    assert "questions: 152" in result.stdout
    assert "completed: 2/2" in result.stdout


def test_validate_examples_accepts_repository_quickstart():
    audit = validate_examples(max_demo_questions=2)

    assert audit.valid
    assert audit.errors == []
    assert "no-key quickstart demo runs successfully" in audit.checks


def test_validate_examples_cli_emits_json():
    result = subprocess.run(
        [
            sys.executable,
            "experiments/validate_examples.py",
            "--max-demo-questions",
            "2",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["valid"] is True
    assert "README documents the no-key quickstart command" in payload["checks"]


def test_examples_gate_is_wired_into_release_check():
    makefile = Path("Makefile").read_text(encoding="utf-8")

    assert "examples-check:" in makefile
    assert "experiments/validate_examples.py" in makefile
    assert (
        "release-check: lint version-check python-support-check doctor-check configs-check docs-check results-check examples-check deployment-check precommit-check deps-check "
        "metadata-check sbom-check coverage-check benchmark-check "
        "release-artifacts-check package-check release-checksums-check"
    ) in makefile
