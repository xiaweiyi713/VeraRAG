"""Tests for runnable examples."""

import subprocess
import sys


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
