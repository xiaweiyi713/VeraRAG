#!/usr/bin/env python3
"""Validate runnable examples and documented quickstart commands."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExampleAudit:
    valid: bool
    errors: list[str]
    checks: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "checks": self.checks,
        }


def validate_examples(
    project_root: str | Path = ".",
    *,
    python: str = sys.executable,
    max_demo_questions: int = 2,
) -> ExampleAudit:
    """Run the no-key quickstart and verify that docs expose the same command."""
    root = Path(project_root)
    errors: list[str] = []
    checks: list[str] = []

    quickstart = root / "examples" / "quickstart.py"
    checks.append("examples/quickstart.py exists")
    if not quickstart.is_file():
        errors.append("examples/quickstart.py is missing")
        return ExampleAudit(False, errors, checks)

    readme = root / "README.md"
    checks.append("README documents the no-key quickstart command")
    try:
        readme_text = readme.read_text(encoding="utf-8")
    except FileNotFoundError:
        errors.append("README.md is missing")
        readme_text = ""
    if "python examples/quickstart.py" not in readme_text:
        errors.append("README missing python examples/quickstart.py quickstart command")

    checks.append("no-key quickstart demo runs successfully")
    if max_demo_questions < 1:
        errors.append("max_demo_questions must be positive")
        return ExampleAudit(False, errors, checks)

    command = [
        python,
        str(quickstart),
        "--max-demo-questions",
        str(max_demo_questions),
    ]
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    result = subprocess.run(
        command,
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        errors.append(
            "quickstart no-key demo failed with exit code "
            f"{result.returncode}: {result.stderr.strip() or result.stdout.strip()}"
        )
    else:
        expected_snippets = (
            "VeraBench loaded",
            "documents: 57",
            "questions: 152",
            f"completed: {max_demo_questions}/{max_demo_questions}",
            "Demo evaluation",
        )
        for snippet in expected_snippets:
            if snippet not in result.stdout:
                errors.append(f"quickstart output missing expected text: {snippet}")

    return ExampleAudit(valid=not errors, errors=errors, checks=checks)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-demo-questions", type=int, default=2)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)

    audit = validate_examples(
        args.project_root,
        python=args.python,
        max_demo_questions=args.max_demo_questions,
    )
    if args.json:
        print(json.dumps(audit.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    elif audit.valid:
        print(f"Examples validated: {', '.join(audit.checks)}.")
    else:
        print("Example validation failed:", file=sys.stderr)
        for error in audit.errors:
            print(f"- {error}", file=sys.stderr)

    if not audit.valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
