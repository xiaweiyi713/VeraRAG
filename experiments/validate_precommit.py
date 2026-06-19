#!/usr/bin/env python3
"""Validate VeraRAG's local pre-commit configuration and contributor docs."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_HOOKS: dict[str, str] = {
    "ruff-check": "python -m ruff check verarag src web experiments examples tests configs scripts demo.py demo_local.py run_web.py",
    "mypy-public-api": "python -m mypy src/ verarag/ --config-file mypy.ini",
    "secret-scan": "python experiments/scan_secrets.py",
    "version-check": "python experiments/validate_version_identity.py",
    "python-support-check": "python experiments/validate_python_support.py",
    "doctor-check": "python experiments/doctor.py",
    "configs-check": "python experiments/validate_configs.py",
    "docs-check": "python experiments/validate_docs.py",
    "results-check": "python experiments/validate_results.py",
    "examples-check": "python experiments/validate_examples.py",
    "deployment-check": "python experiments/validate_deployment.py",
    "deps-check": "python experiments/validate_dependency_metadata.py",
    "metadata-check": "python experiments/validate_project_metadata.py",
}


@dataclass(frozen=True)
class PreCommitAudit:
    valid: bool
    errors: list[str]
    checks: list[str]
    hooks: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "checks": self.checks,
            "hooks": self.hooks,
        }


def validate_precommit(project_root: str | Path = ".") -> PreCommitAudit:
    root = Path(project_root)
    config = _read(root / ".pre-commit-config.yaml")
    makefile = _read(root / "Makefile")
    pyproject = _read(root / "pyproject.toml")
    contributing = _read(root / "CONTRIBUTING.md")
    readme = _read(root / "README.md")
    tests_workflow = _read(root / ".github/workflows/test.yml")

    errors: list[str] = []
    checks: list[str] = []

    checks.append("pre-commit config exists and uses local deterministic hooks")
    if not config.strip():
        errors.append(".pre-commit-config.yaml is missing or empty")
    if "repo: local" not in config:
        errors.append(".pre-commit-config.yaml should use local hooks")
    if re.search(r"repo:\s+https?://", config):
        errors.append(".pre-commit-config.yaml should avoid remote hook repositories")

    checks.append("pre-commit hooks mirror fast local quality gates")
    hook_blocks = _hook_blocks(config)
    for hook_id, entry in REQUIRED_HOOKS.items():
        block = hook_blocks.get(hook_id)
        if block is None:
            errors.append(f"pre-commit missing hook {hook_id}")
            continue
        if f"entry: {entry}" not in block:
            errors.append(f"pre-commit hook {hook_id} entry should be {entry}")
        if "language: system" not in block:
            errors.append(f"pre-commit hook {hook_id} should use language: system")
        if "pass_filenames: false" not in block:
            errors.append(f"pre-commit hook {hook_id} should set pass_filenames: false")

    checks.append("Makefile exposes a precommit-check gate and release-check uses it")
    if "precommit-check:" not in makefile:
        errors.append("Makefile missing precommit-check target")
    if "experiments/validate_precommit.py" not in makefile:
        errors.append("Makefile precommit-check should run experiments/validate_precommit.py")
    if "release-check: lint version-check python-support-check doctor-check configs-check docs-check results-check examples-check deployment-check precommit-check" not in makefile:
        errors.append("release-check should run version-check, python-support-check, doctor-check, and configs-check before docs-check and precommit-check after deployment-check")

    checks.append("public package exposes an installed pre-commit validator")
    expected_script = 'verarag-validate-precommit = "experiments.validate_precommit:main"'
    if expected_script not in pyproject:
        errors.append("pyproject missing verarag-validate-precommit console script")

    checks.append("contributor docs explain installation and all-files execution")
    for snippet in (
        "python -m pip install pre-commit",
        "pre-commit install",
        "pre-commit run --all-files",
        "make configs-check",
        "make precommit-check",
    ):
        if snippet not in contributing:
            errors.append(f"CONTRIBUTING missing pre-commit instruction: {snippet}")
    if "make precommit-check" not in readme:
        errors.append("README missing make precommit-check quality command")
    if "make configs-check" not in readme:
        errors.append("README missing make configs-check quality command")
    if "verarag-validate-precommit" not in readme:
        errors.append("README missing verarag-validate-precommit command")

    checks.append("CI validates pre-commit configuration drift")
    if "Validate pre-commit config" not in tests_workflow:
        errors.append("GitHub Actions workflow missing Validate pre-commit config step")
    if "python experiments/validate_precommit.py" not in tests_workflow:
        errors.append("GitHub Actions workflow should run experiments/validate_precommit.py")
    if "Run environment doctor" not in tests_workflow:
        errors.append("GitHub Actions workflow missing Run environment doctor step")
    if "python experiments/doctor.py" not in tests_workflow:
        errors.append("GitHub Actions workflow should run experiments/doctor.py")
    if "Validate default configs" not in tests_workflow:
        errors.append("GitHub Actions workflow missing Validate default configs step")
    if "python experiments/validate_configs.py" not in tests_workflow:
        errors.append("GitHub Actions workflow should run experiments/validate_configs.py")

    return PreCommitAudit(
        valid=not errors,
        errors=errors,
        checks=checks,
        hooks=sorted(hook_blocks),
    )


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _hook_blocks(config: str) -> dict[str, str]:
    blocks: dict[str, str] = {}
    current_id: str | None = None
    current_lines: list[str] = []
    for line in config.splitlines():
        match = re.match(r"\s*-\s+id:\s+([A-Za-z0-9_.-]+)\s*$", line)
        if match:
            if current_id is not None:
                blocks[current_id] = "\n".join(current_lines)
            current_id = match.group(1)
            current_lines = [line]
            continue
        if current_id is not None:
            current_lines.append(line)
    if current_id is not None:
        blocks[current_id] = "\n".join(current_lines)
    return blocks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".", help="Project root to validate.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args(argv)

    audit = validate_precommit(args.project_root)
    if args.json:
        print(json.dumps(audit.to_dict(), indent=2, ensure_ascii=False))
    elif audit.valid:
        print("Pre-commit config validated: " + ", ".join(audit.hooks))
    else:
        print("Pre-commit config validation failed:")
        for error in audit.errors:
            print(f"- {error}")
    return 0 if audit.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
