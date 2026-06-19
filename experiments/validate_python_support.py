#!/usr/bin/env python3
"""Validate Python support metadata across packaging, CI, tooling, and docs."""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXPECTED_PYTHON_VERSIONS = ("3.10", "3.11", "3.12", "3.13")
MINIMUM_PYTHON = EXPECTED_PYTHON_VERSIONS[0]
RUFF_TARGET = "py310"


@dataclass(frozen=True)
class PythonSupportAudit:
    valid: bool
    errors: list[str]
    checks: list[str]
    requires_python: str
    classifiers: list[str]
    ci_versions: list[str]
    ruff_target: str
    mypy_python_version: str
    pyproject_mypy_python_version: str
    environment_python: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "checks": self.checks,
            "requires_python": self.requires_python,
            "classifiers": self.classifiers,
            "ci_versions": self.ci_versions,
            "ruff_target": self.ruff_target,
            "mypy_python_version": self.mypy_python_version,
            "pyproject_mypy_python_version": self.pyproject_mypy_python_version,
            "environment_python": self.environment_python,
        }


def validate_python_support(project_root: str | Path = ".") -> PythonSupportAudit:
    root = Path(project_root)
    pyproject = _parse_pyproject(root / "pyproject.toml")
    workflow = _read(root / ".github" / "workflows" / "test.yml")
    ruff = _read(root / "ruff.toml")
    mypy = _read(root / "mypy.ini")
    environment = _read(root / "environment.yml")
    readme = _read(root / "README.md")
    contributing = _read(root / "CONTRIBUTING.md")

    requires_python = _metadata_scalar(pyproject, "project", "requires-python")
    classifiers = _metadata_string_list(pyproject, "project", "classifiers")
    ci_versions = _ci_python_versions(workflow)
    ruff_target = _toml_scalar(ruff, "target-version")
    mypy_python_version = _ini_scalar(mypy, "python_version")
    pyproject_mypy_python_version = _metadata_scalar(
        pyproject, "tool.mypy", "python_version"
    )
    environment_python = _environment_python_version(environment)

    errors: list[str] = []
    checks: list[str] = []

    checks.append("pyproject declares the intended Python support floor")
    if requires_python != f">={MINIMUM_PYTHON}":
        errors.append(
            f"pyproject requires-python must be >={MINIMUM_PYTHON}, "
            f"found {requires_python or '<missing>'}"
        )

    checks.append("pyproject classifiers match the supported Python matrix")
    expected_classifiers = {
        "Programming Language :: Python :: 3",
        *{
            f"Programming Language :: Python :: {version}"
            for version in EXPECTED_PYTHON_VERSIONS
        },
    }
    missing_classifiers = sorted(set(expected_classifiers) - set(classifiers))
    for classifier in missing_classifiers:
        errors.append(f"pyproject classifiers missing {classifier}")

    checks.append("GitHub Actions test matrix covers every supported Python version")
    if tuple(ci_versions) != EXPECTED_PYTHON_VERSIONS:
        errors.append(
            "GitHub Actions python-version matrix must be "
            f"{list(EXPECTED_PYTHON_VERSIONS)}, found {ci_versions}"
        )
    if "python-version: ${{ matrix.python-version }}" not in workflow:
        errors.append("GitHub Actions setup-python should use matrix.python-version")

    checks.append("static analysis and conda defaults target the support floor")
    if ruff_target != RUFF_TARGET:
        errors.append(f"ruff.toml target-version must be {RUFF_TARGET}, found {ruff_target or '<missing>'}")
    if mypy_python_version != MINIMUM_PYTHON:
        errors.append(
            f"mypy.ini python_version must be {MINIMUM_PYTHON}, "
            f"found {mypy_python_version or '<missing>'}"
        )
    if pyproject_mypy_python_version != MINIMUM_PYTHON:
        errors.append(
            f"pyproject tool.mypy python_version must be {MINIMUM_PYTHON}, "
            f"found {pyproject_mypy_python_version or '<missing>'}"
        )
    if environment_python != MINIMUM_PYTHON:
        errors.append(
            f"environment.yml python pin must be {MINIMUM_PYTHON}, "
            f"found {environment_python or '<missing>'}"
        )

    checks.append("README and CONTRIBUTING document the public support floor")
    for path, text in (("README.md", readme), ("CONTRIBUTING.md", contributing)):
        if "Python 3.10+" not in text:
            errors.append(f"{path} missing Python 3.10+ support statement")
    if "python-3.10%2B-blue" not in readme:
        errors.append("README Python badge should point at python-3.10+")

    return PythonSupportAudit(
        valid=not errors,
        errors=errors,
        checks=checks,
        requires_python=requires_python,
        classifiers=classifiers,
        ci_versions=ci_versions,
        ruff_target=ruff_target,
        mypy_python_version=mypy_python_version,
        pyproject_mypy_python_version=pyproject_mypy_python_version,
        environment_python=environment_python,
    )


def _parse_pyproject(path: Path) -> dict[str, dict[str, object]]:
    metadata: dict[str, dict[str, object]] = {"project": {}, "tool.mypy": {}}
    section = ""
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line.strip("[]")
            i += 1
            continue
        if section == "project" and "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            if key == "requires-python":
                metadata["project"][key] = _strip_string(value.strip())
            elif key == "classifiers":
                array, i = _collect_toml_array(lines, i)
                metadata["project"][key] = _literal_string_list(array)
                continue
        elif section == "tool.mypy" and line.startswith("python_version"):
            metadata["tool.mypy"]["python_version"] = _strip_string(
                line.split("=", 1)[1].strip()
            )
        i += 1
    return metadata


def _collect_toml_array(lines: list[str], start: int) -> tuple[str, int]:
    collected = [lines[start].split("=", 1)[1].strip()]
    i = start + 1
    while not _toml_array_complete("\n".join(collected)) and i < len(lines):
        collected.append(lines[i].strip())
        i += 1
    return "\n".join(collected), i


def _toml_array_complete(value: str) -> bool:
    depth = 0
    quote: str | None = None
    escaped = False
    saw_array = False
    for char in value:
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
        elif char == "[":
            saw_array = True
            depth += 1
        elif char == "]":
            depth -= 1
    return saw_array and depth == 0 and quote is None


def _literal_string_list(value: str) -> list[str]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected TOML array, found {value}")
    return [str(item) for item in parsed]


def _metadata_scalar(
    metadata: dict[str, dict[str, object]], section: str, key: str
) -> str:
    return str(metadata.get(section, {}).get(key, ""))


def _metadata_string_list(
    metadata: dict[str, dict[str, object]], section: str, key: str
) -> list[str]:
    value = metadata.get(section, {}).get(key, [])
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _ci_python_versions(workflow: str) -> list[str]:
    match = re.search(r"python-version:\s*\[(?P<versions>[^\]]+)\]", workflow)
    if match is None:
        return []
    return re.findall(r"\d+\.\d+", match.group("versions"))


def _toml_scalar(text: str, key: str) -> str:
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=\s*(?P<value>.+?)\s*$", re.MULTILINE)
    match = pattern.search(text)
    return _strip_string(match.group("value").strip()) if match else ""


def _ini_scalar(text: str, key: str) -> str:
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=\s*(?P<value>.+?)\s*$", re.MULTILINE)
    match = pattern.search(text)
    return match.group("value").strip() if match else ""


def _environment_python_version(environment: str) -> str:
    for raw_line in environment.splitlines():
        line = raw_line.strip()
        if line.startswith("- python="):
            return line.split("=", 1)[1].strip()
    return ""


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _strip_string(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".", help="Project root to validate.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args(argv)

    audit = validate_python_support(args.project_root)
    if args.json:
        print(json.dumps(audit.to_dict(), indent=2, ensure_ascii=False))
    elif audit.valid:
        print(
            "Python support validated: "
            f"{audit.requires_python}, CI {', '.join(audit.ci_versions)}."
        )
    else:
        print("Python support validation failed:")
        for error in audit.errors:
            print(f"- {error}")
    return 0 if audit.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
