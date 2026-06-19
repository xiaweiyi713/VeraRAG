#!/usr/bin/env python3
"""Validate dependency metadata across packaging, local install, and docs files."""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DependencyMetadataAudit:
    valid: bool
    errors: list[str]
    checks: list[str]
    project_dependencies: list[str]
    requirement_names: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "checks": self.checks,
            "project_dependencies": self.project_dependencies,
            "requirement_names": self.requirement_names,
        }


def validate_dependency_metadata(project_root: str | Path = ".") -> DependencyMetadataAudit:
    root = Path(project_root)
    pyproject = _parse_pyproject_dependencies(root / "pyproject.toml")
    requirements = _parse_requirements(root / "requirements.txt")
    environment = (root / "environment.yml").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")

    errors: list[str] = []
    checks: list[str] = []
    requirement_names = {_requirement_name(req) for req in requirements}
    requirement_names.discard("")

    checks.append("project core dependencies are mirrored in requirements.txt")
    for dependency in pyproject.project_dependencies:
        name = _requirement_name(dependency)
        if name not in requirement_names:
            errors.append(f"requirements.txt missing project dependency {dependency}")

    checks.append("dev release-tool dependencies are mirrored in requirements.txt")
    for dependency in pyproject.optional_dependencies.get("dev", []):
        name = _requirement_name(dependency)
        if name not in requirement_names:
            errors.append(f"requirements.txt missing dev dependency {dependency}")

    checks.append("all extra references every declared optional group")
    all_extra = pyproject.optional_dependencies.get("all", [])
    referenced_groups = _referenced_extra_groups(all_extra)
    declared_groups = set(pyproject.optional_dependencies) - {"all"}
    missing_groups = sorted(declared_groups - referenced_groups)
    if missing_groups:
        errors.append(f"pyproject optional all extra missing groups: {', '.join(missing_groups)}")

    checks.append("environment.yml delegates pip dependencies to requirements.txt")
    if "- -r requirements.txt" not in environment:
        errors.append("environment.yml does not include pip -r requirements.txt")

    checks.append("environment.yml Python version satisfies pyproject requires-python")
    python_version = _environment_python_version(environment)
    if not python_version:
        errors.append("environment.yml missing python version pin")
    elif not _satisfies_requires_python(python_version, pyproject.requires_python):
        errors.append(
            f"environment.yml python={python_version} does not satisfy "
            f"requires-python {pyproject.requires_python}"
        )

    checks.append("README documents both pip and conda install paths")
    if "pip install -r requirements.txt" not in readme:
        errors.append("README missing pip install -r requirements.txt install path")
    if "conda env create -f environment.yml" not in readme:
        errors.append("README missing conda env create -f environment.yml install path")

    return DependencyMetadataAudit(
        valid=not errors,
        errors=errors,
        checks=checks,
        project_dependencies=pyproject.project_dependencies,
        requirement_names=sorted(requirement_names),
    )


@dataclass(frozen=True)
class _PyprojectDependencies:
    requires_python: str
    project_dependencies: list[str]
    optional_dependencies: dict[str, list[str]]


def _parse_pyproject_dependencies(path: Path) -> _PyprojectDependencies:
    lines = path.read_text(encoding="utf-8").splitlines()
    section = ""
    requires_python = ""
    project_dependencies: list[str] = []
    optional_dependencies: dict[str, list[str]] = {}
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line
            i += 1
            continue
        if not line or line.startswith("#"):
            i += 1
            continue
        if section == "[project]" and line.startswith("requires-python"):
            requires_python = _strip_toml_string(line.split("=", 1)[1].strip())
        elif section == "[project]" and line.startswith("dependencies"):
            value, i = _collect_toml_array(lines, i)
            project_dependencies = _literal_string_list(value)
            continue
        elif section == "[project.optional-dependencies]" and "=" in line:
            key = line.split("=", 1)[0].strip()
            value, i = _collect_toml_array(lines, i)
            optional_dependencies[key] = _literal_string_list(value)
            continue
        i += 1
    if not requires_python:
        raise ValueError("pyproject [project] missing requires-python")
    return _PyprojectDependencies(
        requires_python=requires_python,
        project_dependencies=project_dependencies,
        optional_dependencies=optional_dependencies,
    )


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


def _parse_requirements(path: Path) -> list[str]:
    requirements = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        requirements.append(line)
    return requirements


def _requirement_name(requirement: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9_.-]+)", requirement)
    if not match:
        return ""
    return match.group(1).lower().replace("_", "-")


def _referenced_extra_groups(all_extra: list[str]) -> set[str]:
    groups: set[str] = set()
    for item in all_extra:
        match = re.search(r"\[([^\]]+)\]", item)
        if not match:
            continue
        groups.update(group.strip() for group in match.group(1).split(",") if group.strip())
    return groups


def _environment_python_version(environment: str) -> str | None:
    for raw_line in environment.splitlines():
        line = raw_line.strip()
        if line.startswith("- python="):
            return line.split("=", 1)[1].strip()
    return None


def _satisfies_requires_python(version: str, requires_python: str) -> bool:
    version_tuple = _version_tuple(version)
    for spec in requires_python.split(","):
        spec = spec.strip()
        if not spec:
            continue
        if spec.startswith(">="):
            if version_tuple < _version_tuple(spec[2:]):
                return False
        elif spec.startswith(">"):
            if version_tuple <= _version_tuple(spec[1:]):
                return False
        elif spec.startswith("<="):
            if version_tuple > _version_tuple(spec[2:]):
                return False
        elif spec.startswith("<"):
            if version_tuple >= _version_tuple(spec[1:]):
                return False
        elif spec.startswith("==") and version_tuple != _version_tuple(spec[2:]):
            return False
    return True


def _version_tuple(version: str) -> tuple[int, ...]:
    parts = [int(part) for part in re.findall(r"\d+", version)[:3]]
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def _strip_toml_string(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        default=".",
        help="Source checkout root containing dependency metadata files.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON audit output.")
    args = parser.parse_args(argv)

    audit = validate_dependency_metadata(args.project_root)
    if args.json:
        print(json.dumps(audit.to_dict(), indent=2))
    elif audit.valid:
        print(
            "Dependency metadata validated: "
            f"{len(audit.project_dependencies)} project dependencies, "
            f"{len(audit.requirement_names)} requirements."
        )
    else:
        print("Dependency metadata validation failed:")
        for error in audit.errors:
            print(f"- {error}")

    if not audit.valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
