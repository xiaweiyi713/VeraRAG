#!/usr/bin/env python3
"""Validate release version identity across package, source, citation, and docs."""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VersionIdentityAudit:
    valid: bool
    errors: list[str]
    checks: list[str]
    project_name: str
    project_version: str
    source_version: str
    citation_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "checks": self.checks,
            "project_name": self.project_name,
            "project_version": self.project_version,
            "source_version": self.source_version,
            "citation_version": self.citation_version,
        }


def validate_version_identity(project_root: str | Path = ".") -> VersionIdentityAudit:
    root = Path(project_root)
    pyproject = _parse_pyproject_project(root / "pyproject.toml")
    project_name = pyproject.get("name", "")
    project_version = pyproject.get("version", "")
    source_version = _parse_source_version(root / "src" / "__init__.py")
    citation_version = _parse_citation_version(root / "CITATION.cff")
    readme = _read(root / "README.md")
    releasing = _read(root / "docs" / "RELEASING.md")
    changelog = _read(root / "CHANGELOG.md")

    errors: list[str] = []
    checks: list[str] = []

    checks.append("pyproject declares a normalized semantic project version")
    if not project_name:
        errors.append("pyproject [project] missing name")
    if not project_version:
        errors.append("pyproject [project] missing version")
    elif not re.fullmatch(r"\d+\.\d+\.\d+(?:[a-zA-Z0-9_.+-]+)?", project_version):
        errors.append(f"pyproject project.version is not normalized semver-like: {project_version}")

    checks.append("source checkout fallback version matches package metadata")
    if not source_version:
        errors.append("src/__init__.py missing literal __version__")
    elif source_version != project_version:
        errors.append(
            f"src.__version__ must match pyproject project.version: "
            f"{source_version} != {project_version}"
        )

    checks.append("citation metadata version matches package metadata")
    if not citation_version:
        errors.append("CITATION.cff missing version")
    elif citation_version != project_version:
        errors.append(
            f"CITATION.cff version must match pyproject project.version: "
            f"{citation_version} != {project_version}"
        )

    checks.append("public package reads installed metadata and falls back to source version")
    public_init = _read(root / "verarag" / "__init__.py")
    for snippet in (
        "importlib_metadata.version",
        "PackageNotFoundError",
        "from src import __version__ as _SOURCE_VERSION",
        "__version__ = _read_package_version()",
    ):
        if snippet not in public_init:
            errors.append(f"verarag/__init__.py missing version fallback snippet: {snippet}")

    checks.append("release docs tell maintainers which version files must move together")
    for snippet in (
        "Update version in `pyproject.toml`, `src/__init__.py`, and `CITATION.cff`",
        "make version-check",
        "verarag-validate-version --json",
    ):
        if snippet not in releasing:
            errors.append(f"docs/RELEASING.md missing version instruction: {snippet}")

    checks.append("README and changelog expose the version gate")
    if "make version-check" not in readme:
        errors.append("README missing make version-check quality command")
    if "verarag-validate-version" not in readme:
        errors.append("README missing verarag-validate-version command")
    if "verarag-validate-version" not in changelog:
        errors.append("CHANGELOG missing verarag-validate-version entry")

    return VersionIdentityAudit(
        valid=not errors,
        errors=errors,
        checks=checks,
        project_name=project_name,
        project_version=project_version,
        source_version=source_version,
        citation_version=citation_version,
    )


def _parse_pyproject_project(path: Path) -> dict[str, str]:
    section = ""
    metadata: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line.strip("[]")
            continue
        if section == "project" and "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            if key in {"name", "version"}:
                metadata[key] = _strip_string(value.strip())
    return metadata


def _parse_source_version(path: Path) -> str:
    module = ast.parse(path.read_text(encoding="utf-8"))
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if (
            any(
                isinstance(target, ast.Name) and target.id == "__version__"
                for target in node.targets
            )
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return node.value.value
    return ""


def _parse_citation_version(path: Path) -> str:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("version:"):
            return _strip_string(line.split(":", 1)[1].strip())
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

    audit = validate_version_identity(args.project_root)
    if args.json:
        print(json.dumps(audit.to_dict(), indent=2, ensure_ascii=False))
    elif audit.valid:
        print(
            "Version identity validated: "
            f"{audit.project_name} {audit.project_version}."
        )
    else:
        print("Version identity validation failed:")
        for error in audit.errors:
            print(f"- {error}")
    return 0 if audit.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
