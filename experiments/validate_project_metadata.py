#!/usr/bin/env python3
"""Validate open-source project metadata, governance files, and citation data."""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

REQUIRED_ROOT_FILES = (
    "README.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
    "CITATION.cff",
    "LICENSE",
)
REQUIRED_GITHUB_FILES = (
    ".github/CODEOWNERS",
    ".github/dependabot.yml",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/ISSUE_TEMPLATE/config.yml",
    ".github/ISSUE_TEMPLATE/bug_report.yml",
    ".github/ISSUE_TEMPLATE/feature_request.yml",
    ".github/workflows/codeql.yml",
    ".github/workflows/release.yml",
    ".github/workflows/scorecard.yml",
    ".github/workflows/test.yml",
)
REQUIRED_PROJECT_URLS = (
    "Homepage",
    "Repository",
    "Documentation",
    "Issues",
    "Changelog",
    "Security",
    "Citation",
)
README_REQUIRED_LINKS = (
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
    "CITATION.cff",
    "CHANGELOG.md",
)
CONTRIBUTING_REQUIRED_COMMANDS = (
    "make lint",
    "make version-check",
    "make python-support-check",
    "make docs-check",
    "make results-check",
    "make examples-check",
    "make deployment-check",
    "make precommit-check",
    "make deps-check",
    "make package-check",
    "make release-check",
)
WORKFLOW_ACTION_PATTERN = re.compile(r"^\s*-?\s*uses:\s*([^#\s]+)", re.MULTILINE)
PINNED_ACTION_REF_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+@[0-9a-f]{40}$")


@dataclass(frozen=True)
class ProjectMetadataAudit:
    valid: bool
    errors: list[str]
    checks: list[str]
    project_name: str
    project_version: str
    project_urls: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "checks": self.checks,
            "project_name": self.project_name,
            "project_version": self.project_version,
            "project_urls": self.project_urls,
        }


def validate_project_metadata(project_root: str | Path = ".") -> ProjectMetadataAudit:
    root = Path(project_root)
    pyproject = _parse_pyproject(root / "pyproject.toml")
    citation = _parse_citation(root / "CITATION.cff")
    readme = _read(root / "README.md")
    contributing = _read(root / "CONTRIBUTING.md")
    security = _read(root / "SECURITY.md")
    code_of_conduct = _read(root / "CODE_OF_CONDUCT.md")
    license_text = _read(root / "LICENSE")
    codeowners = _read(root / ".github/CODEOWNERS")
    dependabot = _read(root / ".github/dependabot.yml")
    issue_template_config = _read(root / ".github/ISSUE_TEMPLATE/config.yml")
    codeql_workflow = _read(root / ".github/workflows/codeql.yml")
    release_workflow = _read(root / ".github/workflows/release.yml")
    scorecard_workflow = _read(root / ".github/workflows/scorecard.yml")
    workflow = _read(root / ".github/workflows/test.yml")
    workflow_files = {
        path.relative_to(root).as_posix(): _read(path)
        for path in sorted((root / ".github/workflows").glob("*.yml"))
    }

    errors: list[str] = []
    checks: list[str] = []

    checks.append("required governance files exist and are non-empty")
    for relative_path in (*REQUIRED_ROOT_FILES, *REQUIRED_GITHUB_FILES):
        path = root / relative_path
        if not path.is_file():
            errors.append(f"missing required project file {relative_path}")
        elif not path.read_text(encoding="utf-8").strip():
            errors.append(f"required project file is empty: {relative_path}")

    checks.append("pyproject exposes complete package metadata")
    project = pyproject.get("project", {})
    project_name = str(project.get("name", ""))
    project_version = str(project.get("version", ""))
    for key in ("name", "version", "description", "readme", "requires-python", "license"):
        if not project.get(key):
            errors.append(f"pyproject [project] missing {key}")
    if len(project.get("keywords", [])) < 4:
        errors.append("pyproject [project] should include at least four keywords")
    classifiers = project.get("classifiers", [])
    if any(item.startswith("License ::") for item in classifiers):
        errors.append("pyproject classifiers should not include deprecated license classifiers")
    if not any(item == "Operating System :: OS Independent" for item in classifiers):
        errors.append("pyproject classifiers missing OS independent classifier")

    checks.append("project.urls exposes PyPI/GitHub navigation targets")
    project_urls = dict(pyproject.get("project.urls", {}))
    for key in REQUIRED_PROJECT_URLS:
        value = project_urls.get(key)
        if not value:
            errors.append(f"pyproject [project.urls] missing {key}")
        elif not _is_https_url(value):
            errors.append(f"pyproject [project.urls] {key} is not an https URL")
    repository_url = project_urls.get("Repository", "")
    if repository_url and citation.get("repository-code") != repository_url:
        errors.append("CITATION.cff repository-code must match pyproject Repository URL")

    checks.append("CITATION.cff matches project identity")
    if citation.get("cff-version") != "1.2.0":
        errors.append("CITATION.cff must use cff-version 1.2.0")
    if "VeraRAG" not in str(citation.get("title", "")):
        errors.append("CITATION.cff title should include VeraRAG")
    if str(citation.get("version", "")) != project_version:
        errors.append("CITATION.cff version must match pyproject project.version")
    if str(citation.get("license", "")) != str(project.get("license", "")):
        errors.append("CITATION.cff license must match pyproject project.license")
    if not citation.get("authors"):
        errors.append("CITATION.cff must include at least one author")
    if not _looks_like_iso_date(str(citation.get("date-released", ""))):
        errors.append("CITATION.cff date-released must use YYYY-MM-DD format")

    checks.append("README surfaces governance and citation links")
    for target in README_REQUIRED_LINKS:
        if target not in readme:
            errors.append(f"README missing link or reference to {target}")

    checks.append("CONTRIBUTING points maintainers at current quality gates")
    for command in CONTRIBUTING_REQUIRED_COMMANDS:
        if command not in contributing:
            errors.append(f"CONTRIBUTING missing quality gate command: {command}")
    if "real VeraBench" not in contributing:
        errors.append("CONTRIBUTING should describe evidence for real VeraBench changes")

    checks.append("security and conduct policies mention expected escalation boundaries")
    if "private security advisory" not in security.lower():
        errors.append("SECURITY should mention private security advisory reporting")
    if "revoke or rotate" not in security:
        errors.append("SECURITY should describe key revocation or rotation")
    if "Unacceptable behavior" not in code_of_conduct:
        errors.append("CODE_OF_CONDUCT should define unacceptable behavior")
    if "MIT License" not in license_text:
        errors.append("LICENSE should contain the MIT License text")

    checks.append("GitHub community files route ownership and issue intake")
    if "*" not in codeowners or "@" not in codeowners:
        errors.append("CODEOWNERS should define a repository-wide owner")
    if "blank_issues_enabled: false" not in issue_template_config:
        errors.append("GitHub issue templates should disable blank issues")
    if "security/policy" not in issue_template_config:
        errors.append("GitHub issue templates should route vulnerabilities to SECURITY")

    checks.append("GitHub CodeQL security scanning is enabled")
    if not _uses_action(codeql_workflow, "github/codeql-action/init"):
        errors.append("CodeQL workflow should initialize github/codeql-action")
    if not _uses_action(codeql_workflow, "github/codeql-action/analyze"):
        errors.append("CodeQL workflow should run github/codeql-action/analyze")
    if "security-events: write" not in codeql_workflow:
        errors.append("CodeQL workflow should grant security-events: write")
    if "contents: read" not in codeql_workflow:
        errors.append("CodeQL workflow should grant contents: read")
    if "timeout-minutes:" not in codeql_workflow:
        errors.append("CodeQL workflow should define timeout-minutes")
    if "schedule:" not in codeql_workflow or "cron:" not in codeql_workflow:
        errors.append("CodeQL workflow should run on a recurring schedule")
    if "python" not in codeql_workflow:
        errors.append("CodeQL workflow should analyze Python")

    checks.append("GitHub OpenSSF Scorecard security analysis is enabled")
    if not _uses_action(scorecard_workflow, "ossf/scorecard-action"):
        errors.append("Scorecard workflow should run ossf/scorecard-action")
    if "security-events: write" not in scorecard_workflow:
        errors.append("Scorecard workflow should grant security-events: write")
    if "contents: read" not in scorecard_workflow:
        errors.append("Scorecard workflow should grant contents: read")
    if "timeout-minutes:" not in scorecard_workflow:
        errors.append("Scorecard workflow should define timeout-minutes")
    if "schedule:" not in scorecard_workflow or "cron:" not in scorecard_workflow:
        errors.append("Scorecard workflow should run on a recurring schedule")
    if "publish_results: true" not in scorecard_workflow:
        errors.append("Scorecard workflow should publish results")
    if "scorecard-results.json" not in scorecard_workflow:
        errors.append("Scorecard workflow should preserve a JSON artifact")

    checks.append("GitHub release workflow builds auditable artifacts and uses trusted publishing")
    if "workflow_dispatch:" not in release_workflow:
        errors.append("Release workflow should support workflow_dispatch dry-runs")
    if "tags:" not in release_workflow or '"v*.*.*"' not in release_workflow:
        errors.append("Release workflow should run on version tags")
    if "make release-check PYTHON=python" not in release_workflow:
        errors.append("Release workflow should run make release-check")
    for artifact_path in (
        "dist/*.tar.gz",
        "dist/*.whl",
        "build/sbom/verarag-sbom.cdx.json",
        "build/release-health/release-artifacts-manifest.json",
        "build/release-checksums.json",
    ):
        if artifact_path not in release_workflow:
            errors.append(f"Release workflow should upload {artifact_path}")
    if not _uses_action(release_workflow, "pypa/gh-action-pypi-publish"):
        errors.append("Release workflow should use PyPI trusted publishing action")
    if not _uses_action(release_workflow, "actions/attest-build-provenance"):
        errors.append("Release workflow should attest release artifact provenance")
    if "id-token: write" not in release_workflow:
        errors.append("Release workflow should grant id-token: write for PyPI publishing and attestation")
    if "attestations: write" not in release_workflow:
        errors.append("Release workflow should grant attestations: write for provenance")
    if "subject-path:" not in release_workflow:
        errors.append("Release workflow should attest the uploaded artifact subject paths")
    if "environment: pypi" not in release_workflow:
        errors.append("Release workflow should use a protected pypi environment")
    if "if: startsWith(github.ref, 'refs/tags/v')" not in release_workflow:
        errors.append("Release workflow should publish only from version tags")
    if "gh release create" not in release_workflow:
        errors.append("Release workflow should create GitHub releases for version tags")
    if "contents: write" not in release_workflow:
        errors.append("Release workflow should grant contents: write only for GitHub releases")
    if "actions/download-artifact" not in release_workflow:
        errors.append("Release workflow should download build artifacts for publish jobs")

    checks.append("GitHub Actions dependencies are pinned by commit SHA")
    for workflow_name, workflow_text in workflow_files.items():
        for action in _unpinned_actions(workflow_text):
            errors.append(f"{workflow_name} action should be pinned to a full commit SHA: {action}")

    checks.append("GitHub automation uses least privilege and bounded maintenance")
    if "\npermissions:\n  contents: read\n" not in f"\n{workflow}\n":
        errors.append("GitHub Actions workflow should set top-level contents: read permission")
    if "timeout-minutes:" not in workflow:
        errors.append("GitHub Actions test job should define timeout-minutes")
    if "python experiments/validate_project_metadata.py" not in workflow:
        errors.append("GitHub Actions workflow should run project metadata validation")
    if "python experiments/validate_version_identity.py" not in workflow:
        errors.append("GitHub Actions workflow should run version identity validation")
    if "python experiments/validate_python_support.py" not in workflow:
        errors.append("GitHub Actions workflow should run Python support validation")
    if "python experiments/validate_results.py" not in workflow:
        errors.append("GitHub Actions workflow should run published results validation")
    if "make package-check PYTHON=python" not in workflow:
        errors.append("GitHub Actions workflow should reuse make package-check")
    if 'package-ecosystem: "pip"' not in dependabot:
        errors.append("Dependabot should monitor pip dependencies")
    if 'package-ecosystem: "github-actions"' not in dependabot:
        errors.append("Dependabot should monitor GitHub Actions")
    if dependabot.count('interval: "weekly"') < 2:
        errors.append("Dependabot pip and GitHub Actions updates should run weekly")
    if "open-pull-requests-limit:" not in dependabot:
        errors.append("Dependabot should define open-pull-requests-limit")

    return ProjectMetadataAudit(
        valid=not errors,
        errors=errors,
        checks=checks,
        project_name=project_name,
        project_version=project_version,
        project_urls=project_urls,
    )


def _parse_pyproject(path: Path) -> dict[str, dict[str, object]]:
    metadata: dict[str, dict[str, object]] = {"project": {}, "project.urls": {}}
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
            value = value.strip()
            if key in {"name", "version", "description", "readme", "requires-python", "license"}:
                metadata["project"][key] = _strip_toml_string(value)
            elif key in {"keywords", "classifiers"}:
                array, i = _collect_toml_array(lines, i)
                metadata["project"][key] = _literal_string_list(array)
                continue
        elif section == "project.urls" and "=" in line:
            key, value = line.split("=", 1)
            metadata["project.urls"][key.strip()] = _strip_toml_string(value.strip())
        i += 1
    return metadata


def _parse_citation(path: Path) -> dict[str, object]:
    citation: dict[str, object] = {}
    current_list_key = ""
    current_author: dict[str, str] | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        stripped = raw_line.strip()
        if stripped.endswith(":") and not stripped.startswith("-"):
            current_list_key = stripped[:-1]
            citation.setdefault(current_list_key, [])
            current_author = None
            continue
        if stripped.startswith("- "):
            payload = stripped[2:]
            if ":" in payload:
                key, value = payload.split(":", 1)
                item = {key.strip(): _strip_yaml_scalar(value.strip())}
                if current_list_key:
                    cast_list = citation.setdefault(current_list_key, [])
                    if isinstance(cast_list, list):
                        cast_list.append(item)
                current_author = item
            else:
                cast_list = citation.setdefault(current_list_key, [])
                if isinstance(cast_list, list):
                    cast_list.append(_strip_yaml_scalar(payload.strip()))
                current_author = None
            continue
        if raw_line.startswith("  ") and current_author is not None and ":" in stripped:
            key, value = stripped.split(":", 1)
            current_author[key.strip()] = _strip_yaml_scalar(value.strip())
            continue
        if ":" in stripped:
            key, value = stripped.split(":", 1)
            citation[key.strip()] = _strip_yaml_scalar(value.strip())
            current_list_key = ""
            current_author = None
    return citation


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


def _strip_toml_string(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _strip_yaml_scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _is_https_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.netloc)


def _looks_like_iso_date(value: str) -> bool:
    return re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is not None


def _workflow_actions(workflow: str) -> list[str]:
    return [match.group(1).strip('"\'') for match in WORKFLOW_ACTION_PATTERN.finditer(workflow)]


def _uses_action(workflow: str, action_name: str) -> bool:
    expected_prefix = f"{action_name}@"
    return any(action.startswith(expected_prefix) for action in _workflow_actions(workflow))


def _unpinned_actions(workflow: str) -> list[str]:
    unpinned: list[str] = []
    for action in _workflow_actions(workflow):
        if action.startswith("./"):
            continue
        if not PINNED_ACTION_REF_PATTERN.fullmatch(action):
            unpinned.append(action)
    return unpinned


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        default=".",
        help="Source checkout root containing project metadata files.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON audit output.")
    args = parser.parse_args(argv)

    audit = validate_project_metadata(args.project_root)
    if args.json:
        print(json.dumps(audit.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    elif audit.valid:
        print(
            "Project metadata validated: "
            f"{audit.project_name} {audit.project_version}, "
            f"{len(audit.project_urls)} project URLs."
        )
    else:
        print("Project metadata validation failed:")
        for error in audit.errors:
            print(f"- {error}")

    if not audit.valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
