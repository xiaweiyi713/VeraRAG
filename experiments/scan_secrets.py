#!/usr/bin/env python3
"""Scan repository text files for high-confidence leaked secrets."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "htmlcov",
    "venv",
}
TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".css",
    ".env",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".ts",
    ".txt",
    ".yaml",
    ".yml",
}
MAX_FILE_BYTES = 1_000_000
ALLOWLIST_MARKERS = (
    "${",
    "<key>",
    "dummy",
    "example",
    "placeholder",
    "sk-test",
    "sk-xxx",
    "test-key",
    "your-api-key",
)


@dataclass(frozen=True)
class SecretFinding:
    path: str
    line_number: int
    rule: str
    redacted: str


SECRET_PATTERNS = (
    (
        "openai_or_deepseek_style_key",
        re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9][A-Za-z0-9_-]{20,}\b"),
    ),
    (
        "anthropic_style_key",
        re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"),
    ),
    (
        "github_token",
        re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b"),
    ),
    (
        "github_fine_grained_token",
        re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}\b"),
    ),
    (
        "aws_access_key_id",
        re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    ),
    (
        "private_key_header",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    ),
)
SECRET_RULE_DESCRIPTIONS = {
    "openai_or_deepseek_style_key": "OpenAI or DeepSeek style sk-* API key.",
    "anthropic_style_key": "Anthropic style API key.",
    "github_token": "GitHub classic token.",
    "github_fine_grained_token": "GitHub fine-grained personal access token.",
    "aws_access_key_id": "AWS access key id.",
    "private_key_header": "Private key material header.",
    "generic_secret_assignment": "Generic literal secret assignment.",
    "env_secret_assignment": "Environment-style literal secret assignment.",
}
ASSIGNMENT_PATTERN = re.compile(
    r"""(?ix)
    \b[A-Za-z0-9_]*(?:api[_-]?key|secret|token|access[_-]?key)\b
    \s*[:=]\s*
    ["']
    (?P<value>[A-Za-z0-9][A-Za-z0-9_./+=:-]{19,})
    ["']
    """
)
ENV_ASSIGNMENT_PATTERN = re.compile(
    r"""(?ix)
    ^\s*(?:export\s+)?
    [A-Za-z_][A-Za-z0-9_]*(?:api[_-]?key|secret|token|access[_-]?key)[A-Za-z0-9_]*
    \s*=\s*
    ["']?
    (?P<value>[A-Za-z0-9][A-Za-z0-9_./+=:-]{19,})
    ["']?
    (?:\s*(?:\#.*)?)$
    """
)


def scan_paths(
    paths: Iterable[str | Path], *, include_ignored: bool = False
) -> list[SecretFinding]:
    findings: list[SecretFinding] = []
    for path in _candidate_files(paths, include_ignored=include_ignored):
        findings.extend(_scan_file(path))
    return findings


def _candidate_files(
    paths: Iterable[str | Path], *, include_ignored: bool
) -> list[Path]:
    candidates: list[Path] = []
    for path_like in paths:
        path = Path(path_like)
        if path.is_file():
            candidates.append(path)
        elif path.is_dir():
            candidates.extend(_iter_directory(path, include_ignored=include_ignored))
        else:
            raise FileNotFoundError(path)
    return sorted({path.resolve() for path in candidates})


def _iter_directory(root: Path, *, include_ignored: bool) -> list[Path]:
    if not include_ignored and (git_files := _git_list_files(root)) is not None:
        return [
            root / file_path
            for file_path in git_files
            if _is_scannable_file(root / file_path)
        ]
    return [
        file_path
        for file_path in root.rglob("*")
        if _is_scannable_file(file_path)
    ]


def _git_list_files(root: Path) -> list[Path] | None:
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return [Path(line) for line in result.stdout.splitlines() if line.strip()]


def _is_scannable_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if any(part in SKIP_DIRS for part in path.parts):
        return False
    if not _has_scannable_name(path):
        return False
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return False
    except OSError:
        return False
    return True


def _scan_file(path: Path) -> list[SecretFinding]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []
    findings: list[SecretFinding] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        findings.extend(_scan_line(path, line_number, line))
    return findings


def _scan_line(path: Path, line_number: int, line: str) -> list[SecretFinding]:
    findings: list[SecretFinding] = []
    for rule, pattern in SECRET_PATTERNS:
        for match in pattern.finditer(line):
            token = match.group(0)
            if _is_allowlisted(token, line):
                continue
            findings.append(
                SecretFinding(
                    path=str(path),
                    line_number=line_number,
                    rule=rule,
                    redacted=_redact(token),
                )
            )
    assignment = ASSIGNMENT_PATTERN.search(line)
    if assignment:
        token = assignment.group("value").rstrip("',\"}")
        if not _is_allowlisted(token, line):
            findings.append(
                SecretFinding(
                    path=str(path),
                    line_number=line_number,
                    rule="generic_secret_assignment",
                    redacted=_redact(token),
                )
            )
    env_assignment = None if assignment else ENV_ASSIGNMENT_PATTERN.search(line)
    if env_assignment:
        token = env_assignment.group("value").rstrip("',\"}")
        if not _is_allowlisted(token, line):
            findings.append(
                SecretFinding(
                    path=str(path),
                    line_number=line_number,
                    rule="env_secret_assignment",
                    redacted=_redact(token),
                )
            )
    return findings


def _is_allowlisted(token: str, line: str) -> bool:
    normalized_token = token.lower()
    normalized_line = line.lower()
    return any(
        marker in normalized_token or marker in normalized_line
        for marker in ALLOWLIST_MARKERS
    )


def _redact(token: str) -> str:
    if len(token) <= 10:
        return "***"
    return f"{token[:6]}...{token[-4:]}"


def _has_scannable_name(path: Path) -> bool:
    name = path.name.lower()
    if name == ".env" or name.startswith(".env."):
        return True
    return path.suffix.lower() in TEXT_SUFFIXES


def findings_to_sarif(findings: Iterable[SecretFinding], *, root: Path | None = None) -> dict:
    """Return a SARIF 2.1.0 report without exposing raw secret values."""
    root = (root or Path.cwd()).resolve()
    findings_list = list(findings)
    rule_ids = sorted({finding.rule for finding in findings_list})
    rules = [
        {
            "id": rule_id,
            "name": rule_id,
            "shortDescription": {
                "text": SECRET_RULE_DESCRIPTIONS.get(rule_id, rule_id),
            },
            "fullDescription": {
                "text": (
                    "VeraRAG detected a high-confidence secret pattern. "
                    "Rotate the credential if it was committed or shared."
                ),
            },
            "defaultConfiguration": {"level": "error"},
        }
        for rule_id in rule_ids
    ]
    return {
        "version": "2.1.0",
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "verarag-scan-secrets",
                        "informationUri": "https://github.com/xiaweiyi713/VeraRAG",
                        "rules": rules,
                    }
                },
                "results": [
                    {
                        "ruleId": finding.rule,
                        "level": "error",
                        "message": {
                            "text": (
                                f"Potential secret detected by {finding.rule}: "
                                f"{finding.redacted}"
                            ),
                        },
                        "locations": [
                            {
                                "physicalLocation": {
                                    "artifactLocation": {
                                        "uri": _sarif_uri(Path(finding.path), root=root),
                                    },
                                    "region": {"startLine": finding.line_number},
                                }
                            }
                        ],
                    }
                    for finding in findings_list
                ],
            }
        ],
    }


def _sarif_uri(path: Path, *, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        default=["."],
        help="Files or directories to scan. Defaults to the current repository.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON findings instead of text.",
    )
    parser.add_argument(
        "--sarif",
        action="store_true",
        help="Emit a SARIF 2.1.0 report for code-scanning tools.",
    )
    parser.add_argument(
        "--include-ignored",
        action="store_true",
        help=(
            "Also scan ignored local files such as .env.local. Useful for "
            "workstation audits; CI should normally keep the default."
        ),
    )
    args = parser.parse_args(argv)
    if args.json and args.sarif:
        parser.error("--json and --sarif are mutually exclusive.")

    findings = scan_paths(args.paths, include_ignored=args.include_ignored)
    if args.json:
        print(json.dumps([finding.__dict__ for finding in findings], indent=2))
    elif args.sarif:
        print(json.dumps(findings_to_sarif(findings), indent=2))
    elif findings:
        print("Potential secrets detected:")
        for finding in findings:
            print(
                f"{finding.path}:{finding.line_number}: "
                f"{finding.rule}: {finding.redacted}"
            )
    else:
        print("No high-confidence secrets detected.")

    if findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
