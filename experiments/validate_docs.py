#!/usr/bin/env python3
"""Validate local Markdown links and anchors used by project documentation."""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore[no-redef]

DEFAULT_PATHS = (
    "README.md",
    "CHANGELOG.md",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "docs",
)
DEFAULT_EXCLUDED_RELATIVE_PREFIXES = ("docs/superpowers/",)
LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)\s]+(?:\s+\"[^\"]*\")?)\)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
MAKE_COMMAND_RE = re.compile(r"(?<![\w-])make\s+([A-Za-z0-9][A-Za-z0-9_.-]*)")
CONSOLE_COMMAND_NAME_RE = re.compile(r"^verarag-[A-Za-z0-9][A-Za-z0-9-]*$")
MAKE_TARGET_RE = re.compile(r"^([A-Za-z0-9_.-]+)\s*:(?!=)", re.MULTILINE)
ALLOWED_SCHEMES = {"http", "https", "mailto"}


@dataclass(frozen=True)
class DocLinkIssue:
    path: str
    line_number: int
    target: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "line_number": self.line_number,
            "target": self.target,
            "message": self.message,
        }


def validate_docs(
    paths: list[str | Path] | None = None, *, root: str | Path = "."
) -> dict[str, Any]:
    project_root = Path(root).resolve()
    docs = _collect_markdown_files(project_root, paths or list(DEFAULT_PATHS))
    anchors = {path: _anchors_for(path) for path in docs}
    issues: list[DocLinkIssue] = []
    checked_links = 0
    checked_command_references = 0

    for doc in docs:
        for line_number, target in _iter_markdown_links(doc):
            if _is_external_or_nonlocal(target):
                continue
            checked_links += 1
            issues.extend(_validate_local_target(doc, line_number, target, anchors, project_root))
        command_issues, command_reference_count = _validate_documented_commands(doc, project_root)
        issues.extend(command_issues)
        checked_command_references += command_reference_count

    documented_console_scripts = {
        name
        for doc in docs
        for _, command_type, name in _iter_command_references(doc)
        if command_type == "console"
    }
    console_scripts = _read_console_scripts(project_root)
    for script in sorted(set(console_scripts) - documented_console_scripts):
        issues.append(
            DocLinkIssue(
                path="pyproject.toml",
                line_number=0,
                target=script,
                message="console script is not documented in Markdown docs",
            )
        )

    return {
        "valid": not issues,
        "root": str(project_root),
        "documents": [str(path.relative_to(project_root)) for path in docs],
        "checked_links": checked_links,
        "checked_command_references": checked_command_references,
        "issues": [issue.to_dict() for issue in issues],
    }


def _collect_markdown_files(root: Path, paths: list[str | Path]) -> list[Path]:
    docs: list[Path] = []
    for path_like in paths:
        path = (root / path_like).resolve()
        if path.is_file() and path.suffix.lower() == ".md":
            if not _is_default_excluded(path, root):
                docs.append(path)
        elif path.is_dir():
            docs.extend(
                sorted(
                    p.resolve()
                    for p in path.rglob("*.md")
                    if not _is_default_excluded(p.resolve(), root)
                )
            )
        else:
            raise FileNotFoundError(path)
    return sorted(set(docs))


def _is_default_excluded(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        return False
    return any(relative.startswith(prefix) for prefix in DEFAULT_EXCLUDED_RELATIVE_PREFIXES)


def _anchors_for(path: Path) -> set[str]:
    anchors: set[str] = set()
    counts: Counter[str] = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = HEADING_RE.match(line)
        if not match:
            continue
        base = _github_slug(match.group(2))
        index = counts[base]
        counts[base] += 1
        anchors.add(base if index == 0 else f"{base}-{index}")
    return anchors


def _github_slug(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = text.lower()
    text = re.sub(r"[^\w\s\-\u4e00-\u9fff]", "", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text


def _iter_markdown_links(path: Path) -> list[tuple[int, str]]:
    links: list[tuple[int, str]] = []
    in_fence = False
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in LINK_RE.finditer(line):
            raw_target = match.group(1).strip()
            target = raw_target.split()[0].strip("<>")
            links.append((line_number, target))
    return links


def _iter_command_references(path: Path) -> list[tuple[int, str, str]]:
    references: list[tuple[int, str, str]] = []
    in_fence = False
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            references.extend(_commands_in_text(line_number, line, inline=False))
            continue
        for match in INLINE_CODE_RE.finditer(line):
            references.extend(_commands_in_text(line_number, match.group(1), inline=True))
    return references


def _commands_in_text(
    line_number: int, text: str, *, inline: bool
) -> list[tuple[int, str, str]]:
    references: list[tuple[int, str, str]] = []
    tokens = [_clean_command_token(token) for token in text.split()]
    for index, token in enumerate(tokens):
        if not token:
            continue
        candidate = _console_command_from_token(token)
        previous = tokens[index - 1] if index > 0 else ""
        if candidate and previous not in {"-t", "-s", "--session"} and (
            not inline or len(tokens) > 1
        ):
            references.append((line_number, "console", candidate))
    for match in MAKE_COMMAND_RE.finditer(text):
        references.append((line_number, "make", match.group(1)))
    return references


def _clean_command_token(token: str) -> str:
    return token.strip("`'\"()[]{}:,;")


def _console_command_from_token(token: str) -> str | None:
    if "/" in token:
        if "/bin/verarag-" not in token:
            return None
        token = Path(token).name
    if CONSOLE_COMMAND_NAME_RE.fullmatch(token):
        return token
    return None


def _validate_documented_commands(doc: Path, root: Path) -> tuple[list[DocLinkIssue], int]:
    console_scripts = _read_console_scripts(root)
    make_targets = _read_make_targets(root)
    if not console_scripts and not make_targets:
        return [], 0

    issues: list[DocLinkIssue] = []
    references = _iter_command_references(doc)
    for line_number, command_type, name in references:
        if command_type == "console" and console_scripts and name not in console_scripts:
            issues.append(
                DocLinkIssue(
                    path=str(doc.relative_to(root)),
                    line_number=line_number,
                    target=name,
                    message="documented console script is not declared in pyproject.toml",
                )
            )
        elif command_type == "make" and make_targets and name not in make_targets:
            issues.append(
                DocLinkIssue(
                    path=str(doc.relative_to(root)),
                    line_number=line_number,
                    target=f"make {name}",
                    message="documented make target is not declared in Makefile",
                )
            )
    return issues, len(references)


def _read_console_scripts(root: Path) -> set[str]:
    path = root / "pyproject.toml"
    if not path.is_file():
        return set()
    with path.open("rb") as handle:
        pyproject = tomllib.load(handle)
    scripts = pyproject.get("project", {}).get("scripts", {})
    if not isinstance(scripts, dict):
        return set()
    return {str(name) for name in scripts}


def _read_make_targets(root: Path) -> set[str]:
    path = root / "Makefile"
    if not path.is_file():
        return set()
    return set(MAKE_TARGET_RE.findall(path.read_text(encoding="utf-8")))


def _is_external_or_nonlocal(target: str) -> bool:
    if target.startswith("http://") or target.startswith("https://"):
        return True
    if SCHEME_RE.match(target):
        return target.split(":", 1)[0].lower() in ALLOWED_SCHEMES
    return False


def _validate_local_target(
    doc: Path,
    line_number: int,
    target: str,
    anchors: dict[Path, set[str]],
    root: Path,
) -> list[DocLinkIssue]:
    issues: list[DocLinkIssue] = []
    path_part, fragment = _split_target(target)
    target_path = doc if not path_part else (doc.parent / path_part).resolve()

    if path_part and not _is_within(target_path, root):
        return [
            DocLinkIssue(
                path=str(doc.relative_to(root)),
                line_number=line_number,
                target=target,
                message="local link escapes project root",
            )
        ]
    if path_part and not target_path.exists():
        return [
            DocLinkIssue(
                path=str(doc.relative_to(root)),
                line_number=line_number,
                target=target,
                message="local link target does not exist",
            )
        ]
    if fragment:
        anchor_path = target_path
        if anchor_path.is_dir():
            issues.append(
                DocLinkIssue(
                    path=str(doc.relative_to(root)),
                    line_number=line_number,
                    target=target,
                    message="anchor target is a directory",
                )
            )
        elif anchor_path.suffix.lower() != ".md":
            issues.append(
                DocLinkIssue(
                    path=str(doc.relative_to(root)),
                    line_number=line_number,
                    target=target,
                    message="anchor target is not a Markdown file",
                )
            )
        elif _normalize_fragment(fragment) not in anchors.get(anchor_path, set()):
            issues.append(
                DocLinkIssue(
                    path=str(doc.relative_to(root)),
                    line_number=line_number,
                    target=target,
                    message="Markdown anchor does not exist",
                )
            )
    return issues


def _split_target(target: str) -> tuple[str, str]:
    path_part, _, fragment = target.partition("#")
    path_part = urllib.parse.unquote(path_part)
    return path_part, fragment


def _normalize_fragment(fragment: str) -> str:
    return urllib.parse.unquote(fragment).lower()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Markdown files or directories to validate.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Project root.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)

    report = validate_docs(args.paths or None, root=args.root)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    elif report["valid"]:
        print(
            "Documentation links validated: "
            f"{len(report['documents'])} documents, "
            f"{report['checked_links']} local links, "
            f"{report['checked_command_references']} command references."
        )
    else:
        print("Documentation link issues detected:", file=sys.stderr)
        for issue in report["issues"]:
            print(
                f"{issue['path']}:{issue['line_number']}: "
                f"{issue['target']}: {issue['message']}",
                file=sys.stderr,
            )
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
