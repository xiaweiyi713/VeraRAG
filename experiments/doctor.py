#!/usr/bin/env python3
"""Diagnose the local VeraRAG runtime environment."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MINIMUM_PYTHON = (3, 10)
REQUIRED_MODULES = ("numpy", "sklearn", "yaml", "fastapi", "uvicorn", "cryptography")
OPTIONAL_FEATURES = {
    "dense retrieval": ("sentence_transformers", "faiss"),
    "conflict training": ("torch", "datasets", "accelerate"),
    "ollama provider": ("ollama",),
    "pdf ingestion": ("fitz",),
    "evaluation plots": ("matplotlib", "seaborn"),
}
LLM_ENV_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "DEEPSEEK_API_KEY",
    "DASHSCOPE_API_KEY",
    "ZHIPUAI_API_KEY",
)
REQUIRED_PATHS = (
    "pyproject.toml",
    "README.md",
    "configs/model.yaml",
    "data/verabench/questions.jsonl",
    "data/verabench/corpus.jsonl",
)


@dataclass(frozen=True)
class DoctorReport:
    valid: bool
    errors: list[str]
    warnings: list[str]
    checks: list[str]
    python: dict[str, str]
    package: dict[str, str]
    required_modules: dict[str, bool]
    optional_features: dict[str, dict[str, Any]]
    llm_environment: dict[str, bool]
    paths: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "checks": self.checks,
            "python": self.python,
            "package": self.package,
            "required_modules": self.required_modules,
            "optional_features": self.optional_features,
            "llm_environment": self.llm_environment,
            "paths": self.paths,
        }


def run_doctor(project_root: str | Path = ".") -> DoctorReport:
    """Inspect local runtime prerequisites without importing heavyweight extras."""
    root = Path(project_root)
    checks: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []

    checks.append("Python runtime meets the supported floor")
    if sys.version_info < MINIMUM_PYTHON:
        errors.append(
            "Python runtime must be >= "
            f"{MINIMUM_PYTHON[0]}.{MINIMUM_PYTHON[1]}, found "
            f"{sys.version_info.major}.{sys.version_info.minor}"
        )

    checks.append("required runtime modules are import-discoverable")
    required_modules = {module: _module_available(module) for module in REQUIRED_MODULES}
    for module, available in required_modules.items():
        if not available:
            errors.append(f"required module missing: {module}")

    checks.append("optional feature dependencies are reported without failing the install")
    optional_features: dict[str, dict[str, Any]] = {}
    for feature, modules in OPTIONAL_FEATURES.items():
        module_status = {module: _module_available(module) for module in modules}
        missing = sorted(module for module, available in module_status.items() if not available)
        optional_features[feature] = {
            "available": not missing,
            "modules": module_status,
            "missing": missing,
        }
        if missing:
            warnings.append(
                f"optional feature unavailable: {feature} "
                f"(missing {', '.join(missing)})"
            )

    checks.append("VeraBench data and core repository files are present")
    paths = {path: (root / path).is_file() for path in REQUIRED_PATHS}
    for path, present in paths.items():
        if not present:
            errors.append(f"required project file missing: {path}")

    checks.append("LLM provider environment variables are summarized without exposing values")
    llm_environment = {name: bool(os.environ.get(name)) for name in LLM_ENV_VARS}
    if not any(llm_environment.values()):
        warnings.append(
            "no LLM API key environment variable is configured; demo/BM25 paths still work"
        )

    package = {
        "version": _package_version(),
        "location": str(root.resolve()),
    }
    python = {
        "version": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
    }

    return DoctorReport(
        valid=not errors,
        errors=errors,
        warnings=warnings,
        checks=checks,
        python=python,
        package=package,
        required_modules=required_modules,
        optional_features=optional_features,
        llm_environment=llm_environment,
        paths=paths,
    )


def _module_available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _package_version() -> str:
    try:
        return importlib.metadata.version("verarag")
    except importlib.metadata.PackageNotFoundError:
        return "source-checkout"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".", help="Project root to inspect.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable output.")
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Exit non-zero when optional features or API-key hints produce warnings.",
    )
    args = parser.parse_args(argv)

    report = run_doctor(args.project_root)
    exit_code = 0 if report.valid and not (args.fail_on_warnings and report.warnings) else 1
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        status = "passed" if report.valid else "failed"
        print(f"VeraRAG doctor {status}: {len(report.errors)} errors, {len(report.warnings)} warnings.")
        print(f"Python: {report.python['version']} ({report.python['executable']})")
        print(f"Package: {report.package['version']}")
        for error in report.errors:
            print(f"ERROR: {error}")
        for warning in report.warnings:
            print(f"WARNING: {warning}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
