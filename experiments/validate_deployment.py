#!/usr/bin/env python3
"""Validate Docker and Web deployment configuration drift."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_DOCKERIGNORE_PATTERNS = (
    ".git",
    ".venv",
    ".verarag_key",
    "build/",
    "dist/",
    "htmlcov/",
    "outputs/",
    "results/",
    "verarag.egg-info/",
    "data/verarag.db",
)


@dataclass(frozen=True)
class DeploymentAudit:
    valid: bool
    errors: list[str]
    checks: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "checks": self.checks,
        }


def validate_deployment(project_root: str | Path = ".") -> DeploymentAudit:
    """Validate Dockerfile, .dockerignore, Makefile, README, and Web health route."""
    root = Path(project_root)
    errors: list[str] = []
    checks: list[str] = []

    dockerfile = _read(root / "Dockerfile", errors)
    dockerignore = _read(root / ".dockerignore", errors)
    makefile = _read(root / "Makefile", errors)
    readme = _read(root / "README.md", errors)
    web_api = _read(root / "web" / "api.py", errors)

    checks.append("Dockerfile uses packaged Web entrypoint on port 8000")
    _require(dockerfile, "FROM python:", "Dockerfile should use a Python base image", errors)
    _require(dockerfile, "EXPOSE 8000", "Dockerfile should expose port 8000", errors)
    _require(
        dockerfile,
        'CMD ["verarag-web", "--host", "0.0.0.0", "--port", "8000"]',
        "Dockerfile should run the installed verarag-web entrypoint on 0.0.0.0:8000",
        errors,
    )

    checks.append("Dockerfile installs release package and drops root privileges")
    _require(
        dockerfile,
        "pip install --no-cache-dir .",
        "Dockerfile should install the package, not rely on editable source mode",
        errors,
    )
    _require(dockerfile, "USER verarag", "Dockerfile should run as a non-root user", errors)
    _require(
        dockerfile,
        "chown -R verarag:verarag /app",
        "Dockerfile should make /app writable by the runtime user",
        errors,
    )

    checks.append("Dockerfile healthcheck targets the Web status endpoint")
    _require(dockerfile, "HEALTHCHECK", "Dockerfile should define a HEALTHCHECK", errors)
    _require(
        dockerfile,
        "http://127.0.0.1:8000/api/status",
        "Dockerfile HEALTHCHECK should query /api/status on port 8000",
        errors,
    )
    _require(
        web_api,
        '@router.get("/api/status")',
        "Web API should expose /api/status for container health checks",
        errors,
    )

    checks.append(".dockerignore excludes local secrets, caches, and build artifacts")
    for pattern in REQUIRED_DOCKERIGNORE_PATTERNS:
        if pattern not in dockerignore.splitlines():
            errors.append(f".dockerignore missing {pattern}")

    checks.append("Makefile and README document matching Docker commands")
    _require(makefile, "docker-build:", "Makefile missing docker-build target", errors)
    _require(makefile, "docker build -t verarag .", "docker-build target should build verarag image", errors)
    _require(makefile, "docker-run:", "Makefile missing docker-run target", errors)
    _require(
        makefile,
        "docker run -p 8000:8000 -v $(PWD)/data:/app/data verarag",
        "docker-run target should publish 8000 and mount data volume",
        errors,
    )
    _require(readme, "make docker-build", "README should document make docker-build", errors)
    _require(readme, "make docker-run", "README should document make docker-run", errors)

    return DeploymentAudit(valid=not errors, errors=errors, checks=checks)


def _read(path: Path, errors: list[str]) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        errors.append(f"missing required deployment file: {path.name}")
        return ""


def _require(content: str, needle: str, message: str, errors: list[str]) -> None:
    if needle not in content:
        errors.append(message)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)

    audit = validate_deployment(args.project_root)
    if args.json:
        print(json.dumps(audit.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    elif audit.valid:
        print(f"Deployment config validated: {', '.join(audit.checks)}.")
    else:
        print("Deployment config validation failed:")
        for error in audit.errors:
            print(f"- {error}")

    if not audit.valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
