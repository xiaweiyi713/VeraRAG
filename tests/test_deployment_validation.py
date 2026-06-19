"""Tests for Docker and deployment configuration validation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from experiments.validate_deployment import validate_deployment


def test_deployment_validation_accepts_repository_config():
    audit = validate_deployment()

    assert audit.valid
    assert audit.errors == []
    assert "Dockerfile healthcheck targets the Web status endpoint" in audit.checks


def test_deployment_validation_reports_drift(tmp_path):
    _write_deployment_fixture(
        tmp_path,
        dockerfile="FROM python:3.12-slim\nEXPOSE 8000\nCMD [\"python\"]\n",
        dockerignore=".git\n",
        web_api='@router.get("/api/status")\n',
    )

    audit = validate_deployment(tmp_path)

    assert not audit.valid
    assert "Dockerfile should run the installed verarag-web entrypoint on 0.0.0.0:8000" in audit.errors
    assert "Dockerfile should define a HEALTHCHECK" in audit.errors
    assert "Dockerfile should run as a non-root user" in audit.errors
    assert ".dockerignore missing .venv" in audit.errors


def test_deployment_validation_cli_emits_json():
    result = subprocess.run(
        [
            sys.executable,
            "experiments/validate_deployment.py",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["valid"] is True
    assert payload["errors"] == []


def test_deployment_gate_is_wired_into_release_check():
    makefile = Path("Makefile").read_text(encoding="utf-8")

    assert "deployment-check:" in makefile
    assert "experiments/validate_deployment.py" in makefile
    assert (
        "release-check: lint version-check python-support-check doctor-check configs-check docs-check results-check examples-check deployment-check precommit-check deps-check "
        "metadata-check sbom-check coverage-check benchmark-check "
        "release-artifacts-check package-check release-checksums-check"
    ) in makefile


def _write_deployment_fixture(
    root: Path,
    *,
    dockerfile: str,
    dockerignore: str,
    web_api: str,
) -> None:
    (root / "Dockerfile").write_text(dockerfile, encoding="utf-8")
    (root / ".dockerignore").write_text(dockerignore, encoding="utf-8")
    (root / "Makefile").write_text(
        "docker-build:\n\tdocker build -t verarag .\n"
        "docker-run:\n\tdocker run -p 8000:8000 -v $(PWD)/data:/app/data verarag\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text("make docker-build\nmake docker-run\n", encoding="utf-8")
    web = root / "web"
    web.mkdir()
    (web / "api.py").write_text(web_api, encoding="utf-8")
