"""Tests for release checksum manifest generation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from experiments.generate_release_checksums import (
    build_release_checksum_manifest,
    validate_release_checksum_manifest,
    write_release_checksum_manifest,
)


def test_release_checksum_manifest_includes_package_sbom_and_health_artifacts(tmp_path):
    dist_dir, sbom, health_manifest = _write_release_inputs(tmp_path)

    manifest = build_release_checksum_manifest(
        project_root=tmp_path,
        dist_dir=dist_dir,
        sbom_path=sbom,
        release_health_manifest=health_manifest,
    )

    assert manifest["schema_version"] == 1
    assert manifest["project"] == {"name": "verarag", "version": "0.1.0"}
    roles = {artifact["role"] for artifact in manifest["artifacts"]}
    assert roles == {
        "sdist",
        "wheel",
        "cyclonedx_sbom",
        "release_health_artifact_manifest",
    }
    for artifact in manifest["artifacts"]:
        assert len(artifact["sha256"]) == 64
        assert artifact["bytes"] > 0


def test_release_checksum_manifest_validation_accepts_fresh_manifest(tmp_path):
    dist_dir, sbom, health_manifest = _write_release_inputs(tmp_path)
    manifest_path = tmp_path / "build/release-checksums.json"
    write_release_checksum_manifest(
        build_release_checksum_manifest(
            project_root=tmp_path,
            dist_dir=dist_dir,
            sbom_path=sbom,
            release_health_manifest=health_manifest,
        ),
        manifest_path,
    )

    audit = validate_release_checksum_manifest(manifest_path, manifest_root=tmp_path)

    assert audit.valid
    assert audit.artifact_count == 4
    assert audit.errors == []


def test_release_checksum_manifest_validation_rejects_tampered_artifact(tmp_path):
    dist_dir, sbom, health_manifest = _write_release_inputs(tmp_path)
    manifest_path = tmp_path / "build/release-checksums.json"
    write_release_checksum_manifest(
        build_release_checksum_manifest(
            project_root=tmp_path,
            dist_dir=dist_dir,
            sbom_path=sbom,
            release_health_manifest=health_manifest,
        ),
        manifest_path,
    )
    (dist_dir / "verarag-0.1.0.tar.gz").write_text("changed", encoding="utf-8")

    audit = validate_release_checksum_manifest(manifest_path, manifest_root=tmp_path)

    assert not audit.valid
    assert "dist/verarag-0.1.0.tar.gz: sha256 mismatch" in audit.errors
    assert "dist/verarag-0.1.0.tar.gz: byte size mismatch" in audit.errors


def test_release_checksum_manifest_validation_rejects_path_escape(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    manifest_path = root / "release-checksums.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifacts": [
                    {
                        "role": "sdist",
                        "path": "../escaped.tar.gz",
                        "sha256": "0" * 64,
                        "bytes": 1,
                    },
                    _artifact("wheel", "dist/verarag-0.1.0-py3-none-any.whl"),
                    _artifact("cyclonedx_sbom", "build/sbom/verarag-sbom.cdx.json"),
                    _artifact(
                        "release_health_artifact_manifest",
                        "build/release-health/release-artifacts-manifest.json",
                    ),
                ],
            }
        ),
        encoding="utf-8",
    )

    audit = validate_release_checksum_manifest(manifest_path, manifest_root=root)

    assert not audit.valid
    assert "../escaped.tar.gz: artifact path must stay inside manifest root" in audit.errors


def test_release_checksum_cli_writes_and_validates_json(tmp_path):
    dist_dir, sbom, health_manifest = _write_release_inputs(tmp_path)
    output = tmp_path / "build/release-checksums.json"

    result = subprocess.run(
        [
            sys.executable,
            "experiments/generate_release_checksums.py",
            "--project-root",
            str(tmp_path),
            "--dist-dir",
            str(dist_dir),
            "--sbom",
            str(sbom),
            "--release-health-manifest",
            str(health_manifest),
            "--output",
            str(output),
            "--check",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert output.is_file()
    payload = json.loads(result.stdout)
    assert payload["valid"] is True
    assert payload["artifact_count"] == 4


def test_release_checksum_cli_validates_existing_manifest(tmp_path):
    dist_dir, sbom, health_manifest = _write_release_inputs(tmp_path)
    manifest_path = tmp_path / "build/release-checksums.json"
    write_release_checksum_manifest(
        build_release_checksum_manifest(
            project_root=tmp_path,
            dist_dir=dist_dir,
            sbom_path=sbom,
            release_health_manifest=health_manifest,
        ),
        manifest_path,
    )

    result = subprocess.run(
        [
            sys.executable,
            "experiments/generate_release_checksums.py",
            "--validate-manifest",
            str(manifest_path),
            "--manifest-root",
            str(tmp_path),
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"valid": true' in result.stdout
    assert '"artifact_count": 4' in result.stdout


def test_release_checksum_make_target_runs_after_package_check():
    makefile = Path("Makefile").read_text(encoding="utf-8")

    assert "RELEASE_CHECKSUMS ?= build/release-checksums.json" in makefile
    assert "release-checksums-check:" in makefile
    assert "experiments/generate_release_checksums.py --output $(RELEASE_CHECKSUMS) --check" in makefile
    assert (
        "release-check: lint version-check python-support-check doctor-check configs-check docs-check results-check examples-check deployment-check precommit-check deps-check "
        "metadata-check sbom-check coverage-check benchmark-check "
        "release-artifacts-check package-check release-checksums-check"
    ) in makefile


def _write_release_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = \"verarag\"\nversion = \"0.1.0\"\n",
        encoding="utf-8",
    )
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "verarag-0.1.0.tar.gz").write_bytes(b"sdist")
    (dist / "verarag-0.1.0-py3-none-any.whl").write_bytes(b"wheel")
    sbom = tmp_path / "build/sbom/verarag-sbom.cdx.json"
    sbom.parent.mkdir(parents=True)
    sbom.write_text("{}", encoding="utf-8")
    health = tmp_path / "build/release-health/release-artifacts-manifest.json"
    health.parent.mkdir(parents=True)
    health.write_text("{}", encoding="utf-8")
    return dist, sbom, health


def _artifact(role: str, path: str) -> dict[str, object]:
    return {
        "role": role,
        "path": path,
        "sha256": "0" * 64,
        "bytes": 1,
    }
