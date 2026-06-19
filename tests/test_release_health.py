"""Tests for release-critical VeraBench health validation."""

import hashlib
import json
import subprocess
import sys

import pytest

from experiments.validate_release_health import (
    validate_release_artifact_manifest,
    validate_release_health,
)


def test_validate_release_health_runs_release_critical_checks(tmp_path):
    report = validate_release_health(
        output_dir=tmp_path / "health",
        comparison_resamples=20,
    )

    assert report["valid"]
    assert [check["name"] for check in report["checks"]] == [
        "verabench_audit",
        "external_conflict_fixture",
        "external_annotation_packet",
        "demo_metric_plumbing",
        "demo_paired_comparison",
        "release_artifact_manifest",
    ]
    by_name = {check["name"]: check for check in report["checks"]}
    assert by_name["verabench_audit"]["evidence_refs"] == 208
    assert by_name["verabench_audit"]["dependency_groups"] == 27
    assert by_name["demo_metric_plumbing"]["completed"] == 152
    assert by_name["demo_metric_plumbing"]["answer_f1"] == 1.0
    assert by_name["demo_paired_comparison"]["questions"] == 152
    assert by_name["demo_paired_comparison"]["answer_f1_delta"] == 0.0
    assert by_name["release_artifact_manifest"]["artifacts"] == 5
    assert report["artifact_manifest"].endswith("release-artifacts-manifest.json")


def test_validate_release_health_cli_emits_json(tmp_path):
    output_dir = tmp_path / "health"
    result = subprocess.run(
        [
            sys.executable,
            "experiments/validate_release_health.py",
            "--output-dir",
            str(output_dir),
            "--comparison-resamples",
            "20",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"valid": true' in result.stdout
    assert '"artifact_manifest":' in result.stdout
    assert "Release health validated" not in result.stdout


def test_release_artifact_manifest_rejects_tampered_artifacts(tmp_path):
    output_dir = tmp_path / "health"
    report = validate_release_health(
        output_dir=output_dir,
        comparison_resamples=20,
    )
    (output_dir / "verabench-demo.json").write_text("{}", encoding="utf-8")

    audit = validate_release_artifact_manifest(
        report["artifact_manifest"],
        output_dir=output_dir,
    )

    assert not audit["valid"]
    assert "verabench-demo.json: sha256 mismatch" in audit["errors"]
    assert "verabench-demo.json: byte size mismatch" in audit["errors"]


def test_release_artifact_manifest_cli_validates_existing_manifest(tmp_path):
    output_dir = tmp_path / "health"
    report = validate_release_health(
        output_dir=output_dir,
        comparison_resamples=20,
    )

    result = subprocess.run(
        [
            sys.executable,
            "experiments/validate_release_health.py",
            "--validate-manifest",
            report["artifact_manifest"],
            "--manifest-root",
            str(output_dir),
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"valid": true' in result.stdout
    assert '"artifacts": 5' in result.stdout
    assert "Release artifact manifest validated" not in result.stdout


def test_release_artifact_manifest_cli_rejects_bad_hash(tmp_path):
    root = tmp_path / "health"
    root.mkdir()
    artifact = root / "report.json"
    artifact.write_text("{}", encoding="utf-8")
    manifest = root / "release-artifacts-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifacts": [
                    {
                        "path": "report.json",
                        "role": "report",
                        "sha256": "0" * 64,
                        "bytes": artifact.stat().st_size,
                        "command": "generate report",
                        "summary": {"rows": 1},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "experiments/validate_release_health.py",
            "--validate-manifest",
            str(manifest),
            "--manifest-root",
            str(root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert '"valid": false' in result.stdout
    assert "report.json: sha256 mismatch" in result.stdout
    assert "Release artifact manifest validation failed" in result.stderr


def test_release_artifact_manifest_rejects_path_traversal(tmp_path):
    root = tmp_path / "health"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    manifest = root / "release-artifacts-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifacts": [
                    {
                        "path": "../outside.json",
                        "role": "escaped_report",
                        "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
                        "bytes": outside.stat().st_size,
                        "command": "read escaped report",
                        "summary": {"rows": 1},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    audit = validate_release_artifact_manifest(manifest, output_dir=root)

    assert not audit["valid"]
    assert "../outside.json: artifact path must stay inside manifest root" in audit["errors"]


def test_validate_release_health_rejects_non_positive_resamples(tmp_path):
    with pytest.raises(ValueError, match="comparison_resamples must be positive"):
        validate_release_health(
            output_dir=tmp_path / "health",
            comparison_resamples=0,
        )


def test_validate_release_health_cli_rejects_non_positive_resamples(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "experiments/validate_release_health.py",
            "--output-dir",
            str(tmp_path / "health"),
            "--comparison-resamples",
            "0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "must be positive" in result.stderr
