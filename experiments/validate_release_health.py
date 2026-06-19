#!/usr/bin/env python3
"""Run release-critical VeraBench health checks with machine-readable output."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = Path(tempfile.gettempdir()) / "verarag-release-health"


def validate_release_health(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    comparison_resamples: int = 100,
) -> dict[str, Any]:
    """Run benchmark, fixture, demo, and comparison checks used by release gates."""
    if comparison_resamples <= 0:
        raise ValueError("comparison_resamples must be positive")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []

    _run([sys.executable, "scripts/migrate_verabench_span_traceability_v112.py", "--check"])

    benchmark_audit_path = out / "verabench-audit.json"
    _run([
        sys.executable,
        "experiments/validate_verabench.py",
        "--output",
        str(benchmark_audit_path),
    ])
    benchmark_audit = _load_json(benchmark_audit_path)
    _require(benchmark_audit.get("version") == "1.1.2", "VeraBench version must be 1.1.2")
    traceability = benchmark_audit.get("evidence_traceability", {})
    _require(traceability.get("total") == 208, "VeraBench must expose 208 evidence refs")
    _require(traceability.get("untraceable") == 0, "VeraBench evidence refs must be traceable")
    dependency_groups = benchmark_audit.get("evidence_dependency_groups", {})
    _require(
        dependency_groups.get("count") == 27,
        "VeraBench shared-evidence dependency group count must be 27",
    )
    _require(
        benchmark_audit.get("package_data_in_sync") is True,
        "Repository and packaged VeraBench data must be byte-identical",
    )
    checks.append({
        "name": "verabench_audit",
        "version": benchmark_audit["version"],
        "questions": benchmark_audit["total_questions"],
        "documents": benchmark_audit["total_documents"],
        "evidence_refs": traceability["total"],
        "dependency_groups": dependency_groups["count"],
    })
    artifacts = [
        _artifact_entry(
            benchmark_audit_path,
            output_dir=out,
            role="verabench_dataset_audit",
            command=(
                "python experiments/validate_verabench.py "
                f"--output {benchmark_audit_path}"
            ),
            summary={
                "version": benchmark_audit["version"],
                "questions": benchmark_audit["total_questions"],
                "documents": benchmark_audit["total_documents"],
                "evidence_refs": traceability["total"],
                "dependency_groups": dependency_groups["count"],
            },
        )
    ]

    external_audit_path = out / "external-conflict-audit.json"
    _run([
        sys.executable,
        "experiments/validate_external_conflict_set.py",
        "--data-dir",
        "data/external/conflict_mini_v1",
        "--min-questions",
        "6",
        "--output",
        str(external_audit_path),
    ])
    external_audit = _load_json(external_audit_path)
    external_coverage = external_audit.get("coverage", {})
    conflict_agreement = (
        external_audit.get("agreement", {})
        .get("conflict_present", {})
    )
    _require(external_audit.get("valid") is True, "External conflict fixture must be valid")
    _require(external_coverage.get("questions") == 6, "External conflict fixture must cover 6 questions")
    _require(
        conflict_agreement.get("cohen_kappa") == 1.0,
        "External conflict fixture must retain perfect fixture agreement",
    )
    checks.append({
        "name": "external_conflict_fixture",
        "questions": external_coverage["questions"],
        "annotations": external_coverage["annotations"],
        "adjudications": external_coverage["adjudications"],
        "cohen_kappa": conflict_agreement["cohen_kappa"],
    })
    artifacts.append(
        _artifact_entry(
            external_audit_path,
            output_dir=out,
            role="external_conflict_fixture_audit",
            command=(
                "python experiments/validate_external_conflict_set.py "
                "--data-dir data/external/conflict_mini_v1 --min-questions 6 "
                f"--output {external_audit_path}"
            ),
            summary={
                "questions": external_coverage["questions"],
                "annotations": external_coverage["annotations"],
                "adjudications": external_coverage["adjudications"],
                "cohen_kappa": conflict_agreement["cohen_kappa"],
            },
        )
    )

    packet_dir = out / "external-conflict-packet"
    _run([
        sys.executable,
        "experiments/build_external_annotation_packet.py",
        "--data-dir",
        "data/external/conflict_mini_v1",
        "--output-dir",
        str(packet_dir),
        "--annotator",
        "ann_a",
        "--annotator",
        "ann_b",
        "--overwrite",
    ])
    packet_manifest = _load_json(packet_dir / "packet_manifest.json")
    omitted = set(packet_manifest.get("blind_fields_omitted", []))
    _require(packet_manifest.get("questions") == 6, "Blind packet must include 6 questions")
    _require(
        "ground_truth_answer" in omitted,
        "Blind packet must omit ground_truth_answer",
    )
    checks.append({
        "name": "external_annotation_packet",
        "questions": packet_manifest["questions"],
        "annotators": packet_manifest["annotator_ids"],
        "blind_fields_omitted": sorted(omitted),
    })
    artifacts.append(
        _artifact_entry(
            packet_dir / "packet_manifest.json",
            output_dir=out,
            role="external_blind_annotation_packet_manifest",
            command=(
                "python experiments/build_external_annotation_packet.py "
                "--data-dir data/external/conflict_mini_v1 "
                f"--output-dir {packet_dir} --annotator ann_a --annotator ann_b "
                "--overwrite"
            ),
            summary={
                "questions": packet_manifest["questions"],
                "annotators": packet_manifest["annotator_ids"],
                "blind_fields_omitted": sorted(omitted),
            },
        )
    )

    demo_report_path = out / "verabench-demo.json"
    _run([
        sys.executable,
        "experiments/run_verabench.py",
        "--demo",
        "--output",
        str(demo_report_path),
    ])
    demo = _load_json(demo_report_path)
    _assert_demo_report(demo)
    checks.append({
        "name": "demo_metric_plumbing",
        "completed": demo["completed"],
        "errors": demo["errors"],
        "answer_f1": demo["overall_answer_f1"],
        "evidence_recall": demo["overall_evidence_recall"],
        "behavior_accuracy": demo["behavior_accuracy"],
        "dependency_clusters": demo["dependency_robust_confidence_intervals"]["clusters"],
    })
    artifacts.append(
        _artifact_entry(
            demo_report_path,
            output_dir=out,
            role="verabench_demo_report",
            command=(
                "python experiments/run_verabench.py --demo "
                f"--output {demo_report_path}"
            ),
            summary={
                "completed": demo["completed"],
                "errors": demo["errors"],
                "answer_f1": demo["overall_answer_f1"],
                "evidence_recall": demo["overall_evidence_recall"],
                "behavior_accuracy": demo["behavior_accuracy"],
                "dependency_clusters": demo["dependency_robust_confidence_intervals"]["clusters"],
            },
        )
    )

    comparison_path = out / "verabench-demo-comparison.json"
    _run([
        sys.executable,
        "experiments/compare_verabench_reports.py",
        str(demo_report_path),
        str(demo_report_path),
        "--allow-demo",
        "--resamples",
        str(comparison_resamples),
        "--format",
        "json",
        "--output",
        str(comparison_path),
    ])
    comparison = _load_json(comparison_path)["comparison"]
    _require(comparison.get("questions") == 152, "Demo self-comparison must cover 152 questions")
    answer_delta = comparison["metrics"]["answer_f1"]["delta_candidate_minus_baseline"]
    _require(answer_delta == 0.0, "Demo self-comparison answer_f1 delta must be 0")
    _require(
        comparison["behavior_mcnemar_exact"]["two_sided_exact_p"] == 1.0,
        "Demo self-comparison McNemar p-value must be 1",
    )
    checks.append({
        "name": "demo_paired_comparison",
        "questions": comparison["questions"],
        "answer_f1_delta": answer_delta,
        "mcnemar_p": comparison["behavior_mcnemar_exact"]["two_sided_exact_p"],
        "resamples": comparison["resamples"],
    })
    artifacts.append(
        _artifact_entry(
            comparison_path,
            output_dir=out,
            role="verabench_demo_self_comparison",
            command=(
                "python experiments/compare_verabench_reports.py "
                f"{demo_report_path} {demo_report_path} --allow-demo "
                f"--resamples {comparison_resamples} --format json "
                f"--output {comparison_path}"
            ),
            summary={
                "questions": comparison["questions"],
                "answer_f1_delta": answer_delta,
                "mcnemar_p": comparison["behavior_mcnemar_exact"]["two_sided_exact_p"],
                "resamples": comparison["resamples"],
            },
        )
    )

    manifest_path = out / "release-artifacts-manifest.json"
    manifest = {
        "schema_version": 1,
        "project": "verarag",
        "generator": "experiments.validate_release_health",
        "output_dir": str(out),
        "artifacts": artifacts,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_audit = validate_release_artifact_manifest(manifest_path, output_dir=out)
    _require(manifest_audit["valid"] is True, "Release artifact manifest must validate")
    checks.append({
        "name": "release_artifact_manifest",
        "artifacts": manifest_audit["artifacts"],
        "path": str(manifest_path),
    })

    return {
        "valid": True,
        "schema_version": 1,
        "output_dir": str(out),
        "artifact_manifest": str(manifest_path),
        "checks": checks,
    }


def validate_release_artifact_manifest(
    manifest_path: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Validate hashes and sizes recorded in a release artifact manifest."""
    path = Path(manifest_path)
    manifest = _load_json(path)
    root = Path(output_dir) if output_dir is not None else path.parent
    errors: list[str] = []

    if manifest.get("schema_version") != 1:
        errors.append("manifest.schema_version must be 1")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append("manifest.artifacts must be a non-empty list")
        artifacts = []

    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            errors.append(f"artifact[{index}] must be an object")
            continue
        rel_path = artifact.get("path")
        expected_sha = artifact.get("sha256")
        expected_size = artifact.get("bytes")
        for key in ("role", "command", "summary"):
            if not artifact.get(key):
                errors.append(f"artifact[{index}].{key} is required")
        if not isinstance(rel_path, str) or not rel_path:
            errors.append(f"artifact[{index}].path is required")
            continue
        artifact_rel = Path(rel_path)
        if artifact_rel.is_absolute() or ".." in artifact_rel.parts:
            errors.append(f"{rel_path}: artifact path must stay inside manifest root")
            continue
        artifact_path = root / artifact_rel
        if not artifact_path.is_file():
            errors.append(f"{rel_path}: artifact file is missing")
            continue
        actual_sha = _sha256(artifact_path)
        actual_size = artifact_path.stat().st_size
        if expected_sha != actual_sha:
            errors.append(f"{rel_path}: sha256 mismatch")
        if expected_size != actual_size:
            errors.append(f"{rel_path}: byte size mismatch")

    return {
        "valid": not errors,
        "artifacts": len(artifacts),
        "errors": errors,
    }


def _assert_demo_report(report: dict[str, Any]) -> None:
    _require(report.get("completed") == 152, "Demo report must complete 152 questions")
    _require(report.get("errors") == 0, "Demo report must have zero errors")
    for key in (
        "overall_answer_f1",
        "overall_evidence_recall",
        "overall_evidence_precision",
        "overall_conflict_f1",
        "behavior_accuracy",
        "avg_confidence",
    ):
        _require(report.get(key) == 1.0, f"Demo report {key} must be 1.0")
    for key in ("ece", "brier_score"):
        _require(report.get(key) == 0.0, f"Demo report {key} must be 0.0")
    metadata = report.get("metadata", {})
    _require(metadata.get("mode") == "demo", "Demo report metadata.mode must be demo")
    intervals = report.get("confidence_intervals", {})
    _require(
        intervals.get("method") == "stratified-question-bootstrap-v1",
        "Demo report must include stratified bootstrap intervals",
    )
    _require(
        intervals.get("metrics", {}).get("answer_f1", {}).get("lower") == 1.0,
        "Demo report answer_f1 lower interval must be 1.0",
    )
    robust = report.get("dependency_robust_confidence_intervals", {})
    _require(
        robust.get("method") == "evidence-cluster-bootstrap-v1",
        "Demo report must include dependency-robust intervals",
    )
    _require(
        robust.get("clusters") == 27,
        "Demo report dependency-robust intervals must use 27 clusters",
    )
    _require(
        robust.get("metrics", {}).get("answer_f1", {}).get("lower") == 1.0,
        "Demo report dependency-robust answer_f1 lower interval must be 1.0",
    )


def _run(args: list[str]) -> None:
    result = subprocess.run(
        args,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        if result.stdout:
            print(result.stdout, file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(
            result.returncode,
            args,
            output=result.stdout,
            stderr=result.stderr,
        )


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected top-level JSON object")
    return payload


def _artifact_entry(
    path: Path,
    *,
    output_dir: Path,
    role: str,
    command: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "path": path.relative_to(output_dir).as_posix(),
        "role": role,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "command": command,
        "summary": summary,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for intermediate audit artifacts.",
    )
    parser.add_argument(
        "--comparison-resamples",
        type=_positive_int,
        default=100,
        help="Bootstrap resamples for the demo self-comparison check.",
    )
    parser.add_argument(
        "--validate-manifest",
        type=Path,
        help=(
            "Validate an existing release-artifacts-manifest.json without "
            "rerunning release health checks."
        ),
    )
    parser.add_argument(
        "--manifest-root",
        type=Path,
        help=(
            "Root directory for paths inside --validate-manifest. "
            "Defaults to the manifest parent directory."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON only.")
    args = parser.parse_args(argv)

    if args.validate_manifest:
        audit = validate_release_artifact_manifest(
            args.validate_manifest,
            output_dir=args.manifest_root,
        )
        rendered = json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True)
        print(rendered)
        if not args.json:
            if audit["valid"]:
                print(
                    "Release artifact manifest validated: "
                    f"{audit['artifacts']} artifacts"
                )
            else:
                print(
                    "Release artifact manifest validation failed: "
                    + "; ".join(str(error) for error in audit["errors"]),
                    file=sys.stderr,
                )
        if not audit["valid"]:
            raise SystemExit(1)
        return

    report = validate_release_health(
        output_dir=args.output_dir,
        comparison_resamples=args.comparison_resamples,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if not args.json:
        checks = ", ".join(check["name"] for check in report["checks"])
        print(f"Release health validated: {checks}")


if __name__ == "__main__":
    main()
