#!/usr/bin/env python3
"""Generate and validate SHA-256 checksums for release artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore[no-redef]

DEFAULT_OUTPUT = Path("build/release-checksums.json")
DEFAULT_DIST_DIR = Path("dist")
DEFAULT_SBOM_PATH = Path("build/sbom/verarag-sbom.cdx.json")
DEFAULT_RELEASE_HEALTH_MANIFEST = Path(
    "build/release-health/release-artifacts-manifest.json"
)


@dataclass(frozen=True)
class ReleaseChecksumAudit:
    valid: bool
    errors: list[str]
    manifest_path: str
    artifact_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "manifest_path": self.manifest_path,
            "artifact_count": self.artifact_count,
        }


def build_release_checksum_manifest(
    *,
    project_root: str | Path = ".",
    dist_dir: str | Path = DEFAULT_DIST_DIR,
    sbom_path: str | Path = DEFAULT_SBOM_PATH,
    release_health_manifest: str | Path = DEFAULT_RELEASE_HEALTH_MANIFEST,
) -> dict[str, Any]:
    """Build a release checksum manifest for package and release metadata artifacts."""
    root = Path(project_root)
    project = _read_project_metadata(root / "pyproject.toml")
    dist = Path(dist_dir)
    distribution = _distribution_filename_name(str(project["name"]))
    version = str(project["version"])
    artifacts = [
        _artifact_entry(
            dist / f"{distribution}-{version}.tar.gz",
            role="sdist",
            project_root=root,
        ),
        _artifact_entry(
            dist / f"{distribution}-{version}-py3-none-any.whl",
            role="wheel",
            project_root=root,
        ),
        _artifact_entry(Path(sbom_path), role="cyclonedx_sbom", project_root=root),
        _artifact_entry(
            Path(release_health_manifest),
            role="release_health_artifact_manifest",
            project_root=root,
        ),
    ]
    return {
        "schema_version": 1,
        "project": {
            "name": project["name"],
            "version": version,
        },
        "generator": "experiments.generate_release_checksums",
        "artifacts": artifacts,
    }


def write_release_checksum_manifest(
    manifest: dict[str, Any],
    output_path: str | Path = DEFAULT_OUTPUT,
) -> Path:
    """Write the checksum manifest with stable key ordering."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def validate_release_checksum_manifest(
    manifest_path: str | Path,
    *,
    manifest_root: str | Path = ".",
) -> ReleaseChecksumAudit:
    """Validate artifact paths, sizes, and SHA-256 hashes in a checksum manifest."""
    path = Path(manifest_path)
    errors: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return ReleaseChecksumAudit(False, [f"Manifest does not exist: {path}"], str(path), 0)
    except json.JSONDecodeError as exc:
        return ReleaseChecksumAudit(False, [f"Manifest is not valid JSON: {exc}"], str(path), 0)

    if not isinstance(payload, dict):
        return ReleaseChecksumAudit(False, ["Manifest must be a JSON object"], str(path), 0)
    if payload.get("schema_version") != 1:
        errors.append("manifest.schema_version must be 1")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append("manifest.artifacts must be a non-empty list")
        artifacts = []

    root = Path(manifest_root)
    seen_paths: set[str] = set()
    seen_roles: set[str] = set()
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            errors.append(f"artifact[{index}] must be an object")
            continue
        rel_path = artifact.get("path")
        role = artifact.get("role")
        expected_sha = artifact.get("sha256")
        expected_size = artifact.get("bytes")
        if not isinstance(role, str) or not role:
            errors.append(f"artifact[{index}].role is required")
        elif role in seen_roles:
            errors.append(f"artifact role is duplicated: {role}")
        else:
            seen_roles.add(role)
        if not isinstance(rel_path, str) or not rel_path:
            errors.append(f"artifact[{index}].path is required")
            continue
        artifact_rel = Path(rel_path)
        if artifact_rel.is_absolute() or ".." in artifact_rel.parts:
            errors.append(f"{rel_path}: artifact path must stay inside manifest root")
            continue
        if rel_path in seen_paths:
            errors.append(f"artifact path is duplicated: {rel_path}")
        seen_paths.add(rel_path)
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

    required_roles = {
        "sdist",
        "wheel",
        "cyclonedx_sbom",
        "release_health_artifact_manifest",
    }
    missing_roles = sorted(required_roles - seen_roles)
    for role in missing_roles:
        errors.append(f"manifest missing required artifact role: {role}")

    return ReleaseChecksumAudit(
        valid=not errors,
        errors=errors,
        manifest_path=str(path),
        artifact_count=len(artifacts),
    )


def _artifact_entry(path: Path, *, role: str, project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    artifact_path = path if path.is_absolute() else root / path
    artifact_path = artifact_path.resolve()
    try:
        rel_path = artifact_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Release artifact path must stay inside project root: {path}"
        ) from exc
    if ".." in rel_path.parts:
        raise ValueError(f"Release artifact path must stay inside project root: {path}")
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Expected release artifact {artifact_path}")
    return {
        "role": role,
        "path": rel_path.as_posix(),
        "sha256": _sha256(artifact_path),
        "bytes": artifact_path.stat().st_size,
    }


def _read_project_metadata(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        pyproject = tomllib.load(handle)
    project = pyproject.get("project")
    if not isinstance(project, dict):
        raise ValueError("pyproject.toml missing [project] metadata")
    for key in ("name", "version"):
        if not isinstance(project.get(key), str) or not project[key]:
            raise ValueError(f"pyproject.toml missing project.{key}")
    return project


def _distribution_filename_name(project_name: str) -> str:
    return project_name.replace("-", "_").lower()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dist-dir", type=Path, default=DEFAULT_DIST_DIR)
    parser.add_argument("--sbom", type=Path, default=DEFAULT_SBOM_PATH)
    parser.add_argument(
        "--release-health-manifest",
        type=Path,
        default=DEFAULT_RELEASE_HEALTH_MANIFEST,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the generated checksum manifest after writing it.",
    )
    parser.add_argument(
        "--validate-manifest",
        type=Path,
        help="Validate an existing checksum manifest without regenerating it.",
    )
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=Path("."),
        help="Root directory for artifact paths inside --validate-manifest.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)

    if args.validate_manifest:
        existing_audit = validate_release_checksum_manifest(
            args.validate_manifest,
            manifest_root=args.manifest_root,
        )
        if args.json:
            print(json.dumps(existing_audit.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        elif existing_audit.valid:
            print(
                f"Release checksum manifest validated: "
                f"{existing_audit.artifact_count} artifacts at {existing_audit.manifest_path}."
            )
        else:
            print("Release checksum manifest validation failed:", file=sys.stderr)
            for error in existing_audit.errors:
                print(f"- {error}", file=sys.stderr)
        if not existing_audit.valid:
            raise SystemExit(1)
        return

    manifest = build_release_checksum_manifest(
        project_root=args.project_root,
        dist_dir=args.dist_dir,
        sbom_path=args.sbom,
        release_health_manifest=args.release_health_manifest,
    )
    output_path = write_release_checksum_manifest(manifest, args.output)
    audit: ReleaseChecksumAudit | None = None
    if args.check:
        audit = validate_release_checksum_manifest(
            output_path,
            manifest_root=args.project_root,
        )
        if not audit.valid:
            if args.json:
                print(json.dumps(audit.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
            else:
                print("Release checksum manifest validation failed:", file=sys.stderr)
                for error in audit.errors:
                    print(f"- {error}", file=sys.stderr)
            raise SystemExit(1)

    if args.json:
        payload = {
            "manifest_path": str(output_path),
            "artifact_count": len(manifest["artifacts"]),
            "valid": audit.valid if audit else None,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    elif audit:
        print(
            f"Release checksum manifest validated: "
            f"{audit.artifact_count} artifacts at {audit.manifest_path}."
        )
    else:
        print(f"Release checksum manifest written: {output_path}.")


if __name__ == "__main__":
    main()
