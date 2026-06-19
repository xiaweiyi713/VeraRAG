#!/usr/bin/env python3
"""Validate built VeraRAG sdist and wheel contents before release."""

from __future__ import annotations

import argparse
import configparser
import json
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

REQUIRED_SDIST_PATHS = (
    "CHANGELOG.md",
    "CITATION.cff",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "Dockerfile",
    "LICENSE",
    "README.md",
    "SECURITY.md",
    ".dockerignore",
    ".pre-commit-config.yaml",
    ".github/CODEOWNERS",
    ".github/dependabot.yml",
    ".github/ISSUE_TEMPLATE/config.yml",
    ".github/workflows/codeql.yml",
    ".github/workflows/release.yml",
    ".github/workflows/scorecard.yml",
    ".github/workflows/test.yml",
    "configs/deepseek_rules_only.yaml",
    "data/external/conflict_mini_v1/manifest.json",
    "data/external/conflict_mini_v1/questions.jsonl",
    "data/verabench/questions.jsonl",
    "docs/API.md",
    "docs/EVALUATION.md",
    "docs/GPU_TRAINING.md",
    "docs/RELEASING.md",
    "docs/RESULTS.md",
    "docs/VERABENCH_CARD.md",
    "examples/quickstart.py",
    "experiments/audit_conflict_model.py",
    "experiments/audit_verabench_contamination.py",
    "experiments/build_conflict_training_data.py",
    "experiments/build_external_annotation_packet.py",
    "experiments/doctor.py",
    "experiments/evaluate_retrieval.py",
    "experiments/generate_release_checksums.py",
    "experiments/generate_sbom.py",
    "experiments/plan_retrieval_ablation.py",
    "experiments/scan_secrets.py",
    "experiments/validate_configs.py",
    "experiments/validate_dependency_metadata.py",
    "experiments/validate_deployment.py",
    "experiments/validate_docs.py",
    "experiments/validate_examples.py",
    "experiments/validate_installed_wheel.py",
    "experiments/validate_precommit.py",
    "experiments/validate_project_metadata.py",
    "experiments/validate_python_support.py",
    "experiments/validate_release_health.py",
    "experiments/validate_results.py",
    "experiments/validate_version_identity.py",
    "experiments/validate_package_contents.py",
    "scripts/start_windows_conflict_training_matrix.sh",
    "scripts/start_windows_verabench_eval.sh",
    "src/benchmark/data/verabench/questions.jsonl",
    "src/py.typed",
    "tests/test_docs_validation.py",
    "tests/test_secret_scan.py",
    "tests/test_release_checksums.py",
    "tests/test_sbom_generation.py",
    "verarag/py.typed",
    "web/templates/index.html",
)
REQUIRED_WHEEL_PATHS = (
    "configs/deepseek_rules_only.yaml",
    "experiments/audit_conflict_model.py",
    "experiments/audit_verabench_contamination.py",
    "experiments/build_conflict_training_data.py",
    "experiments/build_external_annotation_packet.py",
    "experiments/doctor.py",
    "experiments/evaluate_retrieval.py",
    "experiments/generate_release_checksums.py",
    "experiments/generate_sbom.py",
    "experiments/plan_retrieval_ablation.py",
    "experiments/scan_secrets.py",
    "experiments/validate_configs.py",
    "experiments/validate_dependency_metadata.py",
    "experiments/validate_deployment.py",
    "experiments/validate_docs.py",
    "experiments/validate_examples.py",
    "experiments/validate_installed_wheel.py",
    "experiments/validate_precommit.py",
    "experiments/validate_project_metadata.py",
    "experiments/validate_python_support.py",
    "experiments/validate_release_health.py",
    "experiments/validate_results.py",
    "experiments/validate_version_identity.py",
    "experiments/validate_package_contents.py",
    "src/benchmark/data/verabench/questions.jsonl",
    "src/py.typed",
    "verarag/py.typed",
    "web/static/app.js",
    "web/templates/index.html",
)
REQUIRED_WHEEL_SUFFIXES = (
    ".data/data/share/doc/verarag/API.md",
    ".data/data/share/doc/verarag/EVALUATION.md",
    ".data/data/share/doc/verarag/GPU_TRAINING.md",
    ".data/data/share/doc/verarag/RELEASING.md",
    ".data/data/share/doc/verarag/RESULTS.md",
    ".data/data/share/doc/verarag/VERABENCH_CARD.md",
)
FORBIDDEN_PACKAGE_SUBSTRINGS = (
    "docs/superpowers",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
)


@dataclass(frozen=True)
class PackageAudit:
    project_name: str
    project_version: str
    sdist: str
    wheel: str
    valid: bool
    errors: list[str]
    pyproject_scripts: list[str]
    wheel_scripts: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "project_name": self.project_name,
            "project_version": self.project_version,
            "sdist": self.sdist,
            "wheel": self.wheel,
            "valid": self.valid,
            "errors": self.errors,
            "pyproject_scripts": self.pyproject_scripts,
            "wheel_scripts": self.wheel_scripts,
        }


def validate_package_contents(
    dist_dir: str | Path = "dist", project_root: str | Path = "."
) -> PackageAudit:
    dist_path = Path(dist_dir)
    root = Path(project_root)
    project_metadata = _parse_pyproject_project_metadata(root / "pyproject.toml")
    project_name = project_metadata["name"]
    project_version = project_metadata["version"]
    distribution_name = _distribution_filename_name(project_name)
    sdist = _required_artifact(dist_path, f"{distribution_name}-{project_version}.tar.gz")
    wheel = _required_artifact(
        dist_path,
        f"{distribution_name}-{project_version}-py3-none-any.whl",
    )

    errors: list[str] = []
    sdist_names = _read_sdist_names(sdist)
    wheel_names = _read_wheel_names(wheel)
    pyproject_entry_points = _parse_pyproject_script_entries(root / "pyproject.toml")
    wheel_entry_points = _parse_wheel_script_entries(wheel)
    pyproject_scripts = sorted(pyproject_entry_points)
    wheel_scripts = sorted(wheel_entry_points)

    sdist_prefix = _sdist_prefix(sdist_names)
    for path in REQUIRED_SDIST_PATHS:
        expected = f"{sdist_prefix}/{path}"
        if expected not in sdist_names:
            errors.append(f"sdist missing {path}")

    for path in REQUIRED_WHEEL_PATHS:
        if path not in wheel_names:
            errors.append(f"wheel missing {path}")

    for suffix in REQUIRED_WHEEL_SUFFIXES:
        if not any(name.endswith(suffix) for name in wheel_names):
            errors.append(f"wheel missing doc suffix {suffix}")

    for forbidden in FORBIDDEN_PACKAGE_SUBSTRINGS:
        if any(forbidden in name for name in sdist_names | wheel_names):
            errors.append(f"package includes forbidden path fragment {forbidden}")

    if not pyproject_scripts:
        errors.append("pyproject has no [project.scripts] entries")
    missing_wheel_scripts = sorted(set(pyproject_scripts) - set(wheel_scripts))
    for script in missing_wheel_scripts:
        errors.append(f"wheel entry_points.txt missing console script {script}")
    extra_wheel_scripts = sorted(set(wheel_scripts) - set(pyproject_scripts))
    for script in extra_wheel_scripts:
        errors.append(f"wheel entry_points.txt has undeclared console script {script}")
    shared_scripts = sorted(set(pyproject_scripts) & set(wheel_scripts))
    for script in shared_scripts:
        expected = pyproject_entry_points[script]
        actual = wheel_entry_points[script]
        if actual != expected:
            errors.append(
                f"wheel console script {script} target mismatch: "
                f"expected {expected}, found {actual}"
            )
            continue
        module_candidates = _console_script_module_candidates(expected)
        if not module_candidates:
            errors.append(f"console script {script} has invalid target {expected}")
            continue
        if not _contains_any(
            sdist_names,
            [f"{sdist_prefix}/{candidate}" for candidate in module_candidates],
        ):
            errors.append(
                f"sdist missing console script {script} target module "
                f"{module_candidates[0]}"
            )
        if not _contains_any(wheel_names, module_candidates):
            errors.append(
                f"wheel missing console script {script} target module "
                f"{module_candidates[0]}"
            )

    return PackageAudit(
        project_name=project_name,
        project_version=project_version,
        sdist=str(sdist),
        wheel=str(wheel),
        valid=not errors,
        errors=errors,
        pyproject_scripts=pyproject_scripts,
        wheel_scripts=wheel_scripts,
    )


def _required_artifact(directory: Path, filename: str) -> Path:
    path = directory / filename
    if not path.is_file():
        raise FileNotFoundError(f"Expected built artifact {path}")
    return path


def _read_sdist_names(path: Path) -> set[str]:
    with tarfile.open(path) as archive:
        return set(archive.getnames())


def _read_wheel_names(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as archive:
        return set(archive.namelist())


def _sdist_prefix(names: set[str]) -> str:
    prefixes = {name.split("/", 1)[0] for name in names if "/" in name}
    if len(prefixes) != 1:
        raise ValueError(f"Expected one sdist top-level directory, found {prefixes}")
    return prefixes.pop()


def _parse_pyproject_script_entries(path: Path) -> dict[str, str]:
    in_scripts = False
    scripts: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_scripts = line == "[project.scripts]"
            continue
        if in_scripts and "=" in line:
            name, target = line.split("=", 1)
            scripts[name.strip()] = _strip_toml_string(target.strip())
    return scripts


def _parse_pyproject_project_metadata(path: Path) -> dict[str, str]:
    in_project = False
    metadata: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_project = line == "[project]"
            continue
        if in_project and "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            if key in {"name", "version"}:
                metadata[key] = _strip_toml_string(value.strip())
    missing = sorted({"name", "version"} - set(metadata))
    if missing:
        raise ValueError(f"pyproject [project] missing {', '.join(missing)}")
    return metadata


def _distribution_filename_name(project_name: str) -> str:
    return project_name.replace("-", "_")


def _parse_wheel_script_entries(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as archive:
        entry_point_names = [
            name for name in archive.namelist() if name.endswith("/entry_points.txt")
        ]
        if len(entry_point_names) != 1:
            return {}
        text = archive.read(entry_point_names[0]).decode("utf-8")
    parser = configparser.ConfigParser()
    parser.read_string(text)
    if not parser.has_section("console_scripts"):
        return {}
    return dict(parser.items("console_scripts"))


def _console_script_module_candidates(target: str) -> tuple[str, ...]:
    module_ref = target.split(":", 1)[0].split("[", 1)[0].strip()
    if not module_ref:
        return ()
    module_path = module_ref.replace(".", "/")
    return (f"{module_path}.py", f"{module_path}/__init__.py")


def _contains_any(names: set[str], candidates: list[str] | tuple[str, ...]) -> bool:
    return any(candidate in names for candidate in candidates)


def _strip_toml_string(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dist-dir",
        default="dist",
        help="Directory containing one VeraRAG sdist and one wheel.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Source checkout root containing pyproject.toml.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON audit output.")
    args = parser.parse_args(argv)

    audit = validate_package_contents(args.dist_dir, args.project_root)
    if args.json:
        print(json.dumps(audit.to_dict(), indent=2))
    elif audit.valid:
        print(
            "Package contents validated: "
            f"{len(audit.pyproject_scripts)} console scripts, "
            "target modules, and required files present."
        )
    else:
        print("Package content validation failed:")
        for error in audit.errors:
            print(f"- {error}")

    if not audit.valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
