"""Tests for package content validation."""

import tarfile
import zipfile
from pathlib import Path

from experiments.validate_package_contents import (
    FORBIDDEN_PACKAGE_SUBSTRINGS,
    REQUIRED_SDIST_PATHS,
    REQUIRED_WHEEL_PATHS,
    REQUIRED_WHEEL_SUFFIXES,
    validate_package_contents,
)


def test_package_validation_accepts_complete_distribution(tmp_path):
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    scripts = ["verarag-web", "verarag-validate-package"]
    _write_pyproject(tmp_path, scripts)
    _write_sdist(dist_dir, scripts=scripts, extra_paths=[])
    _write_wheel(
        dist_dir,
        scripts=scripts,
        extra_paths=[],
    )

    audit = validate_package_contents(dist_dir=dist_dir, project_root=tmp_path)

    assert audit.valid
    assert audit.errors == []
    assert audit.project_name == "verarag"
    assert audit.project_version == "0.1.0"
    assert audit.pyproject_scripts == ["verarag-validate-package", "verarag-web"]
    assert audit.wheel_scripts == ["verarag-validate-package", "verarag-web"]


def test_package_validation_reports_missing_scripts_and_forbidden_paths(tmp_path):
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    scripts = ["verarag-web", "verarag-validate-package"]
    _write_pyproject(tmp_path, scripts)
    _write_sdist(
        dist_dir,
        scripts=scripts,
        extra_paths=[FORBIDDEN_PACKAGE_SUBSTRINGS[0] + "/plan.md"],
    )
    _write_wheel(dist_dir, scripts=["verarag-web"], extra_paths=[])

    audit = validate_package_contents(dist_dir=dist_dir, project_root=tmp_path)

    assert not audit.valid
    assert "wheel entry_points.txt missing console script verarag-validate-package" in audit.errors
    assert (
        f"package includes forbidden path fragment {FORBIDDEN_PACKAGE_SUBSTRINGS[0]}"
        in audit.errors
    )


def test_package_validation_reports_extra_and_mismatched_wheel_scripts(tmp_path):
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    pyproject_scripts = {
        "verarag-web": "web.app:main",
        "verarag-validate-package": "experiments.validate_package_contents:main",
    }
    _write_pyproject(tmp_path, pyproject_scripts)
    _write_sdist(dist_dir, scripts=pyproject_scripts, extra_paths=[])
    _write_wheel(
        dist_dir,
        scripts={
            "verarag-web": "wrong.module:main",
            "verarag-validate-package": "experiments.validate_package_contents:main",
            "verarag-shadow-command": "shadow.module:main",
        },
        extra_paths=[],
    )

    audit = validate_package_contents(dist_dir=dist_dir, project_root=tmp_path)

    assert not audit.valid
    assert "wheel entry_points.txt has undeclared console script verarag-shadow-command" in audit.errors
    assert (
        "wheel console script verarag-web target mismatch: "
        "expected web.app:main, found wrong.module:main"
    ) in audit.errors


def test_package_validation_reports_missing_console_script_target_module(tmp_path):
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    scripts = {"verarag-web": "web.app:main"}
    _write_pyproject(tmp_path, scripts)
    _write_sdist(
        dist_dir,
        scripts=scripts,
        include_script_modules=False,
        extra_paths=[],
    )
    _write_wheel(
        dist_dir,
        scripts=scripts,
        include_script_modules=False,
        extra_paths=[],
    )

    audit = validate_package_contents(dist_dir=dist_dir, project_root=tmp_path)

    assert not audit.valid
    assert "sdist missing console script verarag-web target module web/app.py" in audit.errors
    assert "wheel missing console script verarag-web target module web/app.py" in audit.errors


def test_package_validation_selects_current_version_with_stale_dist_files(tmp_path):
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    scripts = ["verarag-web"]
    _write_pyproject(tmp_path, scripts)
    _write_sdist(dist_dir, version="0.0.9", scripts=scripts, extra_paths=[])
    _write_wheel(dist_dir, version="0.0.9", scripts=scripts, extra_paths=[])
    _write_sdist(dist_dir, version="0.1.0", scripts=scripts, extra_paths=[])
    _write_wheel(dist_dir, version="0.1.0", scripts=scripts, extra_paths=[])

    audit = validate_package_contents(dist_dir=dist_dir, project_root=tmp_path)

    assert audit.valid
    assert audit.sdist.endswith("verarag-0.1.0.tar.gz")
    assert audit.wheel.endswith("verarag-0.1.0-py3-none-any.whl")


def test_release_check_make_target_runs_core_quality_gates():
    makefile = Path("Makefile").read_text(encoding="utf-8")
    gitignore = Path(".gitignore").read_text(encoding="utf-8")

    assert "COVERAGE_MIN ?= 80" in makefile
    assert "RELEASE_HEALTH_DIR ?= build/release-health" in makefile
    assert "RELEASE_CHECKSUMS ?= build/release-checksums.json" in makefile
    assert "coverage-check:" in makefile
    assert "--cov-fail-under=$(COVERAGE_MIN)" in makefile
    assert "coverage.json" in gitignore
    assert (
        "release-check: lint version-check python-support-check doctor-check configs-check docs-check results-check examples-check deployment-check precommit-check deps-check "
        "metadata-check sbom-check coverage-check benchmark-check "
        "release-artifacts-check package-check release-checksums-check"
        in makefile
    )
    assert "version-check:" in makefile
    assert "experiments/validate_version_identity.py" in makefile
    assert "python-support-check:" in makefile
    assert "experiments/validate_python_support.py" in makefile
    assert "doctor-check:" in makefile
    assert "experiments/doctor.py" in makefile
    assert "configs-check:" in makefile
    assert "experiments/validate_configs.py" in makefile
    assert "docs-check:" in makefile
    assert "experiments/validate_docs.py" in makefile
    assert "results-check:" in makefile
    assert "experiments/validate_results.py" in makefile
    assert "examples-check:" in makefile
    assert "experiments/validate_examples.py" in makefile
    assert "deployment-check:" in makefile
    assert "experiments/validate_deployment.py" in makefile
    assert "precommit-check:" in makefile
    assert "experiments/validate_precommit.py" in makefile
    assert "deps-check:" in makefile
    assert "experiments/validate_dependency_metadata.py" in makefile
    assert "metadata-check:" in makefile
    assert "experiments/validate_project_metadata.py" in makefile
    assert "sbom-check:" in makefile
    assert "experiments/generate_sbom.py --output build/sbom/verarag-sbom.cdx.json --check" in makefile
    assert "benchmark-check:" in makefile
    assert "experiments/validate_release_health.py --output-dir $(RELEASE_HEALTH_DIR)" in makefile
    assert "/tmp/verarag-release-health" not in makefile
    assert "build/" in gitignore
    assert "release-artifacts-check:" in makefile
    assert (
        "experiments/validate_release_health.py --validate-manifest "
        "$(RELEASE_HEALTH_DIR)/release-artifacts-manifest.json "
        "--manifest-root $(RELEASE_HEALTH_DIR)"
        in makefile
    )
    assert "release-checksums-check:" in makefile
    assert "experiments/generate_release_checksums.py --output $(RELEASE_CHECKSUMS) --check" in makefile
    assert "package-check: build" in makefile
    assert "$(PYTHON) -m build --sdist --wheel" in makefile
    assert "experiments/validate_package_contents.py --dist-dir dist" in makefile
    assert "experiments/validate_installed_wheel.py --dist-dir dist" in makefile
    assert "installed-wheel-check: build" in makefile
    assert "external-fixture-check:" in makefile


def test_doctor_cli_is_required_in_release_archives():
    assert "experiments/doctor.py" in REQUIRED_SDIST_PATHS
    assert "experiments/doctor.py" in REQUIRED_WHEEL_PATHS
    assert "experiments/validate_configs.py" in REQUIRED_SDIST_PATHS
    assert "experiments/validate_configs.py" in REQUIRED_WHEEL_PATHS


def test_ci_reuses_local_quality_gates_and_uploads_secret_scan_sarif_artifact():
    workflow = Path(".github/workflows/test.yml").read_text(encoding="utf-8")

    assert "python experiments/scan_secrets.py --sarif > secret-scan.sarif || true" in workflow
    assert "if: ${{ always() && matrix.python-version == '3.13' }}" in workflow
    assert "Validate version identity" in workflow
    assert "python experiments/validate_version_identity.py" in workflow
    assert "Validate Python support metadata" in workflow
    assert "python experiments/validate_python_support.py" in workflow
    assert "Run environment doctor" in workflow
    assert "python experiments/doctor.py" in workflow
    assert "Validate default configs" in workflow
    assert "python experiments/validate_configs.py" in workflow
    assert "Validate documentation links" in workflow
    assert "python experiments/validate_docs.py" in workflow
    assert "Validate published results docs" in workflow
    assert "python experiments/validate_results.py" in workflow
    assert "Validate runnable examples" in workflow
    assert "python experiments/validate_examples.py" in workflow
    assert "Validate deployment config" in workflow
    assert "python experiments/validate_deployment.py" in workflow
    assert "Validate pre-commit config" in workflow
    assert "python experiments/validate_precommit.py" in workflow
    assert "Validate dependency metadata" in workflow
    assert "python experiments/validate_dependency_metadata.py" in workflow
    assert "Validate project metadata" in workflow
    assert "python experiments/validate_project_metadata.py" in workflow
    assert "make coverage-check PYTHON=python" in workflow
    assert "Build and validate package" in workflow
    assert "make package-check PYTHON=python" in workflow
    assert "verarag-wheel-smoke" not in workflow
    assert "coverage.json" in workflow
    assert "name: secret-scan-sarif" in workflow
    assert "path: secret-scan.sarif" in workflow


def _write_pyproject(root: Path, scripts: list[str] | dict[str, str]) -> None:
    lines = ['[project]', 'name = "verarag"', 'version = "0.1.0"', "", "[project.scripts]"]
    if isinstance(scripts, dict):
        script_items = scripts.items()
    else:
        script_items = ((script, "pkg.module:main") for script in scripts)
    lines.extend(f'{script} = "{target}"' for script, target in script_items)
    (root / "pyproject.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_sdist(
    dist_dir: Path,
    *,
    version: str = "0.1.0",
    scripts: list[str] | dict[str, str],
    include_script_modules: bool = True,
    extra_paths: list[str],
) -> None:
    path = dist_dir / f"verarag-{version}.tar.gz"
    script_module_paths = _script_module_paths(scripts) if include_script_modules else []
    with tarfile.open(path, "w:gz") as archive:
        for relative_path in _unique_paths(
            [*REQUIRED_SDIST_PATHS, *script_module_paths, *extra_paths]
        ):
            _add_tar_text(archive, f"verarag-{version}/{relative_path}")


def _write_wheel(
    dist_dir: Path,
    *,
    version: str = "0.1.0",
    scripts: list[str] | dict[str, str],
    include_script_modules: bool = True,
    extra_paths: list[str],
) -> None:
    path = dist_dir / f"verarag-{version}-py3-none-any.whl"
    script_module_paths = _script_module_paths(scripts) if include_script_modules else []
    with zipfile.ZipFile(path, "w") as archive:
        for relative_path in _unique_paths(
            [*REQUIRED_WHEEL_PATHS, *script_module_paths, *extra_paths]
        ):
            archive.writestr(relative_path, "")
        for suffix in REQUIRED_WHEEL_SUFFIXES:
            archive.writestr(f"verarag-{version}" + suffix, "")
        archive.writestr(
            f"verarag-{version}.dist-info/entry_points.txt",
            "[console_scripts]\n"
            + "\n".join(
                f"{script} = {target}"
                for script, target in _script_items(scripts)
            )
            + "\n",
        )


def _script_items(scripts: list[str] | dict[str, str]):
    if isinstance(scripts, dict):
        return scripts.items()
    return ((script, "pkg.module:main") for script in scripts)


def _script_module_paths(scripts: list[str] | dict[str, str]) -> list[str]:
    paths = []
    for _, target in _script_items(scripts):
        module_ref = target.split(":", 1)[0].split("[", 1)[0].strip()
        paths.append(module_ref.replace(".", "/") + ".py")
    return paths


def _unique_paths(paths: list[str]) -> list[str]:
    return list(dict.fromkeys(paths))


def _add_tar_text(archive: tarfile.TarFile, name: str) -> None:
    import io

    payload = b""
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))
