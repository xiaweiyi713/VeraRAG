"""Tests for dependency metadata validation."""

from pathlib import Path

from experiments.validate_dependency_metadata import validate_dependency_metadata


def test_dependency_metadata_accepts_consistent_files(tmp_path):
    _write_metadata_fixture(tmp_path)

    audit = validate_dependency_metadata(tmp_path)

    assert audit.valid
    assert audit.errors == []
    assert audit.project_dependencies == ["numpy>=1.21.0", "fastapi>=0.104.0"]
    assert "build" in audit.requirement_names


def test_dependency_metadata_reports_missing_project_requirement(tmp_path):
    _write_metadata_fixture(tmp_path, requirements="fastapi>=0.104.0\nbuild>=1.0.0\n")

    audit = validate_dependency_metadata(tmp_path)

    assert not audit.valid
    assert "requirements.txt missing project dependency numpy>=1.21.0" in audit.errors


def test_dependency_metadata_reports_missing_dev_dependency(tmp_path):
    _write_metadata_fixture(
        tmp_path,
        requirements="numpy>=1.21.0\nfastapi>=0.104.0\npytest>=7.0.0\n",
    )

    audit = validate_dependency_metadata(tmp_path)

    assert not audit.valid
    assert "requirements.txt missing dev dependency build>=1.0.0" in audit.errors


def test_dependency_metadata_reports_environment_not_delegating_to_requirements(tmp_path):
    _write_metadata_fixture(
        tmp_path,
        environment="name: demo\ndependencies:\n  - python=3.10\n",
    )

    audit = validate_dependency_metadata(tmp_path)

    assert not audit.valid
    assert "environment.yml does not include pip -r requirements.txt" in audit.errors


def test_dependency_metadata_reports_incomplete_all_extra(tmp_path):
    _write_metadata_fixture(
        tmp_path,
        all_extra='"demo[dense,dev]"',
    )

    audit = validate_dependency_metadata(tmp_path)

    assert not audit.valid
    assert "pyproject optional all extra missing groups: eval" in audit.errors


def _write_metadata_fixture(
    root: Path,
    *,
    requirements: str | None = None,
    environment: str | None = None,
    all_extra: str = '"demo[dense,eval,dev]"',
) -> None:
    (root / "pyproject.toml").write_text(
        "\n".join(
            [
                "[build-system]",
                'requires = ["setuptools>=77.0", "wheel"]',
                "",
                "[project]",
                'requires-python = ">=3.10"',
                "dependencies = [",
                '    "numpy>=1.21.0",',
                '    "fastapi>=0.104.0",',
                "]",
                "",
                "[project.optional-dependencies]",
                'dense = ["sentence-transformers>=2.2.0"]',
                'eval = ["matplotlib>=3.5.0"]',
                'dev = ["pytest>=7.0.0", "build>=1.0.0"]',
                f"all = [{all_extra}]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "requirements.txt").write_text(
        requirements
        or "\n".join(
            [
                "numpy>=1.21.0",
                "fastapi>=0.104.0",
                "pytest>=7.0.0",
                "build>=1.0.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "environment.yml").write_text(
        environment
        or "name: demo\ndependencies:\n  - python=3.10\n  - pip\n  - pip:\n    - -r requirements.txt\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        "pip install -r requirements.txt\nconda env create -f environment.yml\n",
        encoding="utf-8",
    )
