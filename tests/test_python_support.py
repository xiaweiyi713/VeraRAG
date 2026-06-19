"""Tests for Python support metadata validation."""

import json
from pathlib import Path

from experiments.validate_python_support import main, validate_python_support


def test_python_support_accepts_repository_state():
    audit = validate_python_support()

    assert audit.valid
    assert audit.errors == []
    assert audit.requires_python == ">=3.10"
    assert audit.ci_versions == ["3.10", "3.11", "3.12", "3.13"]


def test_python_support_reports_stale_matrix_and_tooling(tmp_path):
    _write_python_support_fixture(
        tmp_path,
        pyproject_requires='requires-python = ">=3.11"',
        classifiers=[
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.11",
        ],
        workflow='python-version: ["3.11", "3.12"]\npython-version: ${{ matrix.python-version }}\n',
        ruff='target-version = "py311"\n',
        mypy="[mypy]\npython_version = 3.11\n",
        environment="dependencies:\n  - python=3.11\n",
    )

    audit = validate_python_support(tmp_path)

    assert not audit.valid
    assert "pyproject requires-python must be >=3.10, found >=3.11" in audit.errors
    assert "pyproject classifiers missing Programming Language :: Python :: 3.10" in audit.errors
    assert (
        "GitHub Actions python-version matrix must be ['3.10', '3.11', '3.12', '3.13'], "
        "found ['3.11', '3.12']"
        in audit.errors
    )
    assert "ruff.toml target-version must be py310, found py311" in audit.errors
    assert "mypy.ini python_version must be 3.10, found 3.11" in audit.errors
    assert "environment.yml python pin must be 3.10, found 3.11" in audit.errors


def test_python_support_reports_missing_docs(tmp_path):
    _write_python_support_fixture(
        tmp_path,
        readme="No version statement.\n",
        contributing="No version statement.\n",
    )

    audit = validate_python_support(tmp_path)

    assert not audit.valid
    assert "README.md missing Python 3.10+ support statement" in audit.errors
    assert "CONTRIBUTING.md missing Python 3.10+ support statement" in audit.errors
    assert "README Python badge should point at python-3.10+" in audit.errors


def test_python_support_cli_json(capsys):
    exit_code = main(["--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["requires_python"] == ">=3.10"


def _write_python_support_fixture(
    root: Path,
    *,
    pyproject_requires: str = 'requires-python = ">=3.10"',
    classifiers: list[str] | None = None,
    workflow: str | None = None,
    ruff: str = 'target-version = "py310"\n',
    mypy: str = "[mypy]\npython_version = 3.10\n",
    environment: str = "dependencies:\n  - python=3.10\n",
    readme: str | None = None,
    contributing: str | None = None,
) -> None:
    (root / ".github" / "workflows").mkdir(parents=True)
    classifiers = classifiers or [
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ]
    classifier_lines = ",\n".join(f'    "{classifier}"' for classifier in classifiers)
    (root / "pyproject.toml").write_text(
        "\n".join([
            "[project]",
            pyproject_requires,
            "classifiers = [",
            classifier_lines,
            "]",
            "",
            "[tool.mypy]",
            'python_version = "3.10"',
        ])
        + "\n",
        encoding="utf-8",
    )
    (root / ".github" / "workflows" / "test.yml").write_text(
        workflow
        if workflow is not None
        else 'python-version: ["3.10", "3.11", "3.12", "3.13"]\n'
        "python-version: ${{ matrix.python-version }}\n",
        encoding="utf-8",
    )
    (root / "ruff.toml").write_text(ruff, encoding="utf-8")
    (root / "mypy.ini").write_text(mypy, encoding="utf-8")
    (root / "environment.yml").write_text(environment, encoding="utf-8")
    (root / "README.md").write_text(
        readme if readme is not None else "[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]\nPython 3.10+\n",
        encoding="utf-8",
    )
    (root / "CONTRIBUTING.md").write_text(
        contributing if contributing is not None else "Python 3.10+\n",
        encoding="utf-8",
    )
