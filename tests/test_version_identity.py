"""Tests for release version identity validation."""

import json
from pathlib import Path

from experiments.validate_version_identity import main, validate_version_identity


def test_version_identity_accepts_repository_state():
    audit = validate_version_identity()

    assert audit.valid
    assert audit.errors == []
    assert audit.project_name == "verarag"
    assert audit.project_version == audit.source_version == audit.citation_version


def test_version_identity_reports_source_and_citation_mismatch(tmp_path):
    _write_version_fixture(
        tmp_path,
        source_version="0.2.0",
        citation_version="0.0.9",
    )

    audit = validate_version_identity(tmp_path)

    assert not audit.valid
    assert "src.__version__ must match pyproject project.version: 0.2.0 != 0.1.0" in audit.errors
    assert (
        "CITATION.cff version must match pyproject project.version: 0.0.9 != 0.1.0"
        in audit.errors
    )


def test_version_identity_reports_missing_release_instructions(tmp_path):
    _write_version_fixture(
        tmp_path,
        releasing="Update version in pyproject only.\n",
        readme="make version-check\nverarag-validate-version\n",
        changelog="verarag-validate-version\n",
    )

    audit = validate_version_identity(tmp_path)

    assert not audit.valid
    assert (
        "docs/RELEASING.md missing version instruction: "
        "Update version in `pyproject.toml`, `src/__init__.py`, and `CITATION.cff`"
        in audit.errors
    )
    assert "docs/RELEASING.md missing version instruction: make version-check" in audit.errors
    assert (
        "docs/RELEASING.md missing version instruction: verarag-validate-version --json"
        in audit.errors
    )


def test_version_identity_cli_json(capsys):
    exit_code = main(["--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["project_version"] == payload["source_version"]


def _write_version_fixture(
    root: Path,
    *,
    project_version: str = "0.1.0",
    source_version: str = "0.1.0",
    citation_version: str = "0.1.0",
    public_init: str | None = None,
    releasing: str | None = None,
    readme: str | None = None,
    changelog: str | None = None,
) -> None:
    (root / "src").mkdir()
    (root / "verarag").mkdir()
    (root / "docs").mkdir()
    (root / "pyproject.toml").write_text(
        "\n".join([
            "[project]",
            'name = "verarag"',
            f'version = "{project_version}"',
        ])
        + "\n",
        encoding="utf-8",
    )
    (root / "src" / "__init__.py").write_text(
        f'__version__ = "{source_version}"\n',
        encoding="utf-8",
    )
    (root / "CITATION.cff").write_text(
        f"cff-version: 1.2.0\nversion: {citation_version}\n",
        encoding="utf-8",
    )
    (root / "verarag" / "__init__.py").write_text(
        public_init
        if public_init is not None
        else (
            "from src import __version__ as _SOURCE_VERSION\n"
            "import importlib.metadata as importlib_metadata\n"
            "def _read_package_version():\n"
            "    try:\n"
            "        return importlib_metadata.version('verarag')\n"
            "    except importlib_metadata.PackageNotFoundError:\n"
            "        return _SOURCE_VERSION\n"
            "__version__ = _read_package_version()\n"
        ),
        encoding="utf-8",
    )
    (root / "docs" / "RELEASING.md").write_text(
        releasing
        if releasing is not None
        else (
            "Update version in `pyproject.toml`, `src/__init__.py`, and `CITATION.cff`.\n"
            "make version-check\n"
            "verarag-validate-version --json\n"
        ),
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        readme if readme is not None else "make version-check\nverarag-validate-version\n",
        encoding="utf-8",
    )
    (root / "CHANGELOG.md").write_text(
        changelog if changelog is not None else "verarag-validate-version\n",
        encoding="utf-8",
    )
