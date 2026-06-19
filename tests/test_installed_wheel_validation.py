"""Tests for installed wheel smoke validation."""

from pathlib import Path

import pytest

import experiments.validate_installed_wheel as install_validation


def test_installed_wheel_validation_smokes_all_declared_scripts(tmp_path, monkeypatch):
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    _write_pyproject(
        tmp_path,
        {
            "verarag-alpha": "pkg.alpha:main",
            "verarag-beta": "pkg.beta:main",
        },
    )
    wheel = dist_dir / "verarag-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    calls = []

    def fake_run(label, args, *, cwd, env, timeout_seconds):
        calls.append((label, args, cwd, env, timeout_seconds))
        return install_validation._CommandResult(label, True)

    monkeypatch.setattr(
        install_validation,
        "_extract_wheel",
        lambda *_: install_validation._CommandResult("extract wheel", True),
    )
    monkeypatch.setattr(install_validation, "_run", fake_run)

    audit = install_validation.validate_installed_wheel(
        dist_dir=dist_dir,
        project_root=tmp_path,
        timeout_seconds=7,
    )

    assert audit.valid
    assert audit.errors == []
    assert audit.console_scripts == ["verarag-alpha", "verarag-beta"]
    assert [call[0] for call in calls] == [
        "import installed public API and packaged data",
        "verarag-alpha --help",
        "verarag-beta --help",
    ]
    assert all(call[2].name == "work" for call in calls)
    assert all(call[4] == 7 for call in calls)


def test_installed_wheel_validation_reports_script_failure(tmp_path, monkeypatch):
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    _write_pyproject(tmp_path, {"verarag-alpha": "pkg.alpha:main"})
    (dist_dir / "verarag-0.1.0-py3-none-any.whl").write_bytes(b"wheel")

    def fake_run(label, args, *, cwd, env, timeout_seconds):
        if label == "verarag-alpha --help":
            return install_validation._CommandResult(label, False, "script exploded")
        return install_validation._CommandResult(label, True)

    monkeypatch.setattr(
        install_validation,
        "_extract_wheel",
        lambda *_: install_validation._CommandResult("extract wheel", True),
    )
    monkeypatch.setattr(install_validation, "_run", fake_run)

    audit = install_validation.validate_installed_wheel(
        dist_dir=dist_dir,
        project_root=tmp_path,
    )

    assert not audit.valid
    assert audit.errors == ["script exploded"]


def test_installed_wheel_validation_requires_current_wheel(tmp_path):
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    _write_pyproject(tmp_path, {"verarag-alpha": "pkg.alpha:main"})

    with pytest.raises(FileNotFoundError, match=r"verarag-0\.1\.0-py3-none-any\.whl"):
        install_validation.validate_installed_wheel(
            dist_dir=dist_dir,
            project_root=tmp_path,
        )


def _write_pyproject(root: Path, scripts: dict[str, str]) -> None:
    lines = ['[project]', 'name = "verarag"', 'version = "0.1.0"', "", "[project.scripts]"]
    lines.extend(f'{script} = "{target}"' for script, target in scripts.items())
    (root / "pyproject.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")
